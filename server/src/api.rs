use std::{
    io::Write,
    path::{Path as FsPath, PathBuf},
    sync::Arc,
};

use axum::{
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, FromRequest, Multipart, Path, Query, Request, State},
    http::{header, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::{get, post, put},
    Json, Router,
};
use futures_util::StreamExt;
use jiff::{civil::DateTime, tz::TimeZone, Timestamp};
use rusqlite::Connection;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::{io::AsyncWriteExt, sync::Mutex};

pub struct App {
    dir: PathBuf,
    // ponytail: one lock for the db and all file appends; per-recording locks if uploads contend.
    db: Mutex<Connection>,
    tz: TimeZone,
}

impl App {
    pub fn open(dir: PathBuf) -> anyhow::Result<Self> {
        std::fs::create_dir_all(dir.join("recordings"))?;
        let db = crate::db::open(&dir.join("mictap.db"))?;
        Ok(Self {
            dir,
            db: Mutex::new(db),
            tz: TimeZone::system(),
        })
    }

    fn recording_dir(&self, id: &str) -> PathBuf {
        self.dir.join("recordings").join(id)
    }
}

pub fn router(app: Arc<App>) -> Router {
    Router::new()
        .route(
            "/recordings",
            post(upload).layer(DefaultBodyLimit::disable()),
        )
        .route(
            "/recordings/{id}/files/{name}",
            put(put_file).layer(DefaultBodyLimit::max(16 << 20)),
        )
        .route("/recordings/{id}/meta", put(put_meta))
        .route("/recordings/{id}/finish", post(finish))
        .route("/upload", get(|| async { Html(UPLOAD_FORM) }))
        .with_state(app)
}

const UPLOAD_FORM: &str = r#"<!doctype html>
<meta charset="utf-8">
<title>mictap upload</title>
<form method="post" action="/recordings" enctype="multipart/form-data">
<input type="file" name="file" accept="audio/*,video/*" required>
<button>Upload</button>
</form>
"#;

struct Error(StatusCode, String);

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        (self.0, self.1).into_response()
    }
}

impl<E: std::fmt::Display> From<E> for Error {
    fn from(e: E) -> Self {
        Error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    }
}

type Result<T> = std::result::Result<T, Error>;

fn check_name(s: &str) -> Result<()> {
    let ok = !s.is_empty()
        && s.len() <= 128
        && s.as_bytes()[0].is_ascii_alphanumeric()
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
        && s != "meta.json";
    if ok {
        Ok(())
    } else {
        Err(Error(StatusCode::BAD_REQUEST, format!("bad name: {s}")))
    }
}

#[derive(Deserialize)]
struct Offset {
    offset: u64,
}

async fn put_file(
    State(app): State<Arc<App>>,
    Path((id, name)): Path<(String, String)>,
    Query(q): Query<Offset>,
    body: Bytes,
) -> Result<Response> {
    check_name(&id)?;
    check_name(&name)?;
    let db = app.db.lock().await;
    crate::db::ensure_recording(&db, &id, "laptop")?;
    let dir = app.recording_dir(&id);
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(&name);
    let size = std::fs::metadata(&path).map_or(0, |m| m.len());
    if q.offset > size {
        return Ok((StatusCode::CONFLICT, Json(json!({ "size": size }))).into_response());
    }
    let skip = (size - q.offset) as usize;
    let mut size = size;
    if skip < body.len() {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        f.write_all(&body[skip..])?;
        f.sync_data()?;
        size += (body.len() - skip) as u64;
    }
    Ok(Json(json!({ "size": size })).into_response())
}

async fn put_meta(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    body: Bytes,
) -> Result<StatusCode> {
    check_name(&id)?;
    let meta: Value = serde_json::from_slice(&body)
        .map_err(|e| Error(StatusCode::BAD_REQUEST, format!("meta.json: {e}")))?;
    let db = app.db.lock().await;
    crate::db::ensure_recording(&db, &id, "laptop")?;
    if let Some(ms) = meta["started_ms"].as_i64() {
        crate::db::set_started(&db, &id, ms)?;
    }
    let dir = app.recording_dir(&id);
    std::fs::create_dir_all(&dir)?;
    write_atomic(&dir.join("meta.json"), &body)?;
    Ok(StatusCode::OK)
}

async fn finish(State(app): State<Arc<App>>, Path(id): Path<String>) -> Result<StatusCode> {
    check_name(&id)?;
    if crate::db::finish(&*app.db.lock().await, &id)? {
        Ok(StatusCode::OK)
    } else {
        Err(Error(StatusCode::NOT_FOUND, format!("no recording {id}")))
    }
}

fn write_atomic(path: &FsPath, data: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    let mut f = std::fs::File::create(&tmp)?;
    f.write_all(data)?;
    f.sync_data()?;
    std::fs::rename(tmp, path)
}

#[derive(Deserialize)]
struct UploadQuery {
    filename: Option<String>,
    mtime_ms: Option<i64>,
}

async fn upload(
    State(app): State<Arc<App>>,
    Query(q): Query<UploadQuery>,
    req: Request,
) -> Result<Response> {
    let now = Timestamp::now();
    let id = format!(
        "up-{}-{:03}",
        now.strftime("%Y%m%dT%H%M%SZ"),
        now.subsec_millisecond()
    );
    let dir = app.recording_dir(&id);
    std::fs::create_dir(&dir)?;
    let res = receive(&app, &id, &dir, q, now, req).await;
    if res.is_err() {
        let _ = std::fs::remove_dir_all(&dir);
    }
    res
}

async fn receive(
    app: &App,
    id: &str,
    dir: &FsPath,
    q: UploadQuery,
    now: Timestamp,
    req: Request,
) -> Result<Response> {
    let multipart = req
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("multipart/form-data"));
    let (filename, file) = if multipart {
        let mut mp = Multipart::from_request(req, &())
            .await
            .map_err(|e| Error(StatusCode::BAD_REQUEST, e.to_string()))?;
        loop {
            let Some(mut field) = mp.next_field().await? else {
                return Err(Error(StatusCode::BAD_REQUEST, "no file field".into()));
            };
            if field.name() != Some("file") {
                continue;
            }
            let filename = field.file_name().unwrap_or("upload").to_string();
            let (file, mut f) = create_upload(dir, &filename).await?;
            while let Some(chunk) = field.chunk().await? {
                f.write_all(&chunk).await?;
            }
            f.sync_data().await?;
            break (filename, file);
        }
    } else {
        let filename = q.filename.unwrap_or_else(|| "upload".into());
        let (file, mut f) = create_upload(dir, &filename).await?;
        let mut stream = Body::into_data_stream(req.into_body());
        while let Some(chunk) = stream.next().await {
            f.write_all(&chunk?).await?;
        }
        f.sync_data().await?;
        (filename, file)
    };

    let (creation_time, duration_ms) = probe(&dir.join(&file)).await;
    let started_ms = upload_started_ms(
        creation_time.as_deref(),
        &filename,
        q.mtime_ms,
        duration_ms,
        now.as_millisecond(),
        &app.tz,
    );
    let meta = json!({
        "id": id,
        "started_ms": started_ms,
        "app": null,
        "source": "upload",
        "filename": filename,
        "segments": [{"file": file, "key": "upload", "target": null, "offset_ms": 0, "end_ms": duration_ms}],
    });
    write_atomic(&dir.join("meta.json"), &serde_json::to_vec_pretty(&meta)?)?;
    let db = app.db.lock().await;
    crate::db::ensure_recording(&db, id, "upload")?;
    crate::db::set_started(&db, id, started_ms)?;
    crate::db::finish(&db, id)?;
    Ok((StatusCode::CREATED, Json(json!({ "id": id }))).into_response())
}

/// The stored name keeps only the extension, so ffmpeg can sniff the container.
async fn create_upload(dir: &FsPath, filename: &str) -> Result<(String, tokio::fs::File)> {
    let ext: String = FsPath::new(filename)
        .extension()
        .and_then(|e| e.to_str())
        .filter(|e| e.len() <= 8 && e.bytes().all(|b| b.is_ascii_alphanumeric()))
        .unwrap_or("bin")
        .to_ascii_lowercase();
    let file = format!("upload.{ext}");
    let f = tokio::fs::File::create(dir.join(&file)).await?;
    Ok((file, f))
}

/// `(creation_time tag, duration)` from ffprobe; missing on any failure.
async fn probe(path: &FsPath) -> (Option<String>, Option<i64>) {
    let out = tokio::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration:format_tags=creation_time",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .await;
    let Ok(out) = out else {
        return (None, None);
    };
    let v: Value = serde_json::from_slice(&out.stdout).unwrap_or_default();
    let f = &v["format"];
    let creation = f["tags"]["creation_time"].as_str().map(String::from);
    let duration = f["duration"]
        .as_str()
        .and_then(|d| d.parse::<f64>().ok())
        .map(|d| (d * 1000.0) as i64);
    (creation, duration)
}

fn upload_started_ms(
    creation_time: Option<&str>,
    filename: &str,
    mtime_ms: Option<i64>,
    duration_ms: Option<i64>,
    now_ms: i64,
    tz: &TimeZone,
) -> i64 {
    // Recorders without a clock write 1970 or 1904 epochs.
    let year_2000 = 946_684_800_000;
    if let Some(ms) = creation_time
        .and_then(|s| s.parse::<Timestamp>().ok())
        .map(|t| t.as_millisecond())
        .filter(|&ms| ms > year_2000)
    {
        return ms;
    }
    let meet = filename.as_bytes().windows(16).find_map(|w| {
        // strptime skips leading whitespace, which would shift the window.
        if !w[0].is_ascii_digit() {
            return None;
        }
        let dt = DateTime::strptime("%Y_%m_%d %H_%M", std::str::from_utf8(w).ok()?).ok()?;
        Some(dt.to_zoned(tz.clone()).ok()?.timestamp().as_millisecond())
    });
    meet.or(mtime_ms.zip(duration_ms).map(|(m, d)| m - d))
        .unwrap_or(now_ms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn app() -> (tempfile::TempDir, Arc<App>) {
        let tmp = tempfile::tempdir().unwrap();
        let app = Arc::new(App::open(tmp.path().to_path_buf()).unwrap());
        (tmp, app)
    }

    async fn send(app: &Arc<App>, method: &str, uri: &str, body: &[u8]) -> (StatusCode, Vec<u8>) {
        let req = Request::builder()
            .method(method)
            .uri(uri)
            .body(Body::from(body.to_vec()))
            .unwrap();
        let res = router(app.clone()).oneshot(req).await.unwrap();
        let status = res.status();
        (
            status,
            res.into_body().collect().await.unwrap().to_bytes().to_vec(),
        )
    }

    #[tokio::test]
    async fn byte_ranges() {
        let (tmp, app) = app();
        let url = |off| format!("/recordings/r1/files/00-mic.oga?offset={off}");

        let (s, b) = send(&app, "PUT", &url(0), b"hello").await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(b, br#"{"size":5}"#);

        // Append.
        let (s, b) = send(&app, "PUT", &url(5), b" world").await;
        assert_eq!((s, b.as_slice()), (StatusCode::OK, &br#"{"size":11}"#[..]));

        // Already have: retry of a range that landed.
        let (s, b) = send(&app, "PUT", &url(5), b" world").await;
        assert_eq!((s, b.as_slice()), (StatusCode::OK, &br#"{"size":11}"#[..]));

        // Partial overlap appends only the new tail.
        let (s, b) = send(&app, "PUT", &url(6), b"world!").await;
        assert_eq!((s, b.as_slice()), (StatusCode::OK, &br#"{"size":12}"#[..]));

        // Gap.
        let (s, b) = send(&app, "PUT", &url(20), b"x").await;
        assert_eq!(
            (s, b.as_slice()),
            (StatusCode::CONFLICT, &br#"{"size":12}"#[..])
        );

        let data = std::fs::read(tmp.path().join("recordings/r1/00-mic.oga")).unwrap();
        assert_eq!(data, b"hello world!");
    }

    #[tokio::test]
    async fn rejects_bad_names() {
        let (_tmp, app) = app();
        for uri in [
            "/recordings/r1/files/..?offset=0",
            "/recordings/r1/files/.hidden?offset=0",
            "/recordings/r1/files/meta.json?offset=0",
            "/recordings/..%2Fx/files/a?offset=0",
        ] {
            let (s, _) = send(&app, "PUT", uri, b"x").await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{uri}");
        }
    }

    #[tokio::test]
    async fn meta_and_finish() {
        let (tmp, app) = app();
        let (s, _) = send(&app, "POST", "/recordings/r1/finish", b"").await;
        assert_eq!(s, StatusCode::NOT_FOUND);

        let meta = br#"{"id":"r1","started_ms":1727179202000,"app":"Zen","segments":[]}"#;
        let (s, _) = send(&app, "PUT", "/recordings/r1/meta", meta).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(
            std::fs::read(tmp.path().join("recordings/r1/meta.json")).unwrap(),
            meta
        );

        for _ in 0..2 {
            let (s, _) = send(&app, "POST", "/recordings/r1/finish", b"").await;
            assert_eq!(s, StatusCode::OK);
        }
        let row: (i64, i64) = app
            .db
            .lock()
            .await
            .query_row(
                "SELECT started_ms, finished FROM recordings WHERE id = 'r1'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(row, (1727179202000, 1));
    }

    #[tokio::test]
    async fn raw_upload() {
        let (tmp, app) = app();
        let (s, b) = send(&app, "POST", "/recordings?filename=memo.M4A", b"audio").await;
        assert_eq!(s, StatusCode::CREATED);
        let id = serde_json::from_slice::<Value>(&b).unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string();
        let dir = tmp.path().join("recordings").join(&id);
        assert_eq!(std::fs::read(dir.join("upload.m4a")).unwrap(), b"audio");
        let meta: Value =
            serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();
        assert_eq!(meta["source"], "upload");
        assert_eq!(meta["segments"][0]["file"], "upload.m4a");
    }

    #[tokio::test]
    async fn multipart_upload() {
        let (tmp, app) = app();
        let body = "--XX\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.ogg\"\r\n\
                    Content-Type: audio/ogg\r\n\r\nOGGDATA\r\n--XX--\r\n";
        let req = Request::builder()
            .method("POST")
            .uri("/recordings")
            .header(header::CONTENT_TYPE, "multipart/form-data; boundary=XX")
            .body(Body::from(body))
            .unwrap();
        let res = router(app.clone()).oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::CREATED);
        let b = res.into_body().collect().await.unwrap().to_bytes();
        let id = serde_json::from_slice::<Value>(&b).unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string();
        let data =
            std::fs::read(tmp.path().join("recordings").join(id).join("upload.ogg")).unwrap();
        assert_eq!(data, b"OGGDATA");
    }

    #[test]
    fn start_time_rules() {
        let tz = TimeZone::get("Europe/Amsterdam").unwrap();
        let meet = "Team Sync - 2026_08_11 14_59 CEST - Recording.mp4";
        let now = 5_000_000_000_000;

        let tag = upload_started_ms(
            Some("2026-08-11T12:00:00.000000Z"),
            meet,
            None,
            None,
            now,
            &tz,
        );
        assert_eq!(
            tag,
            "2026-08-11T12:00:00Z"
                .parse::<Timestamp>()
                .unwrap()
                .as_millisecond()
        );

        // 14:59 CEST is 12:59 UTC; a bogus 1970 tag is ignored.
        let from_name = upload_started_ms(
            Some("1970-01-01T00:00:00Z"),
            meet,
            Some(1),
            Some(1),
            now,
            &tz,
        );
        assert_eq!(
            from_name,
            "2026-08-11T12:59:00Z"
                .parse::<Timestamp>()
                .unwrap()
                .as_millisecond()
        );

        assert_eq!(
            upload_started_ms(None, "memo.m4a", Some(10_000), Some(3_000), now, &tz),
            7_000
        );
        assert_eq!(
            upload_started_ms(None, "memo.m4a", None, Some(3_000), now, &tz),
            now
        );
    }
}
