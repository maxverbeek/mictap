use std::{
    io::Write,
    path::{Path as FsPath, PathBuf},
    sync::Arc,
};

use axum::{
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, FromRequest, Multipart, Path, Query, Request, State},
    http::{header, HeaderMap, Method, StatusCode},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{get, post, put},
    Json, Router,
};
use futures_util::StreamExt;
use jiff::{civil::DateTime, tz::TimeZone, Timestamp};
use rusqlite::{Connection, OptionalExtension};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::{io::AsyncWriteExt, sync::Mutex};
use tower::ServiceExt;
use tower_http::services::{ServeDir, ServeFile};

use crate::names::Names;

pub struct App {
    dir: PathBuf,
    // ponytail: one lock for the db and all file appends; per-recording locks if uploads contend.
    pub(crate) db: Mutex<Connection>,
    pub(crate) tz: TimeZone,
    pub(crate) vault: PathBuf,
    /// Static files served for every path no route takes: the web page.
    web: PathBuf,
    /// The recording being diarized, and since when (Unix ms).
    pub(crate) diarizing: std::sync::Mutex<Option<(String, i64)>>,
}

impl App {
    pub fn open(dir: PathBuf) -> anyhow::Result<Self> {
        std::fs::create_dir_all(dir.join("recordings"))?;
        let db = crate::db::open(&dir.join("mictap.db"))?;
        let vault = std::env::var_os("MICTAP_VAULT").map_or_else(|| dir.join("vault"), PathBuf::from);
        let web = std::env::var_os("MICTAP_WEB").map_or_else(|| "web".into(), PathBuf::from);
        Ok(Self {
            dir,
            web,
            diarizing: Default::default(),
            db: Mutex::new(db),
            tz: TimeZone::system(),
            vault,
        })
    }

    pub(crate) fn recording_dir(&self, id: &str) -> PathBuf {
        self.dir.join("recordings").join(id)
    }
}

pub fn router(app: Arc<App>) -> Router {
    Router::new()
        .route("/recordings", post(upload).layer(DefaultBodyLimit::disable()).get(list))
        .route("/recordings/{id}", get(show).delete(remove))
        .route("/recordings/{id}/speakers", put(name_speakers))
        .route(
            "/recordings/{id}/files/{name}",
            put(put_file).layer(DefaultBodyLimit::max(16 << 20)),
        )
        .route("/recordings/{id}/meta", put(put_meta))
        .route("/recordings/{id}/finish", post(finish))
        .route("/recordings/{id}/rediarize", post(rediarize))
        .route("/recordings/{id}/outputs", get(outputs))
        .route("/r/{id}/audio.ogg", get(audio))
        .route("/upload", get(|| async { Html(UPLOAD_FORM) }))
        .fallback_service(ServeDir::new(&app.web))
        .layer(middleware::from_fn(guard))
        .with_state(app)
}

async fn guard(req: Request, next: Next) -> Response {
    if allowed(req.method(), req.headers()) {
        next.run(req).await
    } else {
        (StatusCode::FORBIDDEN, "cross-site request").into_response()
    }
}

/// There is no login, so keep browsers out: a page on another site may not change anything
/// (CSRF), and a Host outside the tailnet's names is a DNS rebinding attempt. mictap's
/// client sends neither Origin nor Sec-Fetch-Site.
/// Allowed Hosts: IP literals, single-label names, *.ts.net and MICTAP_URL's host.
fn allowed(method: &Method, headers: &HeaderMap) -> bool {
    let get = |h: &str| headers.get(h).map(|v| v.to_str().unwrap_or("?"));
    let host = get("host");
    if let Some(host) = host {
        let name = match host.rsplit_once(':') {
            Some((n, port)) if port.bytes().all(|b| b.is_ascii_digit()) => n,
            _ => host,
        };
        let name = name.trim_start_matches('[').trim_end_matches(']');
        let own = crate::vault::URL
            .split_once("://")
            .map(|(_, h)| h.split([':', '/']).next().unwrap_or(""));
        if !(name.parse::<std::net::IpAddr>().is_ok()
            || !name.contains('.')
            || name.ends_with(".ts.net")
            || own == Some(name))
        {
            return false;
        }
    }
    if matches!(*method, Method::GET | Method::HEAD) {
        return true;
    }
    if get("sec-fetch-site").is_some_and(|s| s != "same-origin" && s != "none") {
        return false;
    }
    get("origin").is_none_or(|o| o.split_once("://").map(|(_, h)| h) == host)
}

const UPLOAD_FORM: &str = r#"<!doctype html>
<meta charset="utf-8">
<title>mictap upload</title>
<form method="post" action="/recordings" enctype="multipart/form-data">
<input type="file" name="file" accept="audio/*,video/*" required>
<button>Upload</button>
</form>
"#;

const EXPIRED: &str = r#"<!doctype html>
<meta charset="utf-8">
<title>mictap: audio expired</title>
<p>This recording's audio was deleted after 30 days. The transcript stays.</p>
"#;

async fn audio(State(app): State<Arc<App>>, Path(id): Path<String>, req: Request) -> Result<Response> {
    let state: Option<String> = if valid_name(&id) {
        app.db
            .lock()
            .await
            .query_row("SELECT audio FROM recordings WHERE id = ?1", [&id], |r| r.get(0))
            .optional()?
            .flatten()
    } else {
        None
    };
    Ok(match state.as_deref() {
        Some("ready") => ServeFile::new(app.recording_dir(&id).join("audio.ogg"))
            .oneshot(req)
            .await?
            .map(Body::new),
        Some("expired") => (StatusCode::GONE, Html(EXPIRED)).into_response(),
        _ => (StatusCode::NOT_FOUND, "no audio").into_response(),
    })
}

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

/// Safe as a single path component under a recording dir.
pub(crate) fn valid_name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 128
        && s.as_bytes()[0].is_ascii_alphanumeric()
        && s.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
        && s != "meta.json"
        && s != "audio.ogg"
}

fn check_name(s: &str) -> Result<()> {
    if valid_name(s) {
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
        closed(&db, &id)?;
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path)?;
        f.write_all(&body[skip..])?;
        f.sync_data()?;
        size += (body.len() - skip) as u64;
    }
    Ok(Json(json!({ "size": size })).into_response())
}

async fn put_meta(State(app): State<Arc<App>>, Path(id): Path<String>, body: Bytes) -> Result<StatusCode> {
    check_name(&id)?;
    let meta: Value =
        serde_json::from_slice(&body).map_err(|e| Error(StatusCode::BAD_REQUEST, format!("meta.json: {e}")))?;
    let db = app.db.lock().await;
    crate::db::ensure_recording(&db, &id, "laptop")?;
    let dir = app.recording_dir(&id);
    if std::fs::read(dir.join("meta.json")).ok().as_deref() == Some(&body[..]) {
        return Ok(StatusCode::OK);
    }
    closed(&db, &id)?;
    if let Some(ms) = meta["started_ms"].as_i64() {
        crate::db::set_started(&db, &id, ms)?;
    }
    std::fs::create_dir_all(&dir)?;
    write_atomic(&dir.join("meta.json"), &body)?;
    Ok(StatusCode::OK)
}

async fn finish(State(app): State<Arc<App>>, Path(id): Path<String>) -> Result<StatusCode> {
    check_name(&id)?;
    let db = app.db.lock().await;
    if !app.recording_dir(&id).join("meta.json").exists() {
        return Err(Error(StatusCode::CONFLICT, format!("no meta.json for {id} yet")));
    }
    if crate::db::finish(&db, &id)? {
        Ok(StatusCode::OK)
    } else {
        Err(Error(StatusCode::NOT_FOUND, format!("no recording {id}")))
    }
}

/// Every recording, newest first, with what its note shows.
async fn list(State(app): State<Arc<App>>) -> Result<Json<Vec<Value>>> {
    let db = app.db.lock().await;
    type Row = (String, String, Option<i64>, String, Option<String>, Option<String>);
    let rows: Vec<Row> = db
        .prepare(
            "SELECT id, source, started_ms, status, audio, vault_path FROM recordings
             ORDER BY started_ms DESC",
        )?
        .query_map([], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?))
        })?
        .collect::<rusqlite::Result<_>>()?;
    let mut out = Vec::with_capacity(rows.len());
    for (id, source, started_ms, status, audio, transcript) in rows {
        let segs = crate::assemble::lines(&db, &id)?;
        let progress = step(&app, &db, &id, &status, &segs)?;
        let (status, done_ms, total_ms) = crate::vault::progress(&app, &db, &id, &status, &segs)?;
        out.push(json!({
            "id": id, "source": source, "date": date(&app, started_ms), "status": status, "progress": progress,
            "done_ms": done_ms, "total_ms": total_ms, "audio": audio, "transcript": transcript,
        }));
    }
    Ok(Json(out))
}

/// What a recording that isn't done or failed is at: `unstarted`, `transcribing` with a
/// `percent`, or `diarizing` with the Unix ms it started at, if it has.
fn step(app: &App, db: &Connection, id: &str, status: &str, segs: &[crate::assemble::Line]) -> Result<Value> {
    if matches!(status, "diarized" | "done" | "failed") {
        return Ok(Value::Null);
    }
    let (done, undone): (i64, i64) = db.query_row(
        "SELECT COALESCE(SUM(done), 0), COALESCE(SUM(NOT done), 0) FROM windows WHERE recording = ?1",
        [id],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    if status == "windowed" && undone == 0 {
        let since = app
            .diarizing
            .lock()
            .unwrap()
            .as_ref()
            .filter(|(d, _)| d == id)
            .map(|(_, ms)| *ms);
        return Ok(json!({ "step": "diarizing", "since_ms": since }));
    }
    if done == 0 {
        return Ok(json!({ "step": "unstarted" }));
    }
    let (_, done_ms, total_ms) = crate::vault::progress(app, db, id, status, segs)?;
    let percent = (100 * done_ms).checked_div(total_ms).unwrap_or(0);
    Ok(json!({ "step": "transcribing", "percent": percent }))
}

fn date(app: &App, started_ms: Option<i64>) -> Option<String> {
    let t = Timestamp::from_millisecond(started_ms?).ok()?;
    Some(t.to_zoned(app.tz.clone()).strftime("%Y-%m-%d %H:%M").to_string())
}

/// One recording with its lines and speaker names. `editable` once its names can be set.
async fn show(State(app): State<Arc<App>>, Path(id): Path<String>) -> Result<Json<Value>> {
    check_name(&id)?;
    let db = app.db.lock().await;
    type Row = (Option<i64>, String, Option<String>);
    let row: Option<Row> = db
        .query_row(
            "SELECT started_ms, status, audio FROM recordings WHERE id = ?1",
            [&id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .optional()?;
    let Some((started_ms, status, audio)) = row else {
        return Err(Error(StatusCode::NOT_FOUND, format!("no recording {id}")));
    };
    let segs = crate::assemble::lines(&db, &id)?;
    let editable = status == "done";
    let progress = step(&app, &db, &id, &status, &segs)?;
    let (status, done_ms, total_ms) = crate::vault::progress(&app, &db, &id, &status, &segs)?;
    let speakers = crate::names::speakers(&db, &id, segs.iter().filter_map(|l| l.speaker.as_deref()))?;
    Ok(Json(json!({
        "id": id, "date": date(&app, started_ms), "status": status, "done_ms": done_ms,
        "total_ms": total_ms, "audio": audio, "editable": editable, "progress": progress,
        "speakers": speakers, "lines": segs,
    })))
}

/// Confirms speaker names (`label -> name`; `""` leaves a label unnamed, rejecting its
/// suggestion), learns their voices and rewrites the transcript.
async fn name_speakers(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    Json(names): Json<Names>,
) -> Result<StatusCode> {
    check_name(&id)?;
    if let Some(l) = names.keys().find(|l| !crate::names::is_label(l)) {
        return Err(Error(StatusCode::BAD_REQUEST, format!("not a speaker label: {l}")));
    }
    {
        let db = app.db.lock().await;
        let status: Option<String> = db
            .query_row("SELECT status FROM recordings WHERE id = ?1", [&id], |r| r.get(0))
            .optional()?;
        match status.as_deref() {
            None => return Err(Error(StatusCode::NOT_FOUND, format!("no recording {id}"))),
            Some("done") => {}
            Some(s) => return Err(Error(StatusCode::CONFLICT, format!("{id} is {s}, not done"))),
        }
        crate::names::confirm(&db, &id, names)?;
    }
    crate::vault::write(&app, &id).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// The model outputs of a recording, until they expire (see `CONTEXT.md`).
async fn outputs(State(app): State<Arc<App>>, Path(id): Path<String>) -> Result<Json<crate::assemble::Outputs>> {
    check_name(&id)?;
    let db = app.db.lock().await;
    let exists: bool = db.query_row("SELECT EXISTS(SELECT 1 FROM recordings WHERE id = ?1)", [&id], |r| {
        r.get(0)
    })?;
    if !exists {
        return Err(Error(StatusCode::NOT_FOUND, format!("no recording {id}")));
    }
    Ok(Json(crate::assemble::outputs(&db, &id)?))
}

/// Deletes a finished recording's audio and state, including the voices learned from it. The
/// transcript stays in the vault.
async fn remove(State(app): State<Arc<App>>, Path(id): Path<String>) -> Result<StatusCode> {
    check_name(&id)?;
    let db = app.db.lock().await;
    let status: Option<String> = db
        .query_row("SELECT status FROM recordings WHERE id = ?1", [&id], |r| r.get(0))
        .optional()?;
    match status.as_deref() {
        None => return Err(Error(StatusCode::NOT_FOUND, format!("no recording {id}"))),
        Some("done" | "failed") => {}
        Some(_) => return Err(Error(StatusCode::CONFLICT, format!("{id} is still being transcribed"))),
    }
    let tx = db.unchecked_transaction()?;
    for table in [
        "segments",
        "turns",
        "lines",
        "windows",
        "file_progress",
        "clusters",
        "voices",
    ] {
        tx.execute(&format!("DELETE FROM {table} WHERE recording = ?1"), [&id])?;
    }
    tx.execute("DELETE FROM recordings WHERE id = ?1", [&id])?;
    tx.commit()?;
    match std::fs::remove_dir_all(app.recording_dir(&id)) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.into()),
        _ => Ok(StatusCode::NO_CONTENT),
    }
}

/// Runs sherpa and CAM++ on a finished recording's audio again (and whisper, if its segments
/// expired). Its turns, names and voices are dropped; its lines stay until derived anew, names
/// are pre-filled anew from other recordings' voices, and the transcript is rewritten.
async fn rediarize(State(app): State<Arc<App>>, Path(id): Path<String>) -> Result<StatusCode> {
    check_name(&id)?;
    let mut db = app.db.lock().await;
    let row: Option<(String, Option<String>)> = db
        .query_row("SELECT status, audio FROM recordings WHERE id = ?1", [&id], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .optional()?;
    match row {
        None => return Err(Error(StatusCode::NOT_FOUND, format!("no recording {id}"))),
        Some((s, _)) if s != "done" => {
            return Err(Error(StatusCode::CONFLICT, format!("{id} is {s}, not done")));
        }
        Some((_, a)) if a.as_deref() != Some("ready") => {
            return Err(Error(StatusCode::GONE, format!("the audio of {id} is gone")));
        }
        _ => {}
    }
    let tx = db.transaction()?;
    tx.execute("DELETE FROM turns WHERE recording = ?1", [&id])?;
    // whisper's segments expired: transcribe the audio again too.
    tx.execute(
        "UPDATE windows SET done = 0 WHERE recording = ?1
         AND NOT EXISTS (SELECT 1 FROM segments WHERE recording = ?1)",
        [&id],
    )?;
    tx.execute("DELETE FROM voices WHERE recording = ?1", [&id])?;
    tx.execute(
        "UPDATE recordings SET status = 'windowed', attempts = 0, retry_at = 0, speakers = NULL,
         suggested = NULL, done_ms = NULL WHERE id = ?1",
        [&id],
    )?;
    tx.commit()?;
    Ok(StatusCode::ACCEPTED)
}

/// Refuses new data for a recording that was finished (by the client, or for a week without
/// uploads): it would never be transcribed. The client keeps its copy on an error.
fn closed(db: &Connection, id: &str) -> Result<()> {
    let open: bool = db.query_row(
        "SELECT NOT finished AND status = 'receiving' FROM recordings WHERE id = ?1",
        [id],
        |r| r.get(0),
    )?;
    if open {
        Ok(())
    } else {
        Err(Error(StatusCode::GONE, format!("{id} is already finished")))
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

async fn upload(State(app): State<Arc<App>>, Query(q): Query<UploadQuery>, req: Request) -> Result<Response> {
    let now = Timestamp::now();
    let id = format!("up-{}-{:03}", now.strftime("%Y%m%dT%H%M%SZ"), now.subsec_millisecond());
    let dir = app.recording_dir(&id);
    std::fs::create_dir(&dir)?;
    let res = receive(&app, &id, &dir, q, now, req).await;
    if res.is_err() {
        let _ = std::fs::remove_dir_all(&dir);
    }
    res
}

async fn receive(app: &App, id: &str, dir: &FsPath, q: UploadQuery, now: Timestamp, req: Request) -> Result<Response> {
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
    meet.or(mtime_ms.zip(duration_ms).map(|(m, d)| m - d)).unwrap_or(now_ms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::Request;
    use http_body_util::BodyExt;

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
        (status, res.into_body().collect().await.unwrap().to_bytes().to_vec())
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
        assert_eq!((s, b.as_slice()), (StatusCode::CONFLICT, &br#"{"size":12}"#[..]));

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
        // Files, then meta, then finish: finishing without meta.json is refused.
        let (s, _) = send(&app, "PUT", "/recordings/r1/files/00-mic.oga?offset=0", b"x").await;
        assert_eq!(s, StatusCode::OK);
        let (s, _) = send(&app, "POST", "/recordings/r1/finish", b"").await;
        assert_eq!(s, StatusCode::CONFLICT);

        let meta = br#"{"id":"r1","started_ms":1727179202000,"app":"Zen","segments":[]}"#;
        let (s, _) = send(&app, "PUT", "/recordings/r1/meta", meta).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(std::fs::read(tmp.path().join("recordings/r1/meta.json")).unwrap(), meta);

        for _ in 0..2 {
            let (s, _) = send(&app, "POST", "/recordings/r1/finish", b"").await;
            assert_eq!(s, StatusCode::OK);
        }
        let row: (i64, i64) = app
            .db
            .lock()
            .await
            .query_row("SELECT started_ms, finished FROM recordings WHERE id = 'r1'", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap();
        assert_eq!(row, (1727179202000, 1));
    }

    #[tokio::test]
    async fn lists_and_deletes() {
        let (tmp, app) = app();
        send(&app, "PUT", "/recordings/r1/files/00-mic.oga?offset=0", b"abc").await;
        send(
            &app,
            "PUT",
            "/recordings/r1/meta",
            br#"{"id":"r1","started_ms":1,"segments":[]}"#,
        )
        .await;

        let (s, b) = send(&app, "GET", "/recordings", b"").await;
        assert_eq!(s, StatusCode::OK);
        let list: Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(list[0]["id"], "r1");
        assert_eq!(list[0]["status"], "transcribing");

        let (s, _) = send(&app, "DELETE", "/recordings/r1", b"").await;
        assert_eq!(s, StatusCode::CONFLICT, "still transcribing");

        app.db
            .lock()
            .await
            .execute("UPDATE recordings SET status = 'done'", [])
            .unwrap();
        let (_, b) = send(&app, "GET", "/recordings", b"").await;
        let list: Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(list[0]["status"], "done");
        let (s, _) = send(&app, "DELETE", "/recordings/r1", b"").await;
        assert_eq!(s, StatusCode::NO_CONTENT);
        assert!(!tmp.path().join("recordings/r1").exists());
        let (_, b) = send(&app, "GET", "/recordings", b"").await;
        assert_eq!(b, b"[]");
        let (s, _) = send(&app, "DELETE", "/recordings/r1", b"").await;
        assert_eq!(s, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn deletes_a_recording_with_named_speakers() {
        let (_tmp, app) = app();
        app.db
            .lock()
            .await
            .execute_batch(
                "INSERT INTO recordings (id, source, status) VALUES ('r1', 'laptop', 'done');
                 INSERT INTO clusters (recording, label, embedding) VALUES ('r1', 'room/S1', x'00');
                 INSERT INTO voices VALUES ('Max', x'00', 'r1', 'room/S1');",
            )
            .unwrap();
        let (s, b) = send(&app, "DELETE", "/recordings/r1", b"").await;
        assert_eq!(s, StatusCode::NO_CONTENT, "{}", String::from_utf8_lossy(&b));
    }

    #[tokio::test]
    async fn rediarize_resets_labels_names_and_voices() {
        let (_tmp, app) = app();
        {
            let db = app.db.lock().await;
            db.execute_batch(
                "INSERT INTO recordings (id, source, status, audio, speakers) VALUES
                   ('r1', 'laptop', 'done', 'ready', '{\"room/S1\":\"Max\"}'),
                   ('r2', 'laptop', 'transcribing', 'ready', NULL),
                   ('r3', 'laptop', 'done', 'expired', NULL), ('r4', 'laptop', 'done', 'ready', NULL);
                 INSERT INTO windows (id, recording, file, track, offset_ms, start_ms, end_ms, done) VALUES
                   (1, 'r1', '00-mic.oga', 'room', 0, 0, 1000, 1), (2, 'r4', '00-mic.oga', 'room', 0, 0, 1000, 1);
                 INSERT INTO segments (recording, window, track, start_ms, end_ms, text)
                   VALUES ('r1', 1, 'room', 0, 1000, 'a');
                 INSERT INTO turns (recording, track, start_ms, end_ms, speaker)
                   VALUES ('r1', 'room', 0, 1000, 0);
                 INSERT INTO lines (recording, track, start_ms, end_ms, text, speaker)
                   VALUES ('r1', 'room', 0, 1000, 'a', 'room/S1');
                 INSERT INTO clusters (recording, label, embedding) VALUES ('r1', 'room/S1', x'00'), ('r3', 'room/S1', x'00');
                 INSERT INTO voices VALUES ('Max', x'00', 'r1', 'room/S1'), ('Max', x'00', 'r3', 'room/S1');",
            )
            .unwrap();
        }
        for (id, want) in [
            ("r2", StatusCode::CONFLICT),
            ("r3", StatusCode::GONE),
            ("nope", StatusCode::NOT_FOUND),
            ("r1", StatusCode::ACCEPTED),
            ("r4", StatusCode::ACCEPTED),
        ] {
            let (s, _) = send(&app, "POST", &format!("/recordings/{id}/rediarize"), b"").await;
            assert_eq!(s, want, "{id}");
        }

        let db = app.db.lock().await;
        let one = |sql: &str| -> String { db.query_row(sql, [], |r| r.get(0)).unwrap() };
        assert_eq!(
            one("SELECT status || '/' || attempts || '/' || COALESCE(speakers, '-') FROM recordings WHERE id = 'r1'"),
            "windowed/0/-"
        );
        assert_eq!(
            one("SELECT group_concat(speaker) FROM lines"),
            "room/S1",
            "until derived anew"
        );
        assert_eq!(
            one("SELECT group_concat(recording || ':' || done) FROM windows"),
            "r1:1,r4:0",
            "r4's segments expired: transcribed again"
        );
        assert_eq!(one("SELECT CAST(COUNT(*) AS TEXT) FROM turns"), "0");
        assert_eq!(
            one("SELECT group_concat(recording) FROM clusters"),
            "r1,r3",
            "until derived anew"
        );
        assert_eq!(one("SELECT group_concat(recording) FROM voices"), "r3");
    }

    #[tokio::test]
    async fn names_speakers_learns_voices_and_rewrites_the_note() {
        let tmp = tempfile::tempdir().unwrap();
        let mut app = App::open(tmp.path().join("state")).unwrap();
        app.vault = tmp.path().join("vault");
        std::fs::create_dir(&app.vault).unwrap();
        let app = Arc::new(app);
        app.db
            .lock()
            .await
            .execute_batch(
                r#"INSERT INTO recordings (id, source, started_ms, status, speakers)
                     VALUES ('r1', 'laptop', 0, 'done', '{"room/S2":"Eva"}'), ('r2', 'laptop', 0, 'windowed', NULL);
                   INSERT INTO windows (id, recording, file, track, offset_ms, start_ms, end_ms, done)
                     VALUES (1, 'r1', 'a', 'room', 0, 0, 9000, 1);
                   INSERT INTO lines (recording, track, start_ms, end_ms, text, speaker) VALUES
                     ('r1', 'room', 0, 1000, 'Hoi.', 'room/S1'), ('r1', 'room', 1000, 2000, 'Ja.', 'room/S2');
                   INSERT INTO clusters (recording, label, embedding) VALUES ('r1', 'room/S1', x'01'), ('r1', 'room/S2', x'02');
                   INSERT INTO voices VALUES ('Eva', x'02', 'r1', 'room/S2');"#,
            )
            .unwrap();
        let put = |id: &str, body: &str| {
            let req = Request::builder()
                .method("PUT")
                .uri(format!("/recordings/{id}/speakers"))
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .unwrap();
            let res = router(app.clone()).oneshot(req);
            async move { res.await.unwrap().status() }
        };
        assert_eq!(put("r1", r#"{"bad":"x"}"#).await, StatusCode::BAD_REQUEST);
        assert_eq!(put("r2", r#"{"room/S1":"Max"}"#).await, StatusCode::CONFLICT);
        assert_eq!(put("nope", r#"{"room/S1":"Max"}"#).await, StatusCode::NOT_FOUND);
        assert_eq!(
            put("r1", r#"{"room/S1":" Max ","room/S2":""}"#).await,
            StatusCode::NO_CONTENT
        );

        let db = app.db.lock().await;
        let one = |sql: &str| -> String { db.query_row(sql, [], |r| r.get(0)).unwrap() };
        assert_eq!(
            one("SELECT speakers FROM recordings WHERE id = 'r1'"),
            r#"{"room/S1":"Max"}"#
        );
        assert_eq!(
            one("SELECT group_concat(name || ':' || hex(embedding)) FROM voices"),
            "Max:01"
        );
        let note = std::fs::read_dir(&app.vault).unwrap().next().unwrap().unwrap().path();
        let text = std::fs::read_to_string(note).unwrap();
        assert!(text.contains("attendees: [\"[[Max]]\"]\n"), "{text}");
        assert!(
            text.contains("**Max** (room, ") && text.contains("**S2** (room, "),
            "{text}"
        );
    }

    #[tokio::test]
    async fn reports_the_step_until_done() {
        let (_tmp, app) = app();
        app.db
            .lock()
            .await
            .execute_batch(
                "INSERT INTO recordings (id, source, started_ms, status) VALUES
                   ('new', 'laptop', 4, 'receiving'), ('half', 'laptop', 3, 'windowed'),
                   ('dia', 'laptop', 2, 'windowed'), ('done', 'laptop', 1, 'done');
                 INSERT INTO windows (recording, file, track, offset_ms, start_ms, end_ms, done) VALUES
                   ('new', 'a', 'room', 0, 0, 10000, 0),
                   ('half', 'a', 'room', 0, 0, 10000, 1), ('half', 'a', 'room', 0, 10000, 40000, 0),
                   ('dia', 'a', 'room', 0, 0, 10000, 1), ('done', 'a', 'room', 0, 0, 10000, 1);",
            )
            .unwrap();
        let dir = app.recording_dir("half");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("meta.json"),
            r#"{"segments":[{"file":"a","offset_ms":0,"end_ms":40000}]}"#,
        )
        .unwrap();
        *app.diarizing.lock().unwrap() = Some(("dia".into(), 1234));
        let (_, b) = send(&app, "GET", "/recordings", b"").await;
        let list: Value = serde_json::from_slice(&b).unwrap();
        let got: Vec<&Value> = list.as_array().unwrap().iter().map(|r| &r["progress"]).collect();
        assert_eq!(
            got,
            [
                &json!({"step": "unstarted"}),
                &json!({"step": "transcribing", "percent": 25}),
                &json!({"step": "diarizing", "since_ms": 1234}),
                &Value::Null,
            ]
        );
    }

    #[tokio::test]
    async fn exports_model_outputs() {
        let (_tmp, app) = app();
        app.db
            .lock()
            .await
            .execute_batch(
                "INSERT INTO recordings (id, source) VALUES ('r1', 'laptop');
                 INSERT INTO windows (id, recording, file, track, offset_ms, start_ms, end_ms)
                   VALUES (1, 'r1', 'a', 'room', 0, 0, 1);
                 INSERT INTO segments (recording, window, track, start_ms, end_ms, text)
                   VALUES ('r1', 1, 'room', 0, 900, 'hoi');
                 INSERT INTO turns (recording, track, start_ms, end_ms, speaker, embedding)
                   VALUES ('r1', 'room', 0, 1000, 0, x'0000803f');",
            )
            .unwrap();
        let (s, b) = send(&app, "GET", "/recordings/r1/outputs", b"").await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(
            serde_json::from_slice::<Value>(&b).unwrap(),
            json!({"tracks": {"room": {
                "segments": [{"start_ms": 0, "end_ms": 900, "text": "hoi"}],
                "turns": [{"start_ms": 0, "end_ms": 1000, "speaker": 0, "embedding": [1.0]}],
            }}})
        );
        assert_eq!(
            send(&app, "GET", "/recordings/nope/outputs", b"").await.0,
            StatusCode::NOT_FOUND
        );
    }

    #[tokio::test]
    async fn finished_takes_no_new_data() {
        let (tmp, app) = app();
        let meta = br#"{"id":"r1","started_ms":1,"segments":[]}"#;
        send(&app, "PUT", "/recordings/r1/files/00-mic.oga?offset=0", b"abc").await;
        send(&app, "PUT", "/recordings/r1/meta", meta).await;
        // As windows::tick does to a recording idle for a week.
        crate::db::finish(&*app.db.lock().await, "r1").unwrap();

        let (s, b) = send(&app, "PUT", "/recordings/r1/files/00-mic.oga?offset=0", b"abc").await;
        assert_eq!(
            (s, b.as_slice()),
            (StatusCode::OK, &br#"{"size":3}"#[..]),
            "retry of what landed"
        );
        let (s, _) = send(&app, "PUT", "/recordings/r1/files/00-mic.oga?offset=3", b"def").await;
        assert_eq!(s, StatusCode::GONE);
        let (s, _) = send(&app, "PUT", "/recordings/r1/files/01-app.oga?offset=0", b"x").await;
        assert_eq!(s, StatusCode::GONE);
        let (s, _) = send(&app, "PUT", "/recordings/r1/meta", meta).await;
        assert_eq!(s, StatusCode::OK);
        let (s, _) = send(&app, "PUT", "/recordings/r1/meta", br#"{"id":"r1","segments":[1]}"#).await;
        assert_eq!(s, StatusCode::GONE);
        let (s, _) = send(&app, "POST", "/recordings/r1/finish", b"").await;
        assert_eq!(s, StatusCode::OK);

        let dir = tmp.path().join("recordings/r1");
        assert_eq!(std::fs::read(dir.join("00-mic.oga")).unwrap(), b"abc");
        assert!(!dir.join("01-app.oga").exists());
        assert_eq!(std::fs::read(dir.join("meta.json")).unwrap(), meta);
    }

    #[test]
    fn keeps_browsers_out() {
        let ok = |m: Method, h: &[(&str, &str)]| {
            let mut headers = HeaderMap::new();
            for (k, v) in h {
                headers.insert(
                    axum::http::HeaderName::from_bytes(k.as_bytes()).unwrap(),
                    v.parse().unwrap(),
                );
            }
            allowed(&m, &headers)
        };
        let host = ("host", "homeserver:8765");
        // mictap's client, curl, the upload form, a link opened from anywhere.
        assert!(ok(Method::PUT, &[host]));
        assert!(ok(Method::POST, &[]));
        assert!(ok(
            Method::POST,
            &[
                host,
                ("origin", "http://homeserver:8765"),
                ("sec-fetch-site", "same-origin")
            ]
        ));
        assert!(ok(Method::GET, &[host, ("sec-fetch-site", "cross-site")]));
        for h in [
            "100.64.0.3:8765",
            "[fd7a::1]:8765",
            "127.0.0.1",
            "homeserver.tail1234.ts.net:8765",
        ] {
            assert!(ok(Method::GET, &[("host", h)]), "{h}");
        }
        // CSRF from a page on another site, fetch no-cors or a form post.
        assert!(!ok(Method::POST, &[host, ("sec-fetch-site", "cross-site")]));
        assert!(!ok(Method::POST, &[host, ("sec-fetch-site", "same-site")]));
        assert!(!ok(Method::POST, &[host, ("origin", "https://evil.example")]));
        assert!(!ok(Method::POST, &[host, ("origin", "null")]));
        // DNS rebinding: evil.example resolves to homeserver.
        assert!(!ok(Method::GET, &[("host", "evil.example:8765")]));
    }

    #[tokio::test]
    async fn guard_answers_403() {
        let (_tmp, app) = app();
        let req = Request::builder()
            .method("POST")
            .uri("/recordings/r1/finish")
            .header(header::ORIGIN, "https://evil.example")
            .body(Body::empty())
            .unwrap();
        let res = router(app).oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::FORBIDDEN);
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
        let meta: Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();
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
        let data = std::fs::read(tmp.path().join("recordings").join(id).join("upload.ogg")).unwrap();
        assert_eq!(data, b"OGGDATA");
    }

    #[tokio::test]
    async fn serves_audio_ranges_and_expired() {
        let (tmp, app) = app();
        let dir = tmp.path().join("recordings/r1");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("audio.ogg"), b"OggS0123456789").unwrap();
        app.db
            .lock()
            .await
            .execute_batch(
                "INSERT INTO recordings (id, source, audio) VALUES
                   ('r1', 'laptop', 'ready'), ('r2', 'laptop', 'expired'), ('r3', 'laptop', NULL);",
            )
            .unwrap();
        let get = |uri: &str, range: Option<&str>| {
            let mut req = Request::builder().uri(uri);
            if let Some(r) = range {
                req = req.header(header::RANGE, r);
            }
            router(app.clone()).oneshot(req.body(Body::empty()).unwrap())
        };

        let res = get("/r/r1/audio.ogg", Some("bytes=4-7")).await.unwrap();
        assert_eq!(res.status(), StatusCode::PARTIAL_CONTENT);
        assert_eq!(res.headers()[header::CONTENT_TYPE], "audio/ogg");
        let body = res.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(&body[..], b"0123");

        let res = get("/r/r1/audio.ogg", None).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);

        let res = get("/r/r2/audio.ogg", None).await.unwrap();
        assert_eq!(res.status(), StatusCode::GONE);
        let body = res.into_body().collect().await.unwrap().to_bytes();
        assert!(String::from_utf8_lossy(&body).contains("deleted after 30 days"));

        for uri in ["/r/r3/audio.ogg", "/r/nope/audio.ogg", "/r/..%2Fr1/audio.ogg"] {
            let res = get(uri, None).await.unwrap();
            assert_eq!(res.status(), StatusCode::NOT_FOUND, "{uri}");
        }
    }

    #[tokio::test]
    async fn serves_the_web_dir_behind_the_api() {
        let tmp = tempfile::tempdir().unwrap();
        let mut app = App::open(tmp.path().to_path_buf()).unwrap();
        app.web = tmp.path().join("web");
        std::fs::create_dir(&app.web).unwrap();
        std::fs::write(app.web.join("index.html"), "page").unwrap();
        let app = Arc::new(app);
        assert_eq!(send(&app, "GET", "/", b"").await, (StatusCode::OK, b"page".to_vec()));
        assert_eq!(
            send(&app, "GET", "/recordings", b"").await,
            (StatusCode::OK, b"[]".to_vec())
        );
        assert_eq!(send(&app, "GET", "/nope.js", b"").await.0, StatusCode::NOT_FOUND);
    }

    #[test]
    fn start_time_rules() {
        let tz = TimeZone::get("Europe/Amsterdam").unwrap();
        let meet = "Team Sync - 2026_08_11 14_59 CEST - Recording.mp4";
        let now = 5_000_000_000_000;

        let tag = upload_started_ms(Some("2026-08-11T12:00:00.000000Z"), meet, None, None, now, &tz);
        assert_eq!(
            tag,
            "2026-08-11T12:00:00Z".parse::<Timestamp>().unwrap().as_millisecond()
        );

        // 14:59 CEST is 12:59 UTC; a bogus 1970 tag is ignored.
        let from_name = upload_started_ms(Some("1970-01-01T00:00:00Z"), meet, Some(1), Some(1), now, &tz);
        assert_eq!(
            from_name,
            "2026-08-11T12:59:00Z".parse::<Timestamp>().unwrap().as_millisecond()
        );

        assert_eq!(
            upload_started_ms(None, "memo.m4a", Some(10_000), Some(3_000), now, &tz),
            7_000
        );
        assert_eq!(upload_started_ms(None, "memo.m4a", None, Some(3_000), now, &tz), now);
    }
}
