use std::{
    collections::HashMap,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{bail, Context, Result};
use reqwest::StatusCode;
use serde::Deserialize;

/// Nothing leaves the laptop during a recording's first minute, so Discard leaves no trace.
const HOLD_MS: u64 = 60_000;
const RANGE: u64 = 1 << 20;
const TICK: Duration = Duration::from_secs(10);
const MAX_BACKOFF: Duration = Duration::from_secs(300);

#[derive(Deserialize)]
struct Meta {
    started_ms: u64,
    #[serde(default)]
    finished: bool,
    segments: Vec<Seg>,
}

#[derive(Deserialize)]
struct Seg {
    file: String,
}

#[derive(Deserialize)]
struct Size {
    size: u64,
}

pub fn server() -> String {
    std::env::var("MICTAP_SERVER").unwrap_or_else(|_| "http://homeserver:8765".into())
}

/// Posts a whole audio or video file; returns the server's reply (`{"id": ...}`).
pub async fn whole(server: &str, path: &Path) -> Result<String> {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .context("file name is not UTF-8")?;
    let f = tokio::fs::File::open(path)
        .await
        .with_context(|| path.display().to_string())?;
    let mtime_ms = f.metadata().await?.modified()?.duration_since(UNIX_EPOCH)?.as_millis();
    let res = reqwest::Client::new()
        .post(format!("{server}/recordings"))
        .query(&[("filename", name), ("mtime_ms", &mtime_ms.to_string())])
        .body(f)
        .send()
        .await?;
    Ok(check(res).await?.text().await?)
}

pub async fn run(root: PathBuf) {
    let server = server();
    let mut up = Uploader::new(server, root);
    let mut wait = TICK;
    loop {
        tokio::time::sleep(wait).await;
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;
        wait = match up.pass(now).await {
            Ok(()) => TICK,
            Err(e) => {
                eprintln!("upload: {e:#}");
                (wait * 2).min(MAX_BACKOFF)
            }
        };
    }
}

pub struct Uploader {
    http: reqwest::Client,
    server: String,
    root: PathBuf,
    /// (id, file) -> bytes the server has. Unknown after a restart: the server's
    /// reply to the first range (already-have or 409) says where to continue.
    sent: HashMap<(String, String), u64>,
    /// id -> the meta.json the server has
    meta: HashMap<String, Vec<u8>>,
}

impl Uploader {
    pub fn new(server: String, root: PathBuf) -> Self {
        Self {
            http: reqwest::Client::new(),
            server,
            root,
            sent: HashMap::new(),
            meta: HashMap::new(),
        }
    }

    /// Sends every recording past the hold; returns the first error after trying them all.
    pub async fn pass(&mut self, now_ms: u64) -> Result<()> {
        let Ok(entries) = std::fs::read_dir(&self.root) else {
            return Ok(());
        };
        let mut ids: Vec<String> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .filter_map(|e| e.file_name().into_string().ok())
            .collect();
        ids.sort();
        let mut res = Ok(());
        for id in ids {
            if let Err(e) = self.recording(&id, now_ms).await {
                res = res.and(Err(e.context(id)));
            }
        }
        res
    }

    async fn recording(&mut self, id: &str, now_ms: u64) -> Result<()> {
        let dir = self.root.join(id);
        let bytes = match std::fs::read(dir.join("meta.json")) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            r => r.context("meta.json")?,
        };
        let meta: Meta = serde_json::from_slice(&bytes).context("meta.json")?;
        if !meta.finished && now_ms < meta.started_ms + HOLD_MS {
            return Ok(());
        }
        for seg in &meta.segments {
            self.file(id, &dir, &seg.file).await?;
        }
        if self.meta.get(id) != Some(&bytes) {
            let res = self
                .http
                .put(format!("{}/recordings/{id}/meta", self.server))
                .body(bytes.clone())
                .send()
                .await?;
            check(res).await.context("meta.json")?;
            self.meta.insert(id.into(), bytes);
        }
        if meta.finished {
            let res = self
                .http
                .post(format!("{}/recordings/{id}/finish", self.server))
                .send()
                .await?;
            check(res).await.context("finish")?;
            std::fs::remove_dir_all(&dir)?;
            self.meta.remove(id);
            self.sent.retain(|(i, _), _| i != id);
        }
        Ok(())
    }

    async fn file(&mut self, id: &str, dir: &Path, name: &str) -> Result<()> {
        let path = dir.join(name);
        let len = match std::fs::metadata(&path) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            r => r?.len(),
        };
        let key = (id.to_string(), name.to_string());
        let mut off = self.sent.get(&key).copied().unwrap_or(0);
        while off < len {
            let mut buf = vec![0; RANGE.min(len - off) as usize];
            let mut f = std::fs::File::open(&path)?;
            f.seek(SeekFrom::Start(off))?;
            f.read_exact(&mut buf)?;
            let res = self
                .http
                .put(format!("{}/recordings/{id}/files/{name}?offset={off}", self.server))
                .body(buf)
                .send()
                .await?;
            let res = if res.status() == StatusCode::CONFLICT {
                res
            } else {
                check(res).await.context(name.to_string())?
            };
            off = res.json::<Size>().await?.size;
            self.sent.insert(key.clone(), off);
        }
        Ok(())
    }
}

async fn check(res: reqwest::Response) -> Result<reqwest::Response> {
    if res.status().is_success() {
        return Ok(res);
    }
    bail!("{}: {}", res.status(), res.text().await.unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use axum::{
        body::Bytes,
        extract::{Path as P, Query, State},
        http::StatusCode,
        routing::{post, put},
        Json, Router,
    };
    use serde_json::json;

    use super::*;

    #[derive(Default)]
    struct Mock {
        files: HashMap<String, Vec<u8>>,
        meta: Option<Vec<u8>>,
        finished: bool,
    }

    type S = State<Arc<Mutex<Mock>>>;

    #[derive(Deserialize)]
    struct Off {
        offset: usize,
    }

    /// The byte-range protocol of the Decisions table, for one recording.
    async fn mock() -> (String, Arc<Mutex<Mock>>) {
        let state = Arc::new(Mutex::new(Mock::default()));
        let app = Router::new()
            .route(
                "/recordings/{id}/files/{name}",
                put(
                    |State(s): S, P((_, name)): P<(String, String)>, Query(q): Query<Off>, body: Bytes| async move {
                        let mut s = s.lock().unwrap();
                        let f = s.files.entry(name).or_default();
                        if q.offset > f.len() {
                            return (StatusCode::CONFLICT, Json(json!({"size": f.len()})));
                        }
                        let skip = f.len() - q.offset;
                        if skip < body.len() {
                            f.extend_from_slice(&body[skip..]);
                        }
                        (StatusCode::OK, Json(json!({"size": f.len()})))
                    },
                ),
            )
            .route(
                "/recordings/{id}/meta",
                put(|State(s): S, body: Bytes| async move {
                    s.lock().unwrap().meta = Some(body.to_vec());
                }),
            )
            .route(
                "/recordings/{id}/finish",
                post(|State(s): S| async move {
                    let mut s = s.lock().unwrap();
                    if s.meta.is_none() {
                        return StatusCode::CONFLICT;
                    }
                    s.finished = true;
                    StatusCode::OK
                }),
            )
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (url, state)
    }

    fn meta(dir: &Path, finished: bool) {
        let m = json!({"id": "r", "started_ms": 1_000_000, "finished": finished,
            "segments": [{"file": "00-mic.oga"}, {"file": "01-app-7.oga"}]});
        std::fs::write(dir.join("meta.json"), m.to_string()).unwrap();
    }

    fn append(path: &Path, n: usize, seed: u8) {
        use std::io::Write;
        let data: Vec<u8> = (0..n).map(|i| (i as u8).wrapping_mul(31) ^ seed).collect();
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap();
        f.write_all(&data).unwrap();
    }

    #[tokio::test]
    async fn uploads_after_hold_resumes_and_finishes() {
        let (url, server) = mock().await;
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("r");
        std::fs::create_dir(&dir).unwrap();
        let (mic, app) = (dir.join("00-mic.oga"), dir.join("01-app-7.oga"));
        append(&mic, 2_500_000, 1);
        append(&app, 10, 2);
        meta(&dir, false);
        let same = |s: &Mock| {
            s.files["00-mic.oga"] == std::fs::read(&mic).unwrap()
                && s.files["01-app-7.oga"] == std::fs::read(&app).unwrap()
        };

        let mut up = Uploader::new(url.clone(), root.path().into());
        up.pass(1_059_999).await.unwrap();
        assert!(server.lock().unwrap().files.is_empty() && server.lock().unwrap().meta.is_none());

        up.pass(1_060_000).await.unwrap();
        assert!(same(&server.lock().unwrap()));
        assert!(server.lock().unwrap().meta.is_some() && !server.lock().unwrap().finished);

        // Daemon restart: the already-have replies skip what the server has.
        append(&mic, 1_500_000, 3);
        let mut up = Uploader::new(url, root.path().into());
        up.pass(2_000_000).await.unwrap();
        assert!(same(&server.lock().unwrap()));

        // The server lost the tail: its 409 on the next range rewinds the offset.
        server
            .lock()
            .unwrap()
            .files
            .get_mut("00-mic.oga")
            .unwrap()
            .truncate(1000);
        append(&mic, 10, 5);
        append(&app, 5000, 4);
        meta(&dir, true);
        let expected = (std::fs::read(&mic).unwrap(), std::fs::read(&app).unwrap());
        up.pass(2_000_000).await.unwrap();
        let s = server.lock().unwrap();
        assert!(s.files["00-mic.oga"] == expected.0 && s.files["01-app-7.oga"] == expected.1);
        assert!(
            s.finished
                && s.meta
                    .as_deref()
                    .is_some_and(|m| m.windows(15).any(|w| w == b"\"finished\":true"))
        );
        assert!(!dir.exists());
    }

    #[tokio::test]
    async fn whole_file_posts_name_mtime_and_body() {
        type Got = Arc<Mutex<Option<(HashMap<String, String>, Vec<u8>)>>>;
        let got: Got = Default::default();
        let app = Router::new()
            .route(
                "/recordings",
                post(
                    |State(g): State<Got>, Query(q): Query<HashMap<String, String>>, body: Bytes| async move {
                        *g.lock().unwrap() = Some((q, body.to_vec()));
                        (StatusCode::CREATED, Json(json!({"id": "up-1"})))
                    },
                ),
            )
            .layer(axum::extract::DefaultBodyLimit::disable())
            .with_state(got.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("Meet 2026_08_11 14_59 CEST.mp4");
        append(&path, 3_000_000, 7);
        assert_eq!(whole(&url, &path).await.unwrap(), r#"{"id":"up-1"}"#);
        let (q, body) = got.lock().unwrap().take().unwrap();
        assert_eq!(q["filename"], "Meet 2026_08_11 14_59 CEST.mp4");
        let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
        assert_eq!(
            q["mtime_ms"],
            mtime.duration_since(UNIX_EPOCH).unwrap().as_millis().to_string()
        );
        assert!(body == std::fs::read(&path).unwrap());
    }

    #[tokio::test]
    async fn offline_keeps_the_spool() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("r");
        std::fs::create_dir(&dir).unwrap();
        append(&dir.join("00-mic.oga"), 100, 0);
        meta(&dir, true);
        let mut up = Uploader::new("http://127.0.0.1:1".into(), root.path().into());
        assert!(up.pass(0).await.is_err());
        assert!(dir.join("00-mic.oga").exists());
    }
}
