use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::Result;
use serde::Serialize;
use serde_json::json;
use time::{macros::format_description, OffsetDateTime};
use tokio::process::{Child, Command};

use crate::machine::Track;

#[derive(Serialize)]
struct Segment {
    file: String,
    key: String,
    target: String,
    offset_ms: u64,
    end_ms: Option<u64>,
}

pub struct Recorder {
    pub id: String,
    pub started_ms: u64,
    dir: PathBuf,
    clock: Instant,
    app: Option<String>,
    /// key -> (index into segments, pw-record)
    running: HashMap<String, (usize, Child)>,
    segments: Vec<Segment>,
}

impl Recorder {
    pub fn new(root: &Path) -> Result<Self> {
        let now = OffsetDateTime::now_utc();
        let id = now.format(format_description!("[year][month][day]T[hour][minute][second]Z"))?;
        let dir = root.join(&id);
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            id,
            started_ms: (now.unix_timestamp_nanos() / 1_000_000) as u64,
            dir,
            clock: Instant::now(),
            app: None,
            running: HashMap::new(),
            segments: vec![],
        })
    }

    /// Stops tracks that aren't wanted (or whose target changed) and starts the missing ones.
    pub async fn sync(&mut self, want: &[Track], app: Option<&str>) -> Result<()> {
        if app.is_some() {
            self.app = app.map(String::from);
        }
        let stale: Vec<String> = self
            .running
            .iter()
            .filter(|(k, (i, _))| {
                !want
                    .iter()
                    .any(|t| &t.key == *k && t.target == self.segments[*i].target)
            })
            .map(|(k, _)| k.clone())
            .collect();
        for k in &stale {
            self.stop(k).await;
        }
        for t in want {
            if !self.running.contains_key(&t.key) {
                self.spawn(t)?;
            }
        }
        self.write_meta()
    }

    pub async fn finish(mut self) -> Result<()> {
        self.stop_all().await;
        self.write_meta()
    }

    pub async fn discard(mut self) -> Result<()> {
        self.stop_all().await;
        Ok(std::fs::remove_dir_all(&self.dir)?)
    }

    fn spawn(&mut self, t: &Track) -> Result<()> {
        let file = format!("{:02}-{}.oga", self.segments.len(), t.key);
        let child = Command::new("pw-record")
            .args([
                "--target",
                &t.target,
                "--channels",
                "1",
                "--container",
                "oga",
                "--format",
                "opus",
                "-P",
            ])
            // dont-reconnect: when an app stream goes away, don't let the session
            // manager relink the recorder to the default mic.
            .arg(format!("{{ node.name = mictap-{} node.dont-reconnect = true }}", t.key))
            .arg(self.dir.join(&file))
            .kill_on_drop(true)
            .spawn()?;
        self.running.insert(t.key.clone(), (self.segments.len(), child));
        // ponytail: offset is taken at spawn, so it's late by pw-record's startup (tens of ms).
        // Echo dedupe tolerates ±1 s; use PipeWire stream timestamps if alignment ever matters.
        self.segments.push(Segment {
            file,
            key: t.key.clone(),
            target: t.target.clone(),
            offset_ms: self.ms(),
            end_ms: None,
        });
        Ok(())
    }

    async fn stop(&mut self, key: &str) {
        let Some((i, mut child)) = self.running.remove(key) else {
            return;
        };
        // SIGINT lets pw-record flush the last Ogg page.
        if let Some(pid) = child.id() {
            unsafe { libc::kill(pid as i32, libc::SIGINT) };
        }
        let _ = child.wait().await;
        self.segments[i].end_ms = Some(self.ms());
    }

    async fn stop_all(&mut self) {
        let keys: Vec<String> = self.running.keys().cloned().collect();
        for k in &keys {
            self.stop(k).await;
        }
    }

    fn ms(&self) -> u64 {
        self.clock.elapsed().as_millis() as u64
    }

    fn write_meta(&self) -> Result<()> {
        let meta = json!({"id": self.id, "started_ms": self.started_ms, "app": self.app, "segments": self.segments});
        std::fs::write(self.dir.join("meta.json"), serde_json::to_vec_pretty(&meta)?)?;
        Ok(())
    }
}
