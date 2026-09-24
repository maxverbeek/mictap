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
    finished: bool,
}

impl Recorder {
    pub fn new(root: &Path) -> Result<Self> {
        let now = OffsetDateTime::now_utc();
        let stamp = now.format(format_description!("[year][month][day]T[hour][minute][second]Z"))?;
        std::fs::create_dir_all(root)?;
        // A stop and start within one second must not share (and truncate) a directory.
        let (id, dir) = (1..)
            .map(|n| if n == 1 { stamp.clone() } else { format!("{stamp}-{n}") })
            .find_map(|id| match std::fs::create_dir(root.join(&id)) {
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                r => Some(r.map(|()| (id.clone(), root.join(id)))),
            })
            .unwrap()?;
        Ok(Self {
            id,
            started_ms: (now.unix_timestamp_nanos() / 1_000_000) as u64,
            dir,
            clock: Instant::now(),
            app: None,
            running: HashMap::new(),
            segments: vec![],
            finished: false,
        })
    }

    /// Stops tracks that aren't wanted (or whose target changed) and starts the missing ones.
    pub async fn sync(&mut self, want: &[Track], app: Option<&str>) -> Result<()> {
        if app.is_some() {
            self.app = app.map(String::from);
        }
        // A pw-record that quit (PipeWire restarted) is restarted as a new segment.
        let dead: Vec<String> = self
            .running
            .iter_mut()
            .filter_map(|(k, (_, c))| (!matches!(c.try_wait(), Ok(None))).then(|| k.clone()))
            .collect();
        for k in &dead {
            self.stop(k).await;
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
        self.finished = true;
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
        let meta = json!({
            "id": self.id,
            "started_ms": self.started_ms,
            "app": self.app,
            "finished": self.finished,
            "segments": self.segments,
        });
        std::fs::write(self.dir.join("meta.json"), serde_json::to_vec_pretty(&meta)?)?;
        Ok(())
    }
}

/// Marks every recording under `root` finished: at daemon start none is live, so an
/// unfinished one was cut off by a crash. Its open segments keep `end_ms: null`.
pub fn close_orphans(root: &Path) -> Result<()> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Ok(());
    };
    for entry in entries {
        let path = entry?.path().join("meta.json");
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        let mut meta: serde_json::Value = serde_json::from_slice(&bytes)?;
        if meta["finished"] != true {
            meta["finished"] = true.into();
            std::fs::write(&path, serde_json::to_vec_pretty(&meta)?)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique_within_a_second() {
        let root = std::env::temp_dir().join(format!("mictap-test-{}", std::process::id()));
        let a = Recorder::new(&root).unwrap();
        let b = Recorder::new(&root).unwrap();
        let c = Recorder::new(&root).unwrap();
        assert!(a.id != b.id && b.id != c.id && a.id != c.id);
        assert!(root.join(&c.id).is_dir());

        a.write_meta().unwrap();
        close_orphans(&root).unwrap();
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(a.dir.join("meta.json")).unwrap()).unwrap();
        assert_eq!(meta["finished"], true);
        std::fs::remove_dir_all(&root).unwrap();
    }
}
