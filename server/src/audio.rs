use std::{
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};

use anyhow::{bail, Context, Result};
use rusqlite::{params, OptionalExtension};
use tokio::process::Command;

use crate::{
    api::{valid_name, App},
    windows::Meta,
};

const RETENTION: Duration = Duration::from_secs(30 * 24 * 3600);
const DAY: Duration = Duration::from_secs(24 * 3600);

/// Mixes every file of `id` that has audio, at its offset, into `audio.ogg` (Opus).
async fn mixdown(app: &App, id: &str) -> Result<()> {
    let dir = app.recording_dir(id);
    let meta: Meta =
        serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?).context("meta.json")?;
    let decoded: Vec<String> = app
        .db
        .lock()
        .await
        .prepare("SELECT file FROM file_progress WHERE recording = ?1 AND done_ms > 0")?
        .query_map([id], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    let files: Vec<_> = meta
        .segments
        .iter()
        .filter(|s| valid_name(&s.file) && decoded.contains(&s.file))
        .collect();
    if files.is_empty() {
        bail!("no audio");
    }
    let tmp = dir.join(".audio.ogg");
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-nostdin", "-v", "error", "-y"]);
    for s in &files {
        cmd.arg("-i").arg(dir.join(&s.file));
    }
    let offsets: Vec<i64> = files.iter().map(|s| s.offset_ms).collect();
    let out = cmd
        .arg("-filter_complex")
        .arg(crate::diarize::mix_filter(&offsets))
        .args(["-ac", "1", "-c:a", "libopus", "-b:a", "32k", "-f", "ogg"])
        .arg(&tmp)
        .output()
        .await
        .context("ffmpeg")?;
    if !out.status.success() {
        let _ = std::fs::remove_file(&tmp);
        bail!(
            "ffmpeg mixdown: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    std::fs::rename(tmp, dir.join("audio.ogg"))?;
    Ok(())
}

/// Mixes down the oldest finished, windowed recording without audio. Returns false when
/// there is none.
pub async fn step(app: &App) -> Result<bool> {
    let next: Option<String> = app
        .db
        .lock()
        .await
        .query_row(
            "SELECT id FROM recordings WHERE finished AND audio IS NULL
             AND status != 'receiving' ORDER BY id LIMIT 1",
            [],
            |r| r.get(0),
        )
        .optional()?;
    let Some(id) = next else {
        return Ok(false);
    };
    let state = match mixdown(app, &id).await {
        Ok(()) => "ready",
        Err(e) => {
            eprintln!("{id}: audio: {e:#}");
            "failed"
        }
    };
    app.db.lock().await.execute(
        "UPDATE recordings SET audio = ?2 WHERE id = ?1",
        params![id, state],
    )?;
    Ok(true)
}

/// Deletes the audio (mixdown and uploaded tracks, not meta.json) of transcribed recordings
/// whose mixdown is older than `RETENTION` at `now`.
pub fn expire(app: &App, db: &rusqlite::Connection, now: SystemTime) -> Result<()> {
    let ids: Vec<String> = db
        .prepare(
            "SELECT id FROM recordings WHERE audio = 'ready'
             AND status IN ('diarized', 'done', 'failed')",
        )?
        .query_map([], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    for id in ids {
        let dir = app.recording_dir(&id);
        let made = std::fs::metadata(dir.join("audio.ogg")).and_then(|m| m.modified());
        if made.is_ok_and(|t| t + RETENTION > now) {
            continue;
        }
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            if entry.file_name() != "meta.json" {
                std::fs::remove_file(entry.path())?;
            }
        }
        db.execute(
            "UPDATE recordings SET audio = 'expired' WHERE id = ?1",
            [&id],
        )?;
    }
    Ok(())
}

/// Mixes down finished recordings one at a time; expires old audio daily.
pub async fn run(app: Arc<App>) {
    let mut expired: Option<Instant> = None;
    loop {
        if expired.is_none_or(|t| t.elapsed() >= DAY) {
            if let Err(e) = expire(&app, &*app.db.lock().await, SystemTime::now()) {
                eprintln!("audio retention: {e:#}");
            }
            expired = Some(Instant::now());
        }
        match step(&app).await {
            Ok(true) => continue,
            Ok(false) => {}
            Err(e) => eprintln!("audio: {e:#}"),
        }
        tokio::time::sleep(Duration::from_secs(10)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn app() -> (tempfile::TempDir, App) {
        let tmp = tempfile::tempdir().unwrap();
        let app = App::open(tmp.path().to_path_buf()).unwrap();
        (tmp, app)
    }

    #[tokio::test]
    async fn expires_old_audio_only() {
        let (_tmp, app) = app().await;
        let old = SystemTime::now() - RETENTION - DAY;
        for (id, status, audio, age) in [
            ("old", "done", "ready", old),
            ("new", "done", "ready", SystemTime::now()),
            ("busy", "windowed", "ready", old),
        ] {
            let dir = app.recording_dir(id);
            std::fs::create_dir_all(&dir).unwrap();
            for f in ["meta.json", "00-mic.oga", "audio.ogg"] {
                std::fs::write(dir.join(f), "x").unwrap();
            }
            std::fs::File::options()
                .write(true)
                .open(dir.join("audio.ogg"))
                .unwrap()
                .set_modified(age)
                .unwrap();
            app.db
                .lock()
                .await
                .execute(
                    "INSERT INTO recordings (id, source, status, audio) VALUES (?1, 'laptop', ?2, ?3)",
                    params![id, status, audio],
                )
                .unwrap();
        }
        expire(&app, &*app.db.lock().await, SystemTime::now()).unwrap();

        let files = |id| {
            let mut v: Vec<String> = std::fs::read_dir(app.recording_dir(id))
                .unwrap()
                .map(|e| e.unwrap().file_name().into_string().unwrap())
                .collect();
            v.sort();
            v
        };
        assert_eq!(files("old"), ["meta.json"]);
        assert_eq!(files("new"), ["00-mic.oga", "audio.ogg", "meta.json"]);
        assert_eq!(files("busy"), ["00-mic.oga", "audio.ogg", "meta.json"]);
        let states: Vec<(String, String)> = app
            .db
            .lock()
            .await
            .prepare("SELECT id, audio FROM recordings ORDER BY id")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        let s = |a: &str, b: &str| (a.to_string(), b.to_string());
        assert_eq!(
            states,
            [s("busy", "ready"), s("new", "ready"), s("old", "expired")]
        );
    }

    /// Needs ffmpeg (with libopus) and ffprobe on PATH.
    #[tokio::test]
    #[ignore]
    async fn mixes_tracks_at_offsets() {
        let (_tmp, app) = app().await;
        let dir = app.recording_dir("r1");
        std::fs::create_dir_all(&dir).unwrap();
        let fixture = |name: &str| format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
        std::fs::copy(fixture("speech-60s.oga"), dir.join("00-mic.oga")).unwrap();
        std::fs::copy(fixture("dutch-60s.oga"), dir.join("01-app-7.oga")).unwrap();
        std::fs::write(dir.join("02-mic.oga"), "").unwrap();
        std::fs::write(
            dir.join("meta.json"),
            r#"{"id":"r1","started_ms":0,"segments":[
                {"file":"00-mic.oga","key":"mic","offset_ms":0},
                {"file":"01-app-7.oga","key":"app-7","offset_ms":45000},
                {"file":"02-mic.oga","key":"mic","offset_ms":60000}]}"#,
        )
        .unwrap();
        app.db
            .lock()
            .await
            .execute_batch(
                "INSERT INTO recordings (id, source, status, finished) VALUES ('r1', 'laptop', 'windowed', 1);
                 INSERT INTO file_progress (recording, file, done_ms, complete) VALUES
                   ('r1', '00-mic.oga', 60000, 1), ('r1', '01-app-7.oga', 60000, 1),
                   ('r1', '02-mic.oga', 0, 1);",
            )
            .unwrap();
        assert!(step(&app).await.unwrap());
        assert!(!step(&app).await.unwrap());

        let audio: String = app
            .db
            .lock()
            .await
            .query_row("SELECT audio FROM recordings", [], |r| r.get(0))
            .unwrap();
        assert_eq!(audio, "ready");
        let out = std::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_name",
            ])
            .args(["-of", "default=nw=1"])
            .arg(dir.join("audio.ogg"))
            .output()
            .unwrap();
        let out = String::from_utf8(out.stdout).unwrap();
        assert!(out.contains("codec_name=opus"), "{out}");
        let secs: f64 = out
            .lines()
            .find_map(|l| l.strip_prefix("duration="))
            .unwrap()
            .parse()
            .unwrap();
        assert!((104.0..=106.0).contains(&secs), "{secs}");
        assert!(!dir.join(".audio.ogg").exists());
    }
}
