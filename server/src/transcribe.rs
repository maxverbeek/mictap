use std::{path::Path, sync::Arc, time::Duration};

use anyhow::{bail, Context, Result};
use rusqlite::{params, OptionalExtension};
use serde::Deserialize;
use tokio::process::Command;

use crate::{api::App, windows::decode};

const DEFAULT_LANG: &str = "nl";
const SWITCH_P: f64 = 0.8;

/// Whisper's auto-detected language and its probability, from
/// `whisper_full_with_state: auto-detected language: nl (p = 0.997105)`.
fn parse_detected(stderr: &str) -> Option<(String, f64)> {
    let rest = stderr.split("auto-detected language: ").nth(1)?;
    let (lang, rest) = rest.split_once(" (p = ")?;
    let p = rest.split_once(')')?.0.parse().ok()?;
    Some((lang.to_string(), p))
}

/// Keep the auto-detected transcription only if it stays in `current` or is confident
/// enough to switch away from it.
fn accept(current: &str, detected: Option<&(String, f64)>) -> bool {
    detected.is_some_and(|(lang, p)| lang == current || *p >= SWITCH_P)
}

/// `vocabulary.md`, one term per line (list bullets and headings tolerated), as a prompt.
fn prompt(vocabulary: &str) -> String {
    vocabulary
        .lines()
        .map(|l| l.trim().trim_start_matches(['-', '*']).trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect::<Vec<_>>()
        .join(", ")
}

#[derive(Deserialize)]
struct Output {
    transcription: Vec<Piece>,
}

#[derive(Deserialize)]
struct Piece {
    offsets: Offsets,
    text: String,
}

#[derive(Deserialize)]
struct Offsets {
    from: i64,
    to: i64,
}

/// Segments of `wav` as (start_ms, end_ms, text), clamped to `len_ms`, plus whisper's stderr.
// ponytail: one whisper-cli process per window; model load is ~0.2 s against 10+ s of
// transcription per window, well under the 20% that would call for whisper-server.
async fn whisper(wav: &Path, len_ms: i64, lang: &str, prompt: &str) -> Result<(Vec<(i64, i64, String)>, String)> {
    let model = std::env::var("MICTAP_WHISPER_MODEL").context("MICTAP_WHISPER_MODEL not set")?;
    let base = wav.with_extension("");
    let json = wav.with_extension("json");
    let mut cmd = Command::new("whisper-cli");
    cmd.args(["-t", "4", "-m", &model, "-l", lang, "-oj", "-of"]).arg(&base);
    if !prompt.is_empty() {
        cmd.args(["--prompt", prompt]);
    }
    let out = cmd.arg("-f").arg(wav).output().await.context("whisper-cli")?;
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    if !out.status.success() {
        bail!("whisper-cli: {}", stderr.trim());
    }
    let parsed: Output =
        serde_json::from_slice(&std::fs::read(&json).context("whisper json")?).context("whisper json")?;
    let _ = std::fs::remove_file(&json);
    let segs = parsed
        .transcription
        .into_iter()
        .filter_map(|p| {
            let text = p.text.trim();
            let start = p.offsets.from.clamp(0, len_ms);
            (!text.is_empty()).then(|| (start, p.offsets.to.clamp(start, len_ms), text.to_string()))
        })
        .collect();
    Ok((segs, stderr))
}

/// Transcribes `wav` with the sticky language bias. Returns the segments and the language used.
async fn transcribe(wav: &Path, len_ms: i64, current: &str, prompt: &str) -> Result<(Vec<(i64, i64, String)>, String)> {
    let (segs, stderr) = whisper(wav, len_ms, "auto", prompt).await?;
    let detected = parse_detected(&stderr);
    if accept(current, detected.as_ref()) {
        return Ok((segs, detected.unwrap().0));
    }
    let (segs, _) = whisper(wav, len_ms, current, prompt).await?;
    Ok((segs, current.to_string()))
}

/// Transcribes the oldest pending window. Returns false when there is none, or when it
/// failed and will be retried after a pause.
pub async fn step(app: &App) -> Result<bool> {
    let next = app
        .db
        .lock()
        .await
        .query_row(
            "SELECT w.id, w.recording, w.file, w.track, w.offset_ms, w.start_ms, w.end_ms,
                    COALESCE(r.lang, ?1)
             FROM windows w JOIN recordings r ON r.id = w.recording
             WHERE NOT w.done AND r.status != 'failed' ORDER BY w.id LIMIT 1",
            [DEFAULT_LANG],
            |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, i64>(4)?,
                    r.get::<_, i64>(5)?,
                    r.get::<_, i64>(6)?,
                    r.get::<_, String>(7)?,
                ))
            },
        )
        .optional()?;
    let Some((wid, id, file, track, offset_ms, start_ms, end_ms, current)) = next else {
        return Ok(false);
    };
    let dir = app.recording_dir(&id);
    let wav = dir.join(".asr.wav");
    let vocabulary = std::fs::read_to_string(app.vault.join("vocabulary.md")).unwrap_or_default();
    let res = async {
        let len_ms = decode(&dir.join(&file), start_ms, Some(end_ms - start_ms), &wav).await?;
        transcribe(&wav, len_ms, &current, &prompt(&vocabulary)).await
    }
    .await;
    let _ = std::fs::remove_file(&wav);

    let mut db = app.db.lock().await;
    let (segs, lang) = match res {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{id}: transcribing window {wid}: {e:#}");
            // Not final: the window stays pending and is retried after the worker's pause.
            return Ok(crate::db::fail(&db, &id, &format!("transcribing: {e:#}"))?);
        }
    };
    let tx = db.transaction()?;
    let base = offset_ms + start_ms;
    for (s, e, text) in segs {
        tx.execute(
            "INSERT INTO segments (recording, window, track, start_ms, end_ms, text)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![id, wid, track, base + s, base + e, text],
        )?;
    }
    tx.execute("UPDATE windows SET done = 1 WHERE id = ?1", [wid])?;
    tx.execute(
        "UPDATE recordings SET lang = ?2, attempts = 0 WHERE id = ?1",
        params![id, lang],
    )?;
    tx.commit()?;
    Ok(true)
}

/// The single transcription worker: one window at a time.
pub async fn run(app: Arc<App>) {
    loop {
        match step(&app).await {
            Ok(true) => continue,
            Ok(false) => {}
            Err(e) => eprintln!("transcribe: {e:#}"),
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_detected_language() {
        let err = "whisper_full_with_state: auto-detected language: nl (p = 0.997105)\n\
                   whisper_print_timings: load time = 193.40 ms\n";
        assert_eq!(parse_detected(err), Some(("nl".into(), 0.997105)));
        assert_eq!(parse_detected("main: processing 'w.wav'\n"), None);
    }

    #[test]
    fn language_is_sticky() {
        let d = |l: &str, p| Some((l.to_string(), p));
        assert!(accept("nl", d("nl", 0.3).as_ref()));
        assert!(accept("nl", d("en", 0.8).as_ref()));
        assert!(!accept("nl", d("en", 0.79).as_ref()));
        assert!(!accept("nl", None));
    }

    #[test]
    fn vocabulary_prompt() {
        let v = "# Vocabulary\n\nAcme\n- homeserver\n  * Jan \n\n";
        assert_eq!(prompt(v), "Acme, homeserver, Jan");
        assert_eq!(prompt(""), "");
    }

    /// Needs ffmpeg, whisper-vad-speech-segments and whisper-cli on PATH,
    /// MICTAP_VAD_MODEL and MICTAP_WHISPER_MODEL.
    #[tokio::test]
    #[ignore]
    async fn transcribes_dutch() {
        let tmp = tempfile::tempdir().unwrap();
        let mut app = App::open(tmp.path().to_path_buf()).unwrap();
        app.vault = tmp.path().join("vault");
        std::fs::create_dir_all(&app.vault).unwrap();
        std::fs::write(app.vault.join("vocabulary.md"), "Westerwald\n").unwrap();
        let dir = app.recording_dir("r1");
        std::fs::create_dir_all(&dir).unwrap();
        crate::db::ensure_recording(&*app.db.lock().await, "r1", "laptop").unwrap();
        std::fs::write(
            dir.join("meta.json"),
            r#"{"id":"r1","started_ms":0,"app":"Zen","segments":[
                {"file":"00-mic.oga","key":"mic","target":"x","offset_ms":5000,"end_ms":null}]}"#,
        )
        .unwrap();
        std::fs::copy(
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dutch-60s.oga"),
            dir.join("00-mic.oga"),
        )
        .unwrap();
        crate::windows::advance(&app, "r1", true).await.unwrap();
        while step(&app).await.unwrap() {}

        let db = app.db.lock().await;
        let status: String = db
            .query_row("SELECT status || '/' || lang FROM recordings", [], |r| r.get(0))
            .unwrap();
        assert_eq!(status, "receiving/nl");
        let segs: Vec<(String, i64, i64, String)> = db
            .prepare("SELECT track, start_ms, end_ms, text FROM segments ORDER BY id")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        let mut prev = 5_000;
        for (track, s, e, _) in &segs {
            assert_eq!(track, "room");
            assert!(prev <= *s && s <= e && *e <= 65_000, "{segs:?}");
            prev = *e;
        }
        assert!(prev > 55_000, "{segs:?}");
        let text = segs
            .iter()
            .map(|s| s.3.as_str())
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();
        assert!(text.contains("westerwald"), "{text}");
        let dutch = [" de ", " het ", " een ", " en ", " is ", " van "];
        assert!(dutch.iter().filter(|w| text.contains(*w)).count() >= 4, "{text}");
    }
}
