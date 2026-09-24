use std::{io::Read, path::Path, sync::Arc, time::Duration};

use anyhow::{bail, Context, Result};
use rusqlite::{params, OptionalExtension};
use serde::Deserialize;
use tokio::process::Command;

use crate::api::{valid_name, App};

const MAX_MS: i64 = 30_000;
const TAIL_MS: i64 = 2_000;

#[derive(Debug, PartialEq)]
pub struct Window {
    pub start_ms: i64,
    pub end_ms: i64,
}

/// Groups VAD speech into windows of at most `MAX_MS` that end on a pause (a longer
/// speech run is cut hard). A window is closed once `TAIL_MS` of audio follows it, or
/// when `finished`. Times are relative to the decoded audio. Returns the closed windows
/// and where the next pass resumes decoding.
pub fn cut(speech: &[(i64, i64)], audio_ms: i64, finished: bool) -> (Vec<Window>, i64) {
    let mut windows: Vec<Window> = vec![];
    for &(mut s, e) in speech {
        while s < e {
            let end = e.min(s + MAX_MS);
            match windows.last_mut() {
                Some(w) if end - w.start_ms <= MAX_MS => w.end_ms = end,
                _ => windows.push(Window {
                    start_ms: s,
                    end_ms: end,
                }),
            }
            s = end;
        }
    }
    let closed = if finished {
        windows.len()
    } else {
        windows
            .iter()
            .take_while(|w| audio_ms - w.end_ms >= TAIL_MS)
            .count()
    };
    let closed_end = windows[..closed].last().map_or(0, |w| w.end_ms);
    let resume = match windows.get(closed) {
        Some(open) => closed_end.max(open.start_ms - TAIL_MS),
        None if finished => audio_ms,
        None => closed_end.max(audio_ms - TAIL_MS),
    };
    windows.truncate(closed);
    (windows, resume)
}

#[derive(Deserialize)]
pub(crate) struct Meta {
    pub segments: Vec<Segment>,
}

#[derive(Deserialize)]
pub(crate) struct Segment {
    pub file: String,
    key: String,
    pub offset_ms: i64,
}

/// Emits the newly closed windows of every file of `id` and stores how far each file got.
/// `finished` must be read before calling, so every byte of a finished recording is on disk.
pub async fn advance(app: &App, id: &str, finished: bool) -> Result<()> {
    let dir = app.recording_dir(id);
    let meta = match std::fs::read(dir.join("meta.json")) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && !finished => return Ok(()),
        r => r.context("meta.json")?,
    };
    let meta: Meta = serde_json::from_slice(&meta).context("meta.json")?;
    let wav = dir.join(".vad.wav");
    for seg in &meta.segments {
        if !valid_name(&seg.file) {
            bail!("bad file name in meta.json: {}", seg.file);
        }
        let progress: Option<(i64, bool)> = app
            .db
            .lock()
            .await
            .query_row(
                "SELECT done_ms, complete FROM file_progress WHERE recording = ?1 AND file = ?2",
                params![id, seg.file],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        let (done_ms, complete) = progress.unwrap_or((0, false));
        if complete {
            continue;
        }
        let path = dir.join(&seg.file);
        let audio_ms = match decode(&path, done_ms, None, &wav).await {
            Ok(ms) => ms,
            // Not a single Ogg page yet (or an empty segment).
            Err(_) if done_ms == 0 && std::fs::metadata(&path).map_or(0, |m| m.len()) < 4096 => 0,
            Err(e) => return Err(e),
        };
        let speech = if audio_ms > 0 {
            vad(&wav, audio_ms).await?
        } else {
            vec![]
        };
        let _ = std::fs::remove_file(&wav);
        let (windows, resume) = cut(&speech, audio_ms, finished);
        let track = if seg.key.starts_with("app-") {
            "remote"
        } else {
            "room"
        };

        let mut db = app.db.lock().await;
        let tx = db.transaction()?;
        for w in windows {
            tx.execute(
                "INSERT INTO windows (recording, file, track, offset_ms, start_ms, end_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    id,
                    seg.file,
                    track,
                    seg.offset_ms,
                    done_ms + w.start_ms,
                    done_ms + w.end_ms
                ],
            )?;
        }
        tx.execute(
            "INSERT INTO file_progress (recording, file, done_ms, complete) VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT DO UPDATE SET done_ms = excluded.done_ms, complete = excluded.complete",
            params![id, seg.file, done_ms + resume, finished],
        )?;
        tx.commit()?;
    }
    Ok(())
}

/// Advances every recording still receiving; a finished one moves on to `windowed`.
pub async fn run(app: Arc<App>) {
    loop {
        let recordings: rusqlite::Result<Vec<(String, bool)>> = {
            let db = app.db.lock().await;
            db.prepare("SELECT id, finished FROM recordings WHERE status = 'receiving'")
                .and_then(|mut st| st.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?.collect())
        };
        for (id, finished) in recordings.unwrap_or_else(|e| {
            eprintln!("windows: {e}");
            vec![]
        }) {
            let res = match advance(&app, &id, finished).await {
                Ok(()) if finished => app.db.lock().await.execute(
                    "UPDATE recordings SET status = 'windowed' WHERE id = ?1",
                    [&id],
                ),
                Ok(()) => continue,
                Err(e) => {
                    eprintln!("{id}: windowing: {e:#}");
                    if !finished {
                        continue;
                    }
                    crate::db::fail(&*app.db.lock().await, &id, &format!("windowing: {e:#}"))
                        .map(|()| 0)
                }
            };
            if let Err(e) = res {
                eprintln!("{id}: {e}");
            }
        }
        tokio::time::sleep(Duration::from_secs(10)).await;
    }
}

/// Decodes `path` from `from_ms` (for `dur_ms`, else to the end) to a 16 kHz mono wav
/// and returns its length in ms.
pub(crate) async fn decode(
    path: &Path,
    from_ms: i64,
    dur_ms: Option<i64>,
    wav: &Path,
) -> Result<i64> {
    let secs = |ms: i64| format!("{}.{:03}", ms / 1000, ms % 1000);
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-nostdin", "-v", "error", "-y", "-ss"])
        .arg(secs(from_ms));
    if let Some(d) = dur_ms {
        cmd.arg("-t").arg(secs(d));
    }
    let out = cmd
        .arg("-i")
        .arg(path)
        .args(["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"])
        .arg(wav)
        .output()
        .await
        .context("ffmpeg")?;
    if !out.status.success() {
        bail!(
            "ffmpeg {}: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let mut head = vec![];
    std::fs::File::open(wav)?.take(512).read_to_end(&mut head)?;
    let data = head
        .windows(4)
        .position(|w| w == b"data")
        .context("wav without data chunk")?;
    let bytes = std::fs::metadata(wav)?.len() as i64 - data as i64 - 8;
    Ok(bytes / 32)
}

async fn vad(wav: &Path, audio_ms: i64) -> Result<Vec<(i64, i64)>> {
    let model = std::env::var("MICTAP_VAD_MODEL").context("MICTAP_VAD_MODEL not set")?;
    let out = Command::new("whisper-vad-speech-segments")
        .args(["-np", "-t", "4", "-vm", &model, "-f"])
        .arg(wav)
        .output()
        .await
        .context("whisper-vad-speech-segments")?;
    if !out.status.success() {
        bail!(
            "whisper-vad-speech-segments: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(parse_vad(&String::from_utf8_lossy(&out.stdout), audio_ms))
}

/// Parses `Speech segment 0: start = 144.00, end = 397.00` (centiseconds) lines to ms.
fn parse_vad(out: &str, audio_ms: i64) -> Vec<(i64, i64)> {
    let cs = |x: &str| {
        x.trim()
            .parse::<f64>()
            .ok()
            .map(|v| (v * 10.0).round() as i64)
    };
    out.lines()
        .filter_map(|l| {
            let (a, b) = l.split_once("start = ")?.1.split_once(", end = ")?;
            Some((cs(a)?, cs(b)?.min(audio_ms)))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(start_ms: i64, end_ms: i64) -> Window {
        Window { start_ms, end_ms }
    }

    #[test]
    fn groups_up_to_30s_on_pauses() {
        let speech = [
            (0, 5_000),
            (6_000, 20_000),
            (21_000, 29_000),
            (31_000, 40_000),
        ];
        assert_eq!(
            cut(&speech, 50_000, false),
            (vec![w(0, 29_000), w(31_000, 40_000)], 48_000)
        );
    }

    #[test]
    fn closes_only_with_2s_after() {
        let speech = [(0, 5_000), (6_000, 9_000)];
        assert_eq!(cut(&speech, 10_500, false), (vec![], 0));
        assert_eq!(cut(&speech, 11_000, false), (vec![w(0, 9_000)], 9_000));
        assert_eq!(cut(&speech, 9_500, true), (vec![w(0, 9_000)], 9_500));
        // More speech may still join the open window.
        assert_eq!(
            cut(&[(0, 10_000), (10_500, 12_000)], 12_500, false),
            (vec![], 0)
        );
    }

    #[test]
    fn splits_long_speech() {
        assert_eq!(
            cut(&[(1_000, 70_000)], 80_000, false),
            (
                vec![w(1_000, 31_000), w(31_000, 61_000), w(61_000, 70_000)],
                78_000
            )
        );
    }

    #[test]
    fn resumes_near_open_speech_and_skips_silence() {
        let speech = [(0, 3_000), (50_000, 55_000)];
        assert_eq!(cut(&speech, 56_000, false), (vec![w(0, 3_000)], 48_000));
        assert_eq!(cut(&[], 60_000, false), (vec![], 58_000));
        assert_eq!(cut(&[], 1_000, false), (vec![], 0));
        assert_eq!(cut(&[], 1_000, true), (vec![], 1_000));
    }

    #[test]
    fn parses_vad_output() {
        let out = "\nDetected 2 speech segments:\n\
                   Speech segment 0: start = 0.00, end = 89.00\n\
                   Speech segment 1: start = 144.00, end = 6000.00\n";
        assert_eq!(parse_vad(out, 59_990), vec![(0, 890), (1_440, 59_990)]);
        assert_eq!(parse_vad("\nDetected 0 speech segments:\n", 1_000), vec![]);
    }

    /// Needs ffmpeg and whisper-vad-speech-segments on PATH and MICTAP_VAD_MODEL.
    #[tokio::test]
    #[ignore]
    async fn advance_growing_file() {
        let tmp = tempfile::tempdir().unwrap();
        let app = App::open(tmp.path().to_path_buf()).unwrap();
        let dir = app.recording_dir("r1");
        std::fs::create_dir_all(&dir).unwrap();
        crate::db::ensure_recording(&*app.db.lock().await, "r1", "laptop").unwrap();
        std::fs::write(
            dir.join("meta.json"),
            r#"{"id":"r1","started_ms":0,"app":"Zen","segments":[
                {"file":"00-mic.oga","key":"mic","target":"x","offset_ms":0,"end_ms":null},
                {"file":"01-app-427.oga","key":"app-427","target":"y","offset_ms":5000,"end_ms":null}]}"#,
        )
        .unwrap();
        let full = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/speech-60s.oga"
        ))
        .unwrap();

        // App file not uploaded yet, mic file about 36 s in.
        std::fs::write(dir.join("00-mic.oga"), &full[..110_000]).unwrap();
        advance(&app, "r1", false).await.unwrap();
        let rows = |file: &'static str| {
            let app = &app;
            async move {
                let db = app.db.lock().await;
                let mut st = db
                    .prepare("SELECT track, offset_ms, start_ms, end_ms FROM windows WHERE file = ?1 ORDER BY id")
                    .unwrap();
                st.query_map([file], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
                    .unwrap()
                    .collect::<rusqlite::Result<Vec<(String, i64, i64, i64)>>>()
                    .unwrap()
            }
        };
        let early = rows("00-mic.oga").await;
        assert!(!early.is_empty());
        assert!(early.last().unwrap().3 < 36_000);

        std::fs::write(dir.join("00-mic.oga"), &full).unwrap();
        std::fs::write(dir.join("01-app-427.oga"), &full).unwrap();
        advance(&app, "r1", false).await.unwrap();
        advance(&app, "r1", true).await.unwrap();
        // Complete files are skipped: nothing new.
        let n = rows("00-mic.oga").await.len();
        advance(&app, "r1", true).await.unwrap();
        assert_eq!(rows("00-mic.oga").await.len(), n);

        let mic = rows("00-mic.oga").await;
        let app_rows = rows("01-app-427.oga").await;
        assert_eq!(&mic[..early.len()], &early[..]);
        assert!(mic.iter().all(|r| r.0 == "room" && r.1 == 0));
        assert!(app_rows.iter().all(|r| r.0 == "remote" && r.1 == 5_000));
        for rows in [&mic, &app_rows] {
            let mut prev = 0;
            for &(_, _, s, e) in rows.iter() {
                assert!(prev <= s && s < e && e - s <= MAX_MS, "{rows:?}");
                prev = e;
            }
            assert!(prev > 55_000, "{rows:?}");
            let speech: i64 = rows.iter().map(|r| r.3 - r.2).sum();
            assert!(speech > 40_000, "{rows:?}");
        }
    }
}
