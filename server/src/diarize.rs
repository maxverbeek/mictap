use std::{
    collections::{BTreeMap, HashSet},
    ffi::{c_char, CString},
    path::Path,
    sync::Arc,
    time::Duration,
};

use anyhow::{bail, Context, Result};
use rusqlite::OptionalExtension;
use tokio::process::Command;

use crate::{
    api::App,
    assemble::{bytes, derive, normalize, store_turns, Line, Turn},
};

const RATE: usize = 16_000;

/// Parses `59.566 -- 68.324 speaker_02` lines and renumbers speakers by first appearance.
fn parse_turns(out: &str) -> Vec<Turn> {
    let ms = |x: &str| x.parse::<f64>().ok().map(|v| (v * 1000.0).round() as i64);
    let mut raw: Vec<(i64, i64, &str)> = out
        .lines()
        .filter_map(|l| match l.split_whitespace().collect::<Vec<_>>()[..] {
            [a, "--", b, spk] if spk.starts_with("speaker_") => Some((ms(a)?, ms(b)?, spk)),
            _ => None,
        })
        .collect();
    raw.sort_by_key(|t| t.0);
    let mut seen: Vec<&str> = vec![];
    raw.into_iter()
        .map(|(start_ms, end_ms, spk)| {
            let speaker = seen.iter().position(|&s| s == spk).unwrap_or_else(|| {
                seen.push(spk);
                seen.len() - 1
            });
            Turn {
                start_ms,
                end_ms,
                speaker,
                embedding: None,
            }
        })
        .collect()
}

/// ffmpeg filter placing each input at its offset on one timeline.
pub(crate) fn mix_filter(offsets_ms: &[i64]) -> String {
    let mut f = String::new();
    for (i, off) in offsets_ms.iter().enumerate() {
        f += &format!("[{i}:a]adelay={off}:all=1[a{i}];");
    }
    for i in 0..offsets_ms.len() {
        f += &format!("[a{i}]");
    }
    f + &format!("amix=inputs={}:normalize=0", offsets_ms.len())
}

/// Mixes `(file, offset_ms)` into one 16 kHz mono wav that starts at the recording's start.
async fn timeline(dir: &Path, files: &[(String, i64)], wav: &Path) -> Result<()> {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-nostdin", "-v", "error", "-y"]);
    for (file, _) in files {
        cmd.arg("-i").arg(dir.join(file));
    }
    let offsets: Vec<i64> = files.iter().map(|f| f.1).collect();
    let out = cmd
        .arg("-filter_complex")
        .arg(mix_filter(&offsets))
        .args(["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"])
        .arg(wav)
        .output()
        .await
        .context("ffmpeg")?;
    if !out.status.success() {
        bail!("ffmpeg timeline: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    Ok(())
}

async fn diarize(wav: &Path) -> Result<Vec<Turn>> {
    let seg = std::env::var("MICTAP_SEG_MODEL").context("MICTAP_SEG_MODEL not set")?;
    let emb = std::env::var("MICTAP_EMB_MODEL").context("MICTAP_EMB_MODEL not set")?;
    let threshold = std::env::var("MICTAP_CLUSTER_THRESHOLD").unwrap_or_else(|_| "0.9".into());
    let out = Command::new("sherpa-onnx-offline-speaker-diarization")
        .arg(format!("--segmentation.pyannote-model={seg}"))
        .arg(format!("--embedding.model={emb}"))
        .arg(format!("--clustering.cluster-threshold={threshold}"))
        .args(["--segmentation.num-threads=4", "--embedding.num-threads=4"])
        .arg(wav)
        .output()
        .await
        .context("sherpa-onnx-offline-speaker-diarization")?;
    if !out.status.success() {
        bail!(
            "sherpa-onnx-offline-speaker-diarization: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(parse_turns(&String::from_utf8_lossy(&out.stdout)))
}

/// Samples of a 16-bit PCM wav as f32.
fn read_wav(wav: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(wav)?;
    let data = bytes[..bytes.len().min(512)]
        .windows(4)
        .position(|w| w == b"data")
        .context("wav without data chunk")?;
    Ok(bytes[data + 8..]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&b| i16::from_le_bytes(b) as f32 / 32768.0)
        .collect())
}

#[repr(C)]
struct ExtractorConfig {
    model: *const c_char,
    num_threads: i32,
    debug: i32,
    provider: *const c_char,
}

#[repr(C)]
struct RawExtractor {
    _p: [u8; 0],
}

#[repr(C)]
struct RawStream {
    _p: [u8; 0],
}

#[link(name = "sherpa-onnx-c-api")]
extern "C" {
    fn SherpaOnnxCreateSpeakerEmbeddingExtractor(c: *const ExtractorConfig) -> *const RawExtractor;
    fn SherpaOnnxDestroySpeakerEmbeddingExtractor(p: *const RawExtractor);
    fn SherpaOnnxSpeakerEmbeddingExtractorDim(p: *const RawExtractor) -> i32;
    fn SherpaOnnxSpeakerEmbeddingExtractorCreateStream(p: *const RawExtractor) -> *const RawStream;
    fn SherpaOnnxOnlineStreamAcceptWaveform(s: *const RawStream, rate: i32, samples: *const f32, n: i32);
    fn SherpaOnnxOnlineStreamInputFinished(s: *const RawStream);
    fn SherpaOnnxSpeakerEmbeddingExtractorIsReady(p: *const RawExtractor, s: *const RawStream) -> i32;
    fn SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(p: *const RawExtractor, s: *const RawStream) -> *const f32;
    fn SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(v: *const f32);
    fn SherpaOnnxDestroyOnlineStream(s: *const RawStream);
}

struct Extractor(*const RawExtractor);

impl Extractor {
    fn new(model: &str) -> Result<Self> {
        let model = CString::new(model)?;
        let provider = CString::new("cpu")?;
        let config = ExtractorConfig {
            model: model.as_ptr(),
            num_threads: 4,
            debug: 0,
            provider: provider.as_ptr(),
        };
        // SAFETY: the config's strings outlive the call; sherpa copies them.
        let p = unsafe { SherpaOnnxCreateSpeakerEmbeddingExtractor(&config) };
        if p.is_null() {
            bail!("cannot load speaker embedding model {model:?}");
        }
        Ok(Self(p))
    }

    /// The embedding of 16 kHz `samples`, or None when they are too short.
    fn embed(&self, samples: &[f32]) -> Option<Vec<f32>> {
        // SAFETY: self.0 is a live extractor; the stream and embedding are freed here.
        unsafe {
            let dim = SherpaOnnxSpeakerEmbeddingExtractorDim(self.0) as usize;
            let s = SherpaOnnxSpeakerEmbeddingExtractorCreateStream(self.0);
            SherpaOnnxOnlineStreamAcceptWaveform(s, RATE as i32, samples.as_ptr(), samples.len() as i32);
            SherpaOnnxOnlineStreamInputFinished(s);
            let v = (SherpaOnnxSpeakerEmbeddingExtractorIsReady(self.0, s) != 0).then(|| {
                let p = SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(self.0, s);
                let v = std::slice::from_raw_parts(p, dim).to_vec();
                SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(p);
                v
            });
            SherpaOnnxDestroyOnlineStream(s);
            v
        }
    }
}

impl Drop for Extractor {
    fn drop(&mut self) {
        // SAFETY: created by SherpaOnnxCreateSpeakerEmbeddingExtractor, dropped once.
        unsafe { SherpaOnnxDestroySpeakerEmbeddingExtractor(self.0) }
    }
}

/// The normalized embedding of `samples` in `[start_ms, end_ms)`, or None when too short.
fn embed_span(ex: &Extractor, samples: &[f32], start_ms: i64, end_ms: i64) -> Option<Vec<f32>> {
    let at = |ms: i64| (ms.max(0) as usize * RATE / 1000).min(samples.len());
    ex.embed(&samples[at(start_ms)..at(end_ms).max(at(start_ms))])
        .map(|mut v| {
            normalize(&mut v);
            v
        })
}

/// Sets each turn's normalized embedding, or None when it is too short.
fn embed_turns(model: &str, samples: &[f32], turns: &mut [Turn]) -> Result<()> {
    let ex = Extractor::new(model)?;
    for t in turns {
        t.embedding = embed_span(&ex, samples, t.start_ms, t.end_ms);
    }
    Ok(())
}

/// Lines shorter than this get no line voice: too little to tell a voice by.
const LINE_MS: i64 = 300;

/// The line voices of `lines` from their tracks' samples: `(track, start_ms, end_ms, embedding)`.
fn embed_lines(
    model: &str,
    samples: &BTreeMap<String, Vec<f32>>,
    lines: &[Line],
) -> Result<Vec<(String, i64, i64, Vec<f32>)>> {
    let ex = Extractor::new(model)?;
    Ok(lines
        .iter()
        .filter(|l| l.end_ms - l.start_ms >= LINE_MS)
        .filter_map(|l| {
            let v = embed_span(&ex, samples.get(&l.track)?, l.start_ms, l.end_ms)?;
            Some((l.track.clone(), l.start_ms, l.end_ms, v))
        })
        .collect())
}

/// Computes and replaces the line voices of `id` from its tracks' `samples`.
async fn line_voices(app: &App, id: &str, samples: BTreeMap<String, Vec<f32>>) -> Result<usize> {
    let model = std::env::var("MICTAP_EMB_MODEL").context("MICTAP_EMB_MODEL not set")?;
    let lines = crate::assemble::lines(&*app.db.lock().await, id)?;
    let voices = tokio::task::spawn_blocking(move || embed_lines(&model, &samples, &lines)).await??;
    let mut db = app.db.lock().await;
    let tx = db.transaction()?;
    tx.execute("DELETE FROM line_voices WHERE recording = ?1", [id])?;
    for (track, s, e, v) in &voices {
        tx.execute(
            "INSERT INTO line_voices (recording, track, start_ms, end_ms, embedding) VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![id, track, s, e, bytes(v)],
        )?;
    }
    tx.commit()?;
    Ok(voices.len())
}

/// Diarizes the oldest windowed recording whose windows are all transcribed: stores each
/// track's turns and their embeddings, derives its lines, embeds each line and suggests names
/// from known voices. Returns false when there is none, or when it failed and will be retried
/// after a pause.
pub async fn step(app: &App) -> Result<bool> {
    let next: Option<String> = app
        .db
        .lock()
        .await
        .query_row(
            "SELECT id FROM recordings r WHERE status = 'windowed' AND retry_at <= ?1
             AND NOT EXISTS (SELECT 1 FROM windows w WHERE w.recording = r.id AND NOT w.done)
             ORDER BY id LIMIT 1",
            [crate::db::now_ms()],
            |r| r.get(0),
        )
        .optional()?;
    let Some(id) = next else {
        return Ok(false);
    };
    *app.diarizing.lock().unwrap() = Some((id.clone(), crate::db::now_ms()));
    let res = diarized(app, &id).await;
    *app.diarizing.lock().unwrap() = None;
    if let Err(e) = res {
        eprintln!("{id}: diarizing: {e:#}");
        return Ok(crate::db::fail(
            &*app.db.lock().await,
            &id,
            &format!("diarizing: {e:#}"),
        )?);
    }
    Ok(true)
}

async fn diarized(app: &App, id: &str) -> Result<()> {
    let tracks = turns(app, id).await?;
    {
        let mut db = app.db.lock().await;
        let tx = db.transaction()?;
        for (track, (turns, _)) in &tracks {
            store_turns(&tx, id, track, turns)?;
        }
        derive(&tx, id)?;
        tx.commit()?;
    }
    line_voices(app, id, tracks.into_iter().map(|(t, (_, s))| (t, s)).collect()).await?;
    let db = app.db.lock().await;
    // The structure its log teaches over changed: its voices anew.
    crate::names::project(&db, id)?;
    db.execute(
        "UPDATE recordings SET status = 'diarized', attempts = 0 WHERE id = ?1",
        [id],
    )?;
    Ok(())
}

/// Each track of `id` on one 16 kHz timeline, with sherpa's turns on it when `diarize`.
async fn tracks(app: &App, id: &str, diarize: bool) -> Result<BTreeMap<String, (Vec<Turn>, Vec<f32>)>> {
    let mut files: BTreeMap<String, Vec<(String, i64)>> = BTreeMap::new();
    {
        let db = app.db.lock().await;
        let mut st =
            db.prepare("SELECT DISTINCT track, file, offset_ms FROM windows WHERE recording = ?1 ORDER BY file")?;
        for r in st.query_map([id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))? {
            let (track, file, offset): (String, String, i64) = r?;
            files.entry(track).or_default().push((file, offset));
        }
    }
    let dir = app.recording_dir(id);
    let mut out = BTreeMap::new();
    for (track, files) in files {
        let wav = dir.join(format!(".{track}.wav"));
        let res = async {
            timeline(&dir, &files, &wav).await?;
            let turns = if diarize { self::diarize(&wav).await? } else { vec![] };
            anyhow::Ok((turns, read_wav(&wav)?))
        }
        .await;
        let _ = std::fs::remove_file(&wav);
        out.insert(track, res?);
    }
    Ok(out)
}

/// sherpa's turns of each track of `id`, with their embeddings, and the track's samples.
async fn turns(app: &App, id: &str) -> Result<BTreeMap<String, (Vec<Turn>, Vec<f32>)>> {
    let model = std::env::var("MICTAP_EMB_MODEL").context("MICTAP_EMB_MODEL not set")?;
    let mut out = BTreeMap::new();
    for (track, (mut turns, samples)) in tracks(app, id, true).await? {
        let model = model.clone();
        let track_out = tokio::task::spawn_blocking(move || {
            embed_turns(&model, &samples, &mut turns)?;
            anyhow::Ok((turns, samples))
        })
        .await??;
        out.insert(track, track_out);
    }
    Ok(out)
}

/// Done recordings with their audio and lines but no line voices, oldest first: diarized
/// before line voices existed.
fn unembedded(db: &rusqlite::Connection) -> rusqlite::Result<Vec<String>> {
    db.prepare(
        "SELECT id FROM recordings r WHERE status = 'done' AND audio = 'ready'
         AND EXISTS (SELECT 1 FROM lines WHERE recording = r.id)
         AND NOT EXISTS (SELECT 1 FROM line_voices WHERE recording = r.id)
         ORDER BY id",
    )?
    .query_map([], |r| r.get(0))?
    .collect()
}

/// Computes the line voices of one recording that has none and rewrites its transcript, trying
/// each recording once per run (`tried`). Returns false when there is none left.
async fn backfill(app: &App, tried: &mut HashSet<String>) -> Result<bool> {
    let ids = unembedded(&*app.db.lock().await)?;
    let Some(id) = ids.into_iter().find(|id| !tried.contains(id)) else {
        return Ok(false);
    };
    tried.insert(id.clone());
    let samples = tracks(app, &id, false).await?;
    let n = line_voices(app, &id, samples.into_iter().map(|(t, (_, s))| (t, s)).collect()).await?;
    crate::vault::write(app, &id).await?;
    eprintln!("{id}: backfilled {n} line voices");
    Ok(true)
}

/// Diarizes one recording at a time; when there is none, backfills line voices.
pub async fn run(app: Arc<App>) {
    let mut tried = HashSet::new();
    loop {
        match step(&app).await {
            Ok(true) => continue,
            Ok(false) => match backfill(&app, &mut tried).await {
                Ok(true) => continue,
                Ok(false) => {}
                Err(e) => eprintln!("line voices: {e:#}"),
            },
            Err(e) => eprintln!("diarize: {e:#}"),
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(start_ms: i64, end_ms: i64, speaker: usize) -> Turn {
        Turn {
            start_ms,
            end_ms,
            speaker,
            embedding: None,
        }
    }

    #[test]
    fn parses_and_renumbers_by_first_appearance() {
        let out = "OfflineSpeakerDiarizationConfig(...)\nStarted\n\
                   0.031 -- 0.824 speaker_02\n\
                   1.415 -- 3.608 speaker_00\n\
                   4.435 -- 7.878 speaker_02\n\
                   59.566 -- 68.324 speaker_01\n";
        assert_eq!(
            parse_turns(out),
            vec![
                t(31, 824, 0),
                t(1_415, 3_608, 1),
                t(4_435, 7_878, 0),
                t(59_566, 68_324, 2)
            ]
        );
        assert_eq!(parse_turns("Started\n"), vec![]);
    }

    #[test]
    fn mixes_inputs_at_offsets() {
        assert_eq!(
            mix_filter(&[0]),
            "[0:a]adelay=0:all=1[a0];[a0]amix=inputs=1:normalize=0"
        );
        assert_eq!(
            mix_filter(&[0, 60_000]),
            "[0:a]adelay=0:all=1[a0];[1:a]adelay=60000:all=1[a1];[a0][a1]amix=inputs=2:normalize=0"
        );
    }

    #[test]
    fn backfills_done_recordings_with_audio_and_lines_but_no_line_voices() {
        let db = crate::db::open(Path::new(":memory:")).unwrap();
        db.execute_batch(
            "INSERT INTO recordings (id, source, status, audio) VALUES
               ('a', 'laptop', 'done', 'ready'), ('b', 'laptop', 'done', 'expired'),
               ('c', 'laptop', 'windowed', 'ready'), ('d', 'laptop', 'done', 'ready'),
               ('e', 'laptop', 'done', 'ready'), ('f', 'laptop', 'done', 'ready');
             INSERT INTO lines (recording, track, start_ms, end_ms, text) VALUES
               ('a', 'room', 0, 1000, 'x'), ('b', 'room', 0, 1000, 'x'), ('c', 'room', 0, 1000, 'x'),
               ('e', 'room', 0, 1000, 'x'), ('f', 'room', 0, 1000, 'x');
             INSERT INTO line_voices (recording, track, start_ms, end_ms, embedding) VALUES
               ('e', 'room', 0, 1000, x'00');",
        )
        .unwrap();
        assert_eq!(unembedded(&db).unwrap(), ["a", "f"]);
    }

    /// Needs ffmpeg and sherpa-onnx-offline-speaker-diarization on PATH,
    /// MICTAP_SEG_MODEL and MICTAP_EMB_MODEL.
    #[tokio::test]
    #[ignore]
    async fn labels_a_short_track() {
        let tmp = tempfile::tempdir().unwrap();
        let app = App::open(tmp.path().to_path_buf()).unwrap();
        let dir = app.recording_dir("r1");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::copy(
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dutch-8s.oga"),
            dir.join("00-mic.oga"),
        )
        .unwrap();
        {
            let db = app.db.lock().await;
            crate::db::ensure_recording(&db, "r1", "laptop").unwrap();
            db.execute_batch(
                "UPDATE recordings SET status = 'windowed';
                 INSERT INTO windows (id, recording, file, track, offset_ms, start_ms, end_ms, done) VALUES
                   (1, 'r1', '00-mic.oga', 'room', 0, 0, 8000, 1);
                 INSERT INTO segments (recording, window, track, start_ms, end_ms, text) VALUES
                   ('r1', 1, 'room', 1000, 7000, 'a');",
            )
            .unwrap();
        }
        assert!(step(&app).await.unwrap());

        let db = app.db.lock().await;
        let status: String = db.query_row("SELECT status FROM recordings", [], |r| r.get(0)).unwrap();
        assert_eq!(status, "diarized");
        let speaker: String = db.query_row("SELECT speaker FROM lines", [], |r| r.get(0)).unwrap();
        assert_eq!(speaker, "room/S1");
        let emb: Vec<u8> = db
            .query_row("SELECT embedding FROM clusters", [], |r| r.get(0))
            .unwrap();
        assert!(!crate::assemble::floats(&emb).is_empty());
        let n: i64 = db
            .query_row("SELECT COUNT(*) FROM line_voices", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 1);
    }
}
