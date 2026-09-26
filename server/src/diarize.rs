use std::{
    collections::BTreeMap,
    ffi::{c_char, CString},
    path::Path,
    sync::Arc,
    time::Duration,
};

use anyhow::{bail, Context, Result};
use rusqlite::{params, OptionalExtension};
use tokio::process::Command;

use crate::{
    api::App,
    assemble::{cosine, derive, floats, normalize, store_turns, Turn},
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

/// Sets each turn's normalized embedding, or None when it is too short.
fn embed_turns(model: &str, samples: &[f32], turns: &mut [Turn]) -> Result<()> {
    let ex = Extractor::new(model)?;
    let at = |ms: i64| (ms.max(0) as usize * RATE / 1000).min(samples.len());
    for t in turns {
        t.embedding = ex
            .embed(&samples[at(t.start_ms)..at(t.end_ms).max(at(t.start_ms))])
            .map(|mut v| {
                normalize(&mut v);
                v
            });
    }
    Ok(())
}

/// The name of the voice most similar to `emb`, if at least `threshold` similar.
// ponytail: one voice per name; the same person across a room/remote channel change scores
// ~0.6 and stays unnamed (spike S4). Keep one voice per track kind if that matters.
fn best_match<'a>(emb: &[f32], voices: &'a [(String, Vec<f32>)], threshold: f32) -> Option<&'a str> {
    voices
        .iter()
        .map(|(name, v)| (name, cosine(emb, v)))
        .filter(|&(_, c)| c >= threshold)
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(name, _)| name.as_str())
}

/// Diarizes the oldest windowed recording whose windows are all transcribed: stores each
/// track's turns and their embeddings, derives its lines and pre-fills names from known
/// voices. Returns false when there is none, or when it failed and will be retried after a
/// pause.
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
    let res = turns(app, &id).await;
    *app.diarizing.lock().unwrap() = None;
    let mut db = app.db.lock().await;
    let tracks = match res {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{id}: diarizing: {e:#}");
            return Ok(crate::db::fail(&db, &id, &format!("diarizing: {e:#}"))?);
        }
    };
    let tx = db.transaction()?;
    for (track, turns) in &tracks {
        store_turns(&tx, &id, track, turns)?;
    }
    derive(&tx, &id)?;
    lookup_names(&tx, &id, match_threshold()?)?;
    tx.execute(
        "UPDATE recordings SET status = 'diarized', attempts = 0 WHERE id = ?1",
        [&id],
    )?;
    tx.commit()?;
    Ok(true)
}

fn match_threshold() -> Result<f32> {
    match std::env::var("MICTAP_MATCH_THRESHOLD") {
        Ok(v) => v.parse().context("MICTAP_MATCH_THRESHOLD"),
        Err(_) => Ok(0.75),
    }
}

/// sherpa's turns of each track of `id`, with their embeddings.
async fn turns(app: &App, id: &str) -> Result<BTreeMap<String, Vec<Turn>>> {
    let model = std::env::var("MICTAP_EMB_MODEL").context("MICTAP_EMB_MODEL not set")?;
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
            let turns = diarize(&wav).await?;
            let samples = read_wav(&wav)?;
            anyhow::Ok((turns, samples))
        }
        .await;
        let _ = std::fs::remove_file(&wav);
        let (mut turns, samples) = res?;
        let model = model.clone();
        let turns = tokio::task::spawn_blocking(move || {
            embed_turns(&model, &samples, &mut turns)?;
            anyhow::Ok(turns)
        })
        .await??;
        out.insert(track, turns);
    }
    Ok(out)
}

/// Name lookup: pre-fills the names of `id` (unless any are set) with the known voice of
/// another recording most similar to each cluster, if at least `threshold` alike.
fn lookup_names(db: &rusqlite::Connection, id: &str, threshold: f32) -> Result<()> {
    let clusters: Vec<(String, Vec<f32>)> = db
        .prepare("SELECT label, embedding FROM clusters WHERE recording = ?1")?
        .query_map([id], |r| Ok((r.get(0)?, floats(&r.get::<_, Vec<u8>>(1)?))))?
        .collect::<rusqlite::Result<_>>()?;
    let voices: Vec<(String, Vec<f32>)> = db
        .prepare("SELECT name, embedding FROM voices WHERE recording != ?1")?
        .query_map([id], |r| Ok((r.get(0)?, floats(&r.get::<_, Vec<u8>>(1)?))))?
        .collect::<rusqlite::Result<_>>()?;
    let names: BTreeMap<&str, &str> = clusters
        .iter()
        .filter_map(|(label, emb)| Some((label.as_str(), best_match(emb, &voices, threshold)?)))
        .collect();
    db.execute(
        "UPDATE recordings SET speakers = ?2 WHERE id = ?1 AND speakers IS NULL",
        params![id, serde_json::to_string(&names)?],
    )?;
    Ok(())
}

/// Diarizes one recording at a time.
pub async fn run(app: Arc<App>) {
    loop {
        match step(&app).await {
            Ok(true) => continue,
            Ok(false) => {}
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
    fn matches_the_most_similar_voice_above_threshold() {
        let voices = [
            ("Max".to_string(), vec![1.0, 0.0, 0.0]),
            ("Eva".to_string(), vec![0.0, 1.0, 0.0]),
            ("Max".to_string(), vec![0.8, 0.6, 0.0]),
        ];
        assert_eq!(best_match(&[0.9, 0.1, 0.0], &voices, 0.6), Some("Max"));
        assert_eq!(best_match(&[0.3, 2.0, 0.0], &voices, 0.6), Some("Eva"));
        // Unnormalized input: cosine ignores length.
        assert_eq!(best_match(&[8.0, 6.0, 0.0], &voices[1..], 0.99), Some("Max"));
        assert_eq!(best_match(&[0.0, 0.0, 1.0], &voices, 0.6), None);
        assert_eq!(best_match(&[1.0, 1.0, 0.0], &voices[..2], 0.8), None);
        assert_eq!(best_match(&[0.0, 0.0, 0.0], &voices, 0.0), None);
        assert_eq!(best_match(&[1.0, 0.0], &voices, 0.0), None, "other model's dimension");
        assert_eq!(best_match(&[1.0, 0.0, 0.0], &[], 0.0), None);
    }

    #[test]
    fn prefills_matched_clusters_but_keeps_names_set() {
        let db = crate::db::open(Path::new(":memory:")).unwrap();
        let bytes = crate::assemble::bytes;
        for id in ["old", "r1", "r2"] {
            crate::db::ensure_recording(&db, id, "laptop").unwrap();
        }
        db.execute(
            "INSERT INTO voices (name, embedding, recording, label) VALUES
             ('Max', ?1, 'old', 'room/S1'), ('Eva', ?2, 'old', 'remote/S1'), ('Self', ?3, 'r1', 'room/S3')",
            params![
                bytes(&[1.0, 0.0, 0.0]),
                bytes(&[0.0, 1.0, 0.0]),
                bytes(&[0.0, 0.0, 1.0])
            ],
        )
        .unwrap();
        let clusters = |id: &str, xs: &[(&str, [f32; 3])]| {
            for (l, v) in xs {
                db.execute(
                    "INSERT INTO clusters (recording, label, embedding) VALUES (?1, ?2, ?3)",
                    params![id, l, bytes(v)],
                )
                .unwrap();
            }
        };
        let speakers = |id: &str| -> String {
            db.query_row("SELECT speakers FROM recordings WHERE id = ?1", [id], |r| r.get(0))
                .unwrap()
        };
        clusters(
            "r1",
            &[
                ("room/S1", [0.1, 0.9, 0.0]),
                ("room/S2", [0.7, 0.7, 0.1]),
                ("room/S3", [0.0, 0.1, 1.0]),
                ("remote/S1", [0.95, 0.0, 0.1]),
            ],
        );
        lookup_names(&db, "r1", 0.9).unwrap();
        assert_eq!(speakers("r1"), r#"{"remote/S1":"Max","room/S1":"Eva"}"#);

        db.execute(
            "UPDATE recordings SET speakers = '{\"room/S1\":\"Jo\"}' WHERE id = 'r2'",
            [],
        )
        .unwrap();
        clusters("r2", &[("room/S1", [0.0, 1.0, 0.0])]);
        lookup_names(&db, "r2", 0.6).unwrap();
        assert_eq!(speakers("r2"), r#"{"room/S1":"Jo"}"#);
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
        let emb: Vec<u8> = db.query_row("SELECT embedding FROM clusters", [], |r| r.get(0)).unwrap();
        assert!(!floats(&emb).is_empty());
    }
}
