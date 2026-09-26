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

use crate::api::App;

const RATE: usize = 16_000;

#[derive(Debug, PartialEq)]
struct Turn {
    start_ms: i64,
    end_ms: i64,
    /// Index in order of first appearance: 0 is `S1`.
    speaker: usize,
}

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
            }
        })
        .collect()
}

/// The speaker with the most overlap with `[s, e)`, else the one of the nearest turn.
fn assign(turns: &[Turn], s: i64, e: i64) -> Option<usize> {
    let n = turns.iter().map(|t| t.speaker + 1).max()?;
    let mut overlap = vec![0; n];
    for t in turns {
        overlap[t.speaker] += (e.min(t.end_ms) - s.max(t.start_ms)).max(0);
    }
    let (best, &most) = overlap
        .iter()
        .enumerate()
        .max_by_key(|&(i, o)| (*o, std::cmp::Reverse(i)))?;
    if most > 0 {
        return Some(best);
    }
    turns
        .iter()
        .min_by_key(|t| (t.start_ms - e).max(s - t.end_ms))
        .map(|t| t.speaker)
}

/// A speaker's run of words shorter than this joins its neighbour instead of becoming a line.
const MIN_RUN_MS: i64 = 1_000;

/// A cut moves up to this many words to end a line on a sentence or clause.
const SNAP_WORDS: usize = 2;

/// Segment `[s, e)` as `(start_ms, end_ms, text, speaker)` lines, cut where the speaker
/// changes, since whisper segments often span a reply. The speaker of each word is the one
/// its share of the segment overlaps most.
// ponytail: words spread evenly over the segment, as merge.rs does; whisper-cli's token
// timestamps (-ojf) would place cuts better if they land mid-phrase.
fn split(turns: &[Turn], s: i64, e: i64, text: &str) -> Vec<(i64, i64, String, usize)> {
    let Some(whole) = assign(turns, s, e) else {
        return vec![];
    };
    let words: Vec<&str> = text.split_whitespace().collect();
    let step = (e - s) as f64 / words.len().max(1) as f64;
    let at = |i: usize| s + (i as f64 * step).round() as i64;
    // [from, to) word ranges and their speaker.
    let mut runs: Vec<(usize, usize, usize)> = vec![];
    for i in 0..words.len() {
        let k = assign(turns, at(i), at(i + 1)).unwrap_or(whole);
        match runs.last_mut() {
            Some(r) if r.2 == k => r.1 = i + 1,
            _ => runs.push((i, i + 1, k)),
        }
    }
    let short = |r: &(usize, usize, usize)| ((r.1 - r.0) as f64 * step) < MIN_RUN_MS as f64;
    let mut lines: Vec<(usize, usize, usize)> = vec![];
    for r in runs {
        match lines.last_mut() {
            Some(l) if l.2 == r.2 || short(&r) => l.1 = r.1,
            // Only the first line can still be short: it takes the next speaker.
            Some(l) if short(l) => *l = (l.0, r.1, r.2),
            _ => lines.push(r),
        }
    }
    if lines.len() < 2 {
        return vec![(s, e, text.to_string(), whole)];
    }
    for j in 1..lines.len() {
        let cut = lines[j].0;
        let ends = |c: usize| words[c - 1].ends_with(['.', '?', '!', ',', ';', ':']);
        let snap = (0..=SNAP_WORDS)
            .flat_map(|d| [cut.checked_sub(d), Some(cut + d)])
            .flatten()
            .find(|&c| c > lines[j - 1].0 && c < lines[j].1 && ends(c));
        if let Some(c) = snap {
            lines[j - 1].1 = c;
            lines[j].0 = c;
        }
    }
    lines
        .into_iter()
        .map(|(from, to, k)| (at(from), at(to), words[from..to].join(" "), k))
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

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

/// Each turn's normalized embedding, or None when it is too short.
fn turn_embeddings(model: &str, samples: &[f32], turns: &[Turn]) -> Result<Vec<Option<Vec<f32>>>> {
    let ex = Extractor::new(model)?;
    let at = |ms: i64| (ms.max(0) as usize * RATE / 1000).min(samples.len());
    Ok(turns
        .iter()
        .map(|t| {
            let mut v = ex.embed(&samples[at(t.start_ms)..at(t.end_ms).max(at(t.start_ms))])?;
            normalize(&mut v);
            Some(v)
        })
        .collect())
}

/// Per speaker, the normalized mean of its turns' embeddings, weighted by turn length.
fn means(turns: &[Turn], embs: &[Option<Vec<f32>>]) -> Vec<Option<Vec<f32>>> {
    let n = turns.iter().map(|t| t.speaker + 1).max().unwrap_or(0);
    let mut sums: Vec<Option<Vec<f32>>> = vec![None; n];
    for (t, v) in turns.iter().zip(embs) {
        let Some(v) = v else { continue };
        let w = (t.end_ms - t.start_ms) as f32;
        match &mut sums[t.speaker] {
            Some(sum) => sum.iter_mut().zip(v).for_each(|(a, b)| *a += w * b),
            slot => *slot = Some(v.iter().map(|b| w * b).collect()),
        }
    }
    sums.iter_mut().flatten().for_each(|v| normalize(v));
    sums
}

/// Clusters with less speech than this are folded into the most similar larger one.
const MIN_CLUSTER_MS: i64 = 10_000;

/// sherpa splits a speaker into many clusters, most of them fragments of a few seconds (spike
/// S3; a 3-person meeting got 39). Folds the fragments into the most similar larger cluster,
/// then merges the most similar pair while their means are at least `threshold` alike, and
/// renumbers speakers by first appearance. Never splits a cluster.
fn merge(turns: &mut [Turn], embs: &[Option<Vec<f32>>], threshold: f32) {
    let n = turns.iter().map(|t| t.speaker + 1).max().unwrap_or(0);
    let mut len = vec![0; n];
    for t in turns.iter() {
        len[t.speaker] += t.end_ms - t.start_ms;
    }
    let relabel = |turns: &mut [Turn], from: usize, to: usize| {
        turns
            .iter_mut()
            .filter(|t| t.speaker == from)
            .for_each(|t| t.speaker = to);
    };

    let m = means(turns, embs);
    let big: Vec<usize> = (0..n).filter(|&i| len[i] >= MIN_CLUSTER_MS && m[i].is_some()).collect();
    for small in (0..n).filter(|&i| len[i] < MIN_CLUSTER_MS) {
        let Some(v) = &m[small] else { continue };
        let nearest = big
            .iter()
            .map(|&b| (cosine(v, m[b].as_ref().unwrap()), b))
            .filter(|p| p.0.is_finite())
            .max_by(|a, b| a.0.total_cmp(&b.0));
        if let Some((_, b)) = nearest {
            relabel(turns, small, b);
        }
    }

    loop {
        let m = means(turns, embs);
        let live: Vec<usize> = (0..m.len()).filter(|&i| m[i].is_some()).collect();
        let best = live
            .iter()
            .enumerate()
            .flat_map(|(k, &a)| live[k + 1..].iter().map(move |&b| (a, b)))
            .map(|(a, b)| (cosine(m[a].as_ref().unwrap(), m[b].as_ref().unwrap()), a, b))
            .filter(|p| p.0 >= threshold)
            .max_by(|a, b| a.0.total_cmp(&b.0));
        let Some((_, a, b)) = best else { break };
        relabel(turns, b, a);
    }

    let mut seen: Vec<usize> = vec![];
    for t in turns.iter_mut() {
        t.speaker = seen.iter().position(|&s| s == t.speaker).unwrap_or_else(|| {
            seen.push(t.speaker);
            seen.len() - 1
        });
    }
}

fn floats(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect()
}

/// NaN, which matches nothing, for a zero vector or vectors of different lengths.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f32>();
    if a.len() != b.len() {
        return f32::NAN;
    }
    dot(a, b) / (dot(a, a) * dot(b, b)).sqrt()
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

/// Diarizes the oldest windowed recording whose windows are all transcribed: labels its
/// segments `track/S<n>`, stores each cluster's embedding and names the clusters that match a
/// known voice. Returns false when there is none,
/// or when it failed and will be retried after a pause.
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
    let db = || app.db.lock();
    *app.diarizing.lock().unwrap() = Some((id.clone(), crate::db::now_ms()));
    let res = label(app, &id).await;
    *app.diarizing.lock().unwrap() = None;
    match res {
        Ok(()) => {
            db().await.execute(
                "UPDATE recordings SET status = 'diarized', attempts = 0 WHERE id = ?1",
                [&id],
            )?;
            Ok(true)
        }
        Err(e) => {
            eprintln!("{id}: diarizing: {e:#}");
            Ok(crate::db::fail(&*db().await, &id, &format!("diarizing: {e:#}"))?)
        }
    }
}

async fn label(app: &App, id: &str) -> Result<()> {
    let model = std::env::var("MICTAP_EMB_MODEL").context("MICTAP_EMB_MODEL not set")?;
    let threshold: f32 = match std::env::var("MICTAP_MATCH_THRESHOLD") {
        Ok(v) => v.parse().context("MICTAP_MATCH_THRESHOLD")?,
        Err(_) => 0.75,
    };
    let merge_threshold: f32 = match std::env::var("MICTAP_MERGE_THRESHOLD") {
        Ok(v) => v.parse().context("MICTAP_MERGE_THRESHOLD")?,
        Err(_) => 0.75,
    };
    let mut tracks: BTreeMap<String, Vec<(String, i64)>> = BTreeMap::new();
    let mut segments: BTreeMap<String, Vec<(i64, i64, i64, String)>> = BTreeMap::new();
    {
        let db = app.db.lock().await;
        let mut st =
            db.prepare("SELECT DISTINCT track, file, offset_ms FROM windows WHERE recording = ?1 ORDER BY file")?;
        for r in st.query_map([id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))? {
            let (track, file, offset): (String, String, i64) = r?;
            tracks.entry(track).or_default().push((file, offset));
        }
        let mut st = db.prepare("SELECT track, id, start_ms, end_ms, text FROM segments WHERE recording = ?1")?;
        for r in st.query_map([id], |r| Ok((r.get(0)?, (r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))))? {
            let (track, seg): (String, _) = r?;
            segments.entry(track).or_default().push(seg);
        }
    }

    let dir = app.recording_dir(id);
    let mut labels: Vec<(i64, Vec<Line>)> = vec![];
    let mut clusters: Vec<(String, Vec<f32>)> = vec![];
    for (track, files) in &tracks {
        let wav = dir.join(format!(".{track}.wav"));
        let res = async {
            timeline(&dir, files, &wav).await?;
            let turns = diarize(&wav).await?;
            let samples = read_wav(&wav)?;
            anyhow::Ok((turns, samples))
        }
        .await;
        let _ = std::fs::remove_file(&wav);
        let (mut turns, samples) = res?;
        let model = model.clone();
        let (turns, embeddings) = tokio::task::spawn_blocking(move || {
            let embs = turn_embeddings(&model, &samples, &turns)?;
            merge(&mut turns, &embs, merge_threshold);
            let m = means(&turns, &embs);
            anyhow::Ok((turns, m))
        })
        .await??;
        let name = |i: usize| format!("{track}/S{}", i + 1);
        for (sid, s, e, text) in segments.get(track).into_iter().flatten() {
            let lines = split(&turns, *s, *e, text);
            if !lines.is_empty() {
                labels.push((*sid, lines.into_iter().map(|(s, e, t, k)| (s, e, t, name(k))).collect()));
            }
        }
        clusters.extend(
            embeddings
                .into_iter()
                .enumerate()
                .filter_map(|(i, v)| Some((name(i), v?))),
        );
    }

    save(&mut *app.db.lock().await, id, &labels, &clusters, threshold)
}

/// `(start_ms, end_ms, text, label)` of a segment, or of a part of one after `split`.
type Line = (i64, i64, String, String);

/// Labels each segment, replacing it by its lines when it was split, stores the clusters'
/// embeddings, and pre-fills the speakers map with the clusters that match a voice of
/// another recording.
fn save(
    db: &mut rusqlite::Connection,
    id: &str,
    labels: &[(i64, Vec<Line>)],
    clusters: &[(String, Vec<f32>)],
    threshold: f32,
) -> Result<()> {
    let tx = db.transaction()?;
    for (sid, lines) in labels {
        let Some(((_, end, text, label), rest)) = lines.split_first() else {
            continue;
        };
        if rest.is_empty() {
            tx.execute("UPDATE segments SET speaker = ?2 WHERE id = ?1", params![sid, label])?;
            continue;
        }
        tx.execute(
            "UPDATE segments SET end_ms = ?2, text = ?3, speaker = ?4 WHERE id = ?1",
            params![sid, end, text, label],
        )?;
        for (s, e, text, label) in rest {
            tx.execute(
                "INSERT INTO segments (recording, window, track, start_ms, end_ms, text, speaker)
                 SELECT recording, window, track, ?2, ?3, ?4, ?5 FROM segments WHERE id = ?1",
                params![sid, s, e, text, label],
            )?;
        }
    }
    for (label, emb) in clusters {
        let bytes: Vec<u8> = emb.iter().flat_map(|x| x.to_le_bytes()).collect();
        tx.execute(
            "INSERT OR REPLACE INTO clusters (recording, label, embedding) VALUES (?1, ?2, ?3)",
            params![id, label, bytes],
        )?;
    }
    let voices: Vec<(String, Vec<f32>)> = tx
        .prepare("SELECT name, embedding FROM voices WHERE recording != ?1")?
        .query_map([id], |r| Ok((r.get(0)?, floats(&r.get::<_, Vec<u8>>(1)?))))?
        .collect::<rusqlite::Result<_>>()?;
    let names: BTreeMap<&str, &str> = clusters
        .iter()
        .filter_map(|(label, emb)| Some((label.as_str(), best_match(emb, &voices, threshold)?)))
        .collect();
    // Never replaces names already set.
    tx.execute(
        "UPDATE recordings SET speakers = ?2 WHERE id = ?1 AND speakers IS NULL",
        params![id, serde_json::to_string(&names)?],
    )?;
    tx.commit()?;
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
    fn assigns_most_overlap_then_nearest() {
        let turns = [
            t(0, 4_000, 0),
            t(4_000, 6_000, 1),
            t(6_000, 7_000, 0),
            t(20_000, 30_000, 1),
        ];
        // S1 overlaps 1 s + 1 s, S2 1.5 s.
        assert_eq!(assign(&turns, 3_000, 7_000), Some(0));
        assert_eq!(assign(&turns, 3_500, 6_500), Some(1));
        // No overlap: nearest turn.
        assert_eq!(assign(&turns, 8_000, 9_000), Some(0));
        assert_eq!(assign(&turns, 15_000, 19_000), Some(1));
        assert_eq!(assign(&[], 0, 1_000), None);
    }

    #[test]
    fn splits_segments_where_the_speaker_changes() {
        let line = |s: i64, e: i64, text: &str, k: usize| (s, e, text.to_string(), k);
        // 9 words over 8 s; the change at 4 s falls in "Ja", the cut moves back to "Wesley?".
        let turns = [t(0, 4_000, 0), t(4_000, 8_000, 1)];
        assert_eq!(
            split(&turns, 0, 8_000, "Heb jij tijd, Wesley? Ja hoor dat kan wel."),
            [
                line(0, 3_556, "Heb jij tijd, Wesley?", 0),
                line(3_556, 8_000, "Ja hoor dat kan wel.", 1)
            ]
        );
        // Without punctuation nearby, the cut stays at the change.
        assert_eq!(
            split(&turns, 0, 8_000, "een twee drie vier vijf zes zeven acht"),
            [
                line(0, 4_000, "een twee drie vier", 0),
                line(4_000, 8_000, "vijf zes zeven acht", 1)
            ]
        );
        // Runs under a second don't become lines: a blip, or a short first run.
        let text = "een twee drie vier vijf zes zeven acht negen tien";
        let blip = [t(0, 2_000, 0), t(2_000, 2_600, 1), t(2_600, 5_000, 0)];
        assert_eq!(split(&blip, 0, 5_000, text), [line(0, 5_000, text, 0)]);
        let late = [t(0, 400, 0), t(400, 5_000, 1)];
        assert_eq!(split(&late, 0, 5_000, text), [line(0, 5_000, text, 1)]);
        assert_eq!(split(&[], 0, 5_000, text), []);
    }

    #[test]
    fn saves_split_segments_as_lines() {
        let mut db = crate::db::open(Path::new(":memory:")).unwrap();
        crate::db::ensure_recording(&db, "r1", "laptop").unwrap();
        db.execute_batch(
            "INSERT INTO windows (id, recording, file, track, offset_ms, start_ms, end_ms)
               VALUES (1, 'r1', '00-mic.oga', 'room', 0, 0, 30000);
             INSERT INTO segments (id, recording, window, track, start_ms, end_ms, text)
               VALUES (1, 'r1', 1, 'room', 0, 8000, 'a b'), (2, 'r1', 1, 'room', 9000, 9500, 'c');",
        )
        .unwrap();
        let l = |s: i64, e: i64, text: &str, label: &str| (s, e, text.to_string(), label.to_string());
        let labels = [
            (1, vec![l(0, 4_000, "a", "room/S1"), l(4_000, 8_000, "b", "room/S2")]),
            (2, vec![l(9_000, 9_500, "c", "room/S2")]),
        ];
        save(&mut db, "r1", &labels, &[], 0.75).unwrap();
        let rows: Vec<(i64, i64, i64, String, String)> = db
            .prepare("SELECT window, start_ms, end_ms, text, speaker FROM segments ORDER BY start_ms")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        let row = |s: i64, e: i64, text: &str, spk: &str| (1, s, e, text.to_string(), spk.to_string());
        assert_eq!(
            rows,
            [
                row(0, 4_000, "a", "room/S1"),
                row(4_000, 8_000, "b", "room/S2"),
                row(9_000, 9_500, "c", "room/S2")
            ]
        );
    }

    #[test]
    fn folds_fragments_and_merges_alike_clusters() {
        let mut turns = [
            t(0, 20_000, 0),
            t(20_000, 35_000, 1),
            t(35_000, 40_000, 2),
            t(40_000, 70_000, 3),
            t(70_000, 71_000, 4),
            t(71_000, 72_000, 5),
        ];
        let unit = |v: [f32; 3]| {
            let mut v = v.to_vec();
            normalize(&mut v);
            Some(v)
        };
        let embs = [
            unit([1.0, 0.0, 0.0]),
            // Same speaker as S1: merged.
            unit([0.9, 0.1, 0.0]),
            // A fragment, closest to S4 though below the threshold: folded.
            unit([0.0, 1.0, 0.0]),
            unit([0.0, 0.8, 0.6]),
            // A fragment without an embedding: kept.
            None,
            // A fragment alike to nothing big still goes to the nearest.
            unit([0.0, 0.0, -1.0]),
        ];
        merge(&mut turns, &embs, 0.9);
        let speakers: Vec<usize> = turns.iter().map(|t| t.speaker).collect();
        assert_eq!(speakers, [0, 0, 1, 1, 2, 0]);

        let mut two = [t(0, 20_000, 0), t(20_000, 40_000, 1)];
        merge(&mut two, &[unit([1.0, 0.0, 0.0]), unit([0.6, 0.8, 0.0])], 0.7);
        assert_eq!(two.map(|t| t.speaker), [0, 1], "0.6 < 0.7 stays apart");
    }

    #[test]
    fn weights_means_by_turn_length() {
        let turns = [t(0, 3_000, 0), t(3_000, 4_000, 0), t(4_000, 5_000, 2)];
        let m = means(&turns, &[Some(vec![1.0, 0.0]), Some(vec![0.0, 1.0]), None]);
        let s = m[0].as_ref().unwrap();
        assert!((s[0] - 0.9487).abs() < 1e-3 && (s[1] - 0.3162).abs() < 1e-3, "{s:?}");
        assert_eq!(m[1..], [None, None]);
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
        let mut db = crate::db::open(Path::new(":memory:")).unwrap();
        let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
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
        let clusters = |xs: &[(&str, [f32; 3])]| -> Vec<(String, Vec<f32>)> {
            xs.iter().map(|(l, v)| (l.to_string(), v.to_vec())).collect()
        };
        let speakers = |db: &rusqlite::Connection, id: &str| -> String {
            db.query_row("SELECT speakers FROM recordings WHERE id = ?1", [id], |r| r.get(0))
                .unwrap()
        };
        let r1 = clusters(&[
            ("room/S1", [0.1, 0.9, 0.0]),
            ("room/S2", [0.7, 0.7, 0.1]),
            ("room/S3", [0.0, 0.1, 1.0]),
            ("remote/S1", [0.95, 0.0, 0.1]),
        ]);
        save(&mut db, "r1", &[], &r1, 0.9).unwrap();
        assert_eq!(speakers(&db, "r1"), r#"{"remote/S1":"Max","room/S1":"Eva"}"#);
        let stored: Vec<f32> = db
            .query_row(
                "SELECT embedding FROM clusters WHERE recording = 'r1' AND label = 'room/S2'",
                [],
                |r| Ok(floats(&r.get::<_, Vec<u8>>(0)?)),
            )
            .unwrap();
        assert_eq!(stored, [0.7, 0.7, 0.1]);

        db.execute(
            "UPDATE recordings SET speakers = '{\"room/S1\":\"Jo\"}' WHERE id = 'r2'",
            [],
        )
        .unwrap();
        save(&mut db, "r2", &[], &clusters(&[("room/S1", [0.0, 1.0, 0.0])]), 0.6).unwrap();
        assert_eq!(speakers(&db, "r2"), r#"{"room/S1":"Jo"}"#);
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
        let speaker: String = db.query_row("SELECT speaker FROM segments", [], |r| r.get(0)).unwrap();
        assert_eq!(speaker, "room/S1");
        let emb: Vec<u8> = db.query_row("SELECT embedding FROM clusters", [], |r| r.get(0)).unwrap();
        assert!(!floats(&emb).is_empty());
    }
}
