//! Assembly: derives a recording's lines and clusters from its model outputs (see
//! CONTEXT.md). `assemble` is pure; `derive` loads the outputs and stores what it returns.
use std::collections::{BTreeMap, HashMap, HashSet};

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

const NEAR_MS: i64 = 1_000;
const SLICE_MS: i64 = 250;
const DROP_CLUSTER: f64 = 0.7;

/// What the models produced for a recording, per track (`room`, `remote`).
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Outputs {
    pub tracks: BTreeMap<String, Track>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Track {
    /// whisper's, within the recording.
    pub segments: Vec<Segment>,
    /// sherpa's, within the recording; empty until diarized.
    pub turns: Vec<Turn>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Turn {
    pub start_ms: i64,
    pub end_ms: i64,
    /// sherpa's cluster, numbered by first appearance; during assembly, the speaker it was
    /// folded or merged into.
    pub speaker: usize,
    /// CAM++'s, normalized; None when the turn is too short.
    pub embedding: Option<Vec<f32>>,
}

/// A segment, or part of one, attributed to a label (`room/S1`) once diarized.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Line {
    pub track: String,
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
    pub speaker: Option<String>,
}

pub struct Tuning {
    /// Clusters whose means are at least this alike are merged.
    pub merge_threshold: f32,
    /// A room line at least this alike (token Jaccard) to nearby remote speech is an echo.
    pub echo_jaccard: f64,
}

impl Tuning {
    /// `MICTAP_MERGE_THRESHOLD` (default 0.75) and `MICTAP_ECHO_JACCARD` (default 0.6).
    pub fn from_env() -> Self {
        let var = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok());
        Self {
            merge_threshold: var("MICTAP_MERGE_THRESHOLD").unwrap_or(0.75) as f32,
            echo_jaccard: var("MICTAP_ECHO_JACCARD").unwrap_or(0.6),
        }
    }
}

pub struct Assembled {
    /// Chronological, echoes dropped.
    pub lines: Vec<Line>,
    /// Each label's mean embedding, for the labels that have lines.
    pub clusters: Vec<(String, Vec<f32>)>,
}

/// Per track, folds and merges sherpa's clusters into speakers and splits each segment into
/// lines where the speaker changes (segments of an undiarized track stay unlabeled); then
/// merges the tracks and drops room echoes of remote speech.
pub fn assemble(outputs: &Outputs, tuning: &Tuning) -> Assembled {
    let mut lines = vec![];
    let mut clusters = vec![];
    for (track, t) in &outputs.tracks {
        let mut turns = t.turns.clone();
        cluster(&mut turns, tuning.merge_threshold);
        let label = |k: usize| format!("{track}/S{}", k + 1);
        for s in &t.segments {
            let parts = split(&turns, s.start_ms, s.end_ms, &s.text);
            if parts.is_empty() {
                lines.push(Line {
                    track: track.clone(),
                    start_ms: s.start_ms,
                    end_ms: s.end_ms,
                    text: s.text.clone(),
                    speaker: None,
                });
            }
            lines.extend(parts.into_iter().map(|(start_ms, end_ms, text, k)| Line {
                track: track.clone(),
                start_ms,
                end_ms,
                text,
                speaker: Some(label(k)),
            }));
        }
        clusters.extend(
            means(&turns)
                .into_iter()
                .enumerate()
                .filter_map(|(k, v)| Some((label(k), v?))),
        );
    }
    let lines = drop_echoes(lines, tuning.echo_jaccard);
    clusters.retain(|(l, _)| lines.iter().any(|x| x.speaker.as_ref() == Some(l)));
    Assembled { lines, clusters }
}

/// The model outputs stored for `id`.
pub fn outputs(db: &Connection, id: &str) -> rusqlite::Result<Outputs> {
    let mut out = Outputs::default();
    let mut st = db.prepare("SELECT track, start_ms, end_ms, text FROM segments WHERE recording = ?1 ORDER BY id")?;
    for r in st.query_map([id], |r| {
        Ok((
            r.get::<_, String>(0)?,
            Segment {
                start_ms: r.get(1)?,
                end_ms: r.get(2)?,
                text: r.get(3)?,
            },
        ))
    })? {
        let (track, s) = r?;
        out.tracks.entry(track).or_default().segments.push(s);
    }
    let mut st =
        db.prepare("SELECT track, start_ms, end_ms, speaker, embedding FROM turns WHERE recording = ?1 ORDER BY id")?;
    for r in st.query_map([id], |r| {
        Ok((
            r.get::<_, String>(0)?,
            Turn {
                start_ms: r.get(1)?,
                end_ms: r.get(2)?,
                speaker: r.get(3)?,
                embedding: r.get::<_, Option<Vec<u8>>>(4)?.map(|b| floats(&b)),
            },
        ))
    })? {
        let (track, t) = r?;
        out.tracks.entry(track).or_default().turns.push(t);
    }
    Ok(out)
}

/// Replaces the stored turns of `id`'s `track`.
pub fn store_turns(db: &Connection, id: &str, track: &str, turns: &[Turn]) -> rusqlite::Result<()> {
    db.execute(
        "DELETE FROM turns WHERE recording = ?1 AND track = ?2",
        params![id, track],
    )?;
    for t in turns {
        db.execute(
            "INSERT INTO turns (recording, track, start_ms, end_ms, speaker, embedding)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                id,
                track,
                t.start_ms,
                t.end_ms,
                t.speaker,
                t.embedding.as_deref().map(bytes)
            ],
        )?;
    }
    Ok(())
}

/// Assembles `id` from its stored outputs and replaces its lines and clusters. Run it in a
/// transaction with the change to the outputs.
pub fn derive(db: &Connection, id: &str) -> rusqlite::Result<()> {
    let a = assemble(&outputs(db, id)?, &Tuning::from_env());
    db.execute("DELETE FROM lines WHERE recording = ?1", [id])?;
    for l in &a.lines {
        db.execute(
            "INSERT INTO lines (recording, track, start_ms, end_ms, text, speaker)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![id, l.track, l.start_ms, l.end_ms, l.text, l.speaker],
        )?;
    }
    db.execute("DELETE FROM clusters WHERE recording = ?1", [id])?;
    for (label, emb) in &a.clusters {
        db.execute(
            "INSERT INTO clusters (recording, label, embedding) VALUES (?1, ?2, ?3)",
            params![id, label, bytes(emb)],
        )?;
    }
    Ok(())
}

/// The derived lines of `id`, chronological.
pub fn lines(db: &Connection, id: &str) -> rusqlite::Result<Vec<Line>> {
    db.prepare("SELECT track, start_ms, end_ms, text, speaker FROM lines WHERE recording = ?1 ORDER BY id")?
        .query_map([id], |r| {
            Ok(Line {
                track: r.get(0)?,
                start_ms: r.get(1)?,
                end_ms: r.get(2)?,
                text: r.get(3)?,
                speaker: r.get(4)?,
            })
        })?
        .collect()
}

pub(crate) fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub(crate) fn floats(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect()
}

fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect()
}

fn jaccard(a: &HashSet<&str>, b: &HashSet<&str>) -> f64 {
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    a.intersection(b).count() as f64 / union as f64
}

/// The words of `r` spoken within `[s, e]` +- `SLICE_MS`, spreading them evenly over its span.
fn slice<'a>(r: &Line, words: &'a [String], s: i64, e: i64) -> impl Iterator<Item = &'a str> {
    let step = (r.end_ms - r.start_ms) as f64 / words.len().max(1) as f64;
    let (r0, s, e) = (r.start_ms as f64, (s - SLICE_MS) as f64, (e + SLICE_MS) as f64);
    words.iter().enumerate().filter_map(move |(i, w)| {
        let t = r0 + (i as f64 + 0.5) * step;
        (s <= t && t <= e).then_some(w.as_str())
    })
}

/// Whether room segment `s` echoes the remote segments within `NEAR_MS` of it: token Jaccard
/// at least `threshold` with one of them whole, or with the remote words spoken in its time
/// span, since whisper splits the same speech differently on each track.
fn is_echo(s: &Line, remote: &[(&Line, Vec<String>)], threshold: f64) -> bool {
    let words = words(&s.text);
    let own: HashSet<&str> = words.iter().map(String::as_str).collect();
    let near: Vec<_> = remote
        .iter()
        .filter(|(r, _)| r.start_ms <= s.end_ms + NEAR_MS && s.start_ms <= r.end_ms + NEAR_MS)
        .collect();
    let spoken: HashSet<&str> = near
        .iter()
        .flat_map(|(r, w)| slice(r, w, s.start_ms, s.end_ms))
        .collect();
    jaccard(&own, &spoken) >= threshold
        || near
            .iter()
            .any(|(_, w)| jaccard(&own, &w.iter().map(String::as_str).collect()) >= threshold)
}

/// Sorts `segs` chronologically and drops `room` echoes of `remote` speech (see `is_echo`),
/// and every segment of a room cluster that lost more than 70% of its segments that way.
pub fn drop_echoes(mut segs: Vec<Line>, threshold: f64) -> Vec<Line> {
    segs.sort_by_key(|s| (s.start_ms, s.end_ms));
    let remote: Vec<(&Line, Vec<String>)> = segs
        .iter()
        .filter(|s| s.track == "remote")
        .map(|s| (s, words(&s.text)))
        .collect();
    let echo: Vec<bool> = segs
        .iter()
        .map(|s| s.track == "room" && is_echo(s, &remote, threshold))
        .collect();
    let mut clusters: HashMap<&str, (usize, usize)> = HashMap::new();
    for (s, &e) in segs.iter().zip(&echo) {
        if let (Some(spk), "room") = (&s.speaker, s.track.as_str()) {
            let c = clusters.entry(spk).or_default();
            c.0 += e as usize;
            c.1 += 1;
        }
    }
    let dropped: HashSet<String> = clusters
        .into_iter()
        .filter(|(_, (e, n))| *e as f64 > DROP_CLUSTER * *n as f64)
        .map(|(k, _)| k.to_string())
        .collect();
    segs.into_iter()
        .zip(echo)
        .filter(|(s, e)| !e && !s.speaker.as_ref().is_some_and(|k| dropped.contains(k)))
        .map(|(s, _)| s)
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
// ponytail: words spread evenly over the segment, as drop_echoes does; whisper-cli's token
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

pub(crate) fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

/// Per speaker, the normalized mean of its turns' embeddings, weighted by turn length.
fn means(turns: &[Turn]) -> Vec<Option<Vec<f32>>> {
    let n = turns.iter().map(|t| t.speaker + 1).max().unwrap_or(0);
    let mut sums: Vec<Option<Vec<f32>>> = vec![None; n];
    for t in turns {
        let Some(v) = &t.embedding else { continue };
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
fn cluster(turns: &mut [Turn], threshold: f32) {
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

    let m = means(turns);
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
        let m = means(turns);
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

/// NaN, which matches nothing, for a zero vector or vectors of different lengths.
pub(crate) fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f32>();
    if a.len() != b.len() {
        return f32::NAN;
    }
    dot(a, b) / (dot(a, a) * dot(b, b)).sqrt()
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

    fn with(mut turns: Vec<Turn>, embs: impl IntoIterator<Item = Option<Vec<f32>>>) -> Vec<Turn> {
        turns.iter_mut().zip(embs).for_each(|(t, e)| t.embedding = e);
        turns
    }

    fn s(track: &str, start_ms: i64, end_ms: i64, text: &str, speaker: &str) -> Line {
        Line {
            track: track.into(),
            start_ms,
            end_ms,
            text: text.into(),
            speaker: (!speaker.is_empty()).then(|| speaker.into()),
        }
    }

    #[test]
    fn jaccard_on_normalized_tokens() {
        let j = |a, b| {
            let (a, b) = (words(a), words(b));
            jaccard(
                &a.iter().map(String::as_str).collect(),
                &b.iter().map(String::as_str).collect(),
            )
        };
        assert_eq!(j("Hallo, Wereld!", "hallo wereld"), 1.0);
        assert_eq!(j("a b c d", "a b c e"), 0.6);
        assert_eq!(j("", ""), 0.0);
    }

    #[test]
    fn drops_room_echo_near_similar_remote() {
        let segs = vec![
            s("room", 10_000, 12_000, "Het Westerwald is een gebergte", "room/S2"),
            s(
                "remote",
                8_500,
                11_800,
                "het westerwald is een laaggebergte",
                "remote/S1",
            ),
            // Similar but too far away.
            s("room", 13_000, 15_000, "het westerwald is een laaggebergte", "room/S1"),
            // Near but different.
            s("room", 9_000, 11_000, "yes I agree", "room/S1"),
            s("room", 0, 2_000, "good morning", "room/S1"),
        ];
        let out = drop_echoes(segs.clone(), 0.6);
        assert_eq!(
            out,
            vec![segs[4].clone(), segs[1].clone(), segs[3].clone(), segs[2].clone()]
        );
        // The threshold is honored.
        assert_eq!(drop_echoes(segs, 0.9).len(), 5);
    }

    #[test]
    fn drops_echo_split_differently() {
        let long = "Dit is de gesproken versie van het artikel Westerwald, zoals het op 5 \
                    februari 2017 op de Nederlandse Wikipedia stond.";
        let halves = [
            (26_400, 29_560, "Dit is de gesproken versie van het artikel Westerwald."),
            (
                30_020,
                34_280,
                "zoals het op 5 februari 2017 op de Nederlandse Wikipedia stond.",
            ),
        ];
        // One remote segment, echoed as two room segments.
        let mut segs = vec![s("remote", 26_390, 34_130, long, "")];
        segs.extend(halves.map(|(a, b, t)| s("room", a, b, t, "")));
        assert_eq!(drop_echoes(segs, 0.6).len(), 1);
        // Two remote segments, echoed as one.
        let mut segs: Vec<_> = halves.map(|(a, b, t)| s("remote", a, b, t, "")).into();
        segs.push(s("room", 26_540, 34_430, long, ""));
        assert_eq!(drop_echoes(segs, 0.6).len(), 2);
    }

    #[test]
    fn drops_mostly_echoed_room_cluster() {
        let r = |start: i64| s("remote", start, start + 2_000, "precies wat hij zei", "remote/S1");
        let echo = |start: i64| s("room", start + 150, start + 2_150, "precies wat hij zei", "room/S2");
        let mut segs = vec![r(0), echo(0), r(5_000), echo(5_000), r(10_000), echo(10_000)];
        let leftover = s("room", 20_000, 21_000, "hmm", "room/S2");
        let own = s("room", 22_000, 23_000, "ok", "room/S1");
        segs.extend([leftover.clone(), own.clone()]);
        // 3 of 4 echoed (75%): the whole cluster goes.
        let out = drop_echoes(segs.clone(), 0.6);
        assert!(!out.contains(&leftover));
        assert_eq!(out.last(), Some(&own));
        assert_eq!(out.len(), 4);
        // 3 of 5 (60%): the non-echo segments stay.
        segs.push(s("room", 25_000, 26_000, "nog iets", "room/S2"));
        let out = drop_echoes(segs, 0.6);
        assert!(out.contains(&leftover));
        assert_eq!(out.len(), 6);
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
    fn folds_fragments_and_merges_alike_clusters() {
        let turns = vec![
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
        let mut turns = with(turns, embs);
        cluster(&mut turns, 0.9);
        let speakers: Vec<usize> = turns.iter().map(|t| t.speaker).collect();
        assert_eq!(speakers, [0, 0, 1, 1, 2, 0]);

        let mut two = with(
            vec![t(0, 20_000, 0), t(20_000, 40_000, 1)],
            [unit([1.0, 0.0, 0.0]), unit([0.6, 0.8, 0.0])],
        );
        cluster(&mut two, 0.7);
        assert_eq!(
            two.iter().map(|t| t.speaker).collect::<Vec<_>>(),
            [0, 1],
            "0.6 < 0.7 stays apart"
        );
    }

    #[test]
    fn weights_means_by_turn_length() {
        let turns = [t(0, 3_000, 0), t(3_000, 4_000, 0), t(4_000, 5_000, 2)];
        let turns = with(turns.into(), [Some(vec![1.0, 0.0]), Some(vec![0.0, 1.0]), None]);
        let m = means(&turns);
        let s = m[0].as_ref().unwrap();
        assert!((s[0] - 0.9487).abs() < 1e-3 && (s[1] - 0.3162).abs() < 1e-3, "{s:?}");
        assert_eq!(m[1..], [None, None]);
    }

    /// A track: its name, its segments as (start, end, text), its turns.
    type Spec<'a> = (&'a str, Vec<(i64, i64, &'a str)>, Vec<Turn>);

    fn fixture(tracks: &[Spec]) -> Outputs {
        Outputs {
            tracks: tracks
                .iter()
                .map(|(track, segs, turns)| {
                    let segments = segs
                        .iter()
                        .map(|&(start_ms, end_ms, text)| Segment {
                            start_ms,
                            end_ms,
                            text: text.into(),
                        })
                        .collect();
                    (
                        track.to_string(),
                        Track {
                            segments,
                            turns: turns.clone(),
                        },
                    )
                })
                .collect(),
        }
    }

    const TUNING: Tuning = Tuning {
        merge_threshold: 0.75,
        echo_jaccard: 0.6,
    };

    #[test]
    fn assembles_labeled_lines_and_the_clusters_that_kept_any() {
        let e = |v: [f32; 2]| Some(v.to_vec());
        let room = with(
            vec![t(0, 12_000, 0), t(12_000, 24_000, 1), t(30_000, 42_000, 2)],
            [e([1.0, 0.0]), e([0.0, 1.0]), e([-1.0, 0.0])],
        );
        let o = fixture(&[
            (
                "room",
                vec![
                    (1_000, 23_000, "een twee drie vier vijf zes zeven acht negen tien elf"),
                    (31_000, 33_000, "precies wat hij zei"),
                ],
                room,
            ),
            ("remote", vec![(30_900, 33_100, "precies wat hij zei")], vec![]),
        ]);
        let a = assemble(&o, &TUNING);
        let got: Vec<(i64, &str, Option<&str>)> = a
            .lines
            .iter()
            .map(|l| (l.start_ms, l.text.as_str(), l.speaker.as_deref()))
            .collect();
        assert_eq!(
            got,
            [
                (1_000, "een twee drie vier vijf zes", Some("room/S1")),
                (13_000, "zeven acht negen tien elf", Some("room/S2")),
                // Undiarized track: unlabeled. Its room echo is dropped.
                (30_900, "precies wat hij zei", None),
            ]
        );
        // S3, alike to nothing, only had the echo: no lines left, so no cluster.
        let labels: Vec<&str> = a.clusters.iter().map(|(l, _)| l.as_str()).collect();
        assert_eq!(labels, ["room/S1", "room/S2"]);
    }

    #[test]
    fn derives_from_stored_outputs() {
        let db = crate::db::open(std::path::Path::new(":memory:")).unwrap();
        crate::db::ensure_recording(&db, "r1", "laptop").unwrap();
        db.execute_batch(
            "INSERT INTO windows (id, recording, file, track, offset_ms, start_ms, end_ms)
               VALUES (1, 'r1', '00-mic.oga', 'room', 0, 0, 30000);
             INSERT INTO segments (recording, window, track, start_ms, end_ms, text)
               VALUES ('r1', 1, 'room', 0, 8000, 'a b'), ('r1', 1, 'room', 9000, 9500, 'c');",
        )
        .unwrap();
        derive(&db, "r1").unwrap();
        let speakers = |db: &Connection| -> Vec<Option<String>> {
            lines(db, "r1").unwrap().into_iter().map(|l| l.speaker).collect()
        };
        assert_eq!(speakers(&db), [None, None], "before diarization");

        let turns = with(vec![t(0, 9_600, 0)], [Some(vec![0.6, 0.8])]);
        store_turns(&db, "r1", "room", &turns).unwrap();
        assert_eq!(super::outputs(&db, "r1").unwrap().tracks["room"].turns, turns);
        derive(&db, "r1").unwrap();
        assert_eq!(speakers(&db), [Some("room/S1".into()), Some("room/S1".into())]);
        let emb: Vec<u8> = db
            .query_row("SELECT embedding FROM clusters WHERE label = 'room/S1'", [], |r| {
                r.get(0)
            })
            .unwrap();
        let emb = floats(&emb);
        assert!((emb[0] - 0.6).abs() < 1e-6 && (emb[1] - 0.8).abs() < 1e-6, "{emb:?}");
        assert_eq!(
            db.query_row("SELECT COUNT(*) FROM segments", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            2,
            "outputs untouched"
        );
    }
}
