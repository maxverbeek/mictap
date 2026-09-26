//! Interpreting model outputs: folding sherpa's clusters into speakers, splitting whisper
//! segments into lines, and dropping echoes across tracks.
use std::collections::{HashMap, HashSet};

use rusqlite::Connection;

const NEAR_MS: i64 = 1_000;
const SLICE_MS: i64 = 250;
const DROP_CLUSTER: f64 = 0.7;

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Segment {
    pub track: String,
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
    pub speaker: Option<String>,
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
fn slice<'a>(r: &Segment, words: &'a [String], s: i64, e: i64) -> impl Iterator<Item = &'a str> {
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
fn is_echo(s: &Segment, remote: &[(&Segment, Vec<String>)], threshold: f64) -> bool {
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
pub fn drop_echoes(mut segs: Vec<Segment>, threshold: f64) -> Vec<Segment> {
    segs.sort_by_key(|s| (s.start_ms, s.end_ms));
    let remote: Vec<(&Segment, Vec<String>)> = segs
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

/// The recording's segments, merged with `MICTAP_ECHO_JACCARD` (default 0.6).
pub fn merged(conn: &Connection, id: &str) -> rusqlite::Result<Vec<Segment>> {
    let threshold = std::env::var("MICTAP_ECHO_JACCARD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.6);
    Ok(drop_echoes(load(conn, id)?, threshold))
}

fn load(conn: &Connection, id: &str) -> rusqlite::Result<Vec<Segment>> {
    conn.prepare("SELECT track, start_ms, end_ms, text, speaker FROM segments WHERE recording = ?1")?
        .query_map([id], |r| {
            Ok(Segment {
                track: r.get(0)?,
                start_ms: r.get(1)?,
                end_ms: r.get(2)?,
                text: r.get(3)?,
                speaker: r.get(4)?,
            })
        })?
        .collect()
}

#[derive(Debug, PartialEq)]
pub(crate) struct Turn {
    pub start_ms: i64,
    pub end_ms: i64,
    /// Index in order of first appearance: 0 is `S1`.
    pub speaker: usize,
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
pub(crate) fn split(turns: &[Turn], s: i64, e: i64, text: &str) -> Vec<(i64, i64, String, usize)> {
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
pub(crate) fn means(turns: &[Turn], embs: &[Option<Vec<f32>>]) -> Vec<Option<Vec<f32>>> {
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
pub(crate) fn cluster(turns: &mut [Turn], embs: &[Option<Vec<f32>>], threshold: f32) {
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
        }
    }

    fn s(track: &str, start_ms: i64, end_ms: i64, text: &str, speaker: &str) -> Segment {
        Segment {
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
        cluster(&mut turns, &embs, 0.9);
        let speakers: Vec<usize> = turns.iter().map(|t| t.speaker).collect();
        assert_eq!(speakers, [0, 0, 1, 1, 2, 0]);

        let mut two = [t(0, 20_000, 0), t(20_000, 40_000, 1)];
        cluster(&mut two, &[unit([1.0, 0.0, 0.0]), unit([0.6, 0.8, 0.0])], 0.7);
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
}
