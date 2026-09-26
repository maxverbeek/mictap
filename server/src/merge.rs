use std::collections::{HashMap, HashSet};

use rusqlite::Connection;

const NEAR_MS: i64 = 1_000;
const SLICE_MS: i64 = 250;
const DROP_CLUSTER: f64 = 0.7;

#[derive(Debug, Clone, PartialEq)]
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
pub fn merge(mut segs: Vec<Segment>, threshold: f64) -> Vec<Segment> {
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
    Ok(merge(load(conn, id)?, threshold))
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

#[cfg(test)]
mod tests {
    use super::*;

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
        let out = merge(segs.clone(), 0.6);
        assert_eq!(
            out,
            vec![segs[4].clone(), segs[1].clone(), segs[3].clone(), segs[2].clone()]
        );
        // The threshold is honored.
        assert_eq!(merge(segs, 0.9).len(), 5);
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
        assert_eq!(merge(segs, 0.6).len(), 1);
        // Two remote segments, echoed as one.
        let mut segs: Vec<_> = halves.map(|(a, b, t)| s("remote", a, b, t, "")).into();
        segs.push(s("room", 26_540, 34_430, long, ""));
        assert_eq!(merge(segs, 0.6).len(), 2);
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
        let out = merge(segs.clone(), 0.6);
        assert!(!out.contains(&leftover));
        assert_eq!(out.last(), Some(&own));
        assert_eq!(out.len(), 4);
        // 3 of 5 (60%): the non-echo segments stay.
        segs.push(s("room", 25_000, 26_000, "nog iets", "room/S2"));
        let out = merge(segs, 0.6);
        assert!(out.contains(&leftover));
        assert_eq!(out.len(), 6);
    }
}
