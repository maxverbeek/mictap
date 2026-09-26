//! Name storage: confirmed names, names suggested from known voices, and the voices learned
//! from confirmed names (see CONTEXT.md).
use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::Result;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

use crate::assemble::{bytes, cosine, floats, mixed, normalize, Line, Tuning};

/// Speaker label (`room/S1`) -> name.
pub(crate) type Names = BTreeMap<String, String>;

/// A speaker label: `room/S<n>` or `remote/S<n>`.
pub(crate) fn is_label(k: &str) -> bool {
    k.split_once('/').is_some_and(|(track, n)| {
        matches!(track, "room" | "remote")
            && n.strip_prefix('S')
                .is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
    })
}

fn track(label: &str) -> &str {
    label.split_once('/').map_or(label, |(t, _)| t)
}

/// The confirmed names of `id`: typed, or suggestions accepted.
pub(crate) fn confirmed(db: &Connection, id: &str) -> Result<Names> {
    column(db, id, "speakers")
}

/// The labels of `id` whose clusters sound like two speakers or a room (`assemble::mixed`).
fn mixed_labels(db: &Connection, id: &str) -> Result<HashSet<String>> {
    let tuning = Tuning::from_env();
    let rows: Vec<(String, Option<f32>, Option<f32>)> = db
        .prepare("SELECT label, halves_alike, minor_share FROM clusters WHERE recording = ?1")?
        .query_map([id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<rusqlite::Result<_>>()?;
    Ok(rows
        .into_iter()
        .filter(|(_, alike, share)| mixed(alike.zip(*share), &tuning))
        .map(|(label, _, _)| label)
        .collect())
}

fn column(db: &Connection, id: &str, col: &str) -> Result<Names> {
    let json: Option<String> = db.query_row(&format!("SELECT {col} FROM recordings WHERE id = ?1"), [id], |r| {
        r.get(0)
    })?;
    Ok(json.map(|s| serde_json::from_str(&s)).transpose()?.unwrap_or_default())
}

pub(crate) struct Matching {
    /// A voice at least this alike to a cluster can name it.
    pub threshold: f32,
    /// ... if no voice of another name is within this of it.
    pub margin: f32,
}

impl Matching {
    /// `MICTAP_MATCH_THRESHOLD` (default 0.75) and `MICTAP_MATCH_MARGIN` (default 0.05).
    pub fn from_env() -> Self {
        let var = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
        Self {
            threshold: var("MICTAP_MATCH_THRESHOLD").unwrap_or(0.75),
            margin: var("MICTAP_MATCH_MARGIN").unwrap_or(0.05),
        }
    }
}

/// The name whose most similar voice is at least `threshold` alike to `emb` and more than
/// `margin` ahead of every other name's, with that similarity.
fn best<'a>(
    emb: &[f32],
    voices: impl IntoIterator<Item = &'a (String, Vec<f32>)>,
    m: &Matching,
) -> Option<(&'a str, f32)> {
    let mut by_name: HashMap<&str, f32> = HashMap::new();
    for (name, v) in voices {
        let c = cosine(emb, v);
        if c.is_finite() {
            let e = by_name.entry(name).or_insert(c);
            *e = e.max(c);
        }
    }
    let mut ranked: Vec<(&str, f32)> = by_name.into_iter().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    let (name, first) = *ranked.first()?;
    let second = ranked.get(1).map_or(f32::NEG_INFINITY, |r| r.1);
    (first >= m.threshold && first - second > m.margin).then_some((name, first))
}

/// Suggests names for the unconfirmed clusters of `id` from the voices of other recordings, and
/// of its other clusters and named lines.
/// A mixed cluster, or one without a clear match, stays unknown, and no name is suggested twice within a
/// track or for a track where it is already confirmed; the most similar cluster gets it.
pub(crate) fn suggest(db: &Connection, id: &str, m: &Matching) -> Result<()> {
    let confirmed = confirmed(db, id)?;
    let clusters: Vec<(String, Vec<f32>)> = db
        .prepare("SELECT label, COALESCE(core, embedding) FROM clusters WHERE recording = ?1")?
        .query_map([id], |r| Ok((r.get(0)?, floats(&r.get::<_, Vec<u8>>(1)?))))?
        .collect::<rusqlite::Result<_>>()?;
    // Every voice but the ones this very cluster taught: voices from its own recording (other
    // clusters, named lines) count too, so naming one speaker helps guess the others.
    let voices: Vec<(String, String, (String, Vec<f32>))> = db
        .prepare("SELECT recording, label, name, embedding FROM voices")?
        .query_map([], |r| {
            Ok((r.get(0)?, r.get(1)?, (r.get(2)?, floats(&r.get::<_, Vec<u8>>(3)?))))
        })?
        .collect::<rusqlite::Result<_>>()?;
    let mixed = mixed_labels(db, id)?;
    let mut candidates: Vec<(f32, &str, &str)> = clusters
        .iter()
        .filter(|(label, _)| !confirmed.contains_key(label) && !mixed.contains(label))
        .filter_map(|(label, emb)| {
            let others = voices
                .iter()
                .filter(|(r, l, _)| !(r == id && l == label))
                .map(|(_, _, v)| v);
            best(emb, others, m).map(|(name, c)| (c, label.as_str(), name))
        })
        .collect();
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut taken: HashSet<(&str, &str)> = confirmed.iter().map(|(l, n)| (track(l), n.as_str())).collect();
    let suggested: Names = candidates
        .into_iter()
        .filter(|(_, label, name)| taken.insert((track(label), name)))
        .map(|(_, label, name)| (label.to_string(), name.to_string()))
        .collect();
    db.execute(
        "UPDATE recordings SET suggested = ?2 WHERE id = ?1",
        params![id, serde_json::to_string(&suggested)?],
    )?;
    Ok(())
}

/// Suggests anew for every done recording, since a voice learned in one recording can name
/// speakers in all the others.
pub(crate) fn suggest_all(db: &Connection, m: &Matching) -> Result<()> {
    let ids: Vec<String> = db
        .prepare("SELECT id FROM recordings WHERE status = 'done'")?
        .query_map([], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    for id in ids {
        suggest(db, &id, m)?;
    }
    Ok(())
}

/// What the page sends per label: the name, and the snippets of it that were heard.
#[derive(Deserialize)]
pub(crate) struct Naming {
    pub name: String,
    #[serde(default)]
    pub heard: Vec<Heard>,
}

/// A line of the label that was listened to; `correct` when it is only the named speaker.
#[derive(Deserialize)]
pub(crate) struct Heard {
    pub start_ms: i64,
    pub end_ms: i64,
    pub correct: bool,
}

/// Confirms `changes` (`""` leaves the label unnamed, rejecting its suggestion; `?` answers
/// several people or unsure: confirmed, but no name and no voice) and learns
/// voices for each label whose name changed or that was heard: one per correct heard
/// snippet, none if every heard snippet was wrong, else (or when the snippets' turns
/// expired) the cluster's core unless the cluster is mixed. Returns how many voices each
/// label of `changes` learned.
pub(crate) fn confirm(db: &Connection, id: &str, changes: BTreeMap<String, Naming>) -> Result<BTreeMap<String, usize>> {
    let before = confirmed(db, id)?;
    let mut names = before.clone();
    let mut suggested = column(db, id, "suggested")?;
    for (label, n) in &changes {
        suggested.remove(label);
        match n.name.trim() {
            "" => names.remove(label),
            name => names.insert(label.clone(), name.to_string()),
        };
    }
    let heard = |l: &str| changes.get(l).map_or(&[][..], |n| &n.heard[..]);
    let relearn = |l: &String| names.get(l) != before.get(l) || (names.contains_key(l) && !heard(l).is_empty());
    let mixed = mixed_labels(db, id)?;
    let tx = db.unchecked_transaction()?;
    tx.execute(
        "UPDATE recordings SET speakers = ?2, suggested = ?3 WHERE id = ?1",
        params![id, serde_json::to_string(&names)?, serde_json::to_string(&suggested)?],
    )?;
    let labels: HashSet<&String> = before.keys().chain(names.keys()).collect();
    for label in labels.into_iter().filter(|l| relearn(l)) {
        tx.execute(
            "DELETE FROM voices WHERE recording = ?1 AND label = ?2",
            params![id, label],
        )?;
    }
    let mut learned: BTreeMap<String, usize> = changes.keys().map(|l| (l.clone(), 0)).collect();
    for (label, name) in names.iter().filter(|(l, n)| relearn(l) && *n != "?") {
        let heard = heard(label);
        let n = learned.entry(label.clone()).or_default();
        if !heard.is_empty() {
            for h in heard.iter().filter(|h| h.correct) {
                if let Some(v) = snippet(&tx, id, track(label), h.start_ms, h.end_ms)? {
                    *n += tx.execute(
                        "INSERT INTO voices (name, embedding, recording, label, start_ms, end_ms)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                        params![name, bytes(&v), id, label, h.start_ms, h.end_ms],
                    )?;
                }
            }
            if *n > 0 || !heard.iter().any(|h| h.correct) {
                continue;
            }
        }
        if mixed.contains(label) {
            continue;
        }
        *n += tx.execute(
            "INSERT INTO voices (name, embedding, recording, label)
             SELECT ?3, COALESCE(core, embedding), recording, label FROM clusters
             WHERE recording = ?1 AND label = ?2",
            params![id, label, name],
        )?;
    }
    tx.commit()?;
    learned.retain(|l, _| changes.contains_key(l));
    Ok(learned)
}

/// The normalized mean of the embeddings of `id`'s `track` turns overlapping
/// `[start_ms, end_ms)`, weighted by overlap; None once the turns expired.
fn snippet(db: &Connection, id: &str, track: &str, start_ms: i64, end_ms: i64) -> Result<Option<Vec<f32>>> {
    let turns: Vec<(i64, i64, Vec<u8>)> = db
        .prepare(
            "SELECT start_ms, end_ms, embedding FROM turns WHERE recording = ?1 AND track = ?2
             AND embedding IS NOT NULL AND start_ms < ?4 AND end_ms > ?3",
        )?
        .query_map(params![id, track, start_ms, end_ms], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })?
        .collect::<rusqlite::Result<_>>()?;
    let mut sum: Option<Vec<f32>> = None;
    for (s, e, emb) in turns {
        let w = (e.min(end_ms) - s.max(start_ms)) as f32;
        let v = floats(&emb);
        match &mut sum {
            Some(sum) => sum.iter_mut().zip(&v).for_each(|(a, b)| *a += w * b),
            slot => *slot = Some(v.iter().map(|b| w * b).collect()),
        }
    }
    Ok(sum.map(|mut v| {
        normalize(&mut v);
        v
    }))
}

/// Sets the line name of `id`'s line on `track` spanning `[start_ms, end_ms)`, replacing
/// any line name it overlaps (either holds the other's midpoint) and its voice; None or `""`
/// clears it. A name other than `?` learns a voice from the turns under the line, stored
/// under the track as label; returns how many (none once the turns expired).
pub(crate) fn name_line(
    db: &Connection,
    id: &str,
    track: &str,
    start_ms: i64,
    end_ms: i64,
    name: Option<&str>,
) -> Result<usize> {
    let tx = db.unchecked_transaction()?;
    let replaced: Vec<(i64, i64)> = tx
        .prepare(
            "DELETE FROM line_names WHERE recording = ?1 AND track = ?2
             AND ((start_ms + end_ms) / 2 >= ?3 AND (start_ms + end_ms) / 2 < ?4
                  OR (?3 + ?4) / 2 >= start_ms AND (?3 + ?4) / 2 < end_ms)
             RETURNING start_ms, end_ms",
        )?
        .query_map(params![id, track, start_ms, end_ms], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    for (s, e) in replaced {
        tx.execute(
            "DELETE FROM voices WHERE recording = ?1 AND label = ?2 AND start_ms = ?3 AND end_ms = ?4",
            params![id, track, s, e],
        )?;
    }
    let mut learned = 0;
    if let Some(name) = name.map(str::trim).filter(|n| !n.is_empty()) {
        tx.execute(
            "INSERT INTO line_names (recording, track, start_ms, end_ms, name) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, track, start_ms, end_ms, name],
        )?;
        if let Some(v) = snippet(&tx, id, track, start_ms, end_ms)?.filter(|_| name != "?") {
            learned = tx.execute(
                "INSERT INTO voices (name, embedding, recording, label, start_ms, end_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![name, bytes(&v), id, track, start_ms, end_ms],
            )?;
        }
    }
    tx.commit()?;
    Ok(learned)
}

/// Whether `id`'s turns are kept, so naming can still learn voices from its lines.
pub(crate) fn teachable(db: &Connection, id: &str) -> Result<bool> {
    Ok(db.query_row(
        "SELECT EXISTS(SELECT 1 FROM turns WHERE recording = ?1 AND embedding IS NOT NULL)",
        [id],
        |r| r.get(0),
    )?)
}

/// A line with the name it shows: its line name, else its label's confirmed name.
#[derive(Debug, Serialize)]
pub(crate) struct Named {
    #[serde(flatten)]
    pub line: Line,
    pub name: Option<String>,
    pub line_name: Option<String>,
    /// It teaches a voice: named on its own (not `?`), or heard before its label was confirmed.
    pub taught: bool,
}

/// The derived lines of `id` with their names (see `Named`).
pub(crate) fn named(db: &Connection, id: &str) -> Result<Vec<Named>> {
    let names = confirmed(db, id)?;
    let line_names: Vec<(String, i64, i64, String)> = db
        .prepare("SELECT track, start_ms, end_ms, name FROM line_names WHERE recording = ?1")?
        .query_map([id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
        .collect::<rusqlite::Result<_>>()?;
    let heard: Vec<(String, i64, i64)> = db
        .prepare("SELECT label, start_ms, end_ms FROM voices WHERE recording = ?1 AND start_ms IS NOT NULL")?
        .query_map([id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<rusqlite::Result<_>>()?;
    Ok(crate::assemble::lines(db, id)?
        .into_iter()
        .map(|line| {
            let mid = (line.start_ms + line.end_ms) / 2;
            let holds = |t: &str, s: i64, e: i64| t == line.track && s <= mid && mid < e;
            let line_name = line_names
                .iter()
                .find(|(t, s, e, _)| holds(t, *s, *e))
                .map(|n| n.3.clone());
            let taught = match &line_name {
                Some(n) => n != "?",
                None => heard.iter().any(|(l, s, e)| holds(track(l), *s, *e)),
            };
            let name = line_name
                .clone()
                .or_else(|| line.speaker.as_ref().and_then(|l| names.get(l)).cloned());
            Named {
                line,
                name,
                line_name,
                taught,
            }
        })
        .collect())
}

/// Who attended: the confirmed names in label order, then the line names in order, once each.
pub(crate) fn attendees(names: &Names, lines: &[Named]) -> Vec<String> {
    let mut labels: Vec<&String> = names.keys().collect();
    labels.sort_by_key(|l| crate::vault::label_order(l));
    let mut out: Vec<String> = vec![];
    for n in labels
        .into_iter()
        .map(|l| &names[l])
        .chain(lines.iter().filter_map(|l| l.line_name.as_ref()))
    {
        if n != "?" && !out.contains(n) {
            out.push(n.clone());
        }
    }
    out
}

#[derive(Debug, PartialEq, Serialize)]
pub(crate) struct Speaker {
    pub name: Option<String>,
    pub suggested: Option<String>,
    /// Its cluster sounds like two speakers or a room: a name is kept, no voice learned.
    pub mixed: bool,
}

/// Each of `labels` with its confirmed name or else its suggestion.
pub(crate) fn speakers<'a>(
    db: &Connection,
    id: &str,
    labels: impl IntoIterator<Item = &'a str>,
) -> Result<BTreeMap<String, Speaker>> {
    let (names, suggested) = (confirmed(db, id)?, column(db, id, "suggested")?);
    let mixed = mixed_labels(db, id)?;
    Ok(labels
        .into_iter()
        .map(|l| {
            let name = names.get(l).cloned();
            let suggested = suggested.get(l).filter(|_| name.is_none()).cloned();
            let speaker = Speaker {
                name,
                suggested,
                mixed: mixed.contains(l),
            };
            (l.to_string(), speaker)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> Connection {
        let db = crate::db::open(std::path::Path::new(":memory:")).unwrap();
        for id in ["old", "r1"] {
            crate::db::ensure_recording(&db, id, "laptop").unwrap();
        }
        db
    }

    fn voice(db: &Connection, name: &str, label: &str, v: &[f32]) {
        db.execute(
            "INSERT INTO voices (name, embedding, recording, label) VALUES (?1, ?2, 'old', ?3)",
            params![name, bytes(v), label],
        )
        .unwrap();
    }

    fn cluster(db: &Connection, label: &str, v: &[f32]) {
        db.execute(
            "INSERT INTO clusters (recording, label, embedding) VALUES ('r1', ?1, ?2)",
            params![label, bytes(v)],
        )
        .unwrap();
    }

    /// Names without heard snippets, as the page sends a typed name.
    fn named(pairs: &[(&str, &str)]) -> BTreeMap<String, Naming> {
        pairs
            .iter()
            .map(|(l, n)| {
                let naming = Naming {
                    name: n.to_string(),
                    heard: vec![],
                };
                (l.to_string(), naming)
            })
            .collect()
    }

    fn heard(name: &str, snippets: &[(i64, i64, bool)]) -> BTreeMap<String, Naming> {
        let heard = snippets
            .iter()
            .map(|&(start_ms, end_ms, correct)| Heard {
                start_ms,
                end_ms,
                correct,
            })
            .collect();
        let naming = Naming {
            name: name.into(),
            heard,
        };
        [("room/S1".to_string(), naming)].into()
    }

    /// (start_ms, end_ms, embedding) of r1's voices.
    fn learned(db: &Connection) -> Vec<(Option<i64>, Option<i64>, Vec<f32>)> {
        db.prepare("SELECT start_ms, end_ms, embedding FROM voices WHERE recording = 'r1' ORDER BY id")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, floats(&r.get::<_, Vec<u8>>(2)?))))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap()
    }

    fn get(db: &Connection, col: &str) -> Names {
        column(db, "r1", col).unwrap()
    }

    const M: Matching = Matching {
        threshold: 0.75,
        margin: 0.05,
    };

    #[test]
    fn best_needs_the_threshold_and_a_margin_over_other_names() {
        let voices = [
            ("Max".to_string(), vec![1.0, 0.0, 0.0]),
            ("Max".to_string(), vec![0.8, 0.6, 0.0]),
            ("Eva".to_string(), vec![0.0, 1.0, 0.0]),
        ];
        // Max's best voice counts, not his average; another Max voice is no competition.
        assert_eq!(best(&[0.9, 0.1, 0.0], &voices, &M).map(|b| b.0), Some("Max"));
        // Max's second voice 0.96 against Eva's 0.8: clear.
        assert_eq!(best(&[0.6, 0.8, 0.0], &voices, &M).map(|b| b.0), Some("Max"));
        // Halfway between Max's second voice and Eva's, both 0.89: unknown.
        assert_eq!(best(&[0.8, 1.6, 0.0], &voices[1..], &M), None);
        assert_eq!(best(&[0.0, 0.0, 1.0], &voices, &M), None, "alike to no one");
        assert_eq!(best(&[1.0, 0.0], &voices, &M), None, "other model's dimension");
        assert_eq!(best(&[1.0, 0.0, 0.0], &[], &M), None);
    }

    #[test]
    fn suggests_from_the_same_recording_but_not_from_the_cluster_itself() {
        let db = db();
        cluster(&db, "room/S1", &[1.0, 0.0, 0.0]);
        cluster(&db, "room/S2", &[0.0, 1.0, 0.0]);
        for (name, label, v) in [("Max", "room/S1", [0.0, 1.0, 0.0]), ("Eva", "room/S2", [1.0, 0.0, 0.0])] {
            db.execute(
                "INSERT INTO voices (name, embedding, recording, label) VALUES (?1, ?2, 'r1', ?3)",
                params![name, bytes(&v), label],
            )
            .unwrap();
        }
        suggest(&db, "r1", &M).unwrap();
        let got = speakers(&db, "r1", ["room/S1", "room/S2"]).unwrap();
        assert_eq!(got["room/S1"].suggested.as_deref(), Some("Eva"));
        assert_eq!(got["room/S2"].suggested.as_deref(), Some("Max"));
    }

    #[test]
    fn suggests_in_every_done_recording_again() {
        let db = db();
        db.execute("UPDATE recordings SET status = 'done' WHERE id = 'r1'", [])
            .unwrap();
        cluster(&db, "room/S1", &[1.0, 0.0, 0.0]);
        suggest_all(&db, &M).unwrap();
        assert!(speakers(&db, "r1", ["room/S1"]).unwrap()["room/S1"].suggested.is_none());
        voice(&db, "Max", "room/S1", &[1.0, 0.0, 0.0]);
        suggest_all(&db, &M).unwrap();
        assert_eq!(
            speakers(&db, "r1", ["room/S1"]).unwrap()["room/S1"]
                .suggested
                .as_deref(),
            Some("Max")
        );
    }

    #[test]
    fn suggests_each_name_once_per_track_and_not_over_confirmed_ones() {
        let db = db();
        voice(&db, "Max", "room/S1", &[1.0, 0.0, 0.0]);
        voice(&db, "Eva", "room/S2", &[0.0, 1.0, 0.0]);
        voice(&db, "Jan", "remote/S1", &[0.0, 0.0, 1.0]);
        cluster(&db, "room/S1", &[0.9, 0.1, 0.0]);
        cluster(&db, "room/S2", &[0.99, 0.0, 0.1]); // Max too, more alike: gets it
        cluster(&db, "room/S3", &[0.0, 0.2, 1.0]); // Jan: confirmed on the remote track only
        cluster(&db, "remote/S1", &[1.0, 0.05, 0.0]); // Max on another track: fine
        cluster(&db, "remote/S2", &[0.0, 0.0, 1.0]); // Jan, confirmed for remote/S3
        cluster(&db, "remote/S4", &[0.0, 1.0, 0.0]); // Eva, but mixed
        db.execute(
            "UPDATE clusters SET halves_alike = 0.2, minor_share = 0.5 WHERE label = 'remote/S4'",
            [],
        )
        .unwrap();
        db.execute(
            r#"UPDATE recordings SET speakers = '{"remote/S3":"Jan","room/S4":"Bo"}' WHERE id = 'r1'"#,
            [],
        )
        .unwrap();
        suggest(&db, "r1", &M).unwrap();
        let want: Names = [("remote/S1", "Max"), ("room/S2", "Max"), ("room/S3", "Jan")]
            .map(|(l, n)| (l.to_string(), n.to_string()))
            .into();
        assert_eq!(get(&db, "suggested"), want);
    }

    #[test]
    fn confirming_learns_voices_and_clearing_rejects_suggestions() {
        let db = db();
        cluster(&db, "room/S1", &[1.0, 0.0]);
        cluster(&db, "room/S2", &[0.0, 1.0]);
        db.execute(
            r#"UPDATE recordings SET speakers = '{"room/S3":"Bo"}', suggested = '{"room/S1":"Max","room/S2":"Eva"}' WHERE id = 'r1'"#,
            [],
        )
        .unwrap();
        voice(&db, "Bo", "room/S3", &[0.5, 0.5]);
        db.execute("UPDATE voices SET recording = 'r1'", []).unwrap();
        confirm(
            &db,
            "r1",
            named(&[("room/S1", " Max "), ("room/S2", ""), ("room/S3", "")]),
        )
        .unwrap();
        assert_eq!(
            get(&db, "speakers"),
            [("room/S1".to_string(), "Max".to_string())].into()
        );
        assert_eq!(get(&db, "suggested"), Names::new());
        let voices: Vec<(String, String)> = db
            .prepare("SELECT name, label FROM voices ORDER BY label")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(voices, [("Max".to_string(), "room/S1".to_string())]);
    }

    #[test]
    fn a_mixed_cluster_keeps_its_name_but_teaches_no_voice() {
        let db = db();
        cluster(&db, "room/S1", &[1.0, 0.0]);
        cluster(&db, "room/S2", &[0.0, 1.0]);
        db.execute_batch(
            "UPDATE clusters SET halves_alike = 0.3, minor_share = 0.4 WHERE label = 'room/S1';
             UPDATE clusters SET halves_alike = 0.9, minor_share = 0.4 WHERE label = 'room/S2';",
        )
        .unwrap();
        confirm(&db, "r1", named(&[("room/S1", "Room B"), ("room/S2", "Eva")])).unwrap();
        assert_eq!(get(&db, "speakers").len(), 2);
        let learned: String = db
            .query_row("SELECT group_concat(name) FROM voices", [], |r| r.get(0))
            .unwrap();
        assert_eq!(learned, "Eva");
        let got = speakers(&db, "r1", ["room/S1", "room/S2"]).unwrap();
        assert!(got["room/S1"].mixed && !got["room/S2"].mixed);
    }

    #[test]
    fn matches_and_learns_the_core() {
        let db = db();
        voice(&db, "Max", "room/S1", &[1.0, 0.0]);
        cluster(&db, "room/S1", &[0.0, 1.0]);
        db.execute(
            "UPDATE clusters SET core = ?1 WHERE recording = 'r1'",
            [bytes(&[0.99, 0.1])],
        )
        .unwrap();
        suggest(&db, "r1", &M).unwrap();
        assert_eq!(get(&db, "suggested")["room/S1"], "Max");
        confirm(&db, "r1", named(&[("room/S1", "Max")])).unwrap();
        let learned: Vec<u8> = db
            .query_row("SELECT embedding FROM voices WHERE recording = 'r1'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(floats(&learned), [0.99, 0.1]);
    }

    #[test]
    fn a_confirmed_name_hides_the_suggestion() {
        let db = db();
        db.execute(
            r#"UPDATE recordings SET speakers = '{"room/S1":"Max"}', suggested = '{"room/S1":"Eva","room/S2":"Jan"}' WHERE id = 'r1'"#,
            [],
        )
        .unwrap();
        let got = speakers(&db, "r1", ["room/S1", "room/S2", "room/S3"]).unwrap();
        let s = |name: Option<&str>, suggested: Option<&str>| Speaker {
            name: name.map(Into::into),
            suggested: suggested.map(Into::into),
            mixed: false,
        };
        assert_eq!(got["room/S1"], s(Some("Max"), None));
        assert_eq!(got["room/S2"], s(None, Some("Jan")));
        assert_eq!(got["room/S3"], s(None, None));
    }

    fn turns(db: &Connection, turns: &[(i64, i64, &[f32])]) {
        for (s, e, v) in turns {
            db.execute(
                "INSERT INTO turns (recording, track, start_ms, end_ms, speaker, embedding)
                 VALUES ('r1', 'room', ?1, ?2, 0, ?3)",
                params![s, e, bytes(v)],
            )
            .unwrap();
        }
    }

    #[test]
    fn learns_one_voice_per_correct_heard_snippet() {
        let db = db();
        cluster(&db, "room/S1", &[0.6, 0.8]);
        turns(&db, &[(0, 4_000, &[1.0, 0.0]), (4_000, 8_000, &[0.0, 1.0])]);
        let snippets = [(0, 4_000, true), (4_000, 8_000, false), (2_000, 6_000, true)];
        confirm(&db, "r1", heard("Max", &snippets)).unwrap();
        let h = std::f32::consts::FRAC_1_SQRT_2;
        let got = learned(&db);
        assert_eq!(got.len(), 2, "{got:?}");
        assert_eq!(got[0], (Some(0), Some(4_000), vec![1.0, 0.0]));
        assert_eq!((got[1].0, got[1].1), (Some(2_000), Some(6_000)));
        assert!(
            (got[1].2[0] - h).abs() < 1e-6 && (got[1].2[1] - h).abs() < 1e-6,
            "{got:?}"
        );

        // Heard again, all wrong: the name stays, its voices go.
        confirm(&db, "r1", heard("Max", &[(0, 4_000, false)])).unwrap();
        assert_eq!(get(&db, "speakers")["room/S1"], "Max");
        assert_eq!(learned(&db), []);
    }

    #[test]
    fn heard_snippets_whose_turns_expired_teach_the_core() {
        let db = db();
        cluster(&db, "room/S1", &[0.6, 0.8]);
        confirm(&db, "r1", heard("Max", &[(0, 4_000, true)])).unwrap();
        assert_eq!(learned(&db), [(None, None, vec![0.6, 0.8])]);
        // ... unless mixed.
        db.execute("UPDATE clusters SET halves_alike = 0.2, minor_share = 0.5", [])
            .unwrap();
        confirm(&db, "r1", heard("Max", &[(0, 4_000, true)])).unwrap();
        assert_eq!(learned(&db), []);
    }

    #[test]
    fn hearing_a_confirmed_name_again_relearns_it() {
        let db = db();
        cluster(&db, "room/S1", &[0.6, 0.8]);
        turns(&db, &[(0, 4_000, &[1.0, 0.0])]);
        confirm(&db, "r1", named(&[("room/S1", "Max")])).unwrap();
        assert_eq!(learned(&db), [(None, None, vec![0.6, 0.8])]);
        confirm(&db, "r1", heard("Max", &[(0, 4_000, true)])).unwrap();
        assert_eq!(learned(&db), [(Some(0), Some(4_000), vec![1.0, 0.0])]);
        // The same name again without listening: nothing changes.
        confirm(&db, "r1", named(&[("room/S1", "Max")])).unwrap();
        assert_eq!(learned(&db).len(), 1);
    }

    fn lines(db: &Connection, lines: &[(i64, i64, &str)]) {
        db.execute("DELETE FROM lines", []).unwrap();
        for (s, e, label) in lines {
            db.execute(
                "INSERT INTO lines (recording, track, start_ms, end_ms, text, speaker) VALUES ('r1', 'room', ?1, ?2, '', ?3)",
                params![s, e, label],
            )
            .unwrap();
        }
    }

    fn shown(db: &Connection) -> Vec<(Option<String>, bool)> {
        super::named(db, "r1")
            .unwrap()
            .into_iter()
            .map(|l| (l.name, l.taught))
            .collect()
    }

    #[test]
    fn line_names_follow_lines_derived_anew() {
        let db = db();
        db.execute(
            r#"UPDATE recordings SET speakers = '{"room/S1":"Max"}' WHERE id = 'r1'"#,
            [],
        )
        .unwrap();
        lines(
            &db,
            &[
                (0, 2_000, "room/S1"),
                (2_000, 5_000, "room/S1"),
                (5_000, 6_000, "room/S2"),
            ],
        );
        name_line(&db, "r1", "room", 2_000, 5_000, Some(" Eva ")).unwrap();
        name_line(&db, "r1", "room", 5_000, 6_000, Some("?")).unwrap();
        name_line(&db, "r1", "remote", 0, 2_000, Some("Jan")).unwrap();
        let s = |n: Option<&str>, t| (n.map(String::from), t);
        assert_eq!(
            shown(&db),
            [s(Some("Max"), false), s(Some("Eva"), true), s(Some("?"), false)]
        );
        // Derived anew: cut differently, the names go by midpoint.
        lines(
            &db,
            &[
                (0, 2_500, "room/S1"),
                (2_500, 4_800, "room/S2"),
                (4_800, 6_200, "room/S2"),
            ],
        );
        assert_eq!(
            shown(&db),
            [s(Some("Max"), false), s(Some("Eva"), true), s(Some("?"), false)]
        );
        // Naming the new line replaces the name it overlaps; clearing leaves the label's.
        name_line(&db, "r1", "room", 2_500, 4_800, Some("Bo")).unwrap();
        name_line(&db, "r1", "room", 4_800, 6_200, None).unwrap();
        assert_eq!(shown(&db), [s(Some("Max"), false), s(Some("Bo"), true), s(None, false)]);
        let n: i64 = db
            .query_row("SELECT COUNT(*) FROM line_names", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 2, "Bo, and Jan on the remote track");
        let got = attendees(&confirmed(&db, "r1").unwrap(), &super::named(&db, "r1").unwrap());
        assert_eq!(got, ["Max", "Bo"]);
    }

    #[test]
    fn a_line_name_teaches_one_voice_replaced_with_it() {
        let db = db();
        turns(&db, &[(0, 4_000, &[1.0, 0.0]), (4_000, 8_000, &[0.0, 1.0])]);
        assert!(teachable(&db, "r1").unwrap());
        let voices = |db: &Connection| -> Vec<(String, String, i64)> {
            db.prepare("SELECT name, label, start_ms FROM voices ORDER BY id")
                .unwrap()
                .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
                .unwrap()
                .collect::<rusqlite::Result<_>>()
                .unwrap()
        };
        assert_eq!(name_line(&db, "r1", "room", 0, 4_000, Some("Eva")).unwrap(), 1);
        assert_eq!(name_line(&db, "r1", "room", 4_000, 8_000, Some("Jan")).unwrap(), 1);
        assert_eq!(learned(&db)[0].2, [1.0, 0.0]);
        // Renamed: its voice goes with the old name.
        assert_eq!(name_line(&db, "r1", "room", 0, 4_000, Some("Bo")).unwrap(), 1);
        let v = |n: &str, s| (n.to_string(), "room".to_string(), s);
        assert_eq!(voices(&db), [v("Jan", 4_000), v("Bo", 0)]);
        // Unsure, or cleared: no voice.
        assert_eq!(name_line(&db, "r1", "room", 0, 4_000, Some("?")).unwrap(), 0);
        assert_eq!(name_line(&db, "r1", "room", 4_000, 8_000, Some("")).unwrap(), 0);
        assert_eq!(voices(&db), []);
        // A label confirmed later leaves line voices alone.
        name_line(&db, "r1", "room", 4_000, 8_000, Some("Jan")).unwrap();
        cluster(&db, "room/S1", &[0.6, 0.8]);
        confirm(&db, "r1", named(&[("room/S1", "Max")])).unwrap();
        confirm(&db, "r1", named(&[("room/S1", "")])).unwrap();
        assert_eq!(voices(&db), [v("Jan", 4_000)]);
        // Once the turns expired the name is still kept.
        db.execute("DELETE FROM turns", []).unwrap();
        assert!(!teachable(&db, "r1").unwrap());
        assert_eq!(name_line(&db, "r1", "room", 0, 4_000, Some("Eva")).unwrap(), 0);
        let n: i64 = db
            .query_row("SELECT COUNT(*) FROM line_names", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 2);
    }

    #[test]
    fn a_label_answered_unsure_is_confirmed_without_a_name_or_voice() {
        let db = db();
        voice(&db, "Max", "room/S1", &[1.0, 0.0]);
        cluster(&db, "room/S1", &[1.0, 0.0]);
        turns(&db, &[(0, 4_000, &[1.0, 0.0])]);
        lines(&db, &[(0, 4_000, "room/S1")]);
        suggest(&db, "r1", &M).unwrap();
        assert_eq!(get(&db, "suggested")["room/S1"], "Max");
        let learned = confirm(&db, "r1", heard("?", &[(0, 4_000, true)])).unwrap();
        assert_eq!(learned["room/S1"], 0);
        assert_eq!(learned_count(&db), 0);
        suggest(&db, "r1", &M).unwrap();
        assert_eq!(get(&db, "suggested"), Names::new(), "not suggested again");
        assert_eq!(
            speakers(&db, "r1", ["room/S1"]).unwrap()["room/S1"].name.as_deref(),
            Some("?")
        );
        let lines = super::named(&db, "r1").unwrap();
        assert_eq!(lines[0].name.as_deref(), Some("?"));
        assert!(attendees(&confirmed(&db, "r1").unwrap(), &lines).is_empty());
    }

    fn learned_count(db: &Connection) -> i64 {
        db.query_row("SELECT COUNT(*) FROM voices WHERE recording = 'r1'", [], |r| r.get(0))
            .unwrap()
    }
}
