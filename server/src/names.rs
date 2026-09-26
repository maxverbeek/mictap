//! Name storage: confirmed names, names suggested from known voices, and the voices learned
//! from confirmed names (see CONTEXT.md).
use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::Result;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

use crate::assemble::{bytes, cosine, floats, mixed, normalize, Tuning};

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
fn best<'a>(emb: &[f32], voices: &'a [(String, Vec<f32>)], m: &Matching) -> Option<(&'a str, f32)> {
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

/// Suggests names for the unconfirmed clusters of `id` from the voices of other recordings.
/// A mixed cluster, or one without a clear match, stays unknown, and no name is suggested twice within a
/// track or for a track where it is already confirmed; the most similar cluster gets it.
pub(crate) fn suggest(db: &Connection, id: &str, m: &Matching) -> Result<()> {
    let confirmed = confirmed(db, id)?;
    let clusters: Vec<(String, Vec<f32>)> = db
        .prepare("SELECT label, COALESCE(core, embedding) FROM clusters WHERE recording = ?1")?
        .query_map([id], |r| Ok((r.get(0)?, floats(&r.get::<_, Vec<u8>>(1)?))))?
        .collect::<rusqlite::Result<_>>()?;
    let voices: Vec<(String, Vec<f32>)> = db
        .prepare("SELECT name, embedding FROM voices WHERE recording != ?1")?
        .query_map([id], |r| Ok((r.get(0)?, floats(&r.get::<_, Vec<u8>>(1)?))))?
        .collect::<rusqlite::Result<_>>()?;
    let mixed = mixed_labels(db, id)?;
    let mut candidates: Vec<(f32, &str, &str)> = clusters
        .iter()
        .filter(|(label, _)| !confirmed.contains_key(label) && !mixed.contains(label))
        .filter_map(|(label, emb)| best(emb, &voices, m).map(|(name, c)| (c, label.as_str(), name)))
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

/// Confirms `changes` (`""` leaves the label unnamed, rejecting its suggestion) and learns
/// voices for each label whose name changed or that was heard: one per correct heard
/// snippet, none if every heard snippet was wrong, else (or when the snippets' turns
/// expired) the cluster's core unless the cluster is mixed.
pub(crate) fn confirm(db: &Connection, id: &str, changes: BTreeMap<String, Naming>) -> Result<()> {
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
    for (label, name) in names.iter().filter(|(l, _)| relearn(l)) {
        let heard = heard(label);
        if !heard.is_empty() {
            let mut learned = false;
            for h in heard.iter().filter(|h| h.correct) {
                if let Some(v) = snippet(&tx, id, track(label), h.start_ms, h.end_ms)? {
                    tx.execute(
                        "INSERT INTO voices (name, embedding, recording, label, start_ms, end_ms)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                        params![name, bytes(&v), id, label, h.start_ms, h.end_ms],
                    )?;
                    learned = true;
                }
            }
            if learned || !heard.iter().any(|h| h.correct) {
                continue;
            }
        }
        if mixed.contains(label) {
            continue;
        }
        tx.execute(
            "INSERT INTO voices (name, embedding, recording, label)
             SELECT ?3, COALESCE(core, embedding), recording, label FROM clusters
             WHERE recording = ?1 AND label = ?2",
            params![id, label, name],
        )?;
    }
    tx.commit()?;
    Ok(())
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
}
