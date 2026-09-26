//! Name storage: confirmed names, names suggested from known voices, and the voices learned
//! from confirmed names (see CONTEXT.md).
use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::Result;
use rusqlite::{params, Connection};
use serde::Serialize;

use crate::assemble::{cosine, floats};

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
/// A cluster without a clear match stays unknown, and no name is suggested twice within a
/// track or for a track where it is already confirmed; the most similar cluster gets it.
pub(crate) fn suggest(db: &Connection, id: &str, m: &Matching) -> Result<()> {
    let confirmed = confirmed(db, id)?;
    let clusters: Vec<(String, Vec<f32>)> = db
        .prepare("SELECT label, embedding FROM clusters WHERE recording = ?1")?
        .query_map([id], |r| Ok((r.get(0)?, floats(&r.get::<_, Vec<u8>>(1)?))))?
        .collect::<rusqlite::Result<_>>()?;
    let voices: Vec<(String, Vec<f32>)> = db
        .prepare("SELECT name, embedding FROM voices WHERE recording != ?1")?
        .query_map([id], |r| Ok((r.get(0)?, floats(&r.get::<_, Vec<u8>>(1)?))))?
        .collect::<rusqlite::Result<_>>()?;
    let mut candidates: Vec<(f32, &str, &str)> = clusters
        .iter()
        .filter(|(label, _)| !confirmed.contains_key(label))
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

/// Confirms `changes` (label -> name; `""` leaves the label unnamed, rejecting its
/// suggestion) and learns the voice of each newly confirmed name.
pub(crate) fn confirm(db: &Connection, id: &str, changes: Names) -> Result<()> {
    let before = confirmed(db, id)?;
    let mut names = before.clone();
    let mut suggested = column(db, id, "suggested")?;
    for (label, name) in changes {
        suggested.remove(&label);
        match name.trim() {
            "" => names.remove(&label),
            n => names.insert(label, n.to_string()),
        };
    }
    let tx = db.unchecked_transaction()?;
    tx.execute(
        "UPDATE recordings SET speakers = ?2, suggested = ?3 WHERE id = ?1",
        params![id, serde_json::to_string(&names)?, serde_json::to_string(&suggested)?],
    )?;
    for label in before.keys().filter(|l| names.get(*l) != before.get(*l)) {
        tx.execute(
            "DELETE FROM voices WHERE recording = ?1 AND label = ?2",
            params![id, label],
        )?;
    }
    for (label, name) in names.iter().filter(|(l, n)| before.get(*l) != Some(n)) {
        tx.execute(
            "INSERT OR REPLACE INTO voices (name, embedding, recording, label)
             SELECT ?3, embedding, recording, label FROM clusters WHERE recording = ?1 AND label = ?2",
            params![id, label, name],
        )?;
    }
    tx.commit()?;
    Ok(())
}

#[derive(Debug, PartialEq, Serialize)]
pub(crate) struct Speaker {
    pub name: Option<String>,
    pub suggested: Option<String>,
}

/// Each of `labels` with its confirmed name or else its suggestion.
pub(crate) fn speakers<'a>(
    db: &Connection,
    id: &str,
    labels: impl IntoIterator<Item = &'a str>,
) -> Result<BTreeMap<String, Speaker>> {
    let (names, suggested) = (confirmed(db, id)?, column(db, id, "suggested")?);
    Ok(labels
        .into_iter()
        .map(|l| {
            let name = names.get(l).cloned();
            let suggested = suggested.get(l).filter(|_| name.is_none()).cloned();
            (l.to_string(), Speaker { name, suggested })
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assemble::bytes;

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
        let changes: Names = [("room/S1", " Max "), ("room/S2", ""), ("room/S3", "")]
            .map(|(l, n)| (l.to_string(), n.to_string()))
            .into();
        confirm(&db, "r1", changes).unwrap();
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
        };
        assert_eq!(got["room/S1"], s(Some("Max"), None));
        assert_eq!(got["room/S2"], s(None, Some("Jan")));
        assert_eq!(got["room/S3"], s(None, None));
    }
}
