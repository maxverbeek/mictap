//! Name storage: confirmed names, names suggested from known voices, and the voices learned
//! from confirmed names (see CONTEXT.md).
use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::Result;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

use crate::assemble::{bytes, cosine, floats, mixed, normalize, Line, Tuning, Turn};

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
        Self::env("MICTAP_MATCH", 0.75, 0.05)
    }

    /// For a line's own voice: `MICTAP_LINE_THRESHOLD` (default 0.55) and `MICTAP_LINE_MARGIN`
    /// (default 0.1).
    pub fn lines_from_env() -> Self {
        Self::env("MICTAP_LINE", 0.55, 0.1)
    }

    /// Between the names of a cluster naming split, which share its channel: no threshold, and
    /// `MICTAP_SPLIT_MARGIN` (default 0.02).
    pub fn splits_from_env() -> Self {
        Self::env("MICTAP_SPLIT", -1.0, 0.02)
    }

    fn env(prefix: &str, threshold: f32, margin: f32) -> Self {
        let var = |k: &str| {
            std::env::var(format!("{prefix}_{k}"))
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
        };
        Self {
            threshold: var("THRESHOLD").unwrap_or(threshold),
            margin: var("MARGIN").unwrap_or(margin),
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

/// One voice per recording and name: the mean of the ones taught there, so a name taught on many
/// lines of a meeting gets one chance to match rather than one per line, and a wrong one among
/// them is outweighed.
fn pool<'a>(voices: impl IntoIterator<Item = (&'a str, &'a (String, Vec<f32>))>) -> Vec<(String, Vec<f32>)> {
    let mut sums: HashMap<(&str, &str, usize), Vec<f32>> = HashMap::new();
    for (recording, (name, v)) in voices {
        let mut v = v.clone();
        normalize(&mut v);
        add(sums.entry((recording, name, v.len())).or_default(), &v);
    }
    sums.into_iter()
        .map(|((_, name, _), mut sum)| {
            normalize(&mut sum);
            (name.to_string(), sum)
        })
        .collect()
}

fn add(sum: &mut Vec<f32>, v: &[f32]) {
    if sum.is_empty() {
        sum.extend_from_slice(v);
    } else {
        sum.iter_mut().zip(v).for_each(|(a, b)| *a += b);
    }
}

/// Suggests names for the unconfirmed clusters of `id` from the voices of other recordings, and
/// of its other clusters and named lines, `pool`ed.
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
    let mut candidates: Vec<(f32, &str, String)> = clusters
        .iter()
        .filter(|(label, _)| !confirmed.contains_key(label) && !mixed.contains(label))
        .filter_map(|(label, emb)| {
            let others = pool(
                voices
                    .iter()
                    .filter(|(r, l, _)| !(r == id && l == label))
                    .map(|(r, _, v)| (r.as_str(), v)),
            );
            best(emb, &others, m).map(|(name, c)| (c, label.as_str(), name.to_string()))
        })
        .collect();
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut taken: HashSet<(&str, String)> = confirmed.iter().map(|(l, n)| (track(l), n.clone())).collect();
    let suggested: Names = candidates
        .into_iter()
        .filter(|(_, label, name)| taken.insert((track(label), name.clone())))
        .map(|(_, label, name)| (label.to_string(), name))
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
                if let Some(v) = line_voice(&tx, id, track(label), h.start_ms, h.end_ms)? {
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
             SELECT ?3, COALESCE(?4, core, embedding), recording, label FROM clusters
             WHERE recording = ?1 AND label = ?2",
            params![id, label, name, label_core(&tx, id, label, name)?.as_deref().map(bytes)],
        )?;
    }
    tx.commit()?;
    learned.retain(|l, _| changes.contains_key(l));
    Ok(learned)
}

/// The line voice of `id`'s line on `track` spanning `[start_ms, end_ms)`, found by its
/// midpoint, else the turns' `snippet`.
fn line_voice(db: &Connection, id: &str, track: &str, start_ms: i64, end_ms: i64) -> Result<Option<Vec<f32>>> {
    let own: Option<Vec<u8>> = db
        .query_row(
            "SELECT embedding FROM line_voices WHERE recording = ?1 AND track = ?2
             AND start_ms <= ?3 AND ?3 < end_ms",
            params![id, track, (start_ms + end_ms) / 2],
            |r| r.get(0),
        )
        .optional()?;
    match own {
        Some(b) => Ok(Some(floats(&b))),
        None => snippet(db, id, track, start_ms, end_ms),
    }
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
/// any line name it overlaps (either holds the other's midpoint) and the voice any line or heard
/// snippet there taught; None or `""`
/// clears it. A name other than `?` learns a voice (see `line_voice`), stored under the track
/// as label; returns how many (none without a line voice once the turns expired).
pub(crate) fn name_line(
    db: &Connection,
    id: &str,
    track: &str,
    start_ms: i64,
    end_ms: i64,
    name: Option<&str>,
) -> Result<usize> {
    const OVERLAPS: &str = "((start_ms + end_ms) / 2 >= ?3 AND (start_ms + end_ms) / 2 < ?4
        OR (?3 + ?4) / 2 >= start_ms AND (?3 + ?4) / 2 < end_ms)";
    let tx = db.unchecked_transaction()?;
    tx.execute(
        &format!("DELETE FROM line_names WHERE recording = ?1 AND track = ?2 AND {OVERLAPS}"),
        params![id, track, start_ms, end_ms],
    )?;
    // Line names' voices are stored under the track, heard snippets' under their label.
    tx.execute(
        &format!("DELETE FROM voices WHERE recording = ?1 AND (label = ?2 OR label LIKE ?2 || '/%') AND {OVERLAPS}"),
        params![id, track, start_ms, end_ms],
    )?;
    let mut learned = 0;
    if let Some(name) = name.map(str::trim).filter(|n| !n.is_empty()) {
        tx.execute(
            "INSERT INTO line_names (recording, track, start_ms, end_ms, name) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, track, start_ms, end_ms, name],
        )?;
        if let Some(v) = line_voice(&tx, id, track, start_ms, end_ms)?.filter(|_| name != "?") {
            learned = tx.execute(
                "INSERT INTO voices (name, embedding, recording, label, start_ms, end_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![name, bytes(&v), id, track, start_ms, end_ms],
            )?;
        }
    }
    let label: Option<String> = tx
        .query_row(
            "SELECT speaker FROM lines WHERE recording = ?1 AND track = ?2
             AND start_ms <= ?3 AND ?3 < end_ms",
            params![id, track, (start_ms + end_ms) / 2],
            |r| r.get(0),
        )
        .optional()?
        .flatten();
    if let Some(label) = label {
        relearn_core(&tx, id, &label)?;
    }
    tx.commit()?;
    Ok(learned)
}

/// Relearns the voice `label`'s cluster taught (when it taught one) from its `clean_core`, or
/// its core again once no line of it is named otherwise.
fn relearn_core(db: &Connection, id: &str, label: &str) -> Result<()> {
    let Some(name) = confirmed(db, id)?.remove(label).filter(|n| n != "?") else {
        return Ok(());
    };
    db.execute(
        "UPDATE voices SET embedding = COALESCE(?3,
           (SELECT COALESCE(core, embedding) FROM clusters WHERE recording = ?1 AND label = ?2))
         WHERE recording = ?1 AND label = ?2 AND start_ms IS NULL",
        params![id, label, label_core(db, id, label, &name)?.as_deref().map(bytes)],
    )?;
    Ok(())
}

/// Whether `id` has line voices or its turns are kept, so naming can still learn voices from its
/// lines.
pub(crate) fn teachable(db: &Connection, id: &str) -> Result<bool> {
    Ok(db.query_row(
        "SELECT EXISTS(SELECT 1 FROM turns WHERE recording = ?1 AND embedding IS NOT NULL)
             OR EXISTS(SELECT 1 FROM line_voices WHERE recording = ?1)",
        [id],
        |r| r.get(0),
    )?)
}

/// How sure a line's name is (see CONTEXT.md).
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum State {
    Taught,
    Guessed,
    Unknown,
}

/// A line with the name it shows, None when unknown.
#[derive(Debug, Serialize)]
pub(crate) struct Named {
    #[serde(flatten)]
    pub line: Line,
    pub name: Option<String>,
    pub line_name: Option<String>,
    pub state: State,
}

/// The guess for a line not taught, from `g` (its label's confirmed name or suggestion) and
/// the name `voices` give its own `emb`: either one alone, None where they disagree.
pub(crate) fn guess<'a>(
    g: Option<&'a str>,
    emb: Option<&[f32]>,
    voices: impl IntoIterator<Item = &'a (String, Vec<f32>)>,
    m: &Matching,
) -> Option<&'a str> {
    match (emb.and_then(|e| best(e, voices, m)).map(|b| b.0), g) {
        (Some(own), Some(g)) if own != g => None,
        (Some(own), _) => Some(own),
        (None, g) => g,
    }
}

/// Whether `line` is on `track` with its midpoint in `[s, e)`: how names, voices and turns
/// kept by time find the line they belong to.
fn holds(line: &Line, track: &str, s: i64, e: i64) -> bool {
    let mid = (line.start_ms + line.end_ms) / 2;
    line.track == track && s <= mid && mid < e
}

/// `id`'s line names as (track, start_ms, end_ms, name).
fn line_names(db: &Connection, id: &str) -> Result<Vec<(String, i64, i64, String)>> {
    Ok(db
        .prepare("SELECT track, start_ms, end_ms, name FROM line_names WHERE recording = ?1")?
        .query_map([id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
        .collect::<rusqlite::Result<_>>()?)
}

/// `id`'s embedded turns with their tracks.
fn turns(db: &Connection, id: &str) -> Result<Vec<(String, Turn)>> {
    Ok(db
        .prepare(
            "SELECT track, start_ms, end_ms, speaker, embedding FROM turns
             WHERE recording = ?1 AND embedding IS NOT NULL",
        )?
        .query_map([id], |r| {
            let turn = Turn {
                start_ms: r.get(1)?,
                end_ms: r.get(2)?,
                speaker: r.get(3)?,
                embedding: Some(floats(&r.get::<_, Vec<u8>>(4)?)),
            };
            Ok((r.get(0)?, turn))
        })?
        .collect::<rusqlite::Result<_>>()?)
}

/// What a line tells about who speaks in its cluster.
struct Evidence<'a> {
    line: &'a Line,
    /// Its line name, `?` included.
    line_name: Option<&'a str>,
    /// The name it taught (see `State::Taught`).
    taught: Option<&'a str>,
    voice: Option<&'a [f32]>,
}

/// The core of `label`'s turns but those holding one of its lines named other than `name`
/// (`?` included): what the cluster teaches once naming lines showed it holds other people
/// too. None when no turn is left out, or none is kept.
fn clean_core(label: &str, name: &str, ev: &[Evidence], turns: &[(String, Turn)]) -> Option<Vec<f32>> {
    let mut left_out = false;
    let kept: Vec<&Turn> = turns
        .iter()
        .filter(|(track, t)| {
            let mut own = ev
                .iter()
                .filter(|e| e.line.speaker.as_deref() == Some(label) && holds(e.line, track, t.start_ms, t.end_ms))
                .peekable();
            own.peek().is_some() && {
                let clean = own.all(|e| e.line_name.is_none_or(|n| n == name));
                left_out |= !clean;
                clean
            }
        })
        .map(|(_, t)| t)
        .collect();
    left_out.then(|| crate::assemble::core_of(&kept)).flatten()
}

/// `clean_core` of `label` in `id` as named now.
fn label_core(db: &Connection, id: &str, label: &str, name: &str) -> Result<Option<Vec<f32>>> {
    let (lines, names) = (crate::assemble::lines(db, id)?, line_names(db, id)?);
    let ev: Vec<Evidence> = lines
        .iter()
        .map(|line| Evidence {
            line,
            line_name: names.iter().find(|n| holds(line, &n.0, n.1, n.2)).map(|n| n.3.as_str()),
            taught: None,
            voice: None,
        })
        .collect();
    Ok(clean_core(label, name, &ev, &turns(db, id)?))
}

/// A name taught this many times within a cluster (lines with a line voice) splits it.
const SPLIT_SEEDS: usize = 3;

/// The clusters of `ev` that naming lines split: those with, besides the name `label_name`
/// gives them, another name taught `SPLIT_SEEDS` times in them, or two such names. Per split
/// cluster, a centroid per such name: the voices of the lines it taught there and of the
/// turns whose lines it taught all; for the label's own name also its `core_of` the label.
fn splits<'a>(
    ev: &[Evidence<'a>],
    turns: &[(String, Turn)],
    label_name: impl Fn(&str) -> Option<&'a str>,
    core_of: impl Fn(&str, &str) -> Option<Vec<f32>>,
) -> HashMap<&'a str, Vec<(String, Vec<f32>)>> {
    // label -> name -> (lines taught with a line voice, sum of seed voices)
    let mut seeds: HashMap<&str, HashMap<&str, (usize, Vec<f32>)>> = HashMap::new();
    for e in ev {
        if let (Some(label), Some(name), Some(v)) = (e.line.speaker.as_deref(), e.taught, e.voice) {
            let s = seeds.entry(label).or_default().entry(name).or_default();
            s.0 += 1;
            add(&mut s.1, v);
        }
    }
    for (track, t) in turns {
        let mut inside = ev.iter().filter(|e| holds(e.line, track, t.start_ms, t.end_ms));
        let Some(first) = inside.next() else { continue };
        let (Some(label), Some(name)) = (first.line.speaker.as_deref(), first.taught) else {
            continue;
        };
        if inside.all(|e| e.taught == Some(name) && e.line.speaker.as_deref() == Some(label)) {
            let s = seeds.entry(label).or_default().entry(name).or_default();
            add(&mut s.1, t.embedding.as_deref().unwrap_or_default());
        }
    }
    let mut out = HashMap::new();
    for (label, by_name) in seeds {
        let own = label_name(label);
        let mut names: Vec<&str> = by_name.iter().filter(|s| s.1 .0 >= SPLIT_SEEDS).map(|s| *s.0).collect();
        names.extend(own.filter(|o| !names.contains(o)));
        if names.len() < 2 {
            continue;
        }
        let centroids: Vec<(String, Vec<f32>)> = names
            .into_iter()
            .filter_map(|name| {
                let mut sum = by_name.get(name).map(|s| s.1.clone()).unwrap_or_default();
                if Some(name) == own {
                    if let Some(c) = core_of(label, name) {
                        add(&mut sum, &c);
                    }
                }
                (!sum.is_empty()).then(|| {
                    normalize(&mut sum);
                    (name.to_string(), sum)
                })
            })
            .collect();
        if centroids.len() >= 2 {
            out.insert(label, centroids);
        }
    }
    out
}

/// The derived lines of `id` with their names: taught by a line name or a heard voice over
/// it, else guessed anew from the voices known now: within a cluster naming split, by the
/// nearest of its `splits` centroids, else by `guess`.
pub(crate) fn named(db: &Connection, id: &str) -> Result<Vec<Named>> {
    let (names, suggested) = (confirmed(db, id)?, column(db, id, "suggested")?);
    let line_names = line_names(db, id)?;
    let line_voices: Vec<(String, i64, i64, Vec<f32>)> = db
        .prepare("SELECT track, start_ms, end_ms, embedding FROM line_voices WHERE recording = ?1")?
        .query_map([id], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, floats(&r.get::<_, Vec<u8>>(3)?)))
        })?
        .collect::<rusqlite::Result<_>>()?;
    // (recording, label, span if taught from a line, (name, embedding))
    type Voice = (String, String, Option<(i64, i64)>, (String, Vec<f32>));
    let voices: Vec<Voice> = db
        .prepare("SELECT recording, label, start_ms, end_ms, name, embedding FROM voices")?
        .query_map([], |r| {
            let span = r.get::<_, Option<i64>>(2)?.zip(r.get::<_, Option<i64>>(3)?);
            Ok((
                r.get(0)?,
                r.get(1)?,
                span,
                (r.get(4)?, floats(&r.get::<_, Vec<u8>>(5)?)),
            ))
        })?
        .collect::<rusqlite::Result<_>>()?;
    let cores: HashMap<String, Vec<f32>> = db
        .prepare("SELECT label, COALESCE(core, embedding) FROM clusters WHERE recording = ?1")?
        .query_map([id], |r| Ok((r.get(0)?, floats(&r.get::<_, Vec<u8>>(1)?))))?
        .collect::<rusqlite::Result<_>>()?;
    let turns = turns(db, id)?;
    let lines = crate::assemble::lines(db, id)?;
    let own = |line: &Line, (r, l, span, _): &Voice| r == id && span.is_some_and(|(s, e)| holds(line, track(l), s, e));
    let ev: Vec<Evidence> = lines
        .iter()
        .map(|line| {
            let line_name = line_names.iter().find(|n| holds(line, &n.0, n.1, n.2)).map(|n| n.3.as_str());
            let heard = voices.iter().find(|v| own(line, v)).map(|v| v.3 .0.as_str());
            Evidence {
                line,
                line_name,
                taught: line_name.filter(|n| *n != "?").or(heard.filter(|_| line_name.is_none())),
                voice: line_voices.iter().find(|v| holds(line, &v.0, v.1, v.2)).map(|v| &v.3[..]),
            }
        })
        .collect();
    let label_name = |l: &str| match names.get(l) {
        Some(n) => Some(n.as_str()).filter(|n| *n != "?"),
        None => suggested.get(l).map(String::as_str),
    };
    let core = |label: &str, name: &str| clean_core(label, name, &ev, &turns).or_else(|| cores.get(label).cloned());
    let splits = splits(&ev, &turns, label_name, core);
    let (m, sm) = (Matching::lines_from_env(), Matching::splits_from_env());
    // A line is guessed only when it taught no voice, so none needs leaving out.
    let pooled = pool(voices.iter().map(|v| (v.0.as_str(), &v.3)));
    Ok(ev
        .iter()
        .map(|e| {
            let label = e.line.speaker.as_deref();
            let (state, name) = match (e.line_name, e.taught) {
                (Some("?"), _) => (State::Unknown, None),
                (_, Some(n)) => (State::Taught, Some(n.to_string())),
                _ => {
                    let guessed = match label.and_then(|l| splits.get(l)) {
                        Some(cs) => e.voice.and_then(|v| best(v, cs, &sm)).map(|b| b.0),
                        None => guess(label.and_then(label_name), e.voice, &pooled, &m),
                    };
                    match guessed {
                        Some(n) => (State::Guessed, Some(n.to_string())),
                        None => (State::Unknown, None),
                    }
                }
            };
            Named {
                line: e.line.clone(),
                name,
                line_name: e.line_name.map(String::from),
                state,
            }
        })
        .collect())
}

/// Who attended: the confirmed names in label order, then the names lines show in order, once
/// each.
pub(crate) fn attendees(names: &Names, lines: &[Named]) -> Vec<String> {
    let mut labels: Vec<&String> = names.keys().collect();
    labels.sort_by_key(|l| crate::vault::label_order(l));
    let mut out: Vec<String> = vec![];
    for n in labels
        .into_iter()
        .map(|l| &names[l])
        .chain(lines.iter().filter_map(|l| l.name.as_ref()))
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

    fn shown(db: &Connection) -> Vec<(Option<String>, State)> {
        super::named(db, "r1")
            .unwrap()
            .into_iter()
            .map(|l| (l.name, l.state))
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
        use State::*;
        assert_eq!(
            shown(&db),
            [s(Some("Max"), Guessed), s(Some("Eva"), Taught), s(None, Unknown)]
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
            [s(Some("Max"), Guessed), s(Some("Eva"), Taught), s(None, Unknown)]
        );
        // Naming the new line replaces the name it overlaps; clearing leaves the label's.
        name_line(&db, "r1", "room", 2_500, 4_800, Some("Bo")).unwrap();
        name_line(&db, "r1", "room", 4_800, 6_200, None).unwrap();
        assert_eq!(
            shown(&db),
            [s(Some("Max"), Guessed), s(Some("Bo"), Taught), s(None, Unknown)]
        );
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
        // A heard snippet named otherwise unlearns its voice.
        confirm(&db, "r1", heard("Max", &[(0, 4_000, true)])).unwrap();
        name_line(&db, "r1", "room", 0, 4_000, Some("Eva")).unwrap();
        assert_eq!(voices(&db), [v("Jan", 4_000), v("Eva", 0)]);
        name_line(&db, "r1", "room", 0, 4_000, None).unwrap();
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
        assert_eq!((lines[0].name.as_deref(), lines[0].state), (None, State::Unknown));
        assert!(attendees(&confirmed(&db, "r1").unwrap(), &lines).is_empty());
    }

    const L: Matching = Matching {
        threshold: 0.55,
        margin: 0.1,
    };

    #[test]
    fn a_name_counts_once_per_recording() {
        let v = |r: &'static str, n: &str, e: &[f32]| (r, (n.to_string(), e.to_vec()));
        // Max taught five lines in one meeting, one of them wrongly Eva's; Eva taught one.
        let mut voices = vec![v("a", "Max", &[1.0, 0.0]); 4];
        voices.push(v("a", "Max", &[0.0, 1.0]));
        voices.push(v("b", "Eva", &[0.1, 1.0]));
        let pooled = pool(voices.iter().map(|(r, v)| (*r, v)));
        assert_eq!(pooled.len(), 2);
        let all: Vec<_> = voices.iter().map(|v| v.1.clone()).collect();
        assert_eq!(best(&[0.0, 1.0], &all, &M), None, "the wrong voice ties with Eva");
        assert_eq!(best(&[0.0, 1.0], &pooled, &M).map(|b| b.0), Some("Eva"));
        assert_eq!(best(&[1.0, 0.0], &pooled, &M).map(|b| b.0), Some("Max"));
    }

    #[test]
    fn guesses_from_the_label_and_the_line_s_own_voice() {
        let voices = [("Max".to_string(), vec![1.0, 0.0]), ("Eva".to_string(), vec![0.0, 1.0])];
        let max = Some(&[0.9, 0.1][..]);
        let unclear = Some(&[0.7, 0.7][..]);
        assert_eq!(guess(None, max, &voices, &L), Some("Max"), "own alone");
        assert_eq!(guess(Some("Max"), max, &voices, &L), Some("Max"), "both agree");
        assert_eq!(guess(Some("Eva"), max, &voices, &L), None, "they disagree");
        assert_eq!(guess(Some("Eva"), unclear, &voices, &L), Some("Eva"), "own not sure");
        assert_eq!(guess(Some("Eva"), None, &voices, &L), Some("Eva"), "no line voice");
        assert_eq!(guess(None, unclear, &voices, &L), None);
        assert_eq!(guess(None, None, &voices, &L), None);
        assert_eq!(guess(Some("Jan"), max, &[], &L), Some("Jan"), "no voices");
    }

    fn line_voice(db: &Connection, s: i64, e: i64, v: &[f32]) {
        db.execute(
            "INSERT INTO line_voices (recording, track, start_ms, end_ms, embedding) VALUES ('r1', 'room', ?1, ?2, ?3)",
            params![s, e, bytes(v)],
        )
        .unwrap();
    }

    #[test]
    fn a_line_is_taught_guessed_or_unknown() {
        use State::*;
        let db = db();
        voice(&db, "Max", "room/S1", &[1.0, 0.0, 0.0]);
        voice(&db, "Eva", "room/S2", &[0.0, 1.0, 0.0]);
        db.execute(
            r#"UPDATE recordings SET speakers = '{"room/S1":"Max","room/S3":"?"}', suggested = '{"room/S2":"Eva"}' WHERE id = 'r1'"#,
            [],
        )
        .unwrap();
        lines(
            &db,
            &[
                (0, 1_000, "room/S1"),     // Max's label, own voice Max
                (1_000, 2_000, "room/S1"), // Max's label, own voice Eva: check it
                (2_000, 3_000, "room/S1"), // own voice unclear: the label's
                (3_000, 4_000, "room/S2"), // suggested Eva, no line voice
                (4_000, 5_000, "room/S3"), // answered ?, own voice Eva
                (5_000, 6_000, "room/S3"), // answered ?, no line voice
                (6_000, 7_000, "room/S4"), // unnamed, own voice Eva
                (7_000, 8_000, "room/S1"), // named Bo on its own, own voice Eva
                (8_000, 9_000, "room/S1"), // named ?, own voice Max
            ],
        );
        for (s, v) in [
            (0, [0.9, 0.1, 0.0]),
            (1_000, [0.1, 0.9, 0.0]),
            (2_000, [0.6, 0.6, 0.5]),
            (4_000, [0.0, 1.0, 0.0]),
            (6_000, [0.0, 1.0, 0.0]),
            (7_000, [0.0, 1.0, 0.0]),
            (8_000, [1.0, 0.0, 0.0]),
        ] {
            line_voice(&db, s, s + 1_000, &v);
        }
        db.execute_batch(
            "INSERT INTO line_names (recording, track, start_ms, end_ms, name) VALUES
               ('r1', 'room', 7000, 8000, 'Bo'), ('r1', 'room', 8000, 9000, '?');",
        )
        .unwrap();
        let s = |n: Option<&str>, t| (n.map(String::from), t);
        assert_eq!(
            shown(&db),
            [
                s(Some("Max"), Guessed),
                s(None, Unknown),
                s(Some("Max"), Guessed),
                s(Some("Eva"), Guessed),
                s(Some("Eva"), Guessed),
                s(None, Unknown),
                s(Some("Eva"), Guessed),
                s(Some("Bo"), Taught),
                s(None, Unknown),
            ]
        );
        // Heard before its label was confirmed: taught, whatever its own voice says.
        db.execute(
            "INSERT INTO voices (name, embedding, recording, label, start_ms, end_ms)
             VALUES ('Max', ?1, 'r1', 'room/S1', 1000, 2000)",
            [bytes(&[0.0, 0.0, 1.0])],
        )
        .unwrap();
        assert_eq!(shown(&db)[1], s(Some("Max"), Taught));
        // Guessed anew on every read: a voice learned later that the line's own voice matches
        // better than its label's name makes it one to check.
        voice(&db, "Jan", "remote/S9", &[0.6, 0.6, 0.5]);
        assert_eq!(shown(&db)[2], s(None, Unknown));
        // Derived anew, cut differently: line voices go by midpoint.
        lines(&db, &[(0, 1_200, "room/S1"), (1_200, 1_600, "room/S2")]);
        assert_eq!(shown(&db), [s(Some("Max"), Guessed), s(Some("Max"), Taught)]);
    }

    #[test]
    fn teaching_a_line_learns_its_line_voice_over_the_turns() {
        let db = db();
        cluster(&db, "room/S1", &[0.6, 0.8]);
        turns(&db, &[(0, 8_000, &[1.0, 0.0])]);
        line_voice(&db, 0, 4_000, &[0.0, 1.0]);
        name_line(&db, "r1", "room", 0, 4_000, Some("Eva")).unwrap();
        name_line(&db, "r1", "room", 4_000, 8_000, Some("Jan")).unwrap();
        confirm(&db, "r1", heard("Max", &[(0, 4_000, true), (4_000, 8_000, true)])).unwrap();
        let got: Vec<Vec<f32>> = learned(&db).into_iter().map(|v| v.2).collect();
        assert_eq!(got, [vec![0.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0]]);
        // Its turns expired, a line voice still teaches.
        db.execute("DELETE FROM turns", []).unwrap();
        assert!(teachable(&db, "r1").unwrap());
        assert_eq!(name_line(&db, "r1", "room", 0, 4_000, Some("Bo")).unwrap(), 1);
    }

    #[test]
    fn naming_lines_splits_a_cluster_of_two_people() {
        use State::*;
        let db = db();
        db.execute(
            r#"UPDATE recordings SET speakers = '{"room/S1":"Dave"}' WHERE id = 'r1'"#,
            [],
        )
        .unwrap();
        // One cluster holding Dave ([1, 0]) and Erin ([0, 1]), a line per second.
        let own = [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.1, 1.0],
            [0.0, 1.0],
            [0.2, 1.0],
            [1.0, 0.1],
            [1.0, 0.97],
        ];
        let spans: Vec<(i64, i64, &str)> = (0..own.len() as i64).map(|i| (i * 1_000, i * 1_000 + 1_000, "room/S1")).collect();
        lines(&db, &spans);
        for (i, v) in own.iter().enumerate() {
            line_voice(&db, i as i64 * 1_000, i as i64 * 1_000 + 1_000, v);
        }
        cluster(&db, "room/S1", &[0.6, 0.8]);
        turns(&db, &[(0, 1_000, &[1.0, 0.0]), (1_000, 5_000, &[0.0, 1.0]), (5_000, 7_000, &[1.0, 0.0])]);
        let states = |db: &Connection| -> Vec<(Option<String>, State)> { shown(db)[4..].to_vec() };
        let s = |n: Option<&str>, t| (n.map(String::from), t);
        // Erin taught twice: lines sounding like him only disagree with the label.
        name_line(&db, "r1", "room", 1_000, 2_000, Some("Erin")).unwrap();
        name_line(&db, "r1", "room", 2_000, 3_000, Some("Erin")).unwrap();
        assert_eq!(states(&db), [s(None, Unknown), s(Some("Dave"), Guessed), s(None, Unknown)]);
        // A third time splits the cluster: each line goes to the nearer of the two, the unclear
        // one to neither.
        name_line(&db, "r1", "room", 3_000, 4_000, Some("Erin")).unwrap();
        assert_eq!(states(&db), [s(Some("Erin"), Guessed), s(Some("Dave"), Guessed), s(None, Unknown)]);
        // Cleared again: no split.
        name_line(&db, "r1", "room", 3_000, 4_000, None).unwrap();
        assert_eq!(states(&db)[0], s(None, Unknown));
    }

    #[test]
    fn a_cluster_teaches_the_turns_no_line_of_it_is_named_otherwise() {
        let db = db();
        cluster(&db, "room/S1", &[0.6, 0.8]);
        turns(&db, &[(0, 4_000, &[1.0, 0.0]), (4_000, 8_000, &[0.0, 1.0])]);
        lines(&db, &[(0, 4_000, "room/S1"), (4_000, 8_000, "room/S1")]);
        confirm(&db, "r1", named(&[("room/S1", "Max")])).unwrap();
        let core = |db: &Connection| learned(db).into_iter().find(|v| v.0.is_none()).unwrap().2;
        assert_eq!(core(&db), [0.6, 0.8], "nothing named otherwise: the stored core");
        name_line(&db, "r1", "room", 4_000, 8_000, Some("Eva")).unwrap();
        assert_eq!(core(&db), [1.0, 0.0], "Eva's turn left out");
        name_line(&db, "r1", "room", 4_000, 8_000, Some("Max")).unwrap();
        assert_eq!(core(&db), [0.6, 0.8]);
        name_line(&db, "r1", "room", 4_000, 8_000, Some("?")).unwrap();
        assert_eq!(core(&db), [1.0, 0.0], "an unsure line's turn left out too");
        // Confirmed anew, it learns the clean core right away.
        confirm(&db, "r1", named(&[("room/S1", "Bo")])).unwrap();
        assert_eq!(core(&db), [1.0, 0.0]);
    }

    fn learned_count(db: &Connection) -> i64 {
        db.query_row("SELECT COUNT(*) FROM voices WHERE recording = 'r1'", [], |r| r.get(0))
            .unwrap()
    }
}
