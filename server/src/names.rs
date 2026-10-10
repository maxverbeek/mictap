//! Name resolution (see CONTEXT.md). Everything done to a recording's names is an append-only
//! log of events. `teach` derives from it the voices the recording taught and `resolve` what
//! every reader shows; both are pure. The SQL at the bottom keeps the log and, per recording,
//! a cache of `teach`: the library every other recording resolves against.
use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
};

use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

use crate::assemble::{bytes, core_of, cosine, floats, mixed, normalize, Cluster, Line, Turn};

/// Speaker label (`room/S1`) -> name.
pub(crate) type Names = BTreeMap<String, String>;

/// An event's position in the log, one order across all recordings.
pub(crate) type Seq = i64;

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

/// A stretch of a track. Things kept by time belong to the line whose midpoint they hold.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Span {
    pub start_ms: i64,
    pub end_ms: i64,
}

impl Span {
    fn mid(self) -> i64 {
        (self.start_ms + self.end_ms) / 2
    }

    fn holds(self, ms: i64) -> bool {
        self.start_ms <= ms && ms < self.end_ms
    }

    /// Either holds the other's midpoint: what a line name replaces.
    fn overlaps(self, o: Span) -> bool {
        self.holds(o.mid()) || o.holds(self.mid())
    }
}

fn mid(line: &Line) -> i64 {
    (line.start_ms + line.end_ms) / 2
}

/// What was answered for a label or a line: a name, `?` (several people, or unsure) or
/// nothing (undo). JSON: the name, `"?"` or `""`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub(crate) enum Answer {
    Name(String),
    Unsure,
    Clear,
}

impl Answer {
    /// None, `""` and blanks are Clear; `"?"` is Unsure; anything else a trimmed name.
    pub fn parse(s: Option<&str>) -> Self {
        match s.map(str::trim) {
            None | Some("") => Self::Clear,
            Some("?") => Self::Unsure,
            Some(n) => Self::Name(n.to_string()),
        }
    }

    /// The name it shows: a name, or `?` for Unsure.
    fn as_str(&self) -> Option<&str> {
        match self {
            Self::Name(n) => Some(n),
            Self::Unsure => Some("?"),
            Self::Clear => None,
        }
    }

    fn name(&self) -> Option<&str> {
        match self {
            Self::Name(n) => Some(n),
            _ => None,
        }
    }
}

impl From<String> for Answer {
    fn from(s: String) -> Self {
        Self::parse(Some(&s))
    }
}

impl From<Answer> for String {
    fn from(a: Answer) -> Self {
        a.as_str().unwrap_or_default().to_string()
    }
}

/// A line of a label that was listened to before confirming it; `correct` when it is only
/// the named speaker.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Heard {
    pub start_ms: i64,
    pub end_ms: i64,
    pub correct: bool,
}

/// How a label was confirmed: on its own, or as every unconfirmed speaker of its track at
/// once (a one-on-one). Kept for what it says about confidence; nothing weighs it yet.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Scope {
    #[default]
    Label,
    Track,
}

/// One thing done to a recording's names. Stored as JSON, so a new variant or field is a
/// code change, not a migration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum Event {
    /// A label answered, with the snippets of it listened to first.
    Confirmed {
        label: String,
        answer: Answer,
        #[serde(default)]
        heard: Vec<Heard>,
        #[serde(default)]
        scope: Scope,
    },
    /// One line named, `?`, or cleared.
    LineNamed { track: String, span: Span, answer: Answer },
    /// A guessed line played under the name it showed and not renamed. Raw; weighed nowhere
    /// yet, and never a voice.
    Seen { track: String, span: Span, shown: String },
    /// The recording was diarized anew: its labels, and what was confirmed for them, are
    /// gone. Line names are kept by time.
    Rediarized,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Logged {
    pub seq: Seq,
    pub at_ms: i64,
    pub event: Event,
}

/// What assembly and diarizing derived for one recording. Names never write it.
#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct Structure {
    pub lines: Vec<Line>,
    pub clusters: Vec<Cluster>,
    /// Embedded turns, with their track.
    pub turns: Vec<(String, Turn)>,
    /// Line voices: track, span, embedding.
    pub line_voices: Vec<(String, Span, Vec<f32>)>,
}

/// What a voice was learned from. A cluster never matches the voices it taught itself,
/// and the API answers how many each event taught.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum Source {
    /// A confirmed cluster's core (its clean core once lines of it are named otherwise).
    Core { label: String },
    /// A heard snippet kept when its label was confirmed.
    Heard { label: String, span: Span },
    /// A named line: its line voice, else the turns under it.
    Line { track: String, span: Span },
}

impl Source {
    /// The `voices.label` column: the cluster label, or the track for a line.
    fn label(&self) -> &str {
        match self {
            Self::Core { label } | Self::Heard { label, .. } => label,
            Self::Line { track, .. } => track,
        }
    }

    fn span(&self) -> Option<Span> {
        match self {
            Self::Core { .. } => None,
            Self::Heard { span, .. } | Self::Line { span, .. } => Some(*span),
        }
    }

    /// Whether this voice was taught from the line on `track` with midpoint `ms`.
    fn at(&self, track: &str, ms: i64) -> bool {
        self.span().is_some_and(|s| self::track(self.label()) == track && s.holds(ms))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Voice {
    pub recording: String,
    pub name: String,
    /// The event that taught it.
    pub from: Seq,
    pub source: Source,
    pub embedding: Vec<f32>,
}

/// Everything resolution reads for one recording (CONTEXT.md: Known). The library is every
/// recording's cached voices; this recording's own come fresh from `teach`.
#[derive(Serialize, Deserialize)]
pub(crate) struct Known<'a> {
    pub id: String,
    pub events: Vec<Logged>,
    pub structure: Structure,
    pub library: Cow<'a, [Voice]>,
}

pub(crate) struct Matching {
    /// A voice at least this alike to a sample can name it.
    pub threshold: f32,
    /// ... if no voice of another name is within this of it.
    pub margin: f32,
}

impl Matching {
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

/// Every threshold and margin of name resolution.
pub(crate) struct Rules {
    /// Suggesting a name for a cluster: `MICTAP_MATCH_*`.
    pub cluster: Matching,
    /// Guessing a line by its own voice: `MICTAP_LINE_*`.
    pub line: Matching,
    /// Between the names of a split cluster, which share a channel: no threshold, and
    /// `MICTAP_SPLIT_MARGIN`.
    pub split: Matching,
    /// A cluster whose halves are less alike than this is mixed: `MICTAP_MERGE_THRESHOLD`.
    pub merge_threshold: f32,
}

impl Rules {
    pub fn from_env() -> Self {
        Self {
            cluster: Matching::env("MICTAP_MATCH", 0.75, 0.05),
            line: Matching::env("MICTAP_LINE", 0.55, 0.1),
            split: Matching::env("MICTAP_SPLIT", -1.0, 0.02),
            merge_threshold: crate::assemble::Tuning::from_env().merge_threshold,
        }
    }
}

// ---- the fold ----

struct Snippet {
    span: Span,
    correct: bool,
    /// A line named over it since: its voice is unlearned, and it teaches no core either.
    voided: bool,
}

struct Confirm<'a> {
    answer: &'a Answer,
    heard: Vec<Snippet>,
    from: Seq,
}

struct LineName<'a> {
    track: &'a str,
    span: Span,
    /// `?` included.
    name: &'a str,
    from: Seq,
}

/// The log folded: per label its last confirmation that taught anew, and the line names in
/// force.
#[derive(Default)]
struct Folded<'a> {
    labels: BTreeMap<&'a str, Confirm<'a>>,
    line_names: Vec<LineName<'a>>,
}

fn fold(events: &[Logged]) -> Folded<'_> {
    let mut f = Folded::default();
    for l in events {
        match &l.event {
            Event::Confirmed {
                label, answer, heard, ..
            } => {
                if *answer == Answer::Clear {
                    f.labels.remove(label.as_str());
                    continue;
                }
                // The same answer again without listening changes nothing.
                let same = f.labels.get(label.as_str()).is_some_and(|c| c.answer == answer);
                if same && heard.is_empty() {
                    continue;
                }
                let heard = heard
                    .iter()
                    .map(|h| Snippet {
                        span: Span {
                            start_ms: h.start_ms,
                            end_ms: h.end_ms,
                        },
                        correct: h.correct,
                        voided: false,
                    })
                    .collect();
                f.labels.insert(label, Confirm { answer, heard, from: l.seq });
            }
            Event::LineNamed { track, span, answer } => {
                for (_, c) in f.labels.iter_mut().filter(|(l, _)| self::track(l) == track) {
                    for h in c.heard.iter_mut().filter(|h| h.span.overlaps(*span)) {
                        h.voided = true;
                    }
                }
                f.line_names.retain(|n| !(n.track == track && n.span.overlaps(*span)));
                if let Some(name) = answer.as_str() {
                    f.line_names.push(LineName {
                        track,
                        span: *span,
                        name,
                        from: l.seq,
                    });
                }
            }
            Event::Seen { .. } => {}
            Event::Rediarized => f.labels.clear(),
        }
    }
    f
}

fn line_name_of<'a>(names: &[LineName<'a>], line: &Line) -> Option<&'a str> {
    names
        .iter()
        .find(|n| n.track == line.track && n.span.holds(mid(line)))
        .map(|n| n.name)
}

fn mixed_labels<'a>(s: &'a Structure, rules: &Rules) -> HashSet<&'a str> {
    s.clusters
        .iter()
        .filter(|c| mixed(c.halves, rules.merge_threshold))
        .map(|c| c.label.as_str())
        .collect()
}

/// A line's own voice: its line voice (by midpoint), else the normalized mean of the turns
/// under it weighted by overlap; None without either.
fn sample(s: &Structure, track: &str, span: Span) -> Option<Vec<f32>> {
    if let Some((_, _, v)) = s.line_voices.iter().find(|(t, ls, _)| t == track && ls.holds(span.mid())) {
        return Some(v.clone());
    }
    let mut sum: Option<Vec<f32>> = None;
    for (t, turn) in s.turns.iter().filter(|(t, _)| t == track) {
        let Some(e) = &turn.embedding else { continue };
        if !(turn.start_ms < span.end_ms && turn.end_ms > span.start_ms) {
            continue;
        }
        let w = (turn.end_ms.min(span.end_ms) - turn.start_ms.max(span.start_ms)) as f32;
        match &mut sum {
            Some(sum) => sum.iter_mut().zip(e).for_each(|(a, b)| *a += w * b),
            slot => *slot = Some(e.iter().map(|b| w * b).collect()),
        }
        let _ = t;
    }
    sum.map(|mut v| {
        normalize(&mut v);
        v
    })
}

/// The core of `label`'s turns but those holding one of its lines named other than `name`
/// (`?` included): what the cluster teaches once naming lines showed it holds other people
/// too. None when no turn is left out, or none is kept.
fn clean_core(label: &str, name: &str, lines: &[Line], names: &[LineName], turns: &[(String, Turn)]) -> Option<Vec<f32>> {
    let mut left_out = false;
    let kept: Vec<&Turn> = turns
        .iter()
        .filter(|(track, t)| {
            let span = Span {
                start_ms: t.start_ms,
                end_ms: t.end_ms,
            };
            let mut own = lines
                .iter()
                .filter(|l| l.speaker.as_deref() == Some(label) && l.track == *track && span.holds(mid(l)))
                .peekable();
            own.peek().is_some() && {
                let clean = own.all(|l| line_name_of(names, l).is_none_or(|n| n == name));
                left_out |= !clean;
                clean
            }
        })
        .map(|(_, t)| t)
        .collect();
    left_out.then(|| core_of(&kept)).flatten()
}

/// Pure. The voices `events` teach over `s`: per confirmed label one per correct heard
/// snippet that has a sample, else (when no correct snippet had one) its core unless mixed;
/// per named line its sample. `?` and cleared answers teach nothing.
pub(crate) fn teach(id: &str, events: &[Logged], s: &Structure, rules: &Rules) -> Vec<Voice> {
    let f = fold(events);
    let mixed = mixed_labels(s, rules);
    let voice = |name: &str, from, source, embedding| Voice {
        recording: id.to_string(),
        name: name.to_string(),
        from,
        source,
        embedding,
    };
    let mut out = vec![];
    for (label, c) in &f.labels {
        let Some(name) = c.answer.name() else { continue };
        if !c.heard.is_empty() {
            let mut sampled = false;
            for h in c.heard.iter().filter(|h| h.correct) {
                let Some(v) = sample(s, track(label), h.span) else { continue };
                sampled = true;
                if !h.voided {
                    let source = Source::Heard {
                        label: label.to_string(),
                        span: h.span,
                    };
                    out.push(voice(name, c.from, source, v));
                }
            }
            if sampled || !c.heard.iter().any(|h| h.correct) {
                continue;
            }
        }
        if mixed.contains(label) {
            continue;
        }
        let Some(cluster) = s.clusters.iter().find(|c| c.label == *label) else { continue };
        let emb = clean_core(label, name, &s.lines, &f.line_names, &s.turns)
            .or_else(|| cluster.core.clone())
            .unwrap_or_else(|| cluster.mean.clone());
        out.push(voice(name, c.from, Source::Core { label: label.to_string() }, emb));
    }
    for n in &f.line_names {
        if n.name == "?" {
            continue;
        }
        if let Some(v) = sample(s, n.track, n.span) {
            let source = Source::Line {
                track: n.track.to_string(),
                span: n.span,
            };
            out.push(voice(n.name, n.from, source, v));
        }
    }
    out.sort_by_key(|v| v.from);
    out
}

// ---- resolve ----

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

/// One voice per recording and name: the mean of the ones taught there, so a name taught on
/// many lines of a meeting gets one chance to match rather than one per line, and a wrong
/// one among them is outweighed.
fn pool<'a>(voices: impl IntoIterator<Item = &'a Voice>) -> Vec<(String, Vec<f32>)> {
    let mut sums: HashMap<(&str, &str, usize), Vec<f32>> = HashMap::new();
    for v in voices {
        let mut e = v.embedding.clone();
        normalize(&mut e);
        add(sums.entry((&v.recording, &v.name, e.len())).or_default(), &e);
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

/// A name for each unconfirmed, unmixed cluster that one clearly wins, from `voices`
/// pooled; no name twice within a track, nor where the track already confirms it: the most
/// similar cluster gets it.
fn suggest(s: &Structure, confirmed: &Names, mixed: &HashSet<&str>, voices: &[&Voice], m: &Matching) -> Names {
    let pooled = pool(voices.iter().copied());
    let mut candidates: Vec<(f32, &str, &str)> = s
        .clusters
        .iter()
        .filter(|c| !confirmed.contains_key(&c.label) && !mixed.contains(c.label.as_str()))
        .filter_map(|c| {
            best(c.core.as_deref().unwrap_or(&c.mean), &pooled, m).map(|(name, cos)| (cos, c.label.as_str(), name))
        })
        .collect();
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut taken: HashSet<(&str, &str)> = confirmed.iter().map(|(l, n)| (track(l), n.as_str())).collect();
    candidates
        .into_iter()
        .filter(|(_, label, name)| taken.insert((track(label), name)))
        .map(|(_, label, name)| (label.to_string(), name.to_string()))
        .collect()
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

#[derive(Debug, PartialEq, Serialize)]
pub(crate) struct Speaker {
    pub name: Option<String>,
    pub suggested: Option<String>,
    /// Its cluster sounds like two speakers or a room: a name is kept, no voice learned.
    pub mixed: bool,
}

/// What every reader shows. Derived on every read, never stored.
#[derive(Debug, Serialize)]
pub(crate) struct Resolved {
    pub lines: Vec<Named>,
    /// Every label some line carries: its confirmed name, else its suggestion.
    pub speakers: BTreeMap<String, Speaker>,
    pub attendees: Vec<String>,
    /// Naming can still learn voices: there are turns or line voices.
    pub teachable: bool,
}

impl Resolved {
    /// Lines taught, guessed and unknown.
    pub fn counts(&self) -> [usize; 3] {
        let count = |s| self.lines.iter().filter(|l| l.state == s).count();
        [count(State::Taught), count(State::Guessed), count(State::Unknown)]
    }
}

/// The guess for a line not taught, from `g` (its label's confirmed name or suggestion) and
/// the name `voices` give its own `emb`: either one alone, None where they disagree.
fn guess<'a>(
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

/// What a line tells about who speaks in its cluster.
struct Evidence<'a> {
    line: &'a Line,
    /// Its line name, `?` included.
    line_name: Option<&'a str>,
    /// The name it taught (see `State::Taught`).
    taught: Option<&'a str>,
    voice: Option<&'a [f32]>,
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
        let span = Span {
            start_ms: t.start_ms,
            end_ms: t.end_ms,
        };
        let mut inside = ev.iter().filter(|e| e.line.track == *track && span.holds(mid(e.line)));
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

/// Pure, total. Folds the log, teaches this recording's voices anew, suggests a name per
/// cluster from the library and them, and names each line: taught by a line name or a heard
/// snippet over it, else guessed from the voices known now (within a cluster naming split,
/// by the nearest of its centroids), else unknown.
pub(crate) fn resolve(k: &Known, rules: &Rules) -> Resolved {
    let id = k.id.as_str();
    let s = &k.structure;
    let f = fold(&k.events);
    let own = teach(id, &k.events, s, rules);
    let confirmed: Names = f
        .labels
        .iter()
        .filter_map(|(l, c)| Some((l.to_string(), c.answer.as_str()?.to_string())))
        .collect();
    let mixed = mixed_labels(s, rules);
    let voices: Vec<&Voice> = k.library.iter().filter(|v| v.recording != id).chain(&own).collect();
    let suggested = suggest(s, &confirmed, &mixed, &voices, &rules.cluster);
    let cores: HashMap<&str, &[f32]> = s
        .clusters
        .iter()
        .map(|c| (c.label.as_str(), c.core.as_deref().unwrap_or(&c.mean)))
        .collect();
    let ev: Vec<Evidence> = s
        .lines
        .iter()
        .map(|line| {
            let line_name = line_name_of(&f.line_names, line);
            let heard = own
                .iter()
                .find(|v| v.source.at(&line.track, mid(line)))
                .map(|v| v.name.as_str());
            Evidence {
                line,
                line_name,
                taught: line_name.filter(|n| *n != "?").or(heard.filter(|_| line_name.is_none())),
                voice: s
                    .line_voices
                    .iter()
                    .find(|(t, sp, _)| *t == line.track && sp.holds(mid(line)))
                    .map(|v| &v.2[..]),
            }
        })
        .collect();
    let label_name = |l: &str| match confirmed.get(l) {
        Some(n) => Some(n.as_str()).filter(|n| *n != "?"),
        None => suggested.get(l).map(String::as_str),
    };
    let core = |label: &str, name: &str| {
        clean_core(label, name, &s.lines, &f.line_names, &s.turns).or_else(|| cores.get(label).map(|c| c.to_vec()))
    };
    let splits = splits(&ev, &s.turns, label_name, core);
    // A line is guessed only when it taught no voice, so none needs leaving out.
    let pooled = pool(voices.iter().copied());
    let lines: Vec<Named> = ev
        .iter()
        .map(|e| {
            let label = e.line.speaker.as_deref();
            let (state, name) = match (e.line_name, e.taught) {
                (Some("?"), _) => (State::Unknown, None),
                (_, Some(n)) => (State::Taught, Some(n.to_string())),
                _ => {
                    let guessed = match label.and_then(|l| splits.get(l)) {
                        Some(cs) => e.voice.and_then(|v| best(v, cs, &rules.split)).map(|b| b.0),
                        None => guess(label.and_then(label_name), e.voice, &pooled, &rules.line),
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
        .collect();
    let labels: BTreeSet<&str> = s.lines.iter().filter_map(|l| l.speaker.as_deref()).collect();
    let speakers = labels
        .into_iter()
        .map(|l| {
            let name = confirmed.get(l).cloned();
            let speaker = Speaker {
                suggested: suggested.get(l).filter(|_| name.is_none()).cloned(),
                name,
                mixed: mixed.contains(l),
            };
            (l.to_string(), speaker)
        })
        .collect();
    let attendees = attendees(&confirmed, &lines);
    Resolved {
        lines,
        speakers,
        attendees,
        teachable: !s.turns.is_empty() || !s.line_voices.is_empty(),
    }
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

// ---- SQL: the log, and the cache of `teach` ----

/// Every recording's cached voices.
pub(crate) fn library(db: &Connection) -> Result<Vec<Voice>> {
    Ok(db
        .prepare("SELECT recording, name, COALESCE(seq, 0), label, start_ms, end_ms, embedding FROM voices ORDER BY id")?
        .query_map([], |r| {
            let label: String = r.get(3)?;
            let span = r.get::<_, Option<i64>>(4)?.zip(r.get::<_, Option<i64>>(5)?);
            let source = match span {
                Some((start_ms, end_ms)) if is_label(&label) => Source::Heard {
                    label,
                    span: Span { start_ms, end_ms },
                },
                Some((start_ms, end_ms)) => Source::Line {
                    track: label,
                    span: Span { start_ms, end_ms },
                },
                None => Source::Core { label },
            };
            Ok(Voice {
                recording: r.get(0)?,
                name: r.get(1)?,
                from: r.get(2)?,
                source,
                embedding: floats(&r.get::<_, Vec<u8>>(6)?),
            })
        })?
        .collect::<rusqlite::Result<_>>()?)
}

fn events(db: &Connection, id: &str) -> Result<Vec<Logged>> {
    db.prepare("SELECT seq, at_ms, event FROM events WHERE recording = ?1 ORDER BY seq")?
        .query_map([id], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, String>(2)?)))?
        .map(|r| {
            let (seq, at_ms, json) = r?;
            // A variant this build cannot read is an error, never skipped: evidence is not
            // forgotten by a downgrade.
            let event = serde_json::from_str(&json).with_context(|| format!("event {seq}"))?;
            Ok(Logged { seq, at_ms, event })
        })
        .collect()
}

fn structure(db: &Connection, id: &str) -> Result<Structure> {
    let clusters = db
        .prepare("SELECT label, embedding, core, halves_alike, minor_share FROM clusters WHERE recording = ?1")?
        .query_map([id], |r| {
            Ok(Cluster {
                label: r.get(0)?,
                mean: floats(&r.get::<_, Vec<u8>>(1)?),
                core: r.get::<_, Option<Vec<u8>>>(2)?.map(|b| floats(&b)),
                halves: r.get::<_, Option<f32>>(3)?.zip(r.get::<_, Option<f32>>(4)?),
            })
        })?
        .collect::<rusqlite::Result<_>>()?;
    let turns = db
        .prepare(
            "SELECT track, start_ms, end_ms, speaker, embedding FROM turns
             WHERE recording = ?1 AND embedding IS NOT NULL ORDER BY id",
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
        .collect::<rusqlite::Result<_>>()?;
    let line_voices = db
        .prepare("SELECT track, start_ms, end_ms, embedding FROM line_voices WHERE recording = ?1")?
        .query_map([id], |r| {
            let span = Span {
                start_ms: r.get(1)?,
                end_ms: r.get(2)?,
            };
            Ok((r.get(0)?, span, floats(&r.get::<_, Vec<u8>>(3)?)))
        })?
        .collect::<rusqlite::Result<_>>()?;
    Ok(Structure {
        lines: crate::assemble::lines(db, id)?,
        clusters,
        turns,
        line_voices,
    })
}

/// `id`'s Known around `library` (loaded once by callers that resolve many recordings).
pub(crate) fn load<'a>(db: &Connection, id: &str, library: &'a [Voice]) -> Result<Known<'a>> {
    Ok(Known {
        id: id.to_string(),
        events: events(db, id)?,
        structure: structure(db, id)?,
        library: Cow::Borrowed(library),
    })
}

fn project_in(db: &Connection, id: &str) -> Result<Vec<Voice>> {
    let voices = teach(id, &events(db, id)?, &structure(db, id)?, &Rules::from_env());
    db.execute("DELETE FROM voices WHERE recording = ?1", [id])?;
    for v in &voices {
        let span = v.source.span();
        db.execute(
            "INSERT INTO voices (name, embedding, recording, label, start_ms, end_ms, seq)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                v.name,
                bytes(&v.embedding),
                id,
                v.source.label(),
                span.map(|s| s.start_ms),
                span.map(|s| s.end_ms),
                v.from
            ],
        )?;
    }
    Ok(voices)
}

/// Replaces `id`'s cached voices with `teach` of its log over its structure now. Call it
/// after the structure changed (diarized anew).
pub(crate) fn project(db: &Connection, id: &str) -> Result<()> {
    let tx = db.unchecked_transaction()?;
    project_in(&tx, id)?;
    tx.commit()?;
    Ok(())
}

/// `project` for every recording: at startup, since the rules may have changed.
pub(crate) fn rebuild(db: &Connection) -> Result<()> {
    let ids: Vec<String> = db
        .prepare("SELECT id FROM recordings")?
        .query_map([], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    for id in ids {
        project(db, &id)?;
    }
    Ok(())
}

/// Appends `events` to `id`'s log and projects, in one transaction. Returns how many voices
/// each event teaches now. Callers hold the db lock; not inside another transaction.
pub(crate) fn record(db: &Connection, id: &str, events: &[Event]) -> Result<Vec<usize>> {
    let tx = db.unchecked_transaction()?;
    let at_ms = crate::db::now_ms();
    let mut seqs = vec![];
    for e in events {
        tx.execute(
            "INSERT INTO events (recording, at_ms, event) VALUES (?1, ?2, ?3)",
            params![id, at_ms, serde_json::to_string(e)?],
        )?;
        seqs.push(tx.last_insert_rowid());
    }
    let voices = project_in(&tx, id)?;
    tx.commit()?;
    Ok(seqs
        .iter()
        .map(|s| voices.iter().filter(|v| v.from == *s).count())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use State::*;

    const M: Matching = Matching {
        threshold: 0.75,
        margin: 0.05,
    };
    const L: Matching = Matching {
        threshold: 0.55,
        margin: 0.1,
    };
    const R: Rules = Rules {
        cluster: M,
        line: L,
        split: Matching {
            threshold: -1.0,
            margin: 0.02,
        },
        merge_threshold: 0.75,
    };

    fn log(events: &[Event]) -> Vec<Logged> {
        events
            .iter()
            .enumerate()
            .map(|(i, e)| Logged {
                seq: i as i64 + 1,
                at_ms: 0,
                event: e.clone(),
            })
            .collect()
    }

    fn line(track: &str, start_ms: i64, end_ms: i64, label: &str) -> Line {
        Line {
            track: track.into(),
            start_ms,
            end_ms,
            text: String::new(),
            speaker: Some(label.into()),
        }
    }

    fn cluster(label: &str, mean: &[f32]) -> Cluster {
        Cluster {
            label: label.into(),
            mean: mean.to_vec(),
            core: None,
            halves: None,
        }
    }

    /// One line per cluster, a second each, on its track.
    fn lines_of(clusters: &[Cluster]) -> Vec<Line> {
        clusters
            .iter()
            .enumerate()
            .map(|(i, c)| line(track(&c.label), i as i64 * 1_000, i as i64 * 1_000 + 1_000, &c.label))
            .collect()
    }

    fn turn(start_ms: i64, end_ms: i64, v: &[f32]) -> (String, Turn) {
        let turn = Turn {
            start_ms,
            end_ms,
            speaker: 0,
            embedding: Some(v.to_vec()),
        };
        ("room".into(), turn)
    }

    fn lv(start_ms: i64, end_ms: i64, v: &[f32]) -> (String, Span, Vec<f32>) {
        ("room".into(), Span { start_ms, end_ms }, v.to_vec())
    }

    /// A voice of another recording.
    fn lib(name: &str, label: &str, v: &[f32]) -> Voice {
        Voice {
            recording: "old".into(),
            name: name.into(),
            from: 0,
            source: Source::Core { label: label.into() },
            embedding: v.to_vec(),
        }
    }

    fn confirm(label: &str, name: &str) -> Event {
        Event::Confirmed {
            label: label.into(),
            answer: Answer::parse(Some(name)),
            heard: vec![],
            scope: Scope::Label,
        }
    }

    fn heard(label: &str, name: &str, snippets: &[(i64, i64, bool)]) -> Event {
        Event::Confirmed {
            label: label.into(),
            answer: Answer::parse(Some(name)),
            heard: snippets
                .iter()
                .map(|&(start_ms, end_ms, correct)| Heard {
                    start_ms,
                    end_ms,
                    correct,
                })
                .collect(),
            scope: Scope::Label,
        }
    }

    fn name_line(start_ms: i64, end_ms: i64, name: Option<&str>) -> Event {
        Event::LineNamed {
            track: "room".into(),
            span: Span { start_ms, end_ms },
            answer: Answer::parse(name),
        }
    }

    fn known<'a>(events: &[Event], structure: Structure, library: &'a [Voice]) -> Known<'a> {
        Known {
            id: "r1".into(),
            events: log(events),
            structure,
            library: Cow::Borrowed(library),
        }
    }

    fn teach(events: &[Event], s: &Structure) -> Vec<Voice> {
        super::teach("r1", &log(events), s, &R)
    }

    /// (span, embedding) of each voice.
    fn learned(voices: &[Voice]) -> Vec<Learned> {
        voices
            .iter()
            .map(|v| (v.source.span().map(|s| (s.start_ms, s.end_ms)), v.embedding.clone()))
            .collect()
    }

    type Learned = (Option<(i64, i64)>, Vec<f32>);

    fn shown(r: &Resolved) -> Vec<(Option<&str>, State)> {
        r.lines.iter().map(|l| (l.name.as_deref(), l.state)).collect()
    }

    fn suggested(r: &Resolved) -> Names {
        r.speakers
            .iter()
            .filter_map(|(l, s)| Some((l.clone(), s.suggested.clone()?)))
            .collect()
    }

    #[test]
    fn best_needs_the_threshold_and_a_margin_over_other_names() {
        let voices = [
            ("Max".to_string(), vec![1.0, 0.0, 0.0]),
            ("Max".to_string(), vec![0.8, 0.6, 0.0]),
            ("Alice".to_string(), vec![0.0, 1.0, 0.0]),
        ];
        // Max's best voice counts, not his average; another Max voice is no competition.
        assert_eq!(best(&[0.9, 0.1, 0.0], &voices, &M).map(|b| b.0), Some("Max"));
        // Max's second voice 0.96 against Alice's 0.8: clear.
        assert_eq!(best(&[0.6, 0.8, 0.0], &voices, &M).map(|b| b.0), Some("Max"));
        // Halfway between Max's second voice and Alice's, both 0.89: unknown.
        assert_eq!(best(&[0.8, 1.6, 0.0], &voices[1..], &M), None);
        assert_eq!(best(&[0.0, 0.0, 1.0], &voices, &M), None, "alike to no one");
        assert_eq!(best(&[1.0, 0.0], &voices, &M), None, "other model's dimension");
        assert_eq!(best(&[1.0, 0.0, 0.0], &[], &M), None);
    }

    #[test]
    fn answers_and_events_round_trip_as_json() {
        assert_eq!(Answer::parse(Some(" Max ")), Answer::Name("Max".into()));
        assert_eq!(Answer::parse(Some(" ")), Answer::Clear);
        let e = Event::Confirmed {
            label: "room/S1".into(),
            answer: Answer::Unsure,
            heard: vec![],
            scope: Scope::Track,
        };
        let json = serde_json::to_string(&e).unwrap();
        assert_eq!(json, r#"{"kind":"confirmed","label":"room/S1","answer":"?","heard":[],"scope":"track"}"#);
        assert_eq!(serde_json::from_str::<Event>(&json).unwrap(), e);
        // Old events without the fields added since still read.
        let old: Event = serde_json::from_str(r#"{"kind":"confirmed","label":"room/S1","answer":""}"#).unwrap();
        assert_eq!(old, confirm("room/S1", ""));
        assert_eq!(serde_json::to_string(&Event::Rediarized).unwrap(), r#"{"kind":"rediarized"}"#);
    }

    #[test]
    fn suggests_from_the_library_and_the_recording_s_own_named_lines() {
        let clusters = vec![cluster("room/S1", &[1.0, 0.0, 0.0]), cluster("room/S2", &[0.0, 1.0, 0.0])];
        let s = Structure {
            lines: lines_of(&clusters),
            clusters,
            line_voices: vec![lv(0, 1_000, &[1.0, 0.0, 0.0])],
            ..Default::default()
        };
        assert_eq!(suggested(&resolve(&known(&[], s, &[]), &R)), Names::new());
        // Stale rows of this recording in the library are ignored: its own voices come from
        // its log.
        let stale = Voice {
            recording: "r1".into(),
            ..lib("Stale", "room/S2", &[0.0, 1.0, 0.0])
        };
        let library = [lib("Alice", "remote/S1", &[0.0, 1.0, 0.0]), stale];
        let s = Structure {
            lines: lines_of(&[cluster("room/S1", &[1.0, 0.0, 0.0]), cluster("room/S2", &[0.0, 1.0, 0.0])]),
            clusters: vec![cluster("room/S1", &[1.0, 0.0, 0.0]), cluster("room/S2", &[0.0, 1.0, 0.0])],
            line_voices: vec![lv(0, 1_000, &[1.0, 0.0, 0.0])],
            ..Default::default()
        };
        let r = resolve(&known(&[name_line(0, 1_000, Some("Max"))], s, &library), &R);
        let want: Names = [("room/S1", "Max"), ("room/S2", "Alice")]
            .map(|(l, n)| (l.to_string(), n.to_string()))
            .into();
        assert_eq!(suggested(&r), want, "S1 from its own line, S2 from the library");
    }

    #[test]
    fn suggests_each_name_once_per_track_and_not_over_confirmed_ones() {
        let library = [
            lib("Max", "room/S1", &[1.0, 0.0, 0.0]),
            lib("Alice", "room/S2", &[0.0, 1.0, 0.0]),
            lib("Bob", "remote/S1", &[0.0, 0.0, 1.0]),
        ];
        let mut clusters = vec![
            cluster("room/S1", &[0.9, 0.1, 0.0]),
            cluster("room/S2", &[0.99, 0.0, 0.1]),  // Max too, more alike: gets it
            cluster("room/S3", &[0.0, 0.2, 1.0]),   // Bob: confirmed on the remote track only
            cluster("remote/S1", &[1.0, 0.05, 0.0]), // Max on another track: fine
            cluster("remote/S2", &[0.0, 0.0, 1.0]), // Bob, confirmed for remote/S3
            cluster("remote/S4", &[0.0, 1.0, 0.0]), // Alice, but mixed
        ];
        clusters[5].halves = Some((0.2, 0.5));
        let s = Structure {
            lines: lines_of(&clusters),
            clusters,
            ..Default::default()
        };
        let events = [confirm("remote/S3", "Bob"), confirm("room/S4", "Carol")];
        let r = resolve(&known(&events, s, &library), &R);
        let want: Names = [("remote/S1", "Max"), ("room/S2", "Max"), ("room/S3", "Bob")]
            .map(|(l, n)| (l.to_string(), n.to_string()))
            .into();
        assert_eq!(suggested(&r), want);
    }

    #[test]
    fn confirming_learns_the_core_and_clearing_unlearns_it() {
        let s = Structure {
            clusters: vec![
                cluster("room/S1", &[1.0, 0.0]),
                cluster("room/S2", &[0.0, 1.0]),
                cluster("room/S3", &[0.5, 0.5]),
            ],
            ..Default::default()
        };
        let events = [
            confirm("room/S3", "Carol"),
            confirm("room/S1", " Max "),
            confirm("room/S2", ""),
            confirm("room/S3", ""),
        ];
        let voices = teach(&events, &s);
        let got: Vec<(&str, &str)> = voices.iter().map(|v| (v.name.as_str(), v.source.label())).collect();
        assert_eq!(got, [("Max", "room/S1")]);
        let r = resolve(&known(&events, s, &[]), &R);
        assert_eq!(r.attendees, ["Max"]);
    }

    #[test]
    fn a_mixed_cluster_keeps_its_name_but_teaches_no_voice() {
        let mut clusters = vec![cluster("room/S1", &[1.0, 0.0]), cluster("room/S2", &[0.0, 1.0])];
        clusters[0].halves = Some((0.3, 0.4));
        clusters[1].halves = Some((0.9, 0.4));
        let s = Structure {
            lines: lines_of(&clusters),
            clusters,
            ..Default::default()
        };
        let events = [confirm("room/S1", "Room B"), confirm("room/S2", "Alice")];
        let voices = teach(&events, &s);
        let names: Vec<&str> = voices.iter().map(|v| v.name.as_str()).collect();
        assert_eq!(names, ["Alice"]);
        let r = resolve(&known(&events, s, &[]), &R);
        assert!(r.speakers["room/S1"].mixed && !r.speakers["room/S2"].mixed);
        assert_eq!(r.speakers["room/S1"].name.as_deref(), Some("Room B"));
    }

    #[test]
    fn matches_and_learns_the_core() {
        let library = [lib("Max", "room/S1", &[1.0, 0.0])];
        let mut c = cluster("room/S1", &[0.0, 1.0]);
        c.core = Some(vec![0.99, 0.1]);
        let s = Structure {
            lines: lines_of(&[c.clone()]),
            clusters: vec![c],
            ..Default::default()
        };
        let r = resolve(&known(&[], s, &library), &R);
        assert_eq!(r.speakers["room/S1"].suggested.as_deref(), Some("Max"));
        let s = Structure {
            clusters: vec![Cluster {
                core: Some(vec![0.99, 0.1]),
                ..cluster("room/S1", &[0.0, 1.0])
            }],
            ..Default::default()
        };
        assert_eq!(learned(&teach(&[confirm("room/S1", "Max")], &s)), [(None, vec![0.99, 0.1])]);
    }

    #[test]
    fn a_confirmed_name_hides_the_suggestion() {
        let library = [lib("Alice", "room/S1", &[1.0, 0.0]), lib("Bob", "room/S2", &[0.0, 1.0])];
        let clusters = vec![
            cluster("room/S1", &[1.0, 0.0]),
            cluster("room/S2", &[0.0, 1.0]),
            cluster("room/S3", &[0.7, 0.7]),
        ];
        let s = Structure {
            lines: lines_of(&clusters),
            clusters,
            ..Default::default()
        };
        let r = resolve(&known(&[confirm("room/S1", "Max")], s, &library), &R);
        let sp = |name: Option<&str>, suggested: Option<&str>| Speaker {
            name: name.map(Into::into),
            suggested: suggested.map(Into::into),
            mixed: false,
        };
        assert_eq!(r.speakers["room/S1"], sp(Some("Max"), None));
        assert_eq!(r.speakers["room/S2"], sp(None, Some("Bob")));
        assert_eq!(r.speakers["room/S3"], sp(None, None));
    }

    #[test]
    fn learns_one_voice_per_correct_heard_snippet() {
        let s = Structure {
            clusters: vec![cluster("room/S1", &[0.6, 0.8])],
            turns: vec![turn(0, 4_000, &[1.0, 0.0]), turn(4_000, 8_000, &[0.0, 1.0])],
            ..Default::default()
        };
        let snippets = [(0, 4_000, true), (4_000, 8_000, false), (2_000, 6_000, true)];
        let mut events = vec![heard("room/S1", "Max", &snippets)];
        let h = std::f32::consts::FRAC_1_SQRT_2;
        let got = learned(&teach(&events, &s));
        assert_eq!(got.len(), 2, "{got:?}");
        assert_eq!(got[0], (Some((0, 4_000)), vec![1.0, 0.0]));
        assert_eq!(got[1].0, Some((2_000, 6_000)));
        assert!(
            (got[1].1[0] - h).abs() < 1e-6 && (got[1].1[1] - h).abs() < 1e-6,
            "{got:?}"
        );

        // Heard again, all wrong: the name stays, its voices go.
        events.push(heard("room/S1", "Max", &[(0, 4_000, false)]));
        assert_eq!(learned(&teach(&events, &s)), []);
        let r = resolve(&known(&events, s, &[]), &R);
        assert_eq!(r.attendees, ["Max"]);
    }

    #[test]
    fn heard_snippets_whose_turns_expired_teach_the_core() {
        let mut s = Structure {
            clusters: vec![cluster("room/S1", &[0.6, 0.8])],
            ..Default::default()
        };
        let events = [heard("room/S1", "Max", &[(0, 4_000, true)])];
        assert_eq!(learned(&teach(&events, &s)), [(None, vec![0.6, 0.8])]);
        // ... unless mixed.
        s.clusters[0].halves = Some((0.2, 0.5));
        assert_eq!(learned(&teach(&events, &s)), []);
    }

    #[test]
    fn hearing_a_confirmed_name_again_relearns_it() {
        let s = Structure {
            clusters: vec![cluster("room/S1", &[0.6, 0.8])],
            turns: vec![turn(0, 4_000, &[1.0, 0.0])],
            ..Default::default()
        };
        let mut events = vec![confirm("room/S1", "Max")];
        assert_eq!(learned(&teach(&events, &s)), [(None, vec![0.6, 0.8])]);
        events.push(heard("room/S1", "Max", &[(0, 4_000, true)]));
        assert_eq!(learned(&teach(&events, &s)), [(Some((0, 4_000)), vec![1.0, 0.0])]);
        // The same name again without listening: nothing changes.
        events.push(confirm("room/S1", "Max"));
        let voices = teach(&events, &s);
        assert_eq!(learned(&voices), [(Some((0, 4_000)), vec![1.0, 0.0])]);
        assert_eq!(voices[0].from, 2, "still the voice the second event taught");
    }

    #[test]
    fn line_names_follow_lines_derived_anew() {
        let mut s = Structure {
            lines: vec![
                line("room", 0, 2_000, "room/S1"),
                line("room", 2_000, 5_000, "room/S1"),
                line("room", 5_000, 6_000, "room/S2"),
            ],
            ..Default::default()
        };
        let mut events = vec![
            confirm("room/S1", "Max"),
            name_line(2_000, 5_000, Some(" Alice ")),
            name_line(5_000, 6_000, Some("?")),
            Event::LineNamed {
                track: "remote".into(),
                span: Span {
                    start_ms: 0,
                    end_ms: 2_000,
                },
                answer: Answer::parse(Some("Bob")),
            },
        ];
        let want = [(Some("Max"), Guessed), (Some("Alice"), Taught), (None, Unknown)];
        assert_eq!(shown(&resolve(&known(&events, std::mem::take(&mut s), &[]), &R)), want);
        // Derived anew: cut differently, the names go by midpoint.
        s.lines = vec![
            line("room", 0, 2_500, "room/S1"),
            line("room", 2_500, 4_800, "room/S2"),
            line("room", 4_800, 6_200, "room/S2"),
        ];
        let r = resolve(&known(&events, std::mem::take(&mut s), &[]), &R);
        assert_eq!(shown(&r), want);
        s = r.lines.into_iter().map(|l| l.line).collect::<Vec<_>>().into();
        // Naming the new line replaces the name it overlaps; clearing leaves the label's.
        events.push(name_line(2_500, 4_800, Some("Carol")));
        events.push(name_line(4_800, 6_200, None));
        let r = resolve(&known(&events, s, &[]), &R);
        assert_eq!(shown(&r), [(Some("Max"), Guessed), (Some("Carol"), Taught), (None, Unknown)]);
        assert_eq!(fold(&log(&events)).line_names.len(), 2, "Carol, and Bob on the remote track");
        assert_eq!(r.attendees, ["Max", "Carol"]);
    }

    impl From<Vec<Line>> for Structure {
        fn from(lines: Vec<Line>) -> Self {
            Self {
                lines,
                ..Default::default()
            }
        }
    }

    #[test]
    fn a_line_name_teaches_one_voice_replaced_with_it() {
        let mut s = Structure {
            turns: vec![turn(0, 4_000, &[1.0, 0.0]), turn(4_000, 8_000, &[0.0, 1.0])],
            ..Default::default()
        };
        let voices = |events: &[Event], s: &Structure| -> Vec<(String, String, i64)> {
            teach(events, s)
                .iter()
                .map(|v| (v.name.clone(), v.source.label().to_string(), v.source.span().unwrap().start_ms))
                .collect()
        };
        // How many voices the last event taught: what the API answers.
        let taught = |events: &[Event], s: &Structure| {
            let seq = events.len() as i64;
            teach(events, s).iter().filter(|v| v.from == seq).count()
        };
        let v = |n: &str, start| (n.to_string(), "room".to_string(), start);
        let mut events = vec![name_line(0, 4_000, Some("Alice"))];
        assert_eq!(taught(&events, &s), 1);
        events.push(name_line(4_000, 8_000, Some("Bob")));
        assert_eq!(taught(&events, &s), 1);
        assert_eq!(learned(&teach(&events, &s))[0].1, [1.0, 0.0]);
        // Renamed: its voice goes with the old name.
        events.push(name_line(0, 4_000, Some("Carol")));
        assert_eq!(taught(&events, &s), 1);
        assert_eq!(voices(&events, &s), [v("Bob", 4_000), v("Carol", 0)]);
        // Unsure, or cleared: no voice.
        events.push(name_line(0, 4_000, Some("?")));
        assert_eq!(taught(&events, &s), 0);
        events.push(name_line(4_000, 8_000, Some("")));
        assert_eq!(taught(&events, &s), 0);
        assert_eq!(voices(&events, &s), []);
        // A label confirmed later leaves line voices alone.
        events.push(name_line(4_000, 8_000, Some("Bob")));
        s.clusters = vec![cluster("room/S1", &[0.6, 0.8])];
        events.push(confirm("room/S1", "Max"));
        events.push(confirm("room/S1", ""));
        assert_eq!(voices(&events, &s), [v("Bob", 4_000)]);
        // A heard snippet named otherwise unlearns its voice.
        events.push(heard("room/S1", "Max", &[(0, 4_000, true)]));
        events.push(name_line(0, 4_000, Some("Alice")));
        assert_eq!(voices(&events, &s), [v("Bob", 4_000), v("Alice", 0)]);
        events.push(name_line(0, 4_000, None));
        assert_eq!(voices(&events, &s), [v("Bob", 4_000)], "and no core comes back");
        // Once the turns expired the name is still kept.
        s.turns.clear();
        assert!(!resolve(&known(&events, Structure::default(), &[]), &R).teachable);
        events.push(name_line(0, 4_000, Some("Alice")));
        assert_eq!(taught(&events, &s), 0);
        assert_eq!(fold(&log(&events)).line_names.len(), 2);
    }

    #[test]
    fn a_label_answered_unsure_is_confirmed_without_a_name_or_voice() {
        let library = [lib("Max", "room/S1", &[1.0, 0.0])];
        let s = || Structure {
            clusters: vec![cluster("room/S1", &[1.0, 0.0])],
            turns: vec![turn(0, 4_000, &[1.0, 0.0])],
            lines: vec![line("room", 0, 4_000, "room/S1")],
            ..Default::default()
        };
        let r = resolve(&known(&[], s(), &library), &R);
        assert_eq!(r.speakers["room/S1"].suggested.as_deref(), Some("Max"));
        let events = [heard("room/S1", "?", &[(0, 4_000, true)])];
        assert_eq!(teach(&events, &s()), []);
        let r = resolve(&known(&events, s(), &library), &R);
        assert_eq!(r.speakers["room/S1"].suggested, None, "not suggested again");
        assert_eq!(r.speakers["room/S1"].name.as_deref(), Some("?"));
        assert_eq!(shown(&r), [(None, Unknown)]);
        assert!(r.attendees.is_empty());
    }

    #[test]
    fn a_name_counts_once_per_recording() {
        let v = |r: &str, n: &str, e: &[f32]| Voice {
            recording: r.into(),
            ..lib(n, "room/S1", e)
        };
        // Max taught five lines in one meeting, one of them wrongly Alice's; Alice taught one.
        let mut voices = vec![v("a", "Max", &[1.0, 0.0]); 4];
        voices.push(v("a", "Max", &[0.0, 1.0]));
        voices.push(v("b", "Alice", &[0.1, 1.0]));
        let pooled = pool(&voices);
        assert_eq!(pooled.len(), 2);
        let all: Vec<_> = voices.iter().map(|v| (v.name.clone(), v.embedding.clone())).collect();
        assert_eq!(best(&[0.0, 1.0], &all, &M), None, "the wrong voice ties with Alice");
        assert_eq!(best(&[0.0, 1.0], &pooled, &M).map(|b| b.0), Some("Alice"));
        assert_eq!(best(&[1.0, 0.0], &pooled, &M).map(|b| b.0), Some("Max"));
    }

    #[test]
    fn guesses_from_the_label_and_the_line_s_own_voice() {
        let voices = [("Max".to_string(), vec![1.0, 0.0]), ("Alice".to_string(), vec![0.0, 1.0])];
        let max = Some(&[0.9, 0.1][..]);
        let unclear = Some(&[0.7, 0.7][..]);
        assert_eq!(guess(None, max, &voices, &L), Some("Max"), "own alone");
        assert_eq!(guess(Some("Max"), max, &voices, &L), Some("Max"), "both agree");
        assert_eq!(guess(Some("Alice"), max, &voices, &L), None, "they disagree");
        assert_eq!(guess(Some("Alice"), unclear, &voices, &L), Some("Alice"), "own not sure");
        assert_eq!(guess(Some("Alice"), None, &voices, &L), Some("Alice"), "no line voice");
        assert_eq!(guess(None, unclear, &voices, &L), None);
        assert_eq!(guess(None, None, &voices, &L), None);
        assert_eq!(guess(Some("Bob"), max, &[], &L), Some("Bob"), "no voices");
    }

    #[test]
    fn a_line_is_taught_guessed_or_unknown() {
        let mut library = vec![lib("Max", "room/S1", &[1.0, 0.0, 0.0]), lib("Alice", "room/S2", &[0.0, 1.0, 0.0])];
        let clusters = || {
            vec![
                cluster("room/S1", &[1.0, 0.0, 0.0]),
                cluster("room/S2", &[0.0, 1.0, 0.0]), // suggested Alice
                cluster("room/S3", &[0.5, 0.5, 0.7]),
                cluster("room/S4", &[0.5, 0.5, 0.7]),
            ]
        };
        let lines = vec![
            line("room", 0, 1_000, "room/S1"),     // Max's label, own voice Max
            line("room", 1_000, 2_000, "room/S1"), // Max's label, own voice Alice: check it
            line("room", 2_000, 3_000, "room/S1"), // own voice unclear: the label's
            line("room", 3_000, 4_000, "room/S2"), // suggested Alice, no line voice
            line("room", 4_000, 5_000, "room/S3"), // answered ?, own voice Alice
            line("room", 5_000, 6_000, "room/S3"), // answered ?, no line voice
            line("room", 6_000, 7_000, "room/S4"), // unnamed, own voice Alice
            line("room", 7_000, 8_000, "room/S1"), // named Carol on its own
            line("room", 8_000, 9_000, "room/S1"), // named ?, own voice Max
        ];
        let line_voices = [
            (0, [0.9, 0.1, 0.0]),
            (1_000, [0.1, 0.9, 0.0]),
            (2_000, [0.6, 0.6, 0.5]),
            (4_000, [0.0, 1.0, 0.0]),
            (6_000, [0.0, 1.0, 0.0]),
            (7_000, [0.0, 0.0, 1.0]),
            (8_000, [1.0, 0.0, 0.0]),
        ]
        .map(|(s, v)| lv(s, s + 1_000, &v));
        let s = || Structure {
            lines: lines.clone(),
            clusters: clusters(),
            line_voices: line_voices.to_vec(),
            ..Default::default()
        };
        let mut events = vec![
            confirm("room/S1", "Max"),
            confirm("room/S3", "?"),
            name_line(7_000, 8_000, Some("Carol")),
            name_line(8_000, 9_000, Some("?")),
        ];
        assert_eq!(
            shown(&resolve(&known(&events, s(), &library), &R)),
            [
                (Some("Max"), Guessed),
                (None, Unknown),
                (Some("Max"), Guessed),
                (Some("Alice"), Guessed),
                (Some("Alice"), Guessed),
                (None, Unknown),
                (Some("Alice"), Guessed),
                (Some("Carol"), Taught),
                (None, Unknown),
            ]
        );
        // Heard before its label was confirmed: taught, whatever its own voice says.
        events.push(heard("room/S1", "Max", &[(1_000, 2_000, true)]));
        assert_eq!(shown(&resolve(&known(&events, s(), &library), &R))[1], (Some("Max"), Taught));
        // Guessed anew on every read: a voice learned later that the line's own voice matches
        // better than its label's name makes it one to check.
        library.push(lib("Bob", "remote/S9", &[0.6, 0.6, 0.5]));
        assert_eq!(shown(&resolve(&known(&events, s(), &library), &R))[2], (None, Unknown));
        // Derived anew, cut differently: line voices go by midpoint.
        let recut = Structure {
            lines: vec![line("room", 0, 1_200, "room/S1"), line("room", 1_200, 1_600, "room/S2")],
            ..s()
        };
        assert_eq!(
            shown(&resolve(&known(&events, recut, &library), &R)),
            [(Some("Max"), Guessed), (Some("Max"), Taught)]
        );
    }

    #[test]
    fn teaching_a_line_learns_its_line_voice_over_the_turns() {
        let mut s = Structure {
            clusters: vec![cluster("room/S1", &[0.6, 0.8])],
            turns: vec![turn(0, 8_000, &[1.0, 0.0])],
            line_voices: vec![lv(0, 4_000, &[0.0, 1.0])],
            ..Default::default()
        };
        let mut events = vec![
            name_line(0, 4_000, Some("Alice")),
            name_line(4_000, 8_000, Some("Bob")),
            heard("room/S1", "Max", &[(0, 4_000, true), (4_000, 8_000, true)]),
        ];
        let got: Vec<Vec<f32>> = learned(&teach(&events, &s)).into_iter().map(|v| v.1).collect();
        assert_eq!(got, [vec![0.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0]]);
        // Its turns expired, a line voice still teaches.
        s.turns.clear();
        events.push(name_line(0, 4_000, Some("Carol")));
        let voices = teach(&events, &s);
        assert_eq!(voices.iter().filter(|v| v.from == 4).count(), 1);
        assert!(resolve(&known(&events, s, &[]), &R).teachable);
    }

    #[test]
    fn naming_lines_splits_a_cluster_of_two_people() {
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
        let s = || Structure {
            lines: (0..own.len() as i64)
                .map(|i| line("room", i * 1_000, i * 1_000 + 1_000, "room/S1"))
                .collect(),
            clusters: vec![cluster("room/S1", &[0.6, 0.8])],
            turns: vec![
                turn(0, 1_000, &[1.0, 0.0]),
                turn(1_000, 5_000, &[0.0, 1.0]),
                turn(5_000, 7_000, &[1.0, 0.0]),
            ],
            line_voices: own
                .iter()
                .enumerate()
                .map(|(i, v)| lv(i as i64 * 1_000, i as i64 * 1_000 + 1_000, v))
                .collect(),
        };
        let states = |events: &[Event]| -> Vec<(Option<String>, State)> {
            let r = resolve(&known(events, s(), &[]), &R);
            shown(&r)[4..].iter().map(|(n, st)| (n.map(String::from), *st)).collect()
        };
        let st = |n: Option<&str>, state| (n.map(String::from), state);
        // Erin taught twice: lines sounding like him only disagree with the label.
        let mut events = vec![
            heard("room/S1", "Dave", &[(5_000, 6_000, false)]),
            name_line(1_000, 2_000, Some("Erin")),
            name_line(2_000, 3_000, Some("Erin")),
        ];
        assert_eq!(
            states(&events),
            [st(None, Unknown), st(Some("Dave"), Guessed), st(None, Unknown)]
        );
        // A third time splits the cluster: each line goes to the nearer of the two, the unclear
        // one to neither.
        events.push(name_line(3_000, 4_000, Some("Erin")));
        assert_eq!(
            states(&events),
            [st(Some("Erin"), Guessed), st(Some("Dave"), Guessed), st(None, Unknown)]
        );
        // Cleared again: no split.
        events.push(name_line(3_000, 4_000, None));
        assert_eq!(states(&events)[0], st(None, Unknown));
    }

    #[test]
    fn a_cluster_teaches_the_turns_no_line_of_it_is_named_otherwise() {
        let s = Structure {
            clusters: vec![cluster("room/S1", &[0.6, 0.8])],
            turns: vec![turn(0, 4_000, &[1.0, 0.0]), turn(4_000, 8_000, &[0.0, 1.0])],
            lines: vec![line("room", 0, 4_000, "room/S1"), line("room", 4_000, 8_000, "room/S1")],
            ..Default::default()
        };
        let core = |events: &[Event]| {
            teach(events, &s)
                .into_iter()
                .find(|v| matches!(v.source, Source::Core { .. }))
                .unwrap()
                .embedding
        };
        let mut events = vec![confirm("room/S1", "Max")];
        assert_eq!(core(&events), [0.6, 0.8], "nothing named otherwise: the stored core");
        events.push(name_line(4_000, 8_000, Some("Alice")));
        assert_eq!(core(&events), [1.0, 0.0], "Alice's turn left out");
        events.push(name_line(4_000, 8_000, Some("Max")));
        assert_eq!(core(&events), [0.6, 0.8]);
        events.push(name_line(4_000, 8_000, Some("?")));
        assert_eq!(core(&events), [1.0, 0.0], "an unsure line's turn left out too");
        // Confirmed anew, it learns the clean core right away.
        events.push(confirm("room/S1", "Carol"));
        assert_eq!(core(&events), [1.0, 0.0]);
    }

    #[test]
    fn rediarizing_forgets_the_labels_but_not_the_line_names() {
        let s = Structure {
            clusters: vec![cluster("room/S1", &[1.0, 0.0])],
            lines: vec![line("room", 0, 2_000, "room/S1"), line("room", 2_000, 4_000, "room/S1")],
            line_voices: vec![lv(2_000, 4_000, &[0.0, 1.0])],
            ..Default::default()
        };
        let events = [
            confirm("room/S1", "Max"),
            name_line(2_000, 4_000, Some("Alice")),
            Event::Rediarized,
        ];
        let voices = teach(&events, &s);
        let names: Vec<&str> = voices.iter().map(|v| v.name.as_str()).collect();
        assert_eq!(names, ["Alice"]);
        let r = resolve(&known(&events, s, &[]), &R);
        assert_eq!(r.speakers["room/S1"].name, None);
        assert_eq!(shown(&r), [(None, Unknown), (Some("Alice"), Taught)]);
    }

    #[test]
    fn the_log_and_the_cache_round_trip_through_sqlite() {
        let db = crate::db::open(std::path::Path::new(":memory:")).unwrap();
        crate::db::ensure_recording(&db, "r1", "laptop").unwrap();
        db.execute_batch(
            "INSERT INTO clusters (recording, label, embedding) VALUES ('r1', 'room/S1', x'0000803f00000000');
             INSERT INTO lines (recording, track, start_ms, end_ms, text, speaker) VALUES ('r1', 'room', 0, 2000, 'a', 'room/S1');
             INSERT INTO line_voices (recording, track, start_ms, end_ms, embedding) VALUES ('r1', 'room', 0, 2000, x'000000000000803f');",
        )
        .unwrap();
        let events = [confirm("room/S1", "Max"), name_line(0, 2_000, Some("Alice"))];
        assert_eq!(record(&db, "r1", &events).unwrap(), [1, 1]);
        let library = library(&db).unwrap();
        let got: Vec<(&str, &Source, i64)> = library.iter().map(|v| (v.name.as_str(), &v.source, v.from)).collect();
        let eva = Source::Line {
            track: "room".into(),
            span: Span {
                start_ms: 0,
                end_ms: 2_000,
            },
        };
        assert_eq!(
            got,
            [("Max", &Source::Core { label: "room/S1".into() }, 1), ("Alice", &eva, 2)]
        );
        let k = load(&db, "r1", &library).unwrap();
        let logged: Vec<&Event> = k.events.iter().map(|l| &l.event).collect();
        assert_eq!(logged, events.iter().collect::<Vec<_>>());
        assert_eq!(k.structure.lines.len(), 1);
        // The cache is only that: deleted, project brings it back.
        db.execute("DELETE FROM voices", []).unwrap();
        project(&db, "r1").unwrap();
        assert_eq!(super::library(&db).unwrap(), library);
        // The Known is what the replay harness reads.
        let json = serde_json::to_string(&k).unwrap();
        let back: Known = serde_json::from_str(&json).unwrap();
        assert_eq!(back.library.len(), 2);
        assert_eq!(resolve(&back, &R).attendees, ["Max", "Alice"]);
    }
}
