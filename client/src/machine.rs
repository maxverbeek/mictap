use std::time::{Duration, Instant};

use serde::Serialize;

use crate::pw::Graph;

pub const GRACE: Duration = Duration::from_secs(120);
/// How long a meeting's capture stream must exist before recording from it. Opening the mic
/// switches a Bluetooth headset to its call profile, and a recorder linked during that switch
/// keeps the headset from recovering, which stalls the app's audio.
pub const SETTLE: Duration = Duration::from_secs(10);

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Auto,
    Manual,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Track {
    /// "mic" or "app-monitor"
    pub key: String,
    /// What `pw-record --target` gets: a source or sink node.name.
    pub target: String,
}

#[derive(Debug)]
pub struct Session {
    pub mode: Mode,
    pub source: Option<String>,
    pub app: Option<String>,
    pub lost_at: Option<Instant>,
    pub tracks: Vec<Track>,
}

impl Session {
    fn new(mode: Mode, source: Option<String>) -> Self {
        Self {
            mode,
            source,
            app: None,
            lost_at: None,
            tracks: vec![],
        }
    }
}

#[derive(Debug, Default)]
pub struct Machine {
    pub session: Option<Session>,
    meeting: Option<u32>,
    /// The capture stream last seen, and since when.
    seen: Option<(u32, Instant)>,
    suppressed: Option<u32>,
}

impl Machine {
    pub fn mode(&self) -> Option<Mode> {
        self.session.as_ref().map(|s| s.mode)
    }

    pub fn app(&self) -> Option<&str> {
        self.session.as_ref()?.app.as_deref()
    }

    pub fn start(&mut self, source: Option<String>) {
        match &mut self.session {
            Some(s) => {
                s.mode = Mode::Manual;
                s.lost_at = None;
                if source.is_some() {
                    s.source = source;
                }
            }
            None => self.session = Some(Session::new(Mode::Manual, source)),
        }
    }

    pub fn stop(&mut self) {
        self.session = None;
        self.suppressed = self.meeting;
    }

    /// The tracks that should be recording now, or None when idle.
    pub fn tick(&mut self, g: &Graph, now: Instant) -> Option<Vec<Track>> {
        let meeting = g.meeting();
        self.meeting = meeting.as_ref().map(|m| m.stream);
        if self.suppressed != self.meeting {
            self.suppressed = None;
        }
        if self.seen.map(|(id, _)| id) != self.meeting {
            self.seen = self.meeting.map(|id| (id, now));
        }
        let settled = self.seen.is_some_and(|(_, t)| now.duration_since(t) >= SETTLE);
        if self.session.is_none() && settled && self.suppressed.is_none() {
            self.session = Some(Session::new(Mode::Auto, None));
        }
        let s = self.session.as_mut()?;
        match &meeting {
            Some(m) => {
                s.lost_at = None;
                s.app = Some(m.app.clone());
            }
            None if s.mode == Mode::Auto && now.duration_since(*s.lost_at.get_or_insert(now)) >= GRACE => {
                self.session = None;
                return None;
            }
            None => {}
        }
        let mic = s.tracks.iter().find(|t| t.key == "mic").map(|t| t.target.clone());
        let source = s
            .source
            .clone()
            .or_else(|| meeting.as_ref()?.source.clone())
            .or(mic)
            .or_else(|| g.default_source.clone());
        let mut tracks: Vec<Track> = source
            .map(|target| Track {
                key: "mic".into(),
                target,
            })
            .into_iter()
            .collect();
        if let Some(target) = meeting.as_ref().filter(|_| settled).and_then(|m| g.sink(m)) {
            tracks.push(Track {
                key: "app-monitor".into(),
                target,
            });
        }
        s.tracks = tracks.clone();
        Some(tracks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pw::Node;

    fn node(id: u32, class: &str, name: &str, app: &str) -> Node {
        Node {
            id,
            class: class.into(),
            name: name.into(),
            description: String::new(),
            app: app.into(),
            binary: app.to_lowercase(),
        }
    }

    fn quiet() -> Graph {
        Graph {
            nodes: vec![node(1, "Audio/Source", "mic1", "")],
            links: vec![],
            default_source: Some("mic1".into()),
            default_sink: Some("speaker".into()),
        }
    }

    fn meeting(source: &str) -> Graph {
        Graph {
            nodes: vec![
                node(1, "Audio/Source", "mic1", ""),
                node(2, "Audio/Source", "headset", ""),
                node(10, "Stream/Input/Audio", "", "Zen"),
                node(11, "Stream/Output/Audio", "", "Zen"),
            ],
            links: vec![(if source == "mic1" { 1 } else { 2 }, 10)],
            default_source: Some("mic1".into()),
            default_sink: Some("speaker".into()),
        }
    }

    fn keys(t: Option<Vec<Track>>) -> Vec<String> {
        t.unwrap().iter().map(|t| format!("{}={}", t.key, t.target)).collect()
    }

    fn secs(s: u64) -> Duration {
        Duration::from_secs(s)
    }

    #[test]
    fn auto_starts_once_settled_follows_source_and_stops_after_grace() {
        let (mut m, t0) = (Machine::default(), Instant::now());
        assert_eq!(m.tick(&quiet(), t0), None);
        assert_eq!(
            m.tick(&meeting("headset"), t0),
            None,
            "the headset may be switching profiles"
        );
        assert_eq!(m.tick(&meeting("headset"), t0 + secs(9)), None);
        let t = t0 + SETTLE;
        assert_eq!(
            keys(m.tick(&meeting("headset"), t)),
            ["mic=headset", "app-monitor=speaker"]
        );
        assert_eq!(m.mode(), Some(Mode::Auto));
        assert_eq!(keys(m.tick(&meeting("mic1"), t)), ["mic=mic1", "app-monitor=speaker"]);
        assert_eq!(keys(m.tick(&quiet(), t + secs(1))), ["mic=mic1"]);
        assert!(m.tick(&quiet(), t + secs(120)).is_some());
        assert_eq!(m.tick(&quiet(), t + secs(121)), None);
    }

    #[test]
    fn meeting_returning_within_grace_keeps_the_mic_and_waits_to_settle_for_the_rest() {
        let (mut m, t0) = (Machine::default(), Instant::now());
        m.tick(&meeting("mic1"), t0);
        let t = t0 + SETTLE;
        m.tick(&meeting("mic1"), t);
        m.tick(&quiet(), t + secs(1));
        assert_eq!(keys(m.tick(&meeting("mic1"), t + secs(100))), ["mic=mic1"]);
        assert_eq!(
            keys(m.tick(&meeting("mic1"), t + secs(110))),
            ["mic=mic1", "app-monitor=speaker"]
        );
        assert!(m.tick(&quiet(), t + secs(200)).is_some());
    }

    #[test]
    fn stop_suppresses_until_the_meeting_stream_goes_away() {
        let (mut m, t0) = (Machine::default(), Instant::now());
        m.tick(&meeting("mic1"), t0);
        let t = t0 + SETTLE;
        m.tick(&meeting("mic1"), t);
        m.stop();
        assert_eq!(m.tick(&meeting("mic1"), t), None);
        assert_eq!(m.tick(&quiet(), t), None);
        m.tick(&meeting("mic1"), t);
        assert!(m.tick(&meeting("mic1"), t + SETTLE).is_some());
    }

    #[test]
    fn manual_starts_at_once_and_never_times_out() {
        let (mut m, t0) = (Machine::default(), Instant::now());
        m.start(None);
        assert_eq!(keys(m.tick(&quiet(), t0)), ["mic=mic1"]);
        assert_eq!(m.mode(), Some(Mode::Manual));
        m.start(Some("headset".into()));
        assert_eq!(keys(m.tick(&meeting("mic1"), t0)), ["mic=headset"]);
        assert_eq!(
            keys(m.tick(&meeting("mic1"), t0 + SETTLE)),
            ["mic=headset", "app-monitor=speaker"]
        );
        assert!(m.tick(&quiet(), t0 + secs(600)).is_some());
    }

    #[test]
    fn start_during_grace_clears_the_countdown() {
        let (mut m, t0) = (Machine::default(), Instant::now());
        m.tick(&meeting("mic1"), t0);
        m.tick(&meeting("mic1"), t0 + SETTLE);
        m.tick(&quiet(), t0 + secs(11));
        m.start(None);
        assert!(m.tick(&quiet(), t0 + secs(12)).is_some());
        assert_eq!(m.session.as_ref().unwrap().lost_at, None);
    }
}
