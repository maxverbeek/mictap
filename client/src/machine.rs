use std::time::{Duration, Instant};

use serde::Serialize;

use crate::pw::Graph;

pub const GRACE: Duration = Duration::from_secs(120);

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Auto,
    Manual,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Track {
    /// "mic" or "app-<serial>"
    pub key: String,
    /// What `pw-record --target` gets: a node.name or an object.serial.
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
        Self { mode, source, app: None, lost_at: None, tracks: vec![] }
    }
}

#[derive(Debug, Default)]
pub struct Machine {
    pub session: Option<Session>,
    meeting: Option<u32>,
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
        if self.session.is_none() && self.meeting.is_some() && self.suppressed.is_none() {
            self.session = Some(Session::new(Mode::Auto, None));
        }
        let s = self.session.as_mut()?;
        match &meeting {
            Some(m) => {
                s.lost_at = None;
                s.app = Some(m.app.clone());
            }
            None if s.mode == Mode::Auto => {
                if now.duration_since(*s.lost_at.get_or_insert(now)) >= GRACE {
                    self.session = None;
                    return None;
                }
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
        let mut tracks: Vec<Track> = source.map(|target| Track { key: "mic".into(), target }).into_iter().collect();
        if let Some(m) = &meeting {
            tracks.extend(g.playbacks(m).into_iter().map(|n| Track {
                key: format!("app-{}", n.serial),
                target: n.serial.to_string(),
            }));
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
            serial: id as u64,
            class: class.into(),
            name: name.into(),
            description: String::new(),
            app: app.into(),
            binary: app.to_lowercase(),
        }
    }

    fn quiet() -> Graph {
        Graph { nodes: vec![node(1, "Audio/Source", "mic1", "")], links: vec![], default_source: Some("mic1".into()) }
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
        }
    }

    fn keys(t: Option<Vec<Track>>) -> Vec<String> {
        t.unwrap().iter().map(|t| format!("{}={}", t.key, t.target)).collect()
    }

    fn secs(s: u64) -> Duration {
        Duration::from_secs(s)
    }

    #[test]
    fn auto_starts_follows_source_and_stops_after_grace() {
        let (mut m, t0) = (Machine::default(), Instant::now());
        assert_eq!(m.tick(&quiet(), t0), None);
        assert_eq!(keys(m.tick(&meeting("headset"), t0)), ["mic=headset", "app-11=11"]);
        assert_eq!(m.mode(), Some(Mode::Auto));
        assert_eq!(keys(m.tick(&meeting("mic1"), t0)), ["mic=mic1", "app-11=11"]);
        assert_eq!(keys(m.tick(&quiet(), t0 + secs(1))), ["mic=mic1"]);
        assert!(m.tick(&quiet(), t0 + secs(120)).is_some());
        assert_eq!(m.tick(&quiet(), t0 + secs(121)), None);
    }

    #[test]
    fn meeting_returning_within_grace_keeps_recording() {
        let (mut m, t0) = (Machine::default(), Instant::now());
        m.tick(&meeting("mic1"), t0);
        m.tick(&quiet(), t0 + secs(1));
        m.tick(&meeting("mic1"), t0 + secs(100));
        assert!(m.tick(&quiet(), t0 + secs(200)).is_some());
    }

    #[test]
    fn stop_suppresses_until_the_meeting_stream_goes_away() {
        let (mut m, t0) = (Machine::default(), Instant::now());
        m.tick(&meeting("mic1"), t0);
        m.stop();
        assert_eq!(m.tick(&meeting("mic1"), t0), None);
        assert_eq!(m.tick(&quiet(), t0), None);
        assert!(m.tick(&meeting("mic1"), t0).is_some());
    }

    #[test]
    fn manual_uses_override_then_default_and_never_times_out() {
        let (mut m, t0) = (Machine::default(), Instant::now());
        m.start(None);
        assert_eq!(keys(m.tick(&quiet(), t0)), ["mic=mic1"]);
        assert_eq!(m.mode(), Some(Mode::Manual));
        m.start(Some("headset".into()));
        assert_eq!(keys(m.tick(&meeting("mic1"), t0)), ["mic=headset", "app-11=11"]);
        assert!(m.tick(&quiet(), t0 + secs(600)).is_some());
    }
}
