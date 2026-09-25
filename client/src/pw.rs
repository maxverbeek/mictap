use anyhow::{ensure, Result};
use serde_json::Value;

/// Matched case-insensitively against application.name and .process.binary.
/// MICTAP_ALLOWLIST (comma-separated) replaces it.
const ALLOWLIST: &str = "zen,chromium,chrome,zoom,slack";

#[derive(Debug, Clone)]
pub struct Node {
    pub id: u32,
    pub serial: u64,
    pub class: String,
    pub name: String,
    pub description: String,
    pub app: String,
    pub binary: String,
}

#[derive(Debug, Default)]
pub struct Graph {
    pub nodes: Vec<Node>,
    /// (output node id, input node id)
    pub links: Vec<(u32, u32)>,
    pub default_source: Option<String>,
}

#[derive(Debug)]
pub struct Meeting {
    /// Node id of the app's capture stream.
    pub stream: u32,
    pub app: String,
    pub binary: String,
    /// node.name of the source the app captures from, once linked.
    pub source: Option<String>,
}

pub async fn dump() -> Result<Graph> {
    let out = tokio::process::Command::new("pw-dump").output().await?;
    ensure!(out.status.success(), "pw-dump exited with {}", out.status);
    parse(std::str::from_utf8(&out.stdout)?)
}

pub fn parse(json: &str) -> Result<Graph> {
    let objs: Vec<Value> = serde_json::from_str(json)?;
    let s = |v: &Value| v.as_str().unwrap_or("").to_string();
    let id = |v: &Value| v.as_u64().unwrap_or(0) as u32;
    let mut g = Graph::default();
    for o in &objs {
        match o["type"].as_str() {
            Some("PipeWire:Interface:Node") => {
                let p = &o["info"]["props"];
                g.nodes.push(Node {
                    id: id(&o["id"]),
                    serial: p["object.serial"].as_u64().unwrap_or(0),
                    class: s(&p["media.class"]),
                    name: s(&p["node.name"]),
                    description: s(&p["node.description"]),
                    app: s(&p["application.name"]),
                    binary: s(&p["application.process.binary"]),
                });
            }
            Some("PipeWire:Interface:Link") => {
                let i = &o["info"];
                g.links.push((id(&i["output-node-id"]), id(&i["input-node-id"])));
            }
            Some("PipeWire:Interface:Metadata") if o["props"]["metadata.name"] == "default" => {
                for m in o["metadata"].as_array().into_iter().flatten() {
                    if m["key"] == "default.audio.source" {
                        g.default_source = m["value"]["name"].as_str().map(String::from);
                    }
                }
            }
            _ => {}
        }
    }
    Ok(g)
}

fn allowed(n: &Node) -> bool {
    let (app, bin) = (n.app.to_lowercase(), n.binary.to_lowercase());
    let list = std::env::var("MICTAP_ALLOWLIST").unwrap_or_else(|_| ALLOWLIST.into());
    list.split(',')
        .map(|w| w.trim().to_lowercase())
        .any(|w| !w.is_empty() && (app.contains(&w) || bin.contains(&w)))
}

impl Graph {
    pub fn meeting(&self) -> Option<Meeting> {
        let n = self
            .nodes
            .iter()
            .find(|n| n.class == "Stream/Input/Audio" && allowed(n))?;
        let source = self
            .links
            .iter()
            .filter(|(_, input)| *input == n.id)
            .find_map(|(output, _)| self.nodes.iter().find(|s| s.id == *output))
            .map(|s| s.name.clone());
        Some(Meeting {
            stream: n.id,
            app: n.app.clone(),
            binary: n.binary.clone(),
            source,
        })
    }

    pub fn playbacks(&self, m: &Meeting) -> Vec<&Node> {
        self.nodes
            .iter()
            .filter(|n| n.class == "Stream/Output/Audio")
            .filter(|n| {
                if m.binary.is_empty() {
                    n.app == m.app
                } else {
                    n.binary == m.binary
                }
            })
            .collect()
    }

    pub fn sources(&self) -> Vec<&Node> {
        self.nodes
            .iter()
            .filter(|n| n.class.starts_with("Audio/Source"))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DUMP: &str = r#"[
      {"id": 30, "type": "PipeWire:Interface:Metadata", "props": {"metadata.name": "default"},
       "metadata": [{"subject": 0, "key": "default.audio.source", "type": "Spa:String:JSON", "value": {"name": "mic1"}}]},
      {"id": 57, "type": "PipeWire:Interface:Node", "info": {"props": {"media.class": "Audio/Source", "node.name": "mic1", "node.description": "Digital Microphone", "object.serial": 57}}},
      {"id": 58, "type": "PipeWire:Interface:Node", "info": {"props": {"media.class": "Audio/Source", "node.name": "headset", "node.description": "Headset", "object.serial": 58}}},
      {"id": 90, "type": "PipeWire:Interface:Node", "info": {"props": {"media.class": "Stream/Input/Audio", "application.name": "clankertyper", "object.serial": 400}}},
      {"id": 98, "type": "PipeWire:Interface:Node", "info": {"props": {"media.class": "Stream/Input/Audio", "application.name": "Zen", "application.process.binary": "zen", "object.serial": 426}}},
      {"id": 99, "type": "PipeWire:Interface:Node", "info": {"props": {"media.class": "Stream/Output/Audio", "application.name": "Zen", "application.process.binary": "zen", "object.serial": 427}}},
      {"id": 100, "type": "PipeWire:Interface:Node", "info": {"props": {"media.class": "Stream/Output/Audio", "application.name": "spotify", "application.process.binary": "spotify", "object.serial": 428}}},
      {"id": 120, "type": "PipeWire:Interface:Link", "info": {"output-node-id": 58, "input-node-id": 98}},
      {"id": 121, "type": "PipeWire:Interface:Link", "info": {"output-node-id": 57, "input-node-id": 90}}
    ]"#;

    #[test]
    fn finds_allowlisted_capture_its_source_and_playback() {
        let g = parse(DUMP).unwrap();
        let m = g.meeting().unwrap();
        assert_eq!(
            (m.stream, m.app.as_str(), m.source.as_deref()),
            (98, "Zen", Some("headset"))
        );
        assert_eq!(g.playbacks(&m).iter().map(|n| n.serial).collect::<Vec<_>>(), [427]);
        assert_eq!(g.default_source.as_deref(), Some("mic1"));
        assert_eq!(
            g.sources().iter().map(|n| n.name.as_str()).collect::<Vec<_>>(),
            ["mic1", "headset"]
        );
    }

    #[test]
    fn ignores_apps_off_the_allowlist() {
        let g = parse(&DUMP.replace("Zen", "Other").replace("zen", "other")).unwrap();
        assert!(g.meeting().is_none());
    }
}
