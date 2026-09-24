use std::{
    collections::{BTreeMap, HashMap},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime},
};

use anyhow::Result;
use rusqlite::{params, OptionalExtension};
use serde_json::Value;

use crate::{
    api::App,
    vault::{label_order, put},
};

type Names = BTreeMap<String, String>;

/// Splits `---\n<front>---\n<body>` into front (with its last newline) and body.
fn split(text: &str) -> Option<(&str, &str)> {
    let rest = text.strip_prefix("---\n")?;
    let end = rest.find("\n---\n")? + 1;
    Some((&rest[..end], &rest[end + 4..]))
}

/// `id` and the non-empty names of `speakers`, whatever YAML style Obsidian saved them in.
fn parse(front: &str) -> Result<Option<(String, Names)>> {
    let yaml: serde_norway::Value = serde_norway::from_str(front)?;
    let Some(id) = yaml["id"].as_str() else {
        return Ok(None);
    };
    let names = yaml["speakers"]
        .as_mapping()
        .into_iter()
        .flatten()
        .filter_map(|(k, v)| Some((k.as_str()?.to_string(), v.as_str()?.trim().to_string())))
        .filter(|(_, n)| !n.is_empty())
        .collect();
    Ok(Some((id.to_string(), names)))
}

/// What a line of `label` shows: its name, else `S<n>` as the vault writer renders it.
fn display<'a>(label: &'a str, names: &'a Names) -> &'a str {
    names
        .get(label)
        .map_or_else(|| label.split_once('/').map_or(label, |(_, n)| n), String::as_str)
}

/// Relabels `**<old>** (<track>, ...` lines from the `applied` names to `names`.
fn relabel(body: &str, applied: &Names, names: &Names) -> String {
    let mut map: HashMap<(&str, &str), &str> = HashMap::new();
    for label in applied.keys().chain(names.keys()) {
        let track = label.split_once('/').map_or(label.as_str(), |(t, _)| t);
        let (old, new) = (display(label, applied), display(label, names));
        // Lines of two labels shown under one name can't be told apart: on conflict they keep it.
        map.entry((track, old))
            .and_modify(|n| {
                if *n != new {
                    *n = old;
                }
            })
            .or_insert(new);
    }
    body.split_inclusive('\n')
        .map(|line| {
            let relabeled = line
                .strip_prefix("**")
                .and_then(|r| r.split_once("** ("))
                .and_then(|(name, rest)| {
                    let track = rest.split_once(',')?.0;
                    let new = map.get(&(track, name))?;
                    Some(format!("**{new}** ({rest}"))
                });
            relabeled.unwrap_or_else(|| line.to_string())
        })
        .collect()
}

/// Replaces the `attendees` key (flow or block style) in `front`, or appends it.
fn set_attendees(front: &str, names: &Names) -> String {
    let mut labels: Vec<&String> = names.keys().collect();
    labels.sort_by_key(|l| label_order(l));
    let mut unique: Vec<&str> = vec![];
    for l in labels {
        if !unique.contains(&names[l].as_str()) {
            unique.push(&names[l]);
        }
    }
    let links: Vec<String> = unique
        .iter()
        .map(|n| Value::from(format!("[[{n}]]")).to_string())
        .collect();
    let attendees = format!("attendees: [{}]\n", links.join(", "));

    let mut out = String::new();
    let mut replaced = false;
    let mut lines = front.lines().peekable();
    while let Some(line) = lines.next() {
        if !replaced && line.starts_with("attendees:") {
            while lines.peek().is_some_and(|l| l.starts_with([' ', '\t', '-'])) {
                lines.next();
            }
            out += &attendees;
            replaced = true;
        } else {
            out += line;
            out += "\n";
        }
    }
    if !replaced {
        out += &attendees;
    }
    out
}

/// Applies a changed `speakers` map of the transcript at `path` (last modified at `mtime`).
async fn apply(app: &App, path: &Path, mtime: SystemTime) -> Result<()> {
    let text = std::fs::read_to_string(path)?;
    let Some((front, body)) = split(&text) else {
        return Ok(());
    };
    let Some((id, names)) = parse(front)? else {
        return Ok(());
    };
    let db = app.db.lock().await;
    let Some(applied) = db
        .query_row(
            "SELECT speakers FROM recordings WHERE id = ?1 AND written = 'done'",
            [&id],
            |r| r.get::<_, Option<String>>(0),
        )
        .optional()?
    else {
        return Ok(());
    };
    let applied: Names = applied
        .map(|s| serde_json::from_str(&s))
        .transpose()?
        .unwrap_or_default();
    if names == applied {
        return Ok(());
    }
    let content = format!(
        "---\n{}---\n{}",
        set_attendees(front, &names),
        relabel(body, &applied, &names)
    );
    // Edited since it was read: the next poll picks up the newer version.
    if std::fs::metadata(path)?.modified()? != mtime {
        return Ok(());
    }
    put(&app.vault, &id, Some(path), "", &content)?;
    db.execute(
        "UPDATE recordings SET speakers = ?2 WHERE id = ?1",
        params![id, serde_json::to_string(&names)?],
    )?;
    Ok(())
}

async fn scan(app: &App, seen: &mut HashMap<PathBuf, SystemTime>) -> std::io::Result<()> {
    let mut now = HashMap::new();
    for entry in std::fs::read_dir(&app.vault)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "md") {
            continue;
        }
        let mtime = entry.metadata()?.modified()?;
        if seen.get(&path) != Some(&mtime) {
            if let Err(e) = apply(app, &path, mtime).await {
                eprintln!("{}: speakers: {e:#}", path.display());
            }
        }
        now.insert(path, mtime);
    }
    *seen = now;
    Ok(())
}

/// Polls the vault for `speakers` edits.
pub async fn run(app: Arc<App>) {
    let mut seen = HashMap::new();
    loop {
        if let Err(e) = scan(&app, &mut seen).await {
            eprintln!("speakers: {e}");
        }
        tokio::time::sleep(Duration::from_secs(30)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../tests/fixtures/transcript.md");

    async fn setup() -> (tempfile::TempDir, App, PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let mut app = App::open(tmp.path().join("state")).unwrap();
        app.vault = tmp.path().join("vault");
        std::fs::create_dir_all(&app.vault).unwrap();
        app.db
            .lock()
            .await
            .execute(
                "INSERT INTO recordings (id, source, started_ms, written)
                 VALUES ('01J8XTEST', 'laptop', 0, 'done')",
                [],
            )
            .unwrap();
        let path = app.vault.join("Planning.md");
        (tmp, app, path)
    }

    /// Writes `content` as the user would, then runs one poll over it.
    async fn edit(app: &App, path: &Path, content: &str) -> String {
        std::fs::write(path, content).unwrap();
        let mtime = std::fs::metadata(path).unwrap().modified().unwrap();
        apply(app, path, mtime).await.unwrap();
        std::fs::read_to_string(path).unwrap()
    }

    fn sub(s: &str, pairs: &[(&str, &str)]) -> String {
        pairs.iter().fold(s.to_string(), |s, (a, b)| {
            assert!(s.contains(a), "{a:?} not in {s}");
            s.replacen(a, b, 1)
        })
    }

    #[tokio::test]
    async fn relabels_only_matching_lines_and_attendees() {
        let (_tmp, app, path) = setup().await;
        assert_eq!(edit(&app, &path, FIXTURE).await, FIXTURE, "nothing named yet");

        let named = sub(
            FIXTURE,
            &[
                ("  room/S1: \"\"", "  room/S1: Max"),
                ("  remote/S1: \"\"", "  remote/S1: Jan"),
            ],
        );
        let got = edit(&app, &path, &named).await;
        let want = sub(
            &named,
            &[
                ("attendees: []", "attendees: [\"[[Max]]\", \"[[Jan]]\"]"),
                ("**S1** (room, [00:00:02]", "**Max** (room, [00:00:02]"),
                ("**S1** (remote, [00:00:20]", "**Jan** (remote, [00:00:20]"),
                ("**S1** (room, [00:00:31]", "**Max** (room, [00:00:31]"),
            ],
        );
        assert_eq!(got, want);

        // Obsidian re-serializes the frontmatter: quoted keys, reordered keys, block lists.
        let (_, body) = split(&got).unwrap();
        let obsidian = format!(
            "---\nid: \"01J8XTEST\"\ndate: 2026-09-24 14:00\nduration: 1m\nsource: laptop\n\
             status: done\nprogress: 1/1 min\nattendees:\n  - \"[[Max]]\"\n  - \"[[Jan]]\"\n\
             speakers:\n  \"remote/S1\": Jan\n  \"room/S2\": Eva\n  'room/S1': Maxime\n\
             tags: [meeting]\n---\n{body}"
        );
        let got = edit(&app, &path, &obsidian).await;
        let want = sub(
            &obsidian,
            &[
                (
                    "attendees:\n  - \"[[Max]]\"\n  - \"[[Jan]]\"\n",
                    "attendees: [\"[[Maxime]]\", \"[[Eva]]\", \"[[Jan]]\"]\n",
                ),
                ("**Max** (room, [00:00:02]", "**Maxime** (room, [00:00:02]"),
                ("**S2** (room, [00:00:05]", "**Eva** (room, [00:00:05]"),
                ("**Max** (room, [00:00:31]", "**Maxime** (room, [00:00:31]"),
            ],
        );
        assert_eq!(got, want);

        // Clearing a name restores the label.
        let cleared = sub(&got, &[("  'room/S1': Maxime\n", "  room/S1:\n")]);
        let want = sub(
            &cleared,
            &[
                (
                    "attendees: [\"[[Maxime]]\", \"[[Eva]]\", \"[[Jan]]\"]",
                    "attendees: [\"[[Eva]]\", \"[[Jan]]\"]",
                ),
                ("**Maxime** (room, [00:00:02]", "**S1** (room, [00:00:02]"),
                ("**Maxime** (room, [00:00:31]", "**S1** (room, [00:00:31]"),
            ],
        );
        assert_eq!(edit(&app, &path, &cleared).await, want);
    }

    #[tokio::test]
    async fn ignores_unfinished_recordings() {
        let (_tmp, app, path) = setup().await;
        app.db
            .lock()
            .await
            .execute("UPDATE recordings SET written = 'transcribing 0/1 0'", [])
            .unwrap();
        let named = sub(FIXTURE, &[("  room/S1: \"\"", "  room/S1: Max")]);
        assert_eq!(edit(&app, &path, &named).await, named);
    }

    #[test]
    fn same_name_labels_keep_it_on_conflict() {
        let names =
            |pairs: &[(&str, &str)]| -> Names { pairs.iter().map(|(l, n)| (l.to_string(), n.to_string())).collect() };
        let applied = names(&[("room/S1", "Max"), ("room/S2", "Max")]);
        let body = "**Max** (room, x\n**S3** (room, y\n";
        assert_eq!(
            relabel(
                body,
                &applied,
                &names(&[("room/S1", "Max"), ("room/S2", "Eva"), ("room/S3", "Bo")])
            ),
            "**Max** (room, x\n**Bo** (room, y\n"
        );
        assert_eq!(
            relabel(body, &applied, &names(&[("room/S1", "Jo"), ("room/S2", "Jo")])),
            "**Jo** (room, x\n**S3** (room, y\n"
        );
    }
}
