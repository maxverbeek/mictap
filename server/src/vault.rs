use std::{
    collections::BTreeMap,
    io::{ErrorKind, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use anyhow::Result;
use jiff::Timestamp;
use rusqlite::{params, Connection};
use serde_json::Value;

use crate::{api::App, assemble::Segment};

/// Base of the links in transcripts, as browsers reach this server.
pub static URL: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| std::env::var("MICTAP_URL").unwrap_or_else(|_| "http://localhost:8765".into()));

/// Speaker label (`room/S1`) -> name.
pub(crate) type Names = BTreeMap<String, String>;

fn hms(ms: i64) -> String {
    let s = ms / 1000;
    format!("{:02}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
}

/// `room/S2` sorts as (false, 2): room before remote, then by number.
pub(crate) fn label_order(label: &str) -> (bool, u32) {
    let (track, n) = label.split_once("/S").unwrap_or((label, ""));
    (track != "room", n.parse().unwrap_or(u32::MAX))
}

/// A speaker label: `room/S<n>` or `remote/S<n>`.
pub(crate) fn is_label(k: &str) -> bool {
    k.split_once('/').is_some_and(|(track, n)| {
        matches!(track, "room" | "remote")
            && n.strip_prefix('S')
                .is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
    })
}

/// What a line of `label` shows: its name, else `S<n>`.
fn display<'a>(label: &'a str, names: &'a Names) -> &'a str {
    names
        .get(label)
        .map_or_else(|| label.split_once('/').map_or(label, |(_, n)| n), String::as_str)
}

/// The transcript: a small frontmatter, then one line per segment.
fn render(id: &str, date: &str, segs: &[Segment], names: &Names) -> String {
    let mut labels: Vec<&String> = names.keys().collect();
    labels.sort_by_key(|l| label_order(l));
    let mut attendees: Vec<String> = vec![];
    for l in labels {
        let link = Value::from(format!("[[{}]]", names[l])).to_string();
        if !attendees.contains(&link) {
            attendees.push(link);
        }
    }
    let mut out = format!(
        "---\nid: {id}\ndate: {date}\nattendees: [{}]\nlink: {}/#{id}\n---\n\n",
        attendees.join(", "),
        *URL,
    );
    for s in segs {
        let name = s.speaker.as_deref().map_or("?", |l| display(l, names));
        out += &format!(
            "**{name}** ({}, [{}]({}/r/{id}/audio.ogg#t={})): {}\n",
            s.track,
            hms(s.start_ms),
            *URL,
            s.start_ms / 1000,
            s.text
        );
    }
    out
}

/// Whether `path` is a transcript whose frontmatter has `id: <id>`, quoted or not.
fn has_id(path: &Path, id: &str) -> bool {
    let Ok(text) = std::fs::read_to_string(path) else {
        return false;
    };
    let mut lines = text.lines().map(str::trim_end);
    lines.next() == Some("---")
        && lines.take_while(|l| *l != "---").any(|l| {
            l.strip_prefix("id:")
                .is_some_and(|v| v.trim().trim_matches(['"', '\'']) == id)
        })
}

/// The transcript of `id` in `vault`: the cached file name if it still holds it, else a
/// rescan of `*.md`.
fn locate(vault: &Path, id: &str, cached: Option<&str>) -> std::io::Result<Option<PathBuf>> {
    if let Some(p) = cached.map(|c| vault.join(c)).filter(|p| has_id(p, id)) {
        return Ok(Some(p));
    }
    for entry in std::fs::read_dir(vault)? {
        let p = entry?.path();
        if p.extension().is_some_and(|e| e == "md") && has_id(&p, id) {
            return Ok(Some(p));
        }
    }
    Ok(None)
}

/// Writes `content` to a temp file in `vault`, then renames it over `target`, or links it to
/// the first free `<base>.md`, `<base> 2.md`, ... Returns the file name.
fn put(vault: &Path, id: &str, target: Option<&Path>, base: &str, content: &str) -> Result<String> {
    let tmp = vault.join(format!(".mictap-{id}.tmp"));
    let mut f = std::fs::File::create(&tmp)?;
    f.write_all(content.as_bytes())?;
    f.sync_data()?;
    if let Some(t) = target {
        std::fs::rename(&tmp, t)?;
        return Ok(t.file_name().unwrap().to_string_lossy().into_owned());
    }
    let res = (1..)
        .map(|n| match n {
            1 => format!("{base}.md"),
            n => format!("{base} {n}.md"),
        })
        .find_map(|name| match std::fs::hard_link(&tmp, vault.join(&name)) {
            Err(e) if e.kind() == ErrorKind::AlreadyExists => None,
            r => Some(r.map(|()| name)),
        })
        .unwrap();
    std::fs::remove_file(&tmp)?;
    Ok(res?)
}

/// How much of how much audio of `id` is transcribed, and its status as the API shows it.
pub(crate) fn progress(
    app: &App,
    db: &Connection,
    id: &str,
    status: &str,
    segs: &[Segment],
) -> Result<(&'static str, i64, i64)> {
    // Where the earliest untranscribed window starts, else where the last transcribed one ends.
    let transcribed_ms: i64 = db.query_row(
        "SELECT COALESCE(MIN(CASE WHEN NOT done THEN offset_ms + start_ms END),
                         MAX(offset_ms + end_ms), 0)
         FROM windows WHERE recording = ?1",
        [id],
        |r| r.get(0),
    )?;
    let decoded: Vec<(String, i64)> = db
        .prepare("SELECT file, done_ms FROM file_progress WHERE recording = ?1")?
        .query_map([id], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    let meta: Value = std::fs::read(app.recording_dir(id).join("meta.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    let total_ms = meta["segments"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|s| {
            let offset = s["offset_ms"].as_i64().unwrap_or(0);
            let decoded = decoded
                .iter()
                .find(|(f, _)| s["file"].as_str() == Some(f))
                .map_or(0, |(_, ms)| *ms);
            s["end_ms"].as_i64().unwrap_or(0).max(offset + decoded)
        })
        .chain(segs.iter().map(|s| s.end_ms))
        .max()
        .unwrap_or(0);
    let shown = match status {
        "diarized" | "done" => "done",
        "failed" => "failed",
        _ => "transcribing",
    };
    let done_ms = match shown {
        "done" => total_ms,
        _ => transcribed_ms,
    };
    Ok((shown, done_ms, total_ms))
}

/// Writes the transcript of `id`, over its note wherever it was renamed to in the vault
/// folder. A note that was moved out or deleted is left alone, and a recording without
/// speech gets none.
pub async fn write(app: &App, id: &str) -> Result<()> {
    let db = app.db.lock().await;
    let (started_ms, cached, speakers): (i64, Option<String>, Option<String>) = db.query_row(
        "SELECT started_ms, vault_path, speakers FROM recordings WHERE id = ?1",
        [id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    )?;
    let segs = crate::assemble::merged(&db, id)?;
    drop(db);
    let names: Names = speakers
        .map(|s| serde_json::from_str(&s))
        .transpose()?
        .unwrap_or_default();
    let target = locate(&app.vault, id, cached.as_deref())?;
    if segs.is_empty() || (target.is_none() && cached.is_some()) {
        return Ok(());
    }
    let start = Timestamp::from_millisecond(started_ms)?.to_zoned(app.tz.clone());
    let content = render(id, &start.strftime("%Y-%m-%d %H:%M").to_string(), &segs, &names);
    let base = start.strftime("%Y-%m-%d %H%M Meeting").to_string();
    let name = put(&app.vault, id, target.as_deref(), &base, &content)?;
    app.db
        .lock()
        .await
        .execute("UPDATE recordings SET vault_path = ?2 WHERE id = ?1", params![id, name])?;
    Ok(())
}

/// Writes the transcript of each newly diarized recording, which makes it done. Names
/// matched at diarization are kept only for the labels that survived the echo dedupe.
pub async fn run(app: Arc<App>) {
    loop {
        let ids: rusqlite::Result<Vec<String>> = {
            let db = app.db.lock().await;
            db.prepare("SELECT id FROM recordings WHERE status = 'diarized'")
                .and_then(|mut st| st.query_map([], |r| r.get(0))?.collect())
        };
        for id in ids.unwrap_or_else(|e| {
            eprintln!("vault: {e}");
            vec![]
        }) {
            if let Err(e) = done(&app, &id).await {
                eprintln!("{id}: vault: {e:#}");
            }
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

async fn done(app: &App, id: &str) -> Result<()> {
    {
        let db = app.db.lock().await;
        let segs = crate::assemble::merged(&db, id)?;
        let speakers: Option<String> =
            db.query_row("SELECT speakers FROM recordings WHERE id = ?1", [id], |r| r.get(0))?;
        let mut names: Names = speakers
            .map(|s| serde_json::from_str(&s))
            .transpose()?
            .unwrap_or_default();
        names.retain(|l, _| segs.iter().any(|s| s.speaker.as_ref() == Some(l)));
        db.execute(
            "UPDATE recordings SET speakers = ?2 WHERE id = ?1",
            params![id, serde_json::to_string(&names)?],
        )?;
    }
    write(app, id).await?;
    app.db
        .lock()
        .await
        .execute("UPDATE recordings SET status = 'done' WHERE id = ?1", [id])?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(track: &str, start_ms: i64, text: &str, speaker: Option<&str>) -> Segment {
        Segment {
            track: track.into(),
            start_ms,
            end_ms: start_ms + 2_000,
            text: text.into(),
            speaker: speaker.map(Into::into),
        }
    }

    #[test]
    fn renders_named_attendees_and_lines() {
        let segs = [
            seg(
                "room",
                842_000,
                "Zullen we zeggen dat het volgende sprint wordt?",
                Some("room/S1"),
            ),
            seg("remote", 845_400, "Hallo? Zijn jullie er nog?", Some("remote/S1")),
            seg("room", 3_725_000, "Ja, prima.", Some("room/S10")),
            seg("room", 3_726_000, "Hm.", None),
            seg("room", 3_727_000, "Ok.", Some("room/S2")),
        ];
        let names: Names = [("remote/S1", "Jan"), ("room/S1", "Max"), ("room/S2", "Max")]
            .map(|(l, n)| (l.to_string(), n.to_string()))
            .into();
        assert_eq!(
            render("r1", "2026-09-24 14:00", &segs, &names),
            "---\nid: r1\ndate: 2026-09-24 14:00\nattendees: [\"[[Max]]\", \"[[Jan]]\"]\n\
             link: http://localhost:8765/#r1\n---\n\n\
             **Max** (room, [00:14:02](http://localhost:8765/r/r1/audio.ogg#t=842)): Zullen we zeggen dat het volgende sprint wordt?\n\
             **Jan** (remote, [00:14:05](http://localhost:8765/r/r1/audio.ogg#t=845)): Hallo? Zijn jullie er nog?\n\
             **S10** (room, [01:02:05](http://localhost:8765/r/r1/audio.ogg#t=3725)): Ja, prima.\n\
             **?** (room, [01:02:06](http://localhost:8765/r/r1/audio.ogg#t=3726)): Hm.\n\
             **Max** (room, [01:02:07](http://localhost:8765/r/r1/audio.ogg#t=3727)): Ok.\n"
        );
    }

    #[test]
    fn finds_by_id_and_avoids_collisions() {
        let tmp = tempfile::tempdir().unwrap();
        let v = tmp.path();
        std::fs::write(v.join("2026-09-24 1400 Meeting.md"), "---\nid: other\n---\n").unwrap();
        let name = put(v, "r1", None, "2026-09-24 1400 Meeting", "---\nid: r1\n---\n").unwrap();
        assert_eq!(name, "2026-09-24 1400 Meeting 2.md");
        assert_eq!(std::fs::read_dir(v).unwrap().count(), 2, "temp file left behind");

        std::fs::rename(v.join(&name), v.join("Planning.md")).unwrap();
        let found = locate(v, "r1", Some(&name)).unwrap().unwrap();
        assert_eq!(found, v.join("Planning.md"));
        assert_eq!(
            put(v, "r1", Some(&found), "x", "---\nid: r1\nnew\n").unwrap(),
            "Planning.md"
        );
        assert_eq!(std::fs::read_to_string(&found).unwrap(), "---\nid: r1\nnew\n");

        // Obsidian may quote the id; an id in the body doesn't count.
        std::fs::write(&found, "---\nid: \"r1\"\n---\n").unwrap();
        assert!(has_id(&found, "r1"));
        std::fs::write(&found, "---\ntitle: x\n---\nid: r1\n").unwrap();
        assert_eq!(locate(v, "r1", Some("Planning.md")).unwrap(), None);
    }

    async fn setup() -> (tempfile::TempDir, App) {
        let tmp = tempfile::tempdir().unwrap();
        let mut app = App::open(tmp.path().join("state")).unwrap();
        app.vault = tmp.path().join("vault");
        app.tz = jiff::tz::TimeZone::get("Europe/Amsterdam").unwrap();
        std::fs::create_dir_all(&app.vault).unwrap();
        app.db
            .lock()
            .await
            .execute_batch(
                r#"INSERT INTO recordings (id, source, started_ms, status, speakers)
                     VALUES ('r1', 'laptop', 1790431200000, 'diarized', '{"room/S1":"Max","room/S9":"Echo"}');
                   INSERT INTO windows (id, recording, file, track, offset_ms, start_ms, end_ms, done) VALUES
                     (1, 'r1', '00-mic.oga', 'room', 0, 0, 30000, 1),
                     (2, 'r1', '01-app-7.oga', 'remote', 60000, 0, 30000, 1);
                   INSERT INTO segments (recording, window, track, start_ms, end_ms, text, speaker) VALUES
                     ('r1', 1, 'room', 1000, 3000, 'Goedemorgen allemaal.', 'room/S1'),
                     ('r1', 2, 'remote', 61000, 64000, 'Hoi!', 'remote/S1');"#,
            )
            .unwrap();
        (tmp, app)
    }

    async fn one(app: &App, sql: &str) -> String {
        app.db.lock().await.query_row(sql, [], |r| r.get(0)).unwrap()
    }

    #[tokio::test]
    async fn done_writes_the_note_then_rewrites_it_where_it_went() {
        let (_tmp, app) = setup().await;
        done(&app, "r1").await.unwrap();
        let name = "2026-09-26 1600 Meeting.md";
        let text = std::fs::read_to_string(app.vault.join(name)).unwrap();
        assert!(
            text.starts_with("---\nid: r1\ndate: 2026-09-26 16:00\nattendees: [\"[[Max]]\"]\n"),
            "{text}"
        );
        assert!(text.contains("**Max** (room, [00:00:01]"), "{text}");
        assert!(text.contains("**S1** (remote, [00:01:01]"), "{text}");
        assert_eq!(
            one(&app, "SELECT status || ' ' || speakers FROM recordings").await,
            r#"done {"room/S1":"Max"}"#,
            "labels not shown are dropped"
        );

        // Renamed in Obsidian: the next write lands there.
        std::fs::rename(app.vault.join(name), app.vault.join("Kickoff.md")).unwrap();
        app.db
            .lock()
            .await
            .execute(r#"UPDATE recordings SET speakers = '{"remote/S1":"Jan"}'"#, [])
            .unwrap();
        write(&app, "r1").await.unwrap();
        let text = std::fs::read_to_string(app.vault.join("Kickoff.md")).unwrap();
        assert!(text.contains("attendees: [\"[[Jan]]\"]\n"), "{text}");
        assert!(text.contains("**Jan** (remote, [00:01:01]"), "{text}");

        // Moved out: left alone, not recreated.
        std::fs::remove_file(app.vault.join("Kickoff.md")).unwrap();
        write(&app, "r1").await.unwrap();
        assert_eq!(std::fs::read_dir(&app.vault).unwrap().count(), 0);
    }

    #[tokio::test]
    async fn no_speech_no_note() {
        let (_tmp, app) = setup().await;
        app.db.lock().await.execute("DELETE FROM segments", []).unwrap();
        done(&app, "r1").await.unwrap();
        assert_eq!(std::fs::read_dir(&app.vault).unwrap().count(), 0);
        assert_eq!(one(&app, "SELECT status FROM recordings").await, "done");
    }
}
