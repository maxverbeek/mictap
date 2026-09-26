use std::{
    io::{ErrorKind, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use anyhow::Result;
use jiff::Timestamp;
use rusqlite::{params, Connection};
use serde_json::Value;

use crate::{
    api::App,
    merge::Segment,
    speakers::{attendees, display, Names},
};

/// Base of the timestamp links, as browsers reach this server.
pub static URL: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| std::env::var("MICTAP_URL").unwrap_or_else(|_| "http://localhost:8765".into()));

struct Header<'a> {
    id: &'a str,
    date: String,
    source: &'a str,
    status: &'a str,
    done_ms: i64,
    total_ms: i64,
    error: Option<&'a str>,
}

fn minutes(ms: i64) -> i64 {
    (ms + 30_000) / 60_000
}

fn hms(ms: i64) -> String {
    let s = ms / 1000;
    format!("{:02}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
}

/// `room/S2` sorts as (false, 2): room before remote, then by number.
pub(crate) fn label_order(label: &str) -> (bool, u32) {
    let (track, n) = label.split_once("/S").unwrap_or((label, ""));
    (track != "room", n.parse().unwrap_or(u32::MAX))
}

/// The transcript as README specifies it, lines labeled with `names` where set.
fn render(h: &Header, segs: &[Segment], names: &Names) -> String {
    let (done, total) = (minutes(h.done_ms), minutes(h.total_ms));
    let mut out = format!(
        "---\nid: {}\ndate: {}\nduration: {total}m\nsource: {}\nstatus: {}\nprogress: {}/{total} min\n",
        h.id,
        h.date,
        h.source,
        h.status,
        done.min(total),
    );
    if let Some(e) = h.error {
        out += &format!("error: {}\n", Value::from(e));
    }
    let mut labels: Vec<&str> = segs.iter().filter_map(|s| s.speaker.as_deref()).collect();
    labels.sort_by_key(|l| label_order(l));
    labels.dedup();
    out += &attendees(names);
    // One flat property per speaker: Obsidian can only edit flat values.
    for l in labels {
        out += &format!("{l}: {}\n", Value::from(names.get(l).map_or("", String::as_str)));
    }
    out += "---\n\n";
    for s in segs {
        let name = s.speaker.as_deref().map_or("?", |l| display(l, names));
        out += &format!(
            "**{name}** ({}, [{}]({}/r/{}/audio.ogg#t={})): {}\n",
            s.track,
            hms(s.start_ms),
            *URL,
            h.id,
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
pub(crate) fn locate(vault: &Path, id: &str, cached: Option<&str>) -> std::io::Result<Option<PathBuf>> {
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
pub(crate) fn put(vault: &Path, id: &str, target: Option<&Path>, base: &str, content: &str) -> Result<String> {
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

/// Missed syncs (5 s apart) before a transcript that vanished from the folder is given up.
const GONE_AFTER: u32 = 60;

/// What the note shows for `id`: its status, and how much of how much audio is transcribed.
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

/// Brings the vault file of `id` up to date with the database.
pub async fn sync(app: &App, id: &str) -> Result<()> {
    let db = app.db.lock().await;
    let (source, started_ms, status, error, cached, written): (
        String,
        i64,
        String,
        Option<String>,
        Option<String>,
        Option<String>,
    ) = db.query_row(
        "SELECT source, started_ms, status, error, vault_path, written FROM recordings WHERE id = ?1",
        [id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?)),
    )?;
    if matches!(written.as_deref(), Some("done" | "failed" | "gone")) {
        return Ok(());
    }
    let segs = crate::merge::merged(&db, id)?;
    // Names matched at diarization, for the labels that survived the echo dedupe.
    let speakers: Option<String> = db.query_row("SELECT speakers FROM recordings WHERE id = ?1", [id], |r| r.get(0))?;
    let mut names: Names = speakers
        .map(|s| serde_json::from_str(&s))
        .transpose()?
        .unwrap_or_default();
    names.retain(|l, _| segs.iter().any(|s| s.speaker.as_ref() == Some(l)));
    let (shown, done_ms, total_ms) = progress(app, &db, id, &status, &segs)?;
    // No note until there is speech; a recording that ends without any never gets one.
    if cached.is_none() && segs.is_empty() && shown != "failed" {
        if shown == "done" {
            db.execute(
                "UPDATE recordings SET written = 'done', status = 'done' WHERE id = ?1",
                [id],
            )?;
        }
        return Ok(());
    }
    let key = match shown {
        "transcribing" => format!("transcribing {done_ms}/{total_ms} {}", segs.len()),
        s => s.to_string(),
    };
    if written.as_deref() == Some(&key) {
        return Ok(());
    }
    drop(db);

    let start = Timestamp::from_millisecond(started_ms)?.to_zoned(app.tz.clone());
    let header = Header {
        id,
        date: start.strftime("%Y-%m-%d %H:%M").to_string(),
        source: &source,
        status: shown,
        done_ms,
        total_ms,
        error: error.as_deref().filter(|_| shown == "failed"),
    };
    let target = locate(&app.vault, id, cached.as_deref())?;
    let (name, key) = if target.is_none() && cached.is_some() {
        // A rename synced as delete plus upload leaves a gap: only a long absence counts.
        let misses = 1 + written
            .as_deref()
            .and_then(|w| w.strip_prefix("missing "))
            .and_then(|n| n.parse::<u32>().ok())
            .unwrap_or(0);
        if misses < GONE_AFTER {
            (cached, format!("missing {misses}"))
        } else {
            eprintln!("{id}: transcript moved out of the vault folder, no longer writing it");
            (cached, "gone".to_string())
        }
    } else {
        let base = start.strftime("%Y-%m-%d %H%M Meeting").to_string();
        let content = render(&header, &segs, &names);
        (Some(put(&app.vault, id, target.as_deref(), &base, &content)?), key)
    };
    // The lock was released for the file IO, so the status read above may be stale: only
    // 'done' is written back, never the old value. With it, the names shown become the ones
    // C7 compares the file against.
    let done = (shown == "done" && !key.starts_with("missing")).then_some("done");
    app.db.lock().await.execute(
        "UPDATE recordings SET vault_path = ?2, written = ?3, status = COALESCE(?4, status),
         speakers = CASE WHEN ?4 IS NULL THEN speakers ELSE ?5 END
         WHERE id = ?1",
        params![id, name, key, done, serde_json::to_string(&names)?],
    )?;
    Ok(())
}

/// Keeps every unfinished recording's transcript current.
pub async fn run(app: Arc<App>) {
    loop {
        let ids: rusqlite::Result<Vec<String>> = {
            let db = app.db.lock().await;
            db.prepare(
                "SELECT id FROM recordings WHERE started_ms IS NOT NULL
                 AND COALESCE(written, '') NOT IN ('done', 'failed', 'gone')",
            )
            .and_then(|mut st| st.query_map([], |r| r.get(0))?.collect())
        };
        for id in ids.unwrap_or_else(|e| {
            eprintln!("vault: {e}");
            vec![]
        }) {
            if let Err(e) = sync(&app, &id).await {
                eprintln!("{id}: vault: {e:#}");
            }
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
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
    fn renders_readme_format() {
        let h = Header {
            id: "r1",
            date: "2026-09-24 14:00".into(),
            source: "laptop",
            status: "done",
            done_ms: 3_119_000,
            total_ms: 3_119_000,
            error: None,
        };
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
        assert_eq!(
            render(&h, &segs, &Names::new()),
            "---\nid: r1\ndate: 2026-09-24 14:00\nduration: 52m\nsource: laptop\n\
             status: done\nprogress: 52/52 min\nattendees: []\nroom/S1: \"\"\n\
             room/S2: \"\"\nroom/S10: \"\"\nremote/S1: \"\"\n---\n\n\
             **S1** (room, [00:14:02](http://localhost:8765/r/r1/audio.ogg#t=842)): Zullen we zeggen dat het volgende sprint wordt?\n\
             **S1** (remote, [00:14:05](http://localhost:8765/r/r1/audio.ogg#t=845)): Hallo? Zijn jullie er nog?\n\
             **S10** (room, [01:02:05](http://localhost:8765/r/r1/audio.ogg#t=3725)): Ja, prima.\n\
             **?** (room, [01:02:06](http://localhost:8765/r/r1/audio.ogg#t=3726)): Hm.\n\
             **S2** (room, [01:02:07](http://localhost:8765/r/r1/audio.ogg#t=3727)): Ok.\n"
        );
        let h = Header {
            status: "failed",
            done_ms: 100_000,
            error: Some("whisper-cli: \"model\" missing"),
            ..h
        };
        assert_eq!(
            render(&h, &[], &Names::new()),
            "---\nid: r1\ndate: 2026-09-24 14:00\nduration: 52m\nsource: laptop\n\
             status: failed\nprogress: 2/52 min\nerror: \"whisper-cli: \\\"model\\\" missing\"\n\
             attendees: []\n---\n\n"
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
        let dir = app.recording_dir("r1");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("meta.json"),
            r#"{"id":"r1","started_ms":1790431200000,"app":"Zen","segments":[
                {"file":"00-mic.oga","key":"mic","target":"x","offset_ms":0,"end_ms":null},
                {"file":"01-app-7.oga","key":"app-7","target":"y","offset_ms":60000,"end_ms":null}]}"#,
        )
        .unwrap();
        app.db
            .lock()
            .await
            .execute_batch(
                "INSERT INTO recordings (id, source, started_ms) VALUES ('r1', 'laptop', 1790431200000);
                 INSERT INTO file_progress (recording, file, done_ms) VALUES
                   ('r1', '00-mic.oga', 240000), ('r1', '01-app-7.oga', 150000);
                 INSERT INTO windows (id, recording, file, track, offset_ms, start_ms, end_ms, done) VALUES
                   (1, 'r1', '00-mic.oga', 'room', 0, 0, 30000, 1),
                   (2, 'r1', '01-app-7.oga', 'remote', 60000, 0, 30000, 1),
                   (3, 'r1', '00-mic.oga', 'room', 0, 150000, 170000, 0);
                 INSERT INTO segments (recording, window, track, start_ms, end_ms, text) VALUES
                   ('r1', 1, 'room', 1000, 3000, 'Goedemorgen allemaal.'),
                   ('r1', 2, 'remote', 61000, 64000, 'Hoi!');",
            )
            .unwrap();
        (tmp, app)
    }

    fn read(app: &App, name: &str) -> String {
        std::fs::read_to_string(app.vault.join(name)).unwrap()
    }

    #[tokio::test]
    async fn writes_progressively_then_done() {
        let (_tmp, app) = setup().await;
        let name = "2026-09-26 1600 Meeting.md";
        sync(&app, "r1").await.unwrap();
        let text = read(&app, name);
        assert!(
            text.contains(
                "\ndate: 2026-09-26 16:00\nduration: 4m\nsource: laptop\n\
                               status: transcribing\nprogress: 3/4 min\n"
            ),
            "{text}"
        );
        assert!(text.contains("attendees: []\n---\n"), "{text}");
        assert!(text.contains("**?** (remote, [00:01:01]"), "{text}");

        // The user renames it; the next window lands there.
        std::fs::rename(app.vault.join(name), app.vault.join("Kickoff.md")).unwrap();
        app.db
            .lock()
            .await
            .execute_batch(
                "UPDATE windows SET done = 1 WHERE id = 3;
                 INSERT INTO segments (recording, window, track, start_ms, end_ms, text)
                   VALUES ('r1', 3, 'room', 151000, 153000, 'Laatste punt.');",
            )
            .unwrap();
        sync(&app, "r1").await.unwrap();
        let text = read(&app, "Kickoff.md");
        assert!(text.contains("progress: 3/4 min\n"), "{text}");
        assert!(text.contains("Laatste punt."), "{text}");
        assert_eq!(std::fs::read_dir(&app.vault).unwrap().count(), 1);

        app.db
            .lock()
            .await
            .execute_batch(
                "UPDATE recordings SET status = 'diarized', finished = 1;
                 UPDATE segments SET speaker = track || '/S1';",
            )
            .unwrap();
        sync(&app, "r1").await.unwrap();
        let text = read(&app, "Kickoff.md");
        assert!(text.contains("status: done\nprogress: 4/4 min\n"), "{text}");
        assert!(
            text.contains("attendees: []\nroom/S1: \"\"\nremote/S1: \"\"\n---\n"),
            "{text}"
        );
        assert!(
            text.contains(
                "**S1** (room, [00:00:01](http://localhost:8765/r/r1/audio.ogg#t=1)): Goedemorgen allemaal.\n"
            ),
            "{text}"
        );
        let status: String = app
            .db
            .lock()
            .await
            .query_row("SELECT status FROM recordings", [], |r| r.get(0))
            .unwrap();
        assert_eq!(status, "done");

        // The body is the user's now.
        std::fs::write(app.vault.join("Kickoff.md"), "---\nid: r1\n---\nmine\n").unwrap();
        sync(&app, "r1").await.unwrap();
        assert_eq!(read(&app, "Kickoff.md"), "---\nid: r1\n---\nmine\n");
    }

    #[tokio::test]
    async fn prefilled_names_label_lines() {
        let (_tmp, app) = setup().await;
        app.db
            .lock()
            .await
            .execute_batch(
                r#"UPDATE recordings SET status = 'diarized', finished = 1,
                     speakers = '{"room/S1":"Max","room/S9":"Echo"}';
                   UPDATE windows SET done = 1;
                   UPDATE segments SET speaker = track || '/S1';"#,
            )
            .unwrap();
        sync(&app, "r1").await.unwrap();
        let text = read(&app, "2026-09-26 1600 Meeting.md");
        assert!(
            text.contains("attendees: [\"[[Max]]\"]\nroom/S1: \"Max\"\nremote/S1: \"\"\n---\n"),
            "{text}"
        );
        assert!(text.contains("**Max** (room, [00:00:01]"), "{text}");
        assert!(text.contains("**S1** (remote, [00:01:01]"), "{text}");
        let speakers: String = app
            .db
            .lock()
            .await
            .query_row("SELECT speakers FROM recordings", [], |r| r.get(0))
            .unwrap();
        assert_eq!(speakers, r#"{"room/S1":"Max"}"#, "labels not shown are dropped");
    }

    #[tokio::test]
    async fn failed_gets_an_error_line() {
        let (_tmp, app) = setup().await;
        for _ in 0..crate::db::ATTEMPTS {
            crate::db::fail(&*app.db.lock().await, "r1", "transcribing: whisper-cli: boom").unwrap();
        }
        sync(&app, "r1").await.unwrap();
        let text = read(&app, "2026-09-26 1600 Meeting.md");
        assert!(
            text.contains(
                "status: failed\nprogress: 3/4 min\n\
                               error: \"transcribing: whisper-cli: boom\"\n"
            ),
            "{text}"
        );
    }

    #[tokio::test]
    async fn stops_when_moved_out() {
        let (tmp, app) = setup().await;
        sync(&app, "r1").await.unwrap();
        std::fs::rename(
            app.vault.join("2026-09-26 1600 Meeting.md"),
            tmp.path().join("elsewhere.md"),
        )
        .unwrap();
        app.db.lock().await.execute("UPDATE windows SET done = 1", []).unwrap();
        let written = || async {
            app.db
                .lock()
                .await
                .query_row("SELECT written FROM recordings", [], |r| r.get::<_, String>(0))
                .unwrap()
        };
        sync(&app, "r1").await.unwrap();
        assert_eq!(written().await, "missing 1");

        // Back under a new name (a rename synced as delete plus upload): writing resumes.
        std::fs::rename(tmp.path().join("elsewhere.md"), app.vault.join("Kickoff.md")).unwrap();
        sync(&app, "r1").await.unwrap();
        assert!(read(&app, "Kickoff.md").contains("status: transcribing"));
        assert!(written().await.starts_with("transcribing"));

        std::fs::rename(app.vault.join("Kickoff.md"), tmp.path().join("elsewhere.md")).unwrap();
        app.db
            .lock()
            .await
            .execute("UPDATE recordings SET status = 'diarized', finished = 1", [])
            .unwrap();
        for _ in 0..GONE_AFTER {
            sync(&app, "r1").await.unwrap();
        }
        assert_eq!(std::fs::read_dir(&app.vault).unwrap().count(), 0);
        assert_eq!(written().await, "gone");
        let status: String = app
            .db
            .lock()
            .await
            .query_row("SELECT status FROM recordings", [], |r| r.get(0))
            .unwrap();
        assert_eq!(status, "done");
    }

    #[tokio::test]
    async fn no_speech_no_note() {
        let (_tmp, app) = setup().await;
        app.db.lock().await.execute("DELETE FROM segments", []).unwrap();
        sync(&app, "r1").await.unwrap();
        assert_eq!(std::fs::read_dir(&app.vault).unwrap().count(), 0);
        app.db
            .lock()
            .await
            .execute_batch("UPDATE windows SET done = 1; UPDATE recordings SET status = 'diarized', finished = 1;")
            .unwrap();
        sync(&app, "r1").await.unwrap();
        assert_eq!(std::fs::read_dir(&app.vault).unwrap().count(), 0);
        let row: (String, String) = app
            .db
            .lock()
            .await
            .query_row("SELECT status, written FROM recordings", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap();
        assert_eq!(row, ("done".into(), "done".into()));
    }
}
