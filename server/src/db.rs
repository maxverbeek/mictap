use std::path::Path;

use rusqlite::{params, Connection};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS recordings (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    started_ms INTEGER,
    status TEXT NOT NULL DEFAULT 'receiving',
    vault_path TEXT,
    finished INTEGER NOT NULL DEFAULT 0,
    lang TEXT,
    error TEXT,
    -- Failed attempts at the current step; see fail().
    attempts INTEGER NOT NULL DEFAULT 0,
    -- Unix ms before which workers leave a failing recording alone.
    retry_at INTEGER NOT NULL DEFAULT 0,
    -- Confirmed names (label -> name, JSON): typed, or suggestions accepted.
    speakers TEXT,
    -- Names suggested from known voices for unconfirmed labels (label -> name, JSON).
    suggested TEXT,
    -- audio.ogg: NULL until mixed down, then 'ready', 'failed' or 'expired'.
    audio TEXT,
    -- Unix ms it became done; whisper's segments expire MICTAP_OUTPUTS_DAYS later.
    done_ms INTEGER
);
CREATE TABLE IF NOT EXISTS file_progress (
    recording TEXT NOT NULL REFERENCES recordings(id),
    file TEXT NOT NULL,
    done_ms INTEGER NOT NULL DEFAULT 0,
    complete INTEGER NOT NULL DEFAULT 0,
    -- The file's size when done_ms was stored.
    bytes INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (recording, file)
);
-- start_ms/end_ms are within the file; offset_ms is the file's start in the recording.
CREATE TABLE IF NOT EXISTS windows (
    id INTEGER PRIMARY KEY,
    recording TEXT NOT NULL REFERENCES recordings(id),
    file TEXT NOT NULL,
    track TEXT NOT NULL,
    offset_ms INTEGER NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    done INTEGER NOT NULL DEFAULT 0
);
-- Model outputs (CONTEXT.md), never modified. start_ms/end_ms are within the recording.
-- whisper's.
CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY,
    recording TEXT NOT NULL REFERENCES recordings(id),
    window INTEGER NOT NULL REFERENCES windows(id),
    track TEXT NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    text TEXT NOT NULL
);
-- sherpa's, with speaker its cluster and embedding CAM++'s (L2-normalized f32 little-endian).
-- Kept durably, unlike segments: naming lines weighs them anew (names::named).
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY,
    recording TEXT NOT NULL REFERENCES recordings(id),
    track TEXT NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    speaker INTEGER NOT NULL,
    embedding BLOB
);
-- Derived from the model outputs by assemble::derive.
CREATE TABLE IF NOT EXISTS lines (
    id INTEGER PRIMARY KEY,
    recording TEXT NOT NULL REFERENCES recordings(id),
    track TEXT NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    text TEXT NOT NULL,
    speaker TEXT
);
-- embedding: the mean of the cluster's L2-normalized turn embeddings, f32 little-endian.
-- core: see assemble::Cluster::core; halves_alike/minor_share: see assemble::Cluster::halves.
-- All three NULL for clusters derived before they existed.
CREATE TABLE IF NOT EXISTS clusters (
    recording TEXT NOT NULL REFERENCES recordings(id),
    label TEXT NOT NULL,
    embedding BLOB NOT NULL,
    core BLOB,
    halves_alike REAL,
    minor_share REAL,
    PRIMARY KEY (recording, label)
);
-- An embedding learned for a confirmed name: from the cluster (start_ms/end_ms NULL), or
-- from a snippet of it that was heard.
CREATE TABLE IF NOT EXISTS voices (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    embedding BLOB NOT NULL,
    recording TEXT NOT NULL REFERENCES recordings(id),
    label TEXT NOT NULL,
    start_ms INTEGER,
    end_ms INTEGER
);
-- A name set for one line, kept by time since lines are derived anew; '?' when mixed or
-- unsure. A line takes the name whose span holds its midpoint on its track.
CREATE TABLE IF NOT EXISTS line_names (
    recording TEXT NOT NULL REFERENCES recordings(id),
    track TEXT NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    name TEXT NOT NULL
);
-- CAM++'s embedding of one line's own audio (L2-normalized f32 little-endian), kept by time
-- like line_names. Not a model output: kept to guess names anew as voices are learned.
CREATE TABLE IF NOT EXISTS line_voices (
    recording TEXT NOT NULL REFERENCES recordings(id),
    track TEXT NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    embedding BLOB NOT NULL
);
";

pub fn open(path: &Path) -> rusqlite::Result<Connection> {
    let conn = Connection::open(path)?;
    conn.execute_batch(SCHEMA)?;
    if conn.prepare("SELECT retry_at FROM recordings").is_err() {
        conn.execute_batch("ALTER TABLE recordings ADD COLUMN retry_at INTEGER NOT NULL DEFAULT 0")?;
    }
    if conn.prepare("SELECT halves_alike FROM clusters").is_err() {
        conn.execute_batch(
            "ALTER TABLE clusters ADD COLUMN halves_alike REAL;
             ALTER TABLE clusters ADD COLUMN minor_share REAL;",
        )?;
    }
    if conn.prepare("SELECT core FROM clusters").is_err() {
        conn.execute_batch("ALTER TABLE clusters ADD COLUMN core BLOB")?;
    }
    if conn.prepare("SELECT start_ms FROM voices").is_err() {
        let tx = conn.unchecked_transaction()?;
        tx.execute_batch(
            "CREATE TABLE voices_new (
                 id INTEGER PRIMARY KEY,
                 name TEXT NOT NULL,
                 embedding BLOB NOT NULL,
                 recording TEXT NOT NULL REFERENCES recordings(id),
                 label TEXT NOT NULL,
                 start_ms INTEGER,
                 end_ms INTEGER
             );
             INSERT INTO voices_new (name, embedding, recording, label)
               SELECT name, embedding, recording, label FROM voices;
             DROP TABLE voices;
             ALTER TABLE voices_new RENAME TO voices;",
        )?;
        tx.commit()?;
    }
    if conn.prepare("SELECT suggested FROM recordings").is_err() {
        conn.execute_batch("ALTER TABLE recordings ADD COLUMN suggested TEXT")?;
    }
    if conn.prepare("SELECT done_ms FROM recordings").is_err() {
        conn.execute_batch("ALTER TABLE recordings ADD COLUMN done_ms INTEGER")?;
        conn.execute("UPDATE recordings SET done_ms = ?1 WHERE status = 'done'", [now_ms()])?;
    }
    if conn.prepare("SELECT speaker FROM segments").is_ok() {
        lines_from_labeled_segments(&conn)?;
    }
    Ok(conn)
}

/// Before model outputs were kept, diarization split and labeled the segments in place:
/// they become the lines, echoes dropped as reads used to.
fn lines_from_labeled_segments(conn: &Connection) -> rusqlite::Result<()> {
    let tx = conn.unchecked_transaction()?;
    let ids: Vec<String> = tx
        .prepare("SELECT DISTINCT recording FROM segments")?
        .query_map([], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    let echo = crate::assemble::Tuning::from_env().echo_jaccard;
    for id in ids {
        let segs: Vec<crate::assemble::Line> = tx
            .prepare("SELECT track, start_ms, end_ms, text, speaker FROM segments WHERE recording = ?1")?
            .query_map([&id], |r| {
                Ok(crate::assemble::Line {
                    track: r.get(0)?,
                    start_ms: r.get(1)?,
                    end_ms: r.get(2)?,
                    text: r.get(3)?,
                    speaker: r.get(4)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?;
        for l in crate::assemble::drop_echoes(segs, echo) {
            tx.execute(
                "INSERT INTO lines (recording, track, start_ms, end_ms, text, speaker)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![id, l.track, l.start_ms, l.end_ms, l.text, l.speaker],
            )?;
        }
    }
    tx.execute_batch("ALTER TABLE segments DROP COLUMN speaker")?;
    tx.commit()
}

pub fn ensure_recording(conn: &Connection, id: &str, source: &str) -> rusqlite::Result<()> {
    conn.execute(
        "INSERT OR IGNORE INTO recordings (id, source) VALUES (?1, ?2)",
        params![id, source],
    )?;
    Ok(())
}

pub fn set_started(conn: &Connection, id: &str, started_ms: i64) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE recordings SET started_ms = ?2 WHERE id = ?1",
        params![id, started_ms],
    )?;
    Ok(())
}

/// Returns false when the recording doesn't exist.
pub fn finish(conn: &Connection, id: &str) -> rusqlite::Result<bool> {
    Ok(conn.execute("UPDATE recordings SET finished = 1 WHERE id = ?1", params![id])? == 1)
}

/// Deletes whisper's segments of recordings done more than `MICTAP_OUTPUTS_DAYS` (default 7)
/// before `now_ms`. Their lines and turns stay.
pub fn expire_outputs(conn: &Connection, now_ms: i64) -> rusqlite::Result<()> {
    let days: i64 = std::env::var("MICTAP_OUTPUTS_DAYS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7);
    conn.execute(
        "DELETE FROM segments WHERE recording IN (SELECT id FROM recordings WHERE done_ms < ?1)",
        [now_ms - days * 86_400_000],
    )?;
    Ok(())
}

pub fn now_ms() -> i64 {
    jiff::Timestamp::now().as_millisecond()
}

/// Consecutive failures at one step before the recording fails for good. With the backoff
/// below that spans about 3 hours, so an OOM kill or a full disk is waited out.
pub(crate) const ATTEMPTS: i64 = 10;

/// Counts a failed attempt at the current step and holds the recording back from the
/// workers for 30 s, doubling per attempt up to an hour; the `ATTEMPTS`th in a row fails
/// the recording. Returns true when it did. Workers reset `attempts` when a step succeeds.
pub fn fail(conn: &Connection, id: &str, error: &str) -> rusqlite::Result<bool> {
    conn.query_row(
        "UPDATE recordings SET attempts = attempts + 1, error = ?2,
         retry_at = ?4 + MIN(30000 << attempts, 3600000),
         status = CASE WHEN attempts + 1 >= ?3 THEN 'failed' ELSE status END
         WHERE id = ?1 RETURNING status = 'failed'",
        params![id, error, ATTEMPTS, now_ms()],
        |r| r.get(0),
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn backs_off_then_fails() {
        let conn = super::open(std::path::Path::new(":memory:")).unwrap();
        super::ensure_recording(&conn, "r1", "laptop").unwrap();
        let row = || -> (String, i64) {
            conn.query_row("SELECT status, retry_at - ?1 FROM recordings", [super::now_ms()], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap()
        };
        let mut waits = vec![];
        for i in 1..super::ATTEMPTS {
            assert!(!super::fail(&conn, "r1", &format!("e{i}")).unwrap());
            let (status, wait) = row();
            assert_eq!(status, "receiving");
            waits.push((wait + 500) / 1000);
        }
        assert_eq!(waits, [30, 60, 120, 240, 480, 960, 1920, 3600, 3600]);
        assert!(super::fail(&conn, "r1", "last").unwrap());
        let row: (String, String) = conn
            .query_row("SELECT status, error FROM recordings", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap();
        assert_eq!(row, ("failed".into(), "last".into()));
    }

    #[test]
    fn turns_labeled_segments_into_lines() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("old.db");
        rusqlite::Connection::open(&path)
            .unwrap()
            .execute_batch(
                "CREATE TABLE recordings (id TEXT PRIMARY KEY, source TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'receiving');
                 INSERT INTO recordings (id, source, status) VALUES ('r1', 'laptop', 'done');
                 CREATE TABLE segments (id INTEGER PRIMARY KEY, recording TEXT NOT NULL,
                   window INTEGER NOT NULL, track TEXT NOT NULL, start_ms INTEGER NOT NULL,
                   end_ms INTEGER NOT NULL, text TEXT NOT NULL, speaker TEXT);
                 INSERT INTO segments (recording, window, track, start_ms, end_ms, text, speaker) VALUES
                   ('r1', 1, 'room', 0, 1000, 'hoi', 'room/S1'),
                   ('r1', 1, 'remote', 2000, 3000, 'precies wat hij zei', 'remote/S1'),
                   ('r1', 1, 'room', 2100, 3100, 'precies wat hij zei', 'room/S2');",
            )
            .unwrap();
        let conn = super::open(&path).unwrap();
        let lines = crate::assemble::lines(&conn, "r1").unwrap();
        let got: Vec<(&str, Option<&str>)> = lines.iter().map(|l| (l.text.as_str(), l.speaker.as_deref())).collect();
        assert_eq!(
            got,
            [("hoi", Some("room/S1")), ("precies wat hij zei", Some("remote/S1"))]
        );
        assert!(conn.prepare("SELECT speaker FROM segments").is_err());
        super::open(&path).unwrap();
        assert_eq!(crate::assemble::lines(&conn, "r1").unwrap().len(), 2, "once");
    }

    #[test]
    fn expires_outputs_of_long_done_recordings() {
        let conn = super::open(std::path::Path::new(":memory:")).unwrap();
        let day = 86_400_000;
        conn.execute_batch(&format!(
            "INSERT INTO recordings (id, source, status, done_ms) VALUES
               ('old', 'laptop', 'done', 0), ('new', 'laptop', 'done', {}), ('busy', 'laptop', 'windowed', NULL);
             INSERT INTO windows (id, recording, file, track, offset_ms, start_ms, end_ms) VALUES
               (1, 'old', 'a', 'room', 0, 0, 1), (2, 'new', 'a', 'room', 0, 0, 1), (3, 'busy', 'a', 'room', 0, 0, 1);
             INSERT INTO segments (recording, window, track, start_ms, end_ms, text) VALUES
               ('old', 1, 'room', 0, 1, 'a'), ('new', 2, 'room', 0, 1, 'b'), ('busy', 3, 'room', 0, 1, 'c');
             INSERT INTO turns (recording, track, start_ms, end_ms, speaker) VALUES
               ('old', 'room', 0, 1, 0), ('new', 'room', 0, 1, 0);
             INSERT INTO lines (recording, track, start_ms, end_ms, text) VALUES ('old', 'room', 0, 1, 'a');",
            2 * day
        ))
        .unwrap();
        super::expire_outputs(&conn, 8 * day).unwrap();
        let left = |sql: &str| -> String { conn.query_row(sql, [], |r| r.get(0)).unwrap() };
        assert_eq!(left("SELECT group_concat(recording) FROM segments"), "new,busy");
        assert_eq!(left("SELECT group_concat(recording) FROM turns"), "old,new");
        assert_eq!(left("SELECT group_concat(recording) FROM lines"), "old");
    }

    #[test]
    fn keeps_voices_of_an_old_db() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("old.db");
        rusqlite::Connection::open(&path)
            .unwrap()
            .execute_batch(
                "CREATE TABLE recordings (id TEXT PRIMARY KEY, source TEXT NOT NULL,
                   status TEXT NOT NULL DEFAULT 'receiving');
                 INSERT INTO recordings (id, source, status) VALUES ('r1', 'laptop', 'done');
                 CREATE TABLE clusters (recording TEXT NOT NULL, label TEXT NOT NULL,
                   embedding BLOB NOT NULL, PRIMARY KEY (recording, label));
                 CREATE TABLE voices (name TEXT NOT NULL, embedding BLOB NOT NULL,
                   recording TEXT NOT NULL, label TEXT NOT NULL, PRIMARY KEY (recording, label));
                 INSERT INTO voices VALUES ('Max', x'01', 'r1', 'room/S1');",
            )
            .unwrap();
        let conn = super::open(&path).unwrap();
        let row: (String, String, Option<i64>) = conn
            .query_row("SELECT name, label, start_ms FROM voices", [], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })
            .unwrap();
        assert_eq!(row, ("Max".into(), "room/S1".into(), None));
        // Several voices per cluster now.
        conn.execute(
            "INSERT INTO voices (name, embedding, recording, label, start_ms, end_ms)
             VALUES ('Max', x'02', 'r1', 'room/S1', 0, 1000)",
            [],
        )
        .unwrap();
        assert!(conn.prepare("SELECT core FROM clusters").is_ok());
        super::open(&path).unwrap();
    }

    #[test]
    fn adds_retry_at_to_an_old_db() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("old.db");
        rusqlite::Connection::open(&path)
            .unwrap()
            .execute_batch("CREATE TABLE recordings (id TEXT PRIMARY KEY, source TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'receiving')")
            .unwrap();
        let conn = super::open(&path).unwrap();
        conn.execute("INSERT INTO recordings (id, source) VALUES ('r1', 'laptop')", [])
            .unwrap();
        let r: i64 = conn
            .query_row("SELECT retry_at FROM recordings", [], |r| r.get(0))
            .unwrap();
        assert_eq!(r, 0);
        super::open(&path).unwrap();
    }
}
