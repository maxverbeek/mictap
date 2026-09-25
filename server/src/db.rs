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
    -- What the vault file last showed; 'done', 'failed' and 'gone' are final.
    written TEXT,
    -- The speakers map (label -> name, JSON) last applied to the vault file's lines.
    speakers TEXT,
    -- audio.ogg: NULL until mixed down, then 'ready', 'failed' or 'expired'.
    audio TEXT
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
-- start_ms/end_ms are within the recording.
CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY,
    recording TEXT NOT NULL REFERENCES recordings(id),
    window INTEGER NOT NULL REFERENCES windows(id),
    track TEXT NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    text TEXT NOT NULL,
    speaker TEXT
);
-- embedding: the mean of the cluster's L2-normalized turn embeddings, f32 little-endian.
CREATE TABLE IF NOT EXISTS clusters (
    recording TEXT NOT NULL REFERENCES recordings(id),
    label TEXT NOT NULL,
    embedding BLOB NOT NULL,
    PRIMARY KEY (recording, label)
);
-- A cluster the user named: its embedding, copied from clusters.
CREATE TABLE IF NOT EXISTS voices (
    name TEXT NOT NULL,
    embedding BLOB NOT NULL,
    recording TEXT NOT NULL REFERENCES recordings(id),
    label TEXT NOT NULL,
    PRIMARY KEY (recording, label)
);
";

pub fn open(path: &Path) -> rusqlite::Result<Connection> {
    let conn = Connection::open(path)?;
    conn.execute_batch(SCHEMA)?;
    Ok(conn)
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

const ATTEMPTS: i64 = 3;

/// Counts a failed attempt at the current step; the third in a row fails the recording.
/// Returns true when it did. Workers reset `attempts` when a step succeeds.
pub fn fail(conn: &Connection, id: &str, error: &str) -> rusqlite::Result<bool> {
    conn.query_row(
        "UPDATE recordings SET attempts = attempts + 1, error = ?2,
         status = CASE WHEN attempts + 1 >= ?3 THEN 'failed' ELSE status END
         WHERE id = ?1 RETURNING status = 'failed'",
        params![id, error, ATTEMPTS],
        |r| r.get(0),
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn fails_on_the_third_attempt() {
        let conn = super::open(std::path::Path::new(":memory:")).unwrap();
        super::ensure_recording(&conn, "r1", "laptop").unwrap();
        assert!(!super::fail(&conn, "r1", "a").unwrap());
        assert!(!super::fail(&conn, "r1", "b").unwrap());
        let status: String = conn
            .query_row("SELECT status FROM recordings", [], |r| r.get(0))
            .unwrap();
        assert_eq!(status, "receiving");
        assert!(super::fail(&conn, "r1", "c").unwrap());
        let row: (String, String) = conn
            .query_row("SELECT status, error FROM recordings", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap();
        assert_eq!(row, ("failed".into(), "c".into()));
    }
}
