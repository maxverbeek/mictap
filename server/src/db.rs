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
    -- What the vault file last showed; 'done', 'failed' and 'gone' are final.
    written TEXT,
    -- The speakers map (label -> name, JSON) last applied to the vault file's lines.
    speakers TEXT
);
CREATE TABLE IF NOT EXISTS file_progress (
    recording TEXT NOT NULL REFERENCES recordings(id),
    file TEXT NOT NULL,
    done_ms INTEGER NOT NULL DEFAULT 0,
    complete INTEGER NOT NULL DEFAULT 0,
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
    Ok(conn.execute(
        "UPDATE recordings SET finished = 1 WHERE id = ?1",
        params![id],
    )? == 1)
}

pub fn fail(conn: &Connection, id: &str, error: &str) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE recordings SET status = 'failed', error = ?2 WHERE id = ?1",
        params![id, error],
    )?;
    Ok(())
}
