//! Recordings on the server: list, download, delete.

use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use crate::upload::check;

pub async fn list(server: &str) -> Result<Vec<Value>> {
    Ok(check(reqwest::get(format!("{server}/recordings")).await?)
        .await?
        .json()
        .await?)
}

pub async fn download(server: &str, id: &str, to: &Path) -> Result<()> {
    let res = check(reqwest::get(format!("{server}/r/{id}/audio.ogg")).await?).await?;
    tokio::fs::write(to, res.bytes().await?).await?;
    Ok(())
}

pub async fn delete(server: &str, id: &str) -> Result<()> {
    let res = reqwest::Client::new()
        .delete(format!("{server}/recordings/{id}"))
        .send()
        .await?;
    check(res).await?;
    Ok(())
}

pub async fn rediarize(server: &str, id: &str) -> Result<()> {
    let res = reqwest::Client::new()
        .post(format!("{server}/recordings/{id}/rediarize"))
        .send()
        .await?;
    check(res).await?;
    Ok(())
}

/// One line of `mictap recordings`: id, date, source, state.
pub fn line(r: &Value) -> String {
    let min = |v: &Value| (v.as_i64().unwrap_or(0) + 59_999) / 60_000;
    let state = match r["status"].as_str() {
        Some("done") if r["transcript"].is_null() => "done, no speech".to_string(),
        Some("transcribing") => format!(
            "transcribing {}/{} min",
            r["done_ms"].as_i64().unwrap_or(0) / 60_000,
            min(&r["total_ms"])
        ),
        Some(s) => s.to_string(),
        None => "?".to_string(),
    };
    let audio = match r["audio"].as_str() {
        Some("ready") => ", audio kept",
        Some("expired") => ", audio expired",
        _ => "",
    };
    format!(
        "{:<24} {:<16} {:<6} {state}{audio}",
        r["id"].as_str().unwrap_or("?"),
        r["date"].as_str().unwrap_or(""),
        r["source"].as_str().unwrap_or(""),
    )
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    #[test]
    fn lines() {
        let r = json!({"id": "r1", "date": "2026-09-25 13:23", "source": "laptop",
                       "status": "transcribing", "done_ms": 125_000, "total_ms": 250_000,
                       "audio": null, "transcript": "a.md"});
        assert_eq!(
            super::line(&r),
            "r1                       2026-09-25 13:23 laptop transcribing 2/5 min"
        );
        let r = json!({"id": "r2", "date": null, "source": "upload", "status": "done",
                       "audio": "ready", "transcript": null});
        assert!(super::line(&r).ends_with("upload done, no speech, audio kept"));
    }
}
