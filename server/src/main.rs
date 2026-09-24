mod api;
mod db;
mod transcribe;
mod windows;

use std::{path::PathBuf, sync::Arc};

use anyhow::Context;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dir = PathBuf::from(std::env::var("STATE_DIRECTORY").context("STATE_DIRECTORY not set")?);
    let listen = std::env::var("MICTAP_LISTEN").unwrap_or_else(|_| "0.0.0.0:8765".into());
    let app = Arc::new(api::App::open(dir)?);
    tokio::spawn(windows::run(app.clone()));
    tokio::spawn(transcribe::run(app.clone()));
    let listener = tokio::net::TcpListener::bind(&listen)
        .await
        .with_context(|| format!("bind {listen}"))?;
    eprintln!("mictap-server listening on {listen}");
    axum::serve(listener, api::router(app)).await?;
    Ok(())
}
