mod api;
mod assemble;
mod audio;
mod db;
mod diarize;
mod names;
mod transcribe;
mod vault;
mod windows;

use std::{path::PathBuf, sync::Arc};

use anyhow::Context;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dir = PathBuf::from(std::env::var("STATE_DIRECTORY").context("STATE_DIRECTORY not set")?);
    let listen = std::env::var("MICTAP_LISTEN").unwrap_or_else(|_| "127.0.0.1:8765".into());
    let app = Arc::new(api::App::open(dir)?);
    tokio::spawn(windows::run(app.clone()));
    tokio::spawn(transcribe::run(app.clone()));
    tokio::spawn(diarize::run(app.clone()));
    tokio::spawn(vault::run(app.clone()));
    tokio::spawn(audio::run(app.clone()));
    let listener = tokio::net::TcpListener::bind(&listen)
        .await
        .with_context(|| format!("bind {listen}"))?;
    eprintln!("mictap-server listening on {listen}");
    axum::serve(listener, api::router(app))
        .with_graceful_shutdown(terminated())
        .await?;
    Ok(())
}

/// Under PrivatePIDs this is PID 1, which the kernel shields from signals it doesn't handle.
async fn terminated() {
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("SIGTERM handler");
    term.recv().await;
}
