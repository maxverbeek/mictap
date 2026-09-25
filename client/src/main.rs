mod daemon;
mod machine;
mod pw;
mod recorder;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::UnixStream,
};

use daemon::Req;
use mictap::{remote, upload};

#[derive(Parser)]
#[command(version, about = "Meeting recorder")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the recorder (the systemd user service)
    Daemon,
    /// Start recording; manual recordings only stop manually
    Start {
        /// node.name of the mic, default source otherwise
        #[arg(long)]
        source: Option<String>,
    },
    Stop,
    /// Stop if recording, else start from the default source
    Toggle,
    /// Stop and delete the current recording
    Discard,
    Status,
    /// List audio sources
    Sources,
    /// Send an existing audio or video file to the server for transcription
    Upload {
        file: std::path::PathBuf,
    },
    /// List recordings on the server, and any still on this laptop
    Recordings {
        /// One JSON array, as barbell reads it
        #[arg(long)]
        json: bool,
    },
    /// Save a recording's audio (default: <id>.ogg)
    Download {
        id: String,
        file: Option<std::path::PathBuf>,
    },
    /// Delete a finished recording's audio and state from the server; the transcript stays
    Delete {
        id: String,
    },
    /// Diarize a finished recording again; its speaker names are reset
    Rediarize {
        id: String,
    },
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let req = match Cli::parse().cmd {
        Cmd::Daemon => return daemon::run().await,
        Cmd::Start { source } => Req::Start { source },
        Cmd::Stop => Req::Stop,
        Cmd::Toggle => Req::Toggle,
        Cmd::Discard => Req::Discard { id: None },
        Cmd::Status => Req::Status,
        Cmd::Sources => Req::Sources,
        Cmd::Upload { file } => {
            println!("{}", upload::whole(&upload::server(), &file).await?);
            return Ok(());
        }
        Cmd::Recordings { json } => {
            let mut all: Vec<serde_json::Value> = std::fs::read_dir(daemon::spool())
                .into_iter()
                .flatten()
                .flatten()
                .map(|d| serde_json::json!({"id": d.file_name().to_string_lossy(), "source": "laptop", "status": "on laptop"}))
                .collect();
            all.extend(remote::list(&upload::server()).await?);
            if json {
                println!("{}", serde_json::Value::from(all));
            } else {
                for r in &all {
                    println!("{}", remote::line(r));
                }
            }
            return Ok(());
        }
        Cmd::Download { id, file } => {
            let to = file.unwrap_or_else(|| format!("{id}.ogg").into());
            remote::download(&upload::server(), &id, &to).await?;
            println!("{}", to.display());
            return Ok(());
        }
        Cmd::Delete { id } => {
            remote::delete(&upload::server(), &id).await?;
            return Ok(());
        }
        Cmd::Rediarize { id } => {
            remote::rediarize(&upload::server(), &id).await?;
            return Ok(());
        }
    };
    let mut stream = UnixStream::connect(daemon::socket_path()).await?;
    stream
        .write_all(format!("{}\n", serde_json::to_string(&req)?).as_bytes())
        .await?;
    let mut line = String::new();
    BufReader::new(stream).read_line(&mut line).await?;
    print!("{line}");
    Ok(())
}
