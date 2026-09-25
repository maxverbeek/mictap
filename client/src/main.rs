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
    Recordings,
    /// Save a recording's audio (default: <id>.ogg)
    Download {
        id: String,
        file: Option<std::path::PathBuf>,
    },
    /// Delete a finished recording's audio and state from the server; the transcript stays
    Delete {
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
        Cmd::Recordings => {
            if let Ok(dirs) = std::fs::read_dir(daemon::spool()) {
                for d in dirs.flatten() {
                    println!(
                        "{:<24} {:<16} {:<6} still on this laptop",
                        d.file_name().to_string_lossy(),
                        "",
                        "laptop"
                    );
                }
            }
            for r in remote::list(&upload::server()).await? {
                println!("{}", remote::line(&r));
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
