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
    /// Stop and delete the current recording
    Discard,
    Status,
    /// List audio sources
    Sources,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let req = match Cli::parse().cmd {
        Cmd::Daemon => return daemon::run().await,
        Cmd::Start { source } => Req::Start { source },
        Cmd::Stop => Req::Stop,
        Cmd::Discard => Req::Discard { id: None },
        Cmd::Status => Req::Status,
        Cmd::Sources => Req::Sources,
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
