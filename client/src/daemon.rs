use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::UnixListener,
    sync::{mpsc, oneshot, watch},
};

use crate::{
    machine::{Machine, Mode, GRACE},
    pw::{self, Graph},
    recorder::Recorder,
};

#[derive(Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "lowercase")]
pub enum Req {
    Status,
    Subscribe,
    Sources,
    Start { source: Option<String> },
    Stop,
    Toggle,
    Discard { id: Option<String> },
}

type Msg = (Req, oneshot::Sender<String>);

pub fn socket_path() -> PathBuf {
    PathBuf::from(std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".into())).join("mictap.sock")
}

pub(crate) fn spool() -> PathBuf {
    std::env::var("XDG_STATE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap()).join(".local/state"))
        .join("mictap/recordings")
}

pub async fn run() -> Result<()> {
    let path = socket_path();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path)?;
    let (tx, mut rx) = mpsc::channel::<Msg>(16);
    let (status_tx, status_rx) = watch::channel(String::new());
    // Before the uploader starts, so it sends an orphan's finish rather than waiting for one.
    if let Err(e) = crate::recorder::close_orphans(&spool()) {
        eprintln!("closing orphaned recordings: {e:#}");
    }
    tokio::spawn(serve(listener, tx.clone(), status_rx));
    tokio::spawn(mictap::upload::run(spool()));

    let mut machine = Machine::default();
    let mut rec: Option<Recorder> = None;
    let mut graph = Graph::default();
    let mut poll = tokio::time::interval(Duration::from_secs(2));
    loop {
        let mut discard = false;
        let mut reply = None;
        tokio::select! {
            _ = poll.tick() => match pw::dump().await {
                Ok(g) => graph = g,
                Err(e) => eprintln!("pw-dump: {e:#}"),
            },
            Some((req, r)) = rx.recv() => match req {
                Req::Sources => { let _ = r.send(sources(&graph)); }
                req => {
                    match req {
                        Req::Start { source } => machine.start(source),
                        Req::Stop => machine.stop(),
                        Req::Toggle if machine.session.is_some() => machine.stop(),
                        Req::Toggle => machine.start(None),
                        Req::Discard { id } if id.is_none() || id.as_deref() == rec.as_ref().map(|r| r.id.as_str()) => {
                            machine.stop();
                            discard = true;
                        }
                        _ => {}
                    }
                    reply = Some(r);
                }
            },
        }
        // Errors are logged, not returned: exiting would SIGKILL every pw-record mid-meeting.
        let res = match (machine.tick(&graph, Instant::now()), rec.take()) {
            (Some(tracks), r) => {
                let r = match r {
                    Some(r) => Ok(r),
                    None => Recorder::new(&spool()).inspect(|r| {
                        if machine.mode() == Some(Mode::Auto) {
                            notify(r.id.clone(), machine.app(), tx.clone());
                        }
                    }),
                };
                match r {
                    Ok(mut r) => {
                        let res = r.sync(&tracks, machine.app()).await;
                        rec = Some(r);
                        res
                    }
                    Err(e) => Err(e),
                }
            }
            (None, Some(r)) if discard => r.discard().await,
            (None, Some(r)) => r.finish().await,
            (None, None) => Ok(()),
        };
        if let Err(e) = res {
            eprintln!("recording: {e:#}");
        }
        let s = status(&machine, rec.as_ref(), Instant::now());
        status_tx.send_if_modified(|cur| {
            let changed = *cur != s;
            if changed {
                cur.clone_from(&s);
            }
            changed
        });
        if let Some(r) = reply {
            let _ = r.send(s);
        }
    }
}

fn status(m: &Machine, rec: Option<&Recorder>, now: Instant) -> String {
    match (&m.session, rec) {
        (Some(s), Some(r)) => json!({
            "recording": true,
            "id": r.id,
            "started_ms": r.started_ms,
            "mode": s.mode,
            "app": s.app,
            "stopping_in": s.lost_at.map(|t| GRACE.saturating_sub(now - t).as_secs()),
            "tracks": s.tracks.iter().map(|t| &t.key).collect::<Vec<_>>(),
        }),
        _ => json!({"recording": false}),
    }
    .to_string()
}

fn sources(g: &Graph) -> String {
    json!(g
        .sources()
        .iter()
        .map(|n| json!({
            "name": n.name,
            "description": n.description,
            "default": g.default_source.as_deref() == Some(n.name.as_str()),
        }))
        .collect::<Vec<_>>())
    .to_string()
}

/// Offers Discard for an auto-started recording, for as long as the upload
/// hold (60 s). A click after the recording ended is a no-op: the id no longer matches.
fn notify(id: String, app: Option<&str>, tx: mpsc::Sender<Msg>) {
    let body = format!("{} opened the mic", app.unwrap_or("A meeting app"));
    tokio::spawn(async move {
        let out = tokio::process::Command::new("notify-send")
            .args([
                "-a",
                "mictap",
                "-t",
                "60000",
                "-A",
                "discard=Discard",
                "-w",
                "Recording",
                &body,
            ])
            .output()
            .await;
        if out.is_ok_and(|o| o.stdout.starts_with(b"discard")) {
            let (r, _) = oneshot::channel();
            let _ = tx.send((Req::Discard { id: Some(id) }, r)).await;
        }
    });
}

async fn serve(listener: UnixListener, tx: mpsc::Sender<Msg>, status: watch::Receiver<String>) {
    loop {
        let Ok((stream, _)) = listener.accept().await else {
            continue;
        };
        let (tx, mut status) = (tx.clone(), status.clone());
        tokio::spawn(async move {
            let (r, mut w) = stream.into_split();
            let mut line = String::new();
            if BufReader::new(r).read_line(&mut line).await.is_err() {
                return;
            }
            let req: Req = match serde_json::from_str(&line) {
                Ok(req) => req,
                Err(e) => {
                    let _ = w
                        .write_all(format!("{}\n", json!({"error": e.to_string()})).as_bytes())
                        .await;
                    return;
                }
            };
            if let Req::Subscribe = req {
                loop {
                    let s = status.borrow_and_update().clone();
                    if w.write_all(format!("{s}\n").as_bytes()).await.is_err() || status.changed().await.is_err() {
                        return;
                    }
                }
            }
            let (rtx, rrx) = oneshot::channel();
            if tx.send((req, rtx)).await.is_ok() {
                if let Ok(s) = rrx.await {
                    let _ = w.write_all(format!("{s}\n").as_bytes()).await;
                }
            }
        });
    }
}
