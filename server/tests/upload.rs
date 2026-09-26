use std::{
    path::Path,
    process::{Child, Command},
    time::{Duration, Instant},
};

use mictap::upload::Uploader;

struct Server(Child);

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

fn write_meta(dir: &Path, finished: bool) {
    let meta = serde_json::json!({
        "id": "20260924T200000Z", "started_ms": 1_790_280_000_000u64, "app": "Zen", "finished": finished,
        "segments": [
            {"file": "00-mic.oga", "key": "mic", "target": "laptop", "offset_ms": 0, "end_ms": 8_000},
            {"file": "01-app-7.oga", "key": "app-7", "target": "7", "offset_ms": 1_000, "end_ms": 9_000},
            {"file": "02-mic.oga", "key": "mic", "target": "headset", "offset_ms": 8_000, "end_ms": 16_000},
        ],
    });
    std::fs::write(dir.join("meta.json"), meta.to_string()).unwrap();
}

/// Needs ffmpeg, whisper-cli, whisper-vad-speech-segments and
/// sherpa-onnx-offline-speaker-diarization on PATH, and the MICTAP_*_MODEL env vars.
#[tokio::test]
#[ignore]
async fn uploaded_spool_is_transcribed() {
    let tmp = tempfile::tempdir().unwrap();
    let (state, vault, spool) = (
        tmp.path().join("state"),
        tmp.path().join("vault"),
        tmp.path().join("spool"),
    );
    for d in [&state, &vault, &spool] {
        std::fs::create_dir(d).unwrap();
    }
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let _server = Server(
        Command::new(env!("CARGO_BIN_EXE_mictap-server"))
            .env("STATE_DIRECTORY", &state)
            .env("MICTAP_VAULT", &vault)
            .env("MICTAP_LISTEN", format!("127.0.0.1:{port}"))
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + Duration::from_secs(10);
    while std::net::TcpStream::connect(("127.0.0.1", port)).is_err() {
        assert!(Instant::now() < deadline, "server did not start");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Two tracks; the mic switches from laptop to headset at 8 s.
    let files = ["00-mic.oga", "01-app-7.oga", "02-mic.oga"].map(|f| (f, fixture("dutch-8s.oga")));
    let id = "20260924T200000Z";
    let dir = spool.join(id);
    std::fs::create_dir(&dir).unwrap();
    let mut up = Uploader::new(format!("http://127.0.0.1:{port}"), spool.clone());

    // Mid-recording: the files are still growing.
    for (name, data) in &files[..2] {
        std::fs::write(dir.join(name), &data[..data.len() / 2]).unwrap();
    }
    write_meta(&dir, false);
    up.pass(1_790_280_120_000).await.unwrap();

    for (name, data) in &files {
        std::fs::write(dir.join(name), data).unwrap();
    }
    write_meta(&dir, true);
    up.pass(1_790_280_180_000).await.unwrap();
    assert!(!dir.exists());

    let got = state.join("recordings").join(id);
    for (name, data) in &files {
        assert!(std::fs::read(got.join(name)).unwrap() == *data, "{name} differs");
    }

    let deadline = Instant::now() + Duration::from_secs(30 * 60);
    loop {
        let text: String = std::fs::read_dir(&vault)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|x| x == "md"))
            .filter_map(|e| std::fs::read_to_string(e.path()).ok())
            .collect();
        // Written once diarized.
        if text.contains(&format!("id: {id}")) {
            assert!(text.contains("(remote, "), "{text}");
            break;
        }
        assert!(Instant::now() < deadline, "not done: {text}");
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}
