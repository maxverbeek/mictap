# mictap

Self-hosted meeting transcription. The laptop records, the VPS transcribes,
transcripts land in Obsidian. No GUI: everything you read or edit is markdown
in the vault.

```text
laptop (mictap daemon) --byte ranges over tailnet--> homeserver (mictap server)
                                                   |  whisper + diarization
                                                   v
                                     /srv/vault/mictap/*.md
                                                   |  Remotely Save
                                                   v
                                               Obsidian
```

## Recording (laptop)

`mictap daemon` is a Rust systemd user service. It survives barbell restarts,
so reloading the bar never kills a recording.

- **Auto-start.** When an allowlisted app (`Zen`, `chromium`, `zoom`, `slack`,
  ...) opens a mic capture stream in PipeWire, recording starts immediately.
  Non-allowlisted apps (clankertyper, voice notes) never trigger it.
- **Right mic.** It records the source the meeting app is actually capturing
  from, not the default. Switching mics in Meet is followed.
- **Tracks.** The mic and each of the meeting app's playback streams are
  recorded as separate Opus tracks, stamped on one clock. Other apps
  (Spotify) are not recorded.
- **Auto-stop.** 2 minutes after the capture stream disappears. Meet keeps
  the stream open while muted (verified in Zen), so muting doesn't split a
  meeting.
- **Manual.** Start/stop from barbell or the CLI, for physical meetings.
  Uses the default source unless another mic is picked. Manual starts only
  stop manually.
- **Discard.** A notification offers Discard. Nothing is uploaded during the
  first 60 s, so a discarded recording never reaches the VPS. A discard
  after that only deletes the laptop's copy; the server keeps what it got.
- **Spool.** Each track is a growing Opus file in a local recording dir,
  with a `meta.json` describing its segments. Every 10 s the daemon sends
  the new bytes of each file (up to 1 MB per request), then `meta.json`,
  and once the recording is finished and fully sent, `finish`; then it
  deletes the local dir. Failures back off and retry; offline periods just
  queue up.

mictap hears what your mic hears, not what the meeting hears: muted side
discussions in the room are recorded and transcribed.

Control goes through an NDJSON Unix socket (like herdr and stalker):
`status`, `subscribe`, `start [--source X]`, `stop`, `discard`, `sources`.
The same verbs exist as CLI subcommands, plus `mictap upload <file>`.
barbell shows the recording indicator, start/stop, discard and a mic picker.

## Transcription (VPS)

Runs on homeserver (4 vCPU, 8 GB, no GPU). Speed doesn't matter, throughput
does: audio is transcribed as it arrives, so the work starts during the
meeting instead of after it.

- New audio is decoded with `ffmpeg` and re-segmented on speech pauses
  (`whisper-vad-speech-segments`, silero v5.1.2) into windows of at most
  30 s. A window closes once 2 s of audio follow it or the recording is
  finished. Upload byte ranges are arbitrary cuts and never used as windows.
- One `whisper-cli` process per window (nixpkgs whisper.cpp,
  `large-v3-turbo-q5_0`, greedy, `-t 4`), one window at a time. Model load
  is about 1% of a window, so no resident server. See `local/spike.md`.
- Language is detected per window with a sticky bias: the first window
  detected at p >= 0.8 seeds it, and it only switches at p >= 0.8. Short
  code-switches are left to the model (in practice they get dropped).
- `mictap/vocabulary.md` is fed to whisper as the initial prompt.
- **Every track is transcribed and diarized separately**: room speakers
  (mic) and remote speakers (app) are never clustered together. Diarization
  runs once per track over the whole recording after it finishes, so labels
  are consistent across chunks.
- Diarization is the `sherpa-onnx-offline-speaker-diarization` CLI
  (pyannote segmentation-3.0 ONNX, 3D-Speaker CAM++ zh/en embeddings) on a
  timeline wav of the track. Each segment goes to the speaker it overlaps
  most. Each cluster's mean embedding comes from sherpa-onnx's C API and is
  kept in SQLite. No pyannote Python: its models are gated.
- Tracks are merged chronologically and each line is tagged `room` or
  `remote`.
- **Echo dedupe.** Remote voices coming out of a speaker get picked up by the
  mic. Mic segments that overlap remote speech within about 1 s and match
  its text are dropped. A room speaker cluster that is mostly echo (the
  loudspeaker) is dropped entirely.

## Transcripts (Obsidian)

Written to `/srv/vault/mictap/`, which Remotely Save syncs like
any other folder.

```markdown
---
id: 01J8X...
date: 2026-09-24 14:00
duration: 52m
source: laptop            # or upload
status: done              # transcribing | done | failed
progress: 52/52 min
attendees: ["[[Max]]", "[[Jan]]", "[[Eva]]"]
speakers:
  room/S1: Max
  room/S2: Eva
  remote/S1: Jan
  remote/S2: ""           # fill in to name
---

**Max** (room, [00:14:02](http://homeserver:8765/r/01J8X.../audio.ogg#t=842)): Zullen we zeggen dat het volgende sprint wordt?
**Eva** (room, [00:14:05](http://homeserver:8765/r/01J8X.../audio.ogg#t=845)): Ja, prima.
**Jan** (remote, [00:14:20](http://homeserver:8765/r/01J8X.../audio.ogg#t=860)): Hallo? Zijn jullie er nog?
```

- The file appears when the first audio lands and fills in as transcription
  progresses. `status: failed` comes with an `error:` line. Writes go to a
  temp file in `mictap/` that is renamed over the transcript.
- Filenames are `YYYY-MM-DD HHMM Meeting.md`. Rename freely: the server finds
  transcripts by `id`, not by filename. Moving a file out of `mictap/` hands
  it over to you entirely.

Who owns what:

- **Body**: the server's until `status: done`, yours afterwards. The server
  never reads it back, so editing prose has no side effects.
- **`speakers`**: yours, editable any time. Filling in a name makes the
  server rewrite that speaker's line labels (and nothing else in the body)
  and add a wikilink to `attendees`. The server polls every 30 s.
- **`attendees`**: derived by the server from `speakers`.
- **`mictap/vocabulary.md`**: yours. Words whisper keeps getting wrong
  (names, clients, jargon), one per line. Nothing is inferred from your edits.

## Learning names

Naming a speaker stores that cluster's voice embedding. After diarizing a
new recording, each cluster is matched against the stored voices (cosine,
best match at or above `MICTAP_MATCH_THRESHOLD`, default 0.75), and a match
pre-fills `speakers` and labels its lines. Unmatched clusters stay `""`, and
a name you set is never overwritten. Only new transcripts are pre-filled:
blanks in older ones are left alone. Accuracy is modest (see
`local/learning-names.md`): expect misses more than wrong names.

## Audio

Kept for 30 days on the VPS, then deleted. Transcripts stay.

The transcript's timestamps link to a mixdown of all tracks, served over the
tailnet. `#t=` is a standard media fragment, so the browser's own player
opens at that moment. Expired recordings return an "expired" page.

## Upload

For recordings made elsewhere (phone recorder app): a plain upload form
served by the server, or `mictap upload <file>`. An upload is one track, so
there's no alignment or echo dedupe. The start time comes from the file's
`creation_time` tag, else a Meet-style date in the filename
(`YYYY_MM_DD HH_MM`, Amsterdam time), else its mtime minus duration (only
`mictap upload` sends the mtime), else the upload time.

## API

Plain HTTP, tailnet only. No login, so cross-site browser requests (by
`Origin`/`Sec-Fetch-Site`) and Hosts outside the tailnet's names get 403.

- `PUT /recordings/{id}/files/{name}?offset=N`: byte ranges of a growing
  file. `offset` equal to the stored size appends; a range the server
  already has is a 200; `offset` past the size is a 409 with `{"size":N}`,
  so the client resumes from there. Retries are safe.
- `PUT /recordings/{id}/meta`: stores `meta.json` (segments, offsets,
  `finished`).
- `POST /recordings/{id}/finish`: 409 before `meta`, idempotent after.
  Data for a finished recording gets 410.
- `POST /recordings?filename=&mtime_ms=`: whole-file upload, raw body or
  multipart.
- `GET /r/{id}/audio.ogg`: mixdown.
- `GET /upload`: upload form.

## Deployment

Cargo workspace (`client`, `server`) with a flake exposing both binaries;
`meta.json` and the API above are the contract between them. The NixOS
module lives in `~/nixconfig/modules/services/mictap.nix`.

- Listens on `0.0.0.0:8765` (`MICTAP_LISTEN`), but only `tailscale0` is a
  trusted interface, so it's reachable over the tailnet only. No public
  vhost.
- `mictap` user in the `webdav` group, `UMask=0002`. `mictap/` in the vault
  is `2770 webdav:webdav`. No ACL: transcripts are replaced by rename,
  which needs only write access to the folder, so it doesn't matter that
  rclone writes 0644 files.
- `ProtectSystem=strict`, `TemporaryFileSystem=/srv/vault:ro` plus
  `BindPaths=/srv/vault/mictap`: group `webdav` can read the whole
  vault, so the rest of it is hidden, not just read-only. State (SQLite,
  uploads, audio) is in `StateDirectory=mictap`, mode 0750.
- `Nice=19`, `CPUWeight=20`, whisper `-t 4`: homeserver serves other things.
- `TZ=Europe/Amsterdam`; `ffmpeg`, `whisper-cpp` and `sherpa-onnx` on the
  unit's `PATH`.
- Models are pinned with `fetchurl` and passed as `MICTAP_WHISPER_MODEL`,
  `MICTAP_VAD_MODEL`, `MICTAP_SEG_MODEL`, `MICTAP_EMB_MODEL`. Tunables:
  `MICTAP_CLUSTER_THRESHOLD` (0.9), `MICTAP_MATCH_THRESHOLD` (0.75),
  `MICTAP_ECHO_JACCARD` (0.6).
- The laptop's user service points at the server with `MICTAP_SERVER`
  (default `http://homeserver:8765`).

## Non-goals

- Live transcription.
- Per-tab browser capture: one stream per browser is what PipeWire gives.
- Respecting in-meeting mute: not observable from PipeWire.
- Calendar lookups on the server.
- stalker integration: register-hours can read the vault directly.
