# mictap

Self-hosted meeting transcription. The laptop records, the VPS transcribes,
transcripts land in Obsidian. No GUI: everything you read or edit is markdown
in the vault.

```text
laptop (mictap daemon) --chunks over tailnet--> homeserver (mictap server)
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
  first 60 s, so a discarded recording never reaches the VPS.
- **Spool.** Chunks are written locally, uploaded over the tailnet and
  deleted once the server acknowledges them. Offline periods just queue up.

mictap hears what your mic hears, not what the meeting hears: muted side
discussions in the room are recorded and transcribed.

Control goes through an NDJSON Unix socket (like herdr and stalker):
`status`, `subscribe`, `start [--source X]`, `stop`, `discard`, `sources`.
The same verbs exist as CLI subcommands, plus `mictap upload <file>`.
barbell shows the recording indicator, start/stop, discard and a mic picker.

## Transcription (VPS)

Runs on homeserver (4 vCPU, 8 GB, no GPU). Speed doesn't matter, throughput
does: chunks are transcribed as they arrive, so the work starts during the
meeting instead of after it.

- Audio is re-segmented on speech pauses (VAD); upload chunks are arbitrary
  cuts and never used as transcription windows.
- whisper.cpp via `whisper-rs`. Model chosen by the spike (see backlog).
- Language is detected per window with a sticky bias: it only switches away
  from the current language on high confidence. Short code-switches are left
  to the model.
- `mictap/vocabulary.md` is fed to whisper as the initial prompt.
- **Every track is transcribed and diarized separately**: room speakers
  (mic) and remote speakers (app) are never clustered together. Diarization
  runs once per track over the whole recording after it finishes, so labels
  are consistent across chunks.
- Diarization uses sherpa-onnx. If the spike shows pyannote is clearly
  better, it becomes one isolated subprocess (audio in, turns + embeddings
  out as JSON).
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

- The file appears when the first chunk lands and fills in as transcription
  progresses. `status: failed` comes with an error line.
- Filenames are `YYYY-MM-DD HHMM Meeting.md`. Rename freely: the server finds
  transcripts by `id`, not by filename. Moving a file out of `mictap/` hands
  it over to you entirely.

Who owns what:

- **Body**: the server's until `status: done`, yours afterwards. The server
  never reads it back, so editing prose has no side effects.
- **`speakers`**: yours, editable any time. Filling in a name makes the
  server rewrite that speaker's line labels (and nothing else in the body)
  and add a wikilink to `attendees`.
- **`attendees`**: derived by the server from `speakers`.
- **`mictap/vocabulary.md`**: yours. Words whisper keeps getting wrong
  (names, clients, jargon), one per line. Nothing is inferred from your edits.

## Audio

Kept for 30 days on the VPS, then deleted. Transcripts stay.

The transcript's timestamps link to a mixdown of all tracks, served over the
tailnet. `#t=` is a standard media fragment, so the browser's own player
opens at that moment. Expired recordings return an "expired" page.

## Upload

For recordings made elsewhere (phone recorder app): a plain upload form
served by the server, or `mictap upload <file>`. An upload is one track, so
there's no alignment or echo dedupe. The start time comes from the file's
`creation_time` tag, else its mtime minus duration, else the upload time.

## API

Plain HTTP, tailnet only:

- `PUT /recordings/{id}/tracks/{track}/chunks/{n}`: idempotent, retries are
  safe.
- `POST /recordings/{id}/finish`
- `POST /recordings`: whole-file upload.
- `GET /r/{id}/audio.ogg`: mixdown.
- `GET /upload`: upload form.

## Deployment

Cargo workspace (`client`, `server`, `proto`) with a flake exposing both
binaries. The NixOS module lives in `~/nixconfig/modules/services/mictap.nix`.

- Listens on homeserver's tailnet only (`tailscale0` is a trusted
  interface). No public vhost.
- `mictap` user in the `webdav` group. `mictap/` in the vault is
  `2770 webdav:webdav` with a default ACL `g:webdav:rwx`, so files Remotely
  Save uploads there stay writable for mictap.
- `ProtectSystem=strict`, `ReadWritePaths=/srv/vault/mictap
  /var/lib/mictap`: the rest of the vault is off limits.
- Models are pinned with `fetchurl`.

## Non-goals

- Live transcription.
- Per-tab browser capture: one stream per browser is what PipeWire gives.
- Respecting in-meeting mute: not observable from PipeWire.
- Calendar lookups on the server.
- stalker integration: register-hours can read the vault directly.
