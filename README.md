# mictap

Self-hosted meeting transcription. The laptop records, the VPS transcribes,
you name speakers and replay lines on a small web page served by the VPS, and
finished transcripts land in Obsidian. The server only writes to the vault; it
never reads your edits back.

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
- **Tracks.** The mic and the monitor of the sink the meeting app plays to
  are recorded as separate Opus tracks, stamped on one clock. Other apps
  playing to that sink (Spotify) end up in the remote track: tapping the
  app's own streams stalls pipewire-pulse clients (Zen) mid-join.
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
`status`, `subscribe`, `start [--source X]`, `stop`, `toggle`, `discard`,
`sources`. The same verbs exist as CLI subcommands, plus `mictap upload
<file>`. `mictap toggle` is bound to `Mod+M r` (wlr-which-key). barbell is a
consumer: it shows the recording indicator (click stops, right-click
discards) and `r` on an input in the audio menu starts from that mic.

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
  are consistent over the whole meeting.
- Diarization is the `sherpa-onnx-offline-speaker-diarization` CLI
  (pyannote segmentation-3.0 ONNX, 3D-Speaker CAM++ zh/en embeddings) on a
  timeline wav of the track. sherpa over-splits (39 clusters for 3 people),
  so clusters under 10 s are folded into the most similar larger one, and
  clusters whose mean embeddings are at least `MICTAP_MERGE_THRESHOLD` alike
  are merged. Each segment goes to the speaker it overlaps most, and is cut
  into lines where the speaker changes for at least 1 s (words spread evenly
  over the segment, the cut moved to a nearby sentence or clause end). Each cluster's mean embedding comes from sherpa-onnx's C API and is
  kept in SQLite. No pyannote Python: its models are gated.
- Tracks are merged chronologically and each line is tagged `room` or
  `remote`.
- The models' outputs (whisper's segments, sherpa's turns, one embedding per
  turn) are stored as produced and never modified; lines, clusters and their
  mean embeddings are derived from them (`server/src/assemble.rs`) after every
  transcribed window and after diarization. See `CONTEXT.md` for the terms.
  Whisper's segments are deleted `MICTAP_OUTPUTS_DAYS` (default 7) after a
  recording is done; what was derived stays, and so do the turns, which
  naming lines weighs anew (see Learning names). `GET /recordings/{id}/outputs`
  exports the outputs as JSON, to replay a real meeting when tuning assembly.
- **Echo dedupe.** Remote voices coming out of a speaker get picked up by the
  mic. Mic segments that overlap remote speech within about 1 s and match
  its text are dropped. A room speaker cluster that is mostly echo (the
  loudspeaker) is dropped entirely.

## Transcripts (Obsidian)

Written to `/srv/vault/mictap/`, which Remotely Save syncs like
any other folder, once a recording is done (transcribed and diarized).

```markdown
---
id: 01J8X...
date: 2026-09-24 14:00
attendees: ["[[Max]]", "[[Jan]]", "[[Eva]]"]
link: http://homeserver:8765/#01J8X...
---

**Max** (room, [00:14:02](http://homeserver:8765/r/01J8X.../audio.ogg#t=842)): Zullen we zeggen dat het volgende sprint wordt?
**Eva** (room, [00:14:05](http://homeserver:8765/r/01J8X.../audio.ogg#t=845)): Ja, prima.
**?** (remote, [00:14:20](http://homeserver:8765/r/01J8X.../audio.ogg#t=860)): Hallo? Zijn jullie er nog?
```

- `attendees` are the confirmed names, then the names lines show; a line
  shows its name when taught or guessed, else `?` (see Learning names).
  `link` opens the recording on the web page.
- The whole file is the server's: it is rewritten whenever you name a
  speaker on the web page, and edits made in Obsidian are lost then. A
  recording without any speech, or one that failed, gets no file. Writes go
  to a temp file in `mictap/` that is renamed over the transcript.
- Filenames are `YYYY-MM-DD HHMM Meeting.md`. Rename freely: the server finds
  transcripts by `id`, not by filename. A file moved out of `mictap/` or
  deleted is not written again.
- **`mictap/vocabulary.md`** is the one file the server reads: words whisper
  keeps getting wrong (names, clients, jargon), one per line.

## Learning names

A name you type, or a suggestion you confirm, stores **voices**: embeddings of
that speaker to recognize them by later. Lines you listened to before
confirming are the evidence: each one you kept as only that speaker becomes a
voice of its own (its line voice, see below), and one you marked wrong teaches
nothing; if you marked all of them wrong, no voice is stored. Without heard
lines, or when none can teach (no line voice, turns expired), the voice is the
cluster's **core**: the mean of its turns most alike to its mean covering 70%
of its speech, so stray turns and folded fragments are left out. Listening to a confirmed
speaker again and saving replaces its voices. A **line name** (one line named
on its own) teaches a voice of its own, replaced when the line is renamed and
dropped when it is cleared or set to `?`. A label answered `?` teaches nothing
and is never suggested a name.

After diarizing, every line of at least 300 ms gets a **line voice**: CAM++'s
embedding of that line's own audio, kept durably. A line taught (named on its
own, or heard) learns its line voice; without one (older recordings), the
turns under it, and once those expired, nothing: the name is kept. Recordings
diarized before line voices existed get them computed in the background, one
at a time while nothing waits to be diarized, as long as their audio is kept
(one log line each), and their transcript is rewritten.

After diarizing a new recording, and in every recording whenever a voice is
learned, each cluster's core is matched against all voices except the ones it
taught itself (cosine, per name its most similar voice), and the
best name is **suggested** only when it is at least `MICTAP_MATCH_THRESHOLD`
(0.75) alike and more than `MICTAP_MATCH_MARGIN` (0.05) ahead of every other
name. Otherwise the cluster stays unknown, as a guest should. No name is
suggested twice within a track, nor where it is already confirmed for that
track. Suggestions are never learned; a name you set is never overwritten.
Accuracy is modest (see `local/learning-names.md`).

Each line is then **taught**, **guessed** or **unknown**, the same on the page,
in the API and in the transcript. Taught lines show the name they taught.
Otherwise the line's label gives a guess (its confirmed name, else its
suggestion), and so does its line voice: matched against all voices (but ones
taught from that very line), the best name when at least
`MICTAP_LINE_THRESHOLD` (0.55) alike and more than `MICTAP_LINE_MARGIN` (0.1)
ahead of every other name. Either one alone, or both agreeing, is the guess;
when they disagree the line is unknown and worth a look, as is a line with
neither or a line named `?`. Guesses are computed on every read, so teaching
one line can change others at once; a transcript is rewritten when its own
recording's names change.

A cluster can hold more than one voice: two people sherpa lumped together, or
a far meeting room where several people share one mic. Assembly splits each
cluster's turns into its two most different halves; when both carry at least
20% of the speech and are less alike than `MICTAP_MERGE_THRESHOLD`, the
cluster is **mixed**. A mixed cluster gets no suggestion, and naming it keeps
the name but learns no voice from its core, only from lines you heard and kept,
so a room's sound never becomes someone's voice.

Naming lines can show what the mixed check missed: a cluster in which another
name than its own is taught on at least 3 lines (with a line voice) is
**split**. Its lines are then guessed only between the names it holds that
way, plus its own: each name gets a centroid from the line voices it taught
there and the turns whose lines all carry it, and the label's own name also
its core. A line goes to the nearest centroid when more than
`MICTAP_SPLIT_MARGIN` (0.02) ahead of the next, else it is unknown; there is
no threshold, since every name in the cluster came through the same mic. The
core leaves out the turns holding a line named otherwise (or `?`), and the
voice a confirmed cluster taught is relearned from it whenever one of its
lines is named, so a second person in the cluster stops blurring its voice
in other recordings. On a recording where one cluster held two people, this
halved the clicks needed to name every line.

## Audio

Kept for 30 days on the VPS, then deleted. Transcripts stay.

The transcript's timestamps link to a mixdown of all tracks, served over the
tailnet. `#t=` is a standard media fragment, so the browser's own player
opens at that moment. Expired recordings return an "expired" page.

`mictap recordings` lists what the server has (date, transcription progress,
whether the audio is still kept) plus anything not uploaded yet.
`mictap download <id> [file]` saves the audio; `mictap delete <id>` removes a
finished recording's audio and state from the server, including the voices
learned from its named speakers, leaving the transcript.
`mictap rediarize <id>` runs sherpa and CAM++ on a finished recording again
while its audio is kept (after a diarization change), and whisper too once its
segments have expired: its turns, speaker names and their voices are dropped,
the lines are derived anew and embedded again, names are pre-filled anew from
other recordings, and the transcript is rewritten. Line names and their voices
are kept.

## Web page

`web/index.html`, plain JS with no build step, served at `/`: every
recording grouped by day with its attendees or its progress (refreshed while
anything is in progress), and per recording its transcript. Clicking a line's
time plays from there; clicking the rest of the line plays it and opens a menu
to name that line alone (`?` for mixed or unsure; picking its name again
clears it; for a label with no name or suggestion at all, it names the label),
and keys 1 to 9 name the line playing. Lines show taught names dark, guessed
ones light, and unknown ones as an orange `?`. A bar under the title counts
each; its "unknown" button opens the menu on the first unknown line, and
naming a line from its menu moves it on to the next unknown one (arrow keys
move it line by line). Beside the transcript, "Who is this?" asks about each
label with no name or suggestion in turn with three of its lines: pick a name
or `?` (several people, don't know), and ✓ confirms it, the lines not named on
their own taught as heard. Every change is saved at once and rewrites the
transcript in the vault.

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
- `GET /recordings`: every recording, newest first, with its status and
  progress, its `attendees` (as in the transcript) and how many labels are
  `unnamed`. `progress` is `null` once done or failed, else
  `{"step": "unstarted"}`, `{"step": "transcribing", "percent": 42}` or
  `{"step": "diarizing", "since_ms": ...}` (`null` while queued).
- `GET /recordings/{id}`: one recording, its `lines` and `speakers`
  (per label its confirmed `name`, `"?"` when answered unsure, or else its
  `suggested` one); `editable` once names can be set, `teachable` while its
  turns or line voices are kept so naming learns voices. Each line has its
  label (`speaker`), its `line_name` if set, its `state` (`"taught"`,
  `"guessed"` or `"unknown"`) and the `name` it shows (`null` when unknown).
  `counts` has how many lines are in each state: `{"taught": 3, "guessed":
  40, "unknown": 5}`.
- `PUT /recordings/{id}/speakers`: confirms speaker names, learns their
  voices and rewrites the transcript. 409 until the recording is done. Body,
  per label: `{"room/S1": {"name": "Max", "heard": [{"start_ms": 1000,
  "end_ms": 4000, "correct": true}]}}`. `name: ""` leaves the label unnamed
  and rejects its suggestion, `"?"` answers several people or unsure; `heard`
  (optional) lists the lines listened to, `correct` when only that speaker is
  in it (see Learning names). Answers with the voices each label learned:
  `{"room/S1": {"voices": 2}}`.
- `PUT /recordings/{id}/lines`: names one line,
  `{"track": "room", "start_ms": 1000, "end_ms": 4000, "name": "Eva"}`
  (`"?"` for mixed or unsure, `null` or `""` clears it), learns its voice and
  rewrites the transcript. 404 for an unknown recording, 409 until done.
  Answers `{"voices": 1}`, or 0 without a line voice once the turns expired.
- `GET /recordings/{id}/outputs`: its model outputs per track (whisper
  segments, sherpa turns with their embeddings), until they expire.
- `DELETE /recordings/{id}`: audio and state of a finished recording (409
  while transcribing). The transcript stays.
- `GET /r/{id}/audio.ogg`: mixdown.
- `GET /upload`: upload form.
- Any other path: a static file from `MICTAP_WEB` (the `web` option, default
  the package's copy of `web/`, `./web` when unset), so a different frontend
  can be dropped in without touching the server.

## Deployment

Cargo workspace (`client`, `server`); `meta.json` and the API above are the
contract between them. The flake exposes the package and two NixOS modules:

```nix
# the machine that transcribes
imports = [ mictap.nixosModules.server ];
services.mictap.server = {
  enable = true;
  listen = "0.0.0.0:8765";               # default 127.0.0.1:8765
  outputDir = "/srv/notes/vault/mictap";
  user = "syncthing";                     # the user that owns the vault
  group = "syncthing";
};

# the laptop
imports = [ mictap.nixosModules.recorder ];
services.mictap.recorder = {
  enable = true;
  server = "http://myserver:8765";
  allowlist = [ "firefox" "zoom" ];      # apps whose mic use starts a recording
};
```

- **No login.** Listen on loopback or a VPN interface only, and keep the port
  closed in the firewall.
- **Vault access.** Run the server as the user that owns the synced vault,
  so both can overwrite each other's files. The service runs in a chroot
  (`confinement`) holding only its store paths; `outputDir` and the state
  dir (SQLite, uploads, audio) are bind-mounted in, and nothing else of the
  host is visible. `PrivatePIDs`, `ProtectProc=invisible` and
  `SystemCallFilter=@system-service` keep it from seeing or ptracing the
  sync daemon that shares its user.
- **Neighbourly.** `Nice=19`, `CPUWeight=20`: transcription takes every core.
- **Models** are pinned with `fetchurl`; override them with
  `services.mictap.server.models.*`. Tunables go in `settings`:
  `MICTAP_CLUSTER_THRESHOLD` (0.9), `MICTAP_MERGE_THRESHOLD` (0.75),
  `MICTAP_MATCH_THRESHOLD` (0.75), `MICTAP_MATCH_MARGIN` (0.05),
  `MICTAP_LINE_THRESHOLD` (0.55), `MICTAP_LINE_MARGIN` (0.1),
  `MICTAP_ECHO_JACCARD` (0.6),
  `MICTAP_OUTPUTS_DAYS` (7).
- `url` (default `http://<hostname>:<port>`) is the base of the timestamp
  links and the one dotted host name the server accepts besides `*.ts.net`.

## Non-goals

- Live transcription.
- Per-tab browser capture: one stream per browser is what PipeWire gives.
- Respecting in-meeting mute: not observable from PipeWire.
- Calendar lookups on the server.
- stalker integration: register-hours can read the vault directly.
