# Backlog

## Recording (first)

Local only: recordings stay on the laptop, so meetings produce test
material for the spike before any server exists.

- [ ] Cargo workspace (`client`, `server`, `proto`) and flake.
- [ ] Client: PipeWire watcher, allowlist, follow the captured source,
  record the app's playback streams as separate tracks.
- [ ] Client: chunked Opus tracks on one clock in a local spool, discard,
  2-minute stop grace.
- [ ] Client: NDJSON socket and CLI (`start`, `stop`, `status`, `discard`,
  `sources`).
- [ ] barbell: recording indicator, start/stop, discard, mic picker.
- [ ] nixconfig: client user service on the laptop.

## Spike

Run on homeserver with recordings from the recording phase, including a
hybrid (room + remote) meeting.

- [ ] Benchmark `whisper-rs`: large-v3-turbo vs large-v3 (quantized),
  realtime factor and Dutch accuracy.
- [ ] Code-switching: a few seconds of English in a Dutch meeting. Kept as
  English or translated? Turbo vs large-v3.
- [ ] Diarization: sherpa-onnx vs pyannote on the same meeting. Decide
  whether pyannote is worth one Python subprocess.
- [ ] Voiceprint matching across tracks: same person as room vs remote.
- [ ] Echo dedupe on the hybrid recording: does the loudspeaker get its own
  cluster?

## v1 (server)

- [ ] Server: chunk endpoints, SQLite state, spool on disk.
- [ ] Server: VAD re-segmentation, progressive transcription, sticky
  language detection, `vocabulary.md` as prompt.
- [ ] Server: progressive vault file (`status`, `progress`, body).
- [ ] Server: on finish, per-track diarization, chronological merge with
  `room`/`remote` tags, echo dedupe, `status: done`.
- [ ] Server: watch `speakers` frontmatter, rewrite labels, derive
  `attendees` wikilinks.
- [ ] Server: mixdown, `/r/{id}/audio.ogg`, timestamp links, 30-day
  retention with an expired page.
- [ ] Server: upload form and `POST /recordings`, start time from metadata.
- [ ] Client: upload the spool with retry, 60 s upload hold, `upload`
  subcommand.
- [ ] nixconfig: `modules/services/mictap.nix` (server, webdav group, ACL,
  sandboxing).

## Next

- [ ] Learn names: store the embedding when a speaker is named, match new
  clusters against the library, pre-fill `speakers`. Decide whether it also
  fills blanks in older transcripts.

## Later

- [ ] LLM summaries and titles, as a separate pipeline.
- [ ] Android app (background recording, chunked upload).
- [ ] Calendar title and attendees via stalker (maybe).
- [ ] Align several uploads of the same meeting (only if it comes up).
- [ ] Per-utterance language detection (only if the code-switching spike
  fails).
- [ ] Check mute behavior of Teams and Slack huddles (only if used).
