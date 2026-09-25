# Backlog

## Recording (first)

Local only: recordings stay on the laptop, so meetings produce test
material for the spike before any server exists.

- [x] Cargo workspace (`client`, `server`) and flake. No `proto` crate:
  `meta.json` and the HTTP API are the contract.
- [x] Client: PipeWire watcher, allowlist, follow the captured source,
  record the app's playback streams as separate tracks.
- [x] Client: Opus tracks on one clock in a local spool, discard,
  2-minute stop grace.
- [x] Client: NDJSON socket and CLI (`start`, `stop`, `status`, `discard`,
  `sources`).
- [x] barbell: recording indicator, start/stop, discard, mic picker.
- [x] nixconfig: client user service on the laptop.

## Spike

Run on homeserver with the Team Sync recording (English, not Dutch)
and Piper TTS clips for Dutch. Results in `local/spike.md`.

- [x] Benchmark large-v3-turbo vs large-v3 (q5_0, `whisper-cli`): realtime
  factor and Dutch accuracy. Turbo, greedy.
- [x] Code-switching: a few seconds of English in a Dutch meeting. Dropped
  by both models under `-l nl`/`auto`.
- [x] Diarization: sherpa-onnx only. pyannote's HF models are gated, so the
  sherpa ONNX export of segmentation-3.0 is used; CAM++ zh/en embeddings,
  cluster threshold 0.9.
- [x] Voiceprint matching: same speaker across excerpts vs different
  speakers, match threshold 0.75. Same person as room vs remote is untested
  (no hybrid recording yet).
- [ ] Echo dedupe on a real hybrid recording: does the loudspeaker get its
  own cluster? Only a synthetic fixture so far.

## v1 (server)

- [x] Server: byte-range upload endpoints, SQLite state, files on disk.
- [x] Server: VAD re-segmentation, progressive transcription, sticky
  language detection, `vocabulary.md` as prompt.
- [x] Server: progressive vault file (`status`, `progress`, body).
- [x] Server: on finish, per-track diarization, chronological merge with
  `room`/`remote` tags, echo dedupe, `status: done`.
- [x] Server: watch `speakers` frontmatter, rewrite labels, derive
  `attendees` wikilinks.
- [x] Server: mixdown, `/r/{id}/audio.ogg`, timestamp links, 30-day
  retention with an expired page.
- [x] Server: upload form and `POST /recordings`, start time from metadata
  or a Meet-style filename date.
- [x] Client: upload the spool with retry, 60 s upload hold, `upload`
  subcommand.
- [x] nixconfig: `modules/services/mictap.nix` (server, webdav group,
  sandboxing; no ACL needed).

## Next

- [x] Learn names: store the embedding when a speaker is named, match new
  clusters against the library, pre-fill `speakers`. Older transcripts'
  blanks are not filled (decided: no, for now).

## Later

- [ ] LLM summaries and titles, as a separate pipeline.
- [ ] Android app (background recording, upload).
- [ ] Calendar title and attendees via stalker (maybe).
- [ ] Align several uploads of the same meeting (only if it comes up).
- [ ] Per-utterance language detection (the code-switching spike failed:
  English inside a Dutch window is dropped).
- [ ] Check mute behavior of Teams and Slack huddles (only if used).
- [ ] Better voice matching: in the two-halves test (older model and
  thresholds) only 3 of 11 known
  speakers were pre-filled (`local/learning-names.md`).
