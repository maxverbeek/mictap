# Context

Domain terms used across the code and docs.

- **Recording**: one meeting, identified by an id; uploaded as one or more files.
- **Track**: `room` (the mic) or `remote` (the meeting app's playback). Tracks are
  transcribed and diarized separately and never clustered together.
- **Window**: a stretch of a track's audio, at most 30 s, cut on speech pauses; the unit
  whisper transcribes.
- **Model outputs**: what the models produced for a recording, per track, before the app
  interprets it: whisper's **segments** (times and text), sherpa's **turns** (times
  and sherpa's own cluster id), and one CAM++ **embedding** per turn. Never modified;
  expire a configurable number of days (shorter than the audio) after the recording is
  done. Everything the app shows is **derived** from them and kept durably.
- **Rediarize**: discard a recording's turns and embeddings, run sherpa and CAM++ on its
  audio again, and derive anew. Needs the audio; whisper's segments are kept, or
  transcribed again when they expired.
- **Segment**: one piece of whisper output within a window. Often spans a reply by
  someone else.
- **Turn**: one stretch of a track attributed to a single sherpa cluster.
- **Cluster**: a group of turns the app treats as one speaker, after folding sherpa's
  fragments and merging alike ones. Its mean embedding is kept.
- **Label**: a cluster's name within its recording, `room/S1`, numbered by first
  appearance.
- **Line**: a segment, or part of one, attributed to one label; what the transcript shows.
- **Assembly**: derives a recording's lines and clusters (with their means) from its model
  outputs: fold fragments, merge alike clusters, split segments into lines, merge tracks,
  drop echoes. Pure; knows nothing of names.
- **Voice**: a named cluster's mean embedding, one per named cluster (a person named in
  five recordings has five).
- **Name lookup**: matches a recording's clusters against the voices of other recordings
  and pre-fills names; part of name storage, not of assembly.
- **Transcript**: the note written to the vault; derived, never read back.
