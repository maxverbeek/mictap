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
- **Mixed cluster**: a cluster whose turns split into two halves that both carry real
  speech and are less alike than clusters assembly would merge: two people, or a room
  sharing one mic. Gets no suggestion, and its core teaches no voice (heard snippets
  still can).
- **Heard snippet**: a line of a label that was listened to before confirming its name,
  kept (only that speaker) or marked wrong. Each kept one becomes a voice.
- **Label**: a cluster's name within its recording, `room/S1`, numbered by first
  appearance.
- **Line**: a segment, or part of one, attributed to one label; what the transcript shows.
- **Assembly**: derives a recording's lines and clusters (with their means) from its model
  outputs: fold fragments, merge alike clusters, split segments into lines, merge tracks,
  drop echoes. Pure; knows nothing of names.
- **Core**: the mean embedding of a cluster's turns most alike to its mean, covering 70% of
  its speech; outliers and folded fragments left out.
- **Voice**: an embedding learned for a confirmed name: a cluster's core, or a snippet of
  it that was heard. A cluster can teach several; a person confirmed in five recordings
  has at least five.
- **Suggestion**: a name proposed for a cluster by matching it against every voice except the
  ones it taught itself (so other recordings, and other clusters and named lines of its own);
  recomputed whenever a voice is learned; only when one name clearly wins, else the cluster stays unknown. Part of
  name storage, not of assembly. Never learned or written to the transcript.
- **Confirmed name**: a name typed, or a suggestion accepted. Only confirmed names teach
  voices and appear in the transcript. A label can also be answered `?` (several people,
  or unsure): confirmed, so never asked or suggested again, but no name and no voice.
- **Line name**: a name set for one line, overriding its label's; `?` when that line is
  mixed or unsure. Kept by track and time, since lines are derived anew: a line takes the
  line name whose span holds its midpoint.
- **Taught line**: a line that taught a voice: it has a line name other than `?`, or it was
  a heard snippet when its label was confirmed. Its name is shown dark on the page.
- **Transcript**: the note written to the vault; derived, never read back.
