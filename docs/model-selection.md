# Voice selection

The public API exposes five stable, operator-authored profiles: `warm_female`,
`clear_female`, `warm_male`, `clear_male`, and `friendly_neutral` for German,
English, French, and Italian. Qwen3-TTS speaks them in one of two operator-selected
modes (`speech.tts_mode`):

- `voice_clone` (default) drives the Qwen3-TTS base talker with one frozen,
  operator-provided reference recording per voice and language. The recording
  anchors the speaker identity, so the perceived person stays identical across
  segments, turns, and sessions. The recordings were rendered once from the
  VoiceDesign descriptions below, reviewed by ear, and are packaged with the
  service (`hugging_voice_service/voice_refs/`; `speech.voice_ref_dir` overrides
  the directory).
- `voice_design` rebuilds each voice from its text description on every segment
  with the VoiceDesign talker. It supports bounded per-session style
  instructions but provides no persistent speaker-identity state, so the voice
  audibly drifts between segments.

The CustomVoice model remains unused because it has no native German preset.

Per-session style instructions are honored only in `voice_design` mode, where
they are appended to the fixed description and explicitly scoped to delivery.
In `voice_clone` mode the frozen recording fully defines the speaker, so style
instructions are accepted but not applied; clients can read the active mode
from the `tts_mode` field of `session.created`. In both modes clients cannot submit
a base design, model path, speaker name, or reference audio.

Decoding uses sampling by default (`speech.generation.do_sample: true`), which
matches the upstream Qwen3-TTS `generation_config.json`. Greedy decoding is not
recommended: on long generations it drifts into near-silent output and
frequently misses the end-of-speech token.

## Regenerating the frozen reference recordings

The rendering command creates candidate recordings for every profile and
language from each voice's reference transcript, applies acoustic checks
(duration, level, drift, silences), and writes the selected takes plus a
`metadata.json` with full provenance. The procedure and checks are repeatable;
the audio itself is sampled, so takes differ between runs:

```bash
uv run --extra gpu python benchmarks/generate_voice_refs.py \
  --model-root .models --lock models/manifest.lock.json \
  --output-dir benchmarks/reports/voice-refs
```

Rendering (and the `voice_design` mode itself) needs the VoiceDesign talker,
which is not part of the shipped manifest: add
`- path: qwen-talker-1.7b-voicedesign-BF16.gguf` to the Qwen entry in
`models/manifest.yaml` and re-run the prefetch first.

Listen to every recording before committing it to
`services/gpu-service/src/hugging_voice_service/voice_refs/`; these files
freeze the public speaker identities. Reviewers should randomize the WAVs and
independently score intelligibility, native pronunciation, prosody, identity
consistency, artifacts, and listening comfort for every supported language.
Record the panel, hardware, model revision, and score sheet before changing a
frozen recording or the default profile.

For auditioning the `voice_design` mode itself, `benchmarks/voice_audition.py`
still renders all five profiles from their descriptions with identical text,
model revision, sample format, and GPU.
