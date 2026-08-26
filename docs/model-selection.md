# Voice selection

The public API exposes six stable, operator-authored profiles: `thorsten` (the
default), `warm_female`, `clear_female`, `warm_male`, `clear_male`, and
`friendly_neutral` for German, English, French, and Italian. Qwen3-TTS speaks them in one of two operator-selected
modes (`speech.tts_mode`):

- `voice_clone` (default) drives the Qwen3-TTS base talker with one frozen,
  operator-provided reference recording per voice and language. The recording
  anchors the speaker identity, so the perceived person stays identical across
  segments, turns, and sessions. Five of the six recordings were rendered once
  from the VoiceDesign descriptions below, reviewed by ear, and are packaged with
  the service (`hugging_voice_service/voice_refs/`; `speech.voice_ref_dir`
  overrides the directory). `thorsten` is different — see below.
- `voice_design` rebuilds each voice from its text description on every segment
  with the VoiceDesign talker. It supports bounded per-session style
  instructions but provides no persistent speaker-identity state, so the voice
  audibly drifts between segments.

The CustomVoice model remains unused because it has no native German preset.

### `thorsten` — the one voice cloned from a human

The five designed profiles are clones of the service's *own* output: each
recording was rendered from a VoiceDesign description by the same model that
later reads it back. That is two lossy steps away from a person, and it does not
produce native prosody in any language.

`thorsten` breaks that loop. It is an excerpt from
[Thorsten-Voice/TV-44kHz-Full](https://huggingface.co/datasets/Thorsten-Voice/TV-44kHz-Full)
(config `TV-2022.10-Neutral`, speaker Thorsten Müller, Rode Podcaster, **CC0-1.0**,
dataset revision `2b61b98fa8f99abd1ce1587b4bf413d6ebc217d5`): two sentences joined
with a 0.25 s pause, edges trimmed at -40 dBFS, peak-normalised to 0.95, 24 kHz
mono, 11.65 s. Full provenance is in `voice_refs/metadata.json`.

Two consequences follow from it being a real recording:

- **German only.** The `voice_clone` validator requires a reference for every
  configured language, so `en`, `fr`, and `it` reuse the German file. Cross-lingual
  in-context cloning keeps the speaker identity and lets the target text and
  language token drive the output — at the cost of an audible German accent
  outside German. Replacing those three with their own CC0 recordings is a drop-in
  change: three files, three transcripts, no schema or protocol impact.
- **It must never be regenerated.** `benchmarks/generate_voice_refs.py` skips
  every voice in `EXTERNALLY_SOURCED_VOICES` and carries its existing metadata
  entries over verbatim. Without that guard a plain rerun would overwrite a
  licensed human recording with synthetic audio and delete its provenance.

The transcript must match the audio word for word — it anchors the in-context
clone, and a paraphrase makes the voice drift.

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
