# Voice selection

The service uses Qwen3-TTS VoiceDesign because the CustomVoice model has no native
German preset. The public API exposes five stable, operator-authored profiles:
`warm_female`, `clear_female`, `warm_male`, `clear_male`, and
`friendly_neutral`. Each description is rendered for the selected language, so
the same product-facing profile works for German, English, French, and Italian.

Additional per-session style instructions are appended to the fixed description;
they are explicitly scoped to delivery and cannot replace or modify the fixed
speaker identity or native-language requirement. Clients cannot submit a base
design, model path, speaker name, or reference audio.

VoiceDesign decoding is greedy by default (`speech.generation.do_sample: false`)
for lower variation between independent sentence calls. Temperature, top-k,
top-p, and repetition penalty remain operator-configurable for controlled
auditions, but only affect generation when sampling is enabled. This reduces
variation; VoiceDesign still provides no persistent speaker-identity state.

The reproducible audition command renders all five profiles with identical text,
model revision, sample format, and GPU:

```bash
uv run --extra gpu python benchmarks/voice_audition.py \
  --model-root .models --lock models/manifest.lock.json \
  --output-dir benchmarks/reports/voice-audition
```

Reviewers should randomize the WAVs and independently score intelligibility,
native pronunciation, prosody, identity consistency across segments, artifacts,
and listening comfort for every supported language. Record the panel, hardware,
model revision, and score sheet before changing the default profile.
