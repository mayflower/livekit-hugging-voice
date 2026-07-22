# German voice selection

The public API exposes only `de_standard_01`. It currently maps to Qwen speaker
`Aiden`, the initial architecture pin. No listening result is claimed in this
checkout: the verified model volume needed to run the audition was unavailable.

The reproducible audition command renders Aiden, Ryan, Serena, and Sohee with the
same three German sentences, instruction, model revision, sample format, and GPU:

```bash
uv run --extra gpu python benchmarks/voice_audition.py \
  --model-root .models --lock models/manifest.lock.json \
  --output-dir benchmarks/reports/voice-audition
```

Reviewers should randomize the four WAVs and independently score intelligibility,
German pronunciation, prosody, artifacts, and listening comfort. Record the panel,
hardware, model revision, and score sheet in the generated metadata before changing
the internal `QWEN_SPEAKER` constant. A selection changes no protocol or plugin API.
