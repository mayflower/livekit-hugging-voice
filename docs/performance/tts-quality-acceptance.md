# TTS quality acceptance

`benchmarks/tts_quality_sentences.json` is the fixed multilingual corpus for the
CUDA TTS candidate. It contains 20 German and 10 English, French, and Italian
sentences, including numbers, times, names, abbreviations, punctuation, short
utterances, and longer utterances.

No local listening or ASR result is claimed yet. A candidate run must preserve
the generated WAV files and provenance, then record for every public voice:

- reviewer and date;
- profile, exact lock, image digest, GPU and chunk size;
- pronunciation and normalized ASR round-trip result for every corpus item;
- clicks or boundary artifacts;
- speaker consistency within and across sessions;
- any audible reference-audio leakage.

Acceptance requires valid non-empty 24-kHz mono PCM16, no non-finite model
output, documented normalized text fidelity, all five public voices reviewed,
no cross-session voice substitution, and no reference-audio leakage. Upstream
latency numbers are not local measurements.
