# Notices

`livekit-hugging-voice` is an independent project. It uses or interoperates with
third-party software and model artifacts listed in `THIRD_PARTY.md`; their licenses
and notices continue to apply to those components.

No model weights are vendored in this repository or copied into the runtime image.
The image builds a pinned llama.cpp binary and installs version-locked Python and
CUDA runtime components; `THIRD_PARTY.md`, `uv.lock`, installed distribution
metadata, upstream model licenses, and generated image digests are the release
inventory. Redistributors remain responsible for preserving the license and notice
files supplied by those packages, images, binaries, and separately fetched models.
