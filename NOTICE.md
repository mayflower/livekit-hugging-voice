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

## Pipecat Smart Turn

The optional Smart Turn v3.2 model and its inference contract are distributed
under the following BSD 2-Clause notice:

Copyright (c) 2024–2025, Daily

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
