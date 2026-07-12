# Contributing to Ready Video

This covers developing on Ready Video. For using the tool, see [README.md](README.md). For working conventions and pipeline internals, see [CLAUDE.md](CLAUDE.md).

## Development setup

Requires Python 3.11–3.13. (The transcription stack depends on PyTorch wheels not yet available for 3.14.)

```bash
git clone https://github.com/fmontes/ready-video.git
cd ready-video
python -m venv .venv
source .venv/bin/activate
pip install -e ".[transcription,dev]"
```

`-e` installs in editable mode; the `dev` extra adds pytest. `uv` also works (`uv sync`, `uv run pytest`).

Verify the toolchain:

```bash
ready-video doctor --install-missing
```

## Tests

```bash
.venv/bin/python -m pytest -q          # full suite (~65 tests, seconds)
.venv/bin/python -m pytest tests/test_renderer.py -q   # one module
```

Tests find `src/` via `pythonpath` in `pyproject.toml`, so no install is required just to run them.

## Transcription backend

Transcription uses **faster-whisper** (a CTranslate2 Whisper backend) for both text and word-level timestamps — deliberately *not* WhisperX/PyTorch. faster-whisper keeps the install to a handful of light packages (`av`, `ctranslate2`, `onnxruntime`) instead of a multi-GB torch/pyannote/torchcodec stack, and avoids the torchcodec/FFmpeg shared-library fragility that plagued the WhisperX path.

The trade-off: faster-whisper's `word_timestamps` are decoder-derived rather than forced-aligned, and run slightly early (measured median ≈ 66ms). `transcript.py` corrects for this with `_WORD_START_CORRECTION_S` / `_WORD_END_CORRECTION_S`. If you swap models or bump faster-whisper and karaoke timing feels off, re-measure the bias against a reference clip and adjust those constants.

The `transcription` extra pins `faster-whisper~=1.2`; `uv.lock` captures the exact resolved set. To update: bump `pyproject.toml`, run `uv lock`, then do a real `ready-video run` on a clip and spot-check subtitle timing (unit tests mock the backend, so they won't catch a timing regression). Commit `pyproject.toml` and `uv.lock` together.

## Project layout

```text
src/ready_video/
  cli.py         argparse entry point (ready-video console script)
  pipeline.py    stage orchestration: run / approve / doctor
  config.py      pydantic config, layering, presets, YAML generation
  ffmpeg.py      binary resolution + managed download, capability checks
  media.py       probe, sound-based silence analysis, speech.wav
  timeline.py    edit-decision-list timeline; edited↔source mapping
  transcript.py  faster-whisper transcription, word-timing correction & normalization
  refine.py      transcript-driven second-pass silence trim
  subtitles.py   ASS generation, presets, SRT export
  renderer.py    filtergraph construction; single final encode
  identity.py    content/config/job hashing, manifests, idempotency
tests/           pytest, one file per module
```

Read [CLAUDE.md](CLAUDE.md) before touching the pipeline — the timeline time-domain rules and the two-pass silence model are easy to get subtly wrong.

## Release readiness

This scaffold is useful for development and smoke testing, but it is not release-ready yet:

- The managed FFmpeg manifest pins archive URLs and SHA-256 values for macOS Intel, macOS Apple Silicon, Linux x86-64, Linux ARM64, and Windows. Release still needs a legal/license review for those exact binaries and their FFmpeg configure flags. Ready Video's MIT license does not cover FFmpeg or its codec stack.
- faster-whisper lives behind the `transcription` extra; first-run model downloads are not wrapped in a polished installer or progress UI.
- Real transcription needs cross-platform smoke coverage with packaged installs and real model downloads.
- Cross-platform smoke tests with real media, hardware acceleration, and packaged installs are still required.
- **PyPI name:** `https://pypi.org/pypi/ready-video/json` returned `404 Not Found` on 2026-07-11, so the distribution name appeared available then. Recheck immediately before release.

## Current scope

The scaffold includes strict configuration loading, timeline math, FFmpeg-backed probing/silence/audio/render stages, two-pass silence removal (sound + transcript), subtitle generation, and review artifacts.
