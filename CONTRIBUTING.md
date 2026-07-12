# Contributing to Ready Video

This covers developing on Ready Video. For using the tool, see [README.md](README.md). For working conventions and pipeline internals, see [CLAUDE.md](CLAUDE.md).

## Development setup

Requires Python 3.11–3.13. (The transcription stack depends on PyTorch wheels not yet available for 3.14.)

```bash
git clone <repo-url> ready-video
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
.venv/bin/python -m pytest -q          # full suite (~85 tests, seconds)
.venv/bin/python -m pytest tests/test_renderer.py -q   # one module
```

Tests find `src/` via `pythonpath` in `pyproject.toml`, so no install is required just to run them.

## macOS: torchcodec / FFmpeg mismatch (optional, cosmetic)

WhisperX's dependency `pyannote` tries to use `torchcodec`, which loads FFmpeg **4–7** shared libraries. If your system FFmpeg (e.g. Homebrew's) is version 8, torchcodec cannot load and prints a long `libtorchcodec` warning — then WhisperX **silently falls back to a working audio path**, so transcription still succeeds. The warning is harmless.

To make torchcodec load and remove the warning, install the matching FFmpeg shared libs and expose them where torchcodec looks:

```bash
brew install ffmpeg@7    # provides libavutil.59 etc.; keg-only, does not replace your default ffmpeg
for lib in libavutil.59 libavcodec.61 libavformat.61 libavdevice.61 libavfilter.10 libswscale.8 libswresample.5; do
  ln -sf "$(brew --prefix ffmpeg@7)/lib/${lib}.dylib" "/opt/homebrew/lib/${lib}.dylib"
done
```

The `libavdevice.61` symlink is required by torchcodec's build but collides with PyAV's bundled `libavdevice`, producing two harmless objc "Class implemented in both" warnings for AVFoundation capture classes this pipeline never uses. The step is entirely optional — the fallback path produces identical transcription results.

## Project layout

```text
src/ready_video/
  cli.py         argparse entry point (ready-video console script)
  pipeline.py    stage orchestration: run / inbox / approve / doctor
  config.py      pydantic config, layering, presets, YAML generation
  ffmpeg.py      binary resolution + managed download, capability checks
  media.py       probe, sound-based silence analysis, speech.wav
  timeline.py    edit-decision-list timeline; edited↔source mapping
  transcript.py  WhisperX transcription + alignment, timing normalization
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
- WhisperX lives behind the `transcription` extra; first-run model downloads are not wrapped in a polished installer or progress UI.
- Real WhisperX transcription/alignment needs cross-platform smoke coverage with packaged installs and real model downloads.
- Inbox processing is intentionally one-shot and expects OS automation for polling.
- Cross-platform smoke tests with real media, hardware acceleration, and packaged installs are still required.
- **PyPI name:** `https://pypi.org/pypi/ready-video/json` returned `404 Not Found` on 2026-07-11, so the distribution name appeared available then. Recheck immediately before release.

## Current scope

The scaffold includes strict configuration loading, timeline math, FFmpeg-backed probing/silence/audio/render stages, two-pass silence removal (sound + transcript), subtitle generation, review artifacts, and one-shot inbox processing.
