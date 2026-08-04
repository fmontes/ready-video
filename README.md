# Ready Video

Ready Video turns a talking-head clip into a social-media-ready video from the command line. It trims silence, transcribes speech, burns word-level subtitles, normalizes audio, and writes a deterministic vertical render.

```bash
ready-video run talking-head.mp4
```

That's it — no config file, no manual FFmpeg install. The output lands in `./edited`.

> **Status:** Ready Video is not yet on PyPI, so install is from source for now (see below). Working toward a `pip install ready-video` release — track what's left in [CONTRIBUTING.md](CONTRIBUTING.md).

## Install

Requires Python 3.11, 3.12, or 3.13.

Transcription uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — a lightweight, CPU-friendly Whisper backend (no PyTorch). The `transcription` extra is a modest download (roughly a couple hundred MB, plus the speech model on first run). Install with [`uv`](https://docs.astral.sh/uv/) so the pinned `uv.lock` is honored:

```bash
git clone https://github.com/fmontes/ready-video.git
cd ready-video
uv sync --extra transcription
```

That creates a `.venv` with the locked dependency set. Run the tool with `uv run ready-video ...`, or activate the venv (`source .venv/bin/activate`) and call `ready-video` directly.

<details>
<summary>Plain pip (no uv)</summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install ".[transcription]"
```

The `transcription` extra pins compatible version ranges, but pip does not use `uv.lock`, so resolution can still drift. `uv` is recommended for a reproducible install.

</details>

The `transcription` extra provides faster-whisper (speech-to-text + word-level timestamps) and is **required** for subtitles — without it, `ready-video run` stops with an install hint rather than producing a subtitle-free video.

Then verify your machine and let Ready Video fetch a managed FFmpeg if you don't have a compatible one:

```bash
ready-video doctor --install-missing
```

You do not need to install FFmpeg yourself. If no compatible FFmpeg/ffprobe pair is on your `PATH`, Ready Video downloads a checksum-verified managed pair into your user cache.

## Quickstart

Render a clip to a vertical (9:16) video:

```bash
ready-video run /path/to/talking-head.mp4
```

Common one-off overrides:

```bash
ready-video run clip.mp4 --preset clean --language en --aspect 9:16
ready-video run clip.mp4 --preset minimal --aspect 1:1
```

Review before committing to a final render — this writes a preview and a report you can inspect, then approve:

```bash
ready-video run /path/to/talking-head.mp4 --review
ready-video approve <job-id>                 # render using the reviewed config
```

**First run** downloads the Whisper speech model (via faster-whisper / Hugging Face, not controlled by Ready Video). Later runs reuse the cached model.

## Output And File Locations

Finished videos land in `./edited` by default (configurable via `paths.edited`). Each render is `./edited/<stem>_<job-id>.mp4` with an adjacent JSON manifest and a `<stem>_<job-id>.txt` plain-text transcript. The transcript is timestamped one line per segment in edited time (matching the final video), e.g. `[00:03.20] So today we're building...`. Re-running the same input with the same config is a no-op — the existing output is reused, and the transcript sidecar is regenerated if it is missing.

Intermediate per-job artifacts (timeline, transcript, subtitles, review pages) live in an internal work directory under your platform cache and are not something you normally touch.

Caches live in your platform directories. On macOS:

```text
~/Library/Application Support/ready-video/config.yaml   user config
~/Library/Caches/ready-video/ffmpeg                     managed FFmpeg
```

Speech-model caches use Hugging Face defaults (commonly `~/.cache/huggingface`). Point `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, or `XDG_CACHE_HOME` elsewhere if you need to.

## Configuration

Ready Video runs with sensible defaults and needs no config file. To customize, generate one:

```bash
ready-video init            # ./config.yaml for this project
ready-video init --user     # user-wide config
```

Settings are merged from these sources, later ones winning:

1. User config
2. `./config.yaml` in the current project
3. Environment variables
4. `--config <path>`
5. CLI flags (`--preset`, `--language`, `--aspect`)

Any setting can be set via `READY_VIDEO_` plus section and key, separated by double underscores (values are parsed as YAML):

```bash
READY_VIDEO_RENDER__ASPECT=1:1 ready-video run clip.mp4
READY_VIDEO_SUBTITLES__PRESET=clean ready-video run clip.mp4
```

To point at a specific FFmpeg, set both binaries together:

```bash
READY_VIDEO_FFMPEG=/opt/ffmpeg/bin/ffmpeg \
READY_VIDEO_FFPROBE=/opt/ffmpeg/bin/ffprobe \
ready-video doctor
```

## Privacy

Ready Video runs entirely on your machine. It does not send your video, audio, or transcript to any network service. The only network activity is first-run downloads: the managed FFmpeg binaries and the Whisper speech model (both cached and reused afterward). No API keys are read or required.

## FFmpeg Licensing

Ready Video is MIT-licensed, but FFmpeg is a separate project with separate licensing. System FFmpeg builds and managed binaries may include codecs, filters, or build flags that carry LGPL, GPL, patent, or redistribution obligations independent of this repository. Ready Video's MIT license does not cover FFmpeg or its codec stack. See [CONTRIBUTING.md](CONTRIBUTING.md) for the release-time license-review requirements.

## Contributing

Setting up a dev environment, the transcription-backend notes, project layout, and release readiness live in [CONTRIBUTING.md](CONTRIBUTING.md).
