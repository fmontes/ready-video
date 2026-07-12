# Ready Video

Ready Video turns a talking-head clip into a social-media-ready video from the command line. It trims silence, transcribes speech, burns word-level subtitles, adds validated punch-in zooms, normalizes audio, and writes a deterministic vertical render.

```bash
ready-video run talking-head.mp4
```

That's it — no config file, no manual FFmpeg install. The output lands in `./edited`.

> **Status:** Ready Video is not yet on PyPI, so install is from source for now (see below). Working toward a `pip install ready-video` release — track what's left in [CONTRIBUTING.md](CONTRIBUTING.md).

## Install

Requires Python 3.11, 3.12, or 3.13.

```bash
git clone <repo-url> ready-video
cd ready-video
python -m venv .venv
source .venv/bin/activate
pip install ".[transcription]"
```

The `transcription` extra pulls in WhisperX, which does speech-to-text and word alignment. It is required for subtitles; without it, `ready-video run` stops with an install hint rather than producing a subtitle-free video.

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
ready-video run clip.mp4 --no-agent          # skip zoom suggestions
```

Review before committing to a final render — this writes a preview and a report you can inspect, tweak, then approve:

```bash
ready-video run /path/to/talking-head.mp4 --review
ready-video edit-zooms <job-id>              # optional: hand-edit the zooms
ready-video approve <job-id>                 # render using the reviewed config
```

**First run** downloads WhisperX and alignment models (controlled by the WhisperX / Hugging Face / PyTorch stack, not by Ready Video). These can be large. Later runs reuse the cached models.

## Output And File Locations

Runtime directories are created relative to the directory you run the command in, and are configurable:

```text
./edited   final renders and their JSON manifests
./work     per-job intermediate artifacts and review pages
./inbox    files waiting for one-shot inbox processing
./archive  successfully processed inbox originals
./failed   failed inbox originals and error logs
```

Each render is `./edited/<stem>_<job-id>.mp4` with an adjacent manifest. Re-running the same input with the same config is a no-op (the existing output is reused).

Caches live in your platform directories. On macOS:

```text
~/Library/Application Support/ready-video/config.yaml   user config
~/Library/Caches/ready-video/ffmpeg                     managed FFmpeg
```

WhisperX model caches use their upstream defaults (commonly `~/.cache/huggingface`, `~/.cache/torch`). Point `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TORCH_HOME`, or `XDG_CACHE_HOME` elsewhere if you need to.

## Configuration

Ready Video runs with sensible defaults and needs no config file. To customize, generate one:

```bash
ready-video init            # ./config.yaml for this project
ready-video init --user     # user-wide config
```

Settings are merged from these sources, later ones winning:

1. User config
2. `./config.yaml` in the current project
3. Per-input sidecar named like `clip.mp4.yaml`
4. Environment variables
5. `--config <path>`
6. CLI flags (`--preset`, `--language`, `--aspect`, `--no-agent`)

Any setting can be set via `READY_VIDEO_` plus section and key, separated by double underscores (values are parsed as YAML):

```bash
READY_VIDEO_RENDER__ASPECT=1:1 ready-video run clip.mp4
READY_VIDEO_SUBTITLES__PRESET=clean ready-video run clip.mp4
READY_VIDEO_INBOX_PROCESSING__STABILITY_SECONDS=20 ready-video inbox
```

To point at a specific FFmpeg, set both binaries together:

```bash
READY_VIDEO_FFMPEG=/opt/ffmpeg/bin/ffmpeg \
READY_VIDEO_FFPROBE=/opt/ffmpeg/bin/ffprobe \
ready-video doctor
```

## Zoom Suggestions And Privacy

Punch-in zooms can be proposed by a local CLI agent. The default is `auto`, which probes for supported CLIs in order and picks the first usable one, or skips zoom planning entirely:

```text
claude → codex → opencode → none
```

When an agent is selected, Ready Video passes transcript text and job context to that CLI, which **may forward it to that CLI's configured provider**. Ready Video does not broker or redact this traffic. If that isn't what you want:

```bash
ready-video run clip.mp4 --no-agent     # skip for one command
```

```yaml
# or make it the default in config.yaml
agent:
  backend: none
```

Pin a specific agent with `READY_VIDEO_AGENT__BACKEND=codex` (or `claude`, `opencode`).

## Inbox Processing

`ready-video inbox` is a one-shot command: it claims stable files from `./inbox`, processes each, archives successful originals, and preserves failures with an `error.log`. It is not a daemon — pair it with your OS scheduler to process files as they arrive.

Eligible extensions default to `mp4`, `mov`, `mkv`, `webm`. A file must be unchanged for `inbox_processing.stability_seconds` before it is claimed.

<details>
<summary><strong>macOS — launchd (run every minute)</strong></summary>

`~/Library/LaunchAgents/video.ready.inbox.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>video.ready.inbox</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/YOU/Developer/ready-video/.venv/bin/ready-video</string>
    <string>inbox</string>
    <string>--no-agent</string>
  </array>
  <key>WorkingDirectory</key>
  <string>/Users/YOU/ReadyVideoJobs</string>
  <key>StartInterval</key>
  <integer>60</integer>
  <key>StandardOutPath</key>
  <string>/Users/YOU/Library/Logs/ready-video-inbox.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/YOU/Library/Logs/ready-video-inbox.err</string>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/video.ready.inbox.plist
```

</details>

<details>
<summary><strong>Linux — systemd user timer</strong></summary>

`~/.config/systemd/user/ready-video-inbox.service`:

```ini
[Unit]
Description=Ready Video inbox pass

[Service]
Type=oneshot
WorkingDirectory=/home/YOU/ReadyVideoJobs
ExecStart=/home/YOU/ready-video/.venv/bin/ready-video inbox --no-agent
```

`~/.config/systemd/user/ready-video-inbox.timer`:

```ini
[Unit]
Description=Run Ready Video inbox every minute

[Timer]
OnBootSec=1min
OnUnitActiveSec=1min
Unit=ready-video-inbox.service

[Install]
WantedBy=timers.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now ready-video-inbox.timer
```

</details>

<details>
<summary><strong>Windows — Task Scheduler</strong></summary>

```powershell
schtasks /Create /TN "Ready Video Inbox" /SC MINUTE /MO 1 `
  /TR "C:\Users\YOU\ready-video\.venv\Scripts\ready-video.exe inbox --no-agent" `
  /ST 00:00
```

Set the task's "Start in" directory to the folder containing your `config.yaml`, `inbox`, `work`, `edited`, `archive`, and `failed` directories.

</details>

## FFmpeg Licensing

Ready Video is MIT-licensed, but FFmpeg is a separate project with separate licensing. System FFmpeg builds and managed binaries may include codecs, filters, or build flags that carry LGPL, GPL, patent, or redistribution obligations independent of this repository. Ready Video's MIT license does not cover FFmpeg or its codec stack. See [CONTRIBUTING.md](CONTRIBUTING.md) for the release-time license-review requirements.

## Contributing

Setting up a dev environment, the macOS torchcodec/FFmpeg note, project layout, and release readiness live in [CONTRIBUTING.md](CONTRIBUTING.md).
