# CLAUDE.md

See `README.md` for what the tool does, install, and CLI usage. This file covers only what's needed to work in the code.

## Tests

```bash
.venv/bin/python -m pytest -q            # full suite (~85 tests, seconds)
.venv/bin/python -m pytest tests/test_renderer.py -q   # one module
```

Tests find `src/` via `pythonpath` in `pyproject.toml` — no install needed to run them. `uv run pytest` also works.

## Core architecture — read before touching the pipeline

The pipeline is a **canonical edit-decision-list timeline**, not a chain of cut videos. Stages write immutable artifacts into `work/<job_id>/` (`source.json → timeline.json → speech.wav → transcript.json → subs.ass → final mp4 + manifest`). The final video is encoded from the **original source exactly once**; `speech.wav`, previews, and the loudnorm measurement pass are analysis-only.

**All transcript and subtitle timestamps are in EDITED time** (after silence removal). `timeline.json` is the single source of truth for converting between edited time and original source time. Two time domains exist and must never be confused:

- `Timeline.edited_to_source(t)` / `source_to_edited(t)` — single-point mapping. **At a kept-segment boundary an edited timestamp is ambiguous** (it is both the end of one segment and the start of the next) and `edited_to_source` resolves it to the *previous* segment. Do not map a range by calling this on each endpoint — that straddles removed silence.
- `Timeline.edited_range_to_source_ranges(start, end)` — segment-aware range mapping. **Use this for anything that trims a range of edited time back to source** (the renderer does). This was the source of the "silence not removed / subs out of sync" bug: the renderer must build spans in edited time and map them with this method.

The renderer (`renderer.py`) splits the edited timeline at every kept-segment boundary, maps each span to one source range, trims/scales/crops/concats, then burns subtitles **after concat** so subtitle timestamps stay in edited time.

**Silence removal is two passes** (`refine.py`). Pass 1 (`media.analyze_silence`) is sound-based FFmpeg `silencedetect` → `sound_timeline`. Transcription runs on that cut audio, so the transcript is in pass-1 edited time. Pass 2 (`refine_timeline_with_transcript`) compresses inter-word pauses that stayed above the dB threshold (room tone/breath), producing the final tightened `timeline`. Pass 2 only *shortens* gaps and only across **aligned** word boundaries — it never clips speech and never crosses an `interpolated` (uncertain) word. After pass 2 the transcript is re-timed onto the new axis (`retime_transcript`) before subtitles are generated, so everything downstream is already on the tightened timeline. Controlled by `silence.transcript_trim*` config. `work/<job_id>/timeline.json` holds the tightened result; the pass-1 timeline is not persisted separately.

## Conventions & invariants

- **Never mutate the `run` input.** The source passed to `ready-video run` is never edited/moved/deleted, even on failure. Inbox files follow the claim→archive/failed lifecycle instead.
- **Atomic artifacts.** Write to a temp path and rename (`io.atomic_write_*`). A completed stage artifact is never modified in place; re-running a stage replaces only that artifact.
- **Errors** are `ReadyVideoError(code, message, remediation?, details?)` with stable UPPERCASE codes (e.g. `NO_SPEECH_DETECTED`, `TIMELINE_MAPPING_FAILED`, `UNSUPPORTED_HDR`, `RENDER_FAILED`). Add a code rather than raising bare exceptions in pipeline code.
- **No shell strings.** Build FFmpeg argument arrays and run without a shell (`shell=False`). Never interpolate paths into a command string.
- **Idempotency.** `job_id` = first 12 hex of SHA-256(content_sha + config_sha + PIPELINE_VERSION). Same content + same *resolved* config ⇒ skip if a valid manifest+media already exist. `PIPELINE_VERSION` lives in `__init__.py`; bump it when a change alters output for identical inputs.
- **Config safety.** Unknown YAML keys are hard errors (with a "did you mean"). Secret-like keys (`api_key`, `token`, `secret`, `password`) are rejected. `null` means "inherit from preset/default" only for fields explicitly declared inheritable (subtitle fields); elsewhere it is invalid.
- **Fully local.** The pipeline sends nothing to the network except first-run binary/model downloads. There is no agent/LLM step and no transcript ever leaves the machine.

## When changing the renderer or timeline

The unit tests historically did **not** exercise `render_video`'s filtergraph (only helper stubs), so a filtergraph regression can pass CI silently. If you touch span/time mapping, verify end-to-end: render a real multi-segment clip and confirm the output duration equals `timeline.edited_duration` (not `source_duration`), and spot-check a burned subtitle frame against the transcript word at that edited timestamp. Keep the regression tests in `tests/test_renderer.py` that assert spans sum to edited duration and never straddle a silent gap.
