from __future__ import annotations

import importlib
import html
from importlib import metadata
import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

from . import PIPELINE_VERSION
from .config import ResolvedConfig, config_to_yaml, load_config
from .errors import ReadyVideoError
from .ffmpeg import BinaryPair, capability_report, resolve_binaries, run
from .ffmpeg import ffprobe_json
from .identity import config_sha256, file_sha256, job_id, job_sha256, make_manifest, sanitized_stem
from .io import atomic_write_json, atomic_write_text, atomic_write_yaml, ensure_dirs
from .media import SourceInfo, analyze_silence, build_speech_wav, probe_source, write_source
from .refine import TranscriptTrimParams, refine_timeline_with_transcript, retime_transcript
from .renderer import measure_loudness, render_video, target_resolution
from .subtitles import generate_subtitles
from .timeline import Timeline
from .transcript import Transcript, transcribe


def run_file(
    input_path: Path,
    *,
    config_path: Path | None = None,
    review: bool = False,
    cli_overrides: dict | None = None,
) -> Path:
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise ReadyVideoError("INVALID_MEDIA", f"Input file does not exist: {input_path}")
    config = load_config(config_path=config_path, cli_overrides=cli_overrides)
    pair = resolve_binaries()
    ensure_dirs(config.paths.work, config.paths.edited)
    content_hash = file_sha256(input_path)
    cfg_hash = config_sha256(config)
    full_job_hash = job_sha256(content_hash, cfg_hash, PIPELINE_VERSION)
    jid = job_id(full_job_hash)
    output_name = f"{sanitized_stem(input_path)}_{jid}.mp4"
    output_path = config.paths.edited / output_name
    work_dir = config.paths.work / jid
    expected_resolution = target_resolution(config.render.aspect)
    if review:
        if _completed_review(work_dir):
            return work_dir / "review.html"
        review_path = _write_review_from_existing_artifacts(input_path, work_dir, config, pair)
        if review_path is not None:
            return review_path
    elif _completed_output(output_path, full_job_hash, pair, expected_resolution=expected_resolution):
        return output_path
    ensure_dirs(work_dir)
    atomic_write_yaml(work_dir / "resolved-config.yaml", config)
    atomic_write_json(work_dir / "job.json", {"job_id": jid, "job_sha256": full_job_hash, "input": str(input_path)})

    source = probe_source(input_path, pair)
    write_source(work_dir / "source.json", source)
    sound_timeline = analyze_silence(input_path, pair, config, source)
    speech_wav = work_dir / "speech.wav"
    build_speech_wav(input_path, speech_wav, pair, source, sound_timeline)
    transcript = transcribe(speech_wav, work_dir / "transcript.json", pair, config)
    # Second pass: compress inter-word pauses that survived sound-based silence
    # removal (quiet room tone/breath above the dB threshold). Cuts only across
    # aligned word boundaries, never clipping speech. Re-time the transcript onto
    # the tightened axis so subtitles land correctly.
    timeline = refine_timeline_with_transcript(sound_timeline, transcript, _transcript_trim_params(config))
    if timeline is not sound_timeline:
        transcript = retime_transcript(transcript, sound_timeline, timeline)
        atomic_write_json(work_dir / "transcript.json", transcript)
    atomic_write_json(work_dir / "timeline.json", timeline)
    ass_path = work_dir / "subs.ass"
    srt_path = work_dir / "optional.srt" if config.subtitles.export_srt else None
    generate_subtitles(transcript, ass_path, srt_path, config, target_resolution(config.render.aspect))

    if review:
        preview_path = work_dir / "preview.mp4"
        render_video(input_path, preview_path, pair, source, timeline, config, subtitles_path=ass_path if ass_path.exists() else None, preview=True)
        _write_review_artifacts(work_dir, input_path, config, timeline, preview_path, ass_path, srt_path)
        _write_review(work_dir / "review.html", input_path, config, timeline, transcript, preview_path)
        return work_dir / "review.html"

    loudness = None
    if config.render.loudnorm:
        loudness = measure_loudness(input_path, pair, source, timeline, config)
        atomic_write_json(work_dir / "loudness.json", loudness)
    render_video(input_path, output_path, pair, source, timeline, config, subtitles_path=ass_path if ass_path.exists() else None, loudness=loudness)
    _validate_output(output_path, pair, expected_resolution=expected_resolution)
    manifest = make_manifest(
        input_path=input_path,
        output_name=output_name,
        content_hash=content_hash,
        config_hash=cfg_hash,
        job_hash=full_job_hash,
    )
    atomic_write_json(output_path.with_suffix(".json"), manifest)
    return output_path


def _transcript_trim_params(config: ResolvedConfig) -> TranscriptTrimParams:
    return TranscriptTrimParams(
        enabled=config.silence.enabled and config.silence.transcript_trim,
        max_gap_s=config.silence.transcript_trim_max_gap_s,
        word_margin_s=config.silence.transcript_trim_word_margin_s,
    )


def _completed_output(output_path: Path, job_hash: str, pair, *, expected_resolution: tuple[int, int] | None = None) -> bool:
    manifest_path = output_path.with_suffix(".json")
    if not output_path.exists() or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return False
    if manifest.get("job_sha256") != job_hash:
        return False
    if manifest.get("output_name") != output_path.name:
        return False
    try:
        _validate_output(output_path, pair, expected_resolution=expected_resolution)
    except ReadyVideoError:
        return False
    return True


def _completed_review(work_dir: Path) -> bool:
    return all((work_dir / name).exists() for name in ["review.html", "review-artifacts.json", "preview.mp4"])


def _write_review_from_existing_artifacts(input_path: Path, work_dir: Path, config: ResolvedConfig, pair: BinaryPair) -> Path | None:
    source_path = work_dir / "source.json"
    timeline_path = work_dir / "timeline.json"
    transcript_path = work_dir / "transcript.json"
    ass_path = work_dir / "subs.ass"
    required = [source_path, timeline_path, transcript_path, ass_path, work_dir / "job.json", work_dir / "resolved-config.yaml"]
    if not all(path.exists() for path in required):
        return None
    source = SourceInfo.model_validate(json.loads(source_path.read_text()))
    timeline = Timeline.model_validate(json.loads(timeline_path.read_text()))
    transcript = Transcript.model_validate(json.loads(transcript_path.read_text()))
    preview_path = work_dir / "preview.mp4"
    render_video(input_path, preview_path, pair, source, timeline, config, subtitles_path=ass_path, preview=True)
    _write_review_artifacts(work_dir, input_path, config, timeline, preview_path, ass_path, work_dir / "optional.srt")
    _write_review(work_dir / "review.html", input_path, config, timeline, transcript, preview_path)
    return work_dir / "review.html"


def _validate_output(output_path: Path, pair, *, expected_resolution: tuple[int, int] | None = None) -> None:
    if not output_path.exists():
        raise ReadyVideoError("OUTPUT_VALIDATION_FAILED", f"Output was not written: {output_path}")
    try:
        data = ffprobe_json(pair, ["-show_streams", "-show_format", output_path])
    except ReadyVideoError as exc:
        raise ReadyVideoError("OUTPUT_VALIDATION_FAILED", "Rendered output did not pass ffprobe.", details=exc.details) from exc
    streams = data.get("streams") or []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video is None:
        raise ReadyVideoError("OUTPUT_VALIDATION_FAILED", "Rendered output has no video stream.")
    if expected_resolution and (int(video.get("width") or 0), int(video.get("height") or 0)) != expected_resolution:
        raise ReadyVideoError(
            "OUTPUT_VALIDATION_FAILED",
            "Rendered output dimensions do not match the requested aspect.",
            details=f"expected={expected_resolution} actual={(video.get('width'), video.get('height'))}",
        )
    if video.get("pix_fmt") and video.get("pix_fmt") != "yuv420p":
        raise ReadyVideoError("OUTPUT_VALIDATION_FAILED", "Rendered output is not yuv420p.", details=str(video.get("pix_fmt")))
    if video.get("avg_frame_rate") not in {None, "0/0", "30/1"}:
        raise ReadyVideoError("OUTPUT_VALIDATION_FAILED", "Rendered output is not constant 30 fps.", details=str(video.get("avg_frame_rate")))
    expected_color = {"color_space": "bt709", "color_transfer": "bt709", "color_primaries": "bt709"}
    missing_or_wrong = {key: video.get(key) for key, expected in expected_color.items() if video.get(key) != expected}
    if missing_or_wrong:
        raise ReadyVideoError("OUTPUT_VALIDATION_FAILED", "Rendered output is missing BT.709 SDR metadata.", details=str(missing_or_wrong))
    completed = run([pair.ffmpeg, "-v", "error", "-nostdin", "-i", output_path, "-f", "null", "-"], timeout=120)
    if completed.returncode != 0:
        raise ReadyVideoError("OUTPUT_VALIDATION_FAILED", "Rendered output failed full decode validation.", details=completed.stderr)


def approve(job_id_value: str, *, config_path: Path | None = None) -> Path:
    work_dir = _review_work_dir(job_id_value, config_path=config_path)
    job_path = work_dir / "job.json"
    resolved_path = work_dir / "resolved-config.yaml"
    if not job_path.exists() or not resolved_path.exists():
        raise ReadyVideoError("INVALID_MEDIA", f"No reviewed job found for {job_id_value}.")
    config = _load_captured_config(resolved_path)
    job = json.loads(job_path.read_text())
    input_path = Path(job["input"])
    if not input_path.is_file():
        raise ReadyVideoError("INVALID_MEDIA", f"Reviewed input file no longer exists: {input_path}")
    source_path = work_dir / "source.json"
    timeline_path = work_dir / "timeline.json"
    transcript_path = work_dir / "transcript.json"
    missing = [path.name for path in [source_path, timeline_path, transcript_path] if not path.exists()]
    if missing:
        raise ReadyVideoError("INVALID_MEDIA", f"Reviewed job {job_id_value} is missing artifacts: {', '.join(missing)}.")

    pair = resolve_binaries()
    ensure_dirs(config.paths.edited)
    source = SourceInfo.model_validate(json.loads(source_path.read_text()))
    timeline = Timeline.model_validate(json.loads(timeline_path.read_text()))
    ass_path = work_dir / "subs.ass"
    output_name = f"{sanitized_stem(input_path)}_{job_id_value}.mp4"
    output_path = config.paths.edited / output_name
    loudness = None
    if config.render.loudnorm:
        loudness_path = work_dir / "loudness.json"
        if loudness_path.exists():
            loudness = json.loads(loudness_path.read_text())
        else:
            loudness = measure_loudness(input_path, pair, source, timeline, config)
            atomic_write_json(loudness_path, loudness)
    render_video(input_path, output_path, pair, source, timeline, config, subtitles_path=ass_path if ass_path.exists() else None, loudness=loudness)
    _validate_output(output_path, pair, expected_resolution=target_resolution(config.render.aspect))

    content_hash = file_sha256(input_path)
    cfg_hash = config_sha256(config)
    full_job_hash = str(job.get("job_sha256") or job_sha256(content_hash, cfg_hash, PIPELINE_VERSION))
    manifest = make_manifest(
        input_path=input_path,
        output_name=output_name,
        content_hash=content_hash,
        config_hash=cfg_hash,
        job_hash=full_job_hash,
    )
    atomic_write_json(output_path.with_suffix(".json"), manifest)
    return output_path


def _review_work_dir(job_id_value: str, *, config_path: Path | None = None) -> Path:
    candidates: list[Path] = []
    try:
        candidates.append(load_config(config_path=config_path).paths.work / job_id_value)
    except ReadyVideoError:
        pass
    candidates.append(Path("work") / job_id_value)
    for candidate in candidates:
        if (candidate / "job.json").exists() and (candidate / "resolved-config.yaml").exists():
            return candidate
    return candidates[0] if candidates else Path("work") / job_id_value


def _load_captured_config(path: Path) -> ResolvedConfig:
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ReadyVideoError("INVALID_CONFIG", f"Could not parse captured resolved config: {path}", details=str(exc)) from exc
    if not isinstance(raw, dict):
        raise ReadyVideoError("INVALID_CONFIG", f"Captured resolved config must be a mapping: {path}")
    try:
        return ResolvedConfig.model_validate(raw)
    except Exception as exc:
        raise ReadyVideoError("INVALID_CONFIG", f"Captured resolved config is invalid: {path}", details=str(exc)) from exc


def doctor(*, install_missing: bool = False) -> int:
    checks: list[tuple[str, str, str]] = []
    config: ResolvedConfig | None = None

    try:
        config = load_config()
        checks.append(("config", "pass", "loaded"))
    except ReadyVideoError as exc:
        checks.append(("config", "fail", str(exc)))

    try:
        pair = resolve_binaries(install_missing=install_missing)
        checks.append(("ffmpeg.resolution", "pass", _format_binary_resolution(pair)))
        checks.append(("ffmpeg.binary", "pass", str(pair.ffmpeg)))
        checks.append(("ffprobe.binary", "pass", str(pair.ffprobe)))
        try:
            report = capability_report(pair)
            checks.append(("ffmpeg.version", "pass", report.ffmpeg_version))
            checks.append(("ffprobe.version", "pass", report.ffprobe_version))
            checks.append(("ffmpeg.capabilities", _capability_status(report), _format_capability_report(report)))
            checks.append(("ffmpeg.optional_encoders", "pass" if report.available_optional_encoders else "warning", _format_optional_encoders(report)))
        except ReadyVideoError as exc:
            checks.append(("ffmpeg.capabilities", "fail", str(exc)))
    except ReadyVideoError as exc:
        checks.append(("ffmpeg.resolution", "fail", _binary_candidate_details()))
        candidate = _unvalidated_binary_pair()
        if candidate is not None:
            try:
                report = capability_report(candidate)
                checks.append(("ffmpeg.version", "pass", report.ffmpeg_version))
                checks.append(("ffprobe.version", "pass", report.ffprobe_version))
                checks.append(("ffmpeg.capabilities", _capability_status(report), _format_capability_report(report)))
                checks.append(("ffmpeg.optional_encoders", "pass" if report.available_optional_encoders else "warning", _format_optional_encoders(report)))
            except ReadyVideoError as capability_exc:
                checks.append(("ffmpeg.capabilities", "fail", str(capability_exc)))
        checks.append(("ffmpeg", "fail", str(exc)))

    try:
        importlib.import_module("faster_whisper")

        checks.append(("faster-whisper", "pass", f"import ok ({_package_version('faster-whisper')})"))
    except Exception as exc:
        checks.append(("faster-whisper", "fail", f"import failed; install the transcription dependency ({exc.__class__.__name__}: {exc})"))

    if config is not None:
        for name, path in {"edited": config.paths.edited, "work": config.paths.work}.items():
            status, detail = _path_status(Path(path))
            checks.append((f"path.{name}", status, detail))

    for name, status, detail in checks:
        print(f"{status.upper():7} {name}: {detail}")
    return 1 if any(status == "fail" for _, status, _ in checks) else 0


def _format_binary_resolution(pair) -> str:
    details = [f"source={pair.source}"]
    if pair.provider:
        details.append(f"provider={pair.provider}")
    if pair.cache_dir:
        details.append(f"cache_dir={pair.cache_dir}")
    return " ".join(details)


def _binary_candidate_details() -> str:
    env_ffmpeg = os.environ.get("READY_VIDEO_FFMPEG")
    env_ffprobe = os.environ.get("READY_VIDEO_FFPROBE")
    if env_ffmpeg or env_ffprobe:
        return f"READY_VIDEO_FFMPEG={env_ffmpeg or '<unset>'} READY_VIDEO_FFPROBE={env_ffprobe or '<unset>'}"
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    return f"PATH ffmpeg={ffmpeg or '<missing>'} ffprobe={ffprobe or '<missing>'}"


def _unvalidated_binary_pair() -> BinaryPair | None:
    env_ffmpeg = os.environ.get("READY_VIDEO_FFMPEG")
    env_ffprobe = os.environ.get("READY_VIDEO_FFPROBE")
    if env_ffmpeg and env_ffprobe and env_ffmpeg.strip() and env_ffprobe.strip():
        return BinaryPair(Path(env_ffmpeg).expanduser(), Path(env_ffprobe).expanduser(), "environment")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        return BinaryPair(Path(ffmpeg), Path(ffprobe), "system")
    return None


def _format_capability_report(report) -> str:
    missing_filters = sorted(report.missing_filters)
    missing_encoders = sorted(report.missing_encoders)
    ffprobe_release = _doctor_release_token(report.ffprobe_version)
    filter_status = "ok" if not missing_filters else "missing " + ",".join(missing_filters)
    encoder_status = "ok" if not missing_encoders else "missing " + ",".join(missing_encoders)
    details = [f"release={report.release}", f"ffprobe_release={ffprobe_release}", f"required_filters={filter_status}"]
    details.append(f"required_encoders={encoder_status}")
    if report.release != ffprobe_release:
        details.append("binary_pair=version_mismatch")
    return " ".join(details)


def _capability_status(report) -> str:
    if not report.supported:
        return "fail"
    if report.release != _doctor_release_token(report.ffprobe_version):
        return "fail"
    return "pass"


def _doctor_release_token(version_line: str) -> str:
    parts = version_line.split()
    return parts[2] if len(parts) > 2 and parts[0] in {"ffmpeg", "ffprobe"} else version_line


def _format_optional_encoders(report) -> str:
    available = sorted(report.available_optional_encoders)
    missing = sorted(report.optional_encoders - report.available_optional_encoders)
    details = [f"available={','.join(available) if available else 'none'}"]
    if missing:
        details.append(f"missing={','.join(missing)}")
    return " ".join(details)


def _package_version(name: str) -> str:
    try:
        return f"version {metadata.version(name)}"
    except metadata.PackageNotFoundError:
        return "version unknown"


def _path_status(path: Path) -> tuple[str, str]:
    absolute = path.expanduser().resolve(strict=False)
    exists = absolute.exists()
    if exists and not absolute.is_dir():
        return "fail", f"{absolute} exists but is not a directory"
    probe = absolute if exists else _nearest_existing_parent(absolute.parent)
    writable = os.access(probe, os.W_OK)
    state = "exists" if exists else f"missing; parent={probe}"
    detail = f"{absolute} ({state}; writable={'yes' if writable else 'no'}; free={_disk_free(probe)})"
    return ("pass" if writable else "warning", detail)


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate if candidate.exists() else Path.cwd()


def _disk_free(path: Path) -> str:
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        return f"unavailable ({exc})"
    return _format_bytes(usage.free)


def _format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f}{unit}" if unit != "B" else f"{int(amount)}B"
        amount /= 1024
    return f"{value}B"


def _write_review_artifacts(
    work_dir: Path,
    input_path: Path,
    config: ResolvedConfig,
    timeline: Any,
    preview_path: Path,
    ass_path: Path,
    srt_path: Path | None,
) -> None:
    artifacts: dict[str, Any] = {
        "job_id": work_dir.name,
        "input": str(input_path),
        "review_html": str(work_dir / "review.html"),
        "preview": str(preview_path),
        "resolved_config": str(work_dir / "resolved-config.yaml"),
        "job": str(work_dir / "job.json"),
        "source": str(work_dir / "source.json"),
        "timeline": str(work_dir / "timeline.json"),
        "transcript": str(work_dir / "transcript.json"),
        "subtitles_ass": str(ass_path),
        "subtitles_srt": str(srt_path) if srt_path else None,
        "config_sha256": config_sha256(config),
        "edited_duration": timeline.edited_duration,
    }
    atomic_write_json(work_dir / "review-artifacts.json", artifacts)


def _write_review(path: Path, input_path: Path, config: ResolvedConfig, timeline, transcript: Transcript, preview_path: Path) -> None:
    removed = []
    cursor = 0.0
    for segment in timeline.segments:
        if segment.source_start > cursor:
            removed.append((cursor, segment.source_start))
        cursor = segment.source_end
    if cursor < timeline.source_duration:
        removed.append((cursor, timeline.source_duration))
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>Ready Video Review</title>
<h1>Ready Video Review</h1>
<video controls src="{preview_path.name}" style="max-width:360px;width:100%"></video>
<h2>Source</h2>
<p>{input_path.name}: {timeline.source_duration:.2f}s -> {timeline.edited_duration:.2f}s</p>
<h2>Kept ranges</h2>
<pre>{json.dumps([s.model_dump() for s in timeline.segments], indent=2)}</pre>
<h2>Removed ranges</h2>
<pre>{json.dumps(removed, indent=2)}</pre>
<h2>Transcript</h2>
<pre>{_format_review_transcript(transcript)}</pre>
<h2>Resolved config</h2>
<pre>{config_to_yaml(config)}</pre>
"""
    atomic_write_text(path, html)


def _format_review_transcript(transcript: Transcript) -> str:
    if not transcript.words:
        return "No transcript words."
    lines = [
        f"{word.i}: {word.start:.3f}-{word.end:.3f} [{word.timing}] {word.w}"
        for word in transcript.words
    ]
    return html.escape("\n".join(lines))
