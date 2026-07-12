from __future__ import annotations

import json
import re
from pathlib import Path

from .config import ResolvedConfig
from .errors import ReadyVideoError
from .ffmpeg import BinaryPair, available_filters, run
from .media import SourceInfo
from .timeline import Timeline


TARGETS = {"9:16": (1080, 1920), "1:1": (1080, 1080), "16:9": (1920, 1080)}


def build_render_args(
    input_path,
    output_path,
    *,
    mode: str = "final",
    ffmpeg_bin: str = "ffmpeg",
) -> list[str]:
    args = [ffmpeg_bin, "-y", "-i", str(input_path)]
    if mode == "preview":
        args.extend(["-t", "20", "-vf", "scale=-2:720", "-preset", "ultrafast", "-crf", "30"])
    else:
        args.extend(["-c:v", "libx264", "-crf", "18"])
    args.append(str(output_path))
    return args


def target_resolution(aspect: str, *, preview: bool = False) -> tuple[int, int]:
    width, height = TARGETS[aspect]
    if not preview:
        return width, height
    scale = 720 / height if height >= width else 720 / width
    even_width = int(width * scale) // 2 * 2
    even_height = int(height * scale) // 2 * 2
    return max(2, even_width), max(2, even_height)


def render_video(
    input_path: Path,
    output_path: Path,
    pair: BinaryPair,
    source: SourceInfo,
    timeline: Timeline,
    config: ResolvedConfig,
    *,
    subtitles_path: Path | None = None,
    loudness: dict | None = None,
    preview: bool = False,
) -> None:
    if source.color_transfer.lower() in {"smpte2084", "arib-std-b67"} and not {"zscale", "tonemap"} <= available_filters(pair.ffmpeg):
        raise ReadyVideoError("UNSUPPORTED_HDR", "HDR input requires FFmpeg zscale and tonemap filters.")
    width, height = target_resolution(config.render.aspect, preview=preview)
    spans = _render_spans(timeline)
    filters: list[str] = []
    vlabels: list[str] = []
    alabels: list[str] = []
    for i, (edited_start, edited_end) in enumerate(spans):
        source_start, source_end = _span_source_range(timeline, edited_start, edited_end)
        vlabel = f"v{i}"
        alabel = f"a{i}"
        crop_x = f"(iw-{width})/2+({config.render.crop_x_offset})*(iw-{width})/2"
        crop_y = f"(ih-{height})/2+({config.render.crop_y_offset})*(ih-{height})/2"
        filters.append(
            f"[0:{source.video_stream_index}]trim=start={source_start}:end={source_end},setpts=PTS-STARTPTS,"
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}:{crop_x}:{crop_y},fps=30,setsar=1,format=yuv420p[{vlabel}]"
        )
        filters.append(f"[0:{source.audio_stream_index}]atrim=start={source_start}:end={source_end},asetpts=PTS-STARTPTS[{alabel}]")
        vlabels.append(f"[{vlabel}]")
        alabels.append(f"[{alabel}]")
    concat_inputs = "".join(label for pair in zip(vlabels, alabels, strict=False) for label in pair)
    filters.append(f"{concat_inputs}concat=n={len(spans)}:v=1:a=1[vcat][acat]")
    final_video = "[vcat]"
    final_audio = "[acat]"
    if loudness and not preview:
        filters.append(f"[acat]{_loudnorm_filter(config, loudness)}[aout]")
        final_audio = "[aout]"
    if subtitles_path and subtitles_path.exists():
        escaped = str(subtitles_path).replace("\\", "\\\\").replace(":", "\\:")
        filters.append(f"[vcat]ass='{escaped}'[vout]")
        final_video = "[vout]"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
    args: list[str | Path] = [
        pair.ffmpeg,
        "-y",
        "-hide_banner",
        "-nostdin",
        "-i",
        input_path,
        "-filter_complex",
        ";".join(filters),
        "-map",
        final_video,
        "-map",
        final_audio,
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast" if preview else "medium",
        "-crf",
        "30" if preview else str(config.render.crf),
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "30",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        "-c:a",
        "aac",
        "-b:a",
        "128k" if preview else "192k",
        "-movflags",
        "+faststart",
        tmp,
    ]
    completed = run(args, timeout=max(120, int(timeline.source_duration * 8)))
    if completed.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise ReadyVideoError("RENDER_FAILED", "FFmpeg render failed.", details=completed.stderr)
    tmp.replace(output_path)


def measure_loudness(input_path: Path, pair: BinaryPair, source: SourceInfo, timeline: Timeline, config: ResolvedConfig) -> dict:
    filters: list[str] = []
    labels: list[str] = []
    for segment in timeline.segments:
        label = f"a{segment.i}"
        filters.append(
            f"[0:{source.audio_stream_index}]atrim=start={segment.source_start}:end={segment.source_end},asetpts=PTS-STARTPTS[{label}]"
        )
        labels.append(f"[{label}]")
    filters.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[acat]")
    filters.append(f"[acat]loudnorm=I={config.render.loudnorm_target_lufs}:TP=-1.5:LRA=11:print_format=json[aout]")
    completed = run(
        [
            pair.ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-i",
            input_path,
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[aout]",
            "-f",
            "null",
            "-",
        ],
        timeout=max(120, int(timeline.source_duration * 6)),
    )
    if completed.returncode != 0:
        raise ReadyVideoError("RENDER_FAILED", "FFmpeg loudness measurement failed.", details=completed.stderr)
    return _parse_loudnorm_json(completed.stderr or "")


def _parse_loudnorm_json(stderr: str) -> dict:
    matches = re.findall(r"\{(?:.|\n)*?\}", stderr)
    if not matches:
        raise ReadyVideoError("RENDER_FAILED", "FFmpeg loudnorm did not return measurement JSON.", details=stderr)
    try:
        return json.loads(matches[-1])
    except json.JSONDecodeError as exc:
        raise ReadyVideoError("RENDER_FAILED", "FFmpeg loudnorm measurement JSON could not be parsed.", details=str(exc)) from exc


def _loudnorm_filter(config: ResolvedConfig, loudness: dict) -> str:
    required = ["input_i", "input_tp", "input_lra", "input_thresh", "target_offset"]
    missing = [key for key in required if key not in loudness]
    if missing:
        raise ReadyVideoError("RENDER_FAILED", "Loudness measurements are incomplete.", details=f"missing={missing}")
    return (
        f"loudnorm=I={config.render.loudnorm_target_lufs}:TP=-1.5:LRA=11:"
        f"measured_I={loudness['input_i']}:"
        f"measured_TP={loudness['input_tp']}:"
        f"measured_LRA={loudness['input_lra']}:"
        f"measured_thresh={loudness['input_thresh']}:"
        f"offset={loudness['target_offset']}:linear=true:print_format=summary"
    )


def _span_source_range(timeline: Timeline, edited_start: float, edited_end: float) -> tuple[float, float]:
    """Map one render span to its single source range.

    Render spans are split at every kept-segment boundary, so a span lies wholly
    within one timeline segment and maps to exactly one contiguous source range.
    Mapping each endpoint independently with ``edited_to_source`` is wrong: at a
    segment boundary the start would resolve to the *previous* segment's end,
    re-including the removed silent gap. Use the segment-aware range mapping.
    """
    ranges = timeline.edited_range_to_source_ranges(edited_start, edited_end)
    if len(ranges) != 1:
        raise ReadyVideoError(
            "TIMELINE_MAPPING_FAILED",
            "Render span does not map to a single source range.",
            details=f"edited=({edited_start:.6f},{edited_end:.6f}) ranges={ranges}",
        )
    return ranges[0]


def _render_spans(timeline: Timeline) -> list[tuple[float, float]]:
    boundaries = {0.0, timeline.edited_duration}
    for segment in timeline.segments:
        boundaries.add(segment.edited_start)
        boundaries.add(segment.edited_end)
    ordered = sorted(boundaries)
    return [(start, end) for start, end in zip(ordered, ordered[1:], strict=False) if end > start]
