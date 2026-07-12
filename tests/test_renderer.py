from pathlib import Path

import pytest

from ready_video.config import Config
from ready_video.errors import ReadyVideoError
from ready_video.renderer import _loudnorm_filter, _parse_loudnorm_json, _render_spans, _span_source_range, build_render_args
from ready_video.timeline import Timeline, TimelineSegment
from ready_video.zooms import ZoomPlan


def _two_gap_timeline() -> Timeline:
    # Two kept segments separated by a silent gap (source 3.0 -> 8.0 removed).
    return Timeline(
        source_duration=20.0,
        edited_duration=13.0,
        segments=[
            TimelineSegment(i=0, source_start=0.0, source_end=3.0, edited_start=0.0, edited_end=3.0),
            TimelineSegment(i=1, source_start=8.0, source_end=18.0, edited_start=3.0, edited_end=13.0),
        ],
    )


def test_final_render_encodes_once_from_original():
    args = build_render_args(
        Path("input.mp4"),
        Path("edited.mp4"),
        zooms=[],
        mode="final",
        ffmpeg_bin="/usr/bin/ffmpeg",
    )

    assert args[0] == "/usr/bin/ffmpeg"
    assert args.count("-i") == 1
    assert args[args.index("-i") + 1] == "input.mp4"
    assert "-filter_complex" not in args
    assert args[-1] == "edited.mp4"


def test_preview_render_command_adds_short_review_settings():
    args = build_render_args(
        "input.mp4",
        "preview.mp4",
        zooms=[],
        mode="preview",
    )

    assert "-t" in args
    assert args[args.index("-t") + 1] == "20"
    assert "-vf" in args
    assert "scale=-2:720" in args


def test_loudnorm_json_parser_uses_last_json_block():
    payload = _parse_loudnorm_json('noise\n{"input_i":"-18.0","target_offset":"0.1"}\n')

    assert payload["input_i"] == "-18.0"


def test_loudnorm_filter_uses_measured_values():
    measurements = {
        "input_i": "-18.0",
        "input_tp": "-2.0",
        "input_lra": "3.0",
        "input_thresh": "-28.0",
        "target_offset": "0.1",
    }

    text = _loudnorm_filter(Config().resolved(), measurements)

    assert "measured_I=-18.0" in text
    assert "linear=true" in text


def test_loudnorm_filter_rejects_incomplete_measurements():
    with pytest.raises(ReadyVideoError):
        _loudnorm_filter(Config().resolved(), {"input_i": "-18.0"})


def test_render_span_does_not_straddle_silent_gap():
    # A render span starting on a segment boundary must map to the *next*
    # segment's source range, not be pulled back across the removed gap.
    timeline = _two_gap_timeline()

    assert _span_source_range(timeline, 0.0, 3.0) == (0.0, 3.0)
    # Edited 3.0 is the boundary; the second span must start at source 8.0, not 3.0.
    assert _span_source_range(timeline, 3.0, 13.0) == (8.0, 18.0)


def test_render_spans_map_to_edited_duration_not_source_duration():
    # Regression: summing per-span source durations must equal the edited
    # duration (silence removed), never the full source duration.
    timeline = _two_gap_timeline()
    empty_zooms = ZoomPlan.model_validate({"zooms": [], "backend": "none"})

    spans = _render_spans(timeline, empty_zooms)
    total_source = sum(
        end - start
        for edited_start, edited_end, _ in spans
        for start, end in [_span_source_range(timeline, edited_start, edited_end)]
    )

    assert total_source == pytest.approx(timeline.edited_duration)
    assert total_source < timeline.source_duration
