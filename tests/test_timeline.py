import pytest

from ready_video.timeline import (
    TimelineSegment,
    clamp_segments,
    format_ass_timestamp,
    format_srt_timestamp,
    frames_to_seconds,
    seconds_to_frames,
    seconds_to_milliseconds,
    shift_segments,
)


def test_frame_and_time_conversions_round_half_up():
    assert seconds_to_frames(1.5, 24) == 36
    assert seconds_to_frames(1.25, 2) == 3
    assert frames_to_seconds(48, 24) == 2
    assert seconds_to_milliseconds(1.2345) == 1235


def test_subtitle_timestamp_formatting():
    assert format_srt_timestamp(3661.234) == "01:01:01,234"
    assert format_ass_timestamp(3661.235) == "1:01:01.24"


def test_segment_validation_and_helpers():
    with pytest.raises(ValueError):
        TimelineSegment(-0.1, 1)
    with pytest.raises(ValueError):
        TimelineSegment(2, 1)

    segments = [TimelineSegment(0.2, 2.5, "hello")]
    assert shift_segments(segments, -0.1) == [TimelineSegment(0.1, 2.4, "hello")]
    assert clamp_segments(segments, 1.0) == [TimelineSegment(0.2, 1.0, "hello")]

