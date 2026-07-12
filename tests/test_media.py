from pathlib import Path
from types import SimpleNamespace

import pytest

from ready_video.errors import ReadyVideoError
from ready_video.ffmpeg import BinaryPair
from ready_video.media import decode_probe, parse_silencedetect, probe_source


def test_parse_silencedetect_pairs_start_and_end():
    output = """
    [silencedetect @ 0x1] silence_start: 1.5
    [silencedetect @ 0x1] silence_end: 2.75 | silence_duration: 1.25
    """

    assert parse_silencedetect(output) == [{"start": 1.5, "end": 2.75, "duration": 1.25}]


def test_probe_source_records_rotation_and_runs_decode_probe(monkeypatch, tmp_path):
    media = tmp_path / "clip.mov"
    media.write_bytes(b"fake")
    pair = BinaryPair(Path("ffmpeg"), Path("ffprobe"), "test")
    probe_data = {
        "format": {"duration": "12.0", "start_time": "0.0"},
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "duration": "12.0",
                "width": 1080,
                "height": 1920,
                "pix_fmt": "yuv420p",
                "r_frame_rate": "30/1",
                "avg_frame_rate": "30000/1001",
                "sample_aspect_ratio": "1:1",
                "disposition": {"default": 1},
                "tags": {"rotate": "90"},
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "disposition": {"default": 1},
            },
        ],
    }
    run_calls = []

    monkeypatch.setattr("ready_video.media.ffprobe_json", lambda pair, args: probe_data)
    monkeypatch.setattr("ready_video.media.available_filters", lambda ffmpeg: {"zscale", "tonemap"})
    monkeypatch.setattr(
        "ready_video.media.run",
        lambda args, timeout=None: run_calls.append(args) or SimpleNamespace(returncode=0, stderr=""),
    )

    source = probe_source(media, pair)

    assert (source.width, source.height) == (1920, 1080)
    assert source.rotation == 90
    assert source.variable_frame_rate is True
    assert len(run_calls) == 2


def test_probe_source_rejects_hdr_without_tonemap(monkeypatch, tmp_path):
    media = tmp_path / "hdr.mov"
    media.write_bytes(b"fake")
    pair = BinaryPair(Path("ffmpeg"), Path("ffprobe"), "test")
    monkeypatch.setattr(
        "ready_video.media.ffprobe_json",
        lambda pair, args: {
            "format": {"duration": "3.0"},
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "duration": "3.0",
                    "width": 1920,
                    "height": 1080,
                    "pix_fmt": "yuv420p10le",
                    "color_transfer": "smpte2084",
                },
                {"index": 1, "codec_type": "audio"},
            ],
        },
    )
    monkeypatch.setattr("ready_video.media.available_filters", lambda ffmpeg: {"scale"})

    with pytest.raises(ReadyVideoError) as error:
        probe_source(media, pair)

    assert error.value.code == "UNSUPPORTED_HDR"


def test_decode_probe_failure_is_invalid_media(monkeypatch, tmp_path):
    media = tmp_path / "broken.mp4"
    media.write_bytes(b"fake")
    pair = BinaryPair(Path("ffmpeg"), Path("ffprobe"), "test")
    source = SimpleNamespace(start_time=0.0, duration=1.0, video_stream_index=0)
    monkeypatch.setattr(
        "ready_video.media.run",
        lambda args, timeout=None: SimpleNamespace(returncode=1, stderr="decode failed"),
    )

    with pytest.raises(ReadyVideoError) as error:
        decode_probe(media, pair, source)

    assert error.value.code == "INVALID_MEDIA"
