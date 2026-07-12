from __future__ import annotations

import json
from pathlib import Path

import yaml

import ready_video.config as config_module
import ready_video.pipeline as pipeline
from ready_video.config import Config
from ready_video.media import SourceInfo
from ready_video.timeline import Timeline, TimelineSegment
from ready_video.transcript import Transcript, Word


def _isolate_auto_config(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config_module, "user_config_path", lambda: tmp_path / "missing-user-config.yaml")
    monkeypatch.setattr(config_module, "project_config_path", lambda cwd=None: tmp_path / "missing-project-config.yaml")


def _write_config(path: Path, raw: dict) -> None:
    path.write_text(yaml.safe_dump(_jsonable(raw), sort_keys=False))


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _timeline() -> Timeline:
    return Timeline(
        source_duration=10.0,
        edited_duration=8.0,
        segments=[
            TimelineSegment(
                i=0,
                source_start=0.0,
                source_end=8.0,
                edited_start=0.0,
                edited_end=8.0,
            )
        ],
    )


def test_review_writes_artifact_manifest(monkeypatch, tmp_path: Path) -> None:
    _isolate_auto_config(monkeypatch, tmp_path)
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        {
            "paths": {
                "work": tmp_path / "work",
                "edited": tmp_path / "edited",
            },
        },
    )
    source_media = tmp_path / "clip.mp4"
    source_media.write_bytes(b"not real media")

    monkeypatch.setattr(pipeline, "resolve_binaries", lambda: object())
    monkeypatch.setattr(
        pipeline,
        "probe_source",
        lambda input_path, pair: SourceInfo(duration=10.0, width=1920, height=1080, video_stream_index=0, audio_stream_index=1),
    )
    monkeypatch.setattr(pipeline, "analyze_silence", lambda input_path, pair, config, source: _timeline())
    monkeypatch.setattr(pipeline, "build_speech_wav", lambda input_path, speech_wav, pair, source, timeline: speech_wav.write_bytes(b"wav"))

    def fake_transcribe(speech_wav, output_path, pair, config):
        transcript = Transcript(duration=8.0, words=[Word(i=0, w="<hello>", start=0.0, end=0.5, timing="aligned")])
        pipeline.atomic_write_json(output_path, transcript)
        return transcript

    def fake_generate_subtitles(transcript, ass_path, srt_path, config, resolution):
        ass_path.write_text("[Script Info]\n")
        if srt_path:
            srt_path.write_text("")

    def fake_render(input_path, output_path, pair, source, timeline, config, *, subtitles_path=None, preview=False):
        output_path.write_bytes(b"preview" if preview else b"final")

    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    monkeypatch.setattr(pipeline, "generate_subtitles", fake_generate_subtitles)
    monkeypatch.setattr(pipeline, "render_video", fake_render)

    review_path = pipeline.run_file(source_media, config_path=config_path, review=True)

    artifacts_path = review_path.parent / "review-artifacts.json"
    artifacts = json.loads(artifacts_path.read_text())
    assert review_path.name == "review.html"
    assert artifacts["job_id"] == review_path.parent.name
    review_html = review_path.read_text()
    assert "<h2>Transcript</h2>" in review_html
    assert "0: 0.000-0.500 [aligned] &lt;hello&gt;" in review_html
    for key in ["job", "source", "timeline", "transcript", "preview", "resolved_config", "subtitles_ass"]:
        assert Path(artifacts[key]).exists()


def test_approve_uses_captured_resolved_config(monkeypatch, tmp_path: Path) -> None:
    _isolate_auto_config(monkeypatch, tmp_path)
    job_id = "abcdef123456"
    input_path = tmp_path / "clip.mp4"
    input_path.write_bytes(b"reviewed media")
    work_dir = tmp_path / "work" / job_id
    work_dir.mkdir(parents=True)
    captured_edited = tmp_path / "captured-edited"
    ambient_edited = tmp_path / "ambient-edited"
    ambient_config_path = tmp_path / "ambient.yaml"
    _write_config(
        ambient_config_path,
        {
            "paths": {
                "work": tmp_path / "work",
                "edited": ambient_edited,
            },
        },
    )
    captured_config = Config.model_validate(
        {
            "paths": {
                "work": tmp_path / "work",
                "edited": captured_edited,
            },
        }
    ).resolved()
    pipeline.atomic_write_yaml(work_dir / "resolved-config.yaml", captured_config)
    pipeline.atomic_write_json(work_dir / "job.json", {"job_id": job_id, "job_sha256": job_id + ("0" * 52), "input": str(input_path)})
    pipeline.atomic_write_json(
        work_dir / "source.json",
        SourceInfo(duration=10.0, width=1920, height=1080, video_stream_index=0, audio_stream_index=1),
    )
    pipeline.atomic_write_json(work_dir / "timeline.json", _timeline())
    pipeline.atomic_write_json(work_dir / "transcript.json", Transcript(duration=8.0))
    (work_dir / "subs.ass").write_text("[Script Info]\n")

    seen_configs = []

    def fake_render(input_path, output_path, pair, source, timeline, config, *, subtitles_path=None, loudness=None, preview=False):
        seen_configs.append(config)
        output_path.write_bytes(b"approved")

    monkeypatch.setattr(pipeline, "resolve_binaries", lambda: object())
    monkeypatch.setattr(pipeline, "run_file", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("approve should use saved artifacts")))
    monkeypatch.setattr(pipeline, "render_video", fake_render)
    monkeypatch.setattr(
        pipeline,
        "measure_loudness",
        lambda *args, **kwargs: {
            "input_i": "-18.0",
            "input_tp": "-2.0",
            "input_lra": "3.0",
            "input_thresh": "-28.0",
            "target_offset": "0.1",
        },
    )
    monkeypatch.setattr(pipeline, "_validate_output", lambda output_path, pair, *, expected_resolution=None: None)

    output_path = pipeline.approve(job_id, config_path=ambient_config_path)

    assert output_path == captured_edited / f"clip_{job_id}.mp4"
    assert output_path.exists()
    assert seen_configs[0].paths.edited == captured_edited
    assert not ambient_edited.exists()
