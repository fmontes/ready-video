from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import ready_video.pipeline as pipeline
from ready_video.config import Config
from ready_video.identity import config_sha256, file_sha256, job_id, job_sha256, make_manifest, sanitized_stem
from ready_video.pipeline import run_file
from ready_video.media import SourceInfo
from ready_video.timeline import Timeline, TimelineSegment
from ready_video.transcript import Transcript


class DummyPair:
    ffmpeg = Path("ffmpeg")
    ffprobe = Path("ffprobe")
    source = "test"


VALID_VIDEO_PROBE = {
    "streams": [
        {
            "codec_type": "video",
            "width": 1080,
            "height": 1920,
            "pix_fmt": "yuv420p",
            "avg_frame_rate": "30/1",
            "color_space": "bt709",
            "color_transfer": "bt709",
            "color_primaries": "bt709",
        }
    ],
    "format": {"duration": "1.0"},
}


def _config(tmp_path: Path):
    config = Config().resolved()
    config.paths.edited = tmp_path / "edited"
    config.paths.work = tmp_path / "work"
    return config


def test_run_file_skips_existing_manifest_with_valid_media(tmp_path, monkeypatch):
    source = tmp_path / "Clip.mov"
    source.write_bytes(b"not really media")
    config = _config(tmp_path)
    config.paths.edited.mkdir()
    full_job_hash = job_sha256(file_sha256(source), config_sha256(config))
    jid = job_id(full_job_hash)
    output_name = f"{sanitized_stem(source)}_{jid}.mp4"
    output = config.paths.edited / output_name
    output.write_bytes(b"rendered")
    manifest = make_manifest(
        input_path=source,
        output_name=output_name,
        content_hash=file_sha256(source),
        config_hash=config_sha256(config),
        job_hash=full_job_hash,
    )
    output.with_suffix(".json").write_text(json.dumps(manifest.model_dump()))

    monkeypatch.setattr("ready_video.pipeline.load_config", lambda **kwargs: config)
    monkeypatch.setattr("ready_video.pipeline.resolve_binaries", lambda: DummyPair())
    monkeypatch.setattr("ready_video.pipeline.ffprobe_json", lambda pair, args: VALID_VIDEO_PROBE)
    monkeypatch.setattr("ready_video.pipeline.run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr=""))
    monkeypatch.setattr(
        "ready_video.pipeline.probe_source",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("expensive pipeline stage should have been skipped")),
    )

    assert run_file(source) == output


def test_review_mode_ignores_final_output_cache_and_reuses_work_artifacts(tmp_path, monkeypatch):
    source = tmp_path / "Clip.mov"
    source.write_bytes(b"not really media")
    config = _config(tmp_path)
    config.paths.edited.mkdir(parents=True)
    config.paths.work.mkdir(parents=True)
    content_hash = file_sha256(source)
    cfg_hash = config_sha256(config)
    full_job_hash = job_sha256(content_hash, cfg_hash)
    jid = job_id(full_job_hash)
    output_name = f"{sanitized_stem(source)}_{jid}.mp4"
    output = config.paths.edited / output_name
    output.write_bytes(b"rendered")
    manifest = make_manifest(
        input_path=source,
        output_name=output_name,
        content_hash=content_hash,
        config_hash=cfg_hash,
        job_hash=full_job_hash,
    )
    output.with_suffix(".json").write_text(json.dumps(manifest.model_dump()))
    work_dir = config.paths.work / jid
    work_dir.mkdir()
    pipeline.atomic_write_yaml(work_dir / "resolved-config.yaml", config)
    pipeline.atomic_write_json(work_dir / "job.json", {"job_id": jid, "job_sha256": full_job_hash, "input": str(source)})
    pipeline.atomic_write_json(
        work_dir / "source.json",
        SourceInfo(duration=1.0, width=1080, height=1920, video_stream_index=0, audio_stream_index=1),
    )
    pipeline.atomic_write_json(
        work_dir / "timeline.json",
        Timeline(
            source_duration=1.0,
            edited_duration=1.0,
            segments=[TimelineSegment(i=0, source_start=0.0, source_end=1.0, edited_start=0.0, edited_end=1.0)],
        ),
    )
    pipeline.atomic_write_json(work_dir / "transcript.json", Transcript(duration=1.0))
    (work_dir / "subs.ass").write_text("[Script Info]\n")

    monkeypatch.setattr("ready_video.pipeline.load_config", lambda **kwargs: config)
    monkeypatch.setattr("ready_video.pipeline.resolve_binaries", lambda: DummyPair())
    monkeypatch.setattr("ready_video.pipeline.ffprobe_json", lambda pair, args: VALID_VIDEO_PROBE)
    monkeypatch.setattr("ready_video.pipeline.run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr=""))
    monkeypatch.setattr(
        "ready_video.pipeline.probe_source",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("review should reuse existing work artifacts")),
    )

    def fake_render(input_path, output_path, *args, **kwargs):
        assert kwargs["preview"] is True
        output_path.write_bytes(b"preview")

    monkeypatch.setattr("ready_video.pipeline.render_video", fake_render)

    review_path = run_file(source, review=True)

    assert review_path == work_dir / "review.html"
    assert review_path.exists()
    assert (work_dir / "preview.mp4").read_bytes() == b"preview"


def test_run_file_validates_rendered_output(tmp_path, monkeypatch):
    source = tmp_path / "Clip.mov"
    source.write_bytes(b"not really media")
    config = _config(tmp_path)
    config.silence.enabled = False

    class Source:
        duration = 1.0
        video_stream_index = 0
        audio_stream_index = 1
        color_transfer = ""

        def model_dump(self, *args, **kwargs):
            return {"duration": self.duration, "video_stream_index": 0, "audio_stream_index": 1}

    from ready_video.timeline import SilenceParams, timeline_from_silences
    from ready_video.transcript import Transcript

    monkeypatch.setattr("ready_video.pipeline.load_config", lambda **kwargs: config)
    monkeypatch.setattr("ready_video.pipeline.resolve_binaries", lambda: DummyPair())
    monkeypatch.setattr("ready_video.pipeline.probe_source", lambda *args, **kwargs: Source())
    monkeypatch.setattr(
        "ready_video.pipeline.analyze_silence",
        lambda *args, **kwargs: timeline_from_silences(1.0, [], SilenceParams(enabled=False)),
    )
    monkeypatch.setattr("ready_video.pipeline.build_speech_wav", lambda *args, **kwargs: None)
    monkeypatch.setattr("ready_video.pipeline.transcribe", lambda *args, **kwargs: Transcript(duration=1.0))
    monkeypatch.setattr("ready_video.pipeline.generate_subtitles", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "ready_video.pipeline.measure_loudness",
        lambda *args, **kwargs: {
            "input_i": "-18.0",
            "input_tp": "-2.0",
            "input_lra": "3.0",
            "input_thresh": "-28.0",
            "target_offset": "0.1",
        },
    )

    def fake_render(input_path, output_path, *args, **kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"rendered")

    monkeypatch.setattr("ready_video.pipeline.render_video", fake_render)
    probe_calls = []

    def fake_probe(pair, args):
        probe_calls.append(args)
        return VALID_VIDEO_PROBE

    monkeypatch.setattr("ready_video.pipeline.ffprobe_json", fake_probe)
    monkeypatch.setattr("ready_video.pipeline.run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr=""))

    output = run_file(source)

    assert output.exists()
    assert output.with_suffix(".json").exists()
    assert probe_calls


def test_run_file_writes_timestamped_transcript_txt(tmp_path, monkeypatch):
    source = tmp_path / "Clip.mov"
    source.write_bytes(b"not really media")
    config = _config(tmp_path)
    config.silence.enabled = False

    from ready_video.timeline import SilenceParams, timeline_from_silences
    from ready_video.transcript import Transcript, TranscriptSegment, Word

    transcript = Transcript(
        language="en",
        duration=1.0,
        words=[Word(i=0, w="hola", start=0.0, end=0.5), Word(i=1, w="mundo", start=0.5, end=1.0)],
        segments=[TranscriptSegment(id=0, text="hola mundo", start=0.0, end=1.0, word_range=(0, 1))],
    )

    monkeypatch.setattr("ready_video.pipeline.load_config", lambda **kwargs: config)
    monkeypatch.setattr("ready_video.pipeline.resolve_binaries", lambda: DummyPair())
    monkeypatch.setattr(
        "ready_video.pipeline.probe_source",
        lambda *args, **kwargs: SourceInfo(duration=1.0, width=1080, height=1920, video_stream_index=0, audio_stream_index=1),
    )
    monkeypatch.setattr(
        "ready_video.pipeline.analyze_silence",
        lambda *args, **kwargs: timeline_from_silences(1.0, [], SilenceParams(enabled=False)),
    )
    monkeypatch.setattr("ready_video.pipeline.build_speech_wav", lambda *args, **kwargs: None)
    monkeypatch.setattr("ready_video.pipeline.transcribe", lambda *args, **kwargs: transcript)
    monkeypatch.setattr("ready_video.pipeline.generate_subtitles", lambda *args, **kwargs: None)
    monkeypatch.setattr("ready_video.pipeline.render_video", lambda input_path, output_path, *a, **k: output_path.write_bytes(b"rendered"))
    monkeypatch.setattr("ready_video.pipeline.measure_loudness", lambda *args, **kwargs: None)
    config.render.loudnorm = False
    monkeypatch.setattr("ready_video.pipeline.ffprobe_json", lambda pair, args: VALID_VIDEO_PROBE)
    monkeypatch.setattr("ready_video.pipeline.run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr=""))

    output = run_file(source)

    txt = output.with_suffix(".txt")
    assert txt.exists()
    assert txt.read_text() == "[00:00.00] hola mundo\n"


def test_run_file_backfills_transcript_txt_on_cache_hit(tmp_path, monkeypatch):
    source = tmp_path / "Clip.mov"
    source.write_bytes(b"not really media")
    config = _config(tmp_path)
    config.paths.edited.mkdir()
    config.paths.work.mkdir()
    full_job_hash = job_sha256(file_sha256(source), config_sha256(config))
    jid = job_id(full_job_hash)
    output_name = f"{sanitized_stem(source)}_{jid}.mp4"
    output = config.paths.edited / output_name
    output.write_bytes(b"rendered")
    manifest = make_manifest(
        input_path=source,
        output_name=output_name,
        content_hash=file_sha256(source),
        config_hash=config_sha256(config),
        job_hash=full_job_hash,
    )
    output.with_suffix(".json").write_text(json.dumps(manifest.model_dump()))
    work_dir = config.paths.work / jid
    work_dir.mkdir()
    from ready_video.transcript import Transcript, TranscriptSegment, Word

    pipeline.atomic_write_json(
        work_dir / "transcript.json",
        Transcript(
            duration=1.0,
            words=[Word(i=0, w="hi", start=0.2, end=0.5)],
            segments=[TranscriptSegment(id=0, text="hi there", start=0.2, end=0.9, word_range=(0, 0))],
        ),
    )

    monkeypatch.setattr("ready_video.pipeline.load_config", lambda **kwargs: config)
    monkeypatch.setattr("ready_video.pipeline.resolve_binaries", lambda: DummyPair())
    monkeypatch.setattr("ready_video.pipeline.ffprobe_json", lambda pair, args: VALID_VIDEO_PROBE)
    monkeypatch.setattr("ready_video.pipeline.run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr=""))
    monkeypatch.setattr(
        "ready_video.pipeline.probe_source",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache hit should skip probing")),
    )

    assert run_file(source) == output
    assert output.with_suffix(".txt").read_text() == "[00:00.20] hi there\n"
