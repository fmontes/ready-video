from __future__ import annotations

from pathlib import Path

import pytest

from ready_video import agent, pipeline
from ready_video.cli import main
from ready_video.config import Config, PathsConfig
from ready_video.ffmpeg import BinaryPair, CapabilityReport


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    assert "ready-video" in capsys.readouterr().out


def test_cli_doctor_smoke_reports_missing_transcription(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def missing_import(name: str) -> object:
        if name == "whisperx":
            raise ModuleNotFoundError("No module named 'whisperx'")
        return object()

    monkeypatch.setattr(pipeline.importlib, "import_module", missing_import)

    assert main(["doctor"]) == 1
    output = capsys.readouterr().out
    assert "doctor is available" in output
    assert "FAIL    whisperx: import failed; install the transcription dependency" in output


def test_cli_doctor_reports_environment_details(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ffmpeg_path = tmp_path / "ffmpeg"
    ffprobe_path = tmp_path / "ffprobe"
    pair = BinaryPair(ffmpeg_path, ffprobe_path, "environment", provider="test-provider", cache_dir=tmp_path / "cache")
    config = Config(
        paths=PathsConfig(
            inbox=tmp_path / "inbox",
            archive=tmp_path / "archive",
            work=tmp_path / "work",
            edited=tmp_path / "edited",
            failed=tmp_path / "failed",
        )
    ).resolved()
    lock_path = config.paths.inbox / ".ready-video.lock"

    def fake_capability_report(binary_pair: BinaryPair) -> CapabilityReport:
        assert binary_pair == pair
        return CapabilityReport(
            pair=pair,
            ffmpeg_version="ffmpeg version 6.1-ready test-build",
            ffprobe_version="ffprobe version 6.1-ready test-build",
            release="6.1-ready",
            filters=frozenset({"ass", "loudnorm", "silencedetect"}),
            encoders=frozenset({"aac", "libx264", "h264_videotoolbox"}),
            required_filters=frozenset({"ass", "loudnorm", "silencedetect"}),
            required_encoders=frozenset({"aac", "libx264"}),
            optional_encoders=frozenset({"h264_videotoolbox", "h264_nvenc"}),
        )

    monkeypatch.setattr(pipeline, "load_config", lambda: config)
    monkeypatch.setattr(pipeline, "resolve_binaries", lambda *, install_missing: pair)
    monkeypatch.setattr(pipeline, "capability_report", fake_capability_report)
    monkeypatch.setattr(pipeline.importlib, "import_module", lambda name: object())
    monkeypatch.setattr(pipeline.metadata, "version", lambda name: "3.3.0")
    monkeypatch.setattr(agent.shutil, "which", lambda name: "/usr/bin/codex" if name == "codex" else None)

    assert main(["doctor"]) == 0

    output = capsys.readouterr().out
    assert "ffmpeg.resolution: source=environment provider=test-provider" in output
    assert "ffmpeg.capabilities: release=6.1-ready ffprobe_release=6.1-ready required_filters=ok required_encoders=ok" in output
    assert "ffmpeg.optional_encoders: available=h264_videotoolbox missing=h264_nvenc" in output
    assert "whisperx: import ok (version 3.3.0)" in output
    assert "agent.codex: /usr/bin/codex" in output
    assert f"path.inbox: {config.paths.inbox}" in output
    assert f"inbox.lock: absent ({lock_path}" in output
