from __future__ import annotations

import os
import time
from pathlib import Path

import ready_video.pipeline as pipeline
from ready_video.config import Config


def _config(tmp_path: Path):
    return Config.model_validate(
        {
            "paths": {
                "inbox": tmp_path / "inbox",
                "archive": tmp_path / "archive",
                "work": tmp_path / "work",
                "edited": tmp_path / "edited",
                "failed": tmp_path / "failed",
            },
            "inbox_processing": {"stability_seconds": 8},
        }
    ).resolved()


def _write_old(path: Path, data: bytes | str = b"media") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data)
    else:
        path.write_bytes(data)
    old = time.time() - 30
    os.utime(path, (old, old))


def test_inbox_claims_sidecars_archives_success_and_preserves_failures(monkeypatch, tmp_path: Path, capsys) -> None:
    config = _config(tmp_path)
    _write_old(config.paths.inbox / "good.mp4")
    _write_old(config.paths.inbox / "good.mp4.yaml", "render:\n  aspect: '1:1'\n")
    _write_old(config.paths.inbox / "bad.mp4")
    _write_old(config.paths.inbox / "bad.mp4.yaml", "render:\n  aspect: '9:16'\n")
    fresh = config.paths.inbox / "fresh.mp4"
    fresh.write_bytes(b"still uploading")

    monkeypatch.setattr(pipeline, "load_config", lambda **kwargs: config)

    def fake_run_file(path, *, config_path=None, cli_overrides=None, review=False):
        if path.name == "bad.mp4":
            raise RuntimeError("render failed")
        output = config.paths.edited / "good_abcd1234.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"output")
        return output

    monkeypatch.setattr(pipeline, "run_file", fake_run_file)

    assert pipeline.inbox() == 1

    archive_dir = config.paths.archive / "abcd1234"
    fail_dir = config.paths.failed / "bad"
    assert (archive_dir / "good.mp4").exists()
    assert (archive_dir / "good.mp4.yaml").exists()
    assert (fail_dir / "bad.mp4").exists()
    assert (fail_dir / "bad.mp4.yaml").exists()
    assert "render failed" in (fail_dir / "error.log").read_text()
    assert fresh.exists()
    assert not (config.paths.inbox / ".ready-video.lock").exists()
    assert "processed=1 skipped=1 failed=1" in capsys.readouterr().out


def test_inbox_lock_is_nonblocking(monkeypatch, tmp_path: Path, capsys) -> None:
    config = _config(tmp_path)
    lock_path = config.paths.inbox / ".ready-video.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("pid=123\n")
    monkeypatch.setattr(pipeline, "load_config", lambda **kwargs: config)

    assert pipeline.inbox() == 0

    assert lock_path.exists()
    assert "inbox processing already active" in capsys.readouterr().out
