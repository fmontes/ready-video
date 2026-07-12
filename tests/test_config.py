import pytest

pydantic = pytest.importorskip("pydantic")

import yaml

import ready_video.config as config_module
from ready_video.config import ReadyVideoConfig, generate_default_config_yaml, load_config


def test_config_defaults_match_v1_contract():
    config = ReadyVideoConfig()

    assert str(config.paths.inbox) == "inbox"
    assert str(config.paths.archive) == "archive"
    assert str(config.paths.work) == "work"
    assert str(config.paths.edited) == "edited"
    assert str(config.paths.failed) == "failed"
    assert config.silence.enabled is True
    assert config.silence.threshold_db == -35
    assert config.silence.min_duration == 0.45
    assert config.silence.start_margin == 0.12
    assert config.silence.end_margin == 0.18
    assert config.transcription.model == "auto"
    assert config.transcription.language == "auto"
    assert config.transcription.device == "auto"
    assert config.transcription.compute_type == "auto"
    assert config.subtitles.preset == "bold"
    assert config.render.aspect == "9:16"
    assert config.render.crf == 18
    assert config.render.loudnorm is True
    assert config.render.target_lufs == -14
    assert config.render.audio_offset == 0
    assert config.render.video_offset == 0
    assert config.render.hwaccel == "none"
    assert config.inbox.stability_seconds == 8
    assert config.inbox.extensions == ["mp4", "mov", "mkv", "webm"]


def test_config_rejects_unknown_keys():
    with pytest.raises(Exception):
        load_config({"render": {"not_a_real_key": True}})


def test_config_rejects_secret_keys():
    with pytest.raises(ValueError, match="looks like a secret"):
        load_config({"transcription": {"api_key": "do-not-store-this"}})


def test_nullable_subtitle_fields_are_allowed():
    config = load_config({"subtitles": {"font": None, "font_size": None}})

    assert config.subtitles.font is None
    assert config.subtitles.font_size is None


def test_generated_config_round_trips_to_resolved_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(generate_default_config_yaml())

    generated = load_config(config_path=path, cwd=tmp_path)
    defaults = ReadyVideoConfig().resolved()

    assert generated.model_dump(mode="json") == defaults.model_dump(mode="json")
    assert "# Ready Video configuration" in path.read_text()


def test_layer_precedence_user_project_sidecar_env_cli(monkeypatch, tmp_path):
    user_path = tmp_path / "user.yaml"
    project_path = tmp_path / "project.yaml"
    sidecar_path = tmp_path / "clip.mp4.yaml"
    user_path.write_text(yaml.safe_dump({"render": {"aspect": "16:9", "crf": 30}}))
    project_path.write_text(yaml.safe_dump({"render": {"aspect": "1:1"}}))
    sidecar_path.write_text(yaml.safe_dump({"render": {"crf": 22}}))
    monkeypatch.setattr(config_module, "user_config_path", lambda: user_path)
    monkeypatch.setattr(config_module, "project_config_path", lambda cwd=None: project_path)
    monkeypatch.setenv("READY_VIDEO_RENDER__CRF", "24")

    config = load_config(sidecar_path=sidecar_path, cli_overrides={"render": {"aspect": "9:16"}}, cwd=tmp_path)

    assert config.render.aspect == "9:16"
    assert config.render.crf == 24


def test_env_aliases_are_normalized(monkeypatch):
    monkeypatch.setenv("READY_VIDEO_RENDER__TARGET_LUFS", "-16")

    config = load_config()

    assert config.render.loudnorm_target_lufs == -16
