from __future__ import annotations

import difflib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from platformdirs import user_config_dir
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import ReadyVideoError, invalid_config


SECRET_KEYS = {"api_key", "apikey", "token", "secret", "password"}
ASPECTS = {"9:16", "1:1", "16:9"}
SUBTITLE_MODES = {"karaoke", "line", "none"}
SUBTITLE_PRESETS: dict[str, dict[str, Any]] = {
    "bold": {
        "mode": "karaoke",
        "font_family": "Arial",
        "font_size": 86,
        "primary_color": "#FFFFFF",
        "highlight_color": "#33E1FF",
        "outline_color": "#000000",
        "outline_width": 7,
        "shadow": 0,
        "vertical_position": 0.78,
        "max_words_per_line": 3,
        "max_chars_per_line": 22,
        "uppercase": False,
    },
    "clean": {
        "mode": "karaoke",
        "font_family": "Arial",
        "font_size": 66,
        "primary_color": "#FFFFFF",
        "highlight_color": "#FFE066",
        "outline_color": "#111111",
        "outline_width": 4,
        "shadow": 1,
        "vertical_position": 0.80,
        "max_words_per_line": 5,
        "max_chars_per_line": 34,
        "uppercase": False,
    },
    "minimal": {
        "mode": "line",
        "font_family": "Arial",
        "font_size": 48,
        "primary_color": "#FFFFFF",
        "highlight_color": "#FFFFFF",
        "outline_color": "#111111",
        "outline_width": 2,
        "shadow": 0,
        "vertical_position": 0.84,
        "max_words_per_line": 6,
        "max_chars_per_line": 42,
        "uppercase": False,
    },
}

CONFIG_COMMENTS = {
    "paths": "Runtime directories. Relative paths are resolved from the current working directory.",
    "silence": "Silence analysis settings used to build the canonical edit timeline.",
    "transcription": "WhisperX model selection. auto chooses a practical model/device/compute type.",
    "subtitles": "Burned subtitle settings. null values inherit from the selected preset.",
    "render": "Final render settings. Output frame rate and pixel format are fixed internally.",
    "inbox_processing": "One-shot inbox claiming and lifecycle settings.",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PathsConfig(StrictModel):
    inbox: Path = Path("./inbox")
    archive: Path = Path("./archive")
    work: Path = Path("./work")
    edited: Path = Path("./edited")
    failed: Path = Path("./failed")


class SilenceConfig(StrictModel):
    enabled: bool = True
    threshold_db: float = -35.0
    min_silence_s: float = 0.45
    margin_before_s: float = 0.12
    margin_after_s: float = 0.18
    # Transcript-driven second pass: after sound-based silence removal, compress
    # pauses BETWEEN spoken words that survived the dB threshold (quiet room tone,
    # breath). Cuts only across aligned word boundaries, never clipping speech.
    transcript_trim: bool = True
    transcript_trim_max_gap_s: float = 0.35
    transcript_trim_word_margin_s: float = 0.08

    @property
    def threshold(self) -> float:
        return self.threshold_db

    @property
    def min_duration(self) -> float:
        return self.min_silence_s

    @property
    def start_margin(self) -> float:
        return self.margin_before_s

    @property
    def end_margin(self) -> float:
        return self.margin_after_s

    @property
    def margins(self) -> tuple[float, float]:
        return (self.margin_before_s, self.margin_after_s)

    @model_validator(mode="after")
    def validate_ranges(self) -> "SilenceConfig":
        if self.min_silence_s <= 0:
            raise ValueError("min_silence_s must be positive")
        if self.margin_before_s < 0 or self.margin_after_s < 0:
            raise ValueError("silence margins must be nonnegative")
        if self.transcript_trim_max_gap_s < 0:
            raise ValueError("transcript_trim_max_gap_s must be nonnegative")
        if self.transcript_trim_word_margin_s < 0:
            raise ValueError("transcript_trim_word_margin_s must be nonnegative")
        return self


class TranscriptionConfig(StrictModel):
    model: str = "auto"
    language: str = "auto"
    device: str = "auto"
    compute_type: str = "auto"
    initial_prompt: str = ""


class SubtitleConfig(StrictModel):
    mode: str | None = None
    preset: str = "bold"
    export_srt: bool = False
    font_family: str | None = None
    font_file: Path | None = None
    font_size: int | None = None
    primary_color: str | None = None
    highlight_color: str | None = None
    outline_color: str | None = None
    outline_width: int | None = None
    shadow: int | None = None
    vertical_position: float | None = None
    max_words_per_line: int | None = None
    max_chars_per_line: int | None = None
    uppercase: bool | None = None

    @property
    def font(self) -> str | None:
        return self.font_family

    @field_validator("preset")
    @classmethod
    def validate_preset(cls, value: str) -> str:
        if value not in SUBTITLE_PRESETS:
            raise ValueError(f"subtitles.preset must be one of {sorted(SUBTITLE_PRESETS)}")
        return value

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str | None) -> str | None:
        if value is not None and value not in SUBTITLE_MODES:
            raise ValueError(f"subtitles.mode must be one of {sorted(SUBTITLE_MODES)}")
        return value

    @field_validator("primary_color", "highlight_color", "outline_color")
    @classmethod
    def validate_color(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if len(value) != 7 or not value.startswith("#"):
            raise ValueError("colors must use #RRGGBB")
        int(value[1:], 16)
        return value.upper()

    @model_validator(mode="after")
    def validate_font_file(self) -> "SubtitleConfig":
        if self.font_file is not None:
            if self.font_file.suffix.lower() not in {".ttf", ".otf"}:
                raise ValueError("font_file must be a .ttf or .otf file")
            if not self.font_file.is_file():
                raise ValueError(f"font_file does not exist: {self.font_file}")
        if self.vertical_position is not None and not 0 <= self.vertical_position <= 1:
            raise ValueError("vertical_position must be in [0, 1]")
        return self


class RenderConfig(StrictModel):
    aspect: str = "9:16"
    crf: int = 18
    loudnorm: bool = True
    loudnorm_target_lufs: float = -14.0
    crop_x_offset: float = 0.0
    crop_y_offset: float = 0.0
    hwaccel: str = "none"

    @property
    def target_lufs(self) -> float:
        return self.loudnorm_target_lufs

    @property
    def audio_offset(self) -> float:
        return self.crop_x_offset

    @property
    def video_offset(self) -> float:
        return self.crop_y_offset

    @property
    def target(self) -> float:
        return self.loudnorm_target_lufs

    @property
    def offsets(self) -> tuple[float, float]:
        return (self.crop_x_offset, self.crop_y_offset)

    @field_validator("aspect")
    @classmethod
    def validate_aspect(cls, value: str) -> str:
        if value not in ASPECTS:
            raise ValueError(f"render.aspect must be one of {sorted(ASPECTS)}")
        return value

    @model_validator(mode="after")
    def validate_render(self) -> "RenderConfig":
        if not -1 <= self.crop_x_offset <= 1 or not -1 <= self.crop_y_offset <= 1:
            raise ValueError("crop offsets must be in [-1, 1]")
        if not 0 <= self.crf <= 51:
            raise ValueError("render.crf must be in [0, 51]")
        if self.hwaccel not in {"none", "auto", "nvenc", "videotoolbox"}:
            raise ValueError("render.hwaccel must be none, auto, nvenc, or videotoolbox")
        return self


class InboxProcessingConfig(StrictModel):
    stability_seconds: int = 8
    extensions: list[str] = Field(default_factory=lambda: ["mp4", "mov", "mkv", "webm"])
    delete_original_on_success: bool = False

    @property
    def stability(self) -> int:
        return self.stability_seconds


class Config(StrictModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    silence: SilenceConfig = Field(default_factory=SilenceConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    subtitles: SubtitleConfig = Field(default_factory=SubtitleConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    inbox_processing: InboxProcessingConfig = Field(default_factory=InboxProcessingConfig)

    @property
    def inbox(self) -> InboxProcessingConfig:
        return self.inbox_processing

    def resolved(self) -> "ResolvedConfig":
        data = self.model_dump()
        preset = SUBTITLE_PRESETS[self.subtitles.preset]
        subs = data["subtitles"]
        for key, value in preset.items():
            if subs.get(key) is None:
                subs[key] = value
        return ResolvedConfig.model_validate(data)


class ResolvedSubtitleConfig(StrictModel):
    mode: str
    preset: str
    export_srt: bool
    font_family: str
    font_file: Path | None = None
    font_size: int
    primary_color: str
    highlight_color: str
    outline_color: str
    outline_width: int
    shadow: int
    vertical_position: float
    max_words_per_line: int
    max_chars_per_line: int
    uppercase: bool


class ResolvedConfig(Config):
    subtitles: ResolvedSubtitleConfig


def deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _check_secret_keys(data: Any, path: str = "") -> None:
    if isinstance(data, Mapping):
        for key, value in data.items():
            key_text = str(key).lower()
            if key_text in SECRET_KEYS or any(part in SECRET_KEYS for part in key_text.split("_")):
                loc = f"{path}.{key}" if path else str(key)
                raise ValueError(f"Config key {loc} looks like a secret and is not supported")
            _check_secret_keys(value, f"{path}.{key}" if path else str(key))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            _check_secret_keys(item, f"{path}[{i}]")


def _allowed_tree(model: type[BaseModel]) -> dict[str, Any]:
    tree: dict[str, Any] = {}
    for name, field in model.model_fields.items():
        ann = field.annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            tree[name] = _allowed_tree(ann)
        else:
            tree[name] = None
    return tree


def _check_unknown_keys(data: Any, allowed: dict[str, Any], path: str = "") -> None:
    if not isinstance(data, Mapping):
        return
    for key, value in data.items():
        if key not in allowed:
            options = list(allowed)
            suggestion = difflib.get_close_matches(str(key), options, n=1)
            message = f"Unknown config key: {path + '.' if path else ''}{key}"
            if suggestion:
                message += f". Did you mean {suggestion[0]}?"
            raise invalid_config(message)
        child = allowed[key]
        if isinstance(child, dict):
            _check_unknown_keys(value, child, f"{path}.{key}" if path else str(key))


def _normalize_aliases(data: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    silence = normalized.get("silence")
    if isinstance(silence, Mapping):
        silence = dict(silence)
        if "threshold" in silence and "threshold_db" not in silence:
            silence["threshold_db"] = silence.pop("threshold")
        if "min" in silence and "min_silence_s" not in silence:
            silence["min_silence_s"] = silence.pop("min")
        if "min_duration" in silence and "min_silence_s" not in silence:
            silence["min_silence_s"] = silence.pop("min_duration")
        if "start_margin" in silence and "margin_before_s" not in silence:
            silence["margin_before_s"] = silence.pop("start_margin")
        if "end_margin" in silence and "margin_after_s" not in silence:
            silence["margin_after_s"] = silence.pop("end_margin")
        normalized["silence"] = silence
    subtitles = normalized.get("subtitles")
    if isinstance(subtitles, Mapping) and "font" in subtitles and "font_family" not in subtitles:
        subtitles = dict(subtitles)
        subtitles["font_family"] = subtitles.pop("font")
        normalized["subtitles"] = subtitles
    render = normalized.get("render")
    if isinstance(render, Mapping):
        render = dict(render)
        if "target" in render and "loudnorm_target_lufs" not in render:
            render["loudnorm_target_lufs"] = render.pop("target")
        if "target_lufs" in render and "loudnorm_target_lufs" not in render:
            render["loudnorm_target_lufs"] = render.pop("target_lufs")
        if "audio_offset" in render and "crop_x_offset" not in render:
            render["crop_x_offset"] = render.pop("audio_offset")
        if "video_offset" in render and "crop_y_offset" not in render:
            render["crop_y_offset"] = render.pop("video_offset")
        normalized["render"] = render
    if "inbox" in normalized and "inbox_processing" not in normalized:
        normalized["inbox_processing"] = normalized.pop("inbox")
    return normalized


def load_yaml_file(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise invalid_config(f"Could not parse YAML file {path}", str(exc)) from exc
    if not isinstance(raw, dict):
        raise invalid_config(f"YAML config must be a mapping: {path}")
    raw = _normalize_aliases(raw)
    _check_secret_keys(raw)
    _check_unknown_keys(raw, _allowed_tree(Config))
    return raw


def env_overrides(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    env = env or os.environ
    prefix = "READY_VIDEO_"
    data: dict[str, Any] = {}
    for raw_key, raw_value in env.items():
        if not raw_key.startswith(prefix) or raw_key in {"READY_VIDEO_FFMPEG", "READY_VIDEO_FFPROBE"}:
            continue
        parts = raw_key[len(prefix) :].lower().split("__")
        cursor = data
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = yaml.safe_load(raw_value)
    if data:
        data = _normalize_aliases(data)
        _check_secret_keys(data)
        _check_unknown_keys(data, _allowed_tree(Config))
    return data


def project_config_path(cwd: Path | None = None) -> Path:
    return (cwd or Path.cwd()) / "config.yaml"


def user_config_path() -> Path:
    return Path(user_config_dir("ready-video", "ready-video")) / "config.yaml"


def load_config(
    raw: Mapping[str, Any] | None = None,
    *,
    path: Path | None = None,
    config_path: Path | None = None,
    sidecar_path: Path | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
    cwd: Path | None = None,
) -> ResolvedConfig:
    data: dict[str, Any] = {}
    if raw is not None:
        raw = _normalize_aliases(raw)
        _check_secret_keys(raw)
        _check_unknown_keys(raw, _allowed_tree(Config))
        try:
            return Config.model_validate(raw)
        except Exception as exc:
            if isinstance(exc, ReadyVideoError):
                raise
            raise invalid_config(str(exc)) from exc
    if path is not None:
        config_path = path
    for candidate in [user_config_path(), project_config_path(cwd)]:
        if candidate.is_file():
            data = deep_merge(data, load_yaml_file(candidate))
    if sidecar_path and sidecar_path.is_file():
        data = deep_merge(data, load_yaml_file(sidecar_path))
    data = deep_merge(data, env_overrides())
    if config_path:
        data = deep_merge(data, load_yaml_file(config_path))
    if cli_overrides:
        data = deep_merge(data, cli_overrides)
    try:
        return Config.model_validate(data).resolved()
    except Exception as exc:
        if isinstance(exc, ReadyVideoError):
            raise
        raise invalid_config(str(exc)) from exc


def config_to_yaml(config: BaseModel) -> str:
    data = json.loads(config.model_dump_json())
    return yaml.safe_dump(data, sort_keys=False)


def generate_default_config_yaml() -> str:
    data = json.loads(Config().model_dump_json())
    lines = [
        "# Ready Video configuration",
        "# All settings are optional; deleting this file restores built-in defaults.",
        "# null subtitle fields inherit from the selected preset.",
        "",
    ]
    dumped = yaml.safe_dump(data, sort_keys=False).splitlines()
    current_section: str | None = None
    for line in dumped:
        if line and not line.startswith(" ") and line.endswith(":"):
            section = line[:-1]
            comment = CONFIG_COMMENTS.get(section)
            if current_section is not None:
                lines.append("")
            if comment:
                lines.append(f"# {comment}")
            current_section = section
        lines.append(line)
    return "\n".join(lines) + "\n"


ReadyVideoConfig = Config
