from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReadyVideoError(ValueError):
    code: str
    message: str
    remediation: str | None = None
    details: str | None = None

    def __str__(self) -> str:
        text = f"{self.code}: {self.message}"
        if self.remediation:
            text += f" Suggested remediation: {self.remediation}"
        return text


class TranscriptError(ReadyVideoError):
    def __init__(self, message: str, remediation: str | None = None, details: str | None = None):
        super().__init__("TRANSCRIPTION_FAILED", message, remediation, details)


class ZoomValidationError(ReadyVideoError):
    def __init__(self, message: str, remediation: str | None = None, details: str | None = None):
        super().__init__("ZOOM_PLANNING_FAILED", message, remediation, details)


def invalid_config(message: str, remediation: str | None = None) -> ReadyVideoError:
    return ReadyVideoError("INVALID_CONFIG", message, remediation)
