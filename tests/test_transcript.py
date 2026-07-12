import json

import pytest

from ready_video.errors import ReadyVideoError, TranscriptError
from ready_video.config import Config
from ready_video.transcript import _call_whisperx_transcribe, create_transcript, normalize_whisperx, transcribe


def test_empty_transcript_fallback_writes_segments(tmp_path):
    output = tmp_path / "transcript.json"

    payload = create_transcript(tmp_path / "speech.wav", output, fallback="empty")

    assert payload["engine"] == "empty-fallback"
    assert payload["segments"] == []
    assert json.loads(output.read_text())["segments"] == []


def test_missing_whisperx_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    with pytest.raises(TranscriptError, match="WhisperX is not installed"):
        create_transcript(tmp_path / "speech.wav", tmp_path / "transcript.json")


def test_transcribe_requires_whisperx_in_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr("ready_video.transcript.media_duration", lambda speech_wav, pair: 1.0)
    monkeypatch.setattr("ready_video.transcript._transcribe_with_whisperx", lambda speech_wav, duration, config: (_ for _ in ()).throw(ImportError("no whisperx")))

    with pytest.raises(ReadyVideoError) as error:
        transcribe(tmp_path / "speech.wav", tmp_path / "transcript.json", pair=object(), config=object())

    assert error.value.code == "TRANSCRIPTION_UNAVAILABLE"
    assert not (tmp_path / "transcript.json").exists()


def test_whisperx_transcribe_call_filters_kwargs_for_newer_api(tmp_path):
    model = NewWhisperxModel()
    config = Config().resolved()
    config.transcription.language = "en"

    result = _call_whisperx_transcribe(model, tmp_path / "speech.wav", config)

    assert result["language"] == "en"
    assert model.calls == [("speech.wav", {"batch_size": 8, "language": "en"})]


def test_whisperx_transcribe_rejects_unsupported_initial_prompt(tmp_path):
    model = NewWhisperxModel()
    config = Config().resolved()
    config.transcription.initial_prompt = "Names: Ready Video"

    with pytest.raises(ReadyVideoError) as error:
        _call_whisperx_transcribe(model, tmp_path / "speech.wav", config)

    assert error.value.code == "TRANSCRIPTION_FAILED"
    assert "initial_prompt" in error.value.message


def test_normalize_preserves_tokens_and_marks_interpolated_gaps():
    transcript = normalize_whisperx(
        {
            "segments": [
                {
                    "start": 10.0,
                    "end": 20.0,
                    "text": "Well actually hello beautiful world again",
                    "words": [
                        {"word": "Well"},
                        {"word": "actually"},
                        {"word": "hello", "start": 14.0, "end": 15.0},
                        {"word": "beautiful"},
                        {"word": "world", "start": 18.0, "end": 19.0},
                        {"word": "again"},
                    ],
                }
            ]
        },
        30.0,
    )

    assert [word.w for word in transcript.words] == ["Well", "actually", "hello", "beautiful", "world", "again"]
    assert [word.timing for word in transcript.words] == [
        "interpolated",
        "interpolated",
        "aligned",
        "interpolated",
        "aligned",
        "interpolated",
    ]
    assert transcript.words[0].start == pytest.approx(10.0)
    assert transcript.words[0].end == pytest.approx(11.333, abs=0.001)
    assert transcript.words[1].start == pytest.approx(11.333, abs=0.001)
    assert transcript.words[1].end == pytest.approx(14.0)
    assert transcript.words[3].start == pytest.approx(15.0)
    assert transcript.words[3].end == pytest.approx(18.0)
    assert transcript.words[5].start == pytest.approx(19.0)
    assert transcript.words[5].end == pytest.approx(20.0)


class NewWhisperxModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, batch_size=None, language=None):
        from pathlib import Path

        self.calls.append((Path(audio).name, {"batch_size": batch_size, "language": language}))
        return {"language": language, "segments": []}


def test_normalize_keeps_textual_tokens_when_empty_items_are_skipped():
    transcript = normalize_whisperx(
        {
            "segments": [
                {
                    "start": 0.0,
                    "end": 4.0,
                    "text": "alpha beta gamma",
                    "words": [
                        {"word": "alpha"},
                        {"word": ""},
                        {"word": "beta", "start": 1.5, "end": 2.0},
                        {},
                        {"word": "gamma"},
                    ],
                }
            ]
        },
        5.0,
    )

    assert [word.w for word in transcript.words] == ["alpha", "beta", "gamma"]
    assert transcript.words[1].timing == "aligned"
    assert transcript.words[1].start == pytest.approx(1.5)
    assert transcript.words[1].end == pytest.approx(2.0)
    assert transcript.words[2].start == pytest.approx(2.0)
    assert transcript.words[2].end == pytest.approx(4.0)


def test_normalize_allocates_all_untimed_words_across_segment_bounds_by_character_count():
    transcript = normalize_whisperx(
        {
            "segments": [
                {
                    "start": 2.0,
                    "end": 8.0,
                    "text": "a bbbbb",
                    "words": [{"word": "a"}, {"word": "bbbbb"}],
                }
            ]
        },
        10.0,
    )

    assert [word.timing for word in transcript.words] == ["interpolated", "interpolated"]
    assert transcript.words[0].start == pytest.approx(2.0)
    assert transcript.words[0].end == pytest.approx(3.0)
    assert transcript.words[1].start == pytest.approx(3.0)
    assert transcript.words[1].end == pytest.approx(8.0)


def test_normalize_enforces_monotonic_nonnegative_bounded_timestamps_and_warns():
    transcript = normalize_whisperx(
        {
            "segments": [
                {
                    "start": -5.0,
                    "end": 3.0,
                    "text": "hello overlap",
                    "words": [
                        {"word": "hello", "start": -2.0, "end": 1.0},
                        {"word": "overlap", "start": 0.5, "end": 5.0},
                    ],
                },
                {
                    "start": 2.0,
                    "end": 1.0,
                    "text": "bad segment",
                    "words": [{"word": "bad"}],
                },
                {
                    "start": 4.0,
                    "end": 6.0,
                    "text": "no word tokens",
                    "words": [{"word": " "}],
                },
            ]
        },
        3.5,
    )

    assert [(word.start, word.end) for word in transcript.words] == [(0.0, 1.0), (1.0, 3.0)]
    assert all(0.0 <= word.start <= word.end <= transcript.duration for word in transcript.words)
    assert [warning["reason"] for warning in transcript.warnings] == ["invalid_segment_bounds", "no_textual_word_tokens"]
