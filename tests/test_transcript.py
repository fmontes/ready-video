import json

import pytest

from ready_video.errors import ReadyVideoError, TranscriptError
from ready_video.config import Config
from ready_video.transcript import (
    _WORD_START_CORRECTION_S,
    Transcript,
    TranscriptSegment,
    Word,
    _segment_to_dict,
    create_transcript,
    format_transcript_txt,
    normalize_transcript,
    transcribe,
)


def test_empty_transcript_fallback_writes_segments(tmp_path):
    output = tmp_path / "transcript.json"

    payload = create_transcript(tmp_path / "speech.wav", output, fallback="empty")

    assert payload["engine"] == "empty-fallback"
    assert payload["segments"] == []
    assert json.loads(output.read_text())["segments"] == []


def test_missing_transcription_backend_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    with pytest.raises(TranscriptError, match="faster-whisper is not installed"):
        create_transcript(tmp_path / "speech.wav", tmp_path / "transcript.json")


def test_transcribe_reports_missing_backend_as_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr("ready_video.transcript.media_duration", lambda speech_wav, pair: 1.0)
    monkeypatch.setattr(
        "ready_video.transcript._transcribe_with_faster_whisper",
        lambda speech_wav, duration, config: (_ for _ in ()).throw(ImportError("no faster_whisper")),
    )

    with pytest.raises(ReadyVideoError) as error:
        transcribe(tmp_path / "speech.wav", tmp_path / "transcript.json", pair=object(), config=object())

    assert error.value.code == "TRANSCRIPTION_UNAVAILABLE"
    assert not (tmp_path / "transcript.json").exists()


def test_segment_to_dict_applies_word_timing_correction():
    # faster-whisper word starts run early; _segment_to_dict nudges them later.
    class Word:
        def __init__(self, word, start, end):
            self.word, self.start, self.end = word, start, end

    class Segment:
        text = "hola mundo"
        start = 1.0
        end = 2.0
        words = [Word("hola", 1.00, 1.30), Word("mundo", 1.40, 1.90)]

    result = _segment_to_dict(Segment())

    assert result["words"][0]["start"] == pytest.approx(1.00 + _WORD_START_CORRECTION_S)
    assert result["words"][1]["start"] == pytest.approx(1.40 + _WORD_START_CORRECTION_S)
    # correction never pushes a start past its (also-corrected) end
    assert result["words"][0]["end"] >= result["words"][0]["start"]


def test_segment_to_dict_clamps_negative_starts_to_zero():
    class Word:
        def __init__(self, word, start, end):
            self.word, self.start, self.end = word, start, end

    class Segment:
        text = "x"
        start = 0.0
        end = 0.1
        words = [Word("x", 0.0, 0.05)]

    result = _segment_to_dict(Segment())

    assert result["words"][0]["start"] >= 0.0


def test_normalize_preserves_tokens_and_marks_interpolated_gaps():
    transcript = normalize_transcript(
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


def test_normalize_keeps_textual_tokens_when_empty_items_are_skipped():
    transcript = normalize_transcript(
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
    transcript = normalize_transcript(
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
    transcript = normalize_transcript(
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


def test_format_transcript_txt_uses_timestamped_segment_lines():
    transcript = Transcript(
        language="en",
        duration=125.0,
        words=[],
        segments=[
            TranscriptSegment(id=0, text="  So today we're building. ", start=3.2, end=5.0, word_range=(0, 1)),
            TranscriptSegment(id=1, text="Second line here", start=65.07, end=70.0, word_range=(2, 3)),
        ],
    )

    text = format_transcript_txt(transcript)

    assert text == "[00:03.20] So today we're building.\n[01:05.07] Second line here\n"


def test_format_transcript_txt_falls_back_to_words_without_segments():
    transcript = Transcript(
        duration=5.0,
        words=[Word(i=0, w="hello", start=0.0, end=0.5), Word(i=1, w="world", start=0.6, end=1.0)],
        segments=[],
    )

    text = format_transcript_txt(transcript)

    assert text == "[00:00.00] hello\n[00:00.60] world\n"


def test_format_transcript_txt_empty_when_nothing_to_write():
    assert format_transcript_txt(Transcript(duration=0.0)) == ""
