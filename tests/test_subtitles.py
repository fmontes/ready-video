from ready_video.subtitles import SubtitleStyle, cues_from_words, generate_ass, generate_srt
from ready_video.timeline import TimelineSegment


def test_generate_srt_sorts_and_formats_cues():
    srt = generate_srt(
        [
            TimelineSegment(2, 3, "second"),
            TimelineSegment(0.5, 1.25, "first"),
            TimelineSegment(4, 5, "  "),
        ]
    )

    assert srt == (
        "1\n"
        "00:00:00,500 --> 00:00:01,250\n"
        "first\n\n"
        "2\n"
        "00:00:02,000 --> 00:00:03,000\n"
        "second\n"
    )


def test_generate_ass_uses_bold_preset_and_escapes_text():
    ass = generate_ass([TimelineSegment(0, 1.2, "Hello\n{world}")])

    assert "Style: Bold,Arial,72" in ass
    assert ",-1,0,0,0,100,100" in ass
    assert "Dialogue: 0,0:00:00.00,0:00:01.20,Bold,,0,0,0,,Hello\\N\\{world\\}" in ass


def test_generate_ass_accepts_custom_style():
    ass = generate_ass(
        [TimelineSegment(0, 1, "quiet")],
        style=SubtitleStyle(name="Tiny", font_size=24, bold=False),
        resolution=(720, 1280),
    )

    assert "PlayResX: 720" in ass
    assert "Style: Tiny,Arial,24" in ass
    assert ",0,0,0,0,100,100" in ass
    assert "Dialogue: 0,0:00:00.00,0:00:01.00,Tiny" in ass


def test_cues_from_words_groups_on_large_gaps():
    cues = cues_from_words(
        [
            {"start": 0, "end": 0.2, "word": "Ready"},
            {"start": 0.3, "end": 0.5, "word": "video"},
            {"start": 2.0, "end": 2.2, "word": "now"},
        ],
        max_gap=0.8,
    )

    assert cues == [
        TimelineSegment(0, 0.5, "Ready video"),
        TimelineSegment(2.0, 2.2, "now"),
    ]
