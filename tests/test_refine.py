from ready_video.refine import (
    TranscriptTrimParams,
    refine_timeline_with_transcript,
    retime_transcript,
)
from ready_video.timeline import Timeline, TimelineSegment
from ready_video.transcript import Transcript, Word


def _timeline(spans):
    """Build a Timeline from (source_start, source_end) spans, contiguous edited axis."""
    segments = []
    edited = 0.0
    for i, (ss, se) in enumerate(spans):
        dur = se - ss
        segments.append(TimelineSegment(i=i, source_start=ss, source_end=se, edited_start=edited, edited_end=edited + dur))
        edited += dur
    return Timeline(source_duration=spans[-1][1] + 5, edited_duration=edited, segments=segments)


def _transcript(words, duration):
    return Transcript(language="es", duration=duration, words=[Word(i=i, **w) for i, w in enumerate(words)])


def test_disabled_returns_same_timeline():
    tl = _timeline([(0.0, 10.0)])
    tr = _transcript([{"w": "hola", "start": 1.0, "end": 1.5}], 10.0)
    out = refine_timeline_with_transcript(tl, tr, TranscriptTrimParams(enabled=False))
    assert out is tl


def test_long_gap_between_aligned_words_is_compressed():
    # One kept segment; a 6s pause between two aligned words survived sound removal.
    tl = _timeline([(0.0, 10.0)])
    tr = _transcript(
        [
            {"w": "uno", "start": 1.0, "end": 1.5, "timing": "aligned"},
            {"w": "dos", "start": 7.5, "end": 8.0, "timing": "aligned"},
        ],
        10.0,
    )
    out = refine_timeline_with_transcript(tl, tr, TranscriptTrimParams(enabled=True, max_gap_s=0.35, word_margin_s=0.08))
    # 6s gap -> ~0.35s, so edited duration drops by ~5.65s.
    assert out.edited_duration < tl.edited_duration
    assert tl.edited_duration - out.edited_duration > 5.0


def test_interpolated_gap_is_not_cut():
    # Same gap, but the second word is interpolated -> gap must be kept intact.
    tl = _timeline([(0.0, 10.0)])
    aligned = _transcript(
        [
            {"w": "uno", "start": 1.0, "end": 1.5, "timing": "aligned"},
            {"w": "dos", "start": 7.5, "end": 8.0, "timing": "aligned"},
        ],
        10.0,
    )
    interp = _transcript(
        [
            {"w": "uno", "start": 1.0, "end": 1.5, "timing": "aligned"},
            {"w": "dos", "start": 7.5, "end": 8.0, "timing": "interpolated"},
        ],
        10.0,
    )
    params = TranscriptTrimParams(enabled=True, max_gap_s=0.35, word_margin_s=0.08)
    cut = refine_timeline_with_transcript(tl, aligned, params)
    kept = refine_timeline_with_transcript(tl, interp, params)
    assert kept.edited_duration > cut.edited_duration
    # interpolated case leaves the timeline essentially untouched
    assert abs(kept.edited_duration - tl.edited_duration) < 0.2


def test_no_word_is_clipped():
    tl = _timeline([(0.0, 20.0)])
    words = [
        {"w": "a", "start": 0.5, "end": 1.0, "timing": "aligned"},
        {"w": "b", "start": 6.0, "end": 6.6, "timing": "aligned"},
        {"w": "c", "start": 12.0, "end": 12.7, "timing": "aligned"},
        {"w": "d", "start": 18.5, "end": 19.2, "timing": "aligned"},
    ]
    tr = _transcript(words, 20.0)
    out = refine_timeline_with_transcript(tl, tr, TranscriptTrimParams(enabled=True))

    # Every word's source interval must be fully covered by a kept segment.
    for w in tr.words:
        for (ss, se) in tl.edited_range_to_source_ranges(w.start, w.end):
            assert any(s.source_start - 1e-3 <= ss and se <= s.source_end + 1e-3 for s in out.segments), f"word {w.w} clipped"


def test_short_inter_word_gaps_are_preserved():
    # A short gap between two words (below max_gap_s) must not split the segment.
    # Words span nearly the whole timeline so there's no long tail to trim.
    tl = _timeline([(0.0, 5.0)])
    tr = _transcript(
        [
            {"w": "uno", "start": 0.1, "end": 1.5, "timing": "aligned"},
            {"w": "dos", "start": 1.7, "end": 4.9, "timing": "aligned"},
        ],
        5.0,
    )
    out = refine_timeline_with_transcript(tl, tr, TranscriptTrimParams(enabled=True, max_gap_s=0.35, word_margin_s=0.08))
    # 0.2s inter-word gap < 0.35s -> not split; one kept segment covering both words.
    assert len(out.segments) == 1
    assert abs(out.edited_duration - tl.edited_duration) < 1e-3


def test_retime_transcript_keeps_words_monotonic_and_on_new_axis():
    tl = _timeline([(0.0, 10.0)])
    tr = _transcript(
        [
            {"w": "uno", "start": 1.0, "end": 1.5, "timing": "aligned"},
            {"w": "dos", "start": 7.5, "end": 8.0, "timing": "aligned"},
        ],
        10.0,
    )
    new = refine_timeline_with_transcript(tl, tr, TranscriptTrimParams(enabled=True, max_gap_s=0.35))
    retimed = retime_transcript(tr, tl, new)
    assert retimed.duration == new.edited_duration
    # Each word is well-formed, and words are ordered start-to-start.
    for w in retimed.words:
        assert w.start <= w.end
        assert 0 <= w.start <= new.edited_duration + 1e-3
    starts = [w.start for w in retimed.words]
    assert starts == sorted(starts)  # monotonic across words
    # the second word should now start much earlier than before (gap collapsed)
    assert retimed.words[1].start < tr.words[1].start
