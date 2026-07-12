"""Transcript-driven second-pass silence trim (hybrid).

The first pass (``media.analyze_silence``) removes silence by sound: FFmpeg
``silencedetect`` on the audio. Quiet-but-present pauses (room tone, breath)
that stay above the dB threshold survive as long gaps *between spoken words*.

This module runs a second pass over the already-cut timeline using the
transcript's word timings to compress those inter-word gaps. It composes with
the first pass rather than replacing it:

- The sound pass is the safety floor — it never cut anything audible.
- This pass only *shortens* gaps between words; it never extends a kept range,
  so it can never re-introduce removed audio.
- A gap is compressed only when BOTH bounding words are ``aligned``. Near an
  ``interpolated`` (uncertain) word the gap is left intact, so imperfect
  alignment can never clip real speech.

The transcript is in first-pass EDITED time. Each tightened edited range is
mapped back to SOURCE via the first-pass timeline, then a fresh timeline is
emitted (source ranges + a new contiguous edited axis). Callers re-time the
transcript, subtitles, and zooms onto this new axis with
``remap_edited_time``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .timeline import Timeline, TimelineSegment
from .transcript import Transcript

EPS = 1e-6


@dataclass(frozen=True)
class TranscriptTrimParams:
    enabled: bool = True
    max_gap_s: float = 0.35
    word_margin_s: float = 0.08


def refine_timeline_with_transcript(
    timeline: Timeline,
    transcript: Transcript,
    params: TranscriptTrimParams,
) -> Timeline:
    """Return a tightened timeline; the input is returned unchanged when disabled.

    ``transcript`` word timings are in ``timeline``'s edited time.
    """
    if not params.enabled or not transcript.words:
        return timeline

    edited_dur = timeline.edited_duration
    words = sorted(transcript.words, key=lambda w: (w.start, w.end))

    # Build KEEP ranges in edited time: a padded window around every word, plus
    # each inter-word gap capped at max_gap_s. A gap is only capped when both
    # bounding words are aligned; otherwise the full gap is kept.
    keeps: list[tuple[float, float]] = []
    prev_end = 0.0
    prev_aligned = False  # the "word" before the first word is a boundary, not aligned
    for word in words:
        ws = max(0.0, word.start - params.word_margin_s)
        we = min(edited_dur, word.end + params.word_margin_s)
        aligned = word.timing == "aligned"
        gap = ws - prev_end
        if gap > EPS:
            if gap > params.max_gap_s and prev_aligned and aligned:
                # Trustworthy gap: keep only max_gap_s of dead air, split so the
                # tail after the previous word and the lead-in to this word each
                # keep half — feels like a natural beat, not a hard cut.
                half = params.max_gap_s / 2.0
                keeps.append((prev_end, min(prev_end + half, ws)))
                keeps.append((max(ws - half, prev_end + half), ws))
            else:
                # Untrusted (interpolated neighbour) or already short: keep it all.
                keeps.append((prev_end, ws))
        keeps.append((ws, we))
        prev_end = we
        prev_aligned = aligned

    # Trailing tail after the last word.
    if edited_dur - prev_end > EPS:
        if edited_dur - prev_end > params.max_gap_s and prev_aligned:
            keeps.append((prev_end, prev_end + params.max_gap_s))
        else:
            keeps.append((prev_end, edited_dur))

    merged_edited = _merge(keeps)

    # Map each tightened edited keep-range back to source via the first pass.
    source_ranges: list[tuple[float, float]] = []
    for start, end in merged_edited:
        source_ranges.extend(timeline.edited_range_to_source_ranges(start, end))
    merged_source = _merge(source_ranges)

    if not merged_source:
        # Degenerate (e.g. every word untimed): keep the original timeline.
        return timeline

    segments: list[TimelineSegment] = []
    edited = 0.0
    for i, (ss, se) in enumerate(merged_source):
        duration = se - ss
        segments.append(
            TimelineSegment(
                i=i,
                source_start=round(ss, 6),
                source_end=round(se, 6),
                edited_start=round(edited, 6),
                edited_end=round(edited + duration, 6),
            )
        )
        edited += duration
    return Timeline(source_duration=timeline.source_duration, edited_duration=round(edited, 6), segments=segments)


def remap_edited_time(old: Timeline, new: Timeline, edited_time: float) -> float | None:
    """Map an edited timestamp on ``old`` onto ``new`` (via shared source time).

    Returns ``None`` when the point maps to source time that ``new`` dropped
    (i.e. it fell inside a gap the second pass removed).
    """
    source = old.edited_to_source(edited_time)
    return new.source_to_edited(source)


def retime_transcript(transcript: Transcript, old: Timeline, new: Timeline) -> Transcript:
    """Return a copy of ``transcript`` with word/segment times on ``new``'s axis.

    Word start/end are re-derived per word so a word never straddles a trimmed
    gap; only the surrounding gaps shrink. If a boundary lands in a dropped
    region it snaps to the nearest kept edge, keeping timestamps monotonic.
    """
    data = transcript.model_dump()
    for word in data["words"]:
        word["start"] = _snap(old, new, word["start"], prefer="start")
        word["end"] = _snap(old, new, word["end"], prefer="end")
        if word["end"] < word["start"]:
            word["end"] = word["start"]
    for segment in data.get("segments", []):
        segment["start"] = _snap(old, new, segment["start"], prefer="start")
        segment["end"] = _snap(old, new, segment["end"], prefer="end")
        if segment["end"] < segment["start"]:
            segment["end"] = segment["start"]
    data["duration"] = new.edited_duration
    return Transcript.model_validate(data)


def retime_zoom_starts_ends(zooms, transcript: Transcript):
    """Re-derive each zoom's edited start/end from its word indices.

    Word indices are stable across the trim (words are never dropped), so the
    zoom span is rebuilt from the re-timed transcript. Returns the same object
    with start/end updated in place.
    """
    by_index = {w.i: w for w in transcript.words}
    for zoom in zooms.zooms:
        start_word = by_index.get(zoom.start_word_index)
        end_word = by_index.get(zoom.end_word_index)
        if start_word is not None:
            zoom.start = start_word.start
        if end_word is not None:
            zoom.end = end_word.end
        if zoom.end < zoom.start:
            zoom.end = zoom.start
    return zooms


def _snap(old: Timeline, new: Timeline, edited_time: float, *, prefer: str) -> float:
    """Map an old-edited time to new-edited, snapping to the nearest kept edge
    if it fell inside a trimmed gap."""
    source = old.edited_to_source(edited_time)
    mapped = new.source_to_edited(source)
    if mapped is not None:
        return round(mapped, 6)
    # Source time was trimmed away: snap to the nearest kept segment boundary.
    best = 0.0
    best_dist = float("inf")
    for segment in new.segments:
        for src_edge, ed_edge in ((segment.source_start, segment.edited_start), (segment.source_end, segment.edited_end)):
            dist = abs(src_edge - source)
            if dist < best_dist:
                best_dist = dist
                best = ed_edge
    return round(best, 6)


def _merge(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(ranges):
        if end <= start + EPS:
            continue
        if merged and start <= merged[-1][1] + EPS:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
