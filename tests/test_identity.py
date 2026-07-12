from pathlib import Path

from ready_video.identity import build_video_identity, file_fingerprint, short_hash, slugify


def test_slugify_normalizes_source_names():
    assert slugify(" My Video_Final!.MP4 ") == "my-video-final-mp4"
    assert slugify("!!!") == "video"


def test_short_hash_is_stable_and_sized():
    assert short_hash("hello", length=8) == short_hash("hello", length=8)
    assert len(short_hash("hello", length=8)) == 8


def test_build_video_identity_uses_slug_and_fingerprint(tmp_path: Path):
    source = tmp_path / "Launch Clip.mov"
    source.write_bytes(b"video bytes")

    fingerprint = file_fingerprint(source)
    identity = build_video_identity(source)

    assert identity.source_name == "Launch Clip.mov"
    assert identity.slug == "launch-clip"
    assert identity.fingerprint == fingerprint
    assert identity.job_id == f"launch-clip-{fingerprint[:12]}"

