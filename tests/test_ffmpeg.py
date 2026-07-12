from __future__ import annotations

import hashlib
import shutil
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from ready_video import ffmpeg
from ready_video.errors import ReadyVideoError
from ready_video.ffmpeg import BinaryPair, StaticFfmpegProvider, capability_report, resolve_binaries, validate_pair, verify_managed_checksums


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _fake_executable(
    path: Path,
    *,
    program: str,
    release: str = "6.1-ready",
    filters: set[str] | None = None,
    encoders: set[str] | None = None,
) -> Path:
    filter_lines = "\n".join(f" ... {name} description" for name in sorted(filters or set()))
    encoder_lines = "\n".join(f" V..... {name} description" for name in sorted(encoders or set()))
    _write_executable(
        path,
        f"""#!/usr/bin/env python3
import sys

args = sys.argv[1:]
if "-version" in args:
    print({f"{program} version {release} test-build"!r})
    raise SystemExit(0)
if "-filters" in args:
    print({filter_lines!r})
    raise SystemExit(0)
if "-encoders" in args:
    print({encoder_lines!r})
    raise SystemExit(0)
raise SystemExit(2)
""",
    )
    return path


def _fake_pair(tmp_path: Path, *, ffmpeg_release: str = "6.1-ready", ffprobe_release: str = "6.1-ready") -> BinaryPair:
    ffmpeg_path = _fake_executable(
        tmp_path / "ffmpeg",
        program="ffmpeg",
        release=ffmpeg_release,
        filters=set(ffmpeg.REQUIRED_FILTERS),
        encoders=set(ffmpeg.REQUIRED_ENCODERS) | {"h264_videotoolbox"},
    )
    ffprobe_path = _fake_executable(tmp_path / "ffprobe", program="ffprobe", release=ffprobe_release)
    return BinaryPair(ffmpeg_path, ffprobe_path, "test")


def test_environment_pair_must_be_complete(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("READY_VIDEO_FFMPEG", str(tmp_path / "ffmpeg"))
    monkeypatch.delenv("READY_VIDEO_FFPROBE", raising=False)

    with pytest.raises(ReadyVideoError) as error:
        resolve_binaries(install_missing=False)

    assert error.value.code == "MISSING_FFMPEG"
    assert "READY_VIDEO_FFPROBE" in (error.value.details or "")


def test_manifest_pins_supported_managed_platforms() -> None:
    platforms = ffmpeg.FFMPEG_MANIFEST["managed_platforms"]

    assert {"darwin", "darwin_arm64", "linux", "linux_arm64"} <= set(platforms)
    for platform_key, entry in platforms.items():
        assert entry["archive_url"].startswith("https://github.com/zackees/ffmpeg_bins/")
        assert entry["archive_size"] > 0
        for key in ["archive_sha256", "ffmpeg_sha256", "ffprobe_sha256"]:
            assert len(entry[key]) == 64


def test_validate_pair_returns_capability_report(tmp_path: Path) -> None:
    pair = _fake_pair(tmp_path)

    report = validate_pair(pair, explicit=True)

    assert report.supported is True
    assert report.release == "6.1-ready"
    assert report.missing_filters == frozenset()
    assert report.missing_encoders == frozenset()
    assert report.available_optional_encoders == frozenset({"h264_videotoolbox"})
    assert report.as_dict()["supported"] is True


def test_capability_report_keeps_missing_requirements_visible(tmp_path: Path) -> None:
    pair = BinaryPair(
        _fake_executable(
            tmp_path / "ffmpeg",
            program="ffmpeg",
            filters=set(ffmpeg.REQUIRED_FILTERS) - {"ass"},
            encoders=set(),
        ),
        _fake_executable(tmp_path / "ffprobe", program="ffprobe"),
        "test",
    )

    report = capability_report(pair)

    assert report.supported is False
    assert report.missing_filters == frozenset({"ass"})
    assert report.missing_encoders == ffmpeg.REQUIRED_ENCODERS


def test_mismatched_binary_releases_are_rejected(tmp_path: Path) -> None:
    pair = _fake_pair(tmp_path, ffmpeg_release="6.1-ready", ffprobe_release="5.1-ready")

    with pytest.raises(ReadyVideoError) as error:
        validate_pair(pair, explicit=True)

    assert error.value.code == "FFMPEG_VERSION_MISMATCH"


@dataclass
class StubProvider:
    pair: BinaryPair
    name: str = "stub-managed"

    def resolve(self) -> BinaryPair:
        return self.pair


def test_resolve_binaries_accepts_managed_provider_without_network(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("READY_VIDEO_FFMPEG", raising=False)
    monkeypatch.delenv("READY_VIDEO_FFPROBE", raising=False)
    monkeypatch.setattr(ffmpeg.shutil, "which", lambda _: None)
    managed_pair = _fake_pair(tmp_path)

    pair = resolve_binaries(install_missing=True, managed_provider=StubProvider(managed_pair))

    assert pair == managed_pair


def test_static_provider_uses_user_cache_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ffmpeg, "user_cache_dir", lambda appname: str(tmp_path / appname))

    provider = StaticFfmpegProvider()

    assert provider.cache_dir() == tmp_path / "ready-video" / "ffmpeg"


def test_verify_managed_checksums_accepts_pinned_pair(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ffmpeg_path = tmp_path / "ffmpeg"
    ffprobe_path = tmp_path / "ffprobe"
    ffmpeg_path.write_bytes(b"ffmpeg")
    ffprobe_path.write_bytes(b"ffprobe")
    monkeypatch.setitem(
        ffmpeg.FFMPEG_MANIFEST,
        "managed_platforms",
        {
            "test-platform": {
                "ffmpeg_sha256": ffmpeg.file_sha256(ffmpeg_path),
                "ffprobe_sha256": ffmpeg.file_sha256(ffprobe_path),
            }
        },
    )

    verify_managed_checksums(BinaryPair(ffmpeg_path, ffprobe_path, "managed"), "test-platform")


def test_verify_managed_checksums_rejects_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ffmpeg_path = tmp_path / "ffmpeg"
    ffprobe_path = tmp_path / "ffprobe"
    ffmpeg_path.write_bytes(b"ffmpeg")
    ffprobe_path.write_bytes(b"ffprobe")
    monkeypatch.setitem(
        ffmpeg.FFMPEG_MANIFEST,
        "managed_platforms",
        {
            "test-platform": {
                "ffmpeg_sha256": "0" * 64,
                "ffprobe_sha256": ffmpeg.file_sha256(ffprobe_path),
            }
        },
    )

    with pytest.raises(ReadyVideoError) as error:
        verify_managed_checksums(BinaryPair(ffmpeg_path, ffprobe_path, "managed"), "test-platform")

    assert error.value.code == "FFMPEG_CHECKSUM_MISMATCH"


def test_static_provider_installs_pinned_archive_defensively(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    archive = _zip_archive(tmp_path / "ffmpeg.zip", {"ffmpeg": b"ffmpeg-bytes", "ffprobe": b"ffprobe-bytes"})
    monkeypatch.setitem(
        ffmpeg.FFMPEG_MANIFEST,
        "managed_platforms",
        {
            "test-platform": {
                "archive_url": "https://github.com/example/ffmpeg.zip",
                "archive_size": archive.stat().st_size,
                "archive_sha256": _sha256(archive.read_bytes()),
                "ffmpeg_sha256": _sha256(b"ffmpeg-bytes"),
                "ffprobe_sha256": _sha256(b"ffprobe-bytes"),
            }
        },
    )
    monkeypatch.setattr(StaticFfmpegProvider, "platform_key", lambda self: "test-platform")
    monkeypatch.setattr(ffmpeg, "_download_archive", lambda url, destination, *, allowed_hosts: shutil.copyfile(archive, destination))

    pair = StaticFfmpegProvider(download_dir=tmp_path / "cache").resolve()

    assert pair.ffmpeg.read_bytes() == b"ffmpeg-bytes"
    assert pair.ffprobe.read_bytes() == b"ffprobe-bytes"
    assert (tmp_path / "cache" / "test-platform" / "manifest.json").exists()


def test_static_provider_rejects_unsafe_archive_without_final_executable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    archive = _zip_archive(tmp_path / "unsafe.zip", {"../ffmpeg": b"bad", "ffprobe": b"ffprobe-bytes"})
    monkeypatch.setitem(
        ffmpeg.FFMPEG_MANIFEST,
        "managed_platforms",
        {
            "test-platform": {
                "archive_url": "https://github.com/example/ffmpeg.zip",
                "archive_size": archive.stat().st_size,
                "archive_sha256": _sha256(archive.read_bytes()),
                "ffmpeg_sha256": _sha256(b"bad"),
                "ffprobe_sha256": _sha256(b"ffprobe-bytes"),
            }
        },
    )
    monkeypatch.setattr(StaticFfmpegProvider, "platform_key", lambda self: "test-platform")
    monkeypatch.setattr(ffmpeg, "_download_archive", lambda url, destination, *, allowed_hosts: shutil.copyfile(archive, destination))

    with pytest.raises(ReadyVideoError) as error:
        StaticFfmpegProvider(download_dir=tmp_path / "cache").resolve()

    assert error.value.code == "FFMPEG_DOWNLOAD_FAILED"
    assert not (tmp_path / "cache" / "test-platform" / "ffmpeg").exists()


def test_safe_redirect_handler_rejects_untrusted_hosts() -> None:
    handler = ffmpeg._SafeRedirectHandler({"github.com"})

    with pytest.raises(ReadyVideoError) as error:
        handler.redirect_request(None, None, 302, "Found", {}, "https://evil.example/ffmpeg.zip")

    assert error.value.code == "FFMPEG_DOWNLOAD_FAILED"


def _zip_archive(path: Path, files: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return path


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def test_allowed_archive_hosts_include_github_lfs_media_redirect():
    # GitHub /raw/ (Git-LFS) archive URLs 302-redirect to media.githubusercontent.com.
    # Regression: that host must stay allowlisted or the managed download fails.
    hosts = ffmpeg._allowed_archive_hosts("https://github.com/org/repo/raw/main/darwin_arm64.zip")
    assert "media.githubusercontent.com" in hosts
    assert "github.com" in hosts
    assert "objects.githubusercontent.com" in hosts
