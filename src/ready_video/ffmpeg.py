from __future__ import annotations

import json
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Protocol

from platformdirs import user_cache_dir

from .errors import ReadyVideoError


def _load_manifest() -> dict[str, Any]:
    try:
        text = resources.files(__package__).joinpath("ffmpeg_manifest.json").read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ReadyVideoError(
            "INVALID_FFMPEG_MANIFEST",
            "The FFmpeg runtime manifest is missing.",
            "Reinstall ready-video or restore src/ready_video/ffmpeg_manifest.json.",
        )
    try:
        manifest = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReadyVideoError("INVALID_FFMPEG_MANIFEST", "The FFmpeg runtime manifest is not valid JSON.", details=str(exc)) from exc
    required = {"required_filters", "required_encoders", "optional_encoders", "managed_provider", "cache_subdir", "managed_platforms"}
    missing = required - set(manifest)
    if missing:
        raise ReadyVideoError("INVALID_FFMPEG_MANIFEST", "The FFmpeg runtime manifest is incomplete.", details=f"missing={sorted(missing)}")
    _validate_managed_platforms(manifest)
    return manifest


def _validate_managed_platforms(manifest: dict[str, Any]) -> None:
    platforms = manifest.get("managed_platforms")
    if not isinstance(platforms, dict) or not platforms:
        raise ReadyVideoError("INVALID_FFMPEG_MANIFEST", "The FFmpeg runtime manifest has no managed platform entries.")
    required = {"archive_url", "archive_sha256", "archive_size", "ffmpeg_sha256", "ffprobe_sha256"}
    for platform_key, entry in platforms.items():
        if not isinstance(entry, dict):
            raise ReadyVideoError("INVALID_FFMPEG_MANIFEST", "Managed platform entry must be a mapping.", details=str(platform_key))
        missing = required - set(entry)
        if missing:
            raise ReadyVideoError(
                "INVALID_FFMPEG_MANIFEST",
                "Managed platform entry is incomplete.",
                details=f"platform={platform_key} missing={sorted(missing)}",
            )
        for key in ["archive_sha256", "ffmpeg_sha256", "ffprobe_sha256"]:
            value = entry.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
                raise ReadyVideoError(
                    "INVALID_FFMPEG_MANIFEST",
                    "Managed platform checksum must be a SHA-256 hex digest.",
                    details=f"platform={platform_key} key={key}",
                )


FFMPEG_MANIFEST = _load_manifest()
REQUIRED_FILTERS = frozenset(FFMPEG_MANIFEST["required_filters"])
REQUIRED_ENCODERS = frozenset(FFMPEG_MANIFEST["required_encoders"])
OPTIONAL_ENCODERS = frozenset(FFMPEG_MANIFEST["optional_encoders"])


@dataclass(frozen=True)
class BinaryPair:
    ffmpeg: Path
    ffprobe: Path
    source: str
    provider: str | None = None
    cache_dir: Path | None = None


@dataclass(frozen=True)
class CapabilityReport:
    pair: BinaryPair
    ffmpeg_version: str
    ffprobe_version: str
    release: str
    filters: frozenset[str]
    encoders: frozenset[str]
    required_filters: frozenset[str]
    required_encoders: frozenset[str]
    optional_encoders: frozenset[str]

    @property
    def missing_filters(self) -> frozenset[str]:
        return self.required_filters - self.filters

    @property
    def missing_encoders(self) -> frozenset[str]:
        return self.required_encoders - self.encoders

    @property
    def available_optional_encoders(self) -> frozenset[str]:
        return self.optional_encoders & self.encoders

    @property
    def supported(self) -> bool:
        return not self.missing_filters and not self.missing_encoders

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.pair.source,
            "provider": self.pair.provider,
            "ffmpeg": str(self.pair.ffmpeg),
            "ffprobe": str(self.pair.ffprobe),
            "cache_dir": str(self.pair.cache_dir) if self.pair.cache_dir else None,
            "ffmpeg_version": self.ffmpeg_version,
            "ffprobe_version": self.ffprobe_version,
            "release": self.release,
            "supported": self.supported,
            "missing_filters": sorted(self.missing_filters),
            "missing_encoders": sorted(self.missing_encoders),
            "available_optional_encoders": sorted(self.available_optional_encoders),
        }


class FfmpegProvider(Protocol):
    name: str

    def resolve(self) -> BinaryPair:
        ...


@dataclass(frozen=True)
class StaticFfmpegProvider:
    download_dir: Path | None = None
    name: str = str(FFMPEG_MANIFEST["managed_provider"])

    def cache_dir(self) -> Path:
        if self.download_dir:
            return self.download_dir
        return Path(user_cache_dir("ready-video")) / str(FFMPEG_MANIFEST["cache_subdir"])

    def executable_dir(self) -> Path:
        return self.cache_dir() / self.platform_key()

    def platform_key(self) -> str:
        return _managed_platform_key()

    def resolve(self) -> BinaryPair:
        platform_key = self.platform_key()
        cache_dir = self.cache_dir()
        executable_dir = cache_dir / platform_key
        pair = _managed_pair_at(executable_dir, cache_dir, provider=self.name)
        if _managed_pair_valid(pair, platform_key):
            return pair
        with _InstallLock(cache_dir / f".{platform_key}.install.lock"):
            pair = _managed_pair_at(executable_dir, cache_dir, provider=self.name)
            if not _managed_pair_valid(pair, platform_key):
                _install_managed_pair(cache_dir, platform_key, provider=self.name)
                pair = _managed_pair_at(executable_dir, cache_dir, provider=self.name)
        verify_managed_checksums(pair, platform_key)
        return pair


def _managed_platform_key() -> str:
    machine = platform.machine().lower()
    is_arm64 = machine in {"arm64", "aarch64"}
    if sys.platform == "darwin":
        return "darwin_arm64" if is_arm64 else "darwin"
    if sys.platform.startswith("linux"):
        return "linux_arm64" if is_arm64 else "linux"
    if sys.platform.startswith("win"):
        return "win32"
    raise ReadyVideoError(
        "MISSING_FFMPEG",
        f"No verified managed FFmpeg build is pinned for platform {sys.platform}/{machine}.",
        "Use READY_VIDEO_FFMPEG and READY_VIDEO_FFPROBE to point to a compatible verified pair.",
    )


def _managed_pair_at(executable_dir: Path, cache_dir: Path, *, provider: str) -> BinaryPair:
    ffmpeg_name, ffprobe_name = _managed_executable_names()
    return BinaryPair(executable_dir / ffmpeg_name, executable_dir / ffprobe_name, "managed", provider=provider, cache_dir=cache_dir)


def _managed_executable_names() -> tuple[str, str]:
    if sys.platform.startswith("win"):
        return "ffmpeg.exe", "ffprobe.exe"
    return "ffmpeg", "ffprobe"


def _managed_pair_valid(pair: BinaryPair, platform_key: str) -> bool:
    try:
        verify_managed_checksums(pair, platform_key)
    except (OSError, ReadyVideoError):
        return False
    return True


def _install_managed_pair(cache_dir: Path, platform_key: str, *, provider: str) -> None:
    entry = managed_platform_entry(platform_key)
    archive_url = str(entry["archive_url"])
    archive_size = int(entry["archive_size"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    final_dir = cache_dir / platform_key
    temp_parent = Path(tempfile.mkdtemp(prefix=f".{platform_key}.", dir=cache_dir))
    archive_path = temp_parent / "archive.zip"
    staging_dir = temp_parent / "extracted"
    try:
        _announce_managed_download(entry, cache_dir, platform_key)
        _download_archive(archive_url, archive_path, allowed_hosts=_allowed_archive_hosts(archive_url))
        actual_size = archive_path.stat().st_size
        if actual_size != archive_size:
            raise ReadyVideoError(
                "FFMPEG_CHECKSUM_MISMATCH",
                "Managed FFmpeg archive size did not match the pinned manifest.",
                details=f"platform={platform_key} expected={archive_size} actual={actual_size}",
            )
        actual_archive_hash = file_sha256(archive_path)
        if actual_archive_hash != entry["archive_sha256"]:
            raise ReadyVideoError(
                "FFMPEG_CHECKSUM_MISMATCH",
                "Managed FFmpeg archive checksum did not match the pinned manifest.",
                details=f"platform={platform_key} expected={entry['archive_sha256']} actual={actual_archive_hash}",
            )
        _extract_managed_archive(archive_path, staging_dir)
        pair = _managed_pair_at(staging_dir, cache_dir, provider=provider)
        verify_managed_checksums(pair, platform_key)
        _write_install_manifest(staging_dir, entry, platform_key, provider)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(staging_dir, final_dir)
    except Exception:
        if final_dir.exists() and not _managed_pair_valid(_managed_pair_at(final_dir, cache_dir, provider=provider), platform_key):
            shutil.rmtree(final_dir, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(temp_parent, ignore_errors=True)


def _announce_managed_download(entry: dict[str, Any], cache_dir: Path, platform_key: str) -> None:
    parsed = urllib.parse.urlparse(str(entry["archive_url"]))
    print(
        "ready-video: downloading pinned managed FFmpeg "
        f"platform={platform_key} host={parsed.netloc} size={entry['archive_size']} cache={cache_dir}. "
        "FFmpeg media binaries are separately licensed; see the bundled install manifest.",
        file=sys.stderr,
    )


def _download_archive(url: str, destination: Path, *, allowed_hosts: set[str]) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise ReadyVideoError("FFMPEG_DOWNLOAD_FAILED", "Managed FFmpeg archives must be downloaded over HTTPS.")
    if parsed.hostname not in allowed_hosts:
        raise ReadyVideoError("FFMPEG_DOWNLOAD_FAILED", "Managed FFmpeg archive host is not allowlisted.", details=parsed.hostname)
    opener = urllib.request.build_opener(_SafeRedirectHandler(allowed_hosts))
    request = urllib.request.Request(url, headers={"User-Agent": "ready-video/0.1"})
    try:
        with opener.open(request, timeout=120) as response, destination.open("wb") as output:
            final_host = urllib.parse.urlparse(response.geturl()).hostname
            if final_host not in allowed_hosts:
                raise ReadyVideoError("FFMPEG_DOWNLOAD_FAILED", "Managed FFmpeg redirected to an untrusted host.", details=final_host)
            shutil.copyfileobj(response, output)
    except ReadyVideoError:
        raise
    except Exception as exc:
        raise ReadyVideoError("FFMPEG_DOWNLOAD_FAILED", "Managed FFmpeg archive download failed.", details=str(exc)) from exc


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, allowed_hosts: set[str]) -> None:
        self.allowed_hosts = allowed_hosts

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        host = urllib.parse.urlparse(newurl).hostname
        if host not in self.allowed_hosts:
            raise ReadyVideoError("FFMPEG_DOWNLOAD_FAILED", "Managed FFmpeg redirected to an untrusted host.", details=host)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _allowed_archive_hosts(url: str) -> set[str]:
    host = urllib.parse.urlparse(url).hostname
    # GitHub blob (/raw/) URLs backed by Git-LFS 302-redirect to
    # media.githubusercontent.com; release assets go through
    # objects/github-releases.githubusercontent.com. All are GitHub CDN hosts.
    hosts = {
        "github.com",
        "raw.githubusercontent.com",
        "media.githubusercontent.com",
        "objects.githubusercontent.com",
        "github-releases.githubusercontent.com",
    }
    if host:
        hosts.add(host)
    return hosts


def _extract_managed_archive(archive_path: Path, staging_dir: Path) -> None:
    staging_dir.mkdir(parents=True, exist_ok=False)
    ffmpeg_name, ffprobe_name = _managed_executable_names()
    wanted = {ffmpeg_name, ffprobe_name}
    found: dict[str, zipfile.ZipInfo] = {}
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    _validate_zip_member(info)
                    continue
                _validate_zip_member(info)
                basename = Path(info.filename).name
                if basename not in wanted:
                    raise ReadyVideoError("FFMPEG_DOWNLOAD_FAILED", "Managed FFmpeg archive contained an unexpected file.", details=info.filename)
                if basename in found:
                    raise ReadyVideoError("FFMPEG_DOWNLOAD_FAILED", "Managed FFmpeg archive contained duplicate executables.", details=basename)
                found[basename] = info
            missing = wanted - set(found)
            if missing:
                raise ReadyVideoError("FFMPEG_DOWNLOAD_FAILED", "Managed FFmpeg archive did not contain both executables.", details=str(sorted(missing)))
            for name, info in found.items():
                destination = staging_dir / name
                with archive.open(info) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
                if not sys.platform.startswith("win"):
                    destination.chmod(0o755)
    except zipfile.BadZipFile as exc:
        raise ReadyVideoError("FFMPEG_DOWNLOAD_FAILED", "Managed FFmpeg archive was not a valid zip file.", details=str(exc)) from exc


def _validate_zip_member(info: zipfile.ZipInfo) -> None:
    path = Path(info.filename)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ReadyVideoError("FFMPEG_DOWNLOAD_FAILED", "Managed FFmpeg archive contained an unsafe path.", details=info.filename)
    mode = info.external_attr >> 16
    if mode and (mode & 0o170000) == 0o120000:
        raise ReadyVideoError("FFMPEG_DOWNLOAD_FAILED", "Managed FFmpeg archive contained a link.", details=info.filename)


def _write_install_manifest(staging_dir: Path, entry: dict[str, Any], platform_key: str, provider: str) -> None:
    payload = {
        "provider": provider,
        "platform": platform_key,
        "archive_url": entry["archive_url"],
        "archive_sha256": entry["archive_sha256"],
        "archive_size": entry["archive_size"],
        "ffmpeg_sha256": entry["ffmpeg_sha256"],
        "ffprobe_sha256": entry["ffprobe_sha256"],
        "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (staging_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class _InstallLock:
    def __init__(self, path: Path, *, timeout_s: float = 300.0) -> None:
        self.path = path
        self.timeout_s = timeout_s

    def __enter__(self) -> "_InstallLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                self.path.mkdir()
                return self
            except FileExistsError:
                if time.monotonic() > deadline:
                    raise ReadyVideoError("FFMPEG_DOWNLOAD_FAILED", "Timed out waiting for managed FFmpeg installation lock.")
                time.sleep(0.1)

    def __exit__(self, exc_type, exc, traceback) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def managed_platform_entry(platform_key: str) -> dict[str, Any]:
    platforms = FFMPEG_MANIFEST.get("managed_platforms") or {}
    entry = platforms.get(platform_key)
    if not isinstance(entry, dict):
        raise ReadyVideoError(
            "MISSING_FFMPEG",
            f"No verified managed FFmpeg build is pinned for platform {platform_key}.",
            "Use READY_VIDEO_FFMPEG and READY_VIDEO_FFPROBE to point to a compatible verified pair.",
        )
    return entry


def verify_managed_checksums(pair: BinaryPair, platform_key: str) -> None:
    entry = managed_platform_entry(platform_key)
    expected = {
        pair.ffmpeg: entry.get("ffmpeg_sha256"),
        pair.ffprobe: entry.get("ffprobe_sha256"),
    }
    missing = [path.name for path, value in expected.items() if not value]
    if missing:
        raise ReadyVideoError(
            "INVALID_FFMPEG_MANIFEST",
            "The managed FFmpeg manifest is missing executable checksums.",
            details=f"platform={platform_key} missing={missing}",
        )
    for path, expected_hash in expected.items():
        if not path.exists():
            raise ReadyVideoError("MISSING_FFMPEG", f"Managed executable is missing: {path}")
        actual = file_sha256(path)
        if actual != expected_hash:
            raise ReadyVideoError(
                "FFMPEG_CHECKSUM_MISMATCH",
                f"Managed executable checksum mismatch for {path.name}.",
                "Delete the managed FFmpeg cache and retry, or use explicit READY_VIDEO_FFMPEG/READY_VIDEO_FFPROBE overrides.",
                details=f"platform={platform_key} expected={expected_hash} actual={actual}",
            )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: list[str | Path], *, timeout: int | None = None, capture: bool = True) -> subprocess.CompletedProcess[str]:
    text_args = [str(arg) for arg in args]
    return subprocess.run(
        text_args,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        timeout=timeout,
    )


def _version(executable: Path) -> str:
    completed = run([executable, "-version"], timeout=10)
    if completed.returncode != 0:
        raise ReadyVideoError("MISSING_FFMPEG", f"Could not run {executable}.", details=completed.stderr)
    return completed.stdout.splitlines()[0]


def resolve_binaries(*, install_missing: bool = True, managed_provider: FfmpegProvider | None = None) -> BinaryPair:
    pair = _environment_pair()
    if pair:
        validate_pair(pair, explicit=True)
        return pair
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        pair = BinaryPair(Path(ffmpeg), Path(ffprobe), "system")
        try:
            validate_pair(pair, explicit=False)
            return pair
        except ReadyVideoError:
            pass
    if install_missing:
        pair = (managed_provider or StaticFfmpegProvider()).resolve()
        validate_pair(pair, explicit=True)
        return pair
    raise ReadyVideoError(
        "MISSING_FFMPEG",
        "No compatible ffmpeg/ffprobe pair was found.",
        "Install FFmpeg, set READY_VIDEO_FFMPEG and READY_VIDEO_FFPROBE, or run doctor --install-missing to fetch managed binaries.",
    )


def _environment_pair() -> BinaryPair | None:
    values = {
        "READY_VIDEO_FFMPEG": os.environ.get("READY_VIDEO_FFMPEG"),
        "READY_VIDEO_FFPROBE": os.environ.get("READY_VIDEO_FFPROBE"),
    }
    provided = {key for key, value in values.items() if value is not None}
    if not provided:
        return None
    missing = {key for key, value in values.items() if value is None or not value.strip()}
    if missing:
        raise ReadyVideoError(
            "MISSING_FFMPEG",
            "READY_VIDEO_FFMPEG and READY_VIDEO_FFPROBE must be set together.",
            details=f"missing={sorted(missing)}",
        )
    ffmpeg = Path(values["READY_VIDEO_FFMPEG"] or "").expanduser()
    ffprobe = Path(values["READY_VIDEO_FFPROBE"] or "").expanduser()
    if ffmpeg == ffprobe:
        raise ReadyVideoError("MISSING_FFMPEG", "READY_VIDEO_FFMPEG and READY_VIDEO_FFPROBE must point to different executables.")
    return BinaryPair(ffmpeg, ffprobe, "environment")


def validate_pair(pair: BinaryPair, *, explicit: bool) -> CapabilityReport:
    for executable in [pair.ffmpeg, pair.ffprobe]:
        if not executable.exists():
            raise ReadyVideoError("MISSING_FFMPEG", f"Executable does not exist: {executable}")
    report = capability_report(pair)
    ffprobe_release = _release_token(report.ffprobe_version)
    if report.release != ffprobe_release:
        raise ReadyVideoError(
            "FFMPEG_VERSION_MISMATCH",
            "ffmpeg and ffprobe are not from the same release.",
            details=f"{report.ffmpeg_version} / {report.ffprobe_version}",
        )
    missing_filters = report.missing_filters
    missing_encoders = report.missing_encoders
    if missing_filters or missing_encoders:
        message = "FFmpeg is missing required capabilities."
        if not explicit:
            raise ReadyVideoError("UNSUPPORTED_FFMPEG_BUILD", message)
        raise ReadyVideoError(
            "UNSUPPORTED_FFMPEG_BUILD",
            message,
            details=f"missing filters={sorted(missing_filters)} missing encoders={sorted(missing_encoders)}",
        )
    return report


def capability_report(pair: BinaryPair) -> CapabilityReport:
    ffmpeg_version = _version(pair.ffmpeg)
    ffprobe_version = _version(pair.ffprobe)
    return CapabilityReport(
        pair=pair,
        ffmpeg_version=ffmpeg_version,
        ffprobe_version=ffprobe_version,
        release=_release_token(ffmpeg_version),
        filters=frozenset(available_filters(pair.ffmpeg)),
        encoders=frozenset(available_encoders(pair.ffmpeg)),
        required_filters=REQUIRED_FILTERS,
        required_encoders=REQUIRED_ENCODERS,
        optional_encoders=OPTIONAL_ENCODERS,
    )


def _release_token(version_line: str) -> str:
    parts = version_line.split()
    return parts[2] if len(parts) > 2 and parts[0] in {"ffmpeg", "ffprobe"} else version_line


def available_filters(ffmpeg: Path) -> set[str]:
    completed = run([ffmpeg, "-hide_banner", "-filters"], timeout=15)
    if completed.returncode != 0:
        raise ReadyVideoError("UNSUPPORTED_FFMPEG_BUILD", "Could not list FFmpeg filters.", details=completed.stderr)
    filters: set[str] = set()
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0][0] in "T.SA|":
            filters.add(parts[1])
    return filters


def available_encoders(ffmpeg: Path) -> set[str]:
    completed = run([ffmpeg, "-hide_banner", "-encoders"], timeout=15)
    if completed.returncode != 0:
        raise ReadyVideoError("UNSUPPORTED_FFMPEG_BUILD", "Could not list FFmpeg encoders.", details=completed.stderr)
    encoders: set[str] = set()
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            encoders.add(parts[1])
    return encoders


def ffprobe_json(pair: BinaryPair, args: list[str | Path]) -> dict:
    completed = run([pair.ffprobe, "-v", "error", "-of", "json", *args], timeout=60)
    if completed.returncode != 0:
        raise ReadyVideoError("INVALID_MEDIA", "ffprobe could not read the input.", details=completed.stderr)
    return json.loads(completed.stdout or "{}")
