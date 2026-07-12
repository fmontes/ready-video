"""Stable naming, hashing, and manifest helpers."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

from . import PIPELINE_VERSION

PathLike = Union[str, Path]

_SLUG_SEPARATOR_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class VideoIdentity:
    source_path: Path
    source_name: str
    slug: str
    fingerprint: str
    job_id: str


@dataclass(frozen=True)
class OutputManifest:
    job_id: str
    job_sha256: str
    content_sha256: str
    config_sha256: str
    pipeline_version: str
    input_name: str
    output_name: str
    created_at: str

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return asdict(self)


def slugify(value: str, fallback: str = "video") -> str:
    slug = _SLUG_SEPARATOR_RE.sub("-", value.lower()).strip("-")
    return slug or fallback


def sanitized_stem(path: Path) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem).strip("-._")
    return value or "video"


def short_hash(value: Union[str, bytes], length: int = 12) -> str:
    if length <= 0:
        raise ValueError("length must be positive")
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()[:length]


def file_fingerprint(path: PathLike, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    hasher = hashlib.sha256()
    source = Path(path)
    with source.open("rb") as handle:
        _update_hash_from_file(hasher, handle, chunk_size)
    return hasher.hexdigest()


def file_sha256(path: Path) -> str:
    return file_fingerprint(path)


def canonical_json_sha256(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def config_sha256(config: Any) -> str:
    if hasattr(config, "model_dump"):
        data = config.model_dump(mode="json")
    elif hasattr(config, "model_dump_json"):
        data = json.loads(config.model_dump_json())
    else:
        data = config
    return canonical_json_sha256(data)


def job_sha256(content_hash: str, config_hash: str, pipeline_version: str = PIPELINE_VERSION) -> str:
    return hashlib.sha256(f"{content_hash}{config_hash}{pipeline_version}".encode()).hexdigest()


def job_id(job_hash: str) -> str:
    return job_hash[:12]


def make_manifest(
    *,
    input_path: Path,
    output_name: str,
    content_hash: str,
    config_hash: str,
    job_hash: str,
) -> OutputManifest:
    return OutputManifest(
        job_id=job_id(job_hash),
        job_sha256=job_hash,
        content_sha256=content_hash,
        config_sha256=config_hash,
        pipeline_version=PIPELINE_VERSION,
        input_name=input_path.name,
        output_name=output_name,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def build_video_identity(path: PathLike, fingerprint: Optional[str] = None) -> VideoIdentity:
    source = Path(path)
    digest = fingerprint or file_fingerprint(source)
    slug = slugify(source.stem)
    return VideoIdentity(
        source_path=source,
        source_name=source.name,
        slug=slug,
        fingerprint=digest,
        job_id=f"{slug}-{digest[:12]}",
    )


def _update_hash_from_file(hasher: "hashlib._Hash", handle: BinaryIO, chunk_size: int) -> None:
    while True:
        chunk = handle.read(chunk_size)
        if not chunk:
            return
        hasher.update(chunk)
