from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import generate_default_config_yaml, user_config_path
from .errors import ReadyVideoError
from .pipeline import approve, doctor, inbox, run_file


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        overrides = _overrides(args)
        if args.command == "run":
            output = run_file(Path(args.file), config_path=args.config, review=args.review, cli_overrides=overrides)
            print(output)
            return 0
        if args.command == "inbox":
            return inbox(config_path=args.config, cli_overrides=overrides)
        if args.command == "approve":
            print(approve(args.job_id, config_path=args.config))
            return 0
        if args.command == "init":
            target = user_config_path() if args.user else Path("config.yaml")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(generate_default_config_yaml())
            print(target)
            return 0
        if args.command == "doctor":
            print("doctor is available")
            return doctor(install_missing=args.install_missing)
        parser.print_help()
        return 2
    except ReadyVideoError as exc:
        print(str(exc), file=sys.stderr)
        if exc.details:
            print(exc.details, file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ready-video")
    _add_common(parser)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("file")
    run.add_argument("--review", action="store_true")
    _add_common(run, suppress_defaults=True)
    inbox_parser = sub.add_parser("inbox")
    _add_common(inbox_parser, suppress_defaults=True)
    approve_parser = sub.add_parser("approve")
    approve_parser.add_argument("job_id")
    init = sub.add_parser("init")
    init.add_argument("--user", action="store_true")
    doctor_parser = sub.add_parser("doctor")
    doctor_parser.add_argument("--install-missing", action="store_true")
    return parser


def _add_common(parser: argparse.ArgumentParser, *, suppress_defaults: bool = False) -> None:
    default = argparse.SUPPRESS if suppress_defaults else None
    parser.add_argument("--config", type=Path, default=default)
    parser.add_argument("--preset", default=default)
    parser.add_argument("--language", default=default)
    parser.add_argument("--aspect", default=default)


def _overrides(args: argparse.Namespace) -> dict:
    data: dict = {}
    if getattr(args, "preset", None):
        data.setdefault("subtitles", {})["preset"] = args.preset
    if getattr(args, "language", None):
        data.setdefault("transcription", {})["language"] = args.language
    if getattr(args, "aspect", None):
        data.setdefault("render", {})["aspect"] = args.aspect
    return data


if __name__ == "__main__":
    raise SystemExit(main())
