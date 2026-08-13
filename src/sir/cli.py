"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sir import logging as sir_logging
from sir.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sir",
        description="Stupid Inference Router — one endpoint, one scheduler behind it.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the router")
    serve.add_argument("--config", "-c", default="config.yaml", type=Path)
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--log-level", default=None)

    validate = sub.add_parser("validate", help="check a config file and exit")
    validate.add_argument("--config", "-c", default="config.yaml", type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = load_config(args.config)
    except Exception as exc:  # config errors are for humans, not tracebacks
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.command == "validate":
        print(f"config ok: {len(config.models)} model(s): {', '.join(config.model_names)}")
        return 0

    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port
    if args.log_level:
        config.server.log_level = args.log_level

    sir_logging.configure(config.server.log_level)

    import uvicorn  # imported late so `sir validate` doesn't pay for it

    from sir.api import create_app

    uvicorn.run(
        create_app(config),
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
