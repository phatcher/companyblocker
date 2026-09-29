from __future__ import annotations

import argparse
import shutil
import subprocess  # nosec B404 - dev-tooling script; see nosec B603/B607 at its call sites
import webbrowser
from pathlib import Path

from _tooling_common import repository_root

REPO_ROOT = repository_root(Path(__file__))
COMPOSE_FILE = (
    REPO_ROOT / "docs" / "structurizr" / ".structurizr" / "docker-compose.yaml"
)
DEFAULT_URL = "http://localhost:8090"


def _compose_base_cmd() -> list[str]:
    docker_path = shutil.which("docker")
    if docker_path is not None:
        # `docker` is resolved via PATH (developer tooling, not attacker-controlled),
        # and the argv is a fixed literal, not shell=True.
        probe = subprocess.run(  # nosec B603 B607
            ["docker", "compose", "version"],
            check=False,
            text=True,
            capture_output=True,
        )
        if probe.returncode == 0:
            return ["docker", "compose"]

    docker_compose_path = shutil.which("docker-compose")
    if docker_compose_path is not None:
        return ["docker-compose"]

    raise FileNotFoundError(
        "Neither 'docker compose' nor 'docker-compose' is available on PATH. "
        "Install Docker Desktop (Windows/macOS) or Docker Engine + Compose plugin (Linux)."
    )


def _run_compose(
    args: list[str], *, capture_output: bool = False
) -> subprocess.CompletedProcess[str]:
    cmd = [*_compose_base_cmd(), "-f", str(COMPOSE_FILE), *args]
    # `cmd` is built from `_compose_base_cmd()`'s fixed docker/docker-compose literal
    # plus this module's own args; not shell=True, no untrusted input.
    return subprocess.run(  # nosec B603
        cmd,
        check=False,
        text=True,
        capture_output=capture_output,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local Structurizr Lite workspace with Docker Compose.",
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="start",
        choices=["start", "stop", "restart", "status", "logs", "config"],
        help="Compose action. Defaults to 'start'.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Structurizr URL to open after start/restart. Defaults to {DEFAULT_URL}.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the Structurizr URL in a browser.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not COMPOSE_FILE.exists():
        print(f"[error] Compose file not found: {COMPOSE_FILE}")
        return 1

    try:
        if args.action == "start":
            result = _run_compose(["up", "-d"])
            if result.returncode == 0:
                print("[info] Structurizr started")
                print(f"[info] URL: {args.url}")
                if not args.no_open:
                    webbrowser.open(args.url)
            return result.returncode

        if args.action == "stop":
            return _run_compose(["down"]).returncode

        if args.action == "restart":
            down_result = _run_compose(["down"])
            if down_result.returncode != 0:
                return down_result.returncode
            up_result = _run_compose(["up", "-d"])
            if up_result.returncode == 0:
                print("[info] Structurizr restarted")
                print(f"[info] URL: {args.url}")
                if not args.no_open:
                    webbrowser.open(args.url)
            return up_result.returncode

        if args.action == "status":
            return _run_compose(["ps"]).returncode

        if args.action == "logs":
            return _run_compose(["logs", "--tail", "100", "architecture"]).returncode

        if args.action == "config":
            return _run_compose(["config"]).returncode

        parser.print_help()
        return 2
    except FileNotFoundError as exc:
        print(f"[error] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
