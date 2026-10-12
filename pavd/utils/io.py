from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def runtime_info(root: Path) -> dict[str, Any]:
    packages = {
        name: package_version(name)
        for name in ("numpy", "opencv-python", "Pillow", "torch", "faiss-cpu", "open_clip_torch", "transformers")
    }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "git_revision": git_revision(root),
        "packages": packages,
    }
