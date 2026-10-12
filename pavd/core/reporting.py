from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from pavd.config import PAVDConfig
from pavd.utils.io import runtime_info, write_json


class RunManifest:
    def __init__(self, config: PAVDConfig) -> None:
        self.config = config
        self.data: dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "runtime": runtime_info(config.root),
            "config": config.as_dict(),
        }

    def add(self, **values: Any) -> None:
        self.data.update(values)

    def save(self, path: Path) -> None:
        write_json(path, self.data)


def write_query_report(path: Path, report: dict[str, Any]) -> None:
    write_json(path, report)


def write_rows_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
