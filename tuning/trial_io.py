# tuning/trial_io.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())
