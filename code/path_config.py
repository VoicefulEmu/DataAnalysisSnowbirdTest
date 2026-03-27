from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
EXPORTS_DIR = PROJECT_ROOT / "exports"
PHOTOS_DIR = PROJECT_ROOT / "photos"


def ensure_standard_dirs() -> None:
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)


def _newest(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def default_hr_csv() -> Path:
    candidates = list(EXPORTS_DIR.glob("* HR_*.csv"))
    base_candidates = [
        p for p in candidates if not re.search(r"_\d+s_\d+s\.csv$", p.name)
    ]
    selected = _newest(base_candidates) or _newest(candidates)
    if selected is None:
        raise FileNotFoundError("No HR CSV found in exports/ (expected '* HR_*.csv').")
    return selected


def default_lr_csv() -> Path:
    selected = _newest(list(EXPORTS_DIR.glob("* LR_*.csv")))
    if selected is None:
        raise FileNotFoundError("No LR CSV found in exports/ (expected '* LR_*.csv').")
    return selected


def default_summary_csv() -> Path | None:
    return _newest(list(EXPORTS_DIR.glob("*summary*.csv")))


def default_eng_file() -> Path:
    preferred = CODE_DIR / "AeroTech_O5500X-PS.eng"
    if preferred.exists():
        return preferred
    selected = _newest(list(CODE_DIR.glob("*.eng")))
    if selected is None:
        raise FileNotFoundError("No .eng motor file found in code/.")
    return selected


def default_analysis_dir(csv_path: Path) -> Path:
    return EXPORTS_DIR / f"{csv_path.stem}_analysis"
