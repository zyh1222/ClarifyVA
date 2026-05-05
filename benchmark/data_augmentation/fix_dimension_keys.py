"""
fix_dimension_keys.py
---------------------
Remove category prefixes from dimension keys in da-dev-plans.jsonl and
da-dev-constructed.jsonl (e.g., "analysis.method" -> "method") so the
names match the plan_generator output format.

Usage:
    python fix_dimension_keys.py
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "dataset"


def strip_prefix(key: str) -> str:
    """'analysis.method' -> 'method', 'data.source_file' -> 'source_file'."""
    parts = key.split(".")
    return parts[-1] if len(parts) >= 2 else key


def fix_file(path: Path, fields: list[str]) -> int:
    """Update dict keys in the given fields and return the count of changes."""
    if not path.exists():
        print(f"  Skip (missing): {path.name}")
        return 0

    lines = []
    changed = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            modified = False
            for field in fields:
                old = d.get(field)
                if not isinstance(old, dict):
                    continue
                new = {}
                for k, v in old.items():
                    nk = strip_prefix(k)
                    new[nk] = v
                    if nk != k:
                        modified = True
                d[field] = new
            if modified:
                changed += 1
            lines.append(json.dumps(d, ensure_ascii=False))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return changed


def main():
    # 1. da-dev-plans.jsonl: dimension_values
    plans_file = DATA_DIR / "da-dev-plans.jsonl"
    n = fix_file(plans_file, ["dimension_values"])
    print(f"da-dev-plans.jsonl: {n} updated")

    # 2. da-dev-constructed.jsonl: modified_dimensions + original_dimension_values
    constructed_file = DATA_DIR / "da-dev-constructed.jsonl"
    n = fix_file(constructed_file, ["modified_dimensions", "original_dimension_values"])
    print(f"da-dev-constructed.jsonl: {n} updated")


if __name__ == "__main__":
    main()
