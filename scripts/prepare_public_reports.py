"""Replace local absolute paths in saved reports without changing measurements.

Original files and a transformation manifest stay in ignored build/ storage.
Run once before publishing a checkout containing local benchmark reports.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    backup = root / "build/private-publication/report-originals"
    backup.mkdir(parents=True, exist_ok=True)
    mapping = [(root, "<library>"), (root.parent / "myvllm", "<engine>"),
               (Path(sys.prefix), "<python>"), (Path(sys.base_prefix), "<python>")]
    prefixes = sorted([(str(path).replace("\\", "/"), label) for path, label in mapping],
                      key=lambda row: len(row[0]), reverse=True)

    def clean_string(value):
        normal = value.replace("\\", "/")
        for prefix, label in prefixes:
            if normal.casefold() == prefix.casefold() or normal.casefold().startswith(prefix.casefold() + "/"):
                return label + normal[len(prefix):]
        if re.match(r"^[A-Za-z]:/", normal):
            return "<local>/" + normal.rsplit("/", 1)[-1]
        return value

    def clean(value):
        if isinstance(value, str):
            return clean_string(value)
        if isinstance(value, list):
            return [clean(child) for child in value]
        if isinstance(value, dict):
            result = {clean_string(key): clean(child) for key, child in value.items()}
            if len(result) != len(value):
                raise ValueError("sanitized keys collide")
            return result
        return value

    def numbers(value):
        if isinstance(value, dict):
            return [numbers(child) for child in value.values()]
        if isinstance(value, list):
            return [numbers(child) for child in value]
        return None if isinstance(value, str) else value

    rows = []
    for path in sorted(root.glob("bench_*.json")):
        raw = path.read_bytes()
        before = json.loads(raw)
        after = clean(before)
        assert numbers(before) == numbers(after), path.name
        if before == after:
            continue
        original = backup / path.name
        if not original.exists():
            original.write_bytes(raw)
        path.write_text(json.dumps(after, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rows.append({"file": path.name, "original_sha256": hashlib.sha256(raw).hexdigest(),
                     "public_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "numeric_values_preserved": True})
    (backup.parent / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Sanitized {len(rows)} reports; numeric values preserved; originals in ignored build/ storage.")


if __name__ == "__main__":
    main()
