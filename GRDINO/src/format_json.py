#!/usr/bin/env python3
"""Pretty-print JSON files in place.

Usage:
  python format_json.py path/to/file.json
  python format_json.py file1.json file2.json
  python format_json.py --dir path/to/folder

The script reads each JSON file, reformats it with indentation, and writes it
back to the same file path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def format_json_file(path: Path) -> bool:
    """Load JSON from path and rewrite it pretty-printed in place."""
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Invalid JSON: {path} ({exc})")
        return False
    except OSError as exc:
        print(f"[ERROR] Could not read file: {path} ({exc})")
        return False

    try:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"[ERROR] Could not write file: {path} ({exc})")
        return False

    print(f"Formatted {path}")
    return True


def iter_json_files(paths: Iterable[Path], recursive: bool = False) -> Iterable[Path]:
    """Yield JSON files from provided paths and directories."""
    for path in paths:
        if path.is_dir():
            if recursive:
                yield from sorted(path.rglob("*.json"))
            else:
                yield from sorted(path.glob("*.json"))
        else:
            yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretty-print JSON files in place.")
    parser.add_argument("paths", nargs="+", help="JSON file(s) or directory paths")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recurse into directories")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = list(iter_json_files((Path(p) for p in args.paths), recursive=args.recursive))
    if not files:
        print("No JSON files found.")
        return 1

    success = True
    for json_file in files:
        if not format_json_file(json_file):
            success = False

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
