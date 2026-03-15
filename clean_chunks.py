#!/usr/bin/env python3
"""Clean chunk JSONL by removing short and duplicate chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate and filter chunk JSONL")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-chars", type=int, default=120)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    dropped_short = 0
    dropped_dup = 0
    seen: set[str] = set()

    with args.input.open("r", encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            total += 1
            try:
                row = json.loads(line)
            except Exception:
                continue

            text = str(row.get("text", "")).strip()
            if len(text) < args.min_chars:
                dropped_short += 1
                continue

            signature = hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()
            if signature in seen:
                dropped_dup += 1
                continue

            seen.add(signature)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    print(f"total={total}")
    print(f"kept={kept}")
    print(f"dropped_short={dropped_short}")
    print(f"dropped_dup={dropped_dup}")


if __name__ == "__main__":
    main()
