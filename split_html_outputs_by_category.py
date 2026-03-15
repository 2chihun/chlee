#!/usr/bin/env python3
"""Split a full HTML study output into per-category datasets."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("._")
    return cleaned or "unknown"


def load_chunks(chunks_path: Path) -> list[dict]:
    rows: list[dict] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Split HTML study output by top-level category")
    parser.add_argument("--input-output", required=True, type=Path, help="Full output dir (has chunks.jsonl/markdown)")
    parser.add_argument("--dest-root", required=True, type=Path, help="Destination root for category outputs")
    args = parser.parse_args()

    src = args.input_output
    chunks_path = src / "chunks.jsonl"
    md_dir = src / "markdown"

    if not chunks_path.exists():
        raise SystemExit(f"chunks.jsonl not found: {chunks_path}")
    if not md_dir.exists():
        raise SystemExit(f"markdown directory not found: {md_dir}")

    rows = load_chunks(chunks_path)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    html_to_md: dict[str, str] = {}

    for row in rows:
        source = str(row.get("source_html", "")).strip()
        if not source:
            continue
        cat = source.split("\\", 1)[0] if "\\" in source else source.split("/", 1)[0]
        cat = safe_name(cat)
        by_cat[cat].append(row)

        # markdown naming rule from html_study_pipeline.py
        rel_wo_ext = re.sub(r"\.[Hh][Tt][Mm][Ll]?$", "", source)
        md_name = safe_name(rel_wo_ext.replace("\\", "__").replace("/", "__")) + ".md"
        html_to_md[source] = md_name

    args.dest_root.mkdir(parents=True, exist_ok=True)

    total_categories = 0
    for cat, cat_rows in sorted(by_cat.items()):
        out = args.dest_root / cat
        out_md = out / "markdown"
        out_md.mkdir(parents=True, exist_ok=True)

        used_sources = sorted({str(r.get("source_html", "")) for r in cat_rows if r.get("source_html")})
        used_md = sorted({html_to_md[s] for s in used_sources if s in html_to_md})

        with (out / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for row in cat_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        copied = 0
        missing = 0
        for md_name in used_md:
            src_md = md_dir / md_name
            dst_md = out_md / md_name
            if src_md.exists():
                shutil.copy2(src_md, dst_md)
                copied += 1
            else:
                missing += 1

        with (out / "INDEX.md").open("w", encoding="utf-8") as idx:
            idx.write(f"# HTML Study Index - {cat}\n\n")
            idx.write(f"- Category: `{cat}`\n")
            idx.write(f"- Source output: `{src}`\n")
            idx.write(f"- Total docs: {len(used_sources)}\n")
            idx.write(f"- Total chunks: {len(cat_rows)}\n")
            idx.write(f"- Markdown copied: {copied}\n")
            idx.write(f"- Markdown missing: {missing}\n\n")
            idx.write("## Documents\n\n")
            for s in used_sources:
                idx.write(f"- `{s}`\n")

        total_categories += 1
        print(f"Category={cat} docs={len(used_sources)} chunks={len(cat_rows)} copied_md={copied} missing_md={missing}")

    print(f"Done. categories={total_categories}")


if __name__ == "__main__":
    main()
