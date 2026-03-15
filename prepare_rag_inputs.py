#!/usr/bin/env python3
"""Prepare normalized RAG-ready JSONL from cleaned chunk JSONL.

Outputs:
1) Normalized JSONL with stable ids and compact metadata.
2) Category-prioritized sample JSONL for quick retrieval tests.
3) Summary TSV with per-category counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def category_from_source(source_html: str) -> str:
    source_html = source_html.strip()
    if "\\" in source_html:
        return source_html.split("\\", 1)[0].strip().lower() or "unknown"
    if "/" in source_html:
        return source_html.split("/", 1)[0].strip().lower() or "unknown"

    # For bare PDF filenames (no directory path), infer a stable category from filename.
    if source_html.lower().endswith(".pdf"):
        return infer_pdf_category(source_html)

    return "unknown"


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\.[a-z0-9]+$", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def infer_pdf_category(source_pdf: str) -> str:
    name = source_pdf.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    stem = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
    low = stem.lower()

    if "user guide volume" in low:
        m = re.search(r"user\s+guide\s+volume\s+([0-9]+[a-z]?)\s*-\s*(.+)$", low)
        if m:
            vol = m.group(1)
            topic = slugify(m.group(2))
            return f"user_guide_v{vol}_{topic}"
        return "user_guide"

    if "release bulletin" in low:
        return "release_bulletin"
    if "installation and conversion guide" in low:
        return "installation_conversion"
    if "installation guide" in low:
        return "installation"
    if "database definitions" in low:
        return "database_definitions"
    if "file relationships" in low:
        return "file_relationships"
    if "object-based component model" in low:
        return "obcm"
    if low == "85erd":
        return "erd"

    # Fallback to a deterministic slug from filename stem.
    return slugify(stem)


def stable_doc_id(source_html: str, chunk_id: int, text: str) -> str:
    key = f"{source_html}|{chunk_id}|{text[:120]}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def detect_boilerplate_reasons(text: str) -> list[str]:
    """Return heuristic reasons when text looks like cover/footer boilerplate."""
    low = text.lower()
    reasons: list[str] = []

    if "cookie" in low and "javascript" in low and len(text) < 350:
        reasons.append("cookie_js_short")
    if "copyright" in low and len(text) < 250:
        reasons.append("copyright_short")
    if "www.progress.com" in low and len(text) < 400:
        reasons.append("cover_boilerplate")

    return reasons


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_tsv(path: Path, counts: Counter[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("category\tcount\n")
        for cat, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            f.write(f"{cat}\t{n}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare normalized RAG inputs from cleaned chunks")
    parser.add_argument("--input", required=True, type=Path, help="Input cleaned JSONL path")
    parser.add_argument("--output", required=True, type=Path, help="Output normalized JSONL path")
    parser.add_argument(
        "--priority-categories",
        default="langref,proghand,epi",
        help="Comma-separated category priority for sample set",
    )
    parser.add_argument(
        "--per-category-sample",
        type=int,
        default=1200,
        help="Max rows per prioritized category in sample output",
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=Path("rag_sample_priority.jsonl"),
        help="Output sample JSONL path",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("rag_summary_by_category.tsv"),
        help="Output TSV summary by category",
    )
    parser.add_argument(
        "--drop-cover-boilerplate",
        action="store_true",
        help="Drop heuristic cover/footer boilerplate rows from normalized output",
    )
    parser.add_argument(
        "--removed-output",
        type=Path,
        default=None,
        help="Optional JSONL path to write dropped rows metadata",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    raw_rows = load_jsonl(args.input)
    normalized: list[dict] = []
    dropped_rows: list[dict] = []
    counts: Counter[str] = Counter()
    dropped_by_reason: Counter[str] = Counter()

    for row in raw_rows:
        # Support both HTML pipeline rows (source_html) and PDF pipeline rows (source_pdf).
        source_html = str(row.get("source_html") or row.get("source_pdf") or "").strip()
        title = str(row.get("title", "")).strip()
        text = normalize_whitespace(str(row.get("text", "")))
        chunk_id = int(row.get("chunk_id", 0) or 0)
        char_start = int(row.get("char_start", 0) or 0)
        char_end = int(row.get("char_end", 0) or 0)

        if not source_html or not text:
            continue

        reasons = detect_boilerplate_reasons(text)
        if args.drop_cover_boilerplate and reasons:
            dropped_rows.append(
                {
                    "source": source_html,
                    "title": title,
                    "category": category_from_source(source_html),
                    "chunk_id": chunk_id,
                    "char_start": char_start,
                    "char_end": char_end,
                    "len": len(text),
                    "reasons": reasons,
                    "text_preview": " ".join(text.split())[:280],
                }
            )
            for reason in reasons:
                dropped_by_reason[reason] += 1
            continue

        category = category_from_source(source_html)
        rid = stable_doc_id(source_html, chunk_id, text)

        normalized.append(
            {
                "id": rid,
                "text": text,
                "source": source_html,
                "category": category,
                "title": title,
                "chunk_id": chunk_id,
                "char_start": char_start,
                "char_end": char_end,
            }
        )
        counts[category] += 1

    write_jsonl(args.output, normalized)
    write_summary_tsv(args.summary_output, counts)

    priority = [c.strip().lower() for c in args.priority_categories.split(",") if c.strip()]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in normalized:
        grouped[row["category"]].append(row)

    sample_rows: list[dict] = []
    for cat in priority:
        sample_rows.extend(grouped.get(cat, [])[: args.per_category_sample])

    write_jsonl(args.sample_output, sample_rows)

    if args.removed_output:
        write_jsonl(args.removed_output, dropped_rows)

    print(f"input_rows={len(raw_rows)}")
    print(f"normalized_rows={len(normalized)}")
    print(f"output={args.output}")
    print(f"sample_rows={len(sample_rows)}")
    print(f"sample_output={args.sample_output}")
    print(f"summary_output={args.summary_output}")
    print(f"dropped_rows={len(dropped_rows)}")
    if args.removed_output:
        print(f"removed_output={args.removed_output}")
    if dropped_by_reason:
        summary = ", ".join(f"{k}:{v}" for k, v in sorted(dropped_by_reason.items()))
        print(f"dropped_by_reason={summary}")


if __name__ == "__main__":
    main()
