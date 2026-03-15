#!/usr/bin/env python3
"""Convert large source-code trees into study-friendly chunked JSONL files.

Outputs:
1) chunks.jsonl: chunked records compatible with prepare_rag_inputs.py
2) INDEX.md: processing summary and per-root stats
3) source_summary.tsv: per-root and per-extension counts
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SourceRoot:
    label: str
    path: Path


@dataclass
class Chunk:
    source_html: str
    title: str
    chunk_id: int
    char_start: int
    char_end: int
    text: str


TEXT_EXTENSIONS_DEFAULT = (
    ".p",
    ".w",
    ".i",
    ".cls",
    ".df",
    ".txt",
    ".xml",
    ".json",
    ".ini",
    ".yaml",
    ".yml",
    ".sql",
    ".r",
)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def decode_text(raw: bytes) -> str:
    for enc in ("utf-8", "cp949", "euc-kr", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("latin-1", errors="ignore")


def iter_source_files(root: Path, exts: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in exts:
            continue
        files.append(path)
    return files


def split_into_chunks(text: str, source_html: str, title: str, chunk_size: int, overlap: int) -> list[Chunk]:
    if not text:
        return []

    chunks: list[Chunk] = []
    step = max(1, chunk_size - overlap)
    start = 0
    chunk_id = 1

    while start < len(text):
        end = min(len(text), start + chunk_size)
        body = text[start:end].strip()
        if not body:
            break
        chunks.append(
            Chunk(
                source_html=source_html,
                title=title,
                chunk_id=chunk_id,
                char_start=start,
                char_end=end,
                text=body,
            )
        )
        chunk_id += 1
        start += step

    return chunks


def parse_roots(raw_roots: list[str]) -> list[SourceRoot]:
    roots: list[SourceRoot] = []
    for entry in raw_roots:
        if "=" not in entry:
            raise SystemExit(f"Invalid --root format: {entry}. Expected label=path")
        label, path = entry.split("=", 1)
        label = label.strip()
        p = Path(path.strip())
        if not label:
            raise SystemExit(f"Empty root label in --root {entry}")
        if not p.exists():
            raise SystemExit(f"Root path does not exist: {p}")
        roots.append(SourceRoot(label=label, path=p))
    return roots


def run(
    roots: list[SourceRoot],
    output_dir: Path,
    chunk_size: int,
    overlap: int,
    min_text_length: int,
    include_compiled_r: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = output_dir / "chunks.jsonl"
    index_path = output_dir / "INDEX.md"
    summary_path = output_dir / "source_summary.tsv"

    exts = set(TEXT_EXTENSIONS_DEFAULT)
    if not include_compiled_r and ".r" in exts:
        exts.remove(".r")

    total_files = 0
    total_chunks = 0
    skipped_empty = 0
    skipped_too_short = 0
    read_errors = 0

    per_root_files: Counter[str] = Counter()
    per_root_chunks: Counter[str] = Counter()
    per_root_bytes: Counter[str] = Counter()
    per_ext: Counter[str] = Counter()
    docs: defaultdict[str, list[str]] = defaultdict(list)

    with chunks_path.open("w", encoding="utf-8") as out:
        for root in roots:
            files = iter_source_files(root.path, exts)
            print(f"ROOT {root.label}: files={len(files)} path={root.path}")

            for idx, src in enumerate(files, start=1):
                rel = src.relative_to(root.path)
                source_html = f"{root.label}/{rel.as_posix()}"
                title = src.name
                ext = src.suffix.lower() or "(none)"
                size = src.stat().st_size

                total_files += 1
                per_root_files[root.label] += 1
                per_root_bytes[root.label] += size
                per_ext[ext] += 1

                try:
                    raw = src.read_bytes()
                    text = clean_text(decode_text(raw))
                except Exception:
                    read_errors += 1
                    continue

                if not text:
                    skipped_empty += 1
                    continue

                if len(text) < min_text_length:
                    skipped_too_short += 1
                    continue

                chunks = split_into_chunks(
                    text=text,
                    source_html=source_html,
                    title=title,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )

                for c in chunks:
                    out.write(
                        json.dumps(
                            {
                                "source_html": c.source_html,
                                "title": c.title,
                                "chunk_id": c.chunk_id,
                                "char_start": c.char_start,
                                "char_end": c.char_end,
                                "text": c.text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                total_chunks += len(chunks)
                per_root_chunks[root.label] += len(chunks)

                if len(docs[root.label]) < 20:
                    docs[root.label].append(
                        f"- `{source_html}` (bytes: {size}, chars: {len(text)}, chunks: {len(chunks)})"
                    )

                if idx % 500 == 0:
                    print(f"  {root.label}: processed={idx}/{len(files)} total_chunks={total_chunks}")

    with summary_path.open("w", encoding="utf-8") as sf:
        sf.write("section\tkey\tfiles\tchunks\tbytes\n")
        for root in roots:
            sf.write(
                f"root\t{root.label}\t{per_root_files[root.label]}\t{per_root_chunks[root.label]}\t{per_root_bytes[root.label]}\n"
            )
        for ext, n in sorted(per_ext.items(), key=lambda x: (-x[1], x[0])):
            sf.write(f"ext\t{ext}\t{n}\t\t\n")

    with index_path.open("w", encoding="utf-8") as idx:
        idx.write("# Source Study Index\n\n")
        idx.write("## Scope\n\n")
        for root in roots:
            idx.write(f"- `{root.label}`: `{root.path}`\n")
        idx.write("\n")
        idx.write("## Totals\n\n")
        idx.write(f"- Total files scanned: {total_files}\n")
        idx.write(f"- Total chunks emitted: {total_chunks}\n")
        idx.write(f"- Skipped empty text: {skipped_empty}\n")
        idx.write(f"- Skipped short text (<{min_text_length} chars): {skipped_too_short}\n")
        idx.write(f"- Read errors: {read_errors}\n\n")

        idx.write("## By Root\n\n")
        for root in roots:
            idx.write(
                f"- `{root.label}` files={per_root_files[root.label]} chunks={per_root_chunks[root.label]} bytes={per_root_bytes[root.label]}\n"
            )
        idx.write("\n")

        idx.write("## Extension Mix\n\n")
        for ext, n in sorted(per_ext.items(), key=lambda x: (-x[1], x[0])):
            idx.write(f"- `{ext}`: {n}\n")
        idx.write("\n")

        idx.write("## Sample Documents\n\n")
        for root in roots:
            idx.write(f"### {root.label}\n\n")
            if docs[root.label]:
                idx.write("\n".join(docs[root.label]) + "\n\n")
            else:
                idx.write("No sample docs emitted.\n\n")

    print(f"Done. files={total_files}, chunks={total_chunks}")
    print(f"Output: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build study artifacts from source trees")
    parser.add_argument(
        "--root",
        action="append",
        required=True,
        help="Root mapping in form label=path. Repeat for multiple roots.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("study_output_source"),
        help="Output directory for chunks/index/summary",
    )
    parser.add_argument("--chunk-size", type=int, default=2200, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=220, help="Overlap between chunks")
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=40,
        help="Minimum text length required to emit chunks",
    )
    parser.add_argument(
        "--include-compiled-r",
        action="store_true",
        help="Include .r files (compiled artifacts, often low value)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = parse_roots(args.root)
    run(
        roots=roots,
        output_dir=args.output,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        min_text_length=args.min_text_length,
        include_compiled_r=args.include_compiled_r,
    )


if __name__ == "__main__":
    main()
