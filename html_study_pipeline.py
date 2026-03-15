#!/usr/bin/env python3
"""Convert HTML manuals into study-friendly markdown and chunked JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Chunk:
    source_html: str
    title: str
    chunk_id: int
    char_start: int
    char_end: int
    text: str


def clean_text(text: str) -> str:
    """Normalize extracted text for stable chunking and retrieval."""
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_rel_markdown_name(rel_path: Path) -> str:
    """Build a stable markdown filename from relative path to avoid name collisions."""
    raw = "__".join(rel_path.with_suffix("").parts)
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw).strip("._")
    if not name:
        name = "doc"
    if len(name) > 180:
        digest = hashlib.sha1(str(rel_path).encode("utf-8")).hexdigest()[:10]
        name = f"{name[:169]}_{digest}"
    return f"{name}.md"


def iter_html_files(input_dir: Path, excluded_dirs: set[str]) -> Iterable[Path]:
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".html", ".htm"}:
            continue
        parts_lower = {part.lower() for part in path.relative_to(input_dir).parts}
        if parts_lower.intersection(excluded_dirs):
            continue
        yield path


def detect_declared_charset(raw: bytes) -> str | None:
    head = raw[:4096].decode("ascii", errors="ignore")
    m = re.search(r"charset\s*=\s*([A-Za-z0-9_\-]+)", head, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().lower()
    return None


def decode_html(raw: bytes) -> str:
    encodings: list[str] = []
    declared = detect_declared_charset(raw)
    if declared:
        encodings.append(declared)
    encodings.extend(["utf-8", "cp1252", "latin-1"])

    tried: set[str] = set()
    for enc in encodings:
        if enc in tried:
            continue
        tried.add(enc)
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("latin-1", errors="ignore")


def extract_title(html_text: str, fallback: str) -> str:
    m = re.search(r"<title[^>]*>(.*?)</title>", html_text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return fallback
    title = html.unescape(re.sub(r"\s+", " ", m.group(1))).strip()
    return title or fallback


def html_to_text(html_text: str) -> str:
    # Remove non-content regions first.
    text = re.sub(r"<!--.*?-->", " ", html_text, flags=re.DOTALL)
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)

    # Preserve rough paragraph boundaries for readability.
    text = re.sub(r"</(p|div|li|h[1-6]|blockquote|tr|table|section|article)>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)

    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return clean_text(text)


def split_into_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
    source_html: str,
    title: str,
) -> list[Chunk]:
    if not text:
        return []

    chunks: list[Chunk] = []
    step = max(1, chunk_size - overlap)
    start = 0
    chunk_id = 1

    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk_text = text[start:end].strip()
        if not chunk_text:
            break

        chunks.append(
            Chunk(
                source_html=source_html,
                title=title,
                chunk_id=chunk_id,
                char_start=start,
                char_end=end,
                text=chunk_text,
            )
        )
        chunk_id += 1
        start += step

    return chunks


def run(
    input_dir: Path,
    output_dir: Path,
    chunk_size: int,
    overlap: int,
    start_index: int,
    max_files: int,
    min_text_length: int,
    excluded_dirs: set[str],
) -> None:
    output_md = output_dir / "markdown"
    output_jsonl = output_dir / "chunks.jsonl"
    output_index = output_dir / "INDEX.md"

    output_md.mkdir(parents=True, exist_ok=True)

    all_html = list(iter_html_files(input_dir, excluded_dirs))
    selected = all_html[start_index:]
    if max_files > 0:
        selected = selected[:max_files]

    total_files = 0
    total_chunks = 0
    indexed_docs: list[str] = []

    with output_jsonl.open("w", encoding="utf-8") as jf:
        for doc in selected:
            rel_name = str(doc.relative_to(input_dir))
            total_files += 1

            raw = doc.read_bytes()
            html_text = decode_html(raw)
            title = extract_title(html_text, fallback=doc.stem)
            text = html_to_text(html_text)

            md_name = safe_rel_markdown_name(Path(rel_name))
            md_path = output_md / md_name
            md_path.parent.mkdir(parents=True, exist_ok=True)
            with md_path.open("w", encoding="utf-8") as mf:
                mf.write(f"# {rel_name}\n\n")
                mf.write(f"## Title\n\n{title}\n\n")
                if text:
                    mf.write("## Extracted Text\n\n")
                    mf.write(text + "\n")
                else:
                    mf.write("[NO TEXT EXTRACTED]\n")

            chunks: list[Chunk] = []
            if len(text) >= min_text_length:
                chunks = split_into_chunks(
                    text=text,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    source_html=rel_name,
                    title=title,
                )

            for chunk in chunks:
                jf.write(
                    json.dumps(
                        {
                            "source_html": chunk.source_html,
                            "title": chunk.title,
                            "chunk_id": chunk.chunk_id,
                            "char_start": chunk.char_start,
                            "char_end": chunk.char_end,
                            "text": chunk.text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            total_chunks += len(chunks)
            indexed_docs.append(
                f"- `{rel_name}` -> `{(Path('markdown') / md_name).as_posix()}` "
                f"(text_chars: {len(text)}, chunks: {len(chunks)})"
            )
            print(f"Processed: {rel_name} | text_chars={len(text)} chunks={len(chunks)}")

    output_index.parent.mkdir(parents=True, exist_ok=True)
    with output_index.open("w", encoding="utf-8") as idxf:
        idxf.write("# HTML Study Index\n\n")
        idxf.write(f"- Input directory: `{input_dir}`\n")
        idxf.write(f"- Total HTML files: {total_files}\n")
        idxf.write(f"- Total chunks: {total_chunks}\n")
        idxf.write(f"- Excluded dirs: {', '.join(sorted(excluded_dirs))}\n\n")
        idxf.write("## Documents\n\n")
        if indexed_docs:
            idxf.write("\n".join(indexed_docs) + "\n")
        else:
            idxf.write("No HTML files found.\n")

    print(f"Done. HTML files={total_files}, chunks={total_chunks}")
    print(f"Output: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build study artifacts from HTML manuals")
    parser.add_argument("--input", required=True, type=Path, help="HTML root directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("study_output_html"),
        help="Output directory for markdown/chunks/index",
    )
    parser.add_argument("--chunk-size", type=int, default=2000, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap between chunks")
    parser.add_argument("--start-index", type=int, default=0, help="0-based start index")
    parser.add_argument("--max-files", type=int, default=0, help="Process at most N files (0=all)")
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=80,
        help="Minimum extracted text chars required to emit chunks",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=["wwhelp", "wwhimpl", "wwhdata"],
        help="Directory names to exclude (can be repeated)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input directory not found: {args.input}")
    if args.overlap >= args.chunk_size:
        raise SystemExit("--overlap must be smaller than --chunk-size")
    if args.start_index < 0:
        raise SystemExit("--start-index must be >= 0")

    run(
        input_dir=args.input,
        output_dir=args.output,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        start_index=args.start_index,
        max_files=args.max_files,
        min_text_length=args.min_text_length,
        excluded_dirs={d.strip().lower() for d in args.exclude_dir if d and d.strip()},
    )
