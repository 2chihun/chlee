#!/usr/bin/env python3
"""Convert PDFs into study-friendly markdown and chunked JSONL files.

This script extracts text from PDF files recursively and writes:
1) Per-PDF markdown files for reading.
2) A JSONL file containing text chunks for search/RAG training.
3) An index markdown file summarizing processed documents.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue as pyqueue
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text


@dataclass
class Chunk:
    source_pdf: str
    page_start: int
    page_end: int
    chunk_id: int
    text: str


def clean_text(text: str) -> str:
    """Normalize extracted text for more consistent downstream learning."""
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def iter_pdf_files(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.rglob("*.pdf")):
        if path.is_file():
            yield path


def split_into_chunks(
    pages: list[tuple[int, str]],
    chunk_size: int,
    overlap: int,
    source_pdf: str,
) -> list[Chunk]:
    """Chunk by characters while preserving rough page boundaries in metadata."""
    joined = []
    page_map = []
    cursor = 0
    for page_num, text in pages:
        if not text:
            continue
        block = f"[Page {page_num}]\n{text}\n\n"
        joined.append(block)
        page_map.append((cursor, cursor + len(block), page_num))
        cursor += len(block)

    full_text = "".join(joined).strip()
    if not full_text:
        return []

    chunks: list[Chunk] = []
    start = 0
    chunk_id = 1
    step = max(1, chunk_size - overlap)

    while start < len(full_text):
        end = min(len(full_text), start + chunk_size)
        text = full_text[start:end].strip()
        if not text:
            break

        page_start = 1
        page_end = 1
        for s, e, page_num in page_map:
            if s <= start < e:
                page_start = page_num
                break
        for s, e, page_num in page_map:
            if s < end <= e:
                page_end = page_num
                break
            if end > e:
                page_end = page_num

        chunks.append(
            Chunk(
                source_pdf=source_pdf,
                page_start=page_start,
                page_end=page_end,
                chunk_id=chunk_id,
                text=text,
            )
        )
        chunk_id += 1
        start += step

    return chunks


def extract_pdf(pdf_path: Path) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        print(f"WARN: cannot open PDF: {pdf_path} ({exc})")
        return pages

    for idx, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:
            print(f"WARN: text extraction failed: {pdf_path} page={idx} ({exc})")
            raw = ""
        cleaned = clean_text(raw)
        pages.append((idx, cleaned))
    return pages


def extract_pdf_fallback(pdf_path: Path) -> list[tuple[int, str]]:
    """Fallback extraction using pdfminer when pypdf yields no usable pages."""
    try:
        text = pdfminer_extract_text(str(pdf_path)) or ""
    except Exception as exc:
        print(f"WARN: pdfminer fallback failed: {pdf_path} ({exc})")
        return []

    text = clean_text(text)
    if not text:
        return []

    # pdfminer often separates pages with form-feed characters.
    parts = [clean_text(p) for p in text.split("\x0c")]
    parts = [p for p in parts if p]
    if not parts:
        return [(1, text)]
    return [(idx, part) for idx, part in enumerate(parts, start=1)]


def _extract_pdf_worker(pdf_path_str: str, queue: mp.Queue) -> None:
    """Worker process for guarded extraction of one PDF."""
    pdf_path = Path(pdf_path_str)
    try:
        pages = extract_pdf(pdf_path)
        if not any(text for _, text in pages):
            pages = extract_pdf_fallback(pdf_path)
        queue.put({"ok": True, "pages": pages})
    except Exception as exc:
        queue.put({"ok": False, "error": str(exc), "pages": []})


def extract_pdf_with_timeout(pdf_path: Path, timeout_sec: int) -> list[tuple[int, str]]:
    """Run PDF extraction in a child process and terminate on timeout."""
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_extract_pdf_worker, args=(str(pdf_path), queue))
    proc.start()

    # Important: fetch from queue with timeout before join().
    # Joining first can deadlock when a large payload is still being flushed.
    try:
        result = queue.get(timeout=timeout_sec)
    except pyqueue.Empty:
        if proc.is_alive():
            proc.terminate()
            proc.join()
        print(f"WARN: timeout while processing PDF: {pdf_path} (>{timeout_sec}s)")
        return []

    proc.join(timeout=5)
    if proc.is_alive():
        proc.terminate()
        proc.join()

    if not result.get("ok", False):
        print(f"WARN: extraction worker error: {pdf_path} ({result.get('error', 'unknown')})")
        return []
    return result.get("pages", [])


def safe_stem(path: Path) -> str:
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", path.stem)
    return stem[:120] if len(stem) > 120 else stem


def run(
    input_dir: Path,
    output_dir: Path,
    chunk_size: int,
    overlap: int,
    pdf_timeout: int,
    start_index: int,
    max_files: int,
) -> None:
    output_md = output_dir / "markdown"
    output_jsonl = output_dir / "chunks.jsonl"
    output_index = output_dir / "INDEX.md"

    output_md.mkdir(parents=True, exist_ok=True)

    total_pdfs = 0
    total_chunks = 0
    indexed_docs: list[str] = []

    with output_jsonl.open("w", encoding="utf-8") as jf:
        all_pdfs = list(iter_pdf_files(input_dir))
        selected = all_pdfs[start_index:]
        if max_files > 0:
            selected = selected[:max_files]

        for pdf in selected:
            total_pdfs += 1
            rel_name = str(pdf.relative_to(input_dir))
            pages = extract_pdf_with_timeout(pdf, timeout_sec=pdf_timeout)

            md_name = f"{safe_stem(pdf)}.md"
            md_path = output_md / md_name
            # Recreate output folder if it was removed while the pipeline is running.
            md_path.parent.mkdir(parents=True, exist_ok=True)
            with md_path.open("w", encoding="utf-8") as mf:
                mf.write(f"# {rel_name}\n\n")
                for page_num, text in pages:
                    mf.write(f"## Page {page_num}\n\n")
                    mf.write((text or "[NO TEXT EXTRACTED]") + "\n\n")

            chunks = split_into_chunks(
                pages=pages,
                chunk_size=chunk_size,
                overlap=overlap,
                source_pdf=rel_name,
            )
            for chunk in chunks:
                jf.write(
                    json.dumps(
                        {
                            "source_pdf": chunk.source_pdf,
                            "page_start": chunk.page_start,
                            "page_end": chunk.page_end,
                            "chunk_id": chunk.chunk_id,
                            "text": chunk.text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            total_chunks += len(chunks)
            indexed_docs.append(
                f"- `{rel_name}` -> `{(Path('markdown') / md_name).as_posix()}` (pages: {len(pages)}, chunks: {len(chunks)})"
            )
            print(f"Processed: {rel_name} | pages={len(pages)} chunks={len(chunks)}")

    output_index.parent.mkdir(parents=True, exist_ok=True)
    with output_index.open("w", encoding="utf-8") as idxf:
        idxf.write("# PDF Study Index\n\n")
        idxf.write(f"- Input directory: `{input_dir}`\n")
        idxf.write(f"- Total PDFs: {total_pdfs}\n")
        idxf.write(f"- Total chunks: {total_chunks}\n\n")
        idxf.write("## Documents\n\n")
        if indexed_docs:
            idxf.write("\n".join(indexed_docs) + "\n")
        else:
            idxf.write("No PDFs found.\n")

    print(f"Done. PDFs={total_pdfs}, chunks={total_chunks}")
    print(f"Output: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build study artifacts from PDFs")
    parser.add_argument("--input", required=True, type=Path, help="PDF root directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("study_output"),
        help="Output directory for markdown/chunks/index",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Chunk size in characters",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Overlap in characters between chunks",
    )
    parser.add_argument(
        "--pdf-timeout",
        type=int,
        default=120,
        help="Max seconds per PDF before skipping",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="0-based start position in sorted PDF list",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Process at most N PDFs (0 means all)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    mp.freeze_support()
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
        pdf_timeout=args.pdf_timeout,
        start_index=args.start_index,
        max_files=args.max_files,
    )
