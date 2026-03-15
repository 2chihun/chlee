#!/usr/bin/env python3
"""Orchestrate full MFG/Pro source learning in one command.

Pipeline:
1) Run source_study_pipeline.py per root into batch outputs (resume supported).
2) Merge batch chunks into one source chunks.jsonl.
3) Run prepare_rag_inputs.py for normalized source RAG JSONL.
4) Merge with existing ERP/OpenEdge/schema exports.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path


ROOTS = [
    ("pkg85e", Path(r"C:/Package Source Mfg85e")),
    ("ko_2tier", Path(r"Y:/ko")),
    ("w_3tier", Path(r"W:/")),
]


def run_cmd(args: list[str]) -> None:
    print("RUN:", " ".join(args))
    subprocess.run(args, check=True)


def count_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def merge_chunks(batch_dirs: list[Path], merged_chunks: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    merged_chunks.parent.mkdir(parents=True, exist_ok=True)

    with merged_chunks.open("w", encoding="utf-8") as out:
        for bdir in batch_dirs:
            chunks = bdir / "chunks.jsonl"
            label = bdir.name
            if not chunks.exists():
                continue
            with chunks.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    out.write(line)
                    counts[label] += 1
    return counts


def write_plan_and_summary(path: Path, root_file_counts: dict[str, int], chunk_counts: Counter[str]) -> None:
    lines = [
        "# Source Learning Plan",
        "",
        "## Workload Split",
        "",
    ]
    for label in ["pkg85e", "ko_2tier", "w_3tier"]:
        lines.append(
            f"- `{label}` files={root_file_counts.get(label, 0)} chunks={chunk_counts.get(label, 0)}"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Processed in root-level batches to reduce restart risk.",
        "- Re-runnable: completed batches are skipped unless --force-clean is used.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full package source learning pipeline")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("C:/Copilot"),
        help="Workspace root",
    )
    parser.add_argument(
        "--force-clean",
        action="store_true",
        help="Delete previous batch and merged outputs and run from scratch",
    )
    args = parser.parse_args()

    ws = args.workspace
    orchestrator_lock = ws / ".source_learning.lock"
    batch_root = ws / "study_output_source_batches_20260307"
    merged_root = ws / "study_output_source_full_20260307"
    exports_root = ws / "study_output_exports_20260307"

    if orchestrator_lock.exists():
        raise SystemExit(f"Lock exists: {orchestrator_lock}. Another run may be active.")

    if args.force_clean:
        for p in [batch_root, merged_root]:
            if p.exists():
                shutil.rmtree(p)

    orchestrator_lock.write_text(f"started_at={time.ctime()}\n", encoding="utf-8")

    try:
        batch_root.mkdir(parents=True, exist_ok=True)
        merged_root.mkdir(parents=True, exist_ok=True)
        exports_root.mkdir(parents=True, exist_ok=True)

        source_script = ws / "source_study_pipeline.py"
        prep_script = ws / "prepare_rag_inputs.py"
        py = ws / ".venv" / "Scripts" / "python.exe"

        root_file_counts: dict[str, int] = {}
        batch_dirs: list[Path] = []

        for label, root_path in ROOTS:
            if not root_path.exists():
                print(f"WARN: root missing, skip: {root_path}")
                continue

            files = [
                p
                for p in root_path.rglob("*")
                if p.is_file() and p.suffix.lower() in {".p", ".w", ".i", ".cls", ".df", ".txt", ".xml", ".json", ".ini", ".yaml", ".yml", ".sql"}
            ]
            root_file_counts[label] = len(files)

            out_dir = batch_root / label
            done_marker = out_dir / "INDEX.md"
            chunks_file = out_dir / "chunks.jsonl"
            batch_dirs.append(out_dir)

            if done_marker.exists() and chunks_file.exists() and chunks_file.stat().st_size > 0:
                print(f"SKIP (already done): {label}")
                continue

            out_dir.mkdir(parents=True, exist_ok=True)

            run_cmd(
                [
                    str(py),
                    str(source_script),
                    "--root",
                    f"{label}={root_path.as_posix()}",
                    "--output",
                    str(out_dir),
                    "--chunk-size",
                    "2200",
                    "--overlap",
                    "220",
                    "--min-text-length",
                    "40",
                ]
            )

        merged_chunks = merged_root / "chunks.jsonl"
        chunk_counts = merge_chunks(batch_dirs, merged_chunks)

        plan_file = merged_root / "PLAN.md"
        write_plan_and_summary(plan_file, root_file_counts, chunk_counts)

        run_cmd(
            [
                str(py),
                str(prep_script),
                "--input",
                str(merged_chunks),
                "--output",
                str(merged_root / "rag_normalized.jsonl"),
                "--priority-categories",
                "pkg85e,ko_2tier,w_3tier",
                "--per-category-sample",
                "2000",
                "--sample-output",
                str(merged_root / "rag_sample_priority_source.jsonl"),
                "--summary-output",
                str(merged_root / "rag_summary_by_category.tsv"),
            ]
        )

        combined = exports_root / "rag_combined_openedge91e_qad85e_schema_source.jsonl"
        manifest = exports_root / "manifest_source_included.json"

        inputs = [
            ("qad85e", Path(r"C:/Copilot/study_output_qad_85e_full_20260307_final/rag_normalized.jsonl"), "qad_mfgpro_8_5e"),
            ("openedge91e", Path(r"C:/Copilot/study_output_html_full_20260307/rag_normalized.cleaned.noisefiltered.jsonl"), "openedge_91e_webspeed_31e"),
            ("schema_txt", Path(r"C:/Copilot/study_output_exports_20260307/rag_schema_txt.jsonl"), "qad_db_schema"),
            ("package_source", merged_root / "rag_normalized.jsonl", "mfgpro_package_source"),
        ]

        counts: dict[str, int] = {}
        with combined.open("w", encoding="utf-8") as out:
            for name, path, domain in inputs:
                n = 0
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        row["domain"] = domain
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        n += 1
                counts[name] = n

        manifest_obj = {
            "created_at": "2026-03-07",
            "files": {
                "qad": str(inputs[0][1]),
                "openedge": str(inputs[1][1]),
                "schema_txt": str(inputs[2][1]),
                "package_source": str(inputs[3][1]),
                "combined": str(combined),
            },
            "counts": counts,
            "combined_total": sum(counts.values()),
            "schema": ["id", "text", "source", "category", "title", "chunk_id", "char_start", "char_end", "domain"],
        }
        manifest.write_text(json.dumps(manifest_obj, ensure_ascii=False, indent=2), encoding="utf-8")

        print("DONE")
        print(f"combined={combined}")
        print(f"manifest={manifest}")
        print(f"counts={counts}")
        print(f"combined_total={manifest_obj['combined_total']}")

    finally:
        if orchestrator_lock.exists():
            orchestrator_lock.unlink()


if __name__ == "__main__":
    main()
