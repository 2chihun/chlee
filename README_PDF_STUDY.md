# PDF Study Pipeline

This workspace now includes a script to convert PDF documents into study-friendly data.

## What it creates

- `study_output/markdown/*.md`: one markdown file per PDF.
- `study_output/chunks.jsonl`: chunked text records for retrieval or model training.
- `study_output/INDEX.md`: summary index with file mappings.

## Run

```powershell
C:/Copilot/.venv/Scripts/python.exe C:/Copilot/pdf_study_pipeline.py --input "D:/-/Progress DOC V9.1E and WebSpeed 3.1E PDF" --output C:/Copilot/study_output
```

## Optional tuning

```powershell
C:/Copilot/.venv/Scripts/python.exe C:/Copilot/pdf_study_pipeline.py --input "D:/-/Progress DOC V9.1E and WebSpeed 3.1E PDF" --output C:/Copilot/study_output --chunk-size 2500 --overlap 300
```

## Notes

- PDF text extraction quality depends on how the PDF was generated.
- Image-only PDFs require OCR, which is not included in this script.

## HTML Manual Pipeline

For OpenEdge/WebSpeed HTML manuals, use the HTML pipeline script:

```powershell
C:/Copilot/.venv/Scripts/python.exe C:/Copilot/html_study_pipeline.py --input "D:/-/Progress DOC V9.1E and WebSpeed 3.1E HTML" --output C:/Copilot/study_output_html
```

Optional batch run:

```powershell
C:/Copilot/.venv/Scripts/python.exe C:/Copilot/html_study_pipeline.py --input "D:/-/Progress DOC V9.1E and WebSpeed 3.1E HTML" --output C:/Copilot/study_output_html_batch1000 --max-files 1000
```

By default, the HTML pipeline excludes helper directories: `wwhelp`, `wwhimpl`, `wwhdata`.

### Full HTML run (all manuals)

```powershell
C:/Copilot/.venv/Scripts/python.exe C:/Copilot/html_study_pipeline.py --input "D:/-/Progress DOC V9.1E and WebSpeed 3.1E HTML" --output C:/Copilot/study_output_html_full_20260307
```

### Split full output by category

```powershell
C:/Copilot/.venv/Scripts/python.exe C:/Copilot/split_html_outputs_by_category.py --input-output C:/Copilot/study_output_html_full_20260307 --dest-root C:/Copilot/study_output_html_by_category_20260307
```

### Clean chunks (remove short and duplicate)

```powershell
C:/Copilot/.venv/Scripts/python.exe C:/Copilot/clean_chunks.py --input C:/Copilot/study_output_html_full_20260307/chunks.jsonl --output C:/Copilot/study_output_html_full_20260307/chunks.cleaned.min120.dedup.jsonl --min-chars 120
```

### Prepare RAG-ready normalized JSONL

```powershell
C:/Copilot/.venv/Scripts/python.exe C:/Copilot/prepare_rag_inputs.py --input C:/Copilot/study_output_html_full_20260307/chunks.cleaned.min120.dedup.jsonl --output C:/Copilot/study_output_html_full_20260307/rag_normalized.jsonl --priority-categories langref,proghand,epi --per-category-sample 1200 --sample-output C:/Copilot/study_output_html_full_20260307/rag_sample_priority_langref_proghand_epi.jsonl --summary-output C:/Copilot/study_output_html_full_20260307/rag_summary_by_category.tsv
```

### Prepare RAG JSONL with boilerplate filtering

Use this when you want to drop likely cover/footer chunks (copyright/contact/cookie boilerplate):

```powershell
C:/Copilot/.venv/Scripts/python.exe C:/Copilot/prepare_rag_inputs.py --input C:/Copilot/study_output_html_full_20260307/chunks.cleaned.min120.dedup.jsonl --output C:/Copilot/study_output_html_full_20260307/rag_normalized.cleaned.noisefiltered.jsonl --priority-categories langref,proghand,epi --per-category-sample 1200 --sample-output C:/Copilot/study_output_html_full_20260307/rag_sample_priority_langref_proghand_epi.cleaned.noisefiltered.jsonl --summary-output C:/Copilot/study_output_html_full_20260307/rag_summary_by_category.cleaned.noisefiltered.tsv --drop-cover-boilerplate --removed-output C:/Copilot/study_output_html_full_20260307/rag_removed_cover_boilerplate.jsonl
```
