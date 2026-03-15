"""OCR extraction script for ebook pages."""
import os
import sys

from PIL import Image
from rapidocr_onnxruntime import RapidOCR

BASE = r"C:\Users\chlee\ebook_captures\pages\재무제표 모르면 주식투자 절대로 하지마라 - 사경인"
OUTPUT = r"C:\Copilot\ai_trader\tools\ocr_output.txt"
TEMP = r"C:\Copilot\ai_trader\tools\temp_page.png"


def extract_pages(page_ranges, append=False):
    ocr = RapidOCR()
    pages = []
    for r in page_ranges:
        if isinstance(r, tuple):
            pages.extend(range(r[0], r[1] + 1))
        else:
            pages.append(r)

    mode = "a" if append else "w"
    with open(OUTPUT, mode, encoding="utf-8") as f:
        for p in pages:
            img_path = os.path.join(BASE, f"page_{p:04d}.png")
            if not os.path.exists(img_path):
                continue
            # Resize for speed
            img = Image.open(img_path)
            w, h = img.size
            scale = min(1.0, 1500 / max(w, h))
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)))
            img.save(TEMP)
            result, elapse = ocr(TEMP)
            f.write(f"=== Page {p} ===\n")
            if result:
                for line in result:
                    f.write(line[1] + "\n")
            else:
                f.write("(no text)\n")
            f.write("\n")
            f.flush()
            print(f"Page {p} done ({elapse})")

    if os.path.exists(TEMP):
        os.remove(TEMP)


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "toc"

    if phase == "toc":
        extract_pages([(3, 15)])
    elif phase == "all":
        extract_pages([(1, 209)])
    elif phase == "key":
        extract_pages([(3, 15), (50, 80), (100, 140), (150, 180), (195, 209)])
    else:
        parts = phase.split("-")
        extract_pages([(int(parts[0]), int(parts[1]))])
