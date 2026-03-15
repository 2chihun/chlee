"""Korean OCR extraction using Windows built-in OCR (winocr)."""
import asyncio
import os
import sys

import winocr


async def ocr_page(img_path, lang="ko"):
    from PIL import Image
    img = Image.open(img_path)
    result = await winocr.recognize_pil(img, lang)
    lines = []
    for line in result.lines:
        lines.append(line.text)
    return "\n".join(lines)


async def extract_pages(start, end, output_path, append=False):
    base = r"C:\Users\chlee\ebook_captures\pages\재무제표 모르면 주식투자 절대로 하지마라 - 사경인"
    mode = "a" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for p in range(start, end + 1):
            img_path = os.path.join(base, f"page_{p:04d}.png")
            if not os.path.exists(img_path):
                continue
            try:
                text = await ocr_page(img_path)
                f.write(f"=== Page {p} ===\n")
                f.write(text + "\n\n")
                f.flush()
                print(f"Page {p} done")
            except Exception as e:
                print(f"Page {p} error: {e}")


if __name__ == "__main__":
    output = r"C:\Copilot\ai_trader\tools\ocr_output_ko.txt"
    if len(sys.argv) >= 3:
        s, e = int(sys.argv[1]), int(sys.argv[2])
    else:
        s, e = 3, 15
    append = len(sys.argv) >= 4 and sys.argv[3] == "append"
    asyncio.run(extract_pages(s, e, output, append))
    print("Done!")
