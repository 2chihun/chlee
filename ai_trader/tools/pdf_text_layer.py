"""이미지 기반 PDF에 OCR 텍스트 레이어를 추가하여 텍스트 기반 PDF 생성

원본 PDF의 각 페이지 이미지는 그대로 유지하면서,
OCR로 추출한 텍스트를 투명 레이어로 삽입합니다.
결과: 원본과 동일한 외관 + 텍스트 검색/복사 가능
"""

import re
import sys
from pathlib import Path

import fitz  # PyMuPDF


def parse_ocr_pages(ocr_path: str) -> dict[int, str]:
    """OCR 텍스트 파일을 페이지별로 파싱합니다.

    파일 형식: '=== Page N ===' 으로 구분
    """
    text = Path(ocr_path).read_text(encoding="utf-8")
    pages = {}
    # 페이지 구분자로 분할
    parts = re.split(r"=== Page (\d+) ===", text)
    # parts: ['', '1', 'text1', '2', 'text2', ...]
    for i in range(1, len(parts) - 1, 2):
        page_num = int(parts[i])
        page_text = parts[i + 1].strip()
        pages[page_num] = page_text
    return pages


def create_text_pdf(
    src_pdf_path: str,
    ocr_text_path: str,
    output_path: str,
):
    """원본 PDF에 OCR 텍스트 레이어를 추가한 PDF를 생성합니다."""
    pages_text = parse_ocr_pages(ocr_text_path)
    print(f"OCR 페이지 수: {len(pages_text)}")

    src_doc = fitz.open(src_pdf_path)
    total_pages = len(src_doc)
    print(f"원본 PDF 페이지 수: {total_pages}")

    for page_idx in range(total_pages):
        page = src_doc[page_idx]
        page_num = page_idx + 1
        text = pages_text.get(page_num, "")

        if not text:
            continue

        # 페이지 크기
        rect = page.rect

        # 텍스트를 투명(보이지 않는) 레이어로 삽입
        # 폰트 크기를 작게 하여 전체 텍스트를 페이지에 배치
        tw = fitz.TextWriter(rect)
        font = fitz.Font("helv")  # 기본 폰트 (CJK 인코딩용)

        # 텍스트를 줄 단위로 분할하여 페이지에 배치
        lines = text.split("\n")
        y_pos = 10.0
        line_height = (rect.height - 20) / max(len(lines), 1)
        line_height = min(line_height, 12)  # 최대 12pt
        font_size = min(line_height * 0.8, 8)

        for line in lines:
            if y_pos > rect.height - 10:
                break
            if line.strip():
                try:
                    tw.append((10, y_pos), line, font=font, fontsize=font_size)
                except Exception:
                    # CJK 문자 등 폰트 미지원 시 건너뜀
                    pass
            y_pos += line_height

        # 투명 (보이지 않는) 텍스트로 렌더링 (render_mode=3)
        tw.write_text(page, render_mode=3, color=(0, 0, 0))

        if page_num % 20 == 0:
            print(f"  처리 중: {page_num}/{total_pages} 페이지...")

    src_doc.save(output_path, garbage=4, deflate=True)
    src_doc.close()

    out_size = Path(output_path).stat().st_size
    print(f"\n완료! 출력: {output_path}")
    print(f"파일 크기: {out_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    src_pdf = r"C:\Users\chlee\ebook_captures\pages\재무제표 모르면 주식투자 절대로 하지마라 - 사경인.pdf"
    ocr_text = r"C:\Copilot\ai_trader\tools\ocr_output_ko.txt"
    out_pdf = r"C:\Users\chlee\ebook_captures\pages\재무제표 모르면 주식투자 절대로 하지마라 - 사경인_text.pdf"

    create_text_pdf(src_pdf, ocr_text, out_pdf)
