"""매뉴얼 PDF 변환 스크립트

마크다운 매뉴얼을 PDF로 변환합니다.
사전 설치: pip install markdown weasyprint

대안: pip install mdpdf
"""

import os
import sys


def convert_with_mdpdf():
    """mdpdf를 사용한 변환 (가장 간단)"""
    os.system('mdpdf -o docs/AI_Trader_사용자_매뉴얼.pdf docs/user_manual.md')
    os.system('mdpdf -o docs/AI_Trader_관리자_매뉴얼.pdf docs/admin_manual.md')
    print("PDF 변환 완료!")


def convert_with_markdown_and_weasyprint():
    """markdown + weasyprint를 사용한 변환"""
    try:
        import markdown
        from weasyprint import HTML
    except ImportError:
        print("필요 패키지가 없습니다. 아래 명령어로 설치하세요:")
        print("  pip install markdown weasyprint")
        return

    css = """
    @page { size: A4; margin: 2cm; }
    body { font-family: 'Malgun Gothic', sans-serif; font-size: 11pt; line-height: 1.6; }
    h1 { color: #1a5276; border-bottom: 2px solid #1a5276; padding-bottom: 10px; page-break-before: always; }
    h1:first-of-type { page-break-before: avoid; }
    h2 { color: #2471a3; margin-top: 30px; }
    h3 { color: #2e86c1; }
    code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-size: 10pt; }
    pre { background: #2d2d2d; color: #f8f8f2; padding: 15px; border-radius: 5px;
          font-size: 9pt; overflow-x: auto; page-break-inside: avoid; }
    table { border-collapse: collapse; width: 100%; margin: 15px 0; }
    th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
    th { background: #2471a3; color: white; }
    tr:nth-child(even) { background: #f9f9f9; }
    blockquote { border-left: 4px solid #e74c3c; padding: 10px 15px; background: #fef9e7; margin: 15px 0; }
    """

    for src, dst in [
        ("docs/user_manual.md", "docs/AI_Trader_사용자_매뉴얼.pdf"),
        ("docs/admin_manual.md", "docs/AI_Trader_관리자_매뉴얼.pdf"),
    ]:
        with open(src, encoding="utf-8") as f:
            md_content = f.read()

        html_content = markdown.markdown(
            md_content,
            extensions=["tables", "fenced_code", "toc"],
        )

        full_html = f"""
        <!DOCTYPE html>
        <html><head>
        <meta charset="utf-8">
        <style>{css}</style>
        </head><body>{html_content}</body></html>
        """

        HTML(string=full_html).write_pdf(dst)
        print(f"생성: {dst}")

    print("PDF 변환 완료!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--mdpdf":
        convert_with_mdpdf()
    else:
        convert_with_markdown_and_weasyprint()
