"""켄 피셔 책 128~150 페이지 OCR - 기존 파일에 추가"""
import os
import subprocess
from pathlib import Path

os.environ['TESSDATA_PREFIX'] = r'C:\Users\chlee\tessdata'

input_folder = Path(r'C:\Copilot\captures\켄_피셔_주식시장은_어떻게_반복되는가')
output_file = Path(r'C:\Copilot\ken_fisher_text.txt')

# 128~150 파일만 처리
files = sorted(
    f for f in input_folder.glob('*.png')
    if 128 <= int(f.stem) <= 150
)
print(f'총 {len(files)}장 OCR 처리 시작 (128~150)...')

new_text = []
for i, f in enumerate(files, 1):
    try:
        result = subprocess.run(
            [r'C:\Program Files\Tesseract-OCR\tesseract.exe', str(f), 'stdout', '-l', 'kor+eng'],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            env={**os.environ, 'TESSDATA_PREFIX': r'C:\Users\chlee\tessdata'}
        )
        text = result.stdout.strip()
        new_text.append(f'=== {f.name} ===\n{text}')
        print(f'  {i}/{len(files)} {f.name} 완료')
    except Exception as e:
        print(f'  [{f.name}] 오류: {e}')
        new_text.append(f'=== {f.name} ===\n[OCR 실패: {e}]')

# 기존 파일에 추가
with open(output_file, 'a', encoding='utf-8') as fp:
    fp.write('\n\n' + '\n\n'.join(new_text))

print(f'\n완료! 파일 크기: {output_file.stat().st_size / 1024:.1f}KB')
