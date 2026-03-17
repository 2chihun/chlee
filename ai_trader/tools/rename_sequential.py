"""캡처 이미지를 생성일시 순으로 001.png, 002.png ... 로 이름 변경"""

from pathlib import Path
import sys

folder_path = sys.argv[1] if len(sys.argv) > 1 else "."
folder = Path(folder_path)

files = sorted(folder.glob("*.png"))
total = len(files)
print(f"총 {total}장 발견")

if total == 0:
    print("PNG 파일이 없습니다.")
    sys.exit(0)

# 1단계: 임시 이름으로 변경 (기존 파일명과 충돌 방지)
for i, f in enumerate(files, 1):
    f.rename(folder / f"__tmp_{i:03d}.png")

# 2단계: 최종 순번 이름으로 변경
tmp_files = sorted(folder.glob("__tmp_*.png"))
for f in tmp_files:
    num = f.stem.replace("__tmp_", "")
    f.rename(folder / f"{num}.png")

result = sorted(folder.glob("*.png"))
print(f"완료: {result[0].name} ~ {result[-1].name} ({len(result)}장)")
