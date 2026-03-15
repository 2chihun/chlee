"""캡처 이미지의 중복/반복 페이지를 감지하고 삭제하는 스크립트

전체 이미지의 perceptual hash를 비교하여
동일 페이지가 반복 캡처된 경우를 감지합니다.
"""
import sys
from pathlib import Path
from PIL import Image
import imagehash


def analyze_and_remove(folder: str, pattern: str = "book_교과서_*.png",
                       threshold: int = 0, dry_run: bool = False):
    folder = Path(folder)
    images = sorted(folder.glob(pattern))
    print(f"총 {len(images)}장 분석 중...\n")

    # 해시 계산 (전체 이미지, 고해상도 해시)
    hashes = []
    for img_path in images:
        img = Image.open(img_path)
        h = imagehash.phash(img, hash_size=16)
        hashes.append((img_path, h))

    # 순서대로 스캔: 이미 본 해시와 유사하면 중복
    seen = []  # (hash, path)
    to_remove = []

    for path, h in hashes:
        is_dup = False
        for seen_h, seen_path in seen:
            dist = h - seen_h
            if dist <= threshold:
                to_remove.append((path, seen_path, dist))
                is_dup = True
                break
        if not is_dup:
            seen.append((h, path))

    print(f"고유 페이지: {len(seen)}장")
    print(f"중복/반복 : {len(to_remove)}장\n")

    if to_remove:
        print("삭제 대상:")
        for dup_path, orig_path, dist in to_remove:
            dup_idx = int(dup_path.stem.split('_')[-1])
            orig_idx = int(orig_path.stem.split('_')[-1])
            print(f"  {dup_path.name}  (#{dup_idx}) ← 중복 of #{orig_idx}, dist={dist}")

        if not dry_run:
            print(f"\n{len(to_remove)}장 삭제 중...")
            for dup_path, _, _ in to_remove:
                dup_path.unlink()
            print("삭제 완료!")

            # 남은 파일 순번 재정렬
            remaining = sorted(folder.glob(pattern))
            print(f"\n남은 파일: {len(remaining)}장")
            print(f"  첫째: {remaining[0].name}")
            print(f"  마지막: {remaining[-1].name}")
        else:
            print("\n[DRY RUN] 실제 삭제하지 않았습니다.")
    else:
        print("중복 없음!")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "captures"
    dry_run = "--dry-run" in sys.argv
    analyze_and_remove(folder, dry_run=dry_run)
