"""엣지 브라우저 화면 자동 캡처 도구 (개선판)

개선 사항:
    - 루프 내 일반 예외 처리 (연속 오류 한도 초과 시에만 중단)
    - mss 캡처 핸들 부패 시 자동 재생성
    - 해시 충돌 오탐 방지 (인접 페이지 충돌 무시, 먼 과거 일치 시에만 중단)
    - 동일 화면 연속 스킵 한도 (MAX_SKIP 초과 시 마지막 페이지로 판단)
    - 창 소실 연속 한도 (MAX_WINDOW_LOST 초과 시 중단)

사용법:
    python tools/screen_capture_cl.py                     # Edge 창 3초 간격 캡처
    python tools/screen_capture_cl.py --interval 1        # 1초 간격
    python tools/screen_capture_cl.py --count 50          # 50장만 캡처
    python tools/screen_capture_cl.py --fullscreen        # 전체 화면 캡처
    python tools/screen_capture_cl.py --ocr               # OCR 텍스트 추출 포함
    python tools/screen_capture_cl.py --list-windows      # 캡처 가능한 창 목록
    python tools/screen_capture_cl.py --title "제목"      # 특정 창 제목으로 캡처

    # 📖 책 읽기 모드: 우측 > 버튼 랜덤 클릭 + 화면 변경 시 자동 캡처
    #   - 되돌아감 자동 감지 + 중단 / 50페이지마다 10초 휴식
    #   - 전역 단축키: Ctrl+Alt+Shift+B(시작) / Ctrl+Alt+Shift+E(종료)
    python tools/screen_capture_cl.py --book --title "교과서" --monitor 2 --count 290
    python tools/screen_capture_cl.py --book --title "교과서" --monitor 2 --count 290 --rest-interval 50 --rest-seconds 10

    # 🔍 캡처 폴더 중복 이미지 제거 (단독 실행)
    python tools/screen_capture_cl.py --dedup --output captures

    # 📋 캡처 결과 검증 (누락/이상치 확인)
    python tools/screen_capture_cl.py --verify --output captures --title "교과서" --expected 290
"""

import argparse
import ctypes
import random
import re
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

import mss
from PIL import Image

# pygetwindow는 Windows 전용
try:
    import pygetwindow as gw
except ImportError:
    gw = None

# 키보드 자동화
try:
    import pyautogui
    pyautogui.FAILSAFE = True  # 마우스 좌상단으로 이동 시 비상 정지
    pyautogui.PAUSE = 0.05     # 명령 사이 최소 대기
except ImportError:
    pyautogui = None

# 전역 단축키 (Ctrl+Alt+Shift+B=시작, Ctrl+Alt+Shift+E=종료)
try:
    import keyboard as _keyboard
except ImportError:
    _keyboard = None


def activate_window(win):
    """Windows API를 이용해 창을 확실하게 전면으로 가져옵니다."""
    try:
        hwnd = win._hWnd
    except AttributeError:
        return False

    user32 = ctypes.windll.user32

    # 최소화 상태이면 복원
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        time.sleep(0.3)

    # SetForegroundWindow 호출 (실패 방지: 현재 스레드를 대상 스레드에 연결)
    foreground_hwnd = user32.GetForegroundWindow()
    current_thread = user32.GetWindowThreadProcessId(foreground_hwnd, None)
    target_thread = user32.GetWindowThreadProcessId(hwnd, None)

    if current_thread != target_thread:
        user32.AttachThreadInput(current_thread, target_thread, True)
        user32.SetForegroundWindow(hwnd)
        user32.AttachThreadInput(current_thread, target_thread, False)
    else:
        user32.SetForegroundWindow(hwnd)

    time.sleep(0.3)
    return user32.GetForegroundWindow() == hwnd

# 이미지 해시 (중복 감지)
try:
    import imagehash
except ImportError:
    imagehash = None


def list_windows():
    """캡처 가능한 윈도우 목록을 출력합니다."""
    if gw is None:
        print("pygetwindow가 설치되지 않았습니다.")
        return

    windows = gw.getAllWindows()
    print(f"\n{'No':>3}  {'Title':<70} {'Size':>15}")
    print("-" * 92)
    for i, w in enumerate(windows, 1):
        if w.title.strip() and w.visible and w.width > 50 and w.height > 50:
            size = f"{w.width}x{w.height}"
            title = w.title[:68]
            print(f"{i:3}  {title:<70} {size:>15}")
    print()


def find_window(title_keyword: str = "Edge"):
    """키워드로 윈도우를 찾습니다."""
    if gw is None:
        return None

    windows = gw.getAllWindows()
    for w in windows:
        if (title_keyword.lower() in w.title.lower()
                and w.visible and w.width > 50 and w.height > 50):
            return w
    return None


def capture_window(win, sct) -> Image.Image:
    """특정 윈도우 영역을 캡처합니다 (멀티 모니터 지원)."""
    # monitors[0]은 전체 가상 화면 — 음수 좌표도 포함
    virt = sct.monitors[0]
    left = max(win.left, virt["left"])
    top = max(win.top, virt["top"])
    right = min(win.left + win.width, virt["left"] + virt["width"])
    bottom = min(win.top + win.height, virt["top"] + virt["height"])
    width = right - left
    height = bottom - top

    if width <= 0 or height <= 0:
        raise RuntimeError(f"캡처 영역이 유효하지 않습니다: {left},{top} {width}x{height}")

    region = {"left": left, "top": top, "width": width, "height": height}
    raw = sct.grab(region)
    return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")


def capture_fullscreen(sct, monitor: int = 1) -> Image.Image:
    """전체 화면을 캡처합니다.

    Args:
        monitor: 모니터 번호 (0=가상전체, 1=주모니터, 2+=보조모니터)
    """
    if monitor < 0 or monitor >= len(sct.monitors):
        raise ValueError(f"모니터 {monitor} 없음. 사용 가능: 0~{len(sct.monitors)-1}")
    raw = sct.grab(sct.monitors[monitor])
    return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")


def extract_text_ocr(img: Image.Image) -> str:
    """이미지에서 OCR로 텍스트를 추출합니다."""
    try:
        import pytesseract
        text = pytesseract.image_to_string(img, lang="kor+eng")
        return text.strip()
    except Exception as e:
        return f"[OCR 실패: {e}]"


def remove_duplicates(folder: str, threshold: int = 5):
    """이미지 해시를 이용해 중복/유사 이미지를 제거합니다.

    Args:
        folder: 이미지가 저장된 폴더 경로
        threshold: 해시 거리 임계값 (낮을수록 엄격, 기본 5)
    """
    if imagehash is None:
        print("imagehash 패키지가 필요합니다: pip install imagehash")
        return

    folder = Path(folder)
    images = sorted(folder.glob("*.png"))
    if not images:
        print(f"  '{folder}' 에 PNG 파일이 없습니다.")
        return

    print(f"\n  중복 검사 시작: {len(images)}장")
    print(f"  해시 거리 임계값: {threshold} (낮을수록 엄격)\n")

    # 해시 계산
    hashes = []
    for img_path in images:
        try:
            h = imagehash.phash(Image.open(img_path))
            hashes.append((img_path, h))
        except Exception as e:
            print(f"  [오류] {img_path.name}: {e}")

    # 중복 탐지 (앞쪽 이미지를 원본으로 유지)
    to_remove = set()
    for i in range(len(hashes)):
        if hashes[i][0] in to_remove:
            continue
        for j in range(i + 1, len(hashes)):
            if hashes[j][0] in to_remove:
                continue
            dist = hashes[i][1] - hashes[j][1]
            if dist <= threshold:
                to_remove.add(hashes[j][0])

    if not to_remove:
        print(f"  중복 없음. 전체 {len(hashes)}장 유지.")
        return

    print(f"  중복 감지: {len(to_remove)}장 삭제 예정\n")
    for p in sorted(to_remove):
        print(f"    삭제: {p.name}")
        p.unlink()

    kept = len(hashes) - len(to_remove)
    print(f"\n  완료: {len(to_remove)}장 삭제, {kept}장 유지.")


def verify_captures(folder: str, expected_count: int = 0, title_keyword: str = ""):
    """캡처 완료 후 결과를 검증하고 리포트를 출력합니다.

    - 총 캡처 수 vs 예상 수 비교
    - 시퀀스 번호 누락 감지
    - 파일 크기 이상치 탐지 (빈 페이지 / 로딩 화면 의심)
    - 시간 간격 이상치 탐지 (긴 공백 = 중단/재시작 흔적)
    """
    out = Path(folder)
    if not out.exists():
        print(f"  ❌ 폴더가 존재하지 않습니다: {out.resolve()}")
        return

    # book_*.png 파일 수집 (키워드 필터 적용)
    safe_kw = re.sub(r'[^\w]', '', title_keyword)[:10] if title_keyword else ""
    pattern = f"book_{safe_kw}_*.png" if safe_kw else "book_*.png"
    files = sorted(out.glob(pattern))

    if not files:
        print(f"  ❌ '{pattern}' 패턴에 해당하는 파일 없음: {out.resolve()}")
        return

    print(f"\n  ========== 📋 캡처 검증 리포트 ==========")
    print(f"  폴더      : {out.resolve()}")
    print(f"  파일 패턴  : {pattern}")
    print(f"  캡처 파일  : {len(files)}장")
    if expected_count > 0:
        diff = expected_count - len(files)
        status = "✅ 일치" if diff == 0 else f"⚠️  {abs(diff)}장 {'부족' if diff > 0 else '초과'}"
        print(f"  예상 페이지 : {expected_count}장 → {status}")

    # 시퀀스 번호 추출 및 누락 감지
    seq_numbers = []
    file_info = []  # (파일명, 시퀀스번호, 크기, 타임스탬프)
    seq_pattern = re.compile(r'_(\d{4})\.png$')
    ts_pattern = re.compile(r'_(\d{8}_\d{6})_')

    for f in files:
        m_seq = seq_pattern.search(f.name)
        m_ts = ts_pattern.search(f.name)
        seq = int(m_seq.group(1)) if m_seq else -1
        size_kb = f.stat().st_size / 1024
        timestamp = None
        if m_ts:
            try:
                timestamp = datetime.strptime(m_ts.group(1), "%Y%m%d_%H%M%S")
            except ValueError:
                pass
        seq_numbers.append(seq)
        file_info.append((f.name, seq, size_kb, timestamp))

    # 누락된 시퀀스 번호
    if seq_numbers and seq_numbers[0] > 0:
        full_range = set(range(seq_numbers[0], seq_numbers[-1] + 1))
        actual_set = set(seq_numbers)
        missing = sorted(full_range - actual_set)
        if missing:
            print(f"\n  🔍 누락된 시퀀스 번호 ({len(missing)}건):")
            # 연속 구간 묶어서 표시
            ranges = []
            start = missing[0]
            prev = missing[0]
            for m in missing[1:]:
                if m == prev + 1:
                    prev = m
                else:
                    ranges.append((start, prev))
                    start = m
                    prev = m
            ranges.append((start, prev))
            for s, e in ranges:
                if s == e:
                    print(f"     #{s}")
                else:
                    print(f"     #{s} ~ #{e} ({e - s + 1}장)")
        else:
            print(f"\n  ✅ 시퀀스 번호 연속: #{seq_numbers[0]} ~ #{seq_numbers[-1]} (누락 없음)")

    # 파일 크기 이상치 탐지
    sizes = [info[2] for info in file_info]
    if sizes:
        avg_size = sum(sizes) / len(sizes)
        min_threshold = avg_size * 0.15  # 평균의 15% 미만 → 의심
        anomalies = [(info[0], info[1], info[2]) for info in file_info if info[2] < min_threshold]
        if anomalies:
            print(f"\n  ⚠️  파일 크기 이상치 ({len(anomalies)}건, 평균 {avg_size:.0f}KB 대비 15% 미만):")
            for name, seq, size in anomalies:
                print(f"     #{seq:4d}  {name}  ({size:.1f}KB) ← 빈 페이지/로딩 화면 의심")
        else:
            print(f"\n  ✅ 파일 크기 정상 (평균 {avg_size:.0f}KB, 최소 {min(sizes):.0f}KB, 최대 {max(sizes):.0f}KB)")

    # 시간 간격 이상치 탐지
    timestamps = [(info[0], info[1], info[3]) for info in file_info if info[3] is not None]
    if len(timestamps) >= 2:
        time_gaps = []
        for i in range(1, len(timestamps)):
            gap = (timestamps[i][2] - timestamps[i-1][2]).total_seconds()
            time_gaps.append((timestamps[i][0], timestamps[i][1], gap))

        long_gaps = [(name, seq, gap) for name, seq, gap in time_gaps if gap > 30]
        if long_gaps:
            print(f"\n  ⏱️  긴 시간 간격 ({len(long_gaps)}건, 30초 초과):")
            for name, seq, gap in long_gaps:
                mins = int(gap // 60)
                secs = int(gap % 60)
                print(f"     #{seq:4d}  {name}  (이전 페이지와 {mins}분 {secs}초 간격)")

    # 요약
    print(f"\n  ──────── 요약 ────────")
    print(f"  총 캡처    : {len(files)}장")
    print(f"  시퀀스 범위 : #{seq_numbers[0]} ~ #{seq_numbers[-1]}")
    if expected_count > 0 and expected_count > len(files):
        print(f"  추가 필요  : {expected_count - len(files)}장")
    print(f"  =====================================\n")


def _compute_image_hash(img: Image.Image):
    """이미지의 perceptual hash를 계산합니다."""
    if imagehash is not None:
        return imagehash.phash(img)
    # imagehash 없으면 축소 후 bytes 비교용 해시
    import hashlib
    thumb = img.resize((64, 64)).convert("L")
    return hashlib.md5(thumb.tobytes()).hexdigest()


def _images_differ(hash1, hash2, threshold: int = 5) -> bool:
    """두 이미지 해시가 충분히 다른지 판단합니다."""
    if hash1 is None:
        return True
    if imagehash is not None:
        return (hash1 - hash2) > threshold
    return hash1 != hash2


def _create_mss():
    """mss 인스턴스를 생성합니다. 핸들 재생성 시 사용."""
    return mss.mss()


def _safe_capture(sct, win, monitor):
    """캡처를 시도하고, 실패 시 mss 핸들을 재생성하여 재시도합니다.

    Returns:
        (img, sct): 캡처된 이미지와 (갱신된) mss 인스턴스
    """
    try:
        if monitor > 0:
            img = capture_fullscreen(sct, monitor=monitor)
        else:
            img = capture_window(win, sct)
        return img, sct
    except Exception:
        # mss 핸들이 무효화된 경우 재생성
        print(f"  🔄 캡처 핸들 재생성 중...")
        try:
            sct.close()
        except Exception:
            pass
        sct = _create_mss()
        time.sleep(0.5)
        if monitor > 0:
            img = capture_fullscreen(sct, monitor=monitor)
        else:
            img = capture_window(win, sct)
        return img, sct


def run_book_capture(
    title_keyword: str = "Edge",
    click_min: float = 2.0,
    click_max: float = 3.0,
    max_count: int = 0,
    output_dir: str = "captures",
    dedup_threshold: int = 5,
    monitor: int = 0,
    rest_interval: int = 50,
    rest_seconds: float = 10.0,
):
    """책 읽기 모드: > 버튼 랜덤 클릭 + 화면 변경 감지 캡처

    - 우측 > 버튼을 랜덤 시간(click_min~click_max초) 간격으로 랜덤 위치에 클릭
    - 클릭 후 화면이 변경되었을 때만 캡처 (perceptual hash 비교)
    - 이미 캡처한 페이지와 동일한 이미지 감지 시 자동 중단 (되돌아감 방지)
    - rest_interval 페이지마다 rest_seconds초 휴식 (세션 보호)
    - monitor: 캡처할 모니터 번호 (0=창 영역, 1+=해당 모니터 전체)

    개선:
    - 루프 내 예외 처리 (연속 오류 MAX_ERRORS회 초과 시에만 중단)
    - mss 캡처 핸들 부패 시 자동 재생성
    - 해시 충돌 오탐 방지 (인접 페이지 충돌 무시)
    - 동일 화면 MAX_SKIP회 연속 스킵 시 마지막 페이지로 판단
    - 창 소실 MAX_WINDOW_LOST회 연속 시 중단
    """
    if pyautogui is None:
        print("pyautogui 패키지가 필요합니다: pip install pyautogui")
        return
    if gw is None:
        print("pygetwindow 패키지가 필요합니다: pip install pygetwindow")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 대상 창 찾기
    win = find_window(title_keyword)
    if win is None:
        print(f"  '{title_keyword}' 창을 찾을 수 없습니다.")
        return

    capture_mode = f"모니터 {monitor} 전체" if monitor > 0 else "창 영역"
    print(f"\n  ========== 📖 책 읽기 모드 (개선판) ==========")
    print(f"  대상 창   : {win.title[:60]}")
    print(f"  창 크기   : {win.width}x{win.height}")
    print(f"  > 클릭    : {click_min}~{click_max}초 랜덤 간격")
    print(f"  캡처 방식 : 화면 변경 감지 시 저장")
    print(f"  캡처 대상 : {capture_mode}")
    print(f"  최대 횟수 : {'무제한' if max_count <= 0 else max_count}")
    print(f"  휴식     : {rest_interval}페이지마다 {rest_seconds}초")
    print(f"  저장 경로 : {out.resolve()}")
    print(f"  시작 방법 : Ctrl+Alt+Shift+B")
    print(f"  종료 방법 : Ctrl+Alt+Shift+E 또는 Ctrl+C")
    print(f"  ================================================\n")
    print()

    # 전역 단축키 플래그
    start_event = threading.Event()
    stop_event = threading.Event()
    if _keyboard is not None:
        _keyboard.add_hotkey('ctrl+alt+shift+b', lambda: start_event.set())
        _keyboard.add_hotkey('ctrl+alt+shift+e', lambda: stop_event.set())
        print("  ⌨️  전역 단축키 등록 완료")
        print("       시작: Ctrl+Alt+Shift+B")
        print("       종료: Ctrl+Alt+Shift+E")
    else:
        # keyboard 패키지 없으면 5초 카운트다운 후 자동 시작
        print("  ⚠️  keyboard 패키지 없음 → 5초 후 자동 시작")
        for i in range(5, 0, -1):
            print(f"    {i}...")
            time.sleep(1)
        start_event.set()

    # 시작 단축키 대기
    if not start_event.is_set():
        print("\n  ⏳ Ctrl+Alt+Shift+B 를 눌러 시작하세요...")
        while not start_event.is_set():
            if stop_event.is_set():
                print("  ⏹️  시작 전 종료됨")
                if _keyboard is not None:
                    _keyboard.remove_all_hotkeys()
                return
            time.sleep(0.1)
        print("  ▶️  시작!")
    print()

    # 창 활성화 (Windows API 사용)
    activated = activate_window(win)
    if activated:
        print("  ✅ 창 활성화 성공")
    else:
        print("  ⚠️  창 활성화 실패 - 수동으로 책 화면을 클릭해주세요.")
        time.sleep(3)

    # 콘텐츠 영역 계산 (음수 좌표 허용 — 좌측 모니터)
    win_left = win.left
    win_top = win.top
    cx = win_left + win.width // 2
    cy = win_top + win.height // 2

    # 초기 클릭으로 포커스 확보
    pyautogui.click(cx, cy)
    time.sleep(0.5)
    print(f"  📌 콘텐츠 영역 클릭 ({cx}, {cy})")

    # 페이지 넘기기: 우측 > 버튼 기준 좌표 (약 97%, 50%)
    btn_center_x = win_left + int(win.width * 0.97)
    btn_center_y = win_top + int(win.height * 0.50)
    # 랜덤 클릭 범위 (버튼 주변 ±15px)
    click_radius = 15
    print(f"  📄 > 버튼 중심: ({btn_center_x}, {btn_center_y}), 반경 ±{click_radius}px\n")

    # 캡처 이력 해시 저장 (되돌아감 감지용)
    # hash_str → 최초 캡처된 페이지 번호 (해시 충돌 오탐 방지용)
    all_hashes = {}
    count = 0
    prev_hash = None
    skipped = 0
    MAX_SKIP = 30           # 연속 동일 화면 스킵 한도 (초과 시 마지막 페이지로 판단)
    consecutive_errors = 0  # 연속 오류 카운터
    MAX_ERRORS = 10         # 연속 오류 한도
    window_lost_count = 0   # 연속 창 소실 카운터
    MAX_WINDOW_LOST = 15    # 연속 창 소실 한도

    sct = _create_mss()
    try:
        # 초기 캡처 (현재 페이지)
        img, sct = _safe_capture(sct, win, monitor)
        prev_hash = _compute_image_hash(img)
        all_hashes[str(prev_hash)] = 0
        count += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_kw = re.sub(r'[^\w]', '', title_keyword)[:10]
        filename = f"book_{safe_kw}_{ts}_{count:04d}.png"
        filepath = out / filename
        img.save(str(filepath), "PNG", optimize=True)
        print(f"  [{ts}] #{count:4d}  {filepath.name}  ({img.width}x{img.height})  [첫 페이지]")

        while True:
            if 0 < max_count <= count:
                break

            # 전역 단축키 중단 감지
            if stop_event.is_set():
                print(f"\n  ⏹️  Ctrl+Alt+Shift+E 단축키로 중단됨")
                break

            # 휴식 삽입 (rest_interval 페이지마다)
            if rest_interval > 0 and count > 0 and count % rest_interval == 0:
                print(f"  ☕ {count}페이지 도달 → {rest_seconds}초 휴식 중...")
                time.sleep(rest_seconds)

            # 랜덤 대기 (click_min ~ click_max 초)
            wait = random.uniform(click_min, click_max)
            time.sleep(wait)

            # 창 위치 재확인 (활성화는 안 함)
            win = find_window(title_keyword)
            if win is None:
                window_lost_count += 1
                if window_lost_count >= MAX_WINDOW_LOST:
                    print(f"\n  ❌ 창을 {MAX_WINDOW_LOST}회 연속 찾을 수 없어 중단합니다.")
                    break
                print(f"  창을 찾을 수 없습니다. 대기 중... ({window_lost_count}/{MAX_WINDOW_LOST})")
                time.sleep(2)
                continue
            window_lost_count = 0

            # 버튼 좌표 재계산 (창 이동 대비)
            win_left = win.left
            win_top = win.top
            btn_center_x = win_left + int(win.width * 0.97)
            btn_center_y = win_top + int(win.height * 0.50)

            try:
                # 랜덤 위치 계산 → 마우스 이동 후 잠시 대기 → 클릭
                rx = btn_center_x + random.randint(-click_radius, click_radius)
                ry = btn_center_y + random.randint(-click_radius, click_radius)
                pyautogui.moveTo(rx, ry)
                time.sleep(0.3)
                pyautogui.click(rx, ry)

                # 페이지 전환 대기
                time.sleep(0.8)

                # 캡처 (mss 핸들 부패 시 자동 재생성)
                img, sct = _safe_capture(sct, win, monitor)

                # 화면 변경 감지
                curr_hash = _compute_image_hash(img)
                if not _images_differ(prev_hash, curr_hash, dedup_threshold):
                    skipped += 1
                    if skipped % 5 == 0:
                        print(f"  ... 동일 화면 {skipped}회 스킵")
                    if skipped >= MAX_SKIP:
                        print(f"\n  ⚠️  동일 화면 {MAX_SKIP}회 연속 → 마지막 페이지로 판단, 중단합니다.")
                        break
                    continue

                # 되돌아감 감지 (해시 충돌 오탐 방지 개선)
                # hash → 최초 페이지번호 매핑, 인접 페이지(2장 이내) 충돌은 무시
                curr_hash_str = str(curr_hash)
                if curr_hash_str in all_hashes:
                    prev_seen_at = all_hashes[curr_hash_str]
                    # 먼 과거 페이지(3장 이상 차이)와 일치 → 진짜 되돌아감
                    if count - prev_seen_at > 2:
                        print(f"\n  ⚠️  페이지 되돌아감 감지! (#{prev_seen_at + 1} 페이지와 일치)")
                        print(f"     {count}페이지까지 캡처 후 자동 중단합니다.")
                        break
                    else:
                        # 인접 페이지 해시 충돌 → 무시하고 계속 진행
                        print(f"  ⚠️  해시 유사 (#{prev_seen_at + 1}), 충돌 가능성 → 계속 진행")

                # 변경됨 → 저장
                prev_hash = curr_hash
                if curr_hash_str not in all_hashes:
                    all_hashes[curr_hash_str] = count
                skipped = 0
                consecutive_errors = 0
                count += 1
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_kw = re.sub(r'[^\w]', '', title_keyword)[:10]
                filename = f"book_{safe_kw}_{ts}_{count:04d}.png"
                filepath = out / filename
                img.save(str(filepath), "PNG", optimize=True)
                print(f"  [{ts}] #{count:4d}  {filepath.name}  ({img.width}x{img.height})  [{wait:.1f}s → ({rx},{ry})]")

            except KeyboardInterrupt:
                raise  # Ctrl+C는 상위로 전달
            except Exception as e:
                consecutive_errors += 1
                print(f"  ❗ 오류 ({consecutive_errors}/{MAX_ERRORS}): {e}")
                if consecutive_errors >= MAX_ERRORS:
                    print(f"\n  ❌ 연속 오류 {MAX_ERRORS}회 도달, 중단합니다.")
                    break
                time.sleep(1)
                continue

    except KeyboardInterrupt:
        print(f"\n  Ctrl+C 중단")
    finally:
        try:
            sct.close()
        except Exception:
            pass
        if _keyboard is not None:
            _keyboard.remove_all_hotkeys()

    print(f"\n  캡처 종료. 총 {count}장 저장됨 → {out.resolve()}")

    # 캡처 완료 후 자동 검증
    verify_captures(output_dir, expected_count=max_count, title_keyword=title_keyword)


def run_capture(
    title_keyword: str = "Edge",
    interval: float = 3.0,
    max_count: int = 0,
    output_dir: str = "captures",
    fullscreen: bool = False,
    ocr: bool = False,
    quality: int = 90,
    monitor: int = 1,
):
    """주기적 자동 캡처를 실행합니다."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if ocr:
        ocr_dir = out / "ocr_text"
        ocr_dir.mkdir(exist_ok=True)

    mon_label = f'전체 화면 (모니터 {monitor})' if fullscreen else f'창 제목 포함: {title_keyword!r}'
    print(f"  캡처 대상 : {mon_label}")
    print(f"  캡처 간격 : {interval}초")
    print(f"  최대 횟수 : {'무제한' if max_count <= 0 else max_count}")
    print(f"  저장 경로 : {out.resolve()}")
    print(f"  OCR      : {'ON (kor+eng)' if ocr else 'OFF'}")
    print(f"  종료 방법 : Ctrl+C\n")

    count = 0
    sct = _create_mss()
    try:
        while True:
            if 0 < max_count <= count:
                print(f"\n총 {count}장 캡처 완료.")
                break

            try:
                # 이미지 캡처
                if fullscreen:
                    img, sct = _safe_capture(sct, None, monitor)
                else:
                    win = find_window(title_keyword)
                    if win is None:
                        print(f"  [{datetime.now():%H:%M:%S}] '{title_keyword}' 창을 찾을 수 없습니다. 대기 중...")
                        time.sleep(interval)
                        continue
                    img, sct = _safe_capture(sct, win, monitor=0)

                # 파일 저장
                count += 1
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_kw = re.sub(r'[^\w]', '', title_keyword)[:10]
                filename = f"{safe_kw}_{ts}_{count:04d}.png"
                filepath = out / filename
                img.save(str(filepath), "PNG", optimize=True)
                print(f"  [{ts}] #{count:4d}  {filepath.name}  ({img.width}x{img.height})")

                # OCR 처리
                if ocr:
                    text = extract_text_ocr(img)
                    if text:
                        txt_path = ocr_dir / f"{filepath.stem}.txt"
                        txt_path.write_text(text, encoding="utf-8")
                        # 미리보기 (첫 80자)
                        preview = text[:80].replace("\n", " ")
                        print(f"           OCR: {preview}...")

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"  ❗ 캡처 오류: {e}")

            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n\n캡처 종료. 총 {count}장 저장됨 → {out.resolve()}")
    finally:
        try:
            sct.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="엣지 브라우저(또는 특정 창) 자동 주기 캡처 도구 (개선판)"
    )
    parser.add_argument("--title", "-t", default="Edge",
                        help="캡처할 윈도우 제목 키워드 (기본: Edge)")
    parser.add_argument("--interval", "-i", type=float, default=3.0,
                        help="캡처 간격 (초, 기본: 3)")
    parser.add_argument("--count", "-n", type=int, default=0,
                        help="최대 캡처 횟수 (0=무제한, 기본: 0)")
    parser.add_argument("--output", "-o", default="captures",
                        help="저장 폴더 (기본: captures)")
    parser.add_argument("--fullscreen", "-f", action="store_true",
                        help="전체 화면 캡처")
    parser.add_argument("--monitor", "-m", type=int, default=1,
                        help="캡처할 모니터 번호 (0=가상전체, 1=주모니터, 2+=보조, 기본: 1)")
    parser.add_argument("--ocr", action="store_true",
                        help="OCR 텍스트 추출 (Tesseract 필요)")
    parser.add_argument("--quality", "-q", type=int, default=90,
                        help="이미지 품질 (기본: 90)")
    parser.add_argument("--list-windows", "-l", action="store_true",
                        help="캡처 가능한 창 목록 표시")

    # 📖 책 읽기 모드
    parser.add_argument("--book", action="store_true",
                        help="책 읽기 모드: > 버튼 랜덤 클릭 + 화면 변경 감지 캡처")
    parser.add_argument('--click-min', type=float, default=2.0,
                        help="> 버튼 클릭 최소 간격 (초, 기본: 2.0)")
    parser.add_argument('--click-max', type=float, default=3.0,
                        help="> 버튼 클릭 최대 간격 (초, 기본: 3.0)")
    parser.add_argument("--dedup-threshold", type=int, default=5,
                        help="중복 판정 해시 거리 임계값 (기본: 5, 낮을수록 엄격)")
    parser.add_argument("--rest-interval", type=int, default=50,
                        help="휴식 삽입 간격 - N페이지마다 (기본: 50, 0=비활성화)")
    parser.add_argument("--rest-seconds", type=float, default=10.0,
                        help="휴식 시간 (초, 기본: 10)")

    # 🔍 중복 제거 단독 실행
    parser.add_argument("--dedup", action="store_true",
                        help="저장 폴더의 중복 이미지만 제거 (단독 실행)")

    # 📋 캡처 검증 단독 실행
    parser.add_argument("--verify", action="store_true",
                        help="캡처 결과 검증 (누락/이상치 확인, 단독 실행)")
    parser.add_argument("--expected", type=int, default=0,
                        help="예상 페이지 수 (검증 시 비교용, 기본: 0=비교 안 함)")

    args = parser.parse_args()

    if args.list_windows:
        list_windows()
        return

    if args.dedup:
        remove_duplicates(args.output, threshold=args.dedup_threshold)
        return

    if args.verify:
        verify_captures(args.output, expected_count=args.expected, title_keyword=args.title)
        return

    if args.book:
        run_book_capture(
            title_keyword=args.title,
            click_min=args.click_min,
            click_max=args.click_max,
            max_count=args.count,
            output_dir=args.output,
            dedup_threshold=args.dedup_threshold,
            monitor=args.monitor,
            rest_interval=args.rest_interval,
            rest_seconds=args.rest_seconds,
        )
        return

    run_capture(
        title_keyword=args.title,
        interval=args.interval,
        max_count=args.count,
        output_dir=args.output,
        fullscreen=args.fullscreen,
        ocr=args.ocr,
        quality=args.quality,
        monitor=args.monitor,
    )


if __name__ == "__main__":
    main()
