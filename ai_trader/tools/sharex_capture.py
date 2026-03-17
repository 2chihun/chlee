"""
ShareX + Python 자동 책 페이지 캡처 도구
=========================================
ShareX의 고품질 캡처 기능과 Python 자동화를 결합한 도구입니다.

사전 준비:
1. ShareX에서 캡처 단축키 설정:
   - Hotkey settings → Add → Capture → Capture active window (or Screen)
   - 단축키: Ctrl+Alt+C (또는 원하는 키)
   - After capture → Save to folder: C:\Copilot\captures\sharex\
   - File naming: %y%mo%d_%h%mi%s%ms (타임스탬프 기반)

2. 책 뷰어를 좌측 모니터에 열고 첫 페이지로 이동

사용법:
  python sharex_capture.py [옵션]

옵션:
  --total    총 페이지 수 (기본값: 290)
  --delay    페이지 전환 대기시간 초 (기본값: 0.8)
  --hotkey   ShareX 캡처 단축키 (기본값: ctrl+alt+c)
  --output   저장 폴더 (기본값: C:\Copilot\captures\sharex)
  --start    시작 페이지 번호 (기본값: 1)
  --nav      페이지 넘김 키: pagedown/right/space (기본값: pagedown)
  --window   캡처 대상 창 이름 일부 (기본값: 없음, 활성창 캡처)
"""

import argparse
import time
import os
import sys
import glob
import shutil
from datetime import datetime

try:
    import pyautogui
    import pygetwindow as gw
    from PIL import Image
except ImportError:
    print("필요한 패키지 설치: pip install pyautogui pygetwindow pillow")
    sys.exit(1)

# 기본 설정
DEFAULT_CONFIG = {
    "total": 290,
    "delay": 0.8,
    "hotkey": "ctrl+alt+c",
    "output": r"C:\Copilot\captures\sharex",
    "start": 1,
    "nav": "pagedown",
    "window": "",
}


def activate_window(window_name: str) -> bool:
    """창 이름으로 창을 활성화"""
    if not window_name:
        return True

    windows = gw.getWindowsWithTitle(window_name)
    if not windows:
        print(f"  ⚠ 창을 찾을 수 없음: '{window_name}'")
        print(f"  열려있는 창 목록:")
        for w in gw.getAllWindows():
            if w.title:
                print(f"    - {w.title}")
        return False

    win = windows[0]
    win.activate()
    time.sleep(0.3)
    return True


def capture_page(hotkey: str, nav_key: str, delay: float) -> None:
    """ShareX 단축키로 캡처 후 페이지 넘김"""
    # ShareX 캡처 트리거
    pyautogui.hotkey(*hotkey.split("+"))
    time.sleep(0.2)

    # 페이지 넘김
    pyautogui.press(nav_key)

    # 다음 페이지 로딩 대기
    time.sleep(delay)


def rename_captured_files(output_dir: str, start_page: int, count: int) -> list:
    """ShareX가 저장한 파일을 순번 형식으로 변경"""
    # 최근 저장된 파일 찾기 (PNG/JPG)
    patterns = [
        os.path.join(output_dir, "*.png"),
        os.path.join(output_dir, "*.jpg"),
        os.path.join(output_dir, "*.bmp"),
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))

    # 수정시간 기준 정렬
    files.sort(key=os.path.getmtime)

    # 최근 count개만 선택
    recent_files = files[-count:] if len(files) >= count else files

    renamed = []
    for i, fpath in enumerate(recent_files):
        page_num = start_page + i
        ext = os.path.splitext(fpath)[1]
        new_name = f"{page_num:03d}{ext}"
        new_path = os.path.join(output_dir, new_name)

        if fpath != new_path:
            os.rename(fpath, new_path)
        renamed.append(new_path)

    return renamed


def get_latest_file(output_dir: str, before_time: float) -> str:
    """캡처 후 새로 생성된 파일 찾기"""
    patterns = ["*.png", "*.jpg", "*.bmp"]
    newest = None
    newest_time = before_time

    for pat in patterns:
        for fpath in glob.glob(os.path.join(output_dir, pat)):
            mtime = os.path.getmtime(fpath)
            if mtime > newest_time:
                newest_time = mtime
                newest = fpath

    return newest


def run_capture(config: dict) -> None:
    """메인 캡처 루프"""
    total = config["total"]
    delay = config["delay"]
    hotkey = config["hotkey"]
    output = config["output"]
    start = config["start"]
    nav = config["nav"]
    window_name = config["window"]

    # 출력 폴더 생성
    os.makedirs(output, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  ShareX 자동 캡처 시작")
    print(f"{'='*50}")
    print(f"  총 페이지   : {total}")
    print(f"  시작 페이지 : {start}")
    print(f"  캡처 단축키 : {hotkey}")
    print(f"  페이지 넘김 : {nav}")
    print(f"  대기 시간   : {delay}초")
    print(f"  저장 폴더   : {output}")
    print(f"{'='*50}")

    if window_name:
        print(f"\n  대상 창 활성화: {window_name}")
        if not activate_window(window_name):
            print("  ❌ 창 활성화 실패. 수동으로 창을 클릭 후 Enter를 누르세요.")
            input()

    print(f"\n  5초 후 캡처 시작... (Ctrl+C로 취소)")
    for i in range(5, 0, -1):
        print(f"  {i}...", end=" ", flush=True)
        time.sleep(1)
    print()

    # 캡처 루프
    captured = 0
    errors = 0

    try:
        for page in range(start, total + 1):
            capture_num = page - start + 1
            before_time = time.time()

            # 캡처 실행
            capture_page(hotkey, nav, delay)

            # 새 파일 확인
            new_file = get_latest_file(output, before_time - 0.1)

            if new_file:
                # 순번으로 이름 변경
                ext = os.path.splitext(new_file)[1]
                new_name = os.path.join(output, f"{page:03d}{ext}")
                if new_file != new_name and not os.path.exists(new_name):
                    os.rename(new_file, new_name)
                captured += 1
                print(f"\r  진행: {page}/{total} 페이지 ({captured}개 캡처) ", end="", flush=True)
            else:
                errors += 1
                print(f"\n  ⚠ {page}페이지 캡처 파일 미확인 (ShareX 저장 경로 확인 필요)")

    except KeyboardInterrupt:
        print(f"\n\n  ⏹ 사용자 중단: {page-1}/{total} 완료")

    print(f"\n\n{'='*50}")
    print(f"  캡처 완료!")
    print(f"  성공: {captured}개 / 오류: {errors}개")
    print(f"  저장 위치: {output}")
    print(f"{'='*50}\n")


def check_sharex_setup(output_dir: str) -> None:
    """ShareX 설정 확인 가이드 출력"""
    print(f"""
ShareX 설정 확인 사항:
{'─'*45}
1. Hotkey settings 설정:
   - Hotkey: Ctrl+Alt+C
   - Task: Capture active window (또는 Capture)

2. After capture task:
   - ☑ Save to folder: {output_dir}
   - ☑ File naming: %y%mo%d_%h%mi%s%ms
   - ☐ Open in image editor (체크 해제)
   - ☐ Show "after capture" window (체크 해제)

3. 저장 폴더 확인: {output_dir}
{'─'*45}
설정이 완료되면 Enter를 눌러 테스트 캡처를 진행하세요.
""")
    input("Enter를 누르면 테스트 캡처 (1회) 실행...")

    os.makedirs(output_dir, exist_ok=True)
    before = time.time()
    pyautogui.hotkey("ctrl", "alt", "c")
    time.sleep(1.5)

    test_file = get_latest_file(output_dir, before - 0.1)
    if test_file:
        print(f"  ✅ 테스트 성공! 저장된 파일: {os.path.basename(test_file)}")
        # 테스트 파일 삭제
        os.remove(test_file)
    else:
        print(f"  ❌ 파일이 저장되지 않았습니다.")
        print(f"     ShareX 설정을 다시 확인하세요.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="ShareX + Python 자동 책 캡처 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--total",  type=int,   default=DEFAULT_CONFIG["total"],   help=f"총 페이지 수 (기본: {DEFAULT_CONFIG['total']})")
    parser.add_argument("--delay",  type=float, default=DEFAULT_CONFIG["delay"],   help=f"페이지 전환 대기(초) (기본: {DEFAULT_CONFIG['delay']})")
    parser.add_argument("--hotkey", type=str,   default=DEFAULT_CONFIG["hotkey"],  help=f"ShareX 단축키 (기본: {DEFAULT_CONFIG['hotkey']})")
    parser.add_argument("--output", type=str,   default=DEFAULT_CONFIG["output"],  help=f"저장 폴더 (기본: {DEFAULT_CONFIG['output']})")
    parser.add_argument("--start",  type=int,   default=DEFAULT_CONFIG["start"],   help=f"시작 페이지 (기본: {DEFAULT_CONFIG['start']})")
    parser.add_argument("--nav",    type=str,   default=DEFAULT_CONFIG["nav"],     help=f"페이지 넘김 키 (기본: {DEFAULT_CONFIG['nav']})")
    parser.add_argument("--window", type=str,   default=DEFAULT_CONFIG["window"],  help="캡처 창 이름 (부분 일치)")
    parser.add_argument("--setup",  action="store_true", help="ShareX 설정 확인 및 테스트")

    args = parser.parse_args()

    config = {
        "total":  args.total,
        "delay":  args.delay,
        "hotkey": args.hotkey,
        "output": args.output,
        "start":  args.start,
        "nav":    args.nav,
        "window": args.window,
    }

    if args.setup:
        check_sharex_setup(args.output)

    run_capture(config)


if __name__ == "__main__":
    main()
