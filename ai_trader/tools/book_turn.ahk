#Requires AutoHotkey v2.0
; ============================================
; 책 페이지 자동 넘김 (ShareX 자동촬영과 연동)
; ShareX가 캡처 담당, AHK는 페이지 넘김만 담당
; ============================================
; 사용법:
;   1. ShareX 자동촬영 설정 (간격: 2500ms, 좌측모니터 영역)
;   2. 책 뷰어를 첫 페이지로 이동 후 클릭(포커스)
;   3. 이 스크립트 더블클릭 실행
;   4. F9 → ShareX 자동촬영 시작 + 페이지 넘김 동시 시작
;   5. F9 → 일시정지 / F10 → 종료
; ============================================

; ── 설정 ─────────────────────────────────
TOTAL_PAGES  := 260       ; 총 페이지 수
CAPTURE_WAIT := 1500      ; 캡처 후 페이지 넘기기까지 대기(ms)
NAV_WAIT     := 1000      ; 페이지 넘긴 후 다음 사이클까지 대기(ms)
                          ; 전체 사이클 = CAPTURE_WAIT + NAV_WAIT = 2.5초
BOOK_TITLE   := ""        ; 책 뷰어 창 제목 (비워두면 현재 활성창)
NAV_KEY      := "{Right}" ; 페이지 넘김 키
SHAREX_KEY   := "^+!s"   ; ShareX 자동촬영 단축키 (Ctrl+Shift+Alt+S)
; ─────────────────────────────────────────

global isRunning := false
global pageCount := 0

ToolTip("📖 준비완료  F9: 시작  F10: 종료")
SetTimer(() => ToolTip(), 3000)

; ── F9: 시작 / 일시정지 ──────────────────
F9:: {
    global isRunning, pageCount

    isRunning := !isRunning

    if isRunning {
        ; ShareX 자동촬영 시작
        Send(SHAREX_KEY)
        Sleep(300)
        ToolTip("▶ 시작! (0/" TOTAL_PAGES ")`n캡처→1.5초→넘김→1초→반복`nF9:일시정지  F10:종료")
        ; 1.5초 후 첫 페이지 넘김 (첫 캡처 완료 대기)
        SetTimer(TurnPage, CAPTURE_WAIT)
    } else {
        SetTimer(TurnPage, 0)
        Send(SHAREX_KEY)   ; ShareX 자동촬영 정지
        ToolTip("⏸ 일시정지 (" pageCount "/" TOTAL_PAGES ")`nF9:재개  F10:종료")
    }
}

; ── F10: 완전 종료 ────────────────────────
F10:: {
    global isRunning
    isRunning := false
    SetTimer(TurnPage, 0)
    Send(SHAREX_KEY)   ; ShareX 자동촬영 정지
    ToolTip("⏹ 종료 - " pageCount "/" TOTAL_PAGES " 페이지 완료")
    Sleep(3000)
    ToolTip()
    ExitApp()
}

; ── 페이지 넘김 루프 ─────────────────────
TurnPage() {
    global pageCount, isRunning

    if !isRunning
        return

    ; 타이머 중지 (수동 제어)
    SetTimer(TurnPage, 0)

    pageCount++

    ; 책 창 활성화
    if (BOOK_TITLE != "") && WinExist(BOOK_TITLE)
        WinActivate(BOOK_TITLE)

    ; 페이지 넘김
    Send(NAV_KEY)
    ToolTip("📖 " pageCount "/" TOTAL_PAGES " 페이지 넘김`nF9:일시정지  F10:종료")

    ; 완료 체크
    if pageCount >= TOTAL_PAGES {
        Sleep(500)
        Send(SHAREX_KEY)   ; ShareX 자동촬영 정지
        ToolTip("✅ " TOTAL_PAGES "페이지 완료!")
        Sleep(5000)
        ToolTip()
        ExitApp()
        return
    }

    ; 다음 사이클 예약
    if isRunning
        SetTimer(TurnPage, NAV_WAIT + CAPTURE_WAIT)
}
