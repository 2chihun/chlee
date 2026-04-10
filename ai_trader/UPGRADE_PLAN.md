# AI Trader 도서 인사이트 통합 업그레이드 계획

> 생성일: 2026-03-22
> 목적: 9개 투자 도서 인사이트를 ai_trader 프로젝트에 완전 통합
> 인사이트 파일 위치: C:\Copilot\captures\- 텍스트 변환\*_insight.md

## 현재 상태 요약

### 이미 구현된 모듈 (8개 도서)
| 도서 | 모듈 | 크기 | 상태 |
|------|------|------|------|
| 서준식 | features/deep_value.py | 15K | ✅ 완료 |
| 이남우 | features/stock_quality.py | 11K | ✅ 완료 |
| 강방천&존리 | features/value_investor.py | 15K | ✅ 완료 |
| 잭 슈웨거 | features/wizard_discipline.py | 29K | ✅ 완료 |
| 켄 피셔 | features/market_memory.py | 35K | ✅ 완료 |
| 하워드 막스 | features/market_cycle.py + credit_cycle.py | 34K+23K | ✅ 완료 |
| 박병창 | features/candle_patterns.py + market_flow.py | 26K+22K | ⚠️ 부분 |
| 캔들마스터 | features/wave_position.py | 26K | ✅ 완료 |

### 미구현 항목
| # | 항목 | 소스 도서 | 우선순위 |
|---|------|----------|---------|
| 1 | features/bubble_detector.py | 조상철 (2026 대폭락) | HIGH |
| 2 | features/exchange_rate.py | 백석현 (환율) | HIGH |
| 3 | features/execution_analysis.py | 박병창 (보완) | MEDIUM |
| 4 | features/book_integrator.py | 전체 통합 | HIGH |
| 5 | config/settings.py 확장 | 조상철+백석현 Config | HIGH |
| 6 | strategies/swing.py 통합 | 새 모듈 연동 | HIGH |
| 7 | risk/manager.py 보강 | 버블/환율 리스크 | HIGH |
| 8 | tests/test_core.py 확장 | 새 모듈 테스트 | MEDIUM |

---

## 단계별 실행 계획 (8단계)

### Phase 1: BubbleDetectorConfig + BubbleDetector 모듈 ✅ 완료
- **파일**: config/settings.py (BubbleDetectorConfig 추가)
- **파일**: features/bubble_detector.py (신규)
- **소스**: 조상철 인사이트 - AI 버블 감지, 마진콜 시뮬레이션
- **클래스**:
  - `BubbleDetectorConfig` dataclass
  - `BubblePhase` enum: NORMAL / EUPHORIA / PEAK / BURST / PANIC
  - `BubbleSignal` dataclass: phase, bubble_score(0-100), leverage_risk, cascade_risk, position_multiplier
  - `AIBubbleDetector`: CAPEX vs 실제 성과 GAP 감지 (기술적 프록시)
  - `CascadeRiskAnalyzer`: 연쇄 마진콜/강제청산 리스크
  - `SectorLeverageMonitor`: 섹터별 레버리지 집중도
  - `PassiveFundCascadeEstimator`: ETF/패시브 기계적 매도 규모
- **임계값**: CAPEX ROI Gap < 0.5, LTV 50%, 만기집중 경보
- **테스트**: 5개 유닛 테스트

### Phase 2: ExchangeRateConfig + ExchangeRate 모듈 ✅ 완료
- **파일**: config/settings.py (ExchangeRateConfig 추가)
- **파일**: features/exchange_rate.py (신규)
- **소스**: 백석현 인사이트 - 환율 분석, 달러 사이클
- **클래스**:
  - `ExchangeRateConfig` dataclass
  - `DollarCyclePhase` enum: STRONG_EARLY / STRONG_LATE / WEAK_EARLY / WEAK_LATE
  - `ExchangeRateSignal` dataclass: dollar_phase, fx_alarm, risk_barometer, kospi_correlation, position_multiplier
  - `ExchangeRateAnalyzer`: 달러 사이클 판단 (DXY 프록시)
  - `DollarShieldManager`: 달러 방패 비중 관리 (15-25%)
  - `RiskBarometer`: AUD/JPY 글로벌 위험선호도 지표
  - `BiasCorrector`: 부정성 편향/최신 효과/레버리지 가드
- **임계값**: 환율 ±1% 경보, 달러-원 상관 -0.59, 레버리지 최대 1.5x
- **테스트**: 5개 유닛 테스트

### Phase 3: ExecutionAnalysis 보완 모듈 ✅ 완료
- **파일**: features/execution_analysis.py (신규)
- **소스**: 박병창 인사이트 보완 - CANSLIM, PEG, 시간대 전략
- **클래스**:
  - `CANSLIMScreener`: 7요소 스크리닝 (기술적 프록시)
  - `PEGCalculator`: PEG = PER / 이익증가율
  - `TimeBasedStrategy`: 강세장 오전/약세장 오후 최적 시간대
  - `CandleForceAnalyzer`: 장대봉 50% 기준선
- **임계값**: 체결강도 200(극강), PEG < 1.0(저평가), 거래량비 50%
- **테스트**: 4개 유닛 테스트

### Phase 4: BookIntegrator 통합 모듈 ✅ 완료
- **파일**: features/book_integrator.py (신규)
- **소스**: 9개 도서 전체 통합
- **클래스**:
  - `BookIntegrator`: 모든 도서 모듈의 시그널을 가중 합산
  - `IntegratedSignal` dataclass: 통합 점수, 각 모듈별 개별 시그널, 최종 행동 권고
  - `ConflictResolver`: 상충하는 시그널 해소 로직
  - `SignalWeightManager`: 시장 상황별 동적 가중치 조절
- **가중치 기본값**:
  - 하워드 막스 (사이클): 20%
  - 백석현 (환율): 15%
  - 조상철 (버블감지): 15%
  - 잭 슈웨거 (규율): 12%
  - 켄 피셔 (시장기억): 10%
  - 서준식 (딥밸류): 8%
  - 이남우 (품질): 8%
  - 박병창 (집행): 7%
  - 강방천&존리 (가치): 5%
- **테스트**: 3개 유닛 테스트

### Phase 5: config/settings.py 확장 ✅ 완료
- **변경**: BubbleDetectorConfig, ExchangeRateConfig, BookIntegratorConfig 추가
- **변경**: AppConfig에 bubble_detector, exchange_rate, book_integrator 필드 추가
- **변경**: load_config()에 환경변수 로드 추가

### Phase 6: strategies/swing.py 통합 ✅ 완료
- **변경**: BubbleDetector, ExchangeRate, ExecutionAnalysis, BookIntegrator import
- **변경**: generate_signal()에 새 모듈 필터 적용
  - 버블 PEAK/BURST 시 매수 차단
  - 환율 급변 경보 시 포지션 축소
  - CANSLIM 점수 반영
  - BookIntegrator 통합 점수로 최종 confidence 조정

### Phase 7: risk/manager.py 보강 ✅ 완료
- **변경**: BubbleDetector, ExchangeRate 리스크 연동
  - 버블 점수 70+ 시 max_position 40% 축소
  - 환율 급변 시 포지션 30% 축소
  - 레버리지 상한 1.5x 강제 적용

### Phase 8: tests/test_core.py 확장 ✅ 완료
- **추가**: TestBubbleDetector (5개 테스트)
- **추가**: TestExchangeRate (5개 테스트)
- **추가**: TestExecutionAnalysis (4개 테스트)
- **추가**: TestBookIntegrator (3개 테스트)
- **총**: 기존 21개 + 신규 17개 = 38개 테스트

---

## 진행 상태 추적

| Phase | 설명 | 상태 | 완료일 | 비고 |
|-------|------|------|-------|------|
| 1 | BubbleDetector 모듈 | ✅ 완료 | 2026-03-22 | features/bubble_detector.py (신규) |
| 2 | ExchangeRate 모듈 | ✅ 완료 | 2026-03-22 | features/exchange_rate.py (신규) |
| 3 | ExecutionAnalysis 모듈 | ✅ 완료 | 2026-03-22 | features/execution_analysis.py (신규) |
| 4 | BookIntegrator 모듈 | ✅ 완료 | 2026-03-22 | features/book_integrator.py (신규) |
| 5 | config/settings.py 확장 | ✅ 완료 | 2026-03-22 | +3 Config dataclass |
| 6 | strategies/swing.py 통합 | ✅ 완료 | 2026-03-22 | 4개 모듈 연동 |
| 7 | risk/manager.py 보강 | ✅ 완료 | 2026-03-22 | 버블/환율/레버리지 가드 |
| 8 | tests/test_core.py 확장 | ✅ 완료 | 2026-03-22 | 21→38개 (전체 PASS) |

## ✅ Phase 1~8 완료: 2026-03-22
- 전체 38개 테스트 통과 (6.71초)
- 신규 모듈 4개 + 설정 3개 + 전략/리스크 통합 완료

---

## Phase 9: 홍용찬 "실전 퀀트투자" 통합 ✅ 완료

### Phase 9-1: QuantValueConfig + QuantValueAnalyzer 모듈 ✅ 완료
- **파일**: config/settings.py (QuantValueConfig 추가)
- **파일**: features/quant_value.py (신규, ~350줄)
- **소스**: 홍용찬 인사이트 - PER/PBR/PSR/PCR, ROE/ROA, 성장성, 안전성, 배당+흑자, 5-2 모멘텀, 캘린더 효과
- **클래스**:
  - `QuantValueConfig` dataclass (20개 파라미터)
  - `QuantValueSignal` dataclass: value/profitability/growth/safety/dividend/momentum/calendar/composite
  - `ValuePercentileScorer`: PER/PBR/PSR/PCR 기술적 프록시
  - `ProfitabilityScorer`: ROE/ROA/영업이익률 기술적 프록시
  - `GrowthAnalyzer`: 매출/영업이익 성장 4유형 분류
  - `SafetyAnalyzer`: 부채비율/NCAV 기술적 프록시
  - `DividendProfitabilityScorer`: 배당+흑자 복합 필터
  - `MomentumScorer`: 5-2 모멘텀 전략
  - `CalendarEffectAdjuster`: 월말월초/수요일/1월 효과
  - `QuantValueAnalyzer`: 전체 통합 (가치40%+수익성20%+성장15%+안전15%+배당10%)

### Phase 9-2: strategies/swing.py 통합 ✅ 완료
- quant_value import + _HAS_QUANT_VALUE 플래그
- __init__에 QuantValueAnalyzer 초기화
- analyze()에 qv_composite/value/profit/growth/safety/momentum/calendar/deep_value 컬럼 추가

### Phase 9-3: tests/test_core.py 확장 ✅ 완료
- TestQuantValue (4개 테스트): basic, insufficient_data, individual_scorers, deep_value_flag
- **총**: 38개 → **42개** 테스트 (전체 PASS, 6.72초)

### Phase 9-4: 인사이트 파일 ✅ 완료
- `C:\Copilot\captures\- 텍스트 변환\홍용찬-실전_퀀트투자_insight.md` 생성

| Phase | 설명 | 상태 | 완료일 |
|-------|------|------|-------|
| 9-1 | QuantValue 모듈 | ✅ 완료 | 2026-03-25 |
| 9-2 | swing.py 통합 | ✅ 완료 | 2026-03-25 |
| 9-3 | 테스트 확장 (42개) | ✅ 완료 | 2026-03-25 |
| 9-4 | 인사이트 파일 | ✅ 완료 | 2026-03-25 |

## 재개 시 참고사항
- Screenshots 폴더에 9권 추가 도서 이미지 캡처 완료 (OCR 처리 중)
- 이 파일(`UPGRADE_PLAN.md`)의 진행 상태 테이블을 확인하여 중단된 Phase부터 재개
- 각 Phase 완료 시 이 파일의 상태를 ✅로 업데이트
- BOOK_PROGRESS.md 참조: `C:\Copilot\captures\- 텍스트 변환\BOOK_PROGRESS.md`
- 인사이트 원본: `C:\Copilot\captures\- 텍스트 변환\*_insight.md`
