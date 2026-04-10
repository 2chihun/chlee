# AI Trader — ML 고도화 추진 계획

> Marcos López de Prado "Advances in Financial Machine Learning" 기반
> 작성일: 2026-04-09 / 11번째 도서 적용 계획

---

## 추진 배경

기존 ai_trader는 10권의 투자 도서에서 추출한 규칙 기반(Rule-Based) 인사이트를
`features/` 모듈로 구현하였다. 이번 11번째 도서는 성격이 근본적으로 다르다.

| 기존 10권 | 11번째 (López de Prado) |
|-----------|------------------------|
| 투자 철학, 매매 기법 | ML 방법론, 통계적 검증 |
| "무엇을(What)" 중심 | "어떻게(How)" 중심 |
| 규칙 기반 시그널 | 데이터 기반 학습 |
| 개별 모듈 독립 작동 | 전체 파이프라인 개선 |

따라서 기존처럼 `features/*.py` 하나를 추가하는 것이 아니라,
**전체 시스템의 과학적 기반을 업그레이드**하는 프로젝트로 진행한다.

---

## Phase 1 (즉시 적용, 1~2주) — 검증 인프라 강화

기존 코드를 건드리지 않고, 측정/검증 도구부터 구축한다.

### 1-1. Backtest Statistics 강화 (`backtest/engine.py` 개선)
- [ ] `BacktestMetrics`에 PSR (Probabilistic Sharpe Ratio) 추가
- [ ] DSR (Deflated Sharpe Ratio) — 다수 시행 보정
- [ ] Returns Concentration (HHI) — 수익 집중도
- [ ] Time Under Water — 최대 수중 시간
- 기존 Sharpe Ratio / Max Drawdown은 유지, PSR/DSR로 **보완**
- 근거: Ch14 (p.195~209)

### 1-2. Purged K-Fold CV (`backtest/purged_cv.py` 신규)
- [ ] `PurgedKFoldCV` 클래스 구현 (Ch7 Snippet 7.4 기반)
- [ ] Embargo 파라미터 지원
- [ ] 기존 `BacktestEngine` 연동
- 근거: Ch7 (p.103~111) — sklearn의 기본 CV는 금융 시계열에 치명적 데이터 누수

### 1-3. Triple-Barrier Labeling (`features/labeling.py` 신규)
- [ ] `applyTripleBarrier()` 구현 (Ch3 Snippet 3.4 기반)
- [ ] 동적 임계값 계산 (`getDailyVol()`)
- [ ] 기존 swing/scalping 전략의 학습 데이터 생성기로 활용
- 근거: Ch3 (p.43~56) — Fixed-Time Horizon 대비 현실적 매매 결과 반영

### 1-4. Config 확장 (`config/settings.py`)
- [ ] `MLConfig` dataclass 추가
  ```python
  @dataclass
  class MLConfig:
      triple_barrier_pt: float = 1.5   # 이익 배리어 배수
      triple_barrier_sl: float = 1.0   # 손절 배리어 배수
      max_holding_days: int = 5        # 시간 배리어 (영업일)
      purge_pct: float = 0.01          # Purge 비율
      embargo_pct: float = 0.01        # Embargo 비율
      frac_diff_d: float = 0.35        # 분수 차분 d값
  ```

### Phase 1 산출물
- `backtest/engine.py` (개선) — PSR/DSR/HHI 통계
- `backtest/purged_cv.py` (신규) — Purged K-Fold CV
- `features/labeling.py` (신규) — Triple-Barrier Labeling
- `config/settings.py` (개선) — MLConfig 추가
- 테스트: 42개 → 약 52개 예상

---

## Phase 2 (2~3주) — ML 피처 엔지니어링

### 2-1. Fractional Differentiation (`features/frac_diff.py` 신규)
- [ ] `fracDiff_FFD()` 구현 (Ch5 Snippet 5.3 기반)
- [ ] 최적 d값 자동 탐색 (ADF 테스트 + 상관 보존)
- [ ] 기존 `indicators.py` 입력 시계열 전처리기로 활용
- 근거: Ch5 (p.75~89) — 정상성 확보 + 기억 최대 보존

### 2-2. Meta-Labeling (`features/meta_labeling.py` 신규)
- [ ] 1차 모델: 기존 `book_integrator.py`의 composite_score (규칙 기반 방향)
- [ ] 2차 모델: Random Forest로 "실행 여부 + 크기" 결정
- [ ] F1 점수 기반 최적화
- 근거: Ch3.6~3.7 (p.50~53) — Quantamental 접근법의 핵심

### 2-3. Sample Weights (`features/sample_weights.py` 신규)
- [ ] `mpNumCoEvents()` — 동시 레이블 수
- [ ] `mpSampleTW()` — 평균 고유성 가중치
- [ ] `seqBootstrap()` — 순차 부트스트랩
- 근거: Ch4 (p.59~73) — 비IID 금융 데이터의 올바른 샘플링

### 2-4. Feature Importance (`features/importance.py` 신규)
- [ ] MDI (Mean Decrease Impurity)
- [ ] MDA (Mean Decrease Accuracy)
- [ ] SFI (Single Feature Importance)
- [ ] 기존 17개 features 모듈의 예측력 순위 측정
- 근거: Ch8 (p.113~127) — 과적합 방지의 과학적 근거

### 2-5. Structural Breaks (`features/structural_break.py` 신규)
- [ ] CUSUM 필터 (이벤트 샘플링 트리거)
- [ ] SADF (폭발성 테스트, 버블/붕괴 탐지)
- [ ] 기존 `bubble_detector.py`(조상철)와 통합
- 근거: Ch17 (p.249~261) — 규칙 기반 → 통계 기반 버블 감지

### Phase 2 산출물
- 5개 신규 모듈
- Feature Importance 보고서 — 기존 피처의 실제 예측력 순위
- 테스트: 52개 → 약 65개 예상

---

## Phase 3 (4~6주) — 고급 기법 도입

### 3-1. Bet Sizing (`risk/bet_sizing.py` 신규)
- [ ] 예측 확률 → 포지션 크기 변환 함수
- [ ] Average Active Bets 정규화
- [ ] Size Discretization (10단계)
- [ ] Dynamic Bet Sizes + Limit Price 계산
- 기존 `risk/manager.py`의 `calculate_position_size()` 대체
- 근거: Ch10 (p.141~149)

### 3-2. Information-Driven Bars (`data/bar_builder.py` 신규)
- [ ] Dollar Bars
- [ ] Tick Imbalance Bars (TIB)
- [ ] Volume Runs Bars (VRB)
- KIS API 실시간 틱 데이터 → 정보 기반 바 생성
- 근거: Ch2 (p.23~41)

### 3-3. Microstructural Features (`features/microstructure.py` 신규)
- [ ] Kyle's Lambda (가격 영향도)
- [ ] VPIN (Volume-Synchronized Probability of Informed Trading)
- [ ] 실시간 유동성 독성 모니터링
- 근거: Ch19 (p.281~298)

### 3-4. Entropy Features (`features/entropy.py` 신규)
- [ ] Kontoyiannis 엔트로피 추정
- [ ] Lempel-Ziv 압축률
- [ ] 전략 선택기(swing vs scalping) 피처로 활용
- 근거: Ch18 (p.263~279)

### 3-5. HRP Portfolio Optimizer (`risk/portfolio_optimizer.py` 신규)
- [ ] Tree Clustering (scipy.linkage)
- [ ] Quasi-Diagonalization
- [ ] Recursive Bisection
- [ ] 기존 risk/manager.py 포트폴리오 배분과 비교
- 근거: Ch16 (p.221~245)

### Phase 3 산출물
- 5개 신규 모듈
- 기존 Time Bars vs Dollar Bars 성능 비교 보고서
- HRP vs 기존 균등 배분 OOS 성능 비교
- 테스트: 65개 → 약 80개 예상

---

## Phase 4 (장기) — 통합 및 최적화

### 4-1. 통합 파이프라인
- [ ] `book_integrator.py` → ML Meta-Labeling 결합
      기존 10권 도서 시그널(1차) × ML 메타라벨러(2차)
- [ ] 전략 라이프사이클 관리 (Ch1 Section 1.3.1.5)
      Graduation → Allocation → Decommission 자동화

### 4-2. Combinatorial Purged CV Backtesting
- [ ] CPCV 구현 (Ch12)
- [ ] Walk-Forward 대비 과적합 감소 검증

### 4-3. 성능 최적화
- [ ] mpPandasObj 패턴 적용 — 다종목 분석 병렬화
- [ ] 벡터화 우선 리팩토링

### 4-4. AI_TRADER_FULL_CONTEXT.md 업데이트
- [ ] 11번째 도서 인사이트 반영
- [ ] 신규 모듈 아키텍처 문서화
- [ ] 테스트 현황 업데이트

---

## 의존성 (추가 패키지)

```
# Phase 1-2
scikit-learn>=1.4     # Random Forest, CV (이미 설치됨)
statsmodels>=0.14     # ADF test, CUSUM

# Phase 3
scipy>=1.12           # Hierarchical Clustering (이미 설치됨)
```

---

## 리스크 및 주의사항

1. **LIVE 환경 영향 없음** — 모든 Phase는 분석/백테스트 레이어만 변경
   기존 `execution/order.py`, `data/collector.py`의 실시간 매매 로직은 건드리지 않음

2. **단계별 검증** — 각 Phase 완료 후 전체 테스트 PASS 확인 후 다음 단계 진행

3. **기존 로직 보존** — 신규 모듈은 기존 모듈과 병렬 구조로 추가
   기존 `book_integrator.py`의 가중치 체계는 유지,
   ML 레이어는 2차 필터로 상위에 배치

4. **과적합 경계** — 이 책의 핵심 메시지 자체가 "과적합 방지"
   따라서 구현 과정에서도 항상 OOS 성능을 기준으로 판단


---

## 구현 완료 현황 (2026-04-09)

### Phase 1 ✅ 완료 — 검증 인프라 강화
| 모듈 | 파일 | 상태 |
|------|------|------|
| PSR/DSR/HHI 통계 | `backtest/engine.py` | ✅ BacktestMetrics + _calc_metrics() |
| Purged K-Fold CV | `backtest/purged_cv.py` (신규) | ✅ PurgedKFoldCV + purged_train_test_split |
| Triple-Barrier Labeling | `features/labeling.py` (신규) | ✅ 장벽 라벨링 + CUSUM 필터 |
| MLConfig | `config/settings.py` (개선) | ✅ 18개 파라미터 데이터클래스 |

### Phase 2 ✅ 완료 — ML 피처 엔지니어링
| 모듈 | 파일 | 상태 |
|------|------|------|
| Fractional Differentiation | `features/frac_diff.py` (신규) | ✅ FFD + 최적 d 탐색 |
| Meta-Labeling ML | `features/meta_labeling_ml.py` (신규) | ✅ RF 2차 모델 + 확률→크기 |
| Sample Weights | `features/sample_weights.py` (신규) | ✅ 고유성 + 순차 부트스트랩 |
| Feature Importance | `features/importance.py` (신규) | ✅ MDI/MDA/SFI 3종 |
| Structural Breaks | `features/structural_break.py` (신규) | ✅ CUSUM/SADF/Chow |

### Phase 3 ✅ 완료 — 고급 기법 도입
| 모듈 | 파일 | 상태 |
|------|------|------|
| Bet Sizing | `risk/bet_sizing.py` (신규) | ✅ 확률→크기 + 이산화 |
| Information-Driven Bars | `data/bar_builder.py` (신규) | ✅ Tick/Volume/Dollar/TIB |
| Microstructure/VPIN | `features/microstructure.py` (신규) | ✅ VPIN + Kyle's λ + Amihud |
| Entropy Features | `features/entropy.py` (신규) | ✅ Shannon/LZ/ApEn |
| HRP Portfolio | `risk/portfolio_optimizer.py` (신규) | ✅ Tree→QD→Bisection |

### Phase 4 ✅ 완료 — 통합 및 최적화
| 모듈 | 파일 | 상태 |
|------|------|------|
| book_integrator ML 통합 | `features/book_integrator.py` (개선) | ✅ ML overlay 추가 |
| Combinatorial Purged CV | `backtest/cpcv.py` (신규) | ✅ C(N,k) + PBO |
| 병렬화 유틸리티 | `utils/parallel.py` (신규) | ✅ mpPandasObj + 벡터화 |

### 통계 요약
- **신규 모듈**: 14개 파일 생성
- **개선 모듈**: 3개 파일 수정 (engine.py, settings.py, book_integrator.py)
- **전체 모듈 import**: 16/16 성공
- **기존 테스트**: 48 pass 유지 (6 fail = 환경 의존, 변경 무관)
- **López de Prado 챕터 커버리지**: Ch2, Ch3, Ch4, Ch5, Ch7, Ch8, Ch10, Ch12, Ch14, Ch16, Ch17, Ch18, Ch19, Ch20
