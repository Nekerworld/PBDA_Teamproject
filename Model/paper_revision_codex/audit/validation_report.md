# Validation report (실제 실행 기반 검증 결과)

모든 수치는 실제 실행 결과다. v6(§2–§8)는 v6 실행 PC(RTX 5060 Ti)에서, v7과 §9–§26는 두 번째 PC(RTX 3090 Ti)에서 얻었다 (§8 항목 5의 표는 커밋된 v6·v7 선택 파일에서 계산). 원본 표는 `audit/results_v6/`에 있다. 표의 값은 K=50, 세 seed(42/43/44) 평균이다.

## 1. 코드 검증

| 검사 | 결과 |
|---|---|
| 회귀 테스트 (기존 46개 + 신규 16개) | 62/62 통과 (Cycle 2, 첫 PC) |
| 회귀 테스트, 다른 PC의 깨끗한 venv (Cycle 3) | 70/70 통과 (v7 테스트 포함). 이후 `test_cycle3_consistency.py` 9개 추가 → §11 |
| 회귀 테스트 (Cycle 3, v7 완료 후) | 78개 시점: v7 실행 venv 78/78, `requirements-v7-lock.txt`로 새로 만든 venv 78/78. M04 테스트 추가 후 79/79 (두 venv 모두). I-38 수정 후 82/82 (v7 venv). 모두 인자 없는 `python -m pytest` (당시 실행 출력은 저장소에 없음. 현재 수는 Cycle 6 행) |
| 회귀 테스트 (Cycle 4) | `test_cycle4_guards.py` 12개 추가 후 94/94 (v7 venv) → §13 |
| 회귀 테스트 (Cycle 5) | `test_cycle5_guards.py` 12개 추가 후 106/106 (v7 venv) → §14 |
| 회귀 테스트 (Cycle 6) | `test_cycle5_guards.py`에 3개 추가 후 109/109 (v7 venv) → §15 |
| 회귀 테스트 (Cycle 7) | `test_cycle7_guards.py` 4개, `test_policy_tuning_gap.py` 1개 추가 후 114/114 (v7 venv) → §16 |
| 회귀 테스트 (Cycle 8) | `test_cycle8_guards.py` 6개 추가 후 120/120 (v7 venv). 커밋 b9e7134의 새 worktree + lock venv 119 통과, 1 건너뜀 → §17 |
| 회귀 테스트 (Cycle 9) | `test_cycle9_guards.py` 6개, `test_stamp_code_check.py` 1개 추가 후 127/127 (v7 venv). 커밋 ce97790의 새 worktree + lock venv 126 통과, 1 건너뜀 → §18 |
| 회귀 테스트 (Cycle 10) | `test_cycle10_guards.py` 9개 추가 후 136/136 (v7 venv). 커밋 4e166c5의 새 worktree + lock venv 135 통과, 1 건너뜀 → §19 |
| 회귀 테스트 (Cycle 11) | `test_compare_outputs.py` 1개, `test_cycle11_guards.py` 3개 추가 후 140/140 (v7 venv). 커밋 318186b의 새 worktree + lock venv 139 통과, 1 건너뜀 → §20 |
| 회귀 테스트 (Cycle 12) | `test_cycle12_guards.py` 4개, `test_compare_outputs.py` 1개 추가 후 145/145 (v7 venv, 출력 저장 안 함). 커밋 5026680의 새 worktree + lock venv 144 통과, 1 건너뜀 → §21 |
| 회귀 테스트 (Cycle 13) | `test_cycle13_guards.py` 4개 추가 후 149/149 (v7 venv, 출력 저장 안 함). 커밋 cb22505와 db3653e의 새 worktree + lock venv 각각 148 통과, 1 건너뜀 → §22 |
| 회귀 테스트 (Cycle 14) | `test_cycle14_guards.py` 2개 추가 후 151/151 (v7 venv, 출력 저장 안 함). 커밋 41ba20d의 새 worktree + lock venv 150 통과, 1 건너뜀 → §23 |
| 회귀 테스트 (Cycle 15) | `test_cycle15_guards.py` 6개(매개변수화) 추가 후 157/157 (v7 venv, 출력 저장 안 함). 커밋 4c1ddfb의 새 worktree + lock venv 156 통과, 1 건너뜀 → §24 |
| 회귀 테스트 (Cycle 16) | `test_cycle16_guards.py` 1개 추가 후 158/158 (v7 venv, `audit/cycle16/precommit_pytest.txt`). 커밋 6ce754d의 새 worktree + lock venv 157 통과, 1 건너뜀 → §25 |
| 회귀 테스트 (Cycle 17) | `test_cycle17_guards.py` 2개 추가, `test_cycle16_guards.py`를 실제 git으로 바꾼 뒤 160/160 (v7 venv, `audit/cycle17/precommit_pytest.txt`). 커밋 cf1a3fe의 새 worktree + lock venv 159 통과, 1 건너뜀 → §26 |
| v7 시험 지표 독립 재계산 | 270행 최대 차이 1.0e-16, novel 목록 위반 0건 → §12.1 |
| 통합 smoke 실행 (합성 데이터, 전체 파이프라인) | 통과. 재실행 시 시험 접근이 늘지 않음. plan이 바뀌면 거부됨 |
| 누수 검사 결함 검출 | 시험 이벤트 1건을 학습 파일에 넣으면 실패함 |
| 실제 데이터 지표 교차 계산 | v6 평가기와 기존 `ranking_evaluation.ranking_metrics`를 3 fold × 8 모델 × K∈{10,50}, 48건에서 비교. 최대 차이 1.0e-16 (v6 PC의 실행 폴더에서 실행. 비교 결과 파일은 저장소에 없어 이 저장소만으로는 재확인할 수 없음) |
| novel 정책 독립 검사 | 추천 1,010,100건 중 원본 학습 기간에 본 상품 0건, 사용자별 중복 0건 (위와 같이 v6 PC에서 실행, 결과 파일 없음. v7은 커밋된 도구로 다시 실행 가능 → §12.1) |
| 수기 계산 대조 | F1 Recent (0.0938 / 0.1265)가 감사 초기에 따로 계산한 값과 같음 (그 계산 기록은 저장소에 없음. 저장된 추천 목록으로 지표를 다시 계산하는 검사(§12 `recompute_test_metrics.py`)는 v7 F4에만 했다. v6 추천 목록은 v6 PC의 실행 폴더에만 있음) |

## 2. 누수 (원본 데이터로 독립 재계산)

F1, F2, F3 모두 28/28 통과 (`F*/leakage_check.json`). 검사 항목:
- tie-safe cutoff
- core 고정점
- 평가 사용자 정의와 정답
- 학습 파일에 시험 이벤트 없음
- 속성 시점 < cutoff
- `available` = cutoff 이전 마지막 값
- 카탈로그 = 학습 상품
- 내부 validation 구간이 바깥 학습 구간 안에 있음
- validation 정답이 내부 학습에 없음
- GNN 간선 = 학습 쌍
- 미래 전용 쌍이 그래프에 없음

## 3. 평가 모집단

| fold | 시험 구간 | 정책 | 평가 사용자 | 학습 벡터 있음 | 도달 가능한 정답 있음 | 카탈로그 |
|---|---|---|---|---|---|---|
| F1_hist | v1–v5 시험 구간 (이미 노출) | repeat | 854 | 429 | 409 | 58,556 |
| | | novel | 717 | 342 | 316 | |
| F2_mid | F1 학습 구간의 마지막 25% (대부분 v1–v5의 내부 validation 구간) | repeat | 897 | 458 | 431 | 47,979 |
| | | novel | 755 | 372 | 342 | |
| F3_early | 처음 사용하는 앞쪽 구간 | repeat | 628 | 326 | 305 | 39,284 |
| | | novel | 505 | 248 | 225 | |

- 모든 모델의 분모는 같다. 학습 벡터가 없는 사용자는 0점으로 포함한다.
- repeat 정책에서 학습 모델(baseline 4종과 Hybrid)마다 사용자의 73.6–80.8%가 Recall 0이다 (모델별 값, Hybrid는 73.6–77.8%, `per_user_distribution.csv`). ablation `Hybrid_ALS_only`까지 넣으면 73.2–80.8%.
- 적어도 한 seed에서 맞힌 사용자 수(Hybrid, repeat)는 F1 190, F2 205, F3 166명이다.

## 4. 시험 결과

### 4.1 repeat 정책 (원래 프로토콜)

| 모델 | F1 NDCG | F2 NDCG | F3 NDCG | F1 Recall | F2 Recall | F3 Recall |
|---|---|---|---|---|---|---|
| Popularity | 0.0138 | 0.0106 | 0.0097 | 0.0304 | 0.0269 | 0.0293 |
| Recent | 0.0938 | 0.1002 | 0.1188 | 0.1265 | 0.1250 | 0.1475 |
| RecentPop | 0.0958 | 0.1034 | 0.1217 | 0.1333 | 0.1353 | 0.1578 |
| ALS | 0.0966 | 0.0957 | 0.1133 | 0.1477 | 0.1522 | 0.1787 |
| BPR | 0.1056 | 0.1073 | 0.1331 | 0.1500 | 0.1503 | 0.1783 |
| ALS+BPR | 0.1152 | 0.1120 | **0.1365** | 0.1593 | 0.1622 | 0.1893 |
| GNN | 0.1065 | 0.1081 | 0.1305 | **0.1681** | 0.1628 | 0.1907 |
| Hybrid | **0.1166** | **0.1132** | 0.1344 | 0.1668 | **0.1710** | **0.1972** |

### 4.2 novel 정책 (학습 기간에 본 상품 제외)

| 모델 | F1 NDCG | F2 NDCG | F3 NDCG | F1 Recall | F2 Recall | F3 Recall |
|---|---|---|---|---|---|---|
| Popularity = RecentPop | 0.0046 | 0.0064 | 0.0053 | 0.0122 | 0.0140 | 0.0151 |
| Recent | 0 (정의상) | 0 | 0 | 0 | 0 | 0 |
| ALS | 0.0274 | 0.0229 | 0.0304 | 0.0593 | 0.0531 | 0.0616 |
| BPR | 0.0223 | 0.0160 | 0.0208 | 0.0484 | 0.0411 | 0.0491 |
| ALS+BPR | 0.0306 | 0.0258 | 0.0306 | 0.0605 | 0.0637 | 0.0618 |
| GNN | 0.0301 | 0.0226 | 0.0299 | **0.0708** | 0.0544 | 0.0714 |
| Hybrid | **0.0319** | **0.0278** | **0.0347** | 0.0643 | **0.0658** | **0.0786** |

novel 정책에서 학습 모델의 NDCG는 repeat의 약 15–28%, Recall은 약 27–42%로 떨어진다. repeat 지표의 상당 부분은 이미 본 상품을 다시 추천해서 얻은 것이다.

### 4.3 Hybrid와 baseline의 쌍대 비교 (Holm 보정, α = 0.05)

| 비교 | 정책 | 지표 | F1 | F2 | F3 |
|---|---|---|---|---|---|
| Hybrid − GNN | repeat | NDCG | +0.0101 유의 | +0.0051 비유의 (p_holm 0.061) | +0.0038 비유의 |
| | | Recall | −0.0012 비유의 | +0.0082 비유의 (p_holm 0.051) | +0.0065 비유의 |
| | novel | NDCG | +0.0017 비유의 | +0.0052 유의 | +0.0048 유의 |
| | | Recall | −0.0065 비유의 | +0.0115 유의 | +0.0072 비유의 |
| Hybrid − ALS+BPR | repeat | NDCG | +0.0014 비유의 | +0.0012 비유의 | −0.0021 비유의 |
| | | Recall | +0.0075 유의 | +0.0088 유의 | +0.0079 비유의 |
| | novel | NDCG | +0.0013 비유의 | +0.0020 비유의 | +0.0041 비유의 |
| | | Recall | +0.0038 비유의 | +0.0022 비유의 | +0.0169 유의 |
| Hybrid − RecentPop | repeat | NDCG | +0.0208 유의 | +0.0098 유의 | +0.0127 비유의 (p_holm 0.085) |
| | | Recall | +0.0335 유의 | +0.0358 유의 | +0.0394 유의 |
| | novel | NDCG·Recall | 유의 | 유의 | 유의 (novel에서 RecentPop = Popularity) |
| Hybrid − ALS, BPR, Popularity, Recent | 둘 다 | 둘 다 | 대부분 유의 | 대부분 유의 | 대부분 유의 (예외는 `paired_tests.csv`) |

**해석**
- Hybrid는 평균으로는 대부분의 칸에서 1위다 (baseline 7종 대비 12칸 중 9칸).
- `Hybrid_ALS_only`는 Hybrid와 같은 후보(ALS+BPR로 만든 후보)에서, 최종 결합 점수의 MF 부분만 ALS로 바꾼 ablation이다 (GNN 결합과 γ는 같음). 후보 생성에는 BPR이 그대로 쓰이므로 BPR을 완전히 뺀 모델이 아니다. `balanced_union` 후보에서는 BPR 상위 항목이 후보의 절반까지 들어온다. 사전등록 비교 대상이 아니어서 검정하지 않았다.
  - Hybrid와의 평균 차이는 작다 (repeat NDCG +0.0011 – +0.0051).
  - ablation이 더 높은 칸: F3 novel (NDCG 0.0353 vs 0.0347, Recall 0.0805 vs 0.0786), F1 repeat Recall (0.1681 vs 0.1668).
  - 후보 생성에 BPR이 남아 있고 검정도 하지 않았으므로, 이 비교로 BPR의 기여를 판단할 수 없다.
- GNN 결합, γ, 후보 생성, 따로 선택된 MF 결합 비율 α(Hybrid와 ALS+BPR baseline의 α가 fold·정책에 따라 다름)를 합친 효과는 "Hybrid − ALS+BPR" 비교가 보여 준다: NDCG는 6칸 모두 비유의, Recall은 3칸(F1·F2 repeat, F3 novel) 유의.
- 가장 강한 두 baseline(GNN, ALS+BPR)보다 **세 fold 모두에서 유의하게 높은 지표는 없다**.
- ALS+BPR 대비 NDCG는 6개 칸 모두 비유의다.
- GNN 대비 Recall은 F1에서 평균이 더 낮다.
- 따라서 "Hybrid가 모든 baseline보다 우수하다"는 통계적으로 뒷받침되지 않는다. 뒷받침되는 주장은 "단순 규칙, ALS, BPR보다 대체로 유의하게 높고, GNN·ALS+BPR과는 지표·fold에 따라 비슷하거나 약간 높다(평균이 더 낮은 칸도 있음: F1 Recall의 GNN 대비, F3 repeat NDCG의 ALS+BPR 대비)"이다.
- novel 정책에서 Hybrid는 결합 설정을 novel validation으로 다시 고르지만 ALS·GNN 단독은 repeat에서 고른 설정을 쓴다. novel의 Hybrid − GNN, Hybrid − ALS 유의 결과는 이 비대칭 아래의 결과다 (§8 항목 5, I-63).

### 4.4 시간 변동

fold 사이 표준편차(repeat, Hybrid)는 NDCG 0.0114, Recall 0.0165이다. 모델 간 차이(Hybrid − GNN, 평균 NDCG +0.0063)보다 크다. 학습 모델의 세 seed 표준편차는 0.0001–0.0051로 fold 간 변동보다 훨씬 작다.

## 5. 선택과 튜닝 예산 (내부 validation, `selection.json`)

| fold | ALS (학습 24회) | BPR (학습 4회, 체크포인트 20개) | GNN (학습 8회) | GNN 선택 epoch | Hybrid 선택 (repeat) |
|---|---|---|---|---|---|
| F1 | f256 r.02 α10 i32 | lr.005 r.003 e256 | 3층 128차원 lr1e-3 | 160 | blend, k=20000, minmax, ω=.4, γ=1.25 |
| F2 | f256 r.2 α10 i32 | lr.005 r.003 e256 | 2층 128차원 lr1e-3 | 290 | balanced_union, k=500, minmax, ω=.5, γ=1.1 |
| F3 | f256 r.02 α10 i32 | lr.005 r.003 e256 | 2층 128차원 lr1e-3 | 200 | blend, k=all, minmax, ω=.3, γ=1.1 |

- GNN 학습 시간 합계(fold별)는 F1 13,885초, F2 10,032초, F3 6,998초. ALS는 840–1,267초, BPR은 187–409초.
- Hybrid는 추가 학습 없이 추론 설정 8,424개(2,808개 × backbone 3개)를 평가했다.
- Hybrid의 GNN backbone은 GNN 단독 선택과 따로 고른다. F1(2층 64차원 lr3e-3)과 F3(2층 128차원 lr3e-3)에서는 단독 GNN과 다른 설정이 선택돼, 시험 재학습 때 별도 GNN을 학습했다 (`refits.json`의 `hybrid_gnn`).
- **GNN epoch 상한**: 24개 GNN 학습이 모두 400 상한보다 훨씬 이른 50–290 epoch를 선택했다. 연장 규칙은 한 번도 발동하지 않았다. v5의 190/200 문제는 상한 때문이 아니었다.
- **Hybrid 후보 수가 fold마다 다름** (20000 / 500 / all): validation 곡선이 평평해 후보 수가 정해지지 않는다는 뜻이다 (§6.1과 같은 결론).

## 6. 사후 분석 (F1, seed 42, 선택에 사용하지 않음)

### 6.1 후보 수와 비용 (`posthoc_candidate_depth.csv`)

| 후보 수 | 카탈로그 비율 | NDCG | Recall | 후보의 정답 포함률 | 사용자당 ms | Python 최대 메모리 MB |
|---|---|---|---|---|---|---|
| 100 | 0.2% | 0.1162 | 0.1648 | 20.8% | 7.1 | 2.9 |
| 500 | 0.9% | 0.1166 | 0.1663 | 28.2% | 6.3 | 2.9 |
| 2000 | 3.4% | 0.1175 | 0.1685 | 38.6% | 6.8 | 2.9 |
| 20000 | 34% | 0.1175 | 0.1690 | 71.3% | 14.3 | 6.5 |
| 전체 | 100% | 0.1160 | 0.1660 | 100% | 29.2 | 16.8 |

- 전체 카탈로그 GNN 점수 계산(정렬 포함)은 사용자당 0.62 ms다.
- 후보 수 100–20000 사이(표에 없는 200·1000·5000·10000 포함, CSV 전체 행) 지표 차이는 NDCG 0.0023, Recall 0.0079 이내다. 가장 낮은 값은 후보 200개(0.1152 / 0.1611)이다. 후보를 500개로 줄여도 성능이 거의 같고 비용은 절반 이하다.
- 그러나 Hybrid 전체 경로(MF 점수 계산 포함)는 GNN 단독 전체 카탈로그 점수 계산보다 10배 이상 느리다. "후보 생성으로 계산 비용을 줄인다"는 주장은 GNN 단독과 비교할 때 성립하지 않는다.

### 6.2 gamma (재고 가중, `posthoc_gamma.csv`)

| γ | NDCG | Recall | Top-50 중 재고 있음 비율 |
|---|---|---|---|
| 1.0 (중립) | 0.1151 | 0.1625 | 73.4% |
| 1.1 | 0.1169 | 0.1658 | 84.9% |
| 1.25 (선택값) | 0.1175 | 0.1690 | 92.4% |
| 1.5 | 0.1160 | 0.1662 | 97.2% |
| 2.0 | 0.1140 | 0.1614 | 99.6% |

재고 가중의 정확도 효과는 작다 (최대 NDCG +0.0024). 노출되는 상품 구성은 크게 바뀐다. `available`은 학습 종료 시점 이전의 마지막 관측값이며, 시험 기간의 실시간 재고가 아니다.

### 6.3 Isolation Forest (`if_ablation.csv`, `if_removal_by_event.csv`, `if_summary.json`)

- 특징 9개: event_count, view_count, unique_visitors, event_code_mean, event_code_std, first_event_hour, last_event_hour, event_duration_hours, property_count.
- 학습 상품의 13.16%가 이상치로 분류됨. 제거된 이벤트는 필터 전 학습 이벤트의 **62.05%**, 원시 입력 전체의 49.64%다.
- 행동별 제거율: view 60.8%, addtocart 99.1%, transaction **100%** (17,864건 전부).
- 같은 설정(no-IF validation에서 고정)으로 IF 데이터에서 학습한 결과: Hybrid NDCG 0.0079 / Recall 0.0103 (no-IF는 0.1175 / 0.1690). 모든 모델이 같은 수준으로 무너졌다.
- 따라서 최종 모델에서 IF를 뺀 것은 성능을 골라서가 아니라 IF가 구매 신호를 전부 지우기 때문이다. 이 IF는 봇 검출이 검증된 도구가 아니다.
- 한계: 이 ablation은 no-IF에서 고정한 설정을 썼다. IF 데이터 전용으로 튜닝하지는 않았다.

## 7. 결정성 (`determinism_mf_blas1.csv`, `determinism_run1_blas24.csv`)

F3 seed 42를 같은 설정으로 다시 학습해 비교했다.

| 산출물 | 최대 차이 | Top-50 목록 완전 일치 |
|---|---|---|
| ALS (CPU, BLAS 1스레드) | 0 | 100% |
| BPR (GPU) | 0 | 100% |
| Hybrid의 GNN backbone (`hybrid_gnn_seed_42`, GPU, 100 epoch) | 사용자 임베딩 1.1e-4 | 99.7% (목록 평균 겹침 100%) |

- 첫 검사에서는 ALS가 다르게 나왔다 (최대 차이 0.205, 목록 일치 47.5%).
- 원인은 검사 스크립트가 numpy를 import한 뒤에 스레드 환경변수를 설정해 OpenBLAS가 24스레드로 돌았기 때문이다.
- 고친 뒤에는 비트 단위로 같다. **ALS 결과는 BLAS 스레드 수에 따라 달라진다.** `rolling_protocol.py`는 import 전에 1스레드로 고정한다.
- GNN은 CUDA sparse 연산 때문에 비트 단위로는 같지 않다. 차이는 순위에 거의 영향이 없다.
- 고친 뒤 다시 검사(`determinism_mf_blas1.csv`)에는 ALS와 BPR이 들어 있고 둘 다 비트 단위로 같다. GNN은 다시 검사하지 않았으므로 GNN 행은 첫 검사(`determinism_run1_blas24.csv`) 값이다. GPU 학습이라 BLAS 스레드 결함과 무관하다. BPR은 두 검사 모두 비트 단위로 같다.

## 8. 한계와 미해결 사항

1. **MF baseline이 grid 경계에서 선택됨 (I-23, P1)**: validation 전용 경계 탐색(`mf_boundary_probe.csv`)에서 봉인된 ALS 설정을 grid 밖으로 바꾸자(factors 512, α 40, iter 16) ALS validation 점수가 크게 높아졌다.

   | fold | v6 봉인 ALS | 경계 밖 ALS | 증가 | 참고: GNN 최고 / Hybrid 최고 |
   |---|---|---|---|---|
   | F1 | 0.1459 | 0.1669 | +0.0210 | 0.1678 / 0.1759 |
   | F2 | 0.1673 | 0.1782 | +0.0109 | 0.1912 / 0.1966 |
   | F3 | 0.1818 | 0.2090 | +0.0272 | 0.2119 / 0.2171 |

   - 점수는 validation 선택 점수 mean(NDCG@50, Recall@50), seed 42다.
   - 봉인된 설정을 다시 학습한 행은 봉인된 validation 점수를 그대로 재현한다 (차이 0).
   - 첫 probe는 BLAS 스레드 결함(I-25와 같은 원인)과 F2 설정 하드코딩이 있어 `mf_boundary_probe_run1_blas24.csv`로 보존만 한다.
   - BPR의 fold별 최고 증가폭은 +0.0002 – +0.0044로 작다.
   - 따라서 v6 시험의 ALS와 ALS+BPR baseline은 덜 튜닝된 상태일 수 있다. Hybrid도 ALS를 구성요소로 쓰므로 영향의 방향을 단정할 수 없다.
   - 세 fold의 시험을 이미 열었으므로 v6 fold는 다시 실행하지 않았다. 대신 사용자 승인을 받아, 쓰지 않은 구간 F4에서 ALS·BPR·GNN의 grid를 넓힌 v7(Hybrid grid와 ALS+BPR α는 v6와 같음)을 사전등록해 실행했다 (§12). v7에서도 Hybrid와 ALS+BPR의 차이는 네 칸 모두 비유의로, "가장 강한 baseline보다 일관되게 유의하게 높지는 않다"는 v6의 결론이 유지됐다. v6 표의 ALS·ALS+BPR 수치 자체는 덜 튜닝된 조건의 값으로 남는다.
2. **F1 시험 구간은 v1–v5에서 반복 노출됐고, F2 시험 구간(이벤트 비율 0.60–0.80)의 대부분(0.64–0.80, 이벤트의 80%)은 v1–v5에서 설정을 고를 때 쓴 내부 validation 구간이다** (`rolling_protocol.py`의 F2 설명, `v7_feasibility.md` §1 표). F2는 시험으로는 처음 연 구간이지만, v1–v5에서 정한 고정 설계와 독립적이지 않다. v6 전에 평가·선택에 쓰인 적 없는 시험 구간은 F3와 v7의 F4뿐이다(F3 시험 구간의 80%는 같은 v6 실행에서 F2의 내부 validation, 즉 F2 설정 선택에 쓰였지만 F3 자신의 학습·선택에는 들어가지 않음). 데이터셋도 하나(RetailRocket 2015)뿐이다.
3. 평가 사용자가 수백 명이고, 그중 맞힌 사용자는 약 160–205명이다. 작은 차이는 검정력이 부족하다.
4. 사후 분석(후보 수, gamma, IF)은 F1의 seed 42 하나로만 했다.
5. **novel 정책에서 튜닝이 비대칭이다 (I-63, P1)**: 사전등록 규칙상 학습 설정(ALS, BPR, GNN, Hybrid의 GNN backbone)은 repeat validation으로 고르고, 추론 설정만 정책마다 다시 고른다. 그래서 novel에서 Hybrid는 결합 설정 2,808개를, ALS+BPR은 α를 novel validation으로 다시 고르지만, ALS·BPR·GNN 단독은 repeat에서 고른 설정 그대로다. 같은 grid 안에서 novel validation으로 골랐다면 (`report_tools/policy_tuning_gap.py`, `audit/cycle7/policy_tuning_gap_v6.json`, `_v7.json`. validation 점수 mean(NDCG@50, Recall@50), seed 42):

   | fold | GNN 선택 / grid 최고 (선택 순위) | ALS 선택 / 최고 | Hybrid 선택 |
   |---|---|---|---|
   | F1 | 0.0579 / 0.0579 (1/8) | 0.0474 / 0.0477 | 0.0641 |
   | F2 | 0.0598 / 0.0650 (3/8) | 0.0523 / 0.0523 | 0.0708 |
   | F3 | 0.0529 / 0.0581 (4/8) | 0.0493 / 0.0504 | 0.0674 |
   | F4 (v7) | 0.0526 / 0.0659 (19/36) | 0.0541 / 0.0571 | 0.0612 |

   - ALS는 네 fold 모두 novel로 고르면 오른다 (F2는 0.05226 → 0.05232로 매우 작음). BPR은 네 fold 모두 repeat 선택이 novel에서도 최고다.
   - ALS+BPR은 α만 정책별로 고르고, 그 안의 ALS·BPR은 repeat로 고른 것이다. Hybrid도 같은 ALS·BPR을 쓰므로 MF 부분의 조건은 같지만, novel에서 다시 고르는 추론 설정은 Hybrid 2,808개, ALS+BPR α 하나다. ALS·BPR을 novel로 다시 골랐을 때 ALS+BPR이 어떻게 될지는 재학습이 필요해 계산하지 않았다. 그래서 novel F3 Recall의 Hybrid − ALS+BPR 유의 결과(p_holm .010)도 이 조건 아래의 결과로 본다.
   - GNN의 학습 epoch는 설정마다 공통 validation 기준의 early stopping으로 한 번 정해지고, 기록된 novel 점수는 그 epoch의 값이다. novel 기준으로 epoch까지 골랐다면 더 높을 수 있으므로, 위 GNN의 novel 최고값은 하한이다.
   - v7 F4에서는 novel로 고른 GNN(4층 256차원)이 validation에서 Hybrid보다 높다. Hybrid의 backbone 후보도 repeat 상위 3개(모두 1층)뿐이라 이 GNN을 쓰지 못한다.
   - "grid 최고"는 여러 설정 중 최댓값이라 낙관적으로 치우친 값이다 (Hybrid 선택값도 2,808개 중 최댓값). 시험에서 어떻게 될지는 알 수 없고, 이미 연 시험 구간을 다시 열 수 없어 확인하지 않았다.
   - 따라서 novel 정책의 유의한 결과 중 Hybrid − GNN (F2 NDCG·Recall, F3 NDCG, F4 Recall)과 Hybrid − ALS (F2 NDCG·Recall, F3 Recall, F4 Recall)는 "repeat로 튜닝된 단독 모델 대비"로만 해석해야 하며, novel에서 GNN·ALS보다 우수하다는 근거로 쓰면 안 된다. Hybrid − BPR의 novel 결과는 이 문제의 영향을 받지 않는다.

## 9. 다른 PC에서 교차 재현 (Cycle 3, `audit/cross_machine/cross_machine_F3_early.json`)

새로 clone한 저장소와 깨끗한 venv(저장소 밖)로 v6 F3를 다시 전처리하고, ALS/BPR validation grid 전체를 다시 학습했다. 바깥 시험은 읽지 않았다.

| 항목 | 원래 실행 (v6) | 이번 실행 |
|---|---|---|
| PC / GPU | RTX 5060 Ti | RTX 3090 Ti (호스트 `Daisy`) |
| CPU 스택 | Python 3.11.9, numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, sklearn 1.9.0, implicit 0.7.3 | 같음 |
| torch | 2.15.0.dev20260921+cu132 | 2.11.0+cu128 (`requirements-cu128.txt`) |

| 비교 | 결과 |
|---|---|
| fold 절단 (`fold_raw.json`의 5개 항목: 원본·fold events 해시 2개, `end_fraction`, 절단 시각, 규칙) | 모두 같음 |
| 누수 검사 28개 (값 포함) | 모두 같음, 모두 통과 |
| validation 모집단 (repeat/novel) | 같음 |
| ALS grid 24행 (repeat/novel 지표 전체) | **24/24 비트 단위 동일** |
| BPR grid 20행 (GPU) | 20/20 다름. 선택 점수 최대 차이 0.0047 |
| 봉인된 MF 선택 (ALS, BPR, BPR epoch) | 같음 |

- ALS(CPU, BLAS 1스레드)는 PC가 달라도 같은 결과를 낸다.
- BPR(GPU)은 GPU·torch 빌드가 다르면 값이 달라진다. 이 차이(≤ 0.005)는 v6의 일부 모델 간 차이와 비슷한 크기이므로, 소수 셋째 자리의 차이를 해석할 때 seed 표준편차와 함께 고려해야 한다.

## 10. CPU와 GPU (Cycle 3, `audit/cross_machine/device_check_F3_early.json`)

같은 seed와 설정(F3 봉인 GNN: 2층 128차원 lr 1e-3, 30 epoch / 봉인 BPR, 64 epoch)을 CUDA와 CPU에서 학습했다. F3 내부 validation만 썼다.

| 모델 | CUDA 점수 | CPU 점수 | Top-50 평균 겹침 | 완전히 같은 목록 |
|---|---|---|---|---|
| GNN | 0.1912 | 0.1888 | 84.0% | 0% |
| BPR | 0.1899 | 0.1912 | 28.5% | 0% |

- 같은 seed라도 장치가 다르면 다른 모델이 학습된다 (난수 흐름과 연산 순서가 다름). 집계 지표 차이(NDCG 0.0015 – 0.0022, Recall 0.0033 – 0.0047, 선택 점수 0.0013 – 0.0024)는 seed 간 변동과 같은 크기다.
- v6와 v7의 모든 GNN/BPR 결과는 CUDA에서 얻었다. CPU로 재현하면 목록 수준에서는 일치하지 않는다.

## 11. 테스트의 결함 검출력: 변이 테스트 (Cycle 3, `audit/mutation/mutation_results.json`)

핵심 함수에 알려진 결함을 하나씩 넣고 테스트가 실패하는지 확인했다. 프로젝트를 gitignore 대상 사본(`audit_runs/mutation_src`)에 복사해 실행했다 (`report_tools/mutation_check.py`). 공식 결과는 v7 완료 후, 다른 GPU 작업이 없는 상태에서 얻었다.

| 결함 | 기존 테스트 (v6 프로토콜·smoke·revision·codex audit) | 새 테스트 (`test_cycle3_consistency.py`) |
|---|---|---|
| M01 NDCG 이상적 DCG를 카탈로그 안 정답으로만 계산 | **놓침** | 잡음 |
| M02 Recall 분모를 카탈로그 안 정답으로 | 잡음 | 잡음 |
| M03 학습 벡터 없는 사용자를 분모에서 제외 | **놓침** | 잡음 |
| M04 novel 정답에서 원본 대신 필터 후 이력을 뺌 | **놓침** | 잡음 |
| M05 novel 정책의 본 상품 제외를 끔 | 잡음 | 잡음 |
| M06 Holm 단조 보정 제거 | 잡음 | 놓침 |
| M07 sign-flip을 한쪽 검정으로 | **놓침** | 잡음 |
| M08 쌍대 비교를 사용자 ID가 아닌 위치로 짝지음 | **놓침** | 잡음 |
| M09 선택 grid에서 γ가 재고 없는 상품을 올림 | **놓침** | 잡음 |
| M10 선택 grid의 RRF 가중치 뒤바뀜 | **놓침** | 잡음 |
| M11 선택 grid에서 MF min-max 정규화 생략 | **놓침** | 잡음 |
| M12 cutoff와 같은 시각의 이벤트를 학습에 넣음 | 잡음 | 놓침 |
| M13 `available`을 첫 값으로 | 잡음 | 잡음 |
| M14 cutoff 이후 속성 사용 | 잡음 | 잡음 |
| M15 MF 선택이 최솟값을 고름 | **놓침** | 잡음 |
| M16 `best_row`가 최솟값을 고름 | **놓침** | 잡음 |
| M17 재고 플래그 반전 | **놓침** | 잡음 |

- 기존 테스트는 17개 중 11개를 놓쳤다. 두 묶음을 합치면 17개 모두 잡는다. 변이하지 않은 사본은 두 묶음 모두 통과한다.
- 놓친 항목은 구현 결함이 아니다. 코드 정독과 실제 데이터 대조로 구현이 올바름을 확인했다. 예를 들어 v6 `test_metrics.csv`의 분모는 모든 fold·정책에서 평가 사용자 전체(854/717, 897/755, 628/505)와 같고, v7 지표는 원본에서 독립 재계산한 값과 같다 (§12.1). 놓친 항목은 테스트가 해당 성질을 고정하지 않았다는 뜻이다.
- 특히 M09–M11은 "validation 선택에 쓰는 `hybrid_grid` 점수"와 "시험에 쓰는 `hybrid_ranking` 점수"가 같은지를 아무 테스트도 확인하지 않았다는 뜻이다. 새 테스트는 두 정책, 두 결합 방식, γ ∈ {1, 1.5}, 후보 수 3종에서 두 경로의 지표가 1e-12 이내로 같음을 확인한다.
- **M04**: 기존 합성 데이터는 모든 사용자가 같은 상품을 반복해서 보므로 core 필터가 학습 이력을 하나도 지우지 않는다 (필터 전후 이력이 다른 사용자 0명). 그래서 "원본 이력 대신 필터 후 이력"을 쓰는 결함이 이 데이터에서는 정답 코드와 구별되지 않았다. 필터가 지우는 희귀 상품을 본 뒤 시험에서 그 상품을 산 사용자를 넣은 테스트(`test_novel_truth_removes_items_seen_only_before_core_filtering`)를 추가해 잡았다.
- 기록 정정: v7 실행 중 같은 사본으로 한 예비 실행(`audit_runs/mutation.log`, 보존)에서는 M04를 기존 테스트가 잡은 것으로 나왔다. 대상 코드는 같은데 공식 실행에서는 놓침으로 나왔고, 위 설명대로 합성 데이터에서 M04는 정답 코드와 구별되지 않는다. 따라서 예비 실행의 실패는 M04와 무관한 원인이었다 (예비 실행 당시의 mutant 문구 또는 v7과 GPU를 같이 쓴 환경으로 추정하며, 확인하지 못함). 이 문서는 공식 실행만 쓴다. M04 테스트를 넣기 전의 공식 실행 결과는 `mutation_results_run1_before_M04_test.json`으로 보존했다.
- 이 결과 뒤에 `rolling_protocol.py`가 여러 번 바뀌었다 (I-38 코드 해시, Cycle 4·5·6 가드). 대상 문구 17개는 매번 한 번씩 있음을 확인했다. Cycle 4 수정 후 같은 명령으로 다시 실행한 결과(2026-10-22, 49분 38초)가 위 공식 결과 JSON과 완전히 같았다 (17/17 검출, 기존 테스트 11개 놓침, 새 테스트는 M06·M12 놓침).

## 12. v7: 확장 grid, 처음 쓰는 구간 F4 (I-23 확인 실험)

설계와 승인 과정은 `v7_feasibility.md`에 있다. 원본 표는 `audit/results_v7/`, 그림과 표는 `audit/figures_v7/`에 있다.

### 12.1 절차 검증

| 검사 | 결과 |
|---|---|
| plan | `784d0fa9…`, 코드 commit `1170ea5` (작업 트리 깨끗함) |
| 시험 열람 | `sealed_test` 1회. 선택 고정(13:04:26 UTC) 뒤 13:09:41 UTC. 사후 분석 없음 |
| 코드 해시 | 실행 종료 후 현재 코드 해시가 plan의 `code_sha256`과 같음. 열람 시점 git `868b714`는 감사 문서·보조 스크립트만 추가된 commit임 |
| 누수 검사 | 28/28 통과 (`F4_unseen/leakage_check.json`) |
| 구간 독립성 | F4 끝 시각 ≤ v6 세 fold의 내부 validation 시작 시각 (`test_v7_window_ends_at_or_before_v6_inner_validation_timestamps`) |
| 지표 독립 재계산 | `report_tools/recompute_test_metrics.py`: 정답, 원본 학습 이력, 모집단을 `events.csv`에서 다시 만들고 저장된 추천 목록으로 지표를 계산. 270행(정책 2 × seed 3 × 모델 9 × K 5) 최대 차이 1.0e-16, 모집단 일치 (`audit/recompute/recompute_v7_F4_unseen.json`) |
| 추천 목록 | 저장된 추천 487,749건(repeat·novel 목록 합계)에서 사용자별 중복 0건. novel 목록에는 원본 학습 기간에 본 상품 0건 |

### 12.2 선택 (내부 validation, seed 42, repeat 294명 / novel 206명)

| 모델군 | 학습 / 평가한 설정 | 선택 | validation 점수 | grid 끝에 걸린 축 |
|---|---|---|---|---|
| ALS | 192 / 192 | f1024 r.02 α80 i16 | 0.2176 | factors(최대), α(최대) |
| BPR | 24 / 168 | f256 lr.005 r.003, 384 epoch | 0.2154 | factors(최대) |
| ALS+BPR | 추가 학습 없음 / 11 | α = .9 (novel .7) | 0.2219 | - |
| GNN | 36 / 952 | 1층 256차원 lr 3e-3, 90 epoch | 0.2169 | layers(최소), dim(최대), lr(최대) |
| Hybrid | 추가 학습 없음 / 8,424 | repeat: 위 GNN, α .5, balanced_union, 후보 500, minmax, ω .7, γ 1.1 | 0.2250 | 후보 수(최소) |
| | | novel: 같은 GNN, α .75, balanced_union, 후보 20,000, RRF, ω .5, γ 1.25 | 0.0612 | α(최대), 후보 수(balanced_union의 최대) |

- 학습 시간: ALS 3,971초, BPR 1,696초, GNN 16,975초 (36개 합).
- **ALS는 grid를 넓히자 validation에서 GNN 최고값보다 높아졌다** (0.2176 vs 0.2169). v6 경계 탐색(`mf_boundary_probe.csv`)에서 본 경향과 같다.
- 사전등록 규칙대로 경계에 다시 걸린 축은 더 넓히지 않았고, `selection.json`의 `grid_edges`에 기록했다. Hybrid 행의 끝 축은 `audit/cycle13/hybrid_boundary_recheck.json` 기준이다: 커밋된 `selection.json`의 `hybrid_at_grid_boundary`는 α 축을 빼고, 후보 수 최대를 `all`로만 봐서(balanced_union의 최대는 20,000) novel 행을 모두 False로 적었다 (I-97, 선택값은 그대로).
- GNN 36개 중 5개가 epoch 상한 규칙(최고 epoch ≥ 390)에 걸렸다 (400 epoch 4개, 390 epoch 1개). 모두 lr ≤ 1e-3이거나 64차원인 설정이고, validation 순위는 6, 15, 24, 30, 33위다 (최고 0.2138, 선택된 GNN 0.2169). Hybrid backbone 상위 3개와 단독 GNN 선택에는 상한 도달이 없어 연장 규칙은 발동하지 않았다. 연장했다면 이 5개의 순위가 바뀌었을지는 확인하지 않았다.

### 12.3 시험 결과 (F4, 세 seed 평균, K = 50)

| 모델 | repeat NDCG | repeat Recall | novel NDCG | novel Recall |
|---|---|---|---|---|
| Popularity | 0.0100 | 0.0322 | 0.0033 | 0.0114 |
| Recent | 0.1632 | 0.2028 | 0 (정의상) | 0 |
| RecentPop | 0.1647 | 0.2104 | 0.0033 | 0.0114 |
| ALS | 0.1726 | 0.2464 | 0.0368 | 0.0733 |
| BPR | 0.1901 | 0.2474 | 0.0351 | 0.0796 |
| ALS+BPR | 0.1868 | 0.2473 | 0.0389 | 0.0842 |
| GNN | 0.1824 | 0.2464 | 0.0334 | 0.0756 |
| Hybrid | **0.1942** | **0.2550** | **0.0402** | **0.0907** |

- 모집단: repeat 413명(학습 벡터 있음 236, 도달 가능한 정답 있음 221), novel 306명(163 / 146), 카탈로그 26,817개.
- Hybrid가 맞힌 사용자: repeat 평균 127.3명(seed 합집합 129명), novel 38.7명. 학습 모델마다 repeat 사용자의 68 – 70%가 Recall 0이다 (`per_user_distribution.csv`).
- 학습 모델의 seed 표준편차는 0.0003 – 0.0104다.

### 12.4 쌍대 비교 (Holm 보정, α = 0.05)

| 비교 | repeat NDCG | repeat Recall | novel NDCG | novel Recall |
|---|---|---|---|---|
| Hybrid − ALS+BPR | +0.0074 비유의 (p_holm .16) | +0.0077 비유의 (.16) | +0.0013 비유의 (.50) | +0.0065 비유의 (.30) |
| Hybrid − GNN | +0.0118 **유의** (.006) | +0.0086 비유의 (.07) | +0.0068 비유의 (.14) | +0.0151 **유의** (.010) |
| Hybrid − BPR | +0.0040 비유의 (.30) | +0.0076 **유의** (.029) | +0.0052 비유의 (.33) | +0.0111 비유의 (.33) |
| Hybrid − ALS | +0.0215 **유의** (.001) | +0.0086 비유의 (.16) | +0.0034 비유의 (.33) | +0.0174 **유의** (.001) |
| Hybrid − Popularity / Recent / RecentPop | 모두 유의 | 모두 유의 | 모두 유의 | 모두 유의 |

GNN·ALS+BPR 대비 95% CI 반폭은 0.004 – 0.010이다. 사전 추정은 plan의 `power_note`에서 0.005 – 0.014, `v7_feasibility.md` §3 표에서 0.006 – 0.010이었다.

### 12.5 해석

- **I-23의 질문("baseline을 충분히 튜닝해도 결론이 유지되는가")에 대한 답**: 유지된다. ALS·BPR·GNN grid를 모두 넓히자 ALS는 validation에서 GNN을 앞섰지만, 시험에서 Hybrid와 ALS+BPR의 차이는 네 칸 모두 비유의다. v6에서는 NDCG 6칸 모두 비유의, Recall 6칸 중 3칸 유의였다. 따라서 공통된 결론은 "Hybrid가 가장 강한 baseline보다 일관되게 유의하게 높지는 않다"이다.
- Hybrid는 네 칸 모두에서 평균 1위다. 평균이 가장 높은 baseline은 칸마다 다르다: repeat에서는 BPR(NDCG 0.1901, Recall 0.2474. Recall은 ALS+BPR 0.2473과 사실상 같음), novel에서는 ALS+BPR(0.0389, 0.0842). 이 baseline 대비 Hybrid는 repeat Recall(Hybrid − BPR +0.0076, p_holm .029)만 유의하고, 나머지 세 칸은 비유의다. 네 칸 모두에 Hybrid가 유의하게 앞서지 못한 학습 baseline이 하나 이상 있다 (repeat NDCG: BPR·ALS+BPR, repeat Recall: ALS·ALS+BPR·GNN, novel NDCG: 넷 모두, novel Recall: BPR·ALS+BPR). 따라서 v7도 "Hybrid가 모든 baseline보다 우수하다"는 주장을 뒷받침하지 않는다.
- 95% CI는 Holm 보정을 하지 않은 값이다. 그래서 CI가 0을 포함하지 않는데 Holm 기준으로는 비유의한 칸이 있다 (repeat Hybrid − ALS Recall, Hybrid − ALS+BPR NDCG·Recall, Hybrid − GNN Recall, novel Hybrid − ALS+BPR Recall, Hybrid − GNN NDCG). 유의성 판정은 p_holm만 따른다.
- novel의 유의한 결과(Hybrid − GNN Recall, Hybrid − ALS Recall)는 repeat로 고른 GNN·ALS 대비다. novel validation으로 골랐다면 GNN(36개 중 선택 설정은 19위)이 validation에서 Hybrid보다 높았다 (0.0659 vs 0.0612, §8 항목 5). 그래서 v7은 novel에서 Hybrid가 GNN·ALS보다 낫다는 근거가 되지 못한다. "baseline을 충분히 튜닝해도"라는 위 답도 repeat 기준 튜닝에 대한 것이다.
- 단일 fold, 사용자 수백 명이라 0.01 이하의 차이는 이 실험으로 판정할 수 없다 (사전에 적어 둔 검정력 한계와 같다).
- validation과 시험의 순위가 다르다. ALS는 validation에서 단일 모델군(ALS·BPR·GNN) 중 1위였지만(0.2176 vs GNN 0.2169, BPR 0.2154. 결합 모델인 ALS+BPR 0.2219, Hybrid 0.2250은 더 높음), 시험 repeat NDCG에서는 단일 모델군 중 최하위다 (ALS 0.1726, GNN 0.1824, BPR 0.1901). validation 모집단이 294명이라 선택 자체에 잡음이 크다.
- `Hybrid_ALS_only`(같은 후보에서 결합 점수의 MF 부분만 ALS로 바꾼 ablation, 사전등록 비교 대상 아님, 검정 안 함)는 repeat NDCG 0.1814로 Hybrid보다 0.0127 낮다. v6에서는 F3 novel(두 지표)과 F1 repeat Recall에서 반대였다. 후보 생성에는 BPR이 남아 있고 검정도 하지 않았으므로, 이 결과로 BPR의 기여를 판단할 수는 없다.
- v7의 수치를 v6 fold와 평균하지 않는다. grid와 구간이 달라 같은 실험이 아니다.
- 남은 한계: ALS factors·α, BPR factors, GNN layers·dim·lr이 v7에서도 끝에 걸렸다. 사전등록 규칙에 따라 다시 넓히지 않았다. Hybrid grid는 v6와 같게 두었고(v6에서도 끝에 걸린 축이 있었음), v7에서도 repeat 후보 수(최소), novel α(최대)·후보 수(balanced_union의 최대)가 끝에 걸렸다 (§12.2). 따라서 "baseline을 충분히 튜닝해도"라는 위 답은 Hybrid 쪽 결합 설정까지 충분히 탐색했다는 뜻이 아니다.

## 13. Cycle 4: 재개·시험 열람 가드와 기록 (I-39 – I-46)

v6·v7 실행 이후의 코드 수정이다. v6·v7의 선택과 시험 결과는 바뀌지 않는다 (각 항목의 근거는 `issue_register.md`).

| 검사 | 결과 |
|---|---|
| 새 테스트 `test_cycle4_guards.py` | 12/12 통과. 전체 94/94 |
| 새 테스트의 결함 검출력 (옛 동작을 사본에 다시 넣음) | 6/6 검출: 누수 판정 무시(G1), selection 해시 미비교(G2), 커밋 기록 가드 없음(G3), 상한 연장 1회(G4), 스레드 `setdefault`(G5), `end_fraction` 무시(G6). 이 G 변이는 임시 스크립트로 돌려 결과 파일을 남기지 않았다 (Cycle 5의 H 변이는 커밋된 도구와 결과 파일이 있음 → §14) |
| G5 처음 결과 | 놓침. 환경변수를 읽는 테스트였는데, `experiments.py`가 BLAS 로드 뒤 변수를 1로 다시 써서 옛 동작에서도 '1'이 읽혔다. 로드된 라이브러리의 실제 스레드 수를 읽도록 테스트를 고친 뒤 검출 |
| 커밋된 기록으로 시험 열람 가드 | Cycle 4 코드에서 (일회성, 출력 없음): v6·v7 plan 자신은 통과, 새 plan은 F1_hist·F2_mid·F3_early·F4_unseen 모두 거부. Cycle 5부터는 코드도 같아야 예외라, 현재 코드에서는 v6·v7 plan도 거부됨 (§14) |
| 누수 검사 `--end-fraction` (실제 데이터) | 원본 전체 + 0.36으로 v7 F4를 검사: 28개 모두 통과, 커밋된 `leakage_check.json`과 검사값까지 같음. 옵션 없이 실행하면 12개 거짓 실패 (`audit/cycle4/`) |
| 선택·시험 경로의 점수 정밀도 (I-46) | v7 F4 내부 validation, 두 정책 × ALS/BPR: 목록 100% 동일, 지표 차이 0 (`audit/cycle4/score_precision_F4_val.json`) |
| 변이 테스트 17개 (§11) | 수정된 코드로 다시 실행한 결과가 공식 결과와 완전히 같음 (`audit/cycle4/mutation_results_cycle4.json`, 00:42 – 01:31 KST). 커밋 전 작업 폴더 코드로 실행했고(05b9c65 커밋 01:32), 그 사본과 커밋이 같다는 산출물은 없다. 커밋된 코드로는 Cycle 5(fd9bcca)·Cycle 7(8cf8215) 재실행이 있다 |

## 14. Cycle 5: 가드 빈틈과 문서 정정 (I-48 – I-57)

Cycle 4 커밋 뒤 처음부터 다시 한 독립 감사(코드·문서)에서 찾은 항목이다. v6·v7의 선택과 시험 결과는 바뀌지 않는다.

| 검사 | 결과 |
|---|---|
| 새 테스트 `test_cycle5_guards.py` | 12/12 통과. 전체 106/106 (v7 venv) |
| 가드 변이 (옛 동작 12개를 사본에 하나씩 넣음, `report_tools/guard_mutation_check.py`) | 12/12 검출 (`audit/cycle5/guard_mutants.json`. 이 파일은 Cycle 6에서 H13·H14를 더한 14/14 결과로 다시 만들어짐 → §15). 첫 실행에서는 H11(사후 분석 기준 파일을 먼저 씀), H12(selection의 plan 미확인)가 살아남아, 각각 전용 테스트를 추가한 뒤 다시 실행 (첫 실행 결과 파일은 덮어써져 남아 있지 않음) |
| 실제 기록으로 가드 (실제 원본 해시, 이 PC) | 커밋된 `audit/results_v6`·`results_v7`과 커밋 안 된 `audit_runs/v7`을 모두 찾음. 현재 코드에서는 v6 plan, v7 plan, 새 plan 모두 F1–F4가 충돌 (v6·v7은 코드가 바뀌어 재현 예외가 아님. plan commit을 checkout하면 예외). 당시 출력은 없음, Cycle 7에 다시 확인 (§16) |
| plan 코드 확인 (`verify_plan_code.py`) | v6 25개, v7 26개 해시가 plan commit 내용과 일치 (종료 코드 0). `--require-unchanged`는 둘 다 1 (바뀐 파일과 추가된 파일이 있음) |
| 캐시 재사용 규칙을 실제 v7 root에 적용 (I-51, I-60) | plan·selection·`validation_gnn.json`에서 다시 만든 요청 GNN 39개(validation 36 + refit 3)가 모두 있고 학습 설정과 일치, MF 학습 4개 일치 (`audit/cycle5/cache_identity_v7.json`. Cycle 5의 첫 도구는 같은 파일끼리 비교해 의미가 없었고, Cycle 6에서 고쳐 다시 만듦) |
| v7 ALS의 BLAS 스레드 (I-24, I-41 정정) | F4 seed 42 ALS를 같은 설정으로 다시 학습: 1스레드는 저장된 v7 인자와 비트 단위로 같고, 24스레드는 최대 차이 0.49, Top-50 목록 완전 일치 1.3%(평균 겹침 98.6%) (`audit/cycle5/determinism_v7_blas1.csv`, `determinism_v7_blas24.csv`). BPR(GPU)은 둘 다 비트 단위로 같음. 2–23스레드는 검사 안 함 |
| 변이 테스트 17개 (§11) | Cycle 5 코드로 다시 실행: 대상 문구 17개 모두 한 번씩 있음, 수정 안 한 사본은 두 묶음 통과, 결과가 공식 `mutation_results.json`과 완전히 같음 (17/17 검출, `audit/cycle5/mutation_results_cycle5.json`, 56분 36초. 다른 검사와 동시 실행). 실행은 fd9bcca 커밋(02:37 KST) 전인 02:22 KST에 작업 폴더를 복사해 시작했다. 변이 사본은 매번 원래 내용으로 되돌려지므로, 남은 사본 30개 파일이 fd9bcca와 줄바꿈 외에 모두 같음을 Cycle 6에 확인했다 (`audit/cycle6/mutation_src_vs_fd9bcca.json`) |
| Cycle 4 점수 정밀도 결과 (I-55) | 고친 도구(1스레드)로 다시 실행한 결과가 커밋된 `score_precision_F4_val.json`과 완전히 같음 (`audit/cycle5/score_precision_F4_val_blas1.json`) |
| 커밋 후 재검증 (fd9bcca) | 새 worktree + `requirements-v7-lock.txt` venv: 105 통과, 1 건너뜀(원본 `events.csv`가 worktree 기준 상대경로에 없어 v7 원본 경계 테스트가 건너뜀. 원래 작업 폴더에서는 통과). `SHA256SUMS.txt` v6 60/60, v7 28/28 (실행 출력은 저장소에 없음. Cycle 6부터는 `audit/cycle6/`에 보존) |
| 문서 (I-57) | 12건 정정. 결론에 영향을 주는 것은 v7의 "가장 강한 baseline" 서술(§12.5): repeat에서는 BPR이 평균 최고이고, 그 대비 repeat Recall만 유의. "모든 칸에서 Hybrid가 유의하게 앞서지 못한 학습 baseline이 있다"는 결론은 같음 |

## 15. Cycle 6: 시험 기록 비교 기준, 캐시 확인 도구, 문서 정정 (I-58 – I-62)

Cycle 5 커밋 뒤 처음부터 다시 한 독립 감사(코드·문서)에서 찾은 항목이다. v6·v7의 선택과 시험 결과는 바뀌지 않는다. 각 항목은 원본 파일과 실제 실행으로 다시 확인한 뒤 고쳤다.

| 검사 | 결과 |
|---|---|
| 회귀 테스트 | `test_cycle5_guards.py` 3개 추가(events 해시만 비교, git 상태 열, 요청 설정과 캐시 비교), 전체 109/109 (v7 venv) |
| 가드 변이 H1–H14 (H13: 원본 4개 파일 바이트를 모두 비교, H14: git 출력 전체 strip) | 14/14 검출, 수정 안 한 사본 26 통과 (`audit/cycle5/guard_mutants.json`, 실행 출력 `audit/cycle6/guard_mutants_cycle6_log.txt`) |
| 실제 원본으로 시험 기록 가드 (I-59) | 원본 `events.csv`의 원래·LF·CRLF 해시에 v6·v7 plan의 events 해시가 들어 있고, 커밋된 v6·v7 기록과 로컬 `audit_runs/v7`의 기록 5건이 모두 잡힘. 현재 코드에서 v6 plan, v7 plan, 새 plan 모두 F1–F4 충돌 (당시 출력은 없음, Cycle 7에 다시 확인 → §16) |
| 캐시 확인 도구 (I-60) | plan·`validation_gnn.json`·`selection.json`에서 다시 만든 요청 GNN 39개와 MF 4개가 v7 캐시와 모두 일치, 불일치 0 (`audit/cycle5/cache_identity_v7.json`) |
| 변이 테스트 17개 대상 문구 | 현재 코드에서 17개 모두 한 번씩 있음 |
| Cycle 5 변이 재실행의 코드 | 남은 사본 30개 파일이 fd9bcca와 같음 (`audit/cycle6/mutation_src_vs_fd9bcca.json`) |
| 환경 lock | 두 venv의 `uv pip freeze`가 같음 (36개, `audit/cycle6/freeze_*.txt`) |
| plan 코드 확인 (커밋 57dc03b) | v6 25개, v7 26개 해시가 plan commit과 일치 (종료 코드 0). plan 뒤에 바뀐 파일: v6 `leakage_check.py`·`rolling_protocol.py`·`test_v6_smoke.py`, v7 `leakage_check.py`·`rolling_protocol.py`. `--require-unchanged`는 둘 다 1 (`audit/cycle6/plan_code_v6.json`, `plan_code_v7.json`) |
| 커밋 후 재검증 (57dc03b) | 새 worktree + `requirements-v7-lock.txt` venv: 108 통과, 1 건너뜀(원본 `events.csv`가 worktree 기준 상대경로에 없음. 원래 작업 폴더에서는 같은 코드로 통과) (`audit/cycle6/postcommit_57dc03b_pytest.txt`). `SHA256SUMS.txt` v6 60/60, v7 28/28 (`postcommit_57dc03b_sha256.txt`) |
| 문서 (I-58, I-62) | `Hybrid_ALS_only` 설명을 코드에 맞춤(후보는 같고 결합 점수의 MF만 ALS). 문서 오류 11건 정정, 저장소로 확인할 수 없는 기록에 출처 표시. 결론은 바뀌지 않음 |

## 16. Cycle 7: novel 정책 튜닝 비대칭, 병렬 실행 가드, 문서 정정 (I-63 – I-68)

Cycle 6 커밋 뒤 처음부터 다시 한 독립 감사(코드·문서)에서 찾은 항목이다. 각 항목은 코드·커밋된 파일·실제 실행으로 다시 확인한 뒤 고쳤다. v6·v7의 선택과 시험 결과는 바뀌지 않는다. 결론의 해석은 바뀐다: novel 정책의 Hybrid − GNN·ALS 유의 결과는 repeat로 튜닝된 비교 대상 대비로 한정된다 (§8 항목 5).

| 검사 | 결과 |
|---|---|
| novel 튜닝 비대칭 (I-63, validation만) | `report_tools/policy_tuning_gap.py`가 커밋된 선택 파일로 재구성한 repeat 선택은 GNN·ALS·BPR 모두 repeat 1위(테스트. Cycle 7 판의 ALS·BPR 확인은 순환적이어서 Cycle 8에 `selection.json` 설정으로 고침 → §17). novel로 골랐다면 GNN F2·F3·F4, ALS 네 fold 모두(F2는 +0.00006)의 validation 점수가 오르고, F4에서는 GNN이 Hybrid보다 높음 (`audit/cycle7/policy_tuning_gap_v6.json`, `_v7.json`) |
| 회귀 테스트 | 114/114 (v7 venv. 이 실행의 출력은 저장하지 않음. 커밋 후 lock venv 결과는 아래 행) |
| 가드 변이 H1–H18 (H15 시작 때만 확인, H16 다른 원본으로 재현 예외, H17 grid 캐시를 존재만으로 신뢰, H18 git 없이 plan 생성) | 18/18 검출, 수정 안 한 사본 30 통과 (`audit/cycle7/guard_mutants.json`, `guard_mutants_cycle7_log.txt`). 22분 14초는 셸 실행 시간이고 로그에는 시각이 없음. 커밋 0ef6daa 전의 작업 폴더 코드로 실행했고, 그 사본과 커밋이 같다는 산출물은 없음 (Cycle 8에 커밋된 코드로 다시 실행 → §17) |
| 실제 원본으로 가드 (`report_tools/guard_real_data_check.py`) | 원본 4개 파일이 v6·v7 plan 기록과 모두 같음. 기록 root 3개(`audit/results_v6`, `results_v7`, `audit_runs/v7`). v6 fold에서는 v6·v7·새 plan 모두 F1–F3 충돌(3건), v7 fold에서는 모두 F4 충돌(2건) (`audit/cycle7/guard_real_data.json`) |
| 변이 테스트 17개 대상 문구 | 0ef6daa 코드에서 17개 모두 한 번씩 있음 |
| 변이 테스트 17개 (§11), 커밋 8cf8215 그대로(작업 폴더 깨끗함) | 공식 `mutation_results.json`과 완전히 같음 (17/17, 51분 25초, `audit/cycle7/mutation_results_8cf8215.json`, `mutation_8cf8215_run.txt`) |
| 커밋 후 재검증 (0ef6daa) | 새 worktree + `requirements-v7-lock.txt` venv: 113 통과, 1 건너뜀(원본 데이터 상대경로 없음) (`audit/cycle7/postcommit_0ef6daa_pytest.txt`). `SHA256SUMS.txt` v6 60/60, v7 28/28 (`postcommit_0ef6daa_sha256.txt`) |
| 문서 (I-68) | 10건 정정, 일회성 확인에 출력 없음 표시, 실제 원본 가드 확인은 다시 실행해 증거로 남김, 투고본 수치 출처 기재 |

## 17. Cycle 8: 실행 중 코드 고정, 기록 파일 손상, grid 캐시 정밀도, 문서 정정 (I-69 – I-73)

Cycle 7 커밋 뒤 처음부터 다시 한 독립 감사(코드·문서)에서 찾은 항목이다. 새 P0·P1은 없다. 각 항목은 코드·커밋된 파일·실제 실행으로 다시 확인한 뒤 고쳤다. v6·v7의 선택과 시험 결과는 바뀌지 않는다.

| 검사 | 결과 |
|---|---|
| `policy_tuning_gap.py`의 ALS·BPR 선택 재구성 (I-69) | Cycle 7 판은 "repeat 선택 = repeat 1위"로 정의해 테스트가 순환적이었음. `selection.json`의 MF 설정을 grid에서 한 행으로 찾도록 고침(못 찾거나 여러 행이면 오류). 출력 값은 Cycle 7과 같음 (`audit/cycle8/policy_tuning_gap_v*.json`) |
| 실행 중 코드 고정 (I-70) | `require_frozen_code`를 실행 시작, GNN 작업 시작 직전, 새 시험 열람 직전에 확인. v7 산출물 43개에 찍힌 코드 해시 731개가 plan과 모두 같음 (`audit/cycle8/stamp_code_v7.json`). v6 실행 폴더는 v6 PC에만 있어 확인 못 함 |
| 자기 root 오인 (I-71) | 시험 열람 직전의 다른 root 검사에서 자기 root 제외. 코드 확인이 먼저라 원인 메시지가 정확함 |
| grid 캐시 정밀도 (I-72) | v7 grid 6개(각 2,808행): 기본 파서는 모든 행의 값을 몇 ulp 바꾸고, round-trip은 다시 쓰면 줄바꿈(CRLF) 외에는 파일과 같음. 점수 순서는 같음. `selection.json`의 수치 48개 중 26개가 round-trip 값에서 그대로 나오고 기본 파서 값에서는 11개만 나옴 (`audit/cycle8/grid_cache_precision_v7.json`) |
| 기록 파일 손상 (I-73) | 잘린 접근 기록 줄은 그 기록의 모든 fold를 연 것으로 봄(겹치는 구간만 거부, 겹치지 않는 fold는 통과). 읽을 수 없는 `plan.json`·자기 접근 기록은 파일을 지목하는 오류 |
| 실제 원본으로 가드 (`guard_real_data_check.py`, b9e7134 커밋 직전 작업 폴더) | Cycle 7 출력(`audit/cycle7/guard_real_data.json`)과 완전히 같음. 그래서 새로 저장하지 않음 (다시 만들려면 reproducibility 9단계) |
| 회귀 테스트 | 작업 폴더 120/120 (v7 venv, 출력은 저장하지 않음). 커밋 b9e7134의 새 worktree + lock venv: 119 통과, 1 건너뜀(원본 데이터 상대경로 없음). 출력 머리에 commit·worktree·venv·시각 기록 (`audit/cycle8/postcommit_b9e7134_pytest.txt`). `SHA256SUMS.txt` v6 60/60, v7 28/28 (`postcommit_b9e7134_sha256.txt`) |
| 가드 변이 H1–H27 (H19 시험 열람 전 코드 미확인, H20 GNN 작업 전 코드 미확인, H21 자기 root 미제외, H22 잘린 줄에서 예외, H23 잘린 줄 무시, H24·H25 읽을 수 없는 파일을 지목하지 않음, H26 기본 float 파서, H27 열람 직전 검사가 줄바꿈 사본을 못 봄) | 커밋 b9e7134 코드로 실행(출력에 HEAD와 커밋 안 된 파일 기록. 커밋 안 된 것은 문서뿐), 27/27 검출, 수정 안 한 사본 36 통과. 20:38:30–21:14:36 UTC (`audit/cycle8/guard_mutants.json`, `guard_mutants_cycle8_log.txt`). Cycle 7의 H1–H18 실행이 커밋 전 코드였던 점은 이 실행으로 대체됨 |
| 변이 테스트 17개 (§11), 커밋 343e75f 그대로(작업 폴더 깨끗함) | 공식 `mutation_results.json`과 완전히 같음 (17/17, 20:14:17–21:08:52 UTC, `audit/cycle8/mutation_results_343e75f.json`, `mutation_343e75f_run.txt`). b9e7134에서 17개 대상 문구는 모두 한 번씩 있음 |
| novel 튜닝 비대칭 해석 (I-63 보완) | GNN의 epoch는 공통 기준으로 정해지므로 GNN의 novel 최고값은 하한임을 §8 항목 5에 적음 |
| 문서 (I-69) | `manuscript_alignment.md`·두 README에 I-63 한정 추가, ALS+BPR 조건, ALS 네 fold, CPU/GPU 차이 범위, 머리말 PC 범위, `plan_code` 참조, §7 보조 실행 표, 출력 없는 실행 표시 |

## 18. Cycle 9: 재현 예외와 selection, 실제 시험 구간, 완료 파일, 문서 정정 (I-74 – I-79)

Cycle 8 커밋 뒤 처음부터 다시 한 독립 감사(코드·문서)에서 찾은 항목이다. 새 P0·P1·P2는 없고 P3만 있다 (코드 5건, 문서 7건). 각 항목은 코드·커밋된 파일·실제 실행으로 다시 확인한 뒤 고쳤다. v6·v7의 선택과 시험 결과는 바뀌지 않는다.

| 검사 | 결과 |
|---|---|
| 재현 예외와 selection (I-75) | 시험 열람 직전에는 정확한 재현 기록도 같은 fold를 같은 selection 해시로 연 경우에만 예외. 테스트(다른 selection이면 거부, 같으면 통과) |
| 완료 표시 파일 (I-76) | 세 파일과 grid 캐시를 임시 이름에 쓴 뒤 교체. 기존 파일은 모두 완전함(`test_metrics.csv` 5개 모두 머리 + 270행, 사후 분석 9행, IF ablation 18행) |
| 스탬프 확인 도구 (I-77) | `sha256-lf` plan은 plan commit 내용으로 비교. v7 다시 실행: 43개 파일, 731/731 일치 (`audit/cycle9/stamp_code_v7.json`) |
| 재학습 스레드 오류 (I-78) | `start_background`가 작업의 오류를 그대로 다시 냄 |
| 실제 시험 구간 (I-79, `guard_real_data_check.py`) | 분할 규칙으로 원본에서 계산한 구간이 v6 F1–F3·v7 F4 기록 구간과 같음. F4와 명목상 맞닿는 새 fold 5개(끝 0.39, 0.40, 0.43, 0.44, 0.45)는 모두 F4와 이벤트 1개를 공유: 실행 시작 검사(명목 비율)는 0건, 시험 열람 직전 검사(타임스탬프)는 2건(커밋본·실행 폴더) 거부. 기존 항목(raw 일치, 기록 root, v6·v7·새 plan 충돌)은 Cycle 7 출력과 같음 (`audit/cycle9/guard_real_data.json`) |
| 회귀 테스트 | 작업 폴더 127/127 (v7 venv, 출력은 저장하지 않음). 커밋 ce97790의 새 worktree + lock venv: 126 통과, 1 건너뜀(원본 데이터 상대경로 없음) (`audit/cycle9/postcommit_ce97790_pytest.txt`). `SHA256SUMS.txt` v6 60/60, v7 28/28 (`postcommit_ce97790_sha256.txt`) |
| 가드 변이 H1–H35 (H28 selection 무관 재현 예외, H29 기록 구간 무시, H30 기록 없으면 맞닿음을 분리로, H31·H32 열람 직전 검사에 selection·구간을 넘기지 않음, H33 완료 CSV 직접 쓰기, H34 일반 오류 메시지, H35 객체가 아닌 기록 줄 허용. H2·H15·H22·H23은 바뀐 코드에 맞게 대상 문구 수정) | 커밋 ce97790 코드로 실행(커밋 안 된 것은 문서뿐), 35/35 검출, 수정 안 한 사본 42 통과. 21:40:59–22:25:53 UTC (`audit/cycle9/guard_mutants.json`, `guard_mutants_cycle9_log.txt`) |
| 변이 테스트 17개 (§11), 커밋 79b628d 그대로(작업 폴더 깨끗함) | 공식 `mutation_results.json`과 완전히 같음 (17/17, 21:15:30–22:08:20 UTC, `audit/cycle9/mutation_results_79b628d.json`, `mutation_79b628d_run.txt`). ce97790에서 17개 대상 문구는 모두 한 번씩 있음 |
| 문서 (I-74) | 감사가 보고한 7건과 함께 고친 7건 (issue_register I-74 행) |

## 19. Cycle 10: 정확한 재현의 selection 비교, 백그라운드 작업 정지, 재현 명령의 출력 경로 (I-80 – I-84)

Cycle 9 커밋 뒤 처음부터 다시 한 독립 감사(코드·문서)에서 찾은 항목이다. 새 P0·P1은 없다 (코드 P2 1건, P3 3건, 문서 P3 5건). 각 항목은 코드·커밋된 파일·실제 실행으로 다시 확인한 뒤 고쳤다. v6·v7의 선택과 시험 결과는 바뀌지 않는다.

| 검사 | 결과 |
|---|---|
| 정확한 재현의 selection 비교 (I-81, I-75 정정) | (감사자의 원본부터 재현 스크립트를 고친 코드로 다시 실행해 두 번째 root가 완료됨, 출력 저장 안 함. 같은 내용을 아래 테스트가 확인) `selection.json`에는 봉인 시각·절대경로·학습 시간이 들어 있어 원본부터 다시 돌린 재현은 파일 해시가 항상 다름. 이제 결정 내용(`selection_decision`)으로 비교. 합성 원본으로 v7 프로토콜 두 root를 끝까지 돌리는 테스트: 두 번째 root가 시험 평가까지 마침, 파일 해시는 다르고 결정은 같음. 결정이 다르거나(γ) 기록의 파일이 로그와 다르면 거부 |
| 실제 기록의 결정 (`report_tools/guard_real_data_check.py`의 `decisions`) | 기록 5개(v6 F1–F3, v7 커밋본, `audit_runs/v7`) 모두 접근 로그의 selection 해시가 보관된 `selection.json`과 같고(fold마다 서로 다른 해시 1개), v7 커밋본과 실행 폴더의 결정이 같음. 나머지 항목(raw 일치, 기록 root, 구간, 경계 fold, 충돌)은 Cycle 9 출력과 같음 (`audit/cycle10/guard_real_data.json`) |
| 기록의 이상한 분할 필드 (I-82) | null·문자열·실수 값이면 구간 없음으로 처리(맞닿음도 겹침). 실제 기록은 모두 정수 |
| 백그라운드 GNN 작업 (I-83) | 주 단계가 실패하면 stop을 걸고 실행 중인 작업을 종료한 뒤 오류를 냄. 작업의 BaseException도 원인 그대로 |
| 알 수 없는 fold 이름 (I-84) | plan을 만들기 전에 멈춤 |
| 재현 명령의 출력 경로 (I-80) | 재현 5–10단계와 두 변이 도구의 기본 출력이 커밋된 증거를 덮어쓰지 않도록 `audit_runs/`로. 4e166c5 전에 7–10단계를 `audit_runs/repro/`에 실행해 출력 5개가 당시의 커밋된 증거와 같았음(9단계는 `audit/cycle9/guard_real_data.json`과 비교. 출력은 gitignore 대상이고 비교 결과는 저장하지 않음). 남은 경로와 저장된 비교는 Cycle 11 (I-85) |
| 회귀 테스트 | 작업 폴더 136/136 (v7 venv, 출력은 저장하지 않음). 커밋 4e166c5의 새 worktree + lock venv: 135 통과, 1 건너뜀(원본 데이터 상대경로 없음) (`audit/cycle10/postcommit_4e166c5_pytest.txt`). `SHA256SUMS.txt` v6 60/60, v7 28/28 (`postcommit_4e166c5_sha256.txt`) |
| 가드 변이 H1–H44 (H36 열람 직전 검사가 파일 해시를 넘김, H37 기록 파일과 로그의 연결 없음, H38 봉인 시각이 결정에 들어감, H39 분할 필드를 그대로 변환, H40 주 단계 실패 때 정지 없음, H41 작업을 종료하지 않음, H42 작업 루프가 stop을 무시, H43 알 수 없는 fold 허용, H44 worker가 일반 오류만 보관. H15·H20·H31·H34는 바뀐 코드에 맞게 대상 문구 수정. 끝까지 돌리는 재현 테스트는 시간 때문에 제외) | 커밋 4e166c5 코드로 실행(시작 때 커밋 안 된 것은 문서 4개뿐. 실행 중에 가드 테스트가 쓰지 않는 `guard_real_data_check.py`를 고쳤음), 44/44 검출, 수정 안 한 사본 50 통과. 23:06:25–23:58:51 UTC (`audit/cycle10/guard_mutants.json`, `guard_mutants_cycle10_log.txt`) |
| 변이 테스트 17개 (§11), 커밋 ac9b865 그대로(시작 때 작업 폴더 깨끗함. 사본의 `rolling_protocol.py`·`policy_evaluation.py`가 ac9b865와 같음을 확인, 출력 저장 안 함) | 공식 `mutation_results.json`과 완전히 같음 (17/17, 22:27:07–23:17:51 UTC, `audit/cycle10/mutation_results_ac9b865.json`, `mutation_ac9b865_run.txt`). 4e166c5에서 17개 대상 문구는 모두 한 번씩 있음 |

## 20. Cycle 11: 재현 명령 출력과 저장된 비교, 원자적 복제, 강제 종료 때 GNN 작업, 모집단·후보 생성 서술 (I-85 – I-89)

Cycle 10 커밋 뒤 처음부터 다시 한 독립 감사(코드·문서)에서 찾은 항목이다. 새 P0·P1은 없다 (문서 감사 P2 1건·P3 6건, 코드 감사 P3 4건. 코드 감사의 I-88·I-89는 문서·그림 서술). 각 항목은 코드·커밋된 파일·실제 실행으로 다시 확인한 뒤 고쳤다. v6·v7의 선택과 시험 결과는 바뀌지 않는다.

| 검사 | 결과 |
|---|---|
| 재현 명령 출력과 저장된 비교 (I-85) | 재현 4, 7–10단계·plan 코드 확인·재계산을 8d8d0d5의 도구로 `audit_runs/repro/`에 다시 실행해 `compare_outputs`로 커밋된 증거와 비교: 29행(다시 만든 출력 28개. plan 코드 확인 출력 하나를 Cycle 3·Cycle 6 증거 두 곳과 비교했고, 11단계 명령은 Cycle 3 쪽만 비교해 28행) 모두 같음(plan 코드 확인은 시간에 따라 바뀌는 값을 빼고 `commit,scheme,files,all_plan_hashes_match_commit`만 비교) (`audit/cycle11/repro_compare.json`, `repro_run.txt`). 이 비교의 그림은 I-89로 구조도를 고치기 전 것과 같음 |
| 원자적 복제 (I-86) | 6번째 hard link에서 끊으면 대상 폴더가 생기지 않고, 다시 실행하면 하위 `validation`까지 완성되며 `.partial`이 남지 않음(테스트). 실제 v7 실행 폴더 `audit_runs/v7`의 protocol 폴더 51개가 모두 manifest의 파일을 다 갖고 있고 `.partial`이 없음(한 번 확인, 출력 저장 안 함) |
| 강제 종료 때 GNN 작업 (I-87) | 실제 부모 프로세스가 자식을 `kill_with_parent`로 묶은 뒤 부모를 `TerminateProcess`로 끝내면 자식의 heartbeat 파일이 더 자라지 않음(테스트). 모든 작업이 묶이고, Popen이 실패하면 로그가 닫힘(테스트). 원본부터 두 root를 끝까지 돌리는 재현 테스트가 실제 GNN 작업 프로세스로 이 경로를 지남. POSIX에는 이 보호가 없음 |
| 평가 모집단 서술 (I-88) | `manuscript_alignment.md` 공개 사항 2를 `data_protocol.py`의 `reference`(필터 전 원본 학습 기간의 사용자·상품)와 `policy_evaluation.py` 머리말(정답 집합, 0점 처리, novel 제외)에 맞춤 |
| 후보 생성 서술 (I-89) | 커밋된 `selection.json` 4개에서 봉인된 설정 8개 중 `balanced_union` 5개(F1 novel, F2 repeat, F3 novel, F4 repeat·novel), RRF 결합 2개(F1 novel, F4 novel)를 다시 셈. 구조도 2번 상자와 식 8 행에 두 방식을 적음. 그림을 다시 만들어 v6·v7 52개 파일 중 `pipeline.pdf/png/svg` 6개만 바뀜(`compare_outputs`, 출력 저장 안 함) |
| 회귀 테스트 | 작업 폴더 140/140 (v7 venv, 출력은 저장하지 않음). 커밋 318186b의 새 worktree + lock venv: 139 통과, 1 건너뜀(원본 데이터 상대경로 없음) (`audit/cycle11/postcommit_318186b_pytest.txt`). `SHA256SUMS.txt` v6 60/60, v7 28/28 (`postcommit_318186b_sha256.txt`) |
| 가드 변이 H1–H48 (H45 대상 폴더에 바로 링크, H46 작업을 job object에 넣지 않음, H47 kill-on-close 없는 job object, H48 실행 실패 때 로그를 닫지 않음. 끝까지 돌리는 재현 테스트는 시간 때문에 제외) | 커밋 318186b 코드로 실행(시작 때 커밋 안 된 것은 문서 6개, 구조도 6개, `audit/cycle11/`뿐이고 실행 중에 코드를 고치지 않음), 48/48 검출, 수정 안 한 사본 53 통과. 02:43:44–03:46:38 UTC (`audit/cycle11/guard_mutants.json`, `guard_mutants_cycle11_log.txt`) |
| 변이 테스트 17개 (§11), 커밋 8d8d0d5 그대로(사본의 `rolling_protocol.py`·`policy_evaluation.py`가 8d8d0d5와 같음, `audit/cycle11/mutation_scratch_vs_commit.json`) | 공식 `mutation_results.json`과 완전히 같음 (17/17, 23:59:52–00:47:48 UTC, `audit/cycle11/mutation_results_8d8d0d5.json`, `mutation_8d8d0d5_run.txt`) |

## 21. Cycle 12: 전처리 완료 표시, 빈 fold 목록, 비교 도구의 거짓 통과, job object 오류, 문서 정정 (I-90 – I-95)

Cycle 11 커밋 뒤 처음부터 다시 한 독립 감사(코드·문서)에서 찾은 항목이다. 새 P0·P1·P2는 없다 (코드 감사 P3 4건, 문서 감사 P3 3건, 코드 감사의 미확정 의심 1건을 공개로 처리). 코드 감사자는 v7 시험 지표와 paired test를 사용자별 파일에서 독립적으로 다시 계산해 커밋된 값과 1e-16 수준까지 같음을 확인했다(감사자의 임시 출력, 저장 안 함). 각 항목은 코드·커밋된 파일·실제 실행으로 다시 확인한 뒤 고쳤다. v6·v7의 선택과 시험 결과는 바뀌지 않는다.

| 검사 | 결과 |
|---|---|
| 전처리 완료 표시 (I-90) | 바깥 `manifest.json`을 마지막에 한 번만 씀. `finalize_outer`에서 끊으면 바깥 manifest가 없고 재개는 "Incomplete preprocessing output"으로 멈춤, 새로 하면 `input_hashes`·`raw_valid` 행이 있음(테스트). 실제 원본으로 v7 F4 전처리를 고친 코드로 다시 해(39초) 기존 `audit_runs/v7/F4_unseen/prepared`와 비교: 바깥·validation 모두 파일 해시, `protocol_id`, 원본 입력 해시가 같고, 다른 manifest 키는 폴더 경로(`config`)와 전처리 소스 해시뿐 (`audit/cycle12/prep_check_v7_F4.txt`, 실행한 스크립트 포함) |
| 빈 fold 목록 (I-91) | `run(folds=[])`은 plan을 만들기 전에 멈추고, 명령줄 `--folds`만 주면 argparse가 종료 코드 2로 거부(테스트) |
| 비교 도구 (I-92) | 0개 비교, 새 출력에만 있는 파일, 없는 비교 대상, JSON이 아닌 파일의 `#keys`가 통과하지 못함. JSON은 형까지 비교(테스트). Cycle 11 비교 29행을 새 도구로 다시 하면 I-89 전 그림인 v7 `pipeline.*` 3개만 다름. 다시 만든 그림 폴더 2개(52개)는 새 도구로도 모두 같고 새 출력에만 있는 파일 없음(출력 저장 안 함) |
| 끝난 GNN 작업의 job object 오류 (I-93) | 이미 끝난 실제 프로세스는 job object에 못 들어감(OSError, 테스트). 그런 작업은 종료 코드로 "GNN job failed"를 내고, 아직 도는 작업이면 오류가 그대로 나며 작업은 종료됨(테스트) |
| 다른 장치의 정확한 재현 (I-94) | 공개로 처리(`reproducibility.md`) |
| 문서 정정 (I-95) | 재현 비교 개수(29행, 출력 28개), `--require-unchanged` 실패 원인(5026680에서 다시 실행: v6·v7 모두 종료 코드 1, 바뀐 파일 v7 4개·v6 5개, 추가된 테스트 v7 12개·v6 13개, plan 해시는 각 plan commit과 일치. 출력 저장 안 함), Cycle 9 가드 변이 시간 |
| 회귀 테스트 | 작업 폴더 145/145 (v7 venv, 출력은 저장하지 않음). 커밋 5026680의 새 worktree + lock venv: 144 통과, 1 건너뜀(원본 데이터 상대경로 없음) (`audit/cycle12/postcommit_5026680_pytest.txt`). `SHA256SUMS.txt` v6 60/60, v7 28/28 (`postcommit_5026680_sha256.txt`). 주석만 고친 8feac9b도 144 통과, 1 건너뜀, 60/60, 28/28 (`postcommit_8feac9b_*.txt`) |
| 가드 변이 H1–H54 (H49 바깥 manifest를 먼저 씀, H50 빈 fold 목록 허용, H51 끝난 작업의 job object 오류를 그대로 냄, H52 0개 비교 통과, H53 JSON 형 무시, H54 새 출력에만 있는 파일 무시. H46은 바뀐 코드에 맞게 대상 문구 수정. 끝까지 돌리는 재현 테스트는 시간 때문에 제외) | 커밋 5026680 코드로 실행(시작 때 커밋 안 된 것은 문서 2개와 `audit/cycle12/`뿐이고 실행 중에 코드를 고치지 않음), 54/54 검출, 수정 안 한 사본 59 통과. 04:28:49–05:45:08 UTC (`audit/cycle12/guard_mutants.json`, `guard_mutants_cycle12_log.txt`) |
| 변이 테스트 17개 (§11), 커밋 2f1767c 그대로(사본의 `rolling_protocol.py`·`policy_evaluation.py`·`data_protocol.py`·`stats_tools.py`가 2f1767c와 같음, `audit/cycle12/mutation_scratch_vs_commit.json`. 사본은 시작 때 한 번 만들어져, 실행 중 작업 폴더의 Cycle 12 수정은 들어가지 않음) | 공식 `mutation_results.json`과 바이트 단위로 같음 (17/17, 03:47:29–04:36:35 UTC, `audit/cycle12/mutation_results_2f1767c.json`, `mutation_2f1767c_run.txt`) |

## 22. Cycle 13: Hybrid 끝 축 표시, 비교 도구의 키·빈 쌍, 정밀도 도구의 역할 확인, protocol 전환, 문서 정정 (I-96 – I-100)

Cycle 12 커밋 뒤 처음부터 다시 한 독립 감사(코드·문서)에서 찾은 항목이다. 새 P0·P1·P2는 없다 (코드 감사 P3 4건, 문서 감사 P3 5건). 코드 감사자는 커밋된 validation CSV만으로 v7 선택(백본, Hybrid 두 정책, MF 가중, GNN, ALS·BPR, BPR epoch)을 처음부터 다시 도출해 모두 같음을 확인했다(감사자의 임시 출력, 저장 안 함). 각 항목은 코드·커밋된 파일·실제 실행으로 다시 확인한 뒤 고쳤다. v6·v7의 선택과 시험 결과는 바뀌지 않는다. 보고 내용 중 바뀐 것은 Hybrid의 grid 끝 축 표시(§12.2 novel 행, `v7_feasibility.md` §4)뿐이다.

| 검사 | 결과 |
|---|---|
| Hybrid 끝 축 (I-97) | 고친 규칙(선택된 전략의 grid 기준, α 포함)으로 커밋된 8개 정책을 다시 계산: v7 F4 novel은 α .75(최대)와 balanced_union 후보 수 20,000(최대), v6 F2 novel과 F3 repeat는 α .75(최대)가 새로 끝으로 표시되고, 나머지 5개는 기존 표시와 같음. 선택값은 그대로 (`audit/cycle13/hybrid_boundary_recheck.json`, HEAD cb22505에서 만듦) |
| 비교 도구 (I-98) | 없는 `#keys`, JSON 객체가 아닌 파일의 `#keys`는 오류, 커밋 파일이 없는 폴더 쌍은 다른 쌍이 있어도 `all_equal: false`(테스트). 재현 11단계 비교를 Cycle 11 재현 출력에 다시 하면 Cycle 12와 같은 판정(I-89 전 v7 `pipeline.*` 3개만 다름, 빈 쌍 없음) (`audit/cycle13/compare_repro.json`) |
| 정밀도 도구 역할 (I-99) | 실제 v7 F4 `prepared`·`runs/seed_42`(역할 test)는 지표를 계산하기 전에 거부(그 전에 `require_protocol`이 manifest에 적힌 파일, 시험 정답 파일 포함의 해시만 확인함), `val_mf`로 다시 실행한 결과는 Cycle 4 커밋본과 같음 (`audit/cycle13/precision_role_check.txt`, `score_precision_F4_val_c13.json`) |
| protocol 전환 (I-100) | v6, v7, v7→v6, v6→v7→v6→v7 순서 모두 마지막 protocol의 plan 해시가 커밋된 `plan.json`과 같음 (`audit/cycle13/use_protocol_check.txt`) |
| 문서 정정 (I-96) | Cycle 12 가드 변이와 동시 실행한 작업, 두 번째 PC 범위(§9–§21), I-95의 정정 이유(POSIX에서 `launch_detached.py`는 실행을 새 세션으로 띄우고 GNN 작업은 그 그룹을 이어받으므로 그룹 전체 종료도 맞음), "약 10ms"의 근거, 저장 안 한 확인 표시 |
| 회귀 테스트 | 작업 폴더 149/149 (v7 venv, 출력 저장 안 함). 커밋 cb22505의 새 worktree + lock venv: 148 통과, 1 건너뜀(원본 데이터 상대경로 없음) (`audit/cycle13/postcommit_cb22505_pytest.txt`), `SHA256SUMS.txt` v6 60/60, v7 28/28 (`postcommit_cb22505_sha256.txt`). 변이 대상만 고친 db3653e도 148 통과, 1 건너뜀, 60/60, 28/28 (`postcommit_db3653e_*.txt`) |
| 가드 변이 H1–H60 (H55 `#keys`의 없는 키를 None으로 비교, H56 객체가 아닌 JSON의 `#keys`, H57 빈 커밋 폴더 쌍을 다른 쌍이 가림, H58 정밀도 도구가 시험 폴더를 읽음, H59 `use_protocol('v6')`가 v7 값을 남김, H60 α 없는 Hybrid 끝 표시. 끝까지 돌리는 재현 테스트는 시간 때문에 제외) | 커밋 db3653e 코드로 실행(시작 때 커밋 안 된 것은 문서 6개와 `audit/cycle13/`뿐이고 실행 중에 코드를 고치지 않음), 60/60 검출, 수정 안 한 사본 63 통과. 07:24:16–08:48:34 UTC (`audit/cycle13/guard_mutants.json`, `guard_mutants_cycle13_log.txt`). 첫 실행(06:06:59 UTC 시작(셸 기록의 시각, 출력 저장 안 함), cb22505 커밋 06:14:59 전의 작업 폴더. 코드는 cb22505와 같은 수정)은 H52 대상 문구가 없어 H52에서 멈춤(H1–H51 검출, 그 로그는 다시 실행할 때 덮어써 보존 안 됨) |
| 변이 테스트 17개 (§11), 커밋 1e2b35e 그대로(사본의 `.py` 37개가 모두 1e2b35e와 같음, `audit/cycle13/mutation_scratch_vs_commit.json`. 사본은 시작 때 한 번 만들어져 Cycle 13 수정은 들어가지 않음) | 공식 `mutation_results.json`과 바이트 단위로 같음 (17/17, 05:52:27–06:43:43 UTC, `audit/cycle13/mutation_results_1e2b35e.json`, `mutation_1e2b35e_run.txt`) |

## 23. Cycle 14: plan 해시와 Hybrid grid 값, 문서 정정 (I-101 – I-102)

Cycle 13 커밋 뒤 처음부터 다시 한 독립 감사(코드 HEAD cb22505, 문서 HEAD dcbfd0a)에서 찾은 항목이다. 새 P0·P1·P2는 없다 (코드 감사 P3 1건, 문서 감사 P3 4건). 각 항목은 코드·커밋된 파일·실제 실행으로 다시 확인한 뒤 고쳤다. v6·v7의 선택, 시험 결과, plan 해시는 바뀌지 않는다.

| 검사 | 결과 |
|---|---|
| plan의 Hybrid 축 (I-101) | 축을 `HYBRID_GRID`에서 뽑아 기록. 실제 커밋 plan: v6·v7 모두 plan 내용과 해시가 커밋된 `plan.json`과 같음 (`audit/cycle14/plan_hash_check.txt`, HEAD 41ba20d, 커밋 안 된 `.py` 없음). 크기가 같고 값이 다른 grid는 해시가 바뀜(테스트) |
| 문서 정정 (I-102) | I-99 거부 시점, Cycle 13 비교 재실행에 쓴 도구, 가드 변이·공식 변이 테스트의 동시 실행, 등록부 머리말, Cycle 13 가드 첫 실행의 시작 시점 |
| 회귀 테스트 | 작업 폴더 151/151 (v7 venv, 출력 저장 안 함). 커밋 41ba20d의 새 worktree + lock venv: 150 통과, 1 건너뜀 (`audit/cycle14/postcommit_41ba20d_pytest.txt`), `SHA256SUMS.txt` v6 60/60, v7 28/28 (`postcommit_41ba20d_sha256.txt`). Cycle 13 문서 커밋 dcbfd0a(새 테스트 추가 전): 148 통과, 1 건너뜀, 60/60, 28/28 (`postcommit_dcbfd0a_*.txt`) |
| 가드 변이 H1–H61 (H61 plan에 손으로 적은 Hybrid 축. 끝까지 돌리는 재현 테스트는 시간 때문에 제외) | 커밋 41ba20d 코드로 실행(시작 때 커밋 안 된 파일 없음), 61/61 검출, 수정 안 한 사본 65 통과. 09:03:32–10:29:23 UTC (`audit/cycle14/guard_mutants.json`, `guard_mutants_cycle14_log.txt`) |

## 24. Cycle 15: plan 해시와 Hybrid grid 전체, 문서 정정 (I-103 – I-104)

Cycle 14 커밋 뒤 처음부터 다시 한 독립 감사(코드·문서 HEAD 121b40a)에서 찾은 항목이다. 새 P0·P1·P2는 없다 (코드 감사 P3 1건: I-101 수정이 불완전, 문서 감사 P3 2건). 각 항목은 코드·커밋된 파일·실제 실행으로 다시 확인한 뒤 고쳤다. v6·v7의 선택, 시험 결과, plan 해시는 바뀌지 않는다.

| 검사 | 결과 |
|---|---|
| plan의 Hybrid grid (I-104) | 봉인된 grid와 다른 grid는 순서 있는 grid 전체의 sha256을 plan에 기록. 실제 커밋 plan: v6·v7 모두 grid가 봉인된 grid와 같고 plan 내용과 해시가 커밋된 `plan.json`과 같음 (`audit/cycle15/plan_hash_check.txt`, HEAD 4c1ddfb, 커밋 안 된 `.py` 없음). 후보 깊이 값, 전략과 후보 깊이의 짝, 순서만 바꾼 grid는 v6·v7 모두 해시가 바뀜(테스트) |
| 문서 정정 (I-103) | 출력이 남지 않은 Cycle 13 시각 표시, 공식 변이 테스트 재실행의 동시 실행 기록(Cycle 8–10·12·13) |
| 회귀 테스트 | 작업 폴더 157/157 (v7 venv, 출력 저장 안 함). 커밋 4c1ddfb의 새 worktree + lock venv: 156 통과, 1 건너뜀 (`audit/cycle15/postcommit_4c1ddfb_pytest.txt`), `SHA256SUMS.txt` v6 60/60, v7 28/28 (`postcommit_4c1ddfb_sha256.txt`). Cycle 14 문서 커밋 121b40a: 150 통과, 1 건너뜀, 60/60, 28/28 (`postcommit_121b40a_*.txt`) |
| 가드 변이 H1–H62 (H62 plan에 grid sha256 없음. H61은 새 sha256만으로도 해시가 바뀌어 살아남으므로, Cycle 14 테스트에 기록된 축 확인을 더함. 끝까지 돌리는 재현 테스트는 시간 때문에 제외) | 커밋 4c1ddfb 코드로 실행(시작 때 커밋 안 된 것은 문서 3개뿐이고 실행 중에 코드를 고치지 않음), 62/62 검출, 수정 안 한 사본 71 통과. 10:50:02–12:18:48 UTC (`audit/cycle15/guard_mutants.json`, `guard_mutants_cycle15_log.txt`) |

## 25. Cycle 16: v7 grid와 F2 구간 서술, dirty 검사 (I-105 – I-108)

Cycle 15 커밋 뒤 처음부터 다시 한 독립 감사(코드·문서 HEAD c0b52c0)에서 찾은 항목이다. 새 P0·P1은 없다 (코드 감사 P2 1건·P3 1건, 문서 감사 P2 1건·P3 3건). 각 항목은 코드·커밋된 파일·실제 실행으로 다시 확인한 뒤 고쳤다. v6·v7의 선택, 시험 결과, plan 해시는 바뀌지 않는다. 바뀐 것은 해석의 한정(§8 항목 2, §12.5 남은 한계)이다.

| 검사 | 결과 |
|---|---|
| v7 grid 서술 (I-105) | v7은 ALS·BPR·GNN grid만 넓혔고 Hybrid grid(2,808개)와 ALS+BPR α(11개)는 v6와 같음: 커밋된 두 plan의 `hybrid_grid_size`·`mf_alphas`가 같고, 하나의 봉인 grid sha256이 두 plan에 맞음 (`audit/cycle15/plan_hash_check.txt`). "모든 모델군 grid 확장" 6곳을 고치고, §12.5와 I-23 남은 한계에 v7 Hybrid의 끝 축(§12.2)을 더함 |
| F2 시험 구간 (I-106) | F2 시험 구간의 이벤트 80%가 v1–v5 내부 validation 구간 (`v7_feasibility.md` §1 표, `rolling_protocol.py` F2 설명). "F2/F3가 독립적인 확인" 서술을 고침 (§3 표, §8 항목 2, I-01 수정 설명, `manuscript_alignment.md`) |
| 문서 정정 (I-107) | 재현 5단계의 49분 38초는 Cycle 4 재실행 시간(`audit/cycle4/mutation_cycle4_start.txt`), Cycle 3 기준 실행은 시각 출력 없음. Cycle 11 재실행과 재현 실행(`audit/cycle11/repro_run.txt`)의 겹침 추가. Cycle 13 가드 첫 실행 로그가 보존 안 됐음을 `cycle_log.md`에도 표시 |
| dirty 검사 (I-108) | 사전등록 실행 전 dirty 검사가 `audit_runs`로 시작하는 모든 경로(예: `audit_runs_patch.py`)를 빼던 것을 `audit_runs/` 폴더만 빼도록 고침. 실제 저장소에서 `git_state()`는 커밋 안 된 코드·문서를 모두 나열하고 gitignore 대상 `audit_runs/`는 나오지 않음, v6·v7 plan 내용·해시 그대로 (`audit/cycle16/git_state_and_plan_check.txt`) |
| 회귀 테스트 | 커밋 전 작업 폴더 158/158 (v7 venv, 12:34:23–12:40:13 UTC, `audit/cycle16/precommit_pytest.txt`). 커밋 6ce754d의 새 worktree + lock venv: 157 통과, 1 건너뜀 (`audit/cycle16/postcommit_6ce754d_pytest.txt`), `SHA256SUMS.txt` v6 60/60, v7 28/28 (`postcommit_6ce754d_sha256.txt`). Cycle 15 문서 커밋 c0b52c0: 156 통과, 1 건너뜀, 60/60, 28/28 (`postcommit_c0b52c0_*.txt`) |
| 가드 변이 H1–H63 (H63 dirty 검사의 접두사 면제. 끝까지 돌리는 재현 테스트는 시간 때문에 제외) | 커밋 6ce754d 코드로 실행(시작 때 커밋 안 된 것은 문서 7개와 `audit/cycle16/`뿐이고 실행 중에 코드를 고치지 않음), 63/63 검출, 수정 안 한 사본 72 통과. 12:40:29–14:12:07 UTC (`audit/cycle16/guard_mutants.json`, `guard_mutants_cycle16_log.txt`) |

## 26. Cycle 17: dirty 검사를 git pathspec으로, GNN spec 절대 경로, 문서 정정 (I-109 – I-111)

Cycle 16 커밋 뒤 처음부터 다시 한 독립 감사(코드·문서 HEAD 0a0ca40)에서 찾은 항목이다. 새 P0·P1·P2는 없다 (코드 감사 P3 2건, 문서 감사 P3 6건). 각 항목은 코드·커밋된 파일·실제 실행으로 다시 확인한 뒤 고쳤다. v6·v7의 선택, 시험 결과, plan 해시는 바뀌지 않는다.

| 검사 | 결과 |
|---|---|
| dirty 검사 (I-110, I-108 수정이 불완전) | 출력 문자열 대신 `git status --porcelain --untracked-files=normal -- . ':(exclude)audit_runs'`로 git이 `audit_runs`를 뺌. 임시 git 저장소 테스트: `status.showUntrackedFiles=no`에서도 새 코드가 나오고, `audit_runs/`에서 코드 경로로의 이름 변경은 새 경로가 나오며, 따옴표로 싼 `audit_runs/` 경로는 빠짐. 실제 저장소에서 `git_state()` 목록이 일반 `git status`와 같고 `audit_runs/`는 없음, v6·v7 plan 내용·해시 그대로, 커밋된 두 plan의 `dirty_files`는 비어 있음 (`audit/cycle17/git_state_and_plan_check.txt`, cf1a3fe 커밋 전 작업 폴더, 코드는 cf1a3fe와 같은 수정) |
| GNN job spec 경로 (I-111) | spec 폴더를 절대 경로로 넘김. 다른 폴더에서 상대 root로 실행하는 테스트에서 넘긴 경로가 절대 경로이고 파일이 있음 |
| 문서 정정 (I-109) | F3 시험 구간의 80%가 v6 F2의 validation이었음(§8 항목 2, I-106), 등록부 머리말, 가드 변이 docstring, 교수 자료 md 두 개의 감사 후 안내(PDF는 수정 안 함, `professor_materials/README_KO.md`), §1 Cycle 16 행, `v7_feasibility.md` §4의 Hybrid backbone lr 끝 |
| 회귀 테스트 | 커밋 전 작업 폴더 첫 실행은 159 통과 1 실패(Cycle 5 상태 열 테스트가 옛 출력 형식의 가짜 git 출력을 씀, `audit/cycle17/precommit_pytest_first_run_1_failed.txt`). 그 테스트를 git이 지금 돌려주는 출력과 제외 pathspec 요구로 바꾼 뒤 160/160 (`precommit_pytest.txt`). 커밋 cf1a3fe의 새 worktree + lock venv: 159 통과, 1 건너뜀 (`postcommit_cf1a3fe_pytest.txt`), `SHA256SUMS.txt` v6 60/60, v7 28/28. Cycle 16 문서 커밋 0a0ca40: 157 통과, 1 건너뜀, 60/60, 28/28 (`postcommit_0a0ca40_*.txt`) |
| 가드 변이 H1–H65 (H63을 새 방식의 제외 pathspec 삭제로 바꿈, H64 `--untracked-files` 없음, H65 상대 spec 경로. 끝까지 돌리는 재현 테스트는 시간 때문에, git 이력을 읽는 `test_cycle4_guards.py`의 `verify_plan_code` 테스트는 사본이 git 저장소 밖이라 제외. 앞의 것은 그 테스트가 생긴 Cycle 10(4e166c5)부터, 뒤의 것은 Cycle 5(fd9bcca)부터 늘 제외됨) | 커밋 cf1a3fe 코드로 실행(시작 때 커밋 안 된 것은 문서 6개와 `audit/cycle17/`뿐이고 실행 중에 코드를 고치지 않음), 65/65 검출(바꾼 Cycle 5 테스트로 H14도 검출), 수정 안 한 사본 74 통과. 14:40:12–16:12:56 UTC (`audit/cycle17/guard_mutants.json`, `guard_mutants_cycle17_log.txt`) |
