# v10 계획: RetailRocket 메인 실험 (v8 변종 + v9 사다리, repeat 정책)

작성: 2026-11-02 20:40 KST. v10 코드를 쓰기 전, RetailRocket에 v10으로 어떤 시험도 열기 전. 이 문서의 결정은 결과를 보고 바꾸지 않는다. 바꿔야 하면 바꾼 이유와 시점을 `cycle_log.md`와 이 문서 끝에 남기고, 그 변경 뒤에 열린 시험만 v10 결과로 쓴다.

## 1. 배경과 결정 (2026-11-02 회의, 사용자 결정)

| 항목 | 결정 |
|---|---|
| 논문의 메인 데이터 | RetailRocket. Taobao(v8 1% 봉인 시험, v9 2% 봉인 확인 시험)는 다른 데이터셋에서의 확인 결과로 씀 |
| 정책 | repeat만 계산하고 보고. 이유: 쇼핑몰에는 샀던 상품을 다시 사는 수요가 있음(회의 결정). novel은 계산하지 않음(protocol 설정 `POLICIES=('repeat',)`) |
| 모델 | v8 변종 전체(보고 표의 17개 모델)와 v9 사다리(설계 a02) |
| fold | v6·v7과 같은 네 구간 F1–F4 |
| 재고 신호 | v8 설계 그대로 씀(점수 결합 Hybrid의 γ, 재정렬 특징 `i_available`). 값은 cutoff 이전 마지막 값 |
| 통계 | fold별 검정, 주 가설은 네 fold를 합친 하나의 Holm 묶음(§5) |
| 주 가설 | v8 재정렬과 v9 사다리 둘 다, 한 Holm 묶음 |
| 유의하지 않을 때 | "유의한 차이 없음"으로 쓰고 평균과 신뢰구간을 그대로 보고 |
| 실행 시간 | grid를 줄이지 않음 |

정책 결정의 시점: v9 Taobao 확인 시험에서 repeat는 사다리 10% 기준에 미달했고(×1.025, ×1.071, ×1.027, 단계 차이는 모두 Holm 유의), novel은 통과했다. repeat 결정은 그 결과를 본 뒤에 내려졌으며, 원고에 이 사실을 쓴다. Taobao novel 결과는 기록(`audit/results_v8`, `audit/results_v9`)에 그대로 남긴다.

## 2. 데이터와 fold

- 원본: `../../data`(RetailRocket `events.csv` SHA256 `3745aa83…`, 상품 속성 part1·part2, `category_tree.csv`). v6·v7과 같은 파일.
- 전처리: v6–v9와 같음(`use_if=False`, 이벤트 3회 이상 사용자·상품).

| fold | 끝 비율 | 시험 크기 | 시험 구간(전체 이벤트 비율) | 이전에 연 기록 |
|---|---|---|---|---|
| F1_hist | 1.0 | 0.2 | 0.80–1.00 | v1–v5 여러 번, v6(plan `c56ada95…`) |
| F2_mid | 0.8 | 0.25 | 0.60–0.80 | v6. v1–v5에서는 validation |
| F3_early | 0.6 | 0.25 | 0.45–0.60 | v6 |
| F4_unseen | 0.36 | 0.25 | 0.27–0.36 | v7(plan `784d0fa9…`) |

## 3. 시험 구간 재사용과 보호 장치

- 네 구간 모두 이전 버전이 이미 열었다. 이 구간들을 연 모델은 Popularity, Recent, RecentPop, ALS, BPR, ALS+BPR, GNN, 점수 결합 Hybrid(v1)이다.
- v8 재정렬(`Hybrid_LTR`, `Hybrid_LTR_W`)은 RetailRocket 0–27%(F4의 내부 validation, `audit/v8dev`)에서 개발했다. F1–F4의 시험 구간은 설계에 쓰지 않았다.
- v9 사다리는 Taobao에서만 개발했다(`audit/v9_dev_plan.md`). RetailRocket을 쓴 적이 없다.
- 기존 보호 장치 `committed_test_conflicts`는 다른 plan이 이미 연 구간을 다시 여는 실행을 거부한다. v10은 v6·v7의 두 기록(plan `c56ada95…`, `784d0fa9…`)만 plan에 명시해 예외로 둔다(`ACKNOWLEDGED_TEST_RECORDS`). 다른 모든 기록, 특히 v10 이후에 다른 plan으로 같은 구간을 여는 실행은 계속 거부된다. v10 plan 자체가 이 목록을 담으므로 목록을 바꾸면 plan 해시가 바뀐다.
- 원고에는 "RetailRocket의 시험 구간은 이전 버전의 모델 평가에 쓰였고, v8 재정렬과 v9 사다리의 설계에는 쓰이지 않았다"고 쓴다. "봉인 시험"이라고 쓰지 않는다.

## 4. 모델과 선택

- 학습 설정 grid: v8과 같음(ALS 192, BPR 24 × epoch 7, GNN 36, ALS+BPR α 11, seed 42/43/44, 선택 seed 42, 선택 점수 mean(NDCG@50, Recall@50)).
- 점수 결합 Hybrid(v1)와 `Hybrid_ALS_only`: v8 grid(2,808개 × backbone 3개) 그대로.
- v8 재정렬: `Hybrid_LTR`(policy_truth), `Hybrid_LTR_W`(weighted)와 특징 묶음 ablation 6개. 설계·설정 grid·후보·특징은 v8 protocol 그대로. Hybrid backbone GNN 위에서 학습.
- v9 사다리: `hybrid_v9.DESIGNS['a02']` 그대로. 이름은 v8의 점수 결합 Hybrid와 겹치지 않게 바꾼다: `Hybrid_v9`(L3), `Hybrid_v9_MF_Graph`(L2), `Hybrid_v9_MF`(`LTR_MF`), `Hybrid_v9_no_graph`, `Hybrid_v9_no_mf`. 그래프 모듈의 GNN은 단독 GNN baseline으로 선택된 모델(v9 규칙). a02는 Taobao 개발에서 novel 최소 단계로 골랐지만, Taobao 확인 시험과 같은 모델로 유지한다(repeat로 다시 고르지 않음).
- baseline 8개: Popularity, Recent, RecentPop, ALS, BPR, ALS+BPR, GNN, CoVisit.
- 모든 재정렬·사다리 모델은 내부 validation 구간에서만 학습하고, `selection.json`에 SHA256으로 봉인한 뒤 fold의 시험을 한 번 연다.

## 5. 통계와 판정 (repeat, K = 50)

- fold마다 protocol이 계산(사용자 단위 쌍대 bootstrap 95% CI 10,000회, 부호 뒤집기 검정 10,000회, seed 평균 사용자 지표):
  - confirmatory 묶음: Hybrid, Hybrid_LTR, Hybrid_LTR_W, Hybrid_v9 각각 − baseline 8개, NDCG@50·Recall@50.
  - ladder 묶음: Hybrid_v9 − Hybrid_v9_MF_Graph, Hybrid_v9_MF_Graph − ALS+BPR, ALS+BPR − ALS/BPR/GNN.
  - ablation 묶음: Hybrid_v9 − Hybrid_v9_MF/no_graph/no_mf.
  - secondary 묶음: Hybrid_LTR, Hybrid_LTR_W − Hybrid.
- **주 가설 묶음 (네 fold를 합쳐 Holm 한 번, NDCG@50만)**: fold마다 Hybrid_LTR·Hybrid_LTR_W·Hybrid_v9 각각 − baseline 8개(24개)와 사다리 단계 5개, 합계 29개 × fold 4개 = 116개 비교. 각 비교의 보정 전 p는 fold의 `paired_tests.csv` 값을 그대로 쓴다. 계산 도구(`audit/v10protocol/primary_family.py`)는 시험을 열기 전에 합성 데이터로 확인하고 커밋한다.
- 점수 결합 Hybrid(v1)는 RetailRocket 시험 결과를 보며 만든 모델이라 주 가설에 넣지 않는다(fold별 confirmatory 묶음에서 서술용으로 보고).
- 사다리 판정(fold마다): L1 > max(L0), L2 ≥ 1.10 L1, L3 ≥ 1.10 L2, L3 > baseline 8개(seed 평균 NDCG@50). "유의미한 성과"는 비율을 모두 넘고 L3 − L2와 L2 − L1이 주 가설 묶음에서 유의할 때만 쓴다. 비율만 넘으면 "비율은 넘었으나 유의하지 않음".
- v8 재정렬 판정: fold와 baseline마다, 주 가설 묶음에서 유의하게 높을 때만 "높음". 아니면 "유의한 차이 없음".
- fold를 합친 평균이나 메타 분석은 하지 않는다. fold마다 보고하고, 판정을 넘은 fold 수를 함께 쓴다.
- 검정력: v7 F4 repeat 평가 사용자 413명에서 NDCG 최소 검출 차이 약 0.012(`audit/v8dev/power_check.csv`). 116개 묶음에서는 더 크다. Taobao v8 재정렬 차이(+0.0049~0.0060) 크기라면 "유의한 차이 없음"이 나올 가능성이 높다. 결과를 보기 전에 기록한다.

## 6. 누수 방지와 검증 (시험을 열기 전에 모두 통과)

- 특징·후보·그래프는 학습 구간 이벤트만(`check_training_only`), 상품 속성(카테고리·재고)은 cutoff 이전 마지막 값(`data_protocol.py`, `test_available_flag_is_last_value_before_cutoff`).
- 테스트(`test_v10_protocol.py`): v6–v9 plan 해시 유지, v10 plan 기록, 합성 데이터 끝-끝 실행(repeat만, 재정렬·사다리가 봉인 전 학습, 이름 변경, 묶음별 Holm, 재개 때 재학습·재열람 없음), 시험 구간을 섞어도 선택·모델 파일이 같음, 보호 장치 예외는 명시된 plan에만 적용.
- 실행 뒤: 원본 `events.csv`에서 시험 지표 독립 재계산, 봉인 모델 해시 확인, 단계 순서 확인, Holm 재계산, 주 가설 묶음 계산.

## 7. 금지

- 결과를 보고 설계·grid·후보·특징·정책·fold·주 가설을 바꾸는 것, fold를 빼는 것, 시험을 다시 여는 것.
- Hybrid 변종 중 잘 나온 것만 본문에 쓰는 것. 주 가설 세 모델(Hybrid_LTR, Hybrid_LTR_W, Hybrid_v9)과 사다리는 결과와 관계없이 모두 보고한다.
- 유의하지 않은 차이를 "우수"로 쓰는 것.
