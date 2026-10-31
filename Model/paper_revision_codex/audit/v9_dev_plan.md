# v9 개발 계획: Hybrid 사다리 (개발 루프 + 봉인 확인 시험)

작성: 2026-10-31 KST, 개발 데이터를 보기 전. 이 문서의 기준은 개발 중에 바꾸지 않는다. 바꿔야 하면 바꾼 이유와 시점을 `cycle_log.md`에 남기고, 확인 시험 전에 이 문서에 기록한다.

## 1. 성공 기준 (사용자 요청)

점수는 NDCG@50, 주 판단은 repeat 정책(모든 버전에서 학습 설정을 고르는 정책). novel은 같은 표로 함께 보고한다.

| 단계 | 모델 | 내용 |
|---|---|---|
| L0 | ALS, BPR, GNN | 단일 모델 (각자 grid로 튜닝) |
| L1 | MF 모듈 = ALS+BPR | α를 validation으로 고름 |
| L2 | MF + 그래프 Hybrid | 최종 Hybrid 구조에서 MF·그래프 모듈의 신호만 씀 |
| L3 | 최종 Hybrid | 모든 모듈 |

- 기준: L1 > max(L0), L2 ≥ 1.10 × L1, L3 ≥ 1.10 × L2, L3가 모든 baseline(Popularity, Recent, RecentPop, ALS, BPR, ALS+BPR, GNN)보다 높음.
- 우선순위: 두 단계 비율 중 작은 값 min(L2/L1, L3/L2)을 최대화.
- 확인 시험에서는 비율과 함께 사용자 단위 쌍대 검정(Holm)을 보고한다. 비율이 10%를 넘어도 유의하지 않으면 "유의하지 않음"으로 쓴다.

## 2. 데이터 분리

Taobao 사용자는 `dataset_adapters._keep_user`의 seed 42 해시(0–9,999)로 나눈다. 구간이 겹치지 않으므로 사용자가 겹치지 않는다.

| 해시 구간 | 용도 | 상태 |
|---|---|---|
| [0, 100) (1%) | v8 봉인 시험 | 시험 열람 끝. 다시 쓰지 않음 |
| [100, 300) (2%) | v9 개발 | 마음껏 반복 |
| [300, 500) (2%) | v9 확인 시험 (§5에서 [300, 400)을 바꿈) | 개발이 끝날 때까지 변환도 하지 않음 |

RetailRocket은 열지 않은 시험 구간이 없으므로(`v7_feasibility.md` §1, v7 F4가 마지막) 개발 참고용으로만 쓴다.

개발 표본 안의 구간 (v8 fold와 같은 비율: 시험 0.25, validation 0.2):
- D1: 재정렬 학습 구간 (내부 validation의 내부 분할).
- D2: 개발 평가 구간 (내부 validation). 반복해서 본다.
- H: 개발 holdout (개발 표본의 바깥 시험 구간). 개발을 끝낼 후보가 생겼을 때만 보고, 본 횟수를 모두 기록한다.

## 3. 공정성과 누수 방지

- 하위 모듈(ALS, BPR, GNN, ALS+BPR α)은 개발 중 v8 grid로 고른 설정을 그대로 쓰고, 개발 결과를 보고 바꾸지 않는다. 단계 차이는 상위 모듈을 바꿔서만 키운다. 하위 모듈을 약하게 만드는 변경(설정 축소, 학습 데이터 축소, 불리한 후보 처리)은 하지 않는다.
- 확인 시험은 v8 protocol처럼 모든 모델을 같은 grid로 다시 튜닝한다.
- 새 특징과 후보는 학습 구간 이벤트만 읽는다(`check_training_only`). 새 신호마다 누수 테스트와, 라벨 구간을 섞어도 특징이 같은지 확인하는 테스트를 둔다.
- GNN이나 그래프 모듈을 바꾸면 단독 GNN baseline도 같은 모듈을 쓴다(같은 이름의 모델이 사다리와 baseline에서 다르지 않게).

## 4. 종료 조건

1. D2에서 기준을 넘는 설계가 나오면 H에서 한 번 확인한다. H에서도 기준을 넘으면 그 설계와 확인 시험 plan을 고정해 커밋한 뒤 확인 시험을 한 번 실행한다.
2. H에서 무너지면 D2 과적합으로 기록하고 개발을 계속한다.
3. 확인 시험 결과는 기준을 넘든 못 넘든 그대로 보고하고 다시 열지 않는다.
4. 개발 시도는 모두 `audit/v9dev/attempts.csv`에 남긴다(시도 수 공개).

## 5. 변경 기록 (2026-11-01 02:30 KST, D2 시도 a01–a03을 본 뒤, H 결과를 보기 전, 확인 표본 변환 전)

1. 주 판단 정책: repeat → novel. 이유: D2 진단에서 repeat는 기준에 닿을 수 없다고 판단했다. D2 라벨로 D2 안에서 CV 학습한 순위 모델(같은 특징, 같은 후보)도 0.0724(`audit/v9dev/in_window_cv_a02_repeat.json`)로, 기준이 요구하는 1.1 × 1.1 × L1(0.0659) = 0.0797보다 낮다. 특징 하나(`h_weight`)로 0.0658 = ALS+BPR이므로 repeat에서는 이력 신호가 이미 거의 전부를 설명한다. 이 변경은 개발 결과를 본 뒤에 한 것이므로, 확인 시험 보고에 변경 사실과 시점을 함께 쓴다. repeat 결과도 같은 표에 그대로 보고하고 빼지 않는다. repeat가 기준을 넘지 못하면 "repeat에서는 기준 미달"로 쓴다.
2. 확인 시험 판정: 정책마다 기준 비율(L1 > max(L0), L2 ≥ 1.10 L1, L3 ≥ 1.10 L2, L3 > 모든 baseline)과 사다리 단계 쌍대 검정(Hybrid − Hybrid_MF_Graph, Hybrid_MF_Graph − ALS+BPR, ALS+BPR − ALS/BPR/GNN; NDCG@50과 Recall@50; 한 Holm family)을 함께 보고한다. "유의미한 성과"라고 쓰려면 그 정책에서 기준 비율을 모두 넘고, L3 − L2와 L2 − L1의 NDCG@50 차이가 Holm 보정 후 유의해야 한다.
3. baseline 추가: CoVisit(그래프 모듈의 함께 본 상품 그래프 단독)을 baseline에 넣는다. Hybrid가 넘어야 할 모델이 늘어나므로 기준이 더 엄격해진다.
4. 확인 표본: [300, 400) (1%) → [300, 500) (2%). 이유: 검정력. D2 novel에서 L3 − L2 차이의 표준오차가 약 0.00042(평가 사용자 6,246명)였고, 1% 표본은 평가 사용자가 약 절반이라 표준오차가 약 1.4배가 된다. [400, 500)은 지금까지 어떤 용도로도 변환하거나 열지 않았다.
5. 확인 시험 설계: H에서 기준(위 2의 비율)을 넘으면 a02(LambdaRank, weighted 학습 구간, 음성 100개, `hybrid_v9.DESIGNS['a02']`)로 고정한다. 확인 시험은 `rolling_protocol.py --protocol v9`(fold `TB2_full`)로 ALS·BPR·ALS+BPR α·GNN을 v8 grid로 다시 고르고, 그래프 모듈의 GNN은 단독 GNN baseline으로 고른 것과 같은 모델을 쓴다. 순위 모델은 내부 validation 구간에서만 학습하고, 봉인(sha256) 뒤 시험을 한 번 연다. H에서 무너지면 확인 시험을 하지 않고 개발을 계속한다.

## 6. 확인 시험 전 추가 (2026-11-01 07:00 KST, 확인 시험 실행 중, 시험 열람 전)

1. 민감도 분석 "novel 튜닝 baseline"을 미리 정한다. 이유: protocol은 ALS·BPR·GNN의 학습 설정을 repeat validation으로 고르고(v6부터의 규칙), 사다리 순위 모델은 정책마다 그 정책의 라벨로 학습한다. §5에서 novel을 주 판단 정책으로 바꿨으므로 이 비대칭은 novel에서 baseline에 불리하다.
2. 방법(`audit/v9protocol/novel_tuned_baselines.py`): protocol의 validation 기록만 다시 읽어 novel validation 점수로 ALS, BPR(설정과 epoch), GNN을 고르고(새 grid 없음, 같은 선택 점수와 동점 규칙), 그 fit으로 novel validation에서 ALS+BPR α를 고른다. 전체 학습 구간에서 protocol seed 3개로 다시 학습하고, 봉인된 사다리 순위 모델의 시험 목록(바꾸지 않음)과 비교한다. 시험 열람은 같은 plan·selection으로 접근 로그에 먼저 기록한다. 비교는 한 Holm family: Hybrid − ALS_n·BPR_n·ALS+BPR_n·GNN_n, Hybrid_MF_Graph − ALS+BPR_n, ALS+BPR_n − ALS_n·BPR_n·GNN_n (NDCG@50, Recall@50).
3. 판정: novel에서 "유의미한 성과"라고 쓰려면 §5-2(protocol baseline)와, L0·L1을 novel 튜닝 baseline으로 바꾼 같은 기준(L1_n > max(L0_n), L2 ≥ 1.10 L1_n, L3 ≥ 1.10 L2, L3가 두 튜닝의 모든 baseline보다 높음)을 모두 넘고, Hybrid_MF_Graph − ALS+BPR_n의 NDCG@50 차이가 Holm 보정 후 유의해야 한다. 하나라도 못 넘으면 "튜닝 비대칭을 보정하면 기준 미달"로 쓴다.
4. 한계: GNN validation은 epoch 상한(400)에 닿은 설정 중 repeat 1위만 연장했으므로, novel 1위 GNN이 상한에 닿아 있으면 연장 없이 쓴다(결과에 표시).
5. 검증: 작은 합성 실행에서 repeat 점수로 고르면 protocol의 novel baseline(ALS, BPR, ALS+BPR, GNN; seed 3개; K 5개)을 차이 0으로 재현하고, 시험 전에는 거부하며, 다른 GNN을 고르면 고정 epoch 재학습을 함(`audit/v9protocol/check_novel_tuned_tiny.txt`).
