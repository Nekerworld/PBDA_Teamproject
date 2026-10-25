# Manuscript alignment (원고–코드–결과 대조)

원고 파일(투고본 PDF/LaTeX)은 이 저장소에 없다. 따라서 원고를 직접 고치지 않았다.

이 문서는 리뷰어 지적, 원고와 교수 보고 자료(`professor_materials/professor_briefing_ko.md`)의 주장, 검증된 코드와 결과를 대조한다.
- 원고를 고칠 때는 오른쪽 "근거" 열의 파일에 있는 수치만 쓴다.
- 실행하지 않은 실험의 수치는 넣지 않는다.

## 1. 리뷰어 지적별 상태

| 지적 | 코드 상태 | 원고에서 할 일 | 근거 |
|---|---|---|---|
| R2-1 분할 통일 (ALS 90:10 vs LightGCN 80:20 LOO) | 해결. 모든 모델이 같은 시간순 분할, 같은 평가 사용자, 같은 정답을 씀 | 분할을 fold별 시간순 학습:시험(F1 80:20, F2·F3·F4 75:25) + 내부 시간순 validation으로 서술. rolling-origin 3 fold 추가 | `data_protocol.py`, `audit/results_v6/F*/leakage_check.json` |
| R2-2 transductive LightGCN | 해결. 학습 상품만 점수를 받고, 학습 벡터 없는 사용자는 0점으로 분모에 포함 | 평가 모집단과 cold-start 미지원을 명시 | `validation_report.md` §3 |
| R2-3 그림 오류·누락 | 해결. 커밋된 요약 CSV만 읽는 생성기가 모델 비교, Top-K, 쌍대 차이 CI, γ 민감도, 후보 수, 파이프라인 그림과 표를 만듦 | 원고 그림을 `audit/figures_v6`(주 결과), `audit/figures_v7`(확장 grid 확인)로 교체하고 번호를 맞춤 | `report_tools/plot_rolling_results.py`, 각 폴더의 `figure_manifest.json` |
| R2-4 필터 후 통계 | 해결 | 최종 학습 통계 표(사용자 141,660, 상품 58,556, 이벤트 974,155, 쌍 562,419) | `audit/results_v6/F1_hist/dataset_statistics.csv` (`train_final` 행) |
| R2-5 식 9의 gamma | 해결. γ는 정책·fold마다 validation에서 선택됨 (repeat: F1 1.25, F2/F3 1.1. novel: F1 1.5(grid 최대), F2/F3 1.25) | γ의 정의, 탐색 범위 {1, 1.1, 1.25, 1.5}, 선택 방법, 민감도 표 | `validation_report.md` §6.2 |
| R2-6 IF 특징과 제거율 | 해결 | 9개 특징, 제거 62.05%, 구매 100% 제거, IF를 최종 모델에서 뺀 이유와 ablation | `validation_report.md` §6.3 |
| R1 그림 품질 | 해결. PNG(300 dpi), PDF, SVG를 함께 만들고, 다시 실행해도 바이트 단위로 같음 | `professor_materials/figures`(v5 단일 분할 기준)는 쓰지 않음 | `audit/figures_v6`, `audit/figures_v7` |
| R1 데이터 맥락과 한계 | - | 단일 데이터셋, 2015년, 반복 추천 비중, 평가 사용자 수 | `validation_report.md` §3, §4.2, §8 |
| R1 봇 검증 | IF는 검증된 봇 검출기가 아니며 최종 모델에서 뺌 | "bot filtering" 주장을 삭제하거나 한계로 서술 | `if_summary.json` |
| (추가) validation → training 누수 | 해결. 내부 학습 구간만으로 그래프를 만들고, 선택 후 고정 epoch로 재학습 | 수정 사항으로 서술 | `leakage_check.json` (3 fold × 28 통과) |

## 2. 고쳐야 할 주장 (v5 보고 → v6 검증 결과)

| 기존 서술 (v5 / briefing) | 문제 | 검증된 대체 서술 |
|---|---|---|
| "Hybrid가 두 지표의 세 seed 평균 모두 가장 높았다" (F1 = v5 시험 구간) | v6 F1에서 Recall은 GNN이 더 높음 (0.1681 vs 0.1668). F3 NDCG는 ALS+BPR이 더 높음 | "baseline 7종과 비교하면 Hybrid는 12개 칸(3 fold × 2 정책 × 2 지표) 중 9개에서 평균 1위였고, 가장 강한 baseline(GNN, ALS+BPR)보다 세 fold 모두에서 유의하게 높은 지표는 없었다". 같은 후보에서 결합 점수의 MF 부분만 ALS로 바꾼 ablation(`Hybrid_ALS_only`, 후보 생성에는 BPR이 남음, 검정 안 함)은 F3 novel(두 지표)과 F1 repeat Recall에서 Hybrid보다 평균이 높았다 |
| "가장 강한 비교 모델 대비 NDCG +10.83%" | v5는 시험을 본 뒤 설계를 바꾼 결과임. 상대 개선율은 비유의한 차이를 과장함 | GNN·ALS+BPR 대비 차이, 95% CI(보정 안 함), Holm 보정 p값을 그대로 보고 (차이·p_holm은 `validation_report.md` §4.3, CI는 `audit/results_v6/report_paired_tests_all_folds.csv`. v7은 차이·p_holm이 §12.4, CI는 `audit/results_v7/report_paired_tests_all_folds.csv`). 유의성은 p_holm으로만 판정함 (CI가 0을 포함하지 않아도 Holm 기준 비유의한 칸이 있음). 세 fold 모두에서 유의한 우위는 없음 |
| Recall 우위 | v5에서도 CI가 0을 포함함 | Recall 우위를 주장하지 않음 |
| 계산 효율 (retrieve-then-rank) | 후보 수는 fold마다 500–전체로 바뀌고, Hybrid 경로는 GNN 단독 전체 카탈로그 점수 계산보다 10배 이상 느림 | 효율성 주장을 삭제하거나 "후보 500개에서도 정확도 유지"로 한정하되, 근거가 F1·seed 42 한 번의 사후 분석(시험 구간, 선택에 쓰지 않음, `validation_report.md` §6.1)임을 함께 적음 |
| 재고 인식 추천 | 정확도 효과는 γ 1 대비 NDCG +0.0024, Recall +0.0065 수준이지만, 재고 신호는 Hybrid에만 들어가 Hybrid − baseline 차이의 상당 부분을 차지할 수 있음 (공개 사항 9) | "선택된 γ에서는 노출 구성을 재고 있는 상품 쪽으로 바꾸고 정확도 손실은 없었다" 정도로 한정하고, F1·seed 42 한 번의 사후 분석(§6.2)임을 적음. γ=2.0에서는 NDCG가 중립값보다 낮음(0.1140 vs 0.1151) |
| 이종 그래프(속성·카테고리 노드), BPR 손실, IF 기반 정제 | 최종 구현은 이분 그래프, InfoNCE(음성 32개, 온도 0.2), IF 없음 | 식 6–9, 구조도, 손실 함수를 구현에 맞춰 다시 씀 (`model.py`, `recommendation_scoring.py`) |
| 최종 점수 = ALS + GNN | 구현은 ALS+BPR min-max 결합(α) 후 후보 안에서 GNN과 결합(ω, min-max 또는 RRF) × γ. 후보 생성 방식도 fold·정책마다 validation으로 고름: `blend`(α 결합 점수 상위 n개) 또는 `balanced_union`(ALS 상위 ⌈n/2⌉ ∪ BPR 상위 ⌊n/2⌋, 모자라면 α 결합 점수로 채우고 α 결합 점수 순으로 정렬, `policy_evaluation.candidate_indices`). 봉인된 설정 8개(v6 3 fold + v7 F4, 정책 2개) 중 5개가 `balanced_union`, 2개가 RRF | 식 8과 후보 생성 설명을 고침 (구조도 `audit/figures_v*/pipeline.*`는 Cycle 11에서 두 방식을 모두 적도록 고침) |
| (학습 입력 서술) | 행동 가중치, 시간 감쇠, 음성 표본, edge dropout이 위 행들과 구조도에 없었음 (I-132) | 모든 학습 모델의 입력은 사용자–상품 쌍 가중치 w다. 행동별로 조회 1, 장바구니 6, 구매 12를 주고, 학습 구간 마지막 시각 기준 반감기 7일(0.5^(경과일/7))로 감쇠해 쌍마다 더한다. ALS는 신뢰도 1 + α·w (α는 grid에서 선택), BPR은 w를 쌍 손실 가중치로 쓰고 음성은 학습 인기도에 비례해 뽑는다. GNN은 w(평균 1로 정규화)를 InfoNCE 손실 가중치로 쓰고, 전파는 가중치 없이 하며, edge dropout 0.1, 균일 음성 32개를 쓴다. 방법론 절에 이를 적는다 (구조도는 Cycle 21에서 이 내용을 넣어 다시 만듦) | `learning.py`(`BEHAVIOR_WEIGHTS`, `interaction_table`, `train_weighted_bpr`), `als_debug.py`(`train_als`), `model.py`(GNN 손실 가중치), `rolling_protocol.py`(`MF_COMMON`, `BPR_NEGATIVES`, `GNN_FIXED`) |
| 단일 분할 결과 | 시간 변동(fold 간 SD)이 모델 간 차이보다 큼 | rolling-origin 3 fold 결과를 주 결과로 쓰고, F1은 "이전 버전에서 노출된 구간", F2 시험 구간의 대부분(이벤트의 80%)은 "이전 버전에서 설정을 고를 때 쓴 validation 구간", F3 시험 구간의 80%는 "같은 v6 실행에서 F2 설정을 고른 validation 구간(F3 자신의 학습·선택에는 쓰지 않음)"이라고 명시 (I-106, I-113) |
| baseline (ALS/BPR/ALS+BPR/GNN) | 단순 규칙 baseline이 없었음. 반복 추천 비중이 큼 | Popularity, Recent, RecentPop과 novel 정책 결과를 추가 (`validation_report.md` §4) |
| 튜닝 공정성 | v6에서도 ALS가 grid 경계에서 선택됨 (I-23) | 한계로 명시: 경계 밖 validation 탐색에서 ALS가 +0.011 – +0.027 좋아짐 (`mf_boundary_probe.csv`). 확인 실험으로 v7을 보고: 처음 쓰는 구간 F4에서 ALS·BPR·GNN grid를 넓혔더니(Hybrid grid는 v6와 같음) ALS가 validation에서 GNN을 앞섰지만, 시험에서 Hybrid − ALS+BPR은 4칸 모두 비유의 (`validation_report.md` §12) |
| (v7) "확장 grid에서도 Hybrid가 가장 좋다" | v7에서 Hybrid가 4칸 모두 평균 1위지만, 칸마다 Hybrid가 유의하게 앞서지 못한 학습 baseline이 있음. ALS+BPR 대비 유의한 칸은 없음 | "baseline grid를 넓혀도 Hybrid가 모든 학습 baseline보다 유의하게 높은 칸은 없었다(F4, repeat 413명 / novel 306명)". 평균이 가장 높은 baseline은 repeat에서 BPR, novel에서 ALS+BPR이며, 그 대비 유의한 칸은 repeat Recall(Hybrid − BPR +0.0076)뿐. ALS+BPR 대비 차이 +0.0013 – +0.0077은 모두 비유의. GNN 대비는 repeat NDCG(+0.0118)와 novel Recall(+0.0151)만 유의 (`validation_report.md` §12.4). novel의 GNN·ALS 대비 결과는 "repeat validation으로 고른 GNN·ALS 대비"로 함께 적는다 (공개 사항 8) |

## 3. 원고에 넣어야 할 공개 사항

1. v1–v5의 개발 이력: v3 시험 결과를 본 뒤 탐색 공간을 바꿨다. v6는 사전등록 프로토콜로 다시 평가했다 (`experiment_provenance.md`).
2. 평가 모집단: 필터링 전 원본 학습 기간에 관측된 사용자 중, 원본 학습 기간에 등장한 상품에 시험 기간 양성 반응(장바구니·구매)을 하나 이상 남긴 사람만 평가한다 (`data_protocol.py`의 `reference`. core·IF 필터로 학습 데이터에서 빠진 사용자도 포함되며, 학습 벡터가 없는 사용자는 0점으로 계산된다). repeat 정답 집합에는 그 사용자의 시험 기간 양성 상품이 모두 들어간다(학습 카탈로그 밖 상품 포함, 분모에 들어감). novel은 여기서 원본 학습 기간에 접촉한 상품을 빼고, 남는 정답이 없는 사용자는 모집단에서 빠진다 (`policy_evaluation.py` 머리말). 전체 서비스 이용자에 대한 결과가 아니다.
3. 반복 추천 허용 평가와 신규 상품 평가를 따로 보고한다.
4. 투고본의 수치(NDCG 0.3710, Recall 0.7113. 원고 그림 스크립트 `Model/paper/viz.py` 7–8행의 "5. ALS+GNN" 값)와 v6 수치는 분할, 후보 처리, 분모가 달라 직접 비교할 수 없다.
5. 통계 방법: 사용자 단위 paired bootstrap 95% CI, sign-flip 검정, Holm 보정(비교 7개 × 지표 2개, 정책·fold별), seed 3개.
6. v7은 v6의 시험 결과를 본 뒤에 설계했다. 설계의 근거는 validation 전용 경계 탐색(I-23)이었고, F4의 시험·validation 점수는 설계 전에 본 적이 없다. v7은 한 번만 실행했고 결과를 보고 다시 실행하지 않았다. v7 수치는 v6 fold와 평균하지 않고 따로 보고한다.
7. v6와 v7은 서로 다른 PC·GPU에서 실행했다 (RTX 5060 Ti / RTX 3090 Ti). BPR·GNN은 GPU·torch 빌드에 따라 소수점 아래 값이 달라진다 (`reproducibility.md` §4).
8. 학습 설정(ALS, BPR, GNN, Hybrid의 GNN backbone)은 repeat validation으로 고르고, novel에서는 추론 설정(Hybrid 결합, ALS+BPR α)만 다시 골랐다. 그래서 novel의 ALS·GNN 단독은 novel로 튜닝되지 않았다. 같은 grid에서 novel로 골랐다면 GNN의 validation 점수가 F2·F3·F4에서 높아지고, F4에서는 Hybrid보다 높았다 (`validation_report.md` §8 항목 5). novel의 Hybrid − GNN, Hybrid − ALS 유의 결과는 이 조건과 함께 보고한다. 자동 생성 표(`audit/figures_v*/table_paired_tests.*`)의 `claim` 열은 이 조건을 표시하지 않으므로, 표를 원고에 옮길 때 주석을 단다.
9. 재고 신호(학습 종료 시점 이전의 `available`과 가중 γ)는 Hybrid(와 ablation `Hybrid_ALS_only`)에만 들어가고 baseline 7종에는 없다. validation에서 고른 γ는 8개 설정 모두 1보다 크다. 따라서 Hybrid − baseline 차이는 GNN 결합의 효과만이 아니다. 사후 분석이 있는 F1 repeat·seed 42에서는 Hybrid − ALS+BPR(NDCG +0.0031, Recall +0.0123)의 NDCG 약 79%, Recall 약 53%가 γ 1 대비 재고 가중 효과였다 (`validation_report.md` §6.2). 다른 fold·정책에서는 측정하지 않았다. 자동 생성 표와 그림에도 이 조건을 주석으로 단다.

## 4. 원고에 쓰면 안 되는 표현

- "state-of-the-art" 또는 "모든 baseline을 능가".
- 비유의한 차이에 대한 "우수", "개선".
- novel 정책에서 Hybrid가 GNN·ALS보다 "우수"하다는 표현 (비교 대상이 novel로 튜닝되지 않음, 위 공개 사항 8).
- Hybrid − baseline 차이를 GNN 결합만의 효과로 쓰는 표현 (재고 신호는 Hybrid에만 있음, 위 공개 사항 9).
- "봇 제거", "이상 사용자 정제" (최종 모델에 IF가 없음).
- "계산 비용 절감" (GNN 단독 대비 근거 없음).
- "cold-start 해결".
