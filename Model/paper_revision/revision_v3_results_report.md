# RetailRocket revision v3 결과 보고

이 문서는 코드 변경, 합성 검증, 실제 전처리, 본 학습 결과를 구분한다. 제출 PDF는 고치지 않았다. 짧은 epoch 연결 확인과 pytest 숫자는 논문 성능이 아니다.

## 코드와 설계

작업 대상은 `work/research_revision/Model`이다. 원본 프로젝트의 `model.py`, `als_debug.py`, `viz.py`, `EDA/EDA.py` SHA256은 이 작업을 시작하기 전의 값과 같았다. 사용자는 그 파일을 새로 고치지 않았다.

테스트 지표를 보기 전에 고정한 기본 조건은 Isolation Forest를 끄고, 이벤트 행 수 3 기준으로 core filtering을 수렴까지 반복하는 것이다. 고유 이웃 core와 IF는 비교 조건이다. 평가 사용자와 원래 positive는 필터 전 학습 상품 기준 warm-start 집단으로 고정한다. 세 조건의 `reference_id`와 `original_truth_id`는 같다.

행동 가중치는 view 1, addtocart 6, transaction 12다. ALS는 Hu 신뢰도 `1 + 1.5 * r`을 CPU에서 학습한다. BPR과 LightGCN pairwise loss는 그 가중치를 곱한다. implicit BPR 경로는 예외를 내도록 닫았다. LightGCN 그래프 자체는 관측 쌍의 비가중 간선이다. 속성 노드는 학습 구간 최신 값이고, `available`은 그래프에 넣지 않으며 재정렬 gamma로만 쓴다. 음성 표본은 학습에서 본 상품을 빼고, 모든 상품을 본 사용자는 건너뛴다. gamma와 epoch는 시간순 내부 검증의 original NDCG로만 고르고, 선택된 epoch만큼 초기화 후 outer train에서 다시 학습한다.

## 합성 검증

`test_revision.py` 28개가 통과했다. 통과 항목은 조건 간 동일 reference, 필터 탈락 사용자 0점, 세 평가 범위, 음성 표본과 saturated 사용자, 가중치가 BPR과 GNN 손실에 들어가는 것, 속성 값이 그래프를 바꾸는 것, `available`이 노드가 아닌 것, 검증으로 epoch를 고른 뒤 그 길이만큼 재학습하는 것, outer test로 gamma를 고르면 거부하는 것, 계획 해시가 다르면 재개를 거부하는 것이다. 이 테스트의 점수는 RetailRocket 성능이 아니다.

## 실제 전처리

유효 이벤트 시간순 80:20 이후, 필터 전 학습 상품에 positive가 있는 고정 사용자는 854명이다. 세 조건이 같은 reference와 같은 원래 정답 파일을 쓴다.

| 조건 | 학습 이벤트 | 학습 구매 | 조건부 사용자 | 854명 평균 원래 정답 보존율 | 조건부 사용자만의 보존율 |
|---|---:|---:|---:|---:|---:|
| IF 없음, 이벤트 core 3 | 974,155 | 17,305 | 409 | 0.447 | 0.934 |
| IF 있음, 이벤트 core 3 | 142,164 | 0 | 55 | 0.025 | 0.389 |
| IF 없음, 고유 이웃 core 3 | 652,381 | 11,320 | 297 | 0.316 | 0.908 |

IF 조건은 학습 transaction 17,864건을 모두 제거했다. 이상치 라벨은 봇 식별이 아니다. 이웃 기준 core는 구매를 일부 남기지만, 테스트 성능을 보고 기본 조건을 바꾸지 않았다. 기본 조건에서 854명 중 425명은 최종 학습에 없어 원래 정답 평가에서 0점이다.

## 본 학습

출력은 `work/research_revision/retailrocket_v3`이다. 실행 8개가 모두 끝났다. 기본 조건 seed 42, 43, 44와, seed 42의 ablation 다섯 개다. ALS 256회, weighted BPR 128 epoch, LightGCN 최대 200 epoch, patience 25. 환경은 OpenBLAS, OMP, MKL, PyTorch CPU 스레드 1, ALS CPU 8스레드, BPR과 LightGCN은 CUDA다. 기록된 실행 시간 합은 2327초다.

기본 조건의 주 표는 854명, 원래 positive, K=50, 세 seed 평균 ± 표본 표준편차다.

| 모델 | NDCG@50 | Recall@50 |
|---|---:|---:|
| ALS | 0.06746 ± 0.00030 | 0.11834 ± 0.00175 |
| BPR | 0.10178 ± 0.00124 | 0.13536 ± 0.00279 |
| ALS+BPR | 0.09944 ± 0.00049 | 0.14583 ± 0.00079 |
| LightGCN | 0.02233 ± 0.00103 | 0.04140 ± 0.00360 |
| Hybrid | 0.05817 ± 0.00086 | 0.11217 ± 0.00169 |

Hybrid의 사용자 bootstrap 95% 구간은 NDCG 0.0478–0.0690, Recall 0.0945–0.1315다. 이 구간은 끝난 seed에 대한 사용자 재표본이며 시간 불확실성이 아니다. 같은 seed에서 Hybrid NDCG에서 ALS+BPR NDCG를 뺀 paired 차이는 세 seed 모두 약 −0.041이고, 95% 구간에 0이 없다. 수정된 프로토콜에서 Hybrid는 ALS+BPR보다 낮다.

같은 추천의 조건부 지표는 409명만 포함한다. 세 seed 평균 조건부 NDCG@50은 Hybrid 0.124, ALS+BPR 0.213, BPR 0.218이다. 이 값은 425명의 0점을 빼서 높아진 것이며 주 비교가 아니다.

후보 500개의 원래 정답 Recall은 세 seed 평균 0.194 ± 0.003이다. 같은 후보의 reference-warm Recall은 0.198, 조건부 Recall은 0.424다. 후보 단계에서도 원래 정답 기준 회수율은 조건부 숫자보다 낮다.

gamma는 기본 세 seed 모두 검증 original NDCG로 1.5가 선택됐다. 검증에서 1.5가 2.0보다 높고 0.5는 뚜렷이 낮다. LightGCN은 기본 조건에서 epoch 5의 검증 NDCG가 가장 높아 5 epoch만 재학습했다. epoch 25의 학습 손실은 더 낮았지만 검증 NDCG는 epoch 5보다 낮았다.

seed 42 ablation의 Hybrid original NDCG@50은 기본 0.0590, 이진 pairwise 0.0585, 속성 이름만 0.0590, 과거 메타데이터 0.0542, 이웃 core 0.0464, IF 0.0067이다. 과거 메타데이터, 이웃 core, IF의 paired 구간에는 0이 없다. 이진 pairwise와 속성 이름 ablation의 구간에는 0이 들어 있다. ablation은 seed가 하나다.

그림은 `retailrocket_v3/figures`에 있다. `filter_comparison`, `confidence_intervals`, `run_overview`, `gamma_validation`의 PNG와 PDF다. 그림의 숫자는 sealed CSV에서 읽었고, 고정 배열을 넣지 않았다.

## 남은 한계

- 제출 PDF와 그 안의 기존 성능 표는 그대로다. 논문에 넣을 문장은 `PAPER_TEXT_REVISION.md`다.
- 수정 모델은 제출 논문의 모델과 같지 않다. 그래프는 여전히 비가중이고, 행동 차이는 손실 가중치다.
- bootstrap은 시간 분할을 다시 뽑지 않는다. ablation은 한 seed다.
- 조건부 Recall은 원래 정답 전체의 Recall이 아니다.
- IF 라벨을 봇 제거라고 쓰면 안 된다.
- 원본 `Model`에 검증된 코드를 넣었다. 적용 직전 `model.py`, `als_debug.py`, `viz.py`의 SHA256은 작업 시작 시 해시와 같았다. 백업은 `work/original_model_backup`이다. `EDA/EDA.py`와 제출 PDF, 실험 출력 폴더는 원본 저장소에 복사하지 않았다. 원본 `README.md`와 `MODEL_README.md`는 본문을 유지한 채 revision 안내 문장만 앞에 붙인 버전으로 바꿨다. `requirements.txt`에는 `implicit`와 `pytest`를 추가했다.
