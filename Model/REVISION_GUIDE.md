## 1. 목표

고객이 남긴 행동으로, 다음에 장바구니에 넣거나 구매할 상품을 순위대로 추천한다.

행동은 세 가지다.

- `view`: 상품을 봄. 가장 많고, 구매 의사는 약하다.
- `addtocart`: 장바구니에 담음.
- `transaction`: 구매함.

학습에는 세 행동을 모두 쓴다. 맞힘의 기준(정답)은 장바구니와 구매만이다. 조회만 한 사람을 학습에서 지우지는 않는다. 그 사람의 조회는 사용자와 상품의 관계를 배우게 한다. 다만 테스트 구간에 장바구니나 구매가 없으면 “맞혔는지”를 계산할 항목이 없어서, 성적표의 분모에는 넣지 않는다.

추천 시스템은 다섯 모델을 같은 규칙으로 비교한다.

1. ALS. 사용자-상품 행렬을 인수분해한다.
2. BPR. 본 상품이 안 본 상품보다 점수가 높도록 짝을 지어 학습한다.
3. ALS+BPR. 둘의 점수를 섞어 순위를 만든다.
4. LightGCN. 사용자-상품 그래프를 따라 임베딩을 전파한다. 표에서는 GNN으로 적는다.
5. Hybrid. ALS+BPR로 후보를 고른 뒤, 그 후보 안에서 ALS와 LightGCN 점수를 다시 섞고, 재고 정보로 순서를 조정한다.

개정판의 목표는 “Hybrid가 항상 1등”을 만드는 것이 아니다. 데이터 나누기, 필터, 정답, 학습, 점수 계산이 서로 어긋나지 않게 고정하고, 그 규칙 안에서 어떤 모델이 실제로 나은지 보는 것이다.

---

## 2. 데이터에서 사람이 어떻게 갈라지는지

RetailRocket 전체는 이벤트 2,756,101개, 방문자 1,407,580명이다. 장바구니나 구매가 한 번이라도 있는 사람은 38,865명이다. 나머지는 조회만 한다.

개정판은 유효 이벤트를 시간 순으로 앞 80%와 뒤 20%로 나눈다. 같은 시각의 이벤트는 학습과 테스트로 쪼개지 않는다. 뒤 20%가 시험이고, 그 구간의 방문자는 308,549명, 장바구니·구매 이벤트는 18,940개다.

시험 점수를 매길 사람은 여기서 더 줄어든다. 두 조건을 동시에 만족해야 한다.

- 그 사람이 앞 80% 학습 구간에 한 번은 나타나야 한다. 시험에만 처음 나온 사람은 사용자 벡터가 없다.
- 시험 정답 상품도 학습 구간에 존재해야 한다. 학습 때 한 번도 안 나온 상품은 상품 벡터가 없다.

그 교집합에서 장바구니나 구매가 있는 사람이 854명이다. 이 854명은 필터를 바꾸기 전에 고정한다. Isolation Forest를 켜든 끄든, core 기준을 바꾸든, 비교 대상 사람과 정답 상품 ID는 같다.

학습에 쓰는 사람은 854명이 아니다. 이벤트 수가 사용자와 상품 모두 3 이상이 될 때까지 core filtering을 반복하면, 학습 이벤트 974,155개, 사용자 141,660명, 상품 58,556개가 남는다. 이 141,660명으로 ALS와 그래프를 학습한다. 성적표만 854명에게 매긴다.

854명 중 409명만 최종 학습 행렬에 남는다. 425명은 core 기준을 넘기지 못해 사용자 벡터가 없고, 추천 목록이 비어 있다. 854명 평균(original)은 그 빈 목록을 0점으로 넣는다. 409명만 평균 낸 값(conditional)은 따로 두며, 둘을 한 칸에 적지 않는다.

조회만 하고 장바구니·구매가 없는 사람은 학습에는 남을 수 있다. 조회 910,046개가 최종 학습에 들어 있다. 가중치는 조회 1, 장바구니 6, 구매 12다. 그 사람은 시험 구간에 정답이 없으면 854명 성적표에는 없다.

---

## 3. 예전 코드가 하던 일

`Model/paper/`의 파이프라인은 대략 이렇다.

1. Isolation Forest로 이상 상품을 걷어 낸다.
2. ALS로 협업 필터링 추천을 만든다. BPR을 섞을 수 있다.
3. 사용자, 상품, 속성, 카테고리로 그래프를 만들고 GNN 임베딩을 학습한다.
4. ALS와 GNN 점수를 가중 평균해 Hybrid 순위를 만든다.
5. Precision, Recall, F1, Hit Rate, NDCG로 평가한다.

논문 그림에 하드코딩된 예전 성적은 `Model/paper/viz.py`에 있다. ALS+GNN의 NDCG@50은 0.371, Recall@50은 0.711이다. K=10 Recall도 0.510이다. NDCG와 Recall이 둘 다 0.7이었다는 기억과는 다르다. 0.7대는 Recall이다.

이 높은 Recall에는 평가 규칙이 한몫한다. 예전 `als_debug.py`는 추천 순위를 만들 때 학습에서 본 상품을 지운다. 그런데 테스트 정답은 그 삭제에서 빼 둔다.

```python
exclude = train_items_u - test_positives_u
```

학습 때 본 상품은 순위에서 빠지고, 나중에 장바구니나 구매가 된 상품만 순위에 남길 수 있다. 정답이 “가리지 말 상품”으로 지정된 상태에서 Top-50을 만든 것이다. 평균도 정답이 있는 사람만 넣고, 추천이 없는 사람은 건너뛴다.

그 밖에 예전 구현은 이런 점이 논문 문장과 어긋나기 쉬웠다.

- Isolation Forest를 본실험에 쓰면 학습 구매가 크게 줄어든다. 개정 감사에서는 학습 transaction이 전부 사라졌다. 이상치 라벨은 봇을 찾아낸 결과가 아니다.
- implicit 라이브러리의 BPR은 가중치가 달라도 같은 위치면 같게 학습하는 경로가 있었다. 조회와 구매를 다르게 둔다는 문장과 구현이 달랐다.
- 필터를 통과한 사람만으로 점수를 내면, 탈락한 사람이 평균에서 빠져 숫자가 커진다.
- gamma나 epoch를 시험 점수를 보고 고르면, 시험 정보가 학습 선택으로 샌다.

세션 추천 논문의 HR@20이 0.45~0.55인 것과도 자리가 다르다. 그 논문은 세션의 다음 클릭 하나를 맞힌다. 이 프로젝트는 사용자별 장바구니·구매 집합을 전체 상품 목록에서 순위로 매긴다.

---

## 4. 개정판이 고정한 규칙

규칙의 이름은 `chronological-fixed-reference-v3`다. 시험 점수를 보기 전에 기본 조건을 잠갔다. 구현은 `paper_revision/data_protocol.py`, `experiments.py`, `ranking_evaluation.py`에 있다.

- 분할은 필터 전 유효 이벤트의 시간순 80:20이다. 내부 검증은 학습 구간을 다시 시간순으로 나눈다.
- 본실험은 Isolation Forest를 끈다 (`no_if_events`). 켠 조건과 “서로 다른 이웃 수” core는 비교 실험만 한다. 시험 점수로 본실험을 고르지 않는다.
- core는 이벤트 행 수 3 이상이다. 이웃 수가 아니다. 사용자와 상품이 모두 기준을 넘을 때까지 반복한다.
- 정답은 `addtocart`와 `transaction`이다.
- 대표 지표는 854명, 그 사람들의 원래 테스트 정답, K=50이다. 학습에서 빠진 425명은 0점이다.
- 같은 추천에 대해 세 범위를 저장한다. `original`은 854명 전체, `reference_warm`은 필터 전 학습 상품에 속한 정답, `conditional`은 필터 후 남은 사용자·상품만이다. 논문의 주 표는 `original`이다.
- 반복 구매가 가능하므로 학습에서 본 상품도 다시 추천할 수 있다. 테스트 정답으로 후보를 바꾸지 않는다.
- epoch, gamma, Hybrid 융합 가중치는 내부 검증의 original NDCG로만 고른다. 바깥 테스트는 선택에 쓰지 않는다.
- 음성 표본은 그 사용자가 학습에서 본 상품을 빼고 뽑는다. 모든 상품을 본 사용자는 그 학습 행을 건너뛴다.
- seed는 42, 43, 44다. ALS는 CPU, BPR과 LightGCN은 GPU다.

`python -m pytest test_revision.py -q`는 이 규칙이 코드에서 깨지지 않는지 보는 합성 데이터 테스트다. 그 테스트의 점수는 RetailRocket 성능이 아니다. 현재 28개가 통과한다.

---

## 5. 모델을 어떻게 바꿨는지

같은 평가 규칙 안에서 학습 스택만 세 번 올렸다. 데이터 분할과 854명은 같다.

### 5.1 첫 공정 재측정 (revision v3)

예전 파이프라인을 위 규칙에 맞춘 버전이다.

- ALS 신뢰도는 `1 + 1.5 * r`이다. `r`은 조회 1, 장바구니 6, 구매 12를 사용자-상품 쌍에 합친 값이다. 128요인, 256회, CPU.
- BPR은 그 가중치를 손실에 직접 곱하는 torch 구현이다. implicit BPR 경로는 쓰지 않는다. 128 epoch.
- LightGCN은 사용자-상품을 비가중 간선으로 연결하고, 속성 값은 학습 구간의 최신 값만 노드로 둔다. 임베딩 차원 8, 층 2. 재고 `available`은 그래프에 넣지 않고, 재정렬 배수 gamma로만 쓴다.
- Hybrid는 ALS+BPR 상위 500개를 후보로 삼고, 그 안에서 ALS와 LightGCN을 min-max로 섞은 뒤 gamma를 곱한다. 검증에서 gamma는 1.5가 골랐다.

이 단계에서 LightGCN 검증 점수는 초반(약 5 epoch)에 가장 높고 이후 내려갔다. 그래서 짧은 학습만 다시 돌렸다. 그래프 모델이 거의 순위를 못 만들었다.

### 5.2 용량과 융합을 올린 버전 (improvement)

ALS와 BPR을 손보고, 그래프 용량과 Hybrid 선택을 넓혔다. 평가 규칙은 그대로다.

- ALS α를 2.0, 정규화를 0.02로 바꿨다.
- 최근 행동에 더 가중하는 시간 감쇠(반감기 7일)를 넣었다.
- 음성 상품을 균등 추출 대신 인기 상품 쪽에서 더 뽑게 했다.
- LightGCN은 차원 64, 3층, 간선 드롭아웃 0.1, 가중 전파, 아이템-아이템 간선을 추가했다.
- Hybrid의 융합 방식, ALS 비중, 재정렬 폭, 후보 수, gamma는 검증에서만 고른다.

그래프는 커졌지만 검증이 1 epoch 근처에서 꺾여, 사실상 거의 학습되지 않은 체크포인트가 선택됐다. 속성 노드가 약 4,900개 붙는 이종 그래프와 가중 전파, BPR 손실이 서로 잘 맞지 않았다.

### 5.3 지금 논문에 쓸 버전 (paper v2)

LightGCN만 표준에 가깝게 다시 만들었다. ALS와 BPR은 improvement와 같다. 결과 폴더는 `data/research_revision_paper/`다.

- 그래프는 사용자-상품만 잇는다. 속성 노드, 카테고리 노드, 아이템-아이템 간선은 기본으로 끈다.
- 메시지 전달은 비가중이다. 행동 차이는 간선 세기가 아니라 손실 가중치다.
- 손실은 음성 32개의 InfoNCE다. 온도는 0.2다.
- 최소 20 epoch가 지나기 전에는 검증 점수로 체크포인트를 고르지 않는다. 너무 이른 1 epoch 선택을 막기 위해서다.
- 차원 64, 3층, 간선 드롭아웃 0.1은 유지한다.
- seed 42에서는 검증 NDCG가 190 epoch 근처까지 올라 그 길이를 다시 학습했다.

Hybrid의 골격은 같다. 후보를 ALS+BPR에서 가져오고, 그 안에서 ALS와 LightGCN을 섞는다. 검증이 고른 설정은 ALS 비중 0.6, min-max 융합, gamma 1.5 근처다. 후보 수는 seed에 따라 200 또는 1000이다.

---

## 6. 성적

대표 숫자는 original, K=50, 854명, seed 42·43·44의 평균이다. ±는 표본 표준편차다. v3의 seed별 CSV는 `data/research_revision`에 남아 있지 않아, 당시 확정 표(`revision_v3_results_report.md`)의 평균과 표준편차를 썼다. improvement와 paper v2는 `experiment_metrics.csv`다.
 
NDCG@50:

| 모델 | v3 | improvement | paper v2 |
|------|-----:|-----------:|---------:|
| ALS | 0.067 | 0.074 | 0.074 |
| BPR | 0.102 | 0.098 | 0.098 |
| ALS+BPR | 0.099 | 0.105 | 0.105 |
| LightGCN | 0.022 | 0.046 | 0.106 |
| Hybrid | 0.058 | 0.082 | 0.091 |

Recall@50:

| 모델 | v3 | improvement | paper v2 |
|------|-----:|-----------:|---------:|
| ALS | 0.118 | 0.116 | 0.116 |
| BPR | 0.135 | 0.132 | 0.132 |
| ALS+BPR | 0.146 | 0.145 | 0.145 |
| LightGCN | 0.041 | 0.078 | 0.164 |
| Hybrid | 0.112 | 0.127 | 0.146 |

paper v2에서 LightGCN이 NDCG와 Recall 모두 ALS+BPR을 앞선다. 차이는 NDCG에서 작고 Recall에서 크다. Hybrid는 improvement보다 올랐지만, 세 실험 모두에서 ALS+BPR과 LightGCN을 동시에 이긴 적이 없다. paper v2에서는 둘보다 낮다.

검증에서도 같은 순서다. paper v2의 단독 LightGCN 검증 NDCG는 약 0.128이고, 선택된 Hybrid는 약 0.108이다. Hybrid 후보는 ALS+BPR이 고른 상품 안에 갇혀 있어서, 전체 목록을 보는 LightGCN보다 위로 올라오지 못한다.

409명만 보는 conditional은 0점인 425명이 빠져 숫자가 커진다. paper v2, seed 42, K=50에서 LightGCN은 NDCG 약 0.23, Recall 약 0.35다. 이 값은 “학습된 사용자에 대한 순위 품질”이고, 854명 original과 같은 칸에 적으면 안 된다.

예전 ALS+GNN Recall 0.711, NDCG 0.371과 paper v2의 0.146, 0.106을 나란히 두면 모델이 나빠진 것처럼 보인다. 그 차이의 큰 부분은 테스트 정답을 마스킹에서 빼 주던 예전 규칙, 그리고 추천이 없는 사람을 평균에서 빼던 규칙이다. 개정판은 그 두 규칙을 쓰지 않는다.

---

## 7. 무엇이 나아졌고, 무엇을 주장하면 안 되는지

나아진 것:

- 시험 정답이 추천 목록을 만드는 과정에 들어가지 않는다.
- 필터를 바꿔도 비교 대상 854명과 정답이 같다. 필터 때문에 탈락한 사람은 0점으로 남아, 점수가 남은 사람만으로 부풀지 않는다.
- Isolation Forest가 학습 구매를 지운다는 사실을 본실험에서 분리했다.
- 조회와 구매의 가중치가 BPR 손실에 실제로 들어간다.
- gamma와 epoch를 시험 세트가 아니라 시간순 검증으로만 고른다.
- 같은 규칙 안에서 LightGCN이 NDCG 0.022에서 0.106, Recall 0.041에서 0.164로 올라, ALS+BPR을 앞서는 단일 모델이 됐다.
- seed 세 개의 평균과 표준편차, 검증 곡선, 필터 비교 그림을 다시 그렸다.

주장하면 안 되는 것:

- Hybrid가 ALS+BPR이나 LightGCN보다 좋다.
- Isolation Forest가 봇을 제거했다.
- 예전 Recall 0.7과 지금 Recall 0.16이 같은 문제의 점수다.
- 세션 추천 논문의 HR@20과 이 NDCG@50을 직접 비교할 수 있다.
- conditional 점수(409명)를 854명 전체의 성적으로 쓸 수 있다.
- 425명에게 추천 목록을 만들어 주는 것이 이번 협업필터링의 목표다. 그 사람들에게 목록을 주려면 인기 상품이나 속성 기반의 cold-start가 따로 있어야 하고, 지금 모델에는 그 장치가 없다.

---

## 8. 그림과 파일을 어디서 보나

논문에 넣으려고 만든 세 실험 비교 그림은 여기 있다.

`Model/paper_revision/figures/revision_v3/`

- `three_protocol_metrics.png`와 `.pdf`. 다섯 모델의 NDCG@50, Recall@50. 막대 세 개가 v3, improvement, paper v2다.
- `three_protocol_trajectory.png`와 `.pdf`. 같은 숫자를 실험 순서로 이은 그림. LightGCN만 마지막에 ALS+BPR 위로 올라가고 Hybrid는 그 아래에 있다.
- `three_protocol_metrics.csv`. 그림에 쓴 평균과 표준편차.
- 그 폴더의 `filter_comparison`, `confidence_intervals`, `gamma_validation`, `run_overview`는 v3 한 실험의 그림이다. 세 실험 비교가 아니다.

다시 그리려면:

```powershell
cd Model/paper_revision
python plot_three_protocols.py
```

실험 전체를 다시 돌리려면:

```powershell
cd Model/paper_revision
python experiments.py --output-dir ../../data/research_revision_paper --device cuda
python viz.py --experiment-dir ../../data/research_revision_paper --output-dir ../../data/research_revision_paper/figures
python -m pytest test_revision.py -q
```

예전 v3 설정만 다시 돌릴 때는 `python experiments.py --legacy-v3 --output-dir ../../data/research_revision`이다. 출력 폴더에 계획 파일이 이미 있고 코드 해시가 다르면 재개하지 않는다. 새 폴더를 쓴다.

코드 지도:

| 파일 | 역할 |
|------|------|
| `data_protocol.py` | 시간 분할, core, 854명 고정, 세 가지 정답 범위 |
| `learning.py` | 행동 가중, 시간 감쇠, 음성 표본, BPR 손실, InfoNCE |
| `model.py` | ALS, LightGCN, Hybrid 재정렬 |
| `recommendation_scoring.py` | 후보 점수, min-max, RRF, 게이트 융합 |
| `ranking_evaluation.py` | NDCG, Recall과 사용자 단위 기록 |
| `experiments.py` | seed 반복, 검증으로 설정 선택, 그 다음 바깥 테스트 |
| `viz.py` | 한 실험 폴더의 그림 |
| `plot_three_protocols.py` | v3·improvement·paper v2 비교 그림 |
| `test_revision.py` | 프로토콜이 깨졌는지 보는 테스트 |

더 짧은 기록은 `REVISION_README.md`, `revision_v3_results_report.md`, `PAPER_V2_RESULTS.md`에 있다. 그 문서의 숫자는 이 설명과 같은 실험을 가리킨다.
