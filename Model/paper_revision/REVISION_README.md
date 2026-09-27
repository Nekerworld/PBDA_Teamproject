# Revision 실험 프로토콜 v3

이 문서는 제출 논문의 실험 설명을 대체하는 재현 절차다. 아래 설정은 테스트 지표를 보기 전에 고정했다. 연결 확인용 짧은 학습의 숫자는 논문 성능이 아니다.

## 제출 논문과 다른 점

- 기본 학습 데이터에는 Isolation Forest를 쓰지 않는다. 이전 감사에서 IF가 학습 구매를 모두 제거했다. IF 유지와 고유 이웃 기준 core는 비교 조건일 뿐이며, 테스트 성능으로 고르지 않는다.
- 평가 사용자와 원래 정답은 필터 적용 전에 고정한다. 필터로 학습에서 빠진 사용자는 추천이 없고 원래 정답 기준 점수는 0이다.
- 지표는 세 범위로 저장한다. `original`은 고정 사용자의 테스트 positive 전체, `reference_warm`은 필터 전 학습 상품에 속한 그 정답, `conditional`은 필터 후 남은 사용자·상품만이다. 모델 비교의 주 표는 `original`이다.
- 행동 가중치는 view=1, addtocart=6, transaction=12이다. ALS는 Hu 신뢰도 `1 + 1.5 * 가중치`로 학습한다. BPR과 LightGCN의 pairwise loss는 이 가중치를 직접 곱한다. implicit 라이브러리의 BPR은 사용하지 않는다. 그 구현은 값이 달라도 같은 위치를 같게 학습했다.
- LightGCN 메시지 전달은 관측된 사용자·상품 쌍의 비가중 그래프다. 가중치는 그래프의 간선 세기가 아니라 손실에 들어간다. 조회와 구매는 같은 종류의 간선이지만 손실 가중치는 다르다.
- 속성 노드는 기본적으로 이름과 값이다. 같은 timestamp 경계 이전의 최신 값만 쓴다. 값은 공백으로 나누고, 숫자 토큰은 학습 데이터에서만 속성별 10개 분위 구간으로 묶는다. 상품 2개 미만에만 나타난 범주 값은 `<rare>`이다. `categoryid`는 범주 트리로 연결한다. `available`은 그래프에 넣지 않고 재정렬 단계에서만 gamma로 쓴다.
- 음성 표본은 해당 사용자가 학습에서 관측한 모든 상품을 제외한다. 모든 상품을 관측한 사용자는 학습 행을 건너뛴다.
- gamma는 0.5, 1.0, 1.5, 2.0 중 내부 검증의 original NDCG가 가장 큰 값이다. 동점이면 1에 더 가까운 값을 고른다. 외부 테스트는 선택에 쓰지 않는다.
- 반복 추천을 허용한다. 테스트 정답으로 후보를 바꾸지 않는다.

## 고정한 학습 설정

| 항목 | 값 |
|---|---|
| 분할 | 필터 전 유효 이벤트의 시간순 80:20. 같은 timestamp는 나누지 않음 |
| 기본 조건 | `no_if_events`: IF 없음, 이벤트 행 수 core 3 |
| 비교 조건 | `if_events`, `no_if_neighbors`. ablation은 첫 seed만 |
| seed | 42, 43, 44 |
| ALS | CPU, GPU 미사용, 스레드 8, 128 factors, 256 iterations, alpha 1.5 |
| weighted BPR | 128 epoch, batch 65536, Adam lr 0.001 |
| LightGCN | 최대 200 epoch, patience 25, 5 epoch마다 검증, 임베딩 8, layer 2. 선택 후 초기화하여 outer train에서 재학습 |
| 후보 / 대표 K | ALS+BPR 상위 500개, 보고 K=50 |
| 결합 | 후보 안에서 ALS와 LightGCN을 min-max 후 0.4:0.6. 그 다음 재고 gamma |
| 스레드 | `OPENBLAS`, `OMP`, `MKL`, PyTorch CPU 스레드는 1 |

IF를 켜는 비교 조건만 `contamination=auto`, 트리 200개, `n_jobs=1`, seed 42를 쓴다. 이상치 라벨은 봇 신원이 아니다.

## 실행

데이터 폴더에는 `events.csv`, `item_properties_part1.csv`, `item_properties_part2.csv`, `category_tree.csv`가 있어야 한다.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:MPLBACKEND='Agg'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
cd Model/paper_revision
python experiments.py
python viz.py --experiment-dir ../../data/research_revision_improve --output-dir ../../data/research_revision_improve/figures
```

기본은 improvement_v1이다. 자세한 기본값·레거시 옵션은 `IMPROVEMENT_NOTES.md`를 본다. 전처리만 할 때는 `--prepare-only`를 붙인다.

출력 폴더가 이미 있으면 덮어쓰지 않는다. 계획 파일과 코드 해시가 다르면 재개하지 않는다. `als_debug.py`의 implicit BPR 탐색 경로는 닫혀 있다.

## 결과 파일

- `datasets/filter_comparison.csv`: 조건별 학습 규모, 고정 사용자 수, 정답 보존율
- `truth_retention.csv`: 사용자별 원래 정답과 reference 정답의 보존율
- `ranking_metrics.csv`: 대표 K의 original 범위
- `topk_metrics.csv`: original, reference_warm, conditional
- `candidate_metrics.csv`: 후보 500개의 범위별 Recall
- `confidence_intervals.csv`: seed를 사용자별로 평균한 뒤 사용자 단위 bootstrap. 시간적 불확실성이 아니다
- `paired_differences.csv`: 같은 사용자·같은 seed의 Hybrid 차이
- `validation/gamma_validation.csv`: gamma 선택 곡선

검증은 `python -m pytest Model/test_revision.py -q`이다. 이 테스트는 합성 데이터로 프로토콜을 확인하며 RetailRocket 성능이 아니다.

## 완료된 RetailRocket 실행

2026-09-24 실행 폴더는 작업 사본의 `work/research_revision/retailrocket_v3`이다. 8개 실행이 모두 `complete.json`을 남겼고, 집계 CSV는 `experiment_manifest.json`의 해시와 맞다. 그림은 그 CSV만 읽어 `figures/`에 PNG 600 dpi와 PDF로 저장했다.

기록된 실행 시간 합은 2327초다. 환경은 OpenBLAS, OMP, MKL, PyTorch CPU 스레드 1, ALS는 CPU 8스레드이고 GPU를 쓰지 않으며, weighted BPR과 LightGCN은 CUDA(RTX 3090 Ti)다. 기본 조건의 LightGCN은 내부 검증 original NDCG가 epoch 5에서 가장 높아 5 epoch만 재학습했다. 학습 손실은 그 이후에도 줄었지만, epoch 수는 테스트 점수를 보고 바꾸지 않았다.

주 결과와 논문에 넣을 문장은 `PAPER_TEXT_REVISION.md`에 있다. 제출 PDF는 수정하지 않았다.
