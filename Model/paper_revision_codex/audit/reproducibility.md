# 재현성 (Reproducibility)

## 1. 입력 데이터

원본 RetailRocket 파일 4개는 `<repo>/data/`에 있으며 저장소에는 포함되지 않음 (`.gitignore`). v6 `plan.json`의 `raw_input_sha256`에 해시를 기록함. 이 해시는 v1–v5 manifest의 `input_hashes`와 같음 (이 PC에서 확인).

| 파일 | 용도 |
|---|---|
| `events.csv` | 이벤트 (view / addtocart / transaction) |
| `item_properties_part1.csv`, `item_properties_part2.csv` | 시점별 상품 속성 (`available`, `categoryid` 등) |
| `category_tree.csv` | 정적 카테고리 트리 |

## 2. 환경

| 항목 | 이 감사 실행 (2026-10-20) | v1–v5 원래 실행 |
|---|---|---|
| Python | 3.11.9 | 3.12.3 |
| numpy | 2.4.6 | 2.1.3 |
| pandas | 3.0.6 | 2.3.0 |
| scikit-learn | 1.9.0 | 1.6.1 |
| torch / CUDA / GPU | `audit_runs/v6/plan.json` → `environment` | v5 `experiment_plan.json` |

v6 실행 환경 전체(torch, implicit, scipy, CUDA, GPU 이름)는 `plan.json`에 자동으로 기록됨.

## 3. 처음부터 실행하는 방법

```powershell
cd Model/paper_revision_codex
# 1) 회귀 테스트 (62개)
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:MPLBACKEND='Agg'
python -m pytest test_revision.py test_hybrid_improvement.py test_codex_audit.py test_robust_hybrid.py `
  test_fast_fusion.py test_v6_protocol.py test_v6_smoke.py -q -p no:cacheprovider
# 2) 누수 검사만 단독 실행 (전처리된 디렉터리 대상)
python audit_tools/leakage_check.py --processed <processed_dir> --raw ../../data --gnn --output leak.json
# 3) 사전등록 rolling-origin 실험 (git 작업 트리가 깨끗해야 시작됨)
python rolling_protocol.py run --root audit_runs/v6 --raw ../../data
```

- 3단계는 중간에 멈춰도 같은 명령으로 이어서 실행할 수 있음. 완료된 산출물은 다시 만들지 않음.
- 코드나 plan이 바뀐 상태로 같은 `--root`를 쓰면 오류를 내고 멈춤.
- 이미 열린 fold의 시험을 다른 plan으로 다시 열 수 없음.

## 4. 결정성과 seed

- seed {42, 43, 44}를 ALS(`implicit` CPU), BPR(torch), GNN(torch CUDA), numpy RNG에 같은 값으로 전달함.
- 하이퍼파라미터 선택에는 seed 42만 씀 (`selection_seed`). 시험 보고에는 세 seed의 평균과 표준편차를 씀.
- BPR 체크포인트 콜백이 학습 결과를 바꾸지 않음을 확인함 (CPU, 비트 단위 동일: `test_bpr_checkpoints_do_not_change_training`).
- GPU의 sparse 행렬곱은 결정성이 보장되지 않음. 같은 seed로 두 번 학습한 결과 차이는 `validation_report.md`의 결정성 절에 기록함.

## 5. 산출물과 설정의 연결 (provenance)

| 단계 | 기록 위치 |
|---|---|
| 전처리 | 각 분할 디렉터리의 `manifest.json` (protocol_id, 파일 해시, config, 버전) |
| 학습 산출물 | `als_artifacts.json` / `gnn_artifacts.json` (`stamp_artifacts`). 불러올 때 `require_artifacts`로 해시와 protocol_id를 검사 |
| 선택 | fold별 `selection.json` (plan 해시, 예산, 상한 도달 여부) |
| 시험 열람 | `test_access_log.jsonl` (selection 해시, git commit) |
| GNN 작업 | `job_specs/*.json`, 작업별 `job_result.json`, `.log` |

## 6. 저장소에 넣지 않는 것

- 모델, 임베딩, 분할 CSV (수 GB): `audit_runs/` 전체가 gitignore 대상.
- 요약 표는 `audit/results_v6/`에 복사해 추적하고, SHA256을 `experiment_provenance.md`에 기록함.

## 7. 알려진 재현성 한계

1. 라이브러리 버전이 바뀌면서 감사용 요약 `if_feature_distributions.csv`의 해시가 달라짐. 핵심 분할 파일은 바이트 단위로 같음 (I-17).
2. `selection.json`의 `job_dir`는 절대경로임 (I-20). 다른 PC에서는 `refits.json`의 상대경로를 쓰면 됨.
3. v1–v5 결과는 다른 PC의 절대경로를 포함하고, v5는 v3의 학습 산출물을 재사용함. 원래 환경과 산출물 없이는 v5를 비트 단위로 재현할 수 없음. v6가 이를 대체하는 일괄 재실행임.
