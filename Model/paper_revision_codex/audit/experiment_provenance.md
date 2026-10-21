# Experiment provenance (실험 이력과 산출물 출처)

## 1. 실험 이력 (v1–v6)

| 버전 | 위치 | 선택 데이터 | 시험 열람 | 비고 |
|---|---|---|---|---|
| v1–v2 | `results/hybrid_joint_8020_v1`, `_v2` | 내부 validation | v3 이전에는 없음 | 내부 validation 도입, 후보 합집합 수정 |
| v3 | `results/hybrid_joint_8020_v3` | 내부 validation (seed 42) | **최초 열람**, Recall 목표 미달 | 후보 수 상한 확장 |
| v4–v5 | `results/hybrid_joint_8020_v4`, `_v5` | 내부 validation (3 seed 평균) | 같은 시험 구간을 **다시 열람** | backbone 3개와 독립 GNN을 v3 시험을 본 뒤 사람이 정함. v3 학습 산출물 재사용 |
| **v6** | `audit_runs/v6` (요약은 `audit/results_v6`) | 내부 validation (seed 42) | fold당 1회 (아래 로그) | 사전등록 plan, 원본부터 일괄 실행 |

v1–v5의 수치와 코드는 수정하거나 삭제하지 않았다. 과거 소스는 `source_*` 폴더에 그대로 있다.

## 2. v6 실행 식별자

| 항목 | 값 |
|---|---|
| plan SHA256 | `c56ada95ecce…` (`audit/results_v6/plan.json`) |
| 코드 commit (plan 고정 시점) | `29ba891` (`dev`, 작업 트리 깨끗함) |
| 실행 commit (시험 열람 시점) | `074443e` (문서만 추가됨. 코드 해시는 plan과 동일함을 실행 시 검사) |
| 원본 입력 | `events.csv` 외 3개. SHA256은 `plan.json` → `raw_input_sha256` |
| 환경 | Python 3.11.9, numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, sklearn 1.9.0, torch 2.15.0.dev20260921+cu132, implicit 0.7.3, CUDA 13.2, RTX 5060 Ti |
| 실행 시간 | 2026-10-20 07:52 UTC 시작, 중단 후 재개, 2026-10-21 05:09 UTC 완료 (exit 0) |

### 시험 접근 로그 (`test_access_log.jsonl`)

| fold | 목적 | plan | git |
|---|---|---|---|
| F1_hist | sealed_test | c56ada95ecce | 074443e |
| F1_hist | posthoc_candidate_gamma | c56ada95ecce | 074443e |
| F1_hist | posthoc_if_ablation | c56ada95ecce | 074443e |
| F2_mid | sealed_test | c56ada95ecce | 074443e |
| F3_early | sealed_test | c56ada95ecce | 074443e |

각 fold의 시험은 `selection.json`을 고정한 뒤 한 번만 열었다. 사후 분석은 F1에서만 했고, 선택에는 쓰지 않았다.

## 3. 중단과 재개

| 시각 (KST) | 사건 | 처리 |
|---|---|---|
| 10-20 17:26 | 첫 실행이 셸 세션 정리로 외부 종료됨 | 같은 명령으로 재개 |
| 10-20 17:30 | 두 번째 실행이 Ctrl+C로 종료됨 | 사용자 요청으로 중지, 커밋 `074443e` |
| 10-21 08:42 | 재개 직후 다시 Ctrl+C로 종료됨 | Ctrl+C 이벤트를 받지 않는 분리 프로세스로 재실행 (`audit_runs/launch_v6.py`). 이후 완료 |

- 모든 중단은 시험 열람 전, F1 내부 validation 단계에서 일어났다.
- 재개할 때 plan 해시와 코드 해시가 같은지 검사했다.
- 완료된 산출물(MF grid, GNN 작업 4개)은 재사용하고, 끊긴 GNN 작업은 처음부터 다시 학습했다.
- GNN 작업은 학습 시작 시 seed를 고정하므로 중단 여부와 무관하게 같은 절차를 따른다.

## 4. 산출물과 설정의 연결

```
plan.json (plan_sha256, code_sha256, raw_input_sha256, git, environment)
└─ F*/fold_raw/fold_raw.json      fold 끝 시점, 원본/잘린 events 해시
   └─ F*/prepared/manifest.json   protocol_id, 분할 파일 해시
      ├─ F*/leakage_check.json    28개 독립 검사 (모두 통과)
      ├─ F*/selection_mf.json, validation_mf_grid.csv
      ├─ F*/gnn_val/*/job_result.json (+ gnn_artifacts.json: protocol_id, 해시)
      ├─ F*/validation_hybrid_{policy}_{backbone}.csv
      └─ F*/selection.json        plan_sha256, 선택, 예산, 경계 도달 여부
         └─ test_access_log.jsonl (selection_sha256)
            └─ F*/runs/seed_*/ (als_artifacts.json), F*/runs/*gnn_seed_*/ (gnn_artifacts.json)
               └─ F*/report/*.csv
```

산출물을 불러올 때 `require_artifacts`가 protocol_id와 파일 해시를 검사한다. 맞지 않으면 실행이 멈춘다.

## 5. 저장소에 추적하는 요약

`audit/results_v6/`에 61개 파일(`SHA256SUMS.txt` 포함)이 있고, 각 파일의 SHA256은 `audit/results_v6/SHA256SUMS.txt`에 있다. 모델, 임베딩, 분할 CSV, 사용자별 추천 목록은 `audit_runs/`에만 있다(gitignore 대상, 수 GB).

## 6. 감사 보조 실행 (바깥 시험 지표를 계산하지 않음)

- 결정성 검사는 평가 사용자 목록만 쓰고 정답으로 지표를 계산하지 않는다. 비교하는 것은 임베딩 차이와 Top-50 목록 일치도뿐이다.
- 경계 탐색은 내부 validation 분할만 읽는다.

| 실행 | 스크립트 | 출력 |
|---|---|---|
| 결정성 재학습 | `report_tools/determinism_check.py` | `audit/results_v6/determinism_mf_blas1.csv` (첫 실행은 BLAS 24스레드 결함 때문에 `determinism_run1_blas24.csv`로 보존) |
| MF grid 경계 탐색 (validation만 사용) | `report_tools/mf_boundary_probe.py` | `audit/results_v6/mf_boundary_probe.csv` (첫 실행은 같은 결함 때문에 `mf_boundary_probe_run1_blas24.csv`로 보존) |
