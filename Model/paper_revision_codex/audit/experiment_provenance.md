# Experiment provenance (실험 이력과 산출물 출처)

## 1. 실험 이력 (v1–v7)

| 버전 | 위치 | 선택 데이터 | 시험 열람 | 비고 |
|---|---|---|---|---|
| v1–v2 | `results/hybrid_joint_8020_v1`, `_v2` | 내부 validation | v3 이전에는 없음 | 내부 validation 도입, 후보 합집합 수정 |
| v3 | `results/hybrid_joint_8020_v3` | 내부 validation (seed 42) | **최초 열람**, Recall 목표 미달 | 후보 수 상한 확장 |
| v4–v5 | `results/hybrid_joint_8020_v4`, `_v5` | 내부 validation (3 seed 평균) | 같은 시험 구간을 **다시 열람** | backbone 3개와 독립 GNN을 v3 시험을 본 뒤 사람이 정함. v3 학습 산출물 재사용 |
| **v6** | `audit_runs/v6` (요약은 `audit/results_v6`) | 내부 validation (seed 42) | fold마다 봉인 평가 1회, F1 사후 분석 2회 (아래 로그) | 사전등록 plan, 원본부터 일괄 실행 |
| **v7** | `audit_runs/v7` (요약은 `audit/results_v7`) | 내부 validation (seed 42) | F4 봉인 평가 1회, 사후 분석 없음 | I-23 확인용. 모든 모델군 grid 확장, 처음 쓰는 구간 F4 한 fold. 사용자 승인 (`v7_feasibility.md`) |

v1–v5의 수치와 코드는 수정하거나 삭제하지 않았다. 과거 소스는 `source_*` 폴더에 그대로 있다.

## 2. v6 실행 식별자

| 항목 | 값 |
|---|---|
| plan SHA256 | `c56ada95ecce…` (`audit/results_v6/plan.json`) |
| 코드 commit (plan 고정 시점) | `29ba891` (`dev`, 작업 트리 깨끗함) |
| 실행 commit (시험 열람 시점) | `074443e` (문서만 추가됨. 코드 해시는 plan과 동일함을 실행 시 검사) |
| plan 코드와 git | plan의 코드 해시 25개가 모두 `29ba891`의 git 내용(CRLF 형태)과 일치 (`audit/plan_code/plan_code_v6.json`, I-38) |
| 원본 입력 | `events.csv` 외 3개. SHA256은 `plan.json` → `raw_input_sha256` |
| 환경 | Python 3.11.9, numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, sklearn 1.9.0, torch 2.15.0.dev20260921+cu132, implicit 0.7.3, CUDA 13.2, RTX 5060 Ti |
| 실행 시간 | plan 생성 2026-10-20 07:51:28 UTC (`plan.json`의 `created_utc`), 중단 후 재개. 커밋된 마지막 기록은 F3 sealed_test 열람 2026-10-21 05:08:32 UTC (`test_access_log.jsonl`). 완료 시각과 종료 코드는 v6 PC의 실행 로그에만 있어 저장소로는 확인할 수 없음 |

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

- 위 시각과 `launch_v6.py`는 v6를 실행한 PC의 기록이다 (gitignore 대상인 `audit_runs/`, 저장소로 확인 불가). 커밋 `074443e`와 커밋된 접근 로그·결과로는 시험 열람 전 중단과 최종 완료만 확인된다.
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

`audit/results_v7/`에는 28개 파일과 `SHA256SUMS.txt`가 있다. v6 구성에 더해 Hybrid validation grid CSV 6개(약 3.2 MB)를 넣어, Hybrid 선택이 validation 최대값인지 저장소만으로 확인할 수 있게 했다.

그림과 표(`audit/figures_v6`, `audit/figures_v7`)는 `report_tools/plot_rolling_results.py`가 위 요약 CSV만 읽어 만든다. 입력 파일의 SHA256은 각 폴더의 `figure_manifest.json`에 있다.

## 6. v7 실행 식별자

| 항목 | 값 |
|---|---|
| plan SHA256 | `784d0fa97912…` (`audit/results_v7/plan.json`) |
| 코드 commit (plan 고정 시점) | `1170ea5` (`dev`, 작업 트리 깨끗함) |
| 실행 commit (시험 열람 시점) | `868b714` (감사 문서·`report_tools`만 추가됨. 실행 후 코드 해시가 plan과 같음을 확인) |
| plan 코드와 git | plan의 코드 해시 26개가 모두 `1170ea5`의 git 내용과 일치 (`audit/plan_code/plan_code_v7.json`, I-38) |
| 환경 | Python 3.11.9, numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, sklearn 1.9.0, torch 2.11.0+cu128, implicit 0.7.3, CUDA 12.8, RTX 3090 Ti (v6와 다른 PC) |
| 실행 시간 | 2026-10-21 11:16 UTC plan 생성, 13:10 UTC 완료 (`progress.jsonl` "completed") |
| 시험 접근 | `F4_unseen` `sealed_test` 1회 (13:09:41 UTC), selection 고정(13:04:26 UTC) 이후 |

### v7 중단과 재실행

| 시각 (KST) | 사건 | 처리 |
|---|---|---|
| 10-21 20:17 | plan 생성(20:16:46, 커밋된 plan의 `created_utc`) 직후 (실행 시작 시각은 이 PC 기록뿐, 저장소로 확인 불가), 첫 실행을 `DETACHED_PROCESS`로 띄웠더니 전처리 중 Intel Fortran 런타임이 `forrtl: error (200): program aborting due to window-CLOSE event`로 중단 (I-31) | 미완성 전처리 폴더를 `F4_unseen/prepared_incomplete_run1`, 로그를 `audit_runs/v7_run1_forrtl_abort.log`로 보존. `CREATE_NO_WINDOW` 등으로 다시 실행 |

- 중단은 전처리 단계였다. 학습, 점수 계산, 시험 열람은 없었다.
- 두 번째 실행은 첫 실행이 만든 `plan.json`(같은 해시)을 그대로 쓰고, 전처리를 처음부터 다시 해(20:18:43 완료. 커밋된 `audit/results_v7/progress.jsonl`의 `prepared` 11:18:43Z. 첫 실행의 중단 로그와 미완성 폴더는 이 PC의 `audit_runs/`에만 있음) 끝까지 완료됐다.

## 7. 감사 보조 실행 (바깥 시험 지표를 계산하지 않음)

- 결정성 검사는 평가 사용자 목록만 쓰고 정답으로 지표를 계산하지 않는다. 비교하는 것은 임베딩 차이와 Top-50 목록 일치도뿐이다.
- 경계 탐색은 내부 validation 분할만 읽는다.

| 실행 | 스크립트 | 출력 |
|---|---|---|
| 결정성 재학습 | `report_tools/determinism_check.py` | `audit/results_v6/determinism_mf_blas1.csv` (첫 실행은 BLAS 24스레드 결함 때문에 `determinism_run1_blas24.csv`로 보존) |
| MF grid 경계 탐색 (validation만 사용) | `report_tools/mf_boundary_probe.py` | `audit/results_v6/mf_boundary_probe.csv` (첫 실행은 같은 결함 때문에 `mf_boundary_probe_run1_blas24.csv`로 보존) |
| 다른 PC 교차 재현 (v6 F3 전처리, 누수 검사, MF validation grid) | `report_tools/cross_machine_repro.py` | `audit/cross_machine/cross_machine_F3_early.json` |
| CPU와 CUDA 학습 비교 (F3 내부 validation) | `report_tools/device_check.py` | `audit/cross_machine/device_check_F3_early.json` |
| 변이 테스트 (사본에서 실행) | `report_tools/mutation_check.py` | `audit/mutation/mutation_results.json` |
| ALS/BPR 점수 정밀도 비교 (v7 F4 내부 validation, Cycle 4) | `report_tools/check_score_precision.py` | `audit/cycle4/score_precision_F4_val.json` |
| 누수 검사를 원본 전체 + `--end-fraction 0.36`으로 실행 (v7 F4, Cycle 4. 누수 검사이며 지표 계산 아님) | `audit_tools/leakage_check.py` | `audit/cycle4/leakage_F4_fullraw_end036.json` (커밋된 F4 결과와 28개 동일), `leakage_F4_fullraw_without_end.json` (옵션 없이 실행하면 12개 거짓 실패) |
| v7 F4 seed 42 MF 재학습, BLAS 1 / 24스레드 (Cycle 5. 평가 사용자 목록만 쓰고 지표는 계산하지 않음) | `report_tools/determinism_check.py --skip-gnn --blas-threads` | `audit/cycle5/determinism_v7_blas1.csv`, `determinism_v7_blas24.csv` |
| ALS/BPR 점수 정밀도 비교 재실행, 고친 import 순서(1스레드) (Cycle 5) | `report_tools/check_score_precision.py` | `audit/cycle5/score_precision_F4_val_blas1.json` (Cycle 4 파일과 완전히 같음) |
| v7 root의 캐시 설정 일치 확인 (Cycle 5. 학습·정답 읽기 없음) | `report_tools/check_cached_runs.py` | `audit/cycle5/cache_identity_v7.json` |
| 가드 변이 테스트 (사본에서 실행, Cycle 5) | `report_tools/guard_mutation_check.py` | `audit/cycle5/guard_mutants.json` |
| 변이 테스트 재실행 (Cycle 4 / Cycle 5 코드) | `report_tools/mutation_check.py` | `audit/cycle4/mutation_results_cycle4.json`, `audit/cycle5/mutation_results_cycle5.json` |

다음 실행은 이미 열린 시험 결과를 다시 계산만 하며, 새 시험 열람이 아니다 (모델을 학습하거나 선택하지 않음).

| 실행 | 스크립트 | 출력 |
|---|---|---|
| v7 시험 지표 독립 재계산 (저장된 추천 목록 + 원본 `events.csv`) | `report_tools/recompute_test_metrics.py` | `audit/recompute/recompute_v7_F4_unseen.json` |
