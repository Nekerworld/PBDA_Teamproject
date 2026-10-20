# Issue register (paper_revision_codex 독립 감사)

등급: P0 = 주장 자체를 무효화할 수 있음, P1 = 결론을 바꿀 수 있음, P2 = 재현성·해석 위험, P3 = 경미.
상태: `해결·검증` / `수정·검증 대기(실험 필요)` / `환경 제약` / `원고 수정 필요` / `미해결`.
"검증"은 이 PC에서 실제로 실행한 테스트·스크립트만 가리킨다. 결과 JSON을 읽은 것만으로는 검증으로 치지 않는다.

> **현재 상태 (2026-10-20 17:35 KST)**: v6 실제 데이터 실행은 사용자 요청으로 F1 내부 validation 단계에서 중단됨.
> - 완료: MF grid 선택, GNN validation 4/8.
> - 시험 데이터: 열리지 않음 (`test_access_log.jsonl` 없음).
> - 아래 표에서 "v6 결과로 검증"이라고 적은 항목은 **코드 수정과 회귀 테스트만 끝났고, 실제 데이터 결과는 아직 없음**.
> - 같은 명령으로 다시 실행하면 이어서 진행됨 (plan과 코드 해시가 같을 때).

| ID | 등급 | 문제 | 상태 | 커밋 |
|---|---|---|---|---|
| I-01 | P0 | v3 시험 실패 후 v4/v5 설계 변경 (시험 재사용) | 수정 → v6 실행 결과로 검증 | 29ba891, 63145d7 |
| I-02 | P0 | 데이터 누수 (원래 validation→training, 기타 경로) | 해결·검증 (누수 없음 확인) | 46fe0e0 |
| I-03 | P1 | 단순 baseline 부재 (Recent, Popularity) | 수정 → v6 결과로 검증 | 29ba891 |
| I-04 | P1 | 튜닝 예산 비대칭 (ALS/BPR 고정, Hybrid 576×3) | 수정 → v6 결과로 검증 | 0af9a3c, 29ba891 |
| I-05 | P1 | GNN epoch가 상한에 걸림 (190/200) | 수정 → v6 결과로 검증 | 0af9a3c, 29ba891 |
| I-06 | P1 | 통계: 다중비교 보정 없음, Hybrid vs GNN Recall 비유의 | 수정 → v6 결과로 검증 | 29ba891 |
| I-07 | P1 | 반복 추천 허용 평가만 존재 | 수정 → v6 결과로 검증 | 29ba891 |
| I-08 | P1 | 모델군별 선택 기준 불일치 | 수정 → v6 결과로 검증 | 0af9a3c, 29ba891 |
| I-09 | P1 | 후보 수가 grid 상한(20,000 = 카탈로그 34%), 효율성 주장 근거 없음 | 수정 → v6 사후 분석 | 29ba891 |
| I-10 | P2 | `robust_hybrid.py` backbone 하드코딩 | 해결·검증 | 63145d7 |
| I-11 | P2 | `load_state_dict(strict=False)`가 키 불일치를 숨김 | 해결·검증 | 0af9a3c |
| I-12 | P2 | v5가 v3 학습 산출물 재사용, 원본부터의 일괄 재실행 없음 | 수정 → v6 실행 | 29ba891 |
| I-13 | P2 | 지표 CSV가 gitignore, 결과 JSON에 다른 PC 절대경로 | 부분 해결 (audit/ 요약 추적) | 본 문서 커밋 |
| I-14 | P2 | IF 제거 근거(ablation) 부재 | 수정 → v6 F1 사후 분석 | 29ba891 |
| I-15 | P2 | gamma·재고 효과 근거 부족 | 수정 → v6 F1 사후 분석 | 29ba891 |
| I-16 | P2 | seed가 초기화만 바꿈 (분할 고정) | 수정 (3개 시간 fold) | 29ba891 |
| I-17 | P2 | 라이브러리 버전 차이로 `if_feature_distributions.csv` 해시 불일치 | 환경 제약 (원인 추정만 가능) | - |
| I-18 | P2 | novel 정책에서 사용자가 카탈로그 전체를 본 경우 후보 0개 → 예외 | 해결·검증 (커밋 전 발견) | 29ba891 |
| I-19 | P3 | CUDA sparse 연산 결정성 미보장 | 검증 예정 (같은 seed 2회) | - |
| I-20 | P3 | v6 `selection.json`의 `job_dir`가 절대경로 | 미해결 (실행 중 코드 변경 불가) | - |
| I-21 | P3 | v5 성공 기준이 "평균이 모든 baseline보다 높음" (유의성 아님) | 원고 수정 필요 | - |
| I-22 | P3 | 원고 파일이 저장소에 없음 | 원고 수정 필요 (대조표 제공) | - |

---

## I-01 (P0) 시험 데이터 재사용

- **위치**: `robust_hybrid.py` 50행 (수정 전), `results/hybrid_joint_8020_v5/experiment_plan.json` (`historical_test_seen: true`, `prior_test_success: false`).
- **재현**: v3 `result.json`의 `success: false`를 확인한 다음 v4/v5 계획이 작성됨. v5 backbone(`l2_uniform`, `l3_uniform`, `l2_mf_mixed`)은 v3 validation 상위 3개와 다름.
- **근본 원인**: 수치 선택은 validation으로 했지만, 탐색 공간(후보 수 최대 20,000, backbone, 독립 GNN 비교 대상)은 시험 결과를 본 뒤 사람이 정함.
- **영향**: v5의 시험 결과는 selection bias가 있어 일반화 성능 추정치로 볼 수 없음.
- **수정**: `rolling_protocol.py`(v6). 모든 grid·fold·seed·선택 규칙을 실행 전에 `plan.json`에 고정하고 해시로 잠금. fold마다 시험을 한 번만 열고 `test_access_log.jsonl`에 기록. plan이 바뀌면 같은 fold를 다시 열 수 없음. F1은 v1–v5 시험 구간 그대로(이미 노출됐다고 명시), F2/F3는 처음 쓰는 시험 구간.
- **회귀 테스트**: `test_access_log_blocks_reopening_under_new_plan`, `test_rolling_protocol_end_to_end`(재실행해도 시험 접근이 추가되지 않음, plan 변경 시 거부), `test_plan_freezes_grids_and_rules`.
- **기록 보존**: v5 결과와 코드는 그대로 둠. `robust_hybrid.py`에 `BACKBONE_PROVENANCE`를 추가해 시험을 본 뒤 정한 선택임을 기록.

## I-02 (P0) 데이터 누수

- **검증 방법**: `audit_tools/leakage_check.py`가 원본 CSV에서 독립적으로 다시 계산해 전처리 결과와 비교. 검사 항목은 cutoff, core 고정점, 평가 사용자, 정답, 속성 시점, `available`, 카탈로그, 내부 validation 구간, GNN 간선.
- **결과**: 80:20 no-IF 분할에서 28개 항목 모두 통과 (`audit_runs/repro_8020`). v6의 모든 fold도 실행 시작 시 같은 검사를 통과해야 진행되며, F1은 통과함.
- **결함 검출 확인**: 시험 이벤트 1건을 학습 파일에 넣으면 검사가 실패함 (`test_leakage_check_passes_and_detects_injected_leak`).

## I-03 (P1) 단순 baseline

- `policy_evaluation.EvaluationContext.popularity`: 학습 이벤트 수 기준. 동점은 작은 item ID 우선.
- `recent`: 학습 기간에 마지막으로 접촉한 시각이 늦은 순서. 상품이 50개 미만인 사용자는 **짧은 목록을 그대로 둠** (채우지 않음).
- `RecentPop`: Recent 뒤에 아직 없는 인기 상품을 채움. 두 규칙을 모두 보고.
- novel 정책에서 Recent는 정의상 빈 목록이 되고, RecentPop은 인기 상품 채움만 남음.
- 테스트: `test_recent_and_popularity_baselines` (50개 미만 사용자 처리 규칙 포함).

## I-04 (P1) 튜닝 예산

- ALS 24개 (factors {64,128,256} × reg {.02,.2} × α {2,10} × iter {32,256}).
- BPR 4개 설정 × epoch 체크포인트 5개 (16–256). lr {1e-3, 5e-3}, reg {.003, .03}.
- GNN 8개 (layers {2,3} × dim {64,128} × lr {1e-3, 3e-3}). 각각 최대 400 epoch, 10 epoch마다 validation.
- Hybrid는 추론 단계 파라미터 2,808개 × backbone 3개이며, 학습은 추가로 하지 않음.
- `selection.json`의 `budget`에 학습 횟수, 평가한 설정 수, 학습 시간을 따로 기록함.
- 위 설정 수는 모두 아래 회귀 테스트로 고정되어 있음.
- 남은 비대칭: Hybrid는 추론 파라미터 조합 수가 많아 validation 과적합 위험이 더 큼. 따라서 시험 결과는 fold 3개와 통계 검정으로 해석함.

## I-05 (P1) GNN epoch 상한

- 상한을 200에서 400으로 늘림.
- 상한 규칙: 선택된 epoch가 상한에서 validation 간격 1회 이내(≥ 상한 − 10)이면 800 epoch로 다시 학습해 대체함.
- v5의 190/200 패턴이 이 규칙에 걸리는지 `test_gnn_cap_rule_flags_v5_pattern`으로 확인함.
- 실제로 상한에 걸렸는지와 연장 여부는 `validation_gnn.json`의 `at_cap`, `extended`로 보고함.

## I-06 (P1) 통계

- `stats_tools.compare_family`: 사용자 단위 paired bootstrap 95% CI (10,000회), 양측 sign-flip 검정 (10,000회).
- Holm 보정은 Hybrid 대 baseline 7개 × 지표 2개, 정책·fold별로 적용.
- seed 표준편차와 hit 사용자 수(`hit_users`, `hit_users_any_seed`)는 따로 보고.
- 테스트: `test_holm_and_paired_tests`.

## I-07 (P1) 반복 추천과 신규 상품 평가

- `repeat`: 원래 프로토콜. 학습 기간에 접촉한 상품도 추천과 정답에 포함.
- `novel`: 정답에서 원본(필터 전) 학습 기간에 접촉한 상품을 제외. 정답이 비는 사용자는 모집단에서 빠짐. 학습 카탈로그 중 이미 본 상품은 추천 목록에 넣지 않음.
- 두 정책은 별도 표로 보고하며 섞지 않음.
- 테스트: `test_novel_policy_definitions`. 이미 본 상품이 목록에 들어가면 예외가 나는지도 확인함.

## I-08 (P1) 선택 기준 통일

- 모든 모델군이 내부 validation의 mean(NDCG@50, Recall@50) 하나로 선택됨.
- GNN 내부 epoch 선택에도 같은 기준을 씀 (`validation_metric='ndcg_recall_mean'`).
- 기존 기본값(NDCG)은 바꾸지 않음 (`test_gnn_validation_metric_option`).

## I-09 (P1) 후보 수와 비용

- validation grid에 후보 수 {500, …, 20000, 'all'}을 포함해 상한 문제를 없앰.
- 사후 분석(F1, seed 42, 선택에는 사용하지 않음): 후보 수 9개 각각에 대해 다음을 측정.
  - 지표, 후보 recall
  - 단계별 ms/user
  - Python 최대 메모리
  - 전체 카탈로그 GNN 점수 계산 시간
- 결과 파일: `posthoc_candidate_depth.csv`.

## I-10, I-11 (P2) 하드코딩과 strict=False

- `robust_hybrid.py`는 `--backbone NAME:CHECKPOINT`, `--standalone-gnn` 인자를 받음. 기본값은 v5의 과거 목록.
- `model.load_without_adjacency`는 고정 그래프인 `adjacency` 외에 키가 빠지거나 남으면 오류를 냄 (`test_checkpoint_loading_only_tolerates_missing_adjacency`).

## I-12 (P2) 일괄 재실행

v6는 원본 CSV에서 시작함. 학습 산출물을 재사용하지 않고 전처리, 누수 검사, 선택, 재학습, 시험 평가를 한 번에 실행함. `plan.json`에 원본 해시, 코드 해시, git commit, 환경을 기록함.

## I-13 (P2) 추적 가능한 결과

`audit_runs/`는 gitignore 대상 (모델·임베딩·분할 수 GB). 검증된 요약 표만 `audit/results_v6/`로 복사하고, `.gitignore` 예외로 추적함. 각 파일의 SHA256은 `audit/experiment_provenance.md`에 기록.

## I-17 (P2) 라이브러리 버전 차이

- 원래 환경: Python 3.12.3, numpy 2.1.3, pandas 2.3.0, sklearn 1.6.1.
- 현재 환경: Python 3.11.9, numpy 2.4.6, pandas 3.0.6, sklearn 1.9.0.
- 핵심 분할 파일은 바이트 단위로 동일함. 해시가 다른 것은 감사용 요약 `if_feature_distributions.csv` 하나뿐.
- 원본 파일이 이 PC에 없어 내용 비교는 불가능함. 원인은 pandas `describe` 출력 형식 차이로 추정하며, 확인되지 않음.

## I-18 (P2) novel 정책 경계 사례

통합 smoke 테스트에서 발견함: 카탈로그의 모든 상품을 이미 본 사용자는 후보가 0개가 되어 `minmax`가 예외를 냄. `hybrid_grid`와 `top_k(k=0)`를 빈 목록으로 처리하도록 고침. 이후 `test_rolling_protocol_end_to_end`가 통과함.

## I-19 (P3) GPU 결정성

`set_global_seed`가 cuDNN deterministic을 설정하지만, sparse CUDA 행렬곱의 결정성은 보장되지 않음. v6 실행 뒤 같은 설정과 seed로 GNN을 두 번 학습해 임베딩 차이를 측정함 (`validation_report.md`).

## I-20 (P3) 절대경로

v6 `selection.json`과 `validation_gnn.json`의 `job_dir`는 실행 PC의 절대경로임. `refits.json`은 상대경로를 씀. 실행 중에 코드를 바꾸면 코드 해시 잠금이 깨지므로 이번 실행에서는 고치지 않음.

## I-21, I-22 (P3) 원고

원고 파일이 저장소에 없음. `audit/manuscript_alignment.md`에 원고에서 고쳐야 할 주장과 그 근거 파일을 대조표로 정리함.
