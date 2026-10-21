# Issue register (paper_revision_codex 독립 감사)

등급: P0 = 주장 자체를 무효화할 수 있음, P1 = 결론을 바꿀 수 있음, P2 = 재현성·해석 위험, P3 = 경미.
상태: `해결·검증` / `수정·검증 대기(실험 필요)` / `환경 제약` / `원고 수정 필요` / `미해결`.
"검증"은 이 PC에서 실제로 실행한 테스트·스크립트만 가리킨다. 결과 JSON을 읽은 것만으로는 검증으로 치지 않는다.

> **현재 상태 (2026-10-22 KST)**: v6 실제 데이터 실행이 3 fold 모두 완료됨 (exit 0). I-23 확인을 위한 v7(처음 쓰는 구간 F4, 모든 모델군 grid 확장)도 완료됨. Cycle 4 재감사 항목(I-39 – I-47) 처리 완료.
> - 시험 열람: v6는 fold마다 봉인 평가 1회, F1에서만 사후 분석 2회 추가 (`audit/results_v6/test_access_log.jsonl`). v7은 F4 봉인 평가 1회 (`audit/results_v7/test_access_log.jsonl`).
> - 결과와 검증: `validation_report.md` (v7은 §12).
> - Cycle 2 재감사에서 I-23 – I-26, Cycle 2 커밋 후 검증(418e7dd)에서 I-27, Cycle 3에서 I-28 – I-37, Cycle 3 커밋 후 검증에서 I-38, Cycle 4 재감사에서 I-39 이후를 찾음.

| ID | 등급 | 문제 | 상태 | 커밋 |
|---|---|---|---|---|
| I-01 | P0 | v3 시험 실패 후 v4/v5 설계 변경 (시험 재사용) | 해결·검증: 사전등록 v6 완료, fold마다 봉인 평가 1회 (사후 분석은 선택에 쓰지 않음). F1 구간이 과거에 노출된 사실은 되돌릴 수 없어 공개 사항으로 남김 | 29ba891, 63145d7 |
| I-02 | P0 | 데이터 누수 (원래 validation→training, 기타 경로) | 해결·검증: v6 3 fold와 v7 F4 모두 28/28 검사 통과 | 46fe0e0 |
| I-03 | P1 | 단순 baseline 부재 (Recent, Popularity) | 해결·검증: v6 결과에 포함 | 29ba891 |
| I-04 | P1 | 튜닝 예산 비대칭 (ALS/BPR 고정, Hybrid 576×3) | 해결·검증 (한계 공개): v6에서 MF가 경계에서 선택됨 → I-23. v7에서 ALS 192, BPR 24(168 설정), GNN 36회 학습으로 넓혀 확인 | 0af9a3c, 29ba891, c9d3e49 |
| I-05 | P1 | GNN epoch가 상한에 걸림 (190/200) | 해결·검증: v6 validation 학습 24회 모두 50–290/400 epoch에서 멈춤. v7은 36개 중 5개가 상한 규칙(최고 epoch ≥ 390)에 걸렸지만(400 epoch 4개, 390 epoch 1개) 모두 validation 6위 이하(낮은 lr 또는 64차원)였고, 선택·Hybrid backbone 상위 3개는 90–140 epoch. 사전등록 규칙상 연장 대상 아님 | 0af9a3c, 29ba891 |
| I-06 | P1 | 통계: 다중비교 보정 없음, Hybrid vs GNN Recall 비유의 | 해결·검증: Holm 보정 결과 보고 | 29ba891 |
| I-07 | P1 | 반복 추천 허용 평가만 존재 | 해결·검증: novel 정책을 분리 보고. 원본 데이터로 독립 검사 0건 위반 | 29ba891 |
| I-08 | P1 | 모델군별 선택 기준 불일치 | 해결·검증 | 0af9a3c, 29ba891 |
| I-09 | P1 | 후보 수가 grid 상한(20,000 = 카탈로그 34%), 효율성 주장 근거 없음 | 해결·검증: 사후 분석 완료. 효율성 주장은 근거 없음 (원고 수정 필요) | 29ba891 |
| I-10 | P2 | `robust_hybrid.py` backbone 하드코딩 | 해결·검증 | 63145d7 |
| I-11 | P2 | `load_state_dict(strict=False)`가 키 불일치를 숨김 | 해결·검증 | 0af9a3c |
| I-12 | P2 | v5가 v3 학습 산출물 재사용, 원본부터의 일괄 재실행 없음 | 해결·검증: v6는 원본부터 일괄 실행 | 29ba891 |
| I-13 | P2 | 지표 CSV가 gitignore, 결과 JSON에 다른 PC 절대경로 | 해결: `audit/results_v6` 요약 + SHA256 | Cycle 2 커밋 |
| I-14 | P2 | IF 제거 근거(ablation) 부재 | 해결·검증: 구매 100% 제거, 모든 모델 성능 붕괴 | 29ba891 |
| I-15 | P2 | gamma·재고 효과 근거 부족 | 해결·검증: 효과 작음 (원고 수정 필요) | 29ba891 |
| I-16 | P2 | seed가 초기화만 바꿈 (분할 고정) | 해결·검증: 시간 fold 3개 | 29ba891 |
| I-17 | P2 | 라이브러리 버전 차이로 `if_feature_distributions.csv` 해시 불일치 | 환경 제약 (원인 추정만 가능) | - |
| I-18 | P2 | novel 정책에서 사용자가 카탈로그 전체를 본 경우 후보 0개 → 예외 | 해결·검증 (커밋 전 발견) | 29ba891 |
| I-19 | P3 | CUDA sparse 연산 결정성 미보장 | 검증함: GNN 임베딩 차이 1.1e-4, 목록 99.7% 동일. ALS/BPR은 비트 단위 동일 | Cycle 2 커밋 |
| I-20 | P3 | v6 `selection.json`의 `job_dir`가 절대경로 | 미해결 (봉인된 산출물이라 고치지 않음. 다른 PC에서는 `refits.json`의 상대경로를 씀) | - |
| I-21 | P3 | v5 성공 기준이 "평균이 모든 baseline보다 높음" (유의성 아님) | 원고 수정 필요 (`manuscript_alignment.md`) | - |
| I-22 | P3 | 원고 파일이 저장소에 없음 | 원고 수정 필요 (대조표 제공) | - |
| I-23 | P1 | v6의 ALS/BPR이 grid 경계에서 선택됨. 경계 밖에서 ALS validation이 +0.011 – +0.027 | 해결·검증 (한계 공개): 사용자 승인으로 처음 쓰는 구간 F4에서 모든 모델군 grid를 넓힌 v7 실행. ALS가 validation에서 GNN을 앞섰지만(0.2176 vs 0.2169) 시험에서 Hybrid − ALS+BPR은 4칸 모두 비유의 (v6는 NDCG 6칸 비유의, Recall 3칸 유의). "가장 강한 baseline보다 일관되게 유의하게 높지 않다"는 결론 유지. v6 F1–F3 수치는 덜 튜닝된 조건으로 남고, v7에서도 일부 축이 끝에 걸림(사전등록 규칙상 재확장 안 함) | c9d3e49, dea0f20 |
| I-24 | P3 | ALS 결과가 BLAS 스레드 수에 따라 달라짐 | 문서화: v6는 1스레드로 고정. v6·v7 시점 코드는 다른 환경변수가 이미 있으면 그 값이 우선했음 → Cycle 4에서 강제 고정 (I-41) | Cycle 2 커밋 |
| I-25 | P2 | `determinism_check.py`가 numpy를 import한 뒤 스레드 환경변수를 설정함 (거짓 비결정성 보고) | 해결·검증 | Cycle 2 커밋 |
| I-26 | P2 | Hybrid 후보 수가 fold마다 다름 (20000 / 500 / all) | 해석 사항: validation 곡선이 평평함. 원고에서 후보 수를 핵심 설계로 주장하지 않음 | - |
| I-27 | P2 | 커밋된 결과 파일이 `SHA256SUMS.txt`와 불일치 (줄바꿈 자동 변환) | 해결·검증 (커밋 후 검증에서 발견). 새 clone(다른 PC)에서도 60/60 일치 | 418e7dd |
| I-28 | P3 | 인자 없는 `python -m pytest`가 보관용 `source_*` 폴더의 옛 테스트까지 수집해 오류 16개 | 해결·검증: `pytest.ini`로 수집 범위 제한. 다른 PC의 깨끗한 venv에서 70개 통과 | 1170ea5 |
| I-29 | P3 | v7 "미사용 구간" 테스트가 비율만 비교함 (실제 시각 미확인) | 해결·검증: 원본 타임스탬프로 F4 끝 ≤ v6 내부 validation 시작을 확인하는 테스트 추가. 실제 경계는 같은 타임스탬프(겹침 0건). end 0.37 변이를 잡음 | 1170ea5 |
| I-30 | P2 | 평가·선택·통계의 핵심 성질을 고정하는 테스트 부재 (변이 테스트에서 발견) | 해결·검증: 구현은 올바름(정독 + 실제 데이터 대조). 기존 테스트는 변이 17개 중 11개를 놓침. `test_cycle3_consistency.py`(9개) 추가 후 두 묶음 합쳐 17/17. M04는 합성 데이터에서 필터 전후 이력이 같아 구별되지 않던 것을 전용 데이터로 고정함. v7 실행 중 코드 해시 잠금 때문에 v7 종료 후 커밋 | 70f5106 |
| I-31 | P3 | 분리 실행(`DETACHED_PROCESS`) 시 Intel Fortran 런타임이 창 닫기 이벤트로 프로세스를 중단 (`forrtl: error (200)`) | 해결: `CREATE_NO_WINDOW` + `FOR_DISABLE_CONSOLE_CTRL_HANDLER=1`로 재실행. 중단은 v7 전처리 중이었고 시험 열람·학습 없음. 미완성 폴더는 `prepared_incomplete_run1`로 보존 | 문서 |
| I-32 | P2 | 원고용 그림이 v5 단일 분할 결과 기준 (`professor_materials/figures`) | 해결·검증: 커밋된 v6 CSV만 읽는 그림·표 생성기와 결과(`audit/figures_v6`). 재실행 시 바이트 동일 | a2c5445 |
| I-33 | P3 | BPR(GPU) 결과가 GPU·torch 빌드에 따라 달라짐 | 문서화: 다른 PC에서 F3 MF grid 재실행. ALS 24행은 비트 단위 동일, BPR 20행은 선택 점수 최대 0.0047 차이, 선택 결과는 같음 | e7f0240 |
| I-34 | P3 | 같은 seed라도 CPU와 GPU의 학습 결과가 다름 | 문서화: F3 내부 validation에서 GNN 목록 겹침 84%, BPR 28%, validation 점수 차이 0.0024 / 0.0013 (seed 변동 범위). 모든 v6/v7 결과는 CUDA 기준 | 868b714 |
| I-35 | P3 | 그림 생성기 제목에 v6 전용 문구("F1 window was exposed in v1-v5", seed 목록)가 고정돼 v7 그림에도 찍힘. 범례가 막대를 가림 | 해결·검증: 제목을 데이터(fold 이름, `test_metrics.csv`의 seed)에서 만들고 범례를 그림 아래로 옮김. v6/v7 그림을 다시 만들어 두 번 실행 시 바이트 동일 확인 | dea0f20 |
| I-36 | P2 | `requirements.txt`가 scikit-learn 1.9.1을 고정하지만 v6·v7은 1.9.0으로 실행됨. v6·v7 실행에 쓴 분리 실행 스크립트도 저장소에 없음 (`audit_runs/`는 gitignore) | 해결·검증: v7 venv를 그대로 적은 `requirements-v7-lock.txt` 추가. 새 venv에 설치해 패키지 목록이 v7 venv와 완전히 같음을 확인. `report_tools/launch_detached.py` 추가 후 실제 실행으로 동작 확인. `requirements.txt` 자체는 기존 문서의 기준이라 바꾸지 않음 | e9538c8 |
| I-37 | P2 | codex 폴더 첫 문서 `README.md`가 v5 주장("두 지표의 세 seed 평균이 모든 비교 모델보다 높았다")을 감사 결과 안내 없이 제시. `README_HYBRID_KO.md`, `REPRODUCE_KO.md`도 v5 경로만 안내. `professor_materials/README_KO.md`(같은 주장과 "NDCG +10.83%"), 저장소 최상위 `README.md`(v5 재실행을 "최종 Hybrid 실험"으로 안내)도 같음 | 해결: 원문은 이력으로 보존하고, 맨 앞에 시험 재사용 사실과 감사 문서(검증 결과, 재현 절차, 원고 대조표) 안내를 추가 | e9538c8 |
| I-38 | P2 | plan의 코드 해시(`code_hashes`)가 작업 폴더의 원시 바이트를 해시해 checkout의 줄바꿈 변환(`core.autocrlf`)에 따라 달라짐. 같은 commit이라도 다른 PC에서는 plan 검사가 거짓 실패하고, "코드 변경"과 "줄바꿈 차이"를 구별할 수 없음 (v7 plan은 CRLF 19개·LF 7개가 섞여 있었고, 이 PC에서 v6 plan과 비교하면 내용이 같은 파일 7개가 불일치) | 해결·검증: 새 plan은 줄바꿈을 정규화한 해시(`sha256-lf`)와 방식을 기록함. 방식 기록이 없는 v6/v7 plan은 원시 바이트로 비교해 호환 유지. `report_tools/verify_plan_code.py`로 v6 25개, v7 26개 해시가 모두 각 plan commit의 git 내용과 일치함을 확인 (조작한 해시 1개는 검출). 테스트 3개 추가 | b575dac |
| I-39 | P2 | 재개 시 누수 검사 게이트가 `leakage_check.json`의 존재만 확인함. 이 파일은 검사가 실패해도 쓰이므로, 실패한 fold를 재개하면 검사를 건너뛰고 진행됨 | 해결·검증: `leakage_gate`가 저장된 `all_pass`를 읽어 거짓이면 멈춤. 테스트 2개, 변이(옛 동작) 검출. v6 3 fold·v7 F4는 모두 `all_pass: true`라 결과 영향 없음 | 05b9c65 |
| I-40 | P2 | 시험 열람 가드가 같은 root 안에서 plan 해시만 비교함. `selection.json`을 다시 만들어 같은 plan으로 시험을 다시 열거나, 새 root에서 바뀐 plan으로 같은 구간을 여는 것을 막지 못함 (docstring은 "fold마다 한 번"이라고 설명) | 해결·검증: 같은 root에서 `selection_sha256`이 다르면 거부. 실행 시작 시 커밋된 `audit/results_*`의 plan·접근 로그를 읽어 같은 원본 해시·같은 구간(`end_fraction`, `test_size`)을 다른 plan으로 연 기록이 있으면 거부 (같은 plan의 재현은 허용). 실제 커밋 기록으로 F1–F4가 새 plan에 막히고 v6·v7 자신은 막히지 않음을 테스트. 커밋되지 않은 root는 여전히 볼 수 없음(문서화). v6·v7 접근 로그는 fold마다 sealed_test 1회라 결과 영향 없음 | 05b9c65 |
| I-41 | P2 | `rolling_protocol.py`가 BLAS 스레드 변수를 `setdefault`로 설정해 호출자가 준 값이 우선함. 나중에 import되는 `experiments.py`가 변수를 1로 다시 쓰지만 BLAS 로드 뒤라, 기록·확인용으로 변수를 읽으면 실제와 다를 수 있음. 스레드·장치가 plan에 기록되지 않음 | 해결·검증: 값을 강제로 1로 설정. plan `environment`에 변수, 로드된 라이브러리의 실제 스레드 수(threadpoolctl), torch 스레드, 장치 기록. 호출자가 4를 줘도 실제 BLAS 스레드가 1인지 확인하는 테스트(옛 동작 변이 검출). v7은 1 고정 launcher로 실행, v6은 §9 교차 재현이 1스레드를 뒷받침 | 05b9c65 |
| I-42 | P3 | GNN 상한 연장을 한 번만 하고 다시 정렬함. 연장 뒤 다른 상한 도달 설정이 Hybrid backbone 상위 3개에 들어오면 연장 없이 쓰일 수 있음 | 해결·검증: 상위 3개에 연장 안 된 상한 도달 설정이 없을 때까지 반복 (`extend_capped_backbones`). 테스트 2개, 변이 검출. v6·v7은 상위 3개에 상한 도달이 없어 결과 영향 없음 | 05b9c65 |
| I-43 | P3 | 재개 기준 파일 `test_metrics.csv`를 보고서 CSV보다 먼저 씀. 그 사이에 중단되면 재개 때 보고서를 만들지 않음 | 해결: `test_metrics.csv`를 마지막에 씀. 코드 정독으로 확인(전용 테스트 없음). v6·v7은 보고서가 모두 있음 | 05b9c65 |
| I-44 | P3 | `leakage_check.py`를 단독으로 원본 폴더에 대해 실행하면 rolling fold의 끝 비율을 알 수 없음 (manifest에 없음). 원본 전체로 v7 F4를 검사하면 28개 중 12개가 거짓 실패 | 해결·검증: `--end-fraction` 추가. 원본 전체 + 0.36으로 v7 F4를 검사해 커밋된 결과와 28개 모두 같음 (`audit/cycle4/`). 테스트 1개, 변이 검출 | 05b9c65 |
| I-45 | P3 | `verify_plan_code.py`가 코드가 바뀌어도 종료 코드 0을 냄 (plan 해시와 commit만 확인) | 해결·검증: 종료 코드의 뜻을 문서화하고 `--require-unchanged` 추가. 실제 v6 plan으로 0 / 1 확인하는 테스트 | 05b9c65 |
| I-46 | P3 | 선택 경로는 ALS/BPR 원 점수로, 시험 경로는 min-max 후 float32 점수로 순위를 매김. 동점 순서가 달라질 수 있음 | 측정: v7 F4 내부 validation에서 두 정책 × ALS/BPR 모두 목록 100% 동일, 지표 차이 0 (원 점수도 float32). 코드 변경 없음 (`audit/cycle4/score_precision_F4_val.json`) | 05b9c65 |
| I-47 | P2 | 감사 문서의 수치·표현 오류 24건 (v6 Hybrid − ALS+BPR Recall 3칸 유의를 "v6와 같이 모두 비유의"로 쓴 것, Recall 0 비율, 후보 수 범위, γ 정책별 값, GNN 상한 4+1개, 반올림 등). 저장소만으로 다시 확인할 수 없는 주장(v6 PC 전용 검사)이 표시 없이 있었음 | 해결·검증: 각 수치를 커밋된 CSV·JSON에서 다시 계산한 뒤 고침. 확인할 수 없는 주장은 그렇게 표시함 | e3cf44e |

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
- **결과**: 80:20 no-IF 분할에서 28개 항목 모두 통과 (`audit_runs/repro_8020`). v6의 모든 fold와 v7 F4도 실행 시작 시 같은 검사를 통과해야 진행되며, F1–F3와 F4 모두 28/28 통과함.
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

`audit_runs/`는 gitignore 대상 (모델·임베딩·분할 수 GB). 검증된 요약 표만 `audit/results_v6/`로 복사하고, `.gitignore` 예외로 추적함. 각 파일의 SHA256은 `audit/results_v6/SHA256SUMS.txt`에 기록.

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

## I-23 (P1) MF baseline의 grid 경계 선택

v6 사전등록 ALS grid는 축마다 값이 2–3개뿐임 (factors {64, 128, 256}, reg {.02, .2}, α {2, 10}, iterations {32, 256}).
- ALS는 세 fold 모두 factors 256(최대), α 10(최대), iterations 32(최소)를 고름. reg는 F1/F3 .02, F2 .2.
- BPR은 lr .005(최대), reg .003(최소), 256 epoch(최대 체크포인트)를 고름.

Cycle 2에서 `report_tools/mf_boundary_probe.py`로 **validation 구간만** 써서, 각 fold의 봉인된 설정을 경계 밖으로 한 축씩 바꿔 봄 (시험 데이터는 읽지 않음):

| fold | 봉인된 ALS | 경계 밖 최고 ALS (factors 512, α 40, iter 16) | 최고 GNN | 최고 Hybrid |
|---|---|---|---|---|
| F1 | .1459 | .1669 | .1678 | .1759 |
| F2 | .1673 | .1782 | .1912 | .1966 |
| F3 | .1818 | .2090 | .2119 | .2171 |

(값은 validation 선택 점수 = mean(NDCG@50, Recall@50), seed 42. 봉인된 설정을 다시 학습한 행은 봉인 점수를 차이 0으로 재현함.)

- ALS는 경계 밖에서 크게 좋아지고 (+0.011 – +0.027), F1/F3에서는 GNN에 가까워짐. BPR의 증가폭은 작음 (+0.0002 – +0.0044).
- 이 grid 결정은 Hybrid의 ALS+BPR 구성 요소에도 들어가므로, 확장하면 Hybrid도 바뀔 수 있음.
- 따라서 "Hybrid ≥ ALS+BPR" 비교는 baseline이 덜 튜닝된 조건일 수 있음. v6 결과에서 NDCG는 6칸 모두 비유의이고, Recall은 3칸(F1·F2 repeat, F3 novel, p_holm .010 / .024 / .010)에서 유의함. 이 3칸은 덜 튜닝된 baseline 대비 결과라 해석에 주의가 필요함.
- 고치려면 grid를 넓혀 사전등록을 다시 하고 세 fold를 다시 실행해야 함. 이는 같은 시험 구간을 두 번째로 여는 것이고, 시험 결과를 본 뒤의 재실행임. 그래서 감사자가 임의로 하지 않고 사용자 결정으로 남김.
- 산출물: `audit/results_v6/mf_boundary_probe.csv`.

**Cycle 3 처리 (v7)**: F1–F3를 다시 여는 대신, 한 번도 평가하지 않은 구간 F4(이벤트 앞 36%)에서 ALS·BPR·GNN grid를 모두 넓힌 v7을 사전등록하고 사용자 승인 후 실행함 (`v7_feasibility.md`, `validation_report.md` §12).
- ALS: factors 1024, α 80 (둘 다 새 grid의 끝), validation 0.2176. 단독 GNN 최고 0.2169보다 높음.
- 시험: Hybrid − ALS+BPR은 repeat/novel × NDCG/Recall 4칸 모두 Holm 보정 후 비유의 (+0.0013 – +0.0077). v6(NDCG 6칸 비유의, Recall 3칸 유의)와 합쳐 보면 "가장 강한 baseline보다 일관되게 유의하게 높지 않다"는 결론이 같음.
- 남은 한계: v6 표의 ALS·ALS+BPR 수치는 고칠 수 없음. v7에서도 ALS factors·α, BPR factors, GNN layers·dim·lr이 끝에 걸렸고, 사전등록 규칙대로 더 넓히지 않음. F4 한 fold, 시험 사용자 413명이라 0.01 이하의 차이는 판정할 수 없음.

## I-24 (P3) BLAS 스레드 수

`implicit` ALS는 OpenBLAS 스레드 수에 따라 부동소수점 누적 순서가 달라짐. 24스레드와 1스레드의 결과가 달랐음.

- v6·v7 시점의 `rolling_protocol.py`는 numpy import 전에 `OPENBLAS/OMP/MKL/NUMEXPR_NUM_THREADS=1`을 `setdefault`로 설정함. 사용자 환경에 이 변수들이 다른 값으로 이미 있으면 그 값이 우선함.
- v6가 1스레드였다는 근거는 다른 PC에서 1스레드로 다시 학습한 F3 ALS grid 24행이 비트 단위로 같다는 것(`validation_report.md` §9)이고, v7은 1로 고정하는 launcher로 실행함.
- Cycle 4(I-41)부터 값을 강제로 1로 설정하고 실제 스레드 수를 plan에 기록함.

## I-39 – I-47 (Cycle 4 재감사)

Cycle 4의 독립 재감사(코드·문서 각각)가 찾은 항목을 감사자가 직접 확인한 뒤 고침. P0·P1은 없었음. 상세 내용은 위 표와 `cycle_log.md` Cycle 4에 있음.
- 코드 수정(I-39 – I-45)은 모두 v6·v7 실행 이후의 변경이며, v6·v7의 선택·시험 결과를 바꾸지 않음 (항목별 근거는 표의 "결과 영향 없음" 참고).
- 새 테스트 `test_cycle4_guards.py` 12개. 옛 동작을 다시 넣은 변이 6개(G1–G6)를 모두 잡음 (처음에는 스레드 변이 G5를 놓쳐, 환경변수 대신 실제 BLAS 스레드 수를 읽도록 테스트를 고침).

## I-25 (P2) 결정성 검사 스크립트의 import 순서

`report_tools/determinism_check.py` 첫 버전은 numpy를 먼저 import한 뒤 스레드를 설정해 24스레드로 실행됨. 그 결과 ALS가 "비결정적"이라고 잘못 보고함.

- 고친 뒤 같은 검사를 다시 실행: ALS·BPR 인자 차이 0.0, 목록 100% 동일.
- 첫 실행 결과는 `determinism_run1_blas24.csv`로 이름을 바꿔 보존함.
- 재감사에서 `report_tools/mf_boundary_probe.py`에도 같은 결함이 있음을 발견함. 봉인된 ALS 설정을 다시 학습한 점수가 봉인 점수와 달랐음 (F3 −0.0013).
  - 같은 방식으로 고치고, ALS 설정을 fold별 `selection_mf.json`에서 읽게 함 (원래는 하드코딩되어 F2에서 봉인 설정이 아닌 reg .02를 썼음).
  - 다시 실행한 결과, 세 fold의 봉인 ALS·BPR 행이 모두 차이 0으로 재현됨.
  - 첫 결과는 `mf_boundary_probe_run1_blas24.csv`로 보존함.

## I-27 (P2) 결과 파일 해시와 줄바꿈 변환

Windows에서 쓴 결과 파일은 CRLF 줄바꿈임. 이 저장소는 `core.autocrlf=true`라서 git이 LF로 바꿔 저장함. 그래서 커밋된 60개 파일이 모두 `SHA256SUMS.txt`와 달랐음 (Linux/macOS에서 받으면 검증 실패).

- `audit/results_v6/.gitattributes`에 `* -text`를 넣고 다시 저장함.
- 저장소에 저장된 60개 파일 모두 기록된 해시와 일치함을 확인함.

## I-26 (P2) 후보 수의 fold 간 불안정

Hybrid validation에서 선택된 후보 수가 F1 20000, F2 500, F3 all로 다름. 같은 fold에서 후보 수별 validation 점수 차이가 매우 작음 (곡선이 평평함). 그래서 후보 수는 성능의 핵심 요인이 아니며, 원고에서 "retrieve-then-rank로 효율과 정확도를 함께 얻는다"는 주장을 하지 않아야 함.
