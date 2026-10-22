# Issue register (paper_revision_codex 독립 감사)

등급: P0 = 주장 자체를 무효화할 수 있음, P1 = 결론을 바꿀 수 있음, P2 = 재현성·해석 위험, P3 = 경미.
상태: `해결·검증` / `수정·검증 대기(실험 필요)` / `환경 제약` / `원고 수정 필요` / `공개·한정`(시험으로 확인할 수 없어 공개하고 해석을 한정함) / `미해결`.
"검증"은 이 PC에서 실제로 실행한 테스트·스크립트만 가리킨다. 결과 JSON을 읽은 것만으로는 검증으로 치지 않는다.

> **현재 상태 (2026-10-22 KST)**: v6 실제 데이터 실행이 3 fold 모두 완료됨 (3 fold의 봉인 평가 결과가 커밋됨. 종료 코드 0은 v6 PC의 실행 로그 기준). I-23 확인을 위한 v7(처음 쓰는 구간 F4, ALS·BPR·GNN grid 확장)도 완료됨. Cycle 4 재감사 항목(I-39 – I-47), Cycle 5 재감사 항목(I-48 – I-57), Cycle 6 재감사 항목(I-58 – I-62), Cycle 7 재감사 항목(I-63 – I-68), Cycle 8 재감사 항목(I-69 – I-73), Cycle 9 재감사 항목(I-74 – I-79), Cycle 10 재감사 항목(I-80 – I-84), Cycle 11 재감사 항목(I-85 – I-89), Cycle 12 재감사 항목(I-90 – I-95), Cycle 13 재감사 항목(I-96 – I-100), Cycle 14 재감사 항목(I-101 – I-102), Cycle 15 재감사 항목(I-103 – I-104), Cycle 16 재감사 항목(I-105 – I-108), Cycle 17 재감사 항목(I-109 – I-111) 처리 완료. 단 I-63(novel 튜닝 비대칭, P1)은 시험 구간을 다시 열 수 없어 공개·해석 한정으로 처리함.
> - 시험 열람: v6는 fold마다 봉인 평가 1회, F1에서만 사후 분석 2회 추가 (`audit/results_v6/test_access_log.jsonl`). v7은 F4 봉인 평가 1회 (`audit/results_v7/test_access_log.jsonl`).
> - 결과와 검증: `validation_report.md` (v7은 §12).
> - Cycle 2 재감사에서 I-23 – I-26, Cycle 2 커밋 후 검증(418e7dd)에서 I-27, Cycle 3에서 I-28 – I-37, Cycle 3 커밋 후 검증에서 I-38, Cycle 4 재감사에서 I-39 – I-47, Cycle 5 재감사에서 I-48 – I-57, Cycle 6 재감사에서 I-58 – I-62, Cycle 7 재감사에서 I-63 – I-68, Cycle 8 재감사에서 I-69 – I-73, Cycle 9 재감사에서 I-74 – I-79, Cycle 10 재감사에서 I-80 – I-84, Cycle 11 재감사에서 I-85 – I-89, Cycle 12 재감사에서 I-90 – I-95, Cycle 13 재감사에서 I-96 – I-100, Cycle 14 재감사에서 I-101 – I-102, Cycle 15 재감사에서 I-103 – I-104, Cycle 16 재감사에서 I-105 – I-108, Cycle 17 재감사에서 I-109 – I-111을 찾음.

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
| I-23 | P1 | v6의 ALS/BPR이 grid 경계에서 선택됨. 경계 밖에서 ALS validation이 +0.011 – +0.027 | 해결·검증 (한계 공개): 사용자 승인으로 처음 쓰는 구간 F4에서 ALS·BPR·GNN grid를 넓힌 v7 실행(Hybrid grid·ALS+BPR α는 v6와 같음). ALS가 validation에서 GNN을 앞섰지만(0.2176 vs 0.2169) 시험에서 Hybrid − ALS+BPR은 4칸 모두 비유의 (v6는 NDCG 6칸 비유의, Recall 3칸 유의). "가장 강한 baseline보다 일관되게 유의하게 높지 않다"는 결론 유지. v6 F1–F3 수치는 덜 튜닝된 조건으로 남고, v7에서도 일부 축이 끝에 걸림(사전등록 규칙상 재확장 안 함) | c9d3e49, dea0f20 |
| I-24 | P3 | ALS 결과가 BLAS 스레드 수에 따라 달라짐 | 문서화: v6·v7의 실제 스레드 수는 기록되지 않음. v6는 다른 PC 1스레드 재학습과 비트 단위 일치(§9), v7은 F4 seed 42 ALS가 1스레드 재학습과 비트 단위 일치·24스레드와 불일치(Cycle 5)로 1스레드였음을 뒷받침. v6·v7 시점 코드는 다른 환경변수가 이미 있으면 그 값이 우선했음 → Cycle 4에서 강제 고정 (I-41) | Cycle 2 커밋 |
| I-25 | P2 | `determinism_check.py`가 numpy를 import한 뒤 스레드 환경변수를 설정함 (거짓 비결정성 보고) | 해결·검증 | Cycle 2 커밋 |
| I-26 | P2 | Hybrid 후보 수가 fold마다 다름 (20000 / 500 / all) | 해석 사항: validation 곡선이 평평함. 원고에서 후보 수를 핵심 설계로 주장하지 않음 | - |
| I-27 | P2 | 커밋된 결과 파일이 `SHA256SUMS.txt`와 불일치 (줄바꿈 자동 변환) | 해결·검증 (커밋 후 검증에서 발견). 새 clone(다른 PC)에서도 60/60 일치 | 418e7dd |
| I-28 | P3 | 인자 없는 `python -m pytest`가 보관용 `source_*` 폴더의 옛 테스트까지 수집해 오류 16개 | 해결·검증: `pytest.ini`로 수집 범위 제한. 다른 PC의 깨끗한 venv에서 70개 통과 | 1170ea5 |
| I-29 | P3 | v7 "미사용 구간" 테스트가 비율만 비교함 (실제 시각 미확인) | 해결·검증: 원본 타임스탬프로 F4 끝 ≤ v6 내부 validation 시작을 확인하는 테스트 추가. 실제 경계는 같은 타임스탬프(겹침 0건). end 0.37 변이를 잡음 (일회성 수동 확인, 기록 없음) | 1170ea5 |
| I-30 | P2 | 평가·선택·통계의 핵심 성질을 고정하는 테스트 부재 (변이 테스트에서 발견) | 해결·검증: 구현은 올바름(정독 + 실제 데이터 대조). 기존 테스트는 변이 17개 중 11개를 놓침. `test_cycle3_consistency.py`(9개) 추가 후 두 묶음 합쳐 17/17. M04는 합성 데이터에서 필터 전후 이력이 같아 구별되지 않던 것을 전용 데이터로 고정함. v7 실행 중 코드 해시 잠금 때문에 v7 종료 후 커밋 | 70f5106 |
| I-31 | P3 | 분리 실행(`DETACHED_PROCESS`) 시 Intel Fortran 런타임이 창 닫기 이벤트로 프로세스를 중단 (`forrtl: error (200)`) | 해결: `CREATE_NO_WINDOW` + `FOR_DISABLE_CONSOLE_CTRL_HANDLER=1`로 재실행. 중단은 v7 전처리 중이었고 시험 열람·학습 없음. 미완성 폴더는 `prepared_incomplete_run1`로 보존 | 문서 |
| I-32 | P2 | 원고용 그림이 v5 단일 분할 결과 기준 (`professor_materials/figures`) | 해결·검증: 커밋된 v6 CSV만 읽는 그림·표 생성기와 결과(`audit/figures_v6`). 재실행 시 바이트 동일 | a2c5445 |
| I-33 | P3 | BPR(GPU) 결과가 GPU·torch 빌드에 따라 달라짐 | 문서화: 다른 PC에서 F3 MF grid 재실행. ALS 24행은 비트 단위 동일, BPR 20행은 선택 점수 최대 0.0047 차이, 선택 결과는 같음 | e7f0240 |
| I-34 | P3 | 같은 seed라도 CPU와 GPU의 학습 결과가 다름 | 문서화: F3 내부 validation에서 GNN 목록 겹침 84%, BPR 28%, validation 점수 차이 0.0024 / 0.0013 (seed 변동 범위). 모든 v6/v7 결과는 CUDA 기준 | 868b714 |
| I-35 | P3 | 그림 생성기 제목에 v6 전용 문구("F1 window was exposed in v1-v5", seed 목록)가 고정돼 v7 그림에도 찍힘. 범례가 막대를 가림 | 해결·검증: 제목을 데이터(fold 이름, `test_metrics.csv`의 seed)에서 만들고 범례를 그림 아래로 옮김. v6/v7 그림을 다시 만들어 두 번 실행 시 바이트 동일 확인 | dea0f20 |
| I-36 | P2 | `requirements.txt`가 scikit-learn 1.9.1을 고정하지만 v6·v7은 1.9.0으로 실행됨. v6·v7 실행에 쓴 분리 실행 스크립트도 저장소에 없음 (`audit_runs/`는 gitignore) | 해결·검증: v7 venv를 그대로 적은 `requirements-v7-lock.txt` 추가. 새 venv에 설치해 패키지 목록이 v7 venv와 완전히 같음을 확인. `report_tools/launch_detached.py` 추가 후 실제 실행으로 동작 확인 (일회성, 실행 출력은 저장소에 없음). `requirements.txt` 자체는 기존 문서의 기준이라 바꾸지 않음 | e9538c8 |
| I-37 | P2 | codex 폴더 첫 문서 `README.md`가 v5 주장("두 지표의 세 seed 평균이 모든 비교 모델보다 높았다")을 감사 결과 안내 없이 제시. `README_HYBRID_KO.md`, `REPRODUCE_KO.md`도 v5 경로만 안내. `professor_materials/README_KO.md`(같은 주장과 "NDCG +10.83%"), 저장소 최상위 `README.md`(v5 재실행을 "최종 Hybrid 실험"으로 안내)도 같음 | 해결: 원문은 이력으로 보존하고, 맨 앞에 시험 재사용 사실과 감사 문서(검증 결과, 재현 절차, 원고 대조표) 안내를 추가 | e9538c8 |
| I-38 | P2 | plan의 코드 해시(`code_hashes`)가 작업 폴더의 원시 바이트를 해시해 checkout의 줄바꿈 변환(`core.autocrlf`)에 따라 달라짐. 같은 commit이라도 다른 PC에서는 plan 검사가 거짓 실패하고, "코드 변경"과 "줄바꿈 차이"를 구별할 수 없음 (v7 plan은 CRLF 19개·LF 7개가 섞여 있었고, 이 PC에서 v6 plan과 비교하면 내용이 같은 파일 7개가 불일치) | 해결·검증: 새 plan은 줄바꿈을 정규화한 해시(`sha256-lf`)와 방식을 기록함. 방식 기록이 없는 v6/v7 plan은 원시 바이트로 비교해 호환 유지. `report_tools/verify_plan_code.py`로 v6 25개, v7 26개 해시가 모두 각 plan commit의 git 내용과 일치함을 확인 (`audit/plan_code/`. 조작한 해시 1개를 검출한 확인은 일회성으로 출력 없음). 테스트 3개 추가 | b575dac |
| I-39 | P2 | 재개 시 누수 검사 게이트가 `leakage_check.json`의 존재만 확인함. 이 파일은 검사가 실패해도 쓰이므로, 실패한 fold를 재개하면 검사를 건너뛰고 진행됨 | 해결·검증: `leakage_gate`가 저장된 `all_pass`를 읽어 거짓이면 멈춤. 테스트 2개, 변이(옛 동작) 검출. v6 3 fold·v7 F4는 모두 `all_pass: true`라 결과 영향 없음 | 05b9c65 |
| I-40 | P2 | 시험 열람 가드가 같은 root 안에서 plan 해시만 비교함. `selection.json`을 다시 만들어 같은 plan으로 시험을 다시 열거나, 새 root에서 바뀐 plan으로 같은 구간을 여는 것을 막지 못함 (docstring은 "fold마다 한 번"이라고 설명) | 해결·검증: 같은 root에서 `selection_sha256`이 다르면 거부. 실행 시작 시 커밋된 `audit/results_*`의 plan·접근 로그를 읽어 같은 원본 해시·같은 구간(`end_fraction`, `test_size`)을 다른 plan으로 연 기록이 있으면 거부 (같은 plan의 재현은 허용). 실제 커밋 기록으로 F1–F4가 새 plan에 막히고 v6·v7 자신은 막히지 않음을 테스트 (Cycle 4 시점. Cycle 5부터는 코드도 같아야 예외라 현재 코드에서는 v6·v7도 막힘 → I-48). 커밋되지 않은 root는 여전히 볼 수 없음(문서화). v6·v7 접근 로그는 fold마다 sealed_test 1회라 결과 영향 없음 | 05b9c65 |
| I-41 | P2 | `rolling_protocol.py`가 BLAS 스레드 변수를 `setdefault`로 설정해 호출자가 준 값이 우선함. 나중에 import되는 `experiments.py`가 변수를 1로 다시 쓰지만 BLAS 로드 뒤라, 기록·확인용으로 변수를 읽으면 실제와 다를 수 있음. 스레드·장치가 plan에 기록되지 않음 | 해결·검증: 값을 강제로 1로 설정. plan `environment`에 변수, 로드된 라이브러리의 실제 스레드 수(threadpoolctl), torch 스레드, 장치 기록. 호출자가 4를 줘도 실제 BLAS 스레드가 1인지 확인하는 테스트(옛 동작 변이 검출). v6·v7의 실제 스레드 수는 기록되지 않음 (v7을 띄운 launcher는 커밋 전 버전이라 남아 있지 않음. `launch_detached.py`는 v7 뒤에 커밋). 결과로 확인: v6은 §9 교차 재현, v7은 F4 seed 42 ALS가 1스레드 재학습과 비트 단위 일치·24스레드와 불일치 (`audit/cycle5/determinism_v7_blas*.csv`, Cycle 5에서 정정) | 05b9c65 |
| I-42 | P3 | GNN 상한 연장을 한 번만 하고 다시 정렬함. 연장 뒤 다른 상한 도달 설정이 Hybrid backbone 상위 3개에 들어오면 연장 없이 쓰일 수 있음 | 해결·검증: 상위 3개에 연장 안 된 상한 도달 설정이 없을 때까지 반복 (`extend_capped_backbones`). 테스트 2개, 변이 검출. v6·v7은 상위 3개에 상한 도달이 없어 결과 영향 없음 | 05b9c65 |
| I-43 | P3 | 재개 기준 파일 `test_metrics.csv`를 보고서 CSV보다 먼저 씀. 그 사이에 중단되면 재개 때 보고서를 만들지 않음 | 해결: `test_metrics.csv`를 마지막에 씀. 코드 정독으로 확인(전용 테스트 없음). v6·v7은 보고서가 모두 있음 | 05b9c65 |
| I-44 | P3 | `leakage_check.py`를 단독으로 원본 폴더에 대해 실행하면 rolling fold의 끝 비율을 알 수 없음 (manifest에 없음). 원본 전체로 v7 F4를 검사하면 28개 중 12개가 거짓 실패 | 해결·검증: `--end-fraction` 추가. 원본 전체 + 0.36으로 v7 F4를 검사해 커밋된 결과와 28개 모두 같음 (`audit/cycle4/`). 테스트 1개, 변이 검출 | 05b9c65 |
| I-45 | P3 | `verify_plan_code.py`가 코드가 바뀌어도 종료 코드 0을 냄 (plan 해시와 commit만 확인) | 해결·검증: 종료 코드의 뜻을 문서화하고 `--require-unchanged` 추가. 실제 v6 plan으로 0 / 1 확인하는 테스트 | 05b9c65 |
| I-46 | P3 | 선택 경로는 ALS/BPR 원 점수로, 시험 경로는 min-max 후 float32 점수로 순위를 매김. 동점 순서가 달라질 수 있음 | 측정: v7 F4 내부 validation에서 두 정책 × ALS/BPR 모두 목록 100% 동일, 지표 차이 0 (원 점수도 float32). 코드 변경 없음 (`audit/cycle4/score_precision_F4_val.json`) | 05b9c65 |
| I-47 | P2 | 감사 문서의 수치·표현 오류 24건 (v6 Hybrid − ALS+BPR Recall 3칸 유의를 "v6와 같이 모두 비유의"로 쓴 것, Recall 0 비율, 후보 수 범위, γ 정책별 값, GNN 상한 4+1개, 반올림 등). 저장소만으로 다시 확인할 수 없는 주장(v6 PC 전용 검사)이 표시 없이 있었음 | 해결·검증: 각 수치를 커밋된 CSV·JSON에서 다시 계산한 뒤 고침. 확인할 수 없는 주장은 그렇게 표시함 | e3cf44e |
| I-48 | P2 | Cycle 4 시험 열람 가드의 빈틈 세 가지: (1) 다른 root의 기록은 plan 해시만 같으면 면제해, 코드를 바꾸고 같은 plan으로 같은 구간을 다시 열 수 있었음. (2) 구간이 정확히 같을 때만 충돌로 봐, 겹치는 다른 구간(예: end .65, test .2)은 통과. (3) 커밋된 `audit/results_*`만 읽어, 같은 PC의 커밋 안 된 root(`audit_runs/*`)는 보지 않음 | 해결·검증: 면제는 plan 해시와 코드 해시(`code_matches`, 줄바꿈 무관, 파일 목록까지 같음)가 모두 같을 때만. 바깥 시험 구간 (end·(1−test), end)이 겹치면 충돌 (맞닿기만 하면 아님). `audit/`와 `audit_runs/` 바로 아래 root(`*/plan.json`)의 plan·접근 로그를 읽음 (그 밖의 위치에 만든 root는 보지 못함, 문서화). 테스트 추가, 옛 동작 변이 H1–H3 검출. 실제 원본으로 현재 코드에서 v6 plan, v7 plan, 새 plan 모두 F1–F4가 충돌하고, 커밋 안 된 `audit_runs/v7`도 잡힘 (Cycle 5 당시 출력은 없음. Cycle 7 코드로 다시 한 결과 `audit/cycle7/guard_real_data.json`). v6·v7은 이제 코드가 바뀌어 같은 구간을 다시 열 수 없음(재현은 plan commit checkout 필요, 원래 그랬음) | Cycle 5 |
| I-49 | P2 | 원본 데이터 해시를 plan을 처음 만들 때만 계산함. 같은 root를 다른 `--raw`(다른 원본)로 재개해도 막지 않음 | 해결·검증: 매 실행마다 원본 4개 파일을 다시 해시해 plan의 `raw_input_sha256`과 다르면 멈춤. 실행 시작을 `run_started`(git, 실제 BLAS 스레드 수)로 로그. 테스트, 변이 H4 검출 | Cycle 5 |
| I-50 | P3 | 접근 로그의 selection 비교가 같은 목적(purpose) 안에서만 이뤄짐. 또 완료된 단계(`test_metrics.csv`, 사후 분석 CSV가 있음)는 로그 검사 전에 돌아가, 바뀐 plan·selection으로 재개해도 조용히 통과 | 해결·검증: `open_test`가 fold의 모든 목적에 대해 plan·selection 해시를 비교하고, 완료된 단계도 검사만 하고(기록은 안 함) 돌아감. 테스트, 변이 H5·H6 검출 | Cycle 5 |
| I-51 | P3 | 재개 시 캐시된 MF 학습(`mf_fit.json`)과 GNN 작업(`job_result.json`)을 설정 확인 없이 재사용 | 해결·검증: MF는 seed·ALS·BPR·BPR epoch, GNN은 config·fixed·seed·mode·epochs가 요청과 같아야 재사용(장치는 달라도 됨). 테스트, 변이 H7·H8 검출. 실제 v7 root: plan·selection·`validation_gnn.json`에서 다시 만든 요청 GNN 39개(validation 36 + refit 3)와 MF 4개가 캐시와 모두 일치 (`audit/cycle5/cache_identity_v7.json`, Cycle 6에서 도구를 고쳐 다시 만듦 → I-60) | Cycle 5 |
| I-52 | P3 | 재개 시 `selection.json`을 만든 plan이 현재 plan인지 확인하지 않음 | 해결·검증: `sealed_selection`이 다르면 멈춤. 테스트, 변이 H12 검출 | Cycle 5 |
| I-53 | P3 | 사후 분석의 재개 기준 파일(`posthoc_candidate_depth.csv`)을 γ 표보다 먼저 씀. 그 사이에 중단되면 재개 때 γ 표 없이 완료로 처리 | 해결·검증: 기준 파일을 마지막에 씀. γ 단계에서 중단시키는 테스트(기준 파일이 없어야 하고, 재개하면 두 표가 모두 생김), 변이 H11 검출 | Cycle 5 |
| I-54 | P3 | `verify_plan_code.py --require-unchanged`가 plan 뒤에 추가된 파일은 무시함 | 해결: 추가된 파일도 실패로 처리. 실제 v6·v7 plan으로 0 / 1 확인 (둘 다 바뀐 파일도 있어 추가 파일만의 효과는 코드 정독으로 확인) | Cycle 5 |
| I-55 | P3 | 보조 스크립트의 스레드 설정 불일치: `cross_machine_repro`·`determinism_check`·`mf_boundary_probe`·`device_check`가 `setdefault`를 써 호출자 값이 우선. `check_score_precision`은 numpy를 `rolling_protocol`보다 먼저 import해 BLAS가 다중 스레드로 로드됨 | 해결·검증: 모두 1로 강제. `check_score_precision`은 `rolling_protocol`을 먼저 import. 호출자가 4를 줘도 실제 BLAS 스레드가 1인지 확인하는 테스트, 변이 H9 검출. 이전에 커밋된 보조 결과의 실제 스레드 수는 기록되지 않음: `determinism_mf_blas1.csv`(재학습 = 원래 값)와 §9 교차 재현(v6 값과 비트 단위 동일)은 1스레드 결과와 일치하고, `score_precision_F4_val.json`은 고친 도구(1스레드)로 다시 실행한 결과가 커밋된 파일과 완전히 같음 (`audit/cycle5/score_precision_F4_val_blas1.json`). `mf_boundary_probe.csv`는 다시 실행하지 않아 확인하지 않음 | Cycle 5 |
| I-56 | P3 | 실행 중 누수 검사 게이트가 fold의 `fold_raw`(이미 잘린 원본)로 검사해, fold 자르기 자체를 독립적으로 확인하지 못함 | 해결·검증: 원본 전체 + fold의 `end_fraction`으로 검사 (I-44의 옵션 사용). 테스트, 변이 H10 검출. v7 F4는 원본 전체 + 0.36 검사 결과가 커밋된 결과와 28개 동일 (Cycle 4) | Cycle 5 |
| I-57 | P2 | 감사 문서 오류 12건: v7에서 "가장 강한 baseline = ALS+BPR"(repeat에서는 BPR이 평균 최고), "Holm 보정 CI"(CI는 보정 안 함), "ALS가 validation에서 학습 모델 중 1위"(단일 모델군 중 1위), 결정성 재검사에 BPR도 있었음, p_holm 0.062→0.061, v7이 `launch_detached.py`로 실행됐다는 주장, 보조 스크립트 스레드 설정 범위, "v6는 1스레드로 고정"(기록 아닌 정황 근거), v7 전처리 완료 시각, v6 완료 시각(저장소로 확인 불가), "413명"(novel은 306명), "80:20"(F1만) | 해결·검증: 각 항목을 커밋된 CSV·JSON·로그에서 다시 확인한 뒤 고침 | Cycle 5 |
| I-58 | P2 | 문서가 `Hybrid_ALS_only`를 "BPR을 뺀 ablation"으로 설명하고 "BPR 결합의 기여는 구간에 따라 다르다"고 해석함. 코드(`policy_evaluation.hybrid_ranking`, `mf_source='als'`)는 최종 결합 점수에서만 BPR을 빼고, 후보는 Hybrid와 같이 ALS+BPR(`balanced_union`이면 BPR 상위 절반 포함)로 만듦 | 해결: 문서를 "같은 후보에서 결합 점수의 MF 부분만 ALS"로 고치고 BPR 기여에 대한 해석을 삭제. 수치는 맞으므로 결과 변경 없음. BPR을 후보에서도 뺀 ablation은 실행하지 않음 | Cycle 6 |
| I-59 | P3 | 다른 root의 시험 열람 기록을 원본 4개 파일의 바이트 해시가 모두 같을 때만 비교함. 줄바꿈만 다른 사본이나 구간과 무관한 파일(`category_tree.csv` 등)을 바꾸면 기록을 건너뜀 | 해결·검증: 구간을 정하는 `events.csv`의 해시로만 비교하고, 실행 시 그 파일의 원래·LF·CRLF 형태 해시를 함께 씀(`event_digests`). 테스트, 변이 H13 검출. 실제 원본으로 v6·v7 plan의 events 해시가 일치하고 기록 5건이 모두 잡힘 (Cycle 6 당시 출력은 없음. Cycle 7에 `audit/cycle7/guard_real_data.json`으로 다시 확인). 행 내용이 바뀐 `events.csv`는 여전히 다른 데이터로 봄(문서화) | Cycle 6 |
| I-60 | P3 | `check_cached_runs.py`의 GNN 검사가 `job_specs/*.json`과 `job_result.json`을 비교함. 둘은 같은 파일에서 나와 항상 일치하므로, 캐시가 지금 plan·selection이 요청할 설정과 맞는지 확인하지 못함 (I-51의 실제 데이터 근거가 무의미했음) | 해결·검증: plan(grid, 고정 설정, epoch 상한, seed), `validation_gnn.json`(상한 연장), `selection.json`(refit 설정·epoch)에서 요청을 다시 만들어 비교. 어긋난 refit을 잡는 테스트. 다시 만든 v7 결과: 요청 39개 모두 존재·일치, MF 4개 일치 | Cycle 6 |
| I-61 | P3 | `git_state`가 git 출력 전체를 `strip()`해 첫 줄의 상태 열 공백이 사라지고, 경로 비교(`line[3:]`)가 한 글자 밀림. 첫 줄이 `audit_runs` 파일이면 제외되지 않음 | 해결·검증: 끝 줄바꿈만 제거. 테스트, 변이 H14 검출. `audit_runs/`는 gitignore 대상이라 v6·v7 실행에는 영향 없음 | Cycle 6 |
| I-62 | P2 | 감사 문서 오류 11건과 근거 표시 누락: I-24의 "v7은 1로 고정하는 launcher로 실행"(기록 없음), §12.5의 CI가 0을 포함하지 않지만 유의하지 않은 칸 목록에서 2칸 누락(repeat Hybrid − GNN Recall, novel Hybrid − GNN NDCG), "시험 사용자 413명"(novel 306), I-40·§13의 "v6·v7 자신은 통과"(Cycle 5부터는 막힘), v7 전처리 완료 시각이 커밋된 `progress.jsonl`에 있음, `rolling_protocol.py` 변경 횟수, ablation이 더 높은 칸 F1 repeat Recall 누락, Recall 0 비율 범위(ablation 73.2% 제외), I-48 "모든 `plan.json`"(한 단계 아래만), `--require-unchanged` 실패 파일(`leakage_check.py`도 바뀜), §7 표 끊김. 또 저장소로 다시 확인할 수 없는 기록(v6 중단 시각, exit 0, 수기 계산, 일회성 변이, 실행 출력)에 출처 표시가 없었음 | 해결·검증: 각 항목을 커밋된 CSV·JSON·로그와 실제 실행(`verify_plan_code`, 두 venv `uv pip freeze`, 변이 사본과 fd9bcca 비교)으로 확인한 뒤 고치고, 확인할 수 없는 기록에는 출처와 "저장소로 확인 불가"를 표시 | Cycle 6 |
| I-63 | P1 | novel 정책의 튜닝 비대칭: 학습 설정(ALS, BPR, GNN, Hybrid backbone)은 repeat validation으로 고르고(사전등록 `policy_rule`), novel에서는 Hybrid 결합 설정 2,808개와 ALS+BPR α만 다시 고름. ALS·GNN 단독은 novel로 튜닝되지 않음. 문서에 이 영향이 없었음 | 공개·한정 (이미 연 시험 구간을 다시 열 수 없어 시험으로는 확인 불가): validation 전용 도구 `report_tools/policy_tuning_gap.py`로 커밋된 선택 파일에서 계산. novel로 골랐다면 GNN validation 점수 F2 0.0598→0.0650, F3 0.0529→0.0581, F4 0.0526→0.0659(Hybrid 0.0612보다 높음), ALS도 네 fold 모두 상승(F2는 +0.00006), BPR은 변화 없음 (`audit/cycle7/policy_tuning_gap_v*.json`). 테스트: Cycle 7 판은 선택을 repeat 1위로 정의해 순환적이었음 → Cycle 8에 `selection.json`의 MF 설정을 grid에서 찾도록 고치고, 그 설정이 repeat 1위인지 확인 (테스트는 v7 F4만 검사. v6 F1–F3도 도구 출력에서 GNN·ALS·BPR 모두 1위. 값은 같음, `audit/cycle8/policy_tuning_gap_v*.json`). ALS+BPR의 ALS·BPR도 repeat로 고른 것이라 novel의 Hybrid − ALS+BPR 결과도 같은 조건으로 표시. novel의 Hybrid − GNN·ALS 유의 결과를 "repeat로 튜닝된 단독 모델 대비"로 한정하고 원고 공개 사항·금지 표현에 추가 | Cycle 7 |
| I-64 | P2 | 다른 root의 시험 열람 기록을 실행 시작 때만 확인함. 병렬로 시작한 두 root가 겹치는 구간을 둘 다 열 수 있음 | 해결·검증: `open_test`가 시험을 열기 직전에 다시 확인. 테스트, 변이 H15 검출. 확인과 기록 사이의 짧은 경쟁 구간은 남음(파일 잠금 없음, 문서화) | Cycle 7 |
| I-65 | P3 | 정확한 재현(같은 plan·코드) 예외가 `events.csv` 밖의 원본(`available`과 γ를 정하는 `item_properties`)을 보지 않음 | 해결·검증: 예외에 원본 4개 파일 해시 일치를 추가. 테스트, 변이 H16 검출. 실제 원본은 v6·v7 plan 기록과 4개 파일 모두 같음 | Cycle 7 |
| I-66 | P3 | 재개 시 Hybrid validation grid CSV를 존재만으로 재사용. 행 경계에서 잘린 CSV는 오류 없이 읽혀 일부 grid에서 선택함 (감사자가 임시 폴더에서 v7 grid 사본을 1/3로 잘라 선택이 바뀌는 것을 재현. 그 출력은 저장소에 없음) | 해결·검증: 행 수가 grid 크기와 같을 때만 재사용, 쓰기는 임시 파일 후 교체. 테스트, 변이 H17 검출. 기존 v7 grid 6개는 모두 2,808행 | Cycle 7 |
| I-67 | P3 | git을 읽지 못하면(설치 안 됨, 저장소 아님) `git_state`가 commit None을 돌려주고, 커밋 확인을 통과해 plan이 commit 없이 만들어짐 | 해결·검증: commit을 읽지 못하면 plan을 만들지 않음. 테스트, 변이 H18 검출 | Cycle 7 |
| I-68 | P2 | 문서 오류 10건과 근거 표시 누락: `manuscript_alignment.md`의 ablation 문구에 F1 repeat Recall 누락(I-62 정정이 이 문서에 빠짐), 두 README의 "모든 fold에서 유의하게 높지 않았다"(일부 칸은 유의)와 v7 최강 baseline 표기, 가드 변이 시간, `fold_raw.json` "해시 5개"(해시는 2개), lock 파일과 freeze 차이(주석 6줄도 다름), §4.3의 novel Hybrid − RecentPop 행 누락, 결정성 표의 "GNN" 행(실제로는 Hybrid의 GNN backbone), 머리말의 PC 범위, cycle_log의 커밋 해시 3곳. 실제 원본으로 한 가드 확인, 조작 해시 검출, launcher 실행 등 일회성 확인에 출력 없음 표시가 없었고, Cycle 4 변이 재실행은 커밋 전 코드로 실행됨. 투고본 수치의 출처 미기재 | 해결·검증: 각 항목을 CSV·JSON·git으로 확인한 뒤 고침. 실제 원본 가드 확인은 `report_tools/guard_real_data_check.py`로 다시 실행해 `audit/cycle7/guard_real_data.json`에 남김. 나머지 일회성 확인은 표시. 투고본 수치는 `Model/paper/viz.py`의 "5. ALS+GNN" 값으로 출처 기재 | Cycle 7 |
| I-69 | P2 | Cycle 7 정정 뒤 남은 문서·도구 문제: `manuscript_alignment.md` v7 대체 서술과 두 README 안내에 I-63 한정이 없음, ALS+BPR의 ALS·BPR도 repeat로 골랐다는 점 누락, ALS가 novel 튜닝으로 오르는 fold 목록 불완전, `policy_tuning_gap.py`의 ALS·BPR "선택"이 repeat 1위로 정의돼 테스트가 순환적, I-68 행 중복, CPU/GPU 차이 범위(Recall은 0.003을 넘음), 머리말 PC 범위, `plan_code` 현재 결과 참조, §7 보조 실행 표 누락, 출력이 없는 114/114·22분 14초·커밋 전 코드로 한 H1–H18 실행에 표시 없음 | 해결·검증: 각 항목을 커밋된 파일로 확인 후 고침. 도구는 `selection.json`의 MF 설정을 grid에서 찾도록 고쳐 같은 값을 재현(`audit/cycle8/policy_tuning_gap_v*.json`). `guard_mutation_check.py`가 출력에 git HEAD·커밋 안 된 파일·시각을 기록하고 출력 경로를 받도록 함. GNN의 novel 최고값은 epoch를 공통 기준으로 정한 값이라 하한임을 §8 항목 5에 적음 | b9e7134 (도구), 79b628d (문서) |
| I-70 | P2 | 코드 고정을 실행 시작·재개 때만 확인. GNN 작업은 디스크의 코드를 새로 import하는 하위 프로세스이고 일부 모듈은 처음 쓸 때 import되므로, 실행 중(v6은 F1 시험 열람부터 F3까지 3.2시간) 코드를 고치면 바뀐 코드로 학습하고 시험은 옛 plan의 코드 해시 아래 기록될 수 있음 | 해결·검증: `require_frozen_code`를 실행 시작, GNN 작업 시작 직전, 새 시험 열람 직전에 확인 (`test_cycle8_guards.py`, 가드 변이 H19·H20). 실제 영향: v7 산출물 43개에 찍힌 코드 해시 731개(최상위 비테스트 `.py`)가 plan과 모두 같음 (`report_tools/stamp_code_check.py`, `audit/cycle8/stamp_code_v7.json`). v6 실행 폴더는 v6 PC에만 있어 확인하지 못함 | b9e7134 (가드), 79b628d (`stamp_code_check.py`) |
| I-71 | P3 | 실행 중 코드가 바뀌면 I-64 재확인이 자기 root를 "다른 실행"으로 보고 F1 사후 분석 열람을 거부함 (안전한 쪽 실패, 원인 메시지가 틀림) | 해결·검증: 시험 열람 때 코드 확인을 먼저 하고(I-70), 자기 root는 다른 root 검사에서 제외(자기 열람은 접근 기록이 관리). 테스트·H21 | b9e7134 |
| I-72 | P3 | Hybrid grid 캐시를 기본 CSV 파서로 다시 읽어 값이 몇 ulp 달라짐. 재개한 실행의 `selection.json` 수치가 중단 없는 실행과 다를 수 있음 | 해결·검증: `float_precision='round_trip'`. v7 grid 6개: 기본 파서는 2,808행 모두 바꾸고 round-trip은 다시 쓰면 줄바꿈(CRLF) 외에는 파일과 같음, 점수 순서는 둘이 같음. `selection.json`의 수치 26개가 round-trip 값에서 그대로 나오고 기본 파서 값에서는 11개만 나옴 (`audit/cycle8/grid_cache_precision_v7.json`). v7 선택은 캐시 없이 계산돼 영향 없음. 테스트·H26 | b9e7134 |
| I-73 | P3 | 다른 root의 접근 기록 마지막 줄이 잘리면(쓰다 중단) 같은 `events.csv`를 쓰는 모든 실행이 `JSONDecodeError`로 멈춤. 반쯤 쓰인 `plan.json`도 같음 | 해결·검증: 읽을 수 없는 기록 줄은 그 기록의 모든 fold를 연 것으로 봄(겹치는 구간만 거부). 읽을 수 없는 `plan.json`과 자기 접근 기록은 파일을 지목하는 오류. 테스트·H22–H25 | b9e7134 |
| I-74 | P3 | Cycle 9 문서 감사 7건: §7 표에 Cycle 8 공식 변이 재실행 행 없음, 낡은 Cycle 5 가드 변이 행 중복, 이 문서 머리말에 I-69 – I-73과 상태 `공개·한정` 없음, grid 재읽기를 "바이트 단위 재현"이라 함(실제는 CRLF 외 동일), `v7_feasibility.md`의 실행 폴더가 이 PC에 없다는 표시 없음, 원고 권장 문구(후보 500개·재고 가중)에 사후·단일 fold·단일 seed 표시 없음, CI 출처를 §4.3로 잘못 가리킴. 함께 고친 것: §4.3 해석에 평균이 더 낮은 칸 명시, 자동 생성 쌍대 표의 I-63 주석 필요, `policy_tuning_gap` 테스트가 v7만 검사함, 가드 변이 실행 시간, 출력 없음 표시, 커밋 열, `stamp_code_check.py` 설명의 한계 | 해결·검증: 각 항목을 증거 파일·코드·git으로 확인 후 고침 (v6 F1–F3의 repeat 1위는 `audit/cycle8/policy_tuning_gap_v6.json`에서 확인) | Cycle 9 |
| I-75 | P3 | 다른 root의 "정확한 재현"(같은 plan·코드·원본)이 같은 fold를 다른 `selection.json`으로 다시 열 수 있었음 (같은 root에서는 I-40이 막음) | 해결·검증: 시험 열람 직전 검사에서 정확한 재현은 그 기록이 같은 fold를 같은 selection 해시로 연 경우에만 예외 (실행 시작 때는 selection이 아직 없어 이전처럼 예외). 읽을 수 없는 기록 줄은 selection을 모르는 열람으로 봄. 테스트·H28·H31. v6·v7은 fold마다 한 번씩만 열림 **Cycle 10 정정(I-81)**: 비교 대상이 `selection.json` 파일 해시였는데, 이 파일에는 봉인 시각·절대경로·학습 시간이 들어 있어 원본부터 다시 돌린 정확한 재현은 선택이 같아도 항상 거부됐음. 결정 해시로 바꿈 | ce97790, 4e166c5 (I-81) |
| I-76 | P3 | 단계 완료 표시 파일(`test_metrics.csv`, `posthoc_candidate_depth.csv`, `if_ablation.csv`)을 직접 써서, 쓰다 중단되면 잘린 파일이 완료로 취급됨 (I-66은 grid 캐시만 고침) | 해결·검증: `write_csv_atomic`(임시 이름에 쓴 뒤 교체)로 세 파일과 grid 캐시를 씀. 테스트·H33 (호출 위치 세 곳을 되돌리는 변이는 없음). 기존 파일은 모두 완전함: `test_metrics.csv` v6 3개·v7 커밋본·v7 실행 폴더 모두 머리 + 270행(정책 2 × seed 3 × 모델 9 × K 5), v6 F1 `posthoc_candidate_depth.csv` 9행(후보 수 9개), `if_ablation.csv` 18행(모델 9 × 전처리 2) | ce97790 |
| I-77 | P3 | `stamp_code_check.py`가 `sha256-lf` plan(앞으로 만드는 plan)에서 바뀌지 않은 코드도 불일치로 보고 (스탬프는 원시 바이트 해시) | 해결·검증: `sha256-lf` plan은 plan commit의 파일 내용을 git에서 읽어 plan 해시와 맞춰 본 뒤, 스탬프가 그 내용의 LF 또는 CRLF 해시인지 확인 (`test_stamp_code_check.py`). v7(`sha256-raw`)은 다시 실행해 731/731 일치 (`audit/cycle9/stamp_code_v7.json`) | ce97790 |
| I-78 | P3 | GNN 재학습 스레드의 오류가 "GNN refit jobs failed"로 바뀌어 원인(예: 코드 변경)이 가려짐 | 해결·검증: validation과 재학습 스레드 모두 `start_background`로 실행하고 작업의 오류를 그대로 다시 냄. 테스트·H34 | ce97790 |
| I-79 | P3 | 다른 root와의 구간 겹침을 명목 비율로만 판단. 분할은 이벤트 index에서 잘리므로, 명목상 맞닿기만 하는 fold가 경계 이벤트 1개를 공유할 수 있음 | 해결·검증: 시험 열람 직전 검사는 기록된 실제 구간(타임스탬프, `recorded_test_window`)으로 판단하고, 기록된 구간이 없으면 맞닿음도 겹침으로 봄. 실행 시작 검사는 아직 분할 전이라 명목 비율 그대로. 실제 원본: 분할 규칙으로 계산한 구간이 v6·v7 기록 구간 4개와 같고, F4와 명목상 맞닿는 fold 5개(끝 0.39–0.45)는 모두 F4와 이벤트 1개를 공유하며 시작 검사는 통과, 열람 직전 검사는 거부 (`audit/cycle9/guard_real_data.json`). v6·v7에서 연 구간끼리는 공유 이벤트 없음. 테스트·H29·H30·H32 | ce97790 |
| I-80 | P3 | Cycle 10 문서 감사 5건: §18 마지막 행이 I-74의 "함께 고친 것" 7건을 "함께 확인한 6건"이라 함, `stamp_code_check.py`와 `audit/cycle8/stamp_code_v7.json`이 실제로는 79b628d에 추가됐는데 Cycle 8 커밋 기록·I-70 커밋 열은 b9e7134라 함, 재현 6단계·8단계 명령이 커밋된 Cycle 8·7 증거를 내용이 다른 출력으로 덮어씀(6단계에는 Cycle 9 실행 시간도 없음), Cycle 1 근거 `audit_runs/repro_8020`이 이 PC에 없다는 표시 없음. 원인: 재현 명령과 두 변이 도구의 기본 출력이 커밋된 증거 폴더를 가리킴 | 해결·검증: §18·Cycle 8 커밋 기록·I-70 커밋 열 정정. 재현 5–10단계는 gitignore 대상 `audit_runs/repro/`에 쓰고 비교할 커밋 파일을 적음. `mutation_check.py`·`guard_mutation_check.py`의 기본 출력도 `audit_runs/`로 바꿈. 4e166c5 전에 7–10단계를 실제로 실행해 출력 5개(캐시 확인, 정책별 튜닝 차이, 스탬프, grid 재읽기, 실제 원본 가드)가 당시의 커밋된 증거와 같음을 확인(9단계는 Cycle 9 출력과 비교, 비교 결과는 저장하지 않음. 5·6단계는 Cycle 10 공식·가드 변이 실행으로 대신함. 저장된 비교는 I-85). `repro_8020`은 이 PC에 없다고 표시하고, 같은 28개 검사의 커밋된 결과(rolling fold 4개의 `leakage_check.json`, 모두 28/28)를 적음 | 4e166c5 (도구), Cycle 10 문서 커밋 |
| I-81 | P2 | I-75 수정이 지나치게 막음: 다른 root의 정확한 재현은 같은 fold를 같은 `selection.json` 해시로 열었어야 했는데, 이 파일에는 실행마다 달라지는 값(`locked_utc`, 절대경로 `job_dir`, `seconds`·`train_seconds`)이 들어 있어 새 root에서 원본부터 다시 돌리면 선택이 같아도 해시가 반드시 다름. 실행 시작 검사는 통과하므로 학습을 모두 마친 뒤 시험 열람 직전에 거부됨. 감사자가 합성 데이터의 v7 프로토콜로 재현(같은 plan, 다른 해시, 같은 결정. 감사자 스크립트와 출력은 저장소 밖) | 해결·검증: `selection_decision`이 결정 내용(MF·GNN·Hybrid 백본 설정과 epoch, 정책별 MF 가중과 Hybrid 추론 설정)만 해시하고, 상대 기록의 결정은 그 fold의 `selection.json`이 접근 로그에 적힌 열람 해시와 같을 때만 그 파일에서 읽음. 테스트: 실행별 값만 다르면 통과, γ가 다르면 거부, 기록 파일이 로그와 다르면 거부, 합성 원본으로 두 root를 끝까지 돌려 두 번째가 시험 평가까지 마침(파일 해시는 다르고 결정은 같음). 가드 변이 H36–H38. 실제 기록 5개(v6 3 fold, v7 커밋본, `audit_runs/v7`): 로그 해시가 모두 보관된 `selection.json`과 같고, v7 커밋본과 실행 폴더의 결정이 같음 (`audit/cycle10/guard_real_data.json`의 `decisions`). v6·v7은 Cycle 9 전 코드로 fold마다 한 번씩만 열어 영향 없음 | 4e166c5 |
| I-82 | P3 | 다른 root 기록의 분할 필드(`cutoff_timestamp`, `end_cutoff_timestamp`)가 null·문자열이면 `recorded_test_window`가 `try` 밖에서 `int()`로 바꾸다 파일을 지목하지 않는 예외를 내 같은 `events.csv`를 쓰는 모든 시험 열람이 멈춤 | 해결·검증: 정수가 아니면 기록된 구간이 없는 것으로 보고 맞닿음도 겹침으로 판단. 테스트(null, 문자열 2종, 실수)·H39. 실제 기록 5개는 모두 정수 구간 (`guard_real_data.json`의 `windows`가 Cycle 9와 같음) | 4e166c5 |
| I-83 | P3 | `start_background`: 주 스레드(MF validation·재학습)가 실패해도 GNN 작업 스레드가 남은 작업을 계속 띄워 프로세스가 수 시간 끝나지 않음. 작업이 BaseException으로 끝나면 `KeyError('value')`로 원인이 사라짐. 결과 손실은 없고 Cycle 9 전부터 있던 동작 | 해결·검증: `alongside`가 주 단계가 실패하면 stop을 걸고 스레드를 기다린 뒤 오류를 냄. `run_gnn_jobs`는 stop이나 오류가 나면 새 작업을 띄우지 않고 실행 중인 작업을 종료함(종료된 작업은 재개 때 다시 학습). worker는 BaseException도 보관. 테스트·H40–H42·H44. GNN 작업의 오류는 여전히 MF 단계가 끝난 뒤에 드러남(결과에는 영향 없음) | 4e166c5 |
| I-84 | P3 | `--folds`에 protocol에 없는 이름(오타)을 주면 아무 fold도 돌리지 않고 `completed`로 끝나며, 그 root에 plan이 만들어짐 | 해결·검증: plan을 만들기 전에 알 수 없는 fold 이름이면 멈춤. 테스트·H43. 일부러 없는 fold 이름으로 시작 검사만 돌리던 기존 테스트 2개(수집 3건)는 실제 fold 이름과 `run_fold` 대체로 같은 검사를 하게 바꿈 | 4e166c5 |
| I-85 | P2 | Cycle 11 문서 감사 7건 (P2 1, P3 6): plan 코드 확인 명령이 커밋된 Cycle 3 증거(`audit/plan_code/plan_code_v7.json`)를 다른 내용으로 덮어씀(Cycle 11 감사 시점에 바뀐 파일 2개, 추가된 테스트 9개), 그림 재생성·재계산 명령과 도구 4개의 사용 예도 커밋된 폴더에 씀(I-80이 일부 경로만 고침), §19가 가드 변이 실행 시작 때 `guard_real_data_check.py`도 커밋 안 됐다고 함(기록에는 문서 4개뿐, 그 도구는 실행 중에 고침), "7–10단계 출력 5개가 커밋된 증거와 같음"의 9단계 출력은 4e166c5 전 것이고 Cycle 9 출력과 같음, 저장된 출력이 없는 확인 3곳에 표시 없음, `test_cycle3_consistency.py` 개수가 8개·9개로 다름 | 해결·검증: 재현 4단계·plan 코드 확인·재계산과 도구 4개의 사용 예를 `audit_runs/repro/`로 바꾸고, 기본 출력이 `audit/figures_v6`이던 `plot_rolling_results.py`는 `--output`을 필수로 함. `report_tools/compare_outputs.py`(줄바꿈 무시, JSON은 값 비교, 시간에 따라 바뀌는 출력은 지정한 키만 비교, `test_compare_outputs.py`)를 추가하고 4, 7–10단계·plan 코드 확인·재계산을 8d8d0d5의 도구로 다시 실행해 29행(다시 만든 출력 28개. plan 코드 확인 출력 하나를 Cycle 3·Cycle 6 증거 두 곳과 비교했고, 11단계 명령은 Cycle 3 쪽만 비교해 28행)이 모두 커밋된 증거와 같음 (`audit/cycle11/repro_compare.json`, `repro_run.txt`). 서술 3곳 정정, 저장 안 한 확인 3곳 표시, 개수는 "함수 8개, 수집 9개"로 통일 | 318186b (도구), Cycle 11 문서 커밋 |
| I-86 | P3 | `experiments.clone_protocol`이 hard link를 대상 폴더에 순서 없이(set 순회) 바로 만들어, 중간에 끊기면 일부 파일만 있는 폴더가 남음. `manifest.json`이 이미 링크됐으면 재개 경로(`rolling_protocol`은 `manifest.json` 존재, 옛 스크립트는 폴더 존재만 확인)가 이를 완성본으로 보고 넘어가고, 아니면 다시 복제할 때 "대상이 비어 있지 않음" 오류로 멈춰 손으로 지워야 함. 이 PC의 v7 실행 폴더(`audit_runs/v7`)에서는 protocol 폴더 51개가 모두 manifest의 파일을 다 갖고 있음(v6 실행 폴더는 v6 PC에만 있어 확인 못 함) | 해결·검증: 형제 폴더 `<이름>.partial`에 링크를 만든 뒤 `os.replace`로 한 번에 옮김. 끊긴 `.partial`은 다음 실행에서 지우고 다시 만듦(원본 파일은 건드리지 않음). 테스트(6번째 링크에서 끊은 뒤 대상 폴더가 없고, 다시 실행하면 완성됨)·H45 | 318186b |
| I-87 | P3 | 실행 프로세스가 강제로 끝나면(`taskkill /F`, 작업 관리자) `run_gnn_jobs`의 `try/except`가 돌지 않아 GNN 작업 프로세스가 남아 GPU를 계속 씀. 같은 root로 재개하면 남은 작업과 같은 작업 폴더에 동시에 씀. Popen이 실패하면 로그 파일 핸들도 닫히지 않음 | 해결·검증: Windows에서는 각 작업을 KILL_ON_JOB_CLOSE job object에 넣어 실행 프로세스가 어떻게 끝나든 함께 끝나게 함(`kill_with_parent`, 실패하면 오류). Popen이 실패하면 로그를 닫음. 테스트(실제 부모 프로세스를 강제 종료한 뒤 자식의 heartbeat가 멈춤, 모든 작업이 묶임, 실패한 실행의 로그가 닫힘)·H46–H48. POSIX에는 이 보호가 없음(`reproducibility.md`에 적음) | 318186b |
| I-88 | P3 | `manuscript_alignment.md` 공개 사항 2가 평가 모집단을 "시험 양성 상품이 학습 카탈로그에 있는 사람"이라 해, 코드(필터 전 원본 학습 기간의 사용자·상품 기준)와 다름 | 해결: 필터 전 원본 기준, 학습 벡터가 없는 사용자는 0점, repeat·novel 정답 집합 정의를 코드(`data_protocol.py` `reference`, `policy_evaluation.py`)에 맞춰 다시 씀 | Cycle 11 문서 커밋 |
| I-89 | P3 | 구조도(`audit/figures_v*/pipeline.*`)와 `manuscript_alignment.md` 식 8 행이 후보 생성을 α 결합 점수 상위 n개로만 설명함. 봉인된 설정 8개 중 5개는 `balanced_union`(ALS 상위 ⌈n/2⌉ ∪ BPR 상위 ⌊n/2⌋)이고, 결합은 2개가 RRF | 해결·검증: 구조도 2번 상자와 식 8 행에 두 방식을 적음(설정 수는 커밋된 `selection.json` 4개에서 다시 셈). 그림을 다시 만들어 v6·v7 52개 파일 중 `pipeline.pdf/png/svg` 6개만 바뀐 것을 `compare_outputs`로 확인 | 318186b (구조도 문구), Cycle 11 문서 커밋 |
| I-90 | P3 | 전처리(`UnifiedPreprocessor.run`)가 바깥 `manifest.json`을 `finalize_outer`(원본 행 통계 추가, `dataset_statistics.csv` 다시 쓰기, 원본 입력 해시) 전에 써서, 그 사이에 끊기면 재개(`prepare_fold`는 `manifest.json` 존재만 확인)가 원본 출처 기록과 `raw_valid` 행이 없는 폴더를 완성본으로 쓰거나, 원인과 다른 "Shared input changed" 오류로 멈춤 | 해결·검증: 바깥 manifest는 `finalize_outer`에서 한 번만, `.partial`에 쓴 뒤 이름을 바꿔 씀. 끊기면 `prepare_fold`가 "Incomplete preprocessing output"으로 멈춤. 테스트·H49. 실제 원본으로 v7 F4 전처리를 고친 코드로 다시 해 기존 `audit_runs/v7` 폴더와 비교(§21) | 5026680 |
| I-91 | P3 | `--folds`를 이름 없이 주면(빈 셸 변수 등) 빈 목록이 "제한 없음"으로 처리돼 모든 fold와 사후 분석을 실행함(I-84와 같은 종류) | 해결·검증: `--folds`는 이름이 하나 이상 필요(`nargs='+'`), `run(folds=[])`은 plan을 만들기 전에 멈춤. 테스트(함수와 명령줄)·H50 | 5026680 |
| I-92 | P3 | `compare_outputs.py`가 비교한 파일이 0개여도 `all_equal: true`, 새 출력에만 있는 파일은 보고하지 않음, JSON을 `==`로 비교해 `1`·`1.0`·`true`를 같게 봄, NaN에서 `equal`과 다른 키 목록이 서로 어긋남, JSON이 아닌 파일의 `#keys`를 무시함. 경로를 잘못 주면 재현 비교가 거짓으로 통과할 수 있음 | 해결·검증: 0개면 실패, 새 출력에만 있는 파일은 다름으로 기록, 형까지 비교하고 NaN은 NaN과 같음, 비교 대상이 없거나 JSON이 아닌 파일·폴더에 `#keys`를 주면 오류. 테스트·H52–H54. Cycle 11 비교(`audit/cycle11/repro_compare.json`의 29행)를 새 도구로 다시 하면 I-89 전 그림인 v7 `pipeline.*` 3개만 다르고 나머지는 같음(출력 저장 안 함, §21) | 5026680 |
| I-93 | P3 | GNN 작업 프로세스가 job object에 들어가기 전에 끝나면(venv 런처가 바로 죽는 경우 등) `AssignProcessToJobObject`의 "Access is denied"가 실행 전체를 멈춰 "GNN job failed (see .log)"라는 원래 원인을 가림 | 해결·검증: 들어가지 못한 작업이 이미 끝났으면 그 오류를 넘기고 아래의 종료 코드 확인으로 넘어감. 아직 도는 작업이면 그대로 오류(그 작업은 종료됨). 테스트(끝난 실제 프로세스는 job object에 못 들어감, 두 경우의 동작)·H51. 실행과 할당 사이 약 10ms(Cycle 12 코드 감사자가 잰 값: 중앙값 9.6ms, 최대 16.9ms, 출력 저장 안 함) 동안 실행이 강제 종료되면 작업이 남을 수 있음(공개) | 5026680 |
| I-94 | P3 | (Cycle 12 코드 감사의 미확정 의심을 공개로 처리) 다른 root의 정확한 재현 예외는 같은 plan·코드·원본·결정만 보고 장치(CPU/GPU)와 라이브러리 버전은 보지 않음. 다른 장치에서 재현하면 BPR·GNN 결과가 달라질 수 있음(I-34). 접근 로그에는 장치가 없음 | 공개·한정: 예외는 시험을 보고 선택을 바꿀 수 없게 하는 장치이고, 같은 장치에서도 GPU 학습 결과는 실행마다 조금 다름. 열람은 모두 그 root의 접근 로그에 남고, 장치는 각 root의 `plan.json` `environment`에 남음. 보고하는 수치는 처음 봉인된 실행(커밋본)이며 재현 실행의 수치로 바꾸지 않음. `reproducibility.md`에 적음 | Cycle 12 문서 커밋 |
| I-95 | P3 | Cycle 12 문서 감사 3건: Cycle 11 재현 비교의 "29개 파일"은 29행이고 다시 만든 출력은 28개(plan 코드 확인 출력 하나를 Cycle 3·6 증거 두 곳과 비교, 11단계 명령으로는 28행), `--require-unchanged` 실패 원인과 I-85의 "지금은 바뀐 파일 2개" 목록이 Cycle 11 뒤 낡음(`experiments.py`도 바뀜), Cycle 9 가드 변이 시간 44분 54초는 44분 53초 | 해결: 개수와 비교 대상을 적고, 바뀌는 목록은 도구 출력(`changed_since_plan`, `added_since_plan`)을 보도록 하고 기준 커밋을 적음(5026680에서 다시 실행해 v6·v7 모두 실패, v7은 4개·v6은 5개 파일), 시간 정정. 함께: Cycle 11 코드 감사 항목 중 I-88·I-89가 문서·그림 서술임을 표시, 재현 2단계의 누수 검사 출력을 `audit_runs/repro/`로, `kill_with_parent` 설명을 "남은 작업을 직접 끝냄"으로 바꿈(이전 설명의 "프로세스 그룹 전체를 끝냄"도 맞았음: I-96) | 8feac9b (설명), Cycle 12 문서 커밋 |
| I-96 | P3 | Cycle 13 문서 감사 5건: 재현 6단계가 Cycle 12 가드 변이를 커밋 후 전체 테스트와 동시 실행했다고 함(실제로 겹친 것은 공식 변이 테스트뿐, 커밋 후 테스트는 04:22:38–04:28:42와 05:46:18–05:52:07), `validation_report.md` 머리말의 두 번째 PC 범위가 §9–§19, I-95와 커밋 8feac9b 메시지가 "작업은 별도 그룹으로 띄우지 않음"을 정정 이유로 듦(작업은 실행의 그룹을 이어받고 `launch_detached.py`는 POSIX에서 실행을 새 세션으로 띄우므로 이전 설명도 맞았음), "약 10ms"에 근거 표시 없음, 저장 안 한 확인 3종 5곳(145/145 두 곳, Cycle 11 비교 재실행 두 곳, 가드 테스트 15개)에 표시 없음 | 해결: 동시 실행 목록·범위 정정, I-95 문구 정정과 POSIX 안내 보강(그룹 전체를 끝내면 작업도 끝남, 실행만 끝냈으면 남은 작업을 직접 끝냄). 커밋 메시지는 고치지 않음(이력을 다시 쓰지 않음). 10ms는 감사자 측정값(중앙값 9.6ms, 최대 16.9ms, 저장 안 함)으로 표시, 저장 안 한 확인에 표시 | Cycle 13 문서 커밋 |
| I-97 | P3 | Hybrid의 `hybrid_at_grid_boundary`에 α 축이 없고, 후보 수 끝을 `CANDIDATE_KS`의 처음·끝(500, `all`)으로만 봄. balanced_union에는 `all`이 없어 최대 20,000이 끝으로 표시되지 않음. 그 결과 v7 F4 novel(α .75, balanced_union, 20,000)의 끝 축이 `validation_report.md` §12.2에 "-"로 보고됨. v6 F2 novel, F3 repeat의 α .75(최대)도 기록에서 빠짐(`v7_feasibility.md` §4) | 해결·검증: 선택된 전략의 grid에서 축별 끝을 판단하는 `hybrid_at_grid_boundary` 함수(α 포함). 커밋된 `selection.json` 8개 정책에 새 규칙을 읽기 전용으로 적용(`audit/cycle13/hybrid_boundary_recheck.json`): 위 세 곳만 달라지고 선택값은 그대로. 커밋된 `selection.json`은 고치지 않고 보고서 표와 설계 문서를 정정. 선택 결정 해시(`selection_decision`)에는 이 플래그가 없어 재현 판정과 무관. 테스트, 변이 H60 | cb22505 |
| I-98 | P3 | `compare_outputs.py`의 I-92 수정이 불완전: `#keys`에 없는 키를 주면 양쪽 `None`으로 같다고 판정, 최상위가 리스트인 JSON은 `AttributeError`, 커밋 폴더가 빈 쌍이 다른 쌍의 파일 때문에 `all_equal: true`가 될 수 있음 | 해결·검증: 키가 어느 한쪽에라도 없거나 JSON 객체가 아니면 오류, 커밋 파일이 없는 폴더 쌍은 `pairs_without_committed_files`에 기록하고 `all_equal: false`. 실제 재현 출력에 문서의 비교 명령을 다시 실행해 Cycle 12와 같은 판정(I-89 전 v7 `pipeline.*` 3개만 다름, `audit/cycle13/compare_repro.json`). 테스트, 변이 H55–H57(H52는 바뀐 줄에 맞춰 대상 수정) | cb22505, db3653e |
| I-99 | P3 | `check_score_precision.py`가 "내부 validation만"이라고 하면서 역할을 확인하지 않아, 바깥 시험 폴더를 주면 봉인 가드 없이 시험 정답으로 지표 차이를 계산함 | 해결·검증: 지표를 계산하기 전에 manifest 역할이 `validation`이 아니면 거부(그 전에 `require_protocol`이 manifest에 적힌 파일의 해시만 확인함. 시험 정답 파일도 해시 계산으로 읽히지만 내용은 쓰지 않음). 실제 폴더: v7 F4 `prepared`·`runs/seed_42`(역할 test) 거부, `val_mf`로 다시 실행한 결과가 Cycle 4 커밋본과 같음(`audit/cycle13/precision_role_check.txt`, `score_precision_F4_val_c13.json`). 테스트, 변이 H58 | cb22505 |
| I-100 | P3 | `use_protocol('v6')`가 빈 dict라 같은 프로세스에서 v7 뒤에 부르면 v7 grid·fold·plan 값이 남음(실제 실행은 프로세스마다 protocol 하나라 영향 없음) | 해결·검증: v6 항목에 v7이 바꾸는 모든 키의 v6 값을 저장. 실제 커밋 plan: v6, v7, v7→v6, v6→v7→v6→v7 순서 모두 마지막 protocol의 plan 해시가 커밋된 `plan.json`과 같음(`audit/cycle13/use_protocol_check.txt`). 테스트, 변이 H59 | cb22505 |
| I-101 | P3 | plan이 Hybrid grid의 크기(`hybrid_grid_size`)와 손으로 적은 축 목록만 기록해, 크기가 같고 값이 다른 grid(예: γ 1.5→3.0)도 같은 plan 해시가 됨(감사자가 임시 사본에서 재현). 코드 해시 가드가 코드 변경을 따로 잡으므로 v6·v7 결과 영향 없음 | 해결·검증: 축을 `HYBRID_GRID`에서 grid 순서대로 뽑아 기록. v6·v7 plan 내용과 해시가 커밋된 `plan.json`과 그대로 같음(`audit/cycle14/plan_hash_check.txt`, HEAD 41ba20d). 크기가 같고 값이 다른 grid는 해시가 바뀌고, 커밋된 두 plan의 축으로 다시 만든 grid가 `HYBRID_GRID`와 같음(테스트), 변이 H61. 수정이 불완전했음(후보 깊이 값·짝·순서는 여전히 같은 해시) → I-104 | 41ba20d |
| I-102 | P3 | Cycle 14 문서 감사 4건: I-99 "파일을 읽기 전에 거부"(실제로는 `require_protocol`이 manifest의 파일 해시를 먼저 확인), Cycle 13 비교 재실행을 "커밋 cb22505의 도구"로 적음(실제로는 커밋 48초 전 작업 폴더), 가드 변이 동시 실행 목록에서 Cycle 8–10의 커밋 후 테스트와 Cycle 13 공식 변이 테스트의 겹침 누락, 등록부 머리말 목록에 Cycle 13 없음. 같이 확인한 의심 2건도 맞아 정정: Cycle 13 가드 첫 실행이 cb22505 커밋 전(06:06:59 UTC) 작업 폴더에서 시작, 가드 재실행과 겹친 코드 감사는 Cycle 14 것. I-96의 "4곳"은 3종 5곳 | 해결: 문구와 시각 정정(증거 시각은 각 출력 파일의 시작·종료 시각과 커밋 시각. Cycle 13 가드 첫 실행 시작 06:06:59와 작업 폴더 전체 테스트 06:07–06:13은 출력이 남지 않아 셸 기록의 시각이며 그렇게 표시함, I-103) | Cycle 14 문서 커밋 |
| I-103 | P3 | Cycle 15 문서 감사 2건: Cycle 13 가드 첫 실행 시작(06:06:59 UTC)과 작업 폴더 전체 테스트(06:07–06:13) 시각은 커밋된 출력이 없는데 표시 없음, 공식 변이 테스트(재현 5단계)를 "다른 GPU 작업이 없을 때 실행"이라 하면서 Cycle 8–10·12 재실행이 커밋 후 전체 테스트·가드 변이와 겹친 것을 적지 않음(I-102 정정이 가드 변이 쪽만 고침) | 해결: 두 시각에 "셸 기록의 시각, 출력 저장 안 함" 표시, 5단계의 "다른 GPU 작업 없음"을 Cycle 3 기준 실행으로 한정하고 재실행별 겹침을 각 출력 파일·커밋 후 테스트 출력의 시각으로 적음 | Cycle 15 문서 커밋 |
| I-104 | P3 | I-101 수정이 불완전: plan의 축과 `candidate_ks`만으로는 grid가 고정되지 않아, grid 안의 후보 깊이 값(예: 500→300), 전략과 후보 깊이의 짝(예: 20000을 blend 전용으로, `all`을 두 전략 모두에), 순서만 바꾼 grid(`best_row`의 동점 처리를 정함)도 봉인된 plan 해시가 됨(감사자가 임시 사본에서 재현). 코드 해시 가드가 코드 변경을 따로 잡으므로 v6·v7 결과 영향 없음. v7 F4 novel의 최고점은 balanced_union 20,000과 blend `all`이 동점(validation 카탈로그 22,760개라 두 후보 집합이 사실상 같음)이고 grid 순서로 앞의 것이 선택됨(감사자 확인, 저장 안 함) | 해결·검증: grid가 봉인된 v6·v7 grid와 다르면 순서 있는 grid 전체의 sha256(`hybrid_grid_sha256`)을 plan에 추가. 봉인된 grid는 예전 형식 그대로라 v6·v7 plan 내용과 해시가 커밋된 `plan.json`과 같음(`audit/cycle15/plan_hash_check.txt`, HEAD 4c1ddfb). 세 가지 변경 모두 v6·v7에서 해시가 바뀜(테스트 6개), Cycle 14 테스트에 기록된 축이 grid를 따르는지 추가(H61이 살아남지 않게). 변이 H62 | 4c1ddfb |
| I-105 | P2 | Cycle 16 문서 감사: 다섯 문서(6곳, 원고용 권장 문구 포함)가 v7을 "모든 모델군 grid 확장"으로 서술. 실제로는 ALS·BPR·GNN만 넓혔고 Hybrid grid(2,808개)와 ALS+BPR α(11개)는 v6와 같음(두 plan, 하나의 봉인 grid sha256). v6·v7 모두 Hybrid 선택이 grid 끝에 걸렸는데 남은 한계에는 학습 모델군의 끝만 적힘 | 해결: 6곳을 "ALS·BPR·GNN grid 확장(Hybrid grid·ALS+BPR α는 v6와 같음)"으로 고침, `validation_report.md` §12.5와 I-23 남은 한계에 v7 Hybrid의 끝 축과 "baseline을 충분히 튜닝해도"가 Hybrid 결합 설정의 충분한 탐색을 뜻하지 않음을 더함, `v7_feasibility.md` §4에 Hybrid grid를 넓히지 않았음을 적음. 수치 변화 없음 | 0a0ca40 |
| I-106 | P2 | Cycle 16 코드 감사: `validation_report.md` §8 항목 2가 "F2/F3가 독립적인 확인", I-01 수정 설명이 "F2/F3는 처음 쓰는 시험 구간"이라고 함. F2 시험 구간(이벤트 0.60–0.80)의 80%(0.64–0.80)는 v1–v5에서 설정을 고를 때 쓴 내부 validation 구간이라(`rolling_protocol.py`의 F2 설명, `v7_feasibility.md` §1 표) v1–v5의 고정 설계와 독립적이지 않음 | 해결: §8 항목 2, §3 표의 F2 행, I-01 수정 설명, `manuscript_alignment.md`의 단일 분할 행을 "F2는 시험으로는 처음 열었지만 대부분 v1–v5 validation 구간, v6 전에 평가·선택에 쓰인 적 없는 시험 구간은 F3·F4"로 고침(Cycle 17: F3 시험 구간의 80%가 v6 F2의 validation이었음을 더함, I-109). 수치 변화 없음 | 0a0ca40 |
| I-107 | P3 | Cycle 16 문서 감사 3건: 재현 5단계의 49분 38초가 Cycle 3 기준 실행 시간처럼 적힘(실제는 Cycle 4 재실행, Cycle 3 기준 실행은 시각 출력 없음), 공식 변이 재실행의 겹침 목록에 Cycle 11(재현 실행과 겹침)이 빠지고 Cycle 4는 근거 없이 "겹침"에 포함, `cycle_log.md`의 Cycle 13 가드 첫 실행 결과에 로그 미보존 표시 없음 | 해결: 5단계에 각 시간의 출처(70f5106의 Cycle 3 기록, 6364ad4부터 적힌 Cycle 4 재실행 시각)와 Cycle 4 겹침 기록 없음, Cycle 11 겹침, 독립 감사 에이전트의 동시 실행 가능성(시각 저장 안 됨)을 적음. `cycle_log.md`에 미보존 표시 | 0a0ca40 |
| I-108 | P3 | Cycle 16 코드 감사: `git_state`의 dirty 검사 면제가 슬래시 없는 접두사(`audit_runs`)라, 커밋 안 된 `audit_runs_patch.py` 같은 코드 파일도 빠져 사전등록 실행 전 거부가 일어나지 않음(감사자가 porcelain 출력을 바꿔 재현). v6·v7 실행 때 그런 파일은 없었음 | 해결: 면제를 `audit_runs/` 폴더로만 한정, `test_cycle16_guards.py`, 가드 변이 H63. 실제 저장소에서 커밋 안 된 코드·문서가 모두 나열되고 v6·v7 plan 해시 그대로 (`audit/cycle16/git_state_and_plan_check.txt`). 수정이 불완전했음(출력 문자열로 거르는 방식이라 git 설정·이름 변경에 따라 코드를 놓침) → I-110 | 6ce754d |
| I-109 | P3 | Cycle 17 문서 감사 6건: I-106 수정이 "평가·선택에 쓰인 적 없는 시험 구간은 F3·F4"라 했으나 F3 시험 구간(0.45–0.60)의 80%는 같은 v6 실행에서 F2의 내부 validation(F2 설정 선택)이었음, 등록부 머리말 발견 목록에 Cycle 16 누락, 가드 변이 도구 docstring에 Cycle 16 테스트 누락, `professor_briefing_ko.md`·`MEETING_NOTES_KO.md`(와 PDF)에 감사 후 안내 없음(I-37은 README만 고침), `validation_report.md` §1 Cycle 16 행에 "→ §25" 없음, `v7_feasibility.md` §4 표에 v6 Hybrid GNN backbone의 lr 끝(F1·F3 .003) 없음 | 해결: F3 서술을 "v6 전에 평가·선택에 쓰인 적 없음, F3 시험의 80%는 v6 F2의 validation이었지만 F3 자신의 학습·선택에는 안 들어감"으로 한정, 나머지를 고침. 두 md 파일 맨 앞에 안내를 붙이고 원문은 보존. PDF는 수정하지 않고 `README_KO.md`에 안내가 없다고 적음, v5 배포 매니페스트의 md 해시가 지금 파일과 맞지 않음도 적음(PDF는 일치) | Cycle 17 문서 커밋 |
| I-110 | P3 | Cycle 17 코드 감사: I-108 수정이 불완전. `git_state`가 `git status` 출력 문자열에서 `audit_runs/`로 시작하는 줄을 빼는 방식이라, 사용자 설정 `status.showUntrackedFiles=no`이면 추적 안 된 새 코드가 아예 나오지 않고, `audit_runs/`에서 코드 경로로의 이름 변경 줄(`R  audit_runs/x.py -> newcode2.py`)은 통째로 면제되며, 따옴표로 싼 `audit_runs/` 경로는 면제되지 않아 실행이 거부됨(감사자가 임시 저장소에서 재현). v6·v7 plan의 `dirty_files`는 둘 다 비어 있고 plan 코드 해시가 커밋과 같음 | 해결·검증: `git status --porcelain --untracked-files=normal -- . ':(exclude)audit_runs'`로 git이 직접 빼게 함. 실제 git 저장소로 하는 테스트(`test_cycle16_guards.py`를 실제 git으로 바꿈, `test_cycle17_guards.py`), Cycle 5 상태 열 테스트는 git이 이제 돌려주는 출력과 제외 pathspec을 요구하도록 바꿈(첫 전체 실행에서 이 테스트가 옛 출력 형식 때문에 실패, `audit/cycle17/precommit_pytest_first_run_1_failed.txt`). 가드 변이 H63(제외 없음)·H64(`--untracked-files` 없음). `--no-renames`도 넣었다가 제외 pathspec만으로 이름 변경이 짝지어지지 않아(변이가 살아남음) 뺐음. 실제 저장소에서 `git_state()`가 일반 `git status`와 같은 목록을 내고 `audit_runs/`는 나오지 않음, v6·v7 plan 해시 그대로 (`audit/cycle17/git_state_and_plan_check.txt`) | cf1a3fe |
| I-111 | P3 | Cycle 17 코드 감사: `run_gnn_jobs`가 job spec 경로를 `Path(root) / 'job_specs'`(상대 경로 그대로)로 GNN 하위 프로세스에 넘기는데 하위 프로세스는 프로젝트 폴더에서 실행됨. 다른 폴더에서 상대 `--root`로 실행하면 모든 GNN job이 spec을 찾지 못해 실패하고, 오류는 MF 단계가 끝난 뒤에야 드러남(감사자가 재현). 문서의 실행 위치는 프로젝트 폴더라 v6·v7 영향 없음 | 해결·검증: spec 폴더를 절대 경로로 바꿈, `test_cycle17_guards.py`(다른 폴더에서 상대 root로 실행해 넘긴 경로가 절대 경로이고 파일이 있음), 가드 변이 H65 | cf1a3fe |

---

## I-01 (P0) 시험 데이터 재사용

- **위치**: `robust_hybrid.py` 50행 (수정 전), `results/hybrid_joint_8020_v5/experiment_plan.json` (`historical_test_seen: true`, `prior_test_success: false`).
- **재현**: v3 `result.json`의 `success: false`를 확인한 다음 v4/v5 계획이 작성됨. v5 backbone(`l2_uniform`, `l3_uniform`, `l2_mf_mixed`)은 v3 validation 상위 3개와 다름.
- **근본 원인**: 수치 선택은 validation으로 했지만, 탐색 공간(후보 수 최대 20,000, backbone, 독립 GNN 비교 대상)은 시험 결과를 본 뒤 사람이 정함.
- **영향**: v5의 시험 결과는 selection bias가 있어 일반화 성능 추정치로 볼 수 없음.
- **수정**: `rolling_protocol.py`(v6). 모든 grid·fold·seed·선택 규칙을 실행 전에 `plan.json`에 고정하고 해시로 잠금. fold마다 시험을 한 번만 열고 `test_access_log.jsonl`에 기록. plan이 바뀌면 같은 fold를 다시 열 수 없음. F1은 v1–v5 시험 구간 그대로(이미 노출됐다고 명시), F2/F3는 시험으로는 처음 여는 구간(단 F2 시험 구간의 이벤트 80%는 v1–v5에서 설정을 고를 때 쓴 내부 validation 구간이라 설계와 독립적이지 않음, I-106).
- **회귀 테스트**: `test_access_log_blocks_reopening_under_new_plan`, `test_rolling_protocol_end_to_end`(재실행해도 시험 접근이 추가되지 않음, plan 변경 시 거부), `test_plan_freezes_grids_and_rules`.
- **기록 보존**: v5 결과와 코드는 그대로 둠. `robust_hybrid.py`에 `BACKBONE_PROVENANCE`를 추가해 시험을 본 뒤 정한 선택임을 기록.

## I-02 (P0) 데이터 누수

- **검증 방법**: `audit_tools/leakage_check.py`가 원본 CSV에서 독립적으로 다시 계산해 전처리 결과와 비교. 검사 항목은 cutoff, core 고정점, 평가 사용자, 정답, 속성 시점, `available`, 카탈로그, 내부 validation 구간, GNN 간선.
- **결과**: 80:20 no-IF 분할에서 28개 항목 모두 통과 (`audit_runs/repro_8020`, gitignore 대상으로 Cycle 1을 실행한 PC에만 있고 이 PC에는 없음. 같은 28개 검사의 커밋된 결과는 rolling fold마다 `audit/results_v*/<fold>/leakage_check.json`, 4개 모두 28/28). v6의 모든 fold와 v7 F4도 실행 시작 시 같은 검사를 통과해야 진행되며, F1–F3와 F4 모두 28/28 통과함.
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

v6 `selection.json`과 `validation_gnn.json`의 `job_dir`는 실행 PC의 절대경로임. `refits.json`은 상대경로를 씀. 이 실행의 코드 해시 잠금을 지키기 위해 고치지 않음 (Cycle 8 전에는 코드를 실행 시작과 재개 때만 확인했고, b9e7134부터 GNN 작업 시작과 시험 열람 때마다도 확인함. I-70).

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
- 남은 한계: v6 표의 ALS·ALS+BPR 수치는 고칠 수 없음. v7에서도 ALS factors·α, BPR factors, GNN layers·dim·lr이 끝에 걸렸고, 사전등록 규칙대로 더 넓히지 않음. Hybrid grid는 v6와 같게 두었고(v6에서도 끝에 걸린 축이 있었음), v7에서도 repeat 후보 수(최소), novel α(최대)·후보 수(balanced_union의 최대)가 끝에 걸렸다. F4 한 fold, 시험 사용자 repeat 413명 / novel 306명이라 0.01 이하의 차이는 판정할 수 없음.

## I-24 (P3) BLAS 스레드 수

`implicit` ALS는 OpenBLAS 스레드 수에 따라 부동소수점 누적 순서가 달라짐. 24스레드와 1스레드의 결과가 달랐음.

- v6·v7 시점의 `rolling_protocol.py`는 numpy import 전에 `OPENBLAS/OMP/MKL/NUMEXPR_NUM_THREADS=1`을 `setdefault`로 설정함. 사용자 환경에 이 변수들이 다른 값으로 이미 있으면 그 값이 우선함.
- v6가 1스레드였다는 근거는 다른 PC에서 1스레드로 다시 학습한 F3 ALS grid 24행이 비트 단위로 같다는 것(`validation_report.md` §9)이다. v7의 실제 스레드 수도 기록되지 않았고(실행한 launcher는 남아 있지 않음), F4 seed 42 ALS가 1스레드 재학습과 비트 단위로 같고 24스레드와는 다르다는 것이 근거다 (Cycle 5, `audit/cycle5/determinism_v7_blas*.csv`).
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
