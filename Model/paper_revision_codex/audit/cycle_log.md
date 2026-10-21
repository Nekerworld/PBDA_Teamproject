# Cycle log

각 사이클은 A 분석 → B 문제 식별 → C 수정 → D 검증 → E 커밋 → F 커밋 후 검증 → G 처음부터 재감사 순서로 진행함.

## Cycle 1 (2026-10-20)

**A. 분석**
- 코드를 정독함: 전처리, 학습, 평가, 선택, 보고 스크립트.
- 원본 RetailRocket으로 80:20 분할을 새 환경에서 재현함 (`audit_runs/repro_8020`). 핵심 파일은 바이트 단위로 같음.
- v1–v5 계획과 결과 파일로 실험 이력을 재구성함.

**B. 식별**: I-01 – I-17, I-19 – I-22 (`issue_register.md`).

**C. 수정**
- `learning.py`: BPR learning rate, epoch 체크포인트.
- `model.py`: GNN validation 기준 옵션, 엄격한 체크포인트 로딩.
- `improve_hybrid.py`: 엄격한 체크포인트 로딩.
- `robust_hybrid.py`: backbone을 명시적 인자로 바꾸고 출처를 기록.
- 새 파일:
  - `audit_tools/leakage_check.py`
  - `policy_evaluation.py`
  - `stats_tools.py`
  - `rolling_protocol.py`
  - 테스트 2개

**D. 검증**
- 원본 데이터 누수 검사 28/28 통과.
- 새 테스트 16개 추가, 전체 62개 통과.
- 통합 smoke 실행에서 I-18(novel 정책 후보 0개)을 발견해 고친 뒤 통과.
- 코드를 다시 읽어 결함 4개를 고침 (모두 커밋 전):
  - novel 정책 backbone 제한 로직이 동작하지 않던 코드
  - 로그 파일을 상위 폴더 생성 전에 엶
  - GNN 예산에서 연장 실행이 빠짐
  - 상한 규칙이 `>` (v5의 190/200 패턴을 놓침)

**E. 커밋** (`dev`)
- 0af9a3c: BPR/GNN 옵션, 엄격한 로딩
- 63145d7: robust_hybrid 출처
- 46fe0e0: 누수 검사 도구
- 29ba891: v6 프로토콜, 평가 정책, 통계, 테스트

**F. 커밋 후 검증**
- 작업 트리가 깨끗한 것을 확인함.
- `test_v6_protocol.py` 15개를 다시 실행해 통과함.

**실제 데이터 실행**
- `rolling_protocol.py run --root audit_runs/v6` 실행.
- F1 누수 검사 통과.
- 첫 실행은 셸 세션 정리로 외부에서 종료됨 (Python 오류 없음).
- WMI로 셸과 분리된 프로세스로 다시 실행함. 완료된 산출물은 재사용하고, 끊긴 GNN 작업 4개는 처음부터 다시 학습함.
- 같은 plan 해시이고 시험은 아직 열리지 않음.

**중단 (17:30 KST, 사용자 요청)**
- 진행 상황: F1 MF 선택 완료, GNN validation 4/8 완료 (나머지 4개는 학습 도중 중단).
- 시험 데이터는 열리지 않음 (접근 로그 없음).
- 중간 산출물은 `audit_runs/v6`에 보존됨 (gitignore 대상).
- `python rolling_protocol.py run --root audit_runs/v6 --raw ../../data`로 다시 시작할 수 있음.
- Cycle 1의 F(실제 데이터 검증)와 G(재감사)는 끝나지 않음.

**관찰 (plan 고정 이후라 바꾸지 않고 보고만 함)**: F1 내부 validation에서 ALS는 factors 256(상한), α 10(상한), iterations 32(하한)를, BPR은 256 epoch(상한)를 선택함. baseline도 grid 경계에서 선택됨 → `validation_report.md`의 한계 절.

**커밋**: 074443e (중단 시점의 감사 문서).

## 재개와 v6 완료 (2026-10-21)

- 사용자 요청으로 074443e에서 재개함. 작업 트리가 깨끗하고 코드 해시가 plan과 같은 것을 확인한 뒤 같은 명령으로 이어서 실행함.
- 백그라운드 실행이 셸의 Ctrl+C로 다시 끊김 → `CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB`로 분리된 프로세스로 실행해 끝까지 완료함 (exit 0).
- 시험 열람 5회: 세 fold의 `sealed_test` 각 1회, F1 사후 분석 2회(`posthoc_candidate_gamma`, `posthoc_if_ablation`). 모두 같은 plan 해시와 git commit `074443e`.

## Cycle 2 (2026-10-21): 결과 검증과 재감사

**A. 분석**
- fold별 누수 검사, 모집단, 선택 기록, 접근 로그, 보고 CSV를 원본 데이터와 대조함.
- 구 평가기와 v6 평가기의 지표를 비교함: 48개 비교 모두 차이 1e-16 이하.
- novel 목록에서 이미 본 상품을 원본으로 다시 셈: 1,010,100개 중 0개.
- (위 두 검사는 v6 PC의 실행 폴더에서 했고 결과 파일은 저장소에 없다. Cycle 4 문서 감사에서 "저장소만으로는 재확인 불가"로 표시함.)

**B. 식별**
- I-23: MF grid 경계 선택. ALS가 경계 밖에서 validation +0.011 – +0.027.
- I-24: ALS가 BLAS 스레드 수에 따라 달라짐.
- I-25: `determinism_check.py`의 import 순서 결함.
- I-26: 후보 수가 fold마다 다름.
- 재감사 중 같은 import 순서 결함이 `mf_boundary_probe.py`에도 있음을 발견함.
  - 증상: probe가 다시 학습한 봉인 ALS 설정의 점수가 봉인된 validation 점수와 달랐음 (F3 −0.0013).
  - 별도 결함: probe의 ALS 설정이 하드코딩되어, F2에서는 봉인된 설정(reg .2)이 아닌 reg .02를 기준으로 탐색함.

**C. 수정**
- 두 스크립트 모두 numpy import 전에 스레드 환경변수를 설정함.
- probe는 fold별 봉인된 ALS 설정을 기준으로 한 축씩 바꾸도록 고침.

**D. 검증**
- 결정성 검사 재실행(ALS만, `determinism_mf_blas1.csv`): ALS 비트 단위 동일. BPR(비트 단위 동일)과 GNN(임베딩 최대 차이 1.1e-4, Top-50 목록 99.7% 동일)은 첫 검사(`determinism_run1_blas24.csv`) 값이다. 둘 다 GPU 학습이라 BLAS 스레드 결함과 무관해 다시 돌리지 않았다.
- probe 재실행: 봉인된 설정 행이 봉인된 validation 점수를 그대로 재현함.
- 첫 실행 결과는 `*_run1_blas24.csv`로 이름을 바꿔 보존함.
- 전체 테스트 62개 통과.

**E. 커밋** (`dev`)
- 6326c5a: 결정성 검사, MF 경계 탐색 스크립트
- 6ae2c67: v6 결과 요약과 SHA256 목록
- 418e7dd: 결과 파일을 바이트 그대로 저장 (I-27)
- 감사 문서 커밋 (이 문서 포함)

**F. 커밋 후 검증**
- 커밋된 파일을 `SHA256SUMS.txt`와 대조하다가 60개 모두 불일치를 발견함 (I-27: 줄바꿈 자동 변환). 고친 뒤 60/60 일치.
- `plan.json`의 코드 해시 25개가 현재 코드와 모두 같음 (v6 이후 실험 코드는 바뀌지 않음).

**G. 최종 재감사**
- 문서의 수치를 CSV·JSON과 다시 대조함 (K=50 표, 선택 표, 경계 탐색, epoch, 모집단).
- 고친 서술 3개 (코드 문제는 아님):
  - "Hybrid 12칸 중 9칸 1위"는 baseline 7종 대비라는 조건을 붙임.
  - `Hybrid_ALS_only`는 GNN이 없는 ablation이 아니라 BPR만 뺀 ablation임을 명시함.
  - "fold당 1회 열람"을 "봉인 평가 1회 + F1 사후 분석 2회"로 고침.
- Hybrid backbone이 F1/F3에서 단독 GNN과 다르다는 점을 추가함.
- 새 코드 결함은 발견되지 않음. 남은 미해결 사항은 I-23(사용자 결정 필요), I-17/I-20(환경·봉인 산출물), I-21/I-22(원고 수정).

## Cycle 3 (2026-10-21): I-23 확인 실험(v7), 다른 PC 재현, 테스트 검출력

**A. 분석**
- I-23을 F1–F3를 다시 열지 않고 확인할 방법을 검토함 (`v7_feasibility.md`). 시험·validation 점수를 본 적 없는 구간은 이벤트 앞 36%뿐임.
- 다른 PC(RTX 3090 Ti)에 저장소를 새로 clone하고 저장소 밖에 깨끗한 venv를 만들어 재현을 확인함.
- 테스트가 결함을 실제로 잡는지 확인하려고 변이 테스트를 설계함 (알려진 결함 17종).

**B. 식별**
- I-28, I-29 (pytest 수집 범위, v7 구간 테스트가 비율만 비교). I-27(커밋된 결과 줄바꿈)은 Cycle 2 커밋 후 검증에서 찾아 418e7dd로 고쳤다.
- I-30 (핵심 성질을 고정하는 테스트 부재), I-31 (분리 실행 시 Fortran 런타임 중단)
- I-32 (원고용 그림이 v5 기준), I-33 / I-34 (GPU·torch 빌드, CPU/GPU 차이)
- v7 완료 후: I-35 (그림 제목 하드코딩), I-36 (`requirements.txt`와 실행 환경 불일치, launcher 미추적), I-37 (첫 안내 문서들이 v5 주장을 그대로 제시)

**C. 수정**
- v7 프로토콜 추가 (모든 모델군 grid 확장, F4 한 fold). 사용자 승인 후 사전등록.
- `pytest.ini`, v7 구간을 원본 타임스탬프로 확인하는 테스트.
- 보조 스크립트: `plot_rolling_results.py`, `cross_machine_repro.py`, `device_check.py`, `mutation_check.py`, `recompute_test_metrics.py`, `launch_detached.py`.
- `test_cycle3_consistency.py` (8개). v7 실행 중에는 코드 해시 잠금 때문에 추가하지 않고, v7 완료 후 추가함.
- `requirements-v7-lock.txt`, 안내 문서 맨 앞의 감사 안내 (원문 보존).

**D. 검증**
- v7 실제 데이터 실행 완료 (11:16 – 13:10 UTC). 시험 열람 1회, 선택 고정 이후. 실행 후 코드 해시가 plan과 같음.
- v7 지표를 원본 `events.csv`와 저장된 추천 목록으로 독립 재계산함: 270행 최대 차이 1e-16, novel 위반 0건.
- 다른 PC: v6 결과 60/60 SHA256 일치. F3 전처리·누수 검사·ALS grid 비트 단위 동일, BPR 최대 0.0047 차이.
- CPU와 GPU: 같은 seed라도 다른 모델이 학습됨 (집계 차이 0.001 – 0.003).
- 회귀 테스트 78/78 (v7 venv, lock 파일로 새로 만든 venv 둘 다). M04 테스트 추가 후 두 venv 모두 79/79.
- 변이 테스트 (공식, v7 완료 후): 기존 테스트는 17개 중 11개를 놓침.
  - 첫 공식 실행에서 새 테스트를 더해도 M04 하나가 살아남음. 원인은 합성 데이터에서 core 필터가 이력을 지우지 않아 M04가 정답 코드와 구별되지 않은 것.
  - 전용 테스트를 추가하고 다시 실행해 17/17을 잡음.
  - v7 실행 중 예비 실행에서 "기존 테스트가 M04를 잡음"으로 나온 기록은 재현되지 않아 폐기함 (`validation_report.md` §11).
- 그림: v6/v7 두 번 생성해 바이트 동일.

**E. 커밋** (`dev`)
- 1170ea5: pytest 수집 범위, v7 구간 타임스탬프 테스트 (v7 plan 고정 commit)
- a2c5445: v6 그림·표 생성기와 결과
- e7f0240: 다른 PC 교차 재현
- 868b714: CPU/CUDA 비교
- dea0f20: v7 결과 요약, 그림, 독립 재계산
- e9538c8: 환경 lock 파일, 분리 실행 launcher, 안내 문서의 감사 안내
- 이 문서와 함께: `test_cycle3_consistency.py`, `mutation_check.py`, 변이 테스트 결과, 감사 문서

**F. 커밋 후 검증**
- `git archive`로 꺼낸 `results_v7` 28/28, `results_v6` 60/60이 `SHA256SUMS.txt`와 일치.
- 새 산출물 폴더가 모두 추적됨 (gitignore에 걸리지 않음).

**G. 재감사 (v7 완료 후)**
- v7 절차: 접근 로그 1회, 선택 고정 이후, 코드 해시 일치, 상한 연장 규칙이 상위 3개에만 적용됨을 코드로 확인.
- 보고 경로: fold가 하나일 때 시간 변동 표준편차는 빈칸(NaN)으로 남아 값이 지어지지 않음. 생성된 표가 요약 CSV와 일치.
- 문서 수치를 CSV·JSON과 다시 대조하다가 고친 서술: CI 반폭 범위, Recall 0 사용자 비율이 모델별 값이라는 점, ALS 최하위가 repeat NDCG에 한정된다는 점, seed 표준편차 최솟값, 변이 테스트 개수(문서마다 8/9/10으로 달랐음 → 공식 실행 11).
- 새로 찾아 고친 것: I-35, I-36, I-37, M04 테스트 공백(I-30에 포함).

**F. 커밋(70f5106) 후 검증에서 추가로 찾은 것: I-38**
- 현재 코드 해시를 v6/v7 plan과 비교했더니, v6 이후 바꾸지 않은 파일 7개까지 불일치함 (Cycle 4에서 다시 계산: v6 plan 대비 원시 바이트 불일치 9개 중 `plan_code_v6.json`의 변경 목록에 없는 파일 7개).
- 원인: plan 코드 해시가 작업 폴더의 원시 바이트 기준이라 checkout의 줄바꿈 변환에 따라 달라짐 (v6 plan은 모두 CRLF, 이 PC의 checkout은 일부 LF).
- 수정: 새 plan은 줄바꿈을 정규화해 해시하고 방식을 기록함. 기존 plan은 원시 바이트로 비교. `verify_plan_code.py`로 v6 25개·v7 26개 해시가 각 plan commit의 git 내용과 모두 일치함을 확인하고, 조작한 해시 1개를 검출함.
- 테스트 3개 추가. 기존 plan 분기를 일부러 망가뜨리면 테스트가 실패함을 확인한 뒤 복원함.
