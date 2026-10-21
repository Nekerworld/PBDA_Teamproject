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
- 결정성 검사 재실행: ALS·BPR 비트 단위 동일. GNN은 임베딩 최대 차이 1.1e-4, Top-50 목록 99.7% 동일.
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
