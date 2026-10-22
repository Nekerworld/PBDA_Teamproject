# Cycle log

각 사이클은 A 분석 → B 문제 식별 → C 수정 → D 검증 → E 커밋 → F 커밋 후 검증 → G 처음부터 재감사 순서로 진행함.

## Cycle 1 (2026-10-20)

**A. 분석**
- 코드를 정독함: 전처리, 학습, 평가, 선택, 보고 스크립트.
- 원본 RetailRocket으로 80:20 분할을 새 환경에서 재현함 (`audit_runs/repro_8020`, gitignore 대상으로 Cycle 1을 실행한 PC에만 있고 이 PC에는 없음). 핵심 파일은 바이트 단위로 같음.
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
- 백그라운드 실행이 셸의 Ctrl+C로 다시 끊김 → `CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB`로 분리된 프로세스로 실행해 끝까지 완료함 (exit 0은 v6 PC의 실행 로그 기준, 저장소로 확인 불가. 완료 자체는 커밋된 결과·접근 로그로 확인됨).
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
- 결정성 검사 재실행(ALS·BPR, `determinism_mf_blas1.csv`): 둘 다 비트 단위 동일. GNN(임베딩 최대 차이 1.1e-4, Top-50 목록 99.7% 동일)은 첫 검사(`determinism_run1_blas24.csv`) 값이다. GPU 학습이라 BLAS 스레드 결함과 무관해 다시 돌리지 않았다.
- probe 재실행: 봉인된 설정 행이 봉인된 validation 점수를 그대로 재현함.
- 첫 실행 결과는 `*_run1_blas24.csv`로 이름을 바꿔 보존함.
- 전체 테스트 62개 통과.

**E. 커밋** (`dev`)
- 6326c5a: 결정성 검사, MF 경계 탐색 스크립트
- 6ae2c67: v6 결과 요약과 SHA256 목록
- 418e7dd: 결과 파일을 바이트 그대로 저장 (I-27)
- d309369: 감사 문서 (이 문서 포함)

**F. 커밋 후 검증**
- 커밋된 파일을 `SHA256SUMS.txt`와 대조하다가 60개 모두 불일치를 발견함 (I-27: 줄바꿈 자동 변환). 고친 뒤 60/60 일치.
- `plan.json`의 코드 해시 25개가 현재 코드와 모두 같음 (v6 이후 실험 코드는 바뀌지 않음).

**G. 최종 재감사**
- 문서의 수치를 CSV·JSON과 다시 대조함 (K=50 표, 선택 표, 경계 탐색, epoch, 모집단).
- 고친 서술 3개 (코드 문제는 아님):
  - "Hybrid 12칸 중 9칸 1위"는 baseline 7종 대비라는 조건을 붙임.
  - `Hybrid_ALS_only`는 GNN이 없는 ablation이 아니라 BPR만 뺀 ablation임을 명시함. (Cycle 6에서 정정: 결합 점수에서만 BPR을 빼고 후보 생성에는 BPR이 남음, I-58)
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
- v7 프로토콜 추가 (ALS·BPR·GNN grid 확장, Hybrid grid·ALS+BPR α는 v6와 같음, F4 한 fold). 사용자 승인 후 사전등록.
- `pytest.ini`, v7 구간을 원본 타임스탬프로 확인하는 테스트.
- 보조 스크립트: `plot_rolling_results.py`, `cross_machine_repro.py`, `device_check.py`, `mutation_check.py`, `recompute_test_metrics.py`, `launch_detached.py`.
- `test_cycle3_consistency.py` (테스트 함수 8개, 하나가 정책별로 나뉘어 수집 9개). v7 실행 중에는 코드 해시 잠금 때문에 추가하지 않고, v7 완료 후 추가함.
- `requirements-v7-lock.txt`, 안내 문서 맨 앞의 감사 안내 (원문 보존).

**D. 검증**
- v7 실제 데이터 실행 완료 (11:16 – 13:10 UTC). 시험 열람 1회, 선택 고정 이후. 실행 후 코드 해시가 plan과 같음.
- v7 지표를 원본 `events.csv`와 저장된 추천 목록으로 독립 재계산함: 270행 최대 차이 1e-16, novel 위반 0건.
- 다른 PC: v6 결과 60/60 SHA256 일치. F3 전처리·누수 검사·ALS grid 비트 단위 동일, BPR 최대 0.0047 차이.
- CPU와 GPU: 같은 seed라도 다른 모델이 학습됨 (NDCG 차이 0.0015 – 0.0022, Recall 0.0033 – 0.0047).
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
- 70f5106: `test_cycle3_consistency.py`, `mutation_check.py`, 변이 테스트 결과, 감사 문서

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
- 테스트 3개 추가. 기존 plan 분기를 일부러 망가뜨리면 테스트가 실패함을 확인한 뒤 복원함 (일회성, 출력 없음).

## Cycle 4 (2026-10-22): 처음부터 다시 한 독립 감사

**A. 분석**
- 코드 감사와 문서 수치 감사를 서로 독립인 감사자 둘에게 맡김. 두 감사자는 이전 결론을 전제로 하지 않고 코드, 커밋된 CSV·JSON, 문서를 처음부터 다시 읽음.
- 보고된 항목은 그대로 받아들이지 않고 감사자가 직접 원본 파일로 다시 확인함. 확인되지 않은 항목은 고치지 않음.

**B. 식별 (P0·P1 없음)**
- 문서: 수치·표현 오류 24건 (I-47). 가장 중요한 것은 "v6와 마찬가지로 Hybrid − ALS+BPR 4칸 모두 비유의"라는 서술. 실제 v6는 NDCG 6칸 비유의, Recall 3칸 유의(F1·F2 repeat, F3 novel)였음. 결론("일관되게 유의하게 높지 않다")은 같지만 근거 서술이 틀렸음.
- 코드: 재개 시 누수 판정 무시(I-39), 시험 열람 가드가 root·plan 단위뿐(I-40), 스레드 변수 `setdefault`와 미기록(I-41), 상한 연장 1회(I-42), 재개 기준 파일 쓰기 순서(I-43), 누수 검사 CLI의 끝 비율(I-44), `verify_plan_code` 종료 코드(I-45), 점수 정밀도 경로 차이(I-46).
- 확인 중 새로 안 것: `experiments.py`가 BLAS 로드 뒤 스레드 변수를 1로 다시 써서, 변수를 읽는 방식으로는 실제 스레드 수를 확인할 수 없음 (I-41에 포함).

**C. 수정**
- 문서: 각 수치를 커밋된 CSV·JSON에서 다시 계산한 뒤 고침 (e3cf44e). 저장소만으로 확인할 수 없는 주장(v6 PC 전용 검사 2개)은 그렇게 표시함.
- 코드: `leakage_gate`, `access_log`의 selection 해시 비교, `committed_test_conflicts`, 스레드 강제 설정과 `native_threads` 기록, `extend_capped_backbones`, `test_metrics.csv`를 마지막에 쓰기, `leakage_check.py --end-fraction`, `verify_plan_code.py --require-unchanged`.

**D. 검증**
- 새 테스트 12개(`test_cycle4_guards.py`), 전체 94/94.
- 옛 동작 변이 6개를 사본에 넣어 새 테스트가 모두 잡는지 확인. 처음에는 스레드 변이(G5)를 놓침 → 환경변수 대신 로드된 BLAS의 실제 스레드 수를 읽도록 테스트를 고친 뒤 검출.
- 실제 데이터: 원본 전체 + `--end-fraction 0.36`으로 v7 F4 누수 검사 28개가 커밋된 결과와 같음. 옵션 없이는 12개 거짓 실패.
- 점수 정밀도: v7 F4 내부 validation에서 ALS/BPR 목록 100% 동일 (시험은 읽지 않음).
- 17개 변이 대상 문구가 수정 후에도 각각 한 번씩 있음. 공식 변이 테스트를 수정된 코드로 다시 실행한 결과가 커밋된 `mutation_results.json`과 완전히 같음 (49분 38초. 커밋 전 작업 폴더 코드로 실행했고, 05b9c65와 같다는 산출물은 없음).

**E. 커밋**
- e3cf44e: 문서 수치·표현 정정 (I-47)
- 05b9c65: 코드 가드, 테스트, 증거 파일 (I-39 – I-46)
- 6364ad4: Cycle 4 감사 문서 기록

**F. 커밋 후 재검증**: Cycle 5의 독립 감사로 이어짐 (아래).

## Cycle 5 (2026-10-22): Cycle 4 커밋 뒤 다시 한 독립 감사

**A. 분석**
- Cycle 4와 같은 방식: 코드 감사자와 문서 감사자를 따로 두고 처음부터 읽게 함. 보고된 항목은 원본 파일·실제 실행으로 다시 확인한 뒤에만 고침.

**B. 식별 (P0·P1 없음)**
- 코드 (I-48 – I-56): Cycle 4에 넣은 시험 열람 가드가 같은 plan + 바뀐 코드, 겹치는 구간, 커밋 안 된 로컬 root를 놓침(I-48). 재개 시 원본 데이터를 다시 확인하지 않음(I-49). selection 비교가 목적별이고 끝난 단계는 검사를 건너뜀(I-50). 캐시된 MF·GNN 학습을 설정 확인 없이 재사용(I-51). `selection.json`의 plan 미확인(I-52). 사후 분석 기준 파일 쓰기 순서(I-53). `verify_plan_code`가 추가 파일 무시(I-54). 보조 스크립트 스레드 설정 불일치(I-55). 실행 중 누수 검사가 잘린 원본만 봄(I-56).
- 문서 (I-57): 12건. 결론에 닿는 것은 v7의 "가장 강한 baseline(ALS+BPR)" 서술(repeat에서는 BPR이 평균 최고)과 "Holm 보정 CI"(CI는 보정 안 함). 또 "v7은 `launch_detached.py`로 실행"은 틀림 (그 파일은 v7 뒤에 커밋됨).

**C. 수정**
- 코드: `code_matches`, 구간 겹침 판정, `audit/`+`audit_runs/` 검색, 매 실행 원본 재해시와 `run_started` 로그, `open_test`(모든 목적·끝난 단계도 검사), MF·GNN 캐시 설정 확인, `sealed_selection`, 사후 분석 기준 파일을 마지막에 쓰기, 누수 게이트에 원본 전체 + `end_fraction`, 보조 스크립트 스레드 강제.
- 도구: `guard_mutation_check.py`(가드 변이), `check_cached_runs.py`(끝난 root에 캐시 규칙 적용), `determinism_check.py --blas-threads`.
- 문서: 12건을 CSV·JSON·로그로 다시 확인한 뒤 고침. v7 스레드 주장은 기록이 없어, 실제 재학습으로 확인한 사실로 바꿈.

**D. 검증**
- 새 테스트 12개(`test_cycle5_guards.py`), 전체 106/106.
- 가드 변이 H1–H12: 첫 실행에서 H11·H12가 살아남음 → 중단 시뮬레이션 테스트와 `sealed_selection` 테스트를 추가한 뒤 12/12 검출.
- 실제 데이터: v7 캐시 43개(GNN 39, MF 4) 설정 일치. v7 F4 seed 42 ALS는 1스레드 재학습과 비트 단위로 같고 24스레드와는 다름. 실제 원본 해시로 가드를 돌리면 커밋된 v6·v7 기록과 로컬 `audit_runs/v7`이 모두 잡힘 (당시 출력 없음, Cycle 7에 다시 확인).
- v6·v7 plan 코드 해시가 각 plan commit과 일치 (25 / 26개).
- 공식 변이 테스트 17개를 수정된 코드로 다시 실행: 공식 결과와 완전히 같음 (`audit/cycle5/mutation_results_cycle5.json`).
- `check_score_precision`을 고친 import 순서로 다시 실행: Cycle 4 결과와 완전히 같음.

**E. 커밋**
- fd9bcca: 코드 가드, 도구, 테스트, 증거 파일 (I-48 – I-56)
- 3e6f2ef: 문서 정정(I-57)과 Cycle 5 기록, 변이 재실행 증거

**F. 커밋 후 재검증 (fd9bcca)**
- 새 worktree, `requirements-v7-lock.txt` venv: 105 통과, 1 건너뜀(원본 데이터 상대경로 없음). 결과 파일 SHA256 v6 60/60, v7 28/28. (실행 출력은 저장소에 없음.)

## Cycle 6 (2026-10-22): Cycle 5 커밋 뒤 다시 한 독립 감사

**A. 분석**
- 코드 감사자와 문서 감사자를 새로 두고 처음부터 읽게 함. 보고 항목은 원본 파일과 실제 실행으로 다시 확인한 뒤에만 고침.

**B. 식별 (P0·P1 없음)**
- 코드: 시험 기록을 원본 4개 파일 바이트가 모두 같을 때만 비교(I-59). Cycle 5의 캐시 확인 도구가 같은 출처의 두 파일을 비교해 의미가 없었음(I-60). `git_state`가 첫 줄 상태 열을 지움(I-61).
- 문서: `Hybrid_ALS_only`를 "BPR을 뺀 ablation"으로 설명(I-58. 실제로는 후보 생성에 BPR이 남음). 수치·서술 오류 11건과 저장소로 확인할 수 없는 기록의 출처 미표시(I-62). 가장 큰 것은 I-24의 "v7은 1스레드 launcher로 실행"(기록 없음)과 §12.5의 "CI는 0을 포함하지 않지만 유의하지 않은 칸" 목록에서 2칸 누락.

**C. 수정**
- 코드: `event_digests`와 events 해시 비교, `git_state`는 끝 줄바꿈만 제거, `check_cached_runs.py`는 plan·selection·`validation_gnn.json`에서 요청을 다시 만들어 비교.
- 도구: 가드 변이 H13·H14 추가.
- 문서: 각 항목을 CSV·JSON·로그와 실제 실행으로 확인한 뒤 고침. 확인할 수 없는 기록은 출처와 "저장소로 확인 불가"를 표시. 다시 확인할 수 있는 것은 다시 실행해 증거를 남김(두 venv freeze, 변이 사본과 fd9bcca 비교).

**D. 검증**
- 전체 109/109 (v7 venv). 가드 변이 14/14.
- 실제 데이터: v7 캐시 요청 43개 일치, 실제 원본으로 기록 5건 모두 잡힘 (가드 확인은 당시 출력 없음, Cycle 7에 다시 확인).

**E. 커밋**
- 57dc03b: 코드, 테스트, 증거 (I-59 – I-61)
- 8cf8215: 문서 정정(I-58, I-62), Cycle 6 기록과 증거

**F. 커밋 후 재검증 (57dc03b)**
- 새 worktree, lock venv: 108 통과, 1 건너뜀(원본 데이터 상대경로 없음). 결과 파일 SHA256 v6 60/60, v7 28/28. plan 코드 v6 25개, v7 26개 일치. 출력은 `audit/cycle6/`에 보존.

## Cycle 7 (2026-10-22): Cycle 6 커밋 뒤 다시 한 독립 감사

**A. 분석**
- 코드 감사자와 문서 감사자를 새로 두고 처음부터 읽게 함. 보고 항목은 코드·커밋된 파일·실제 실행으로 다시 확인한 뒤에만 고침.

**B. 식별 (P0 없음)**
- P1 (I-63): novel 정책에서 Hybrid는 결합 설정을 novel validation으로 다시 고르지만 ALS·BPR·GNN 단독은 repeat에서 고른 설정을 씀 (사전등록 규칙). 문서에 그 영향이 없었음. validation으로 계산하면 v7 F4에서 novel로 고른 GNN이 Hybrid보다 높음.
- 코드: 다른 root 확인이 시작 때만(I-64), 재현 예외가 `events.csv` 밖 원본을 안 봄(I-65), Hybrid grid 캐시를 존재만으로 재사용(I-66), git 없이 plan 생성(I-67).
- 문서 (I-68): 10건과 일회성 확인의 출력 없음 표시 누락.

**C. 수정**
- I-63은 시험 구간을 다시 열 수 없어 시험으로는 확인하지 않음. validation 전용 도구로 크기를 계산해 공개하고, novel의 Hybrid − GNN·ALS 결과 해석을 한정함 (`validation_report.md` §8 항목 5, `manuscript_alignment.md` 공개 사항 8).
- 코드: `open_test`에서 다시 확인, 재현 예외에 원본 4개 파일, `cached_grid`, git commit 필수.
- 도구: `policy_tuning_gap.py`, `guard_real_data_check.py`, 가드 변이 H15–H18.

**D. 검증**
- 전체 114/114 (v7 venv). 가드 변이 18/18. 실제 원본 가드 확인 결과 저장.
- 공식 변이 테스트를 커밋 8cf8215 그대로(깨끗한 작업 폴더) 다시 실행: 공식 결과와 완전히 같음 (`audit/cycle7/mutation_results_8cf8215.json`).

**E. 커밋**
- 0ef6daa: 코드, 도구, 테스트, 증거 (I-63 – I-67)
- 343e75f: 문서 정정(I-63, I-68)과 Cycle 7 기록

**F. 커밋 후 재검증 (0ef6daa)**
- 새 worktree, lock venv: 113 통과, 1 건너뜀(원본 데이터 상대경로 없음). 결과 파일 SHA256 v6 60/60, v7 28/28. 출력은 `audit/cycle7/`에 보존.

## Cycle 8 (2026-10-22): Cycle 7 커밋 뒤 다시 한 독립 감사

**A. 분석**
- 코드 감사자와 문서 감사자를 새로 두고 처음부터 읽게 함. 보고 항목은 코드·커밋된 파일·실제 실행으로 다시 확인한 뒤에만 고침.

**B. 식별 (P0·P1 없음)**
- 코드: P2 1건(I-70 실행 중 코드 고정 미확인), P3 3건(I-71 자기 root 오인, I-72 grid 캐시 정밀도, I-73 기록 파일 손상).
- 문서·도구 (I-69, P2 1건과 P3 9건을 한 항목으로): I-63 한정이 일부 문서에 없음, `policy_tuning_gap.py` 테스트가 순환적, 범위·참조 오류, 출력 없는 실행 표시 누락.

**C. 수정**
- 코드: `require_frozen_code`(시작·GNN 작업·시험 열람), 자기 root 제외, 잘린 기록 줄은 모든 fold 연 것으로, round-trip float 파싱.
- 도구: `policy_tuning_gap.py`가 `selection.json` 설정을 찾음, `guard_mutation_check.py`가 HEAD·커밋 안 된 파일·시각 기록, `stamp_code_check.py`, `grid_cache_precision_check.py`, 가드 변이 H19–H27.
- 문서: I-69 항목들, GNN novel 최고값이 하한이라는 점.

**D. 검증**
- 작업 폴더 120/120 (v7 venv). 실제 원본 가드 출력은 Cycle 7과 같음(새 출력은 저장하지 않음). v7 코드 해시 731개 모두 plan과 같음. v7 grid 재읽기 확인.
- 가드 변이 H1–H27을 커밋 b9e7134 코드로 실행: 27/27.
- 공식 변이 테스트를 커밋 343e75f 그대로 다시 실행: 공식 결과와 완전히 같음.

**E. 커밋**
- b9e7134: 코드, 도구(`stamp_code_check.py` 제외), 테스트, 증거 일부 (I-70 – I-73, I-69의 도구 부분)
- 79b628d: 문서 정정(I-69), Cycle 8 기록과 증거, `stamp_code_check.py`와 그 출력 `audit/cycle8/stamp_code_v7.json`(I-70 근거), `guard_mutation_check.py`의 git 상태 기록 줄 앞 공백이 잘리던 것 수정

**F. 커밋 후 재검증 (b9e7134)**
- 새 worktree, lock venv: 119 통과, 1 건너뜀(원본 데이터 상대경로 없음). 결과 파일 SHA256 v6 60/60, v7 28/28. 출력은 `audit/cycle8/`에 보존.

## Cycle 9 (2026-10-22): Cycle 8 커밋 뒤 다시 한 독립 감사

**A. 분석**
- 코드 감사자와 문서 감사자를 새로 두고 처음부터 읽게 함. 보고 항목은 코드·커밋된 파일·실제 실행으로 다시 확인한 뒤에만 고침.

**B. 식별 (P0·P1·P2 없음)**
- 코드 P3 5건: I-75 다른 root의 재현이 다른 selection으로 다시 열 수 있음, I-76 완료 표시 파일 직접 쓰기, I-77 스탬프 도구가 `sha256-lf` plan에서 거짓 불일치, I-78 재학습 스레드 오류 원인 가려짐, I-79 명목 비율 겹침 검사가 경계 이벤트 공유를 놓침.
- 문서 P3 7건 (I-74).

**C. 수정**
- 코드: 열람 직전 검사에 selection·실제 구간, `write_csv_atomic`, `start_background`, 접근 기록 줄 형식 확인.
- 도구: `stamp_code_check.py` 해시 방식별 비교, `guard_real_data_check.py` 실제 구간·경계 fold, 가드 변이 H28–H35.
- 문서: I-74 항목들, 시험 열람 가드 설명(selection 조건, 실제 구간).

**D. 검증**
- 작업 폴더 127/127 (v7 venv). 실제 원본으로 구간·경계 fold 확인, v7 스탬프 731/731.
- 가드 변이 H1–H35를 커밋 ce97790 코드로 실행: 35/35.
- 공식 변이 테스트를 커밋 79b628d 그대로 다시 실행: 공식 결과와 완전히 같음.

**E. 커밋**
- ce97790: 코드, 도구, 테스트, 증거 일부 (I-75 – I-79)
- 이 문서와 함께: 문서 정정(I-74), Cycle 9 기록과 증거

**F. 커밋 후 재검증 (ce97790)**
- 새 worktree, lock venv: 126 통과, 1 건너뜀(원본 데이터 상대경로 없음). 결과 파일 SHA256 v6 60/60, v7 28/28. 출력은 `audit/cycle9/`에 보존.

## Cycle 10 (2026-10-22): Cycle 9 커밋 뒤 다시 한 독립 감사

**A. 분석**
- 코드 감사자와 문서 감사자를 새로 두고 처음부터 읽게 함. 보고 항목은 코드·커밋된 파일·실제 실행으로 다시 확인한 뒤에만 고침.

**B. 식별 (P0·P1 없음)**
- 코드 P2 1건: I-81 I-75 수정이 원본부터 다시 돌린 정확한 재현을 항상 거부(selection 파일에 실행마다 다른 값).
- 코드 P3 3건: I-82 기록의 분할 필드가 정수가 아니면 이름 없는 예외, I-83 주 단계가 실패해도 GNN 작업이 계속 돎, I-84 알 수 없는 fold 이름이면 아무것도 안 하고 완료.
- 문서 P3 5건 (I-80). 그중 재현 명령이 커밋된 증거를 덮어쓰는 문제는 도구 기본 출력까지 고침.

**C. 수정**
- 코드: `selection_decision`·`opened_decision`, 분할 필드 형식 확인, `alongside`와 `run_gnn_jobs`의 stop·종료, fold 이름 확인.
- 도구: 두 변이 도구의 기본 출력을 `audit_runs/`로, `guard_real_data_check.py`에 `decisions`, 가드 변이 H36–H44.
- 테스트: `test_cycle10_guards.py`(원본부터 두 root를 돌리는 재현 포함). 없는 fold 이름으로 시작 검사만 돌리던 테스트는 실제 fold와 `run_fold` 대체로 바꿈.
- 문서: I-80 항목들, 시험 열람 가드 설명(결정 비교, 결정이 다르면 학습 뒤에 거부됨), 백그라운드 작업 동작.

**D. 검증**
- 작업 폴더 136/136 (v7 venv). 감사자의 원본부터 재현 스크립트를 고친 코드로 다시 실행해 두 번째 root가 완료됨(출력 저장 안 함. 같은 내용을 `test_exact_reproduction_from_raw_reaches_its_test_opening`이 확인).
- 실제 기록 5개의 결정 확인. 4e166c5 전에 재현 7–10단계 출력 5개가 당시의 커밋된 증거와 같음(비교 결과 저장 안 함. Cycle 11에 저장된 비교로 다시 확인).
- 가드 변이 H1–H44를 커밋 4e166c5 코드로 실행: 44/44.
- 공식 변이 테스트를 커밋 ac9b865 그대로 다시 실행: 공식 결과와 완전히 같음.

**E. 커밋**
- 4e166c5: 코드, 테스트, 변이 도구 (I-81 – I-84, I-80의 도구 부분)
- 이 문서와 함께: 문서 정정(I-80), `guard_real_data_check.py`의 `decisions`, Cycle 10 기록과 증거

**F. 커밋 후 재검증 (4e166c5)**
- 새 worktree, lock venv: 135 통과, 1 건너뜀(원본 데이터 상대경로 없음). 결과 파일 SHA256 v6 60/60, v7 28/28. 출력은 `audit/cycle10/`에 보존.

## Cycle 11 (2026-10-22): Cycle 10 커밋 뒤 다시 한 독립 감사

**A. 분석**
- 코드 감사자와 문서 감사자를 새로 두고 처음부터 읽게 함. 보고 항목은 코드·커밋된 파일·실제 실행으로 다시 확인한 뒤에만 고침.

**B. 식별 (P0·P1 없음)**
- 문서 P2 1건, P3 6건 (I-85): plan 코드 확인 명령이 커밋된 Cycle 3 증거를 덮어씀, 그림·재계산 명령과 도구 사용 예의 출력 경로, §19 서술 3곳, 저장 안 한 확인 표시, 테스트 개수.
- 코드 감사 P3 4건 (I-88·I-89는 코드가 아니라 문서·그림 서술): I-86 끊긴 hard link 복제가 완성본으로 남음, I-87 실행 프로세스를 강제로 끝내면 GNN 작업이 남음(Popen 실패 때 로그도 안 닫힘), I-88 평가 모집단 서술이 코드와 다름, I-89 구조도·식 8 행이 `balanced_union`을 빠뜨림.

**C. 수정**
- 코드: `clone_protocol`을 `.partial` 폴더에 만든 뒤 옮김, GNN 작업을 KILL_ON_JOB_CLOSE job object에 넣음(`kill_with_parent`), Popen 실패 때 로그 닫기.
- 도구: `compare_outputs.py` 추가, 도구 4개의 사용 예와 재현 명령 출력을 `audit_runs/repro/`로, `plot_rolling_results.py`의 `--output` 필수, 구조도 후보 생성 문구, 가드 변이 H45–H48.
- 테스트: `test_cycle11_guards.py` 3개(실제 부모 프로세스를 강제 종료하는 확인 포함), `test_compare_outputs.py` 1개. `test_cycle10_guards.py`의 가짜 작업 테스트는 `kill_with_parent`를 대체함.
- 문서: I-85 항목들, 평가 모집단과 식 8 행(`manuscript_alignment.md`), 재현 문서의 job object(POSIX에는 없음)와 원자적 복제.

**D. 검증**
- 재현 4, 7–10단계·plan 코드 확인·재계산을 8d8d0d5 도구로 다시 실행해 29행(다시 만든 출력 28개. plan 코드 확인 출력 하나를 Cycle 3·Cycle 6 증거 두 곳과 비교했고, 11단계 명령은 Cycle 3 쪽만 비교해 28행)이 모두 커밋된 증거와 같음(`audit/cycle11/repro_compare.json`).
- 공식 변이 테스트를 커밋 8d8d0d5 그대로 다시 실행: 공식 결과와 완전히 같음(17/17).
- 구조도를 다시 만들어 52개 중 `pipeline.*` 6개만 바뀜. 봉인된 설정 8개 중 `balanced_union` 5개를 커밋된 `selection.json`에서 다시 셈.
- `audit_runs/v7`의 protocol 폴더 51개가 모두 완전함(남은 `.partial` 없음).
- H45–H48을 `test_cycle11_guards.py`만으로 먼저 확인: 넷 다 검출, 수정 안 한 사본 3 통과(출력 저장 안 함).
- 작업 폴더 140/140 (v7 venv, 출력 저장 안 함).
- 가드 변이 H1–H48을 커밋 318186b 코드로 실행: 48/48, 수정 안 한 사본 53 통과.

**E. 커밋**
- 318186b: 코드, 테스트, 도구 (I-86, I-87, I-85·I-89의 도구 부분, H45–H48)
- 이 문서와 함께: 문서 정정(I-85, I-88, I-89), 다시 만든 구조도, Cycle 11 기록과 증거

**F. 커밋 후 재검증 (318186b)**
- 새 worktree, lock venv: 139 통과, 1 건너뜀(원본 데이터 상대경로 없음). 결과 파일 SHA256 v6 60/60, v7 28/28. 출력은 `audit/cycle11/`에 보존.

## Cycle 12 (2026-10-22): Cycle 11 커밋 뒤 다시 한 독립 감사

**A. 분석**
- 코드 감사자(HEAD 318186b)와 문서 감사자(HEAD 2f1767c)를 새로 두고 처음부터 읽게 함. 보고 항목은 코드·커밋된 파일·실제 실행으로 다시 확인한 뒤에만 고침.

**B. 식별 (P0·P1·P2 없음)**
- 코드 감사 P3 4건: I-90 전처리가 끝나기 전에 바깥 manifest를 써서 끊긴 전처리가 완성본으로 재개됨, I-91 빈 `--folds`가 모든 fold를 실행, I-92 `compare_outputs`가 0개 비교·새 출력에만 있는 파일·형이 다른 JSON 값을 통과시킴, I-93 이미 끝난 GNN 작업의 job object 오류가 원래 원인을 가림.
- 코드 감사의 미확정 의심 중 하나(다른 장치의 정확한 재현)를 I-94로 공개 처리. 나머지(겹치는 fold 이름의 재현 예외, job 할당 전 짧은 구간(감사자 측정 약 10ms, 저장 안 함), `os.replace` 실패 가능성, epoch 상한 표시 구간, cold 사용자 0점, IF ablation 누수 검사 없음 등)는 확인되지 않았거나 설계·공개 사항.
- 문서 감사 P3 3건 (I-95): 재현 비교 개수, `--require-unchanged` 실패 원인 목록, Cycle 9 실행 시간.

**C. 수정**
- 코드: 바깥 manifest는 `finalize_outer`에서 한 번만(`.partial` 후 이름 바꿈), 빈 fold 목록 거부(`nargs='+'`), 끝난 작업의 job object 오류는 종료 코드 확인으로 넘김.
- 도구: `compare_outputs.py` 엄격화, 가드 변이 H49–H54(H46 대상 문구 수정).
- 테스트: `test_cycle12_guards.py` 4개, `test_compare_outputs.py` 1개 추가. 새 테스트가 H46 변이에서 끝없이 돌던 것을 sleep 횟수 제한으로 고침(변이 확인 중 발견, 약 20분 멈춤).
- 문서: I-95 항목들, 재현 문서(장치, job 할당 전 짧은 구간, 빈 `--folds`, 끊긴 전처리, 누수 검사 출력 경로), Cycle 11 분류 표시.

**D. 검증**
- 새 테스트와 관련 테스트, H46·H49–H54를 해당 테스트만으로 먼저 확인: 일곱 다 검출, 수정 안 한 사본 9 통과(출력 저장 안 함).
- 작업 폴더 145/145 (v7 venv, 출력 저장 안 함).
- 실제 원본으로 v7 F4 전처리를 다시 해 기존 실행 폴더와 파일·`protocol_id`·원본 해시가 같음(`audit/cycle12/prep_check_v7_F4.txt`).
- Cycle 11 재현 비교를 새 도구로 다시 함: I-89 전 그림 3개만 다름(출력 저장 안 함).
- 공식 변이 테스트를 커밋 2f1767c 그대로: 공식 결과와 바이트 단위로 같음(17/17).
- 가드 변이 H1–H54를 커밋 5026680 코드로 실행: 54/54, 수정 안 한 사본 59 통과.

**E. 커밋**
- 5026680: 코드, 테스트, 도구 (I-90 – I-93, H49–H54)
- 8feac9b: `kill_with_parent` 설명 정정 (I-95, 주석만 바뀜. 관련 가드 테스트 15개 통과(출력 저장 안 함), 54개 변이 대상 문구는 그대로 한 번씩 있음)
- 이 문서와 함께: 문서 정정(I-94, I-95), Cycle 12 기록과 증거

**F. 커밋 후 재검증 (5026680)**
- 새 worktree, lock venv: 144 통과, 1 건너뜀(원본 데이터 상대경로 없음). 결과 파일 SHA256 v6 60/60, v7 28/28. 출력은 `audit/cycle12/`에 보존.
- 8feac9b도 같은 방법으로: 144 통과, 1 건너뜀, SHA256 60/60, 28/28 (`audit/cycle12/postcommit_8feac9b_*.txt`).

## Cycle 13 (2026-10-22): Cycle 12 커밋 뒤 다시 한 독립 감사

**A. 분석**
- 코드 감사자(HEAD 1e2b35e)와 문서 감사자(HEAD 1e2b35e)를 새로 두고 처음부터 읽게 함. 보고 항목은 코드·커밋된 파일·실제 실행으로 다시 확인한 뒤에만 고침.

**B. 식별 (P0·P1·P2 없음)**
- 코드 감사 P3 4건: I-97 Hybrid 끝 축 표시에 α가 없고 balanced_union의 후보 수 최대(20,000)를 놓쳐 v7 novel 행이 "-"로 보고됨, I-98 `compare_outputs`의 없는 `#keys`·객체가 아닌 JSON·빈 커밋 폴더 쌍, I-99 `check_score_precision`이 역할을 확인하지 않음, I-100 `use_protocol('v6')`가 v7 값을 되돌리지 않음.
- 코드 감사의 미확정 의심(IF ablation 누수 검사 없음, `prepare_conditions`의 이른 manifest, 원자적이지 않은 JSON 표시 등)은 확인되지 않았거나 거부하는 방향.
- 문서 감사 P3 5건 (I-96): Cycle 12 가드 변이와 동시 실행한 작업, 두 번째 PC 범위, I-95의 정정 이유, "약 10ms"의 근거, 저장 안 한 확인 표시.

**C. 수정**
- 코드: `hybrid_at_grid_boundary` 함수(선택된 전략의 grid, α 포함), v6 protocol 값 저장, 정밀도 도구의 역할 확인.
- 도구: `compare_outputs.py`의 키·객체 확인과 쌍별 빈 폴더 표시, 가드 변이 H55–H60.
- 테스트: `test_cycle13_guards.py` 4개.
- 문서: §12.2 Hybrid novel 행과 `v7_feasibility.md` §4의 끝 축(커밋된 `selection.json`은 고치지 않음), I-96 항목들, 재현 문서(plan commit으로 다시 실행하면 옛 끝 표시가 나옴).
- 가드 변이 첫 실행이 H52에서 "대상 문구 없음"으로 멈춤(H1–H51은 모두 검출): cb22505에서 바꾼 `all_equal` 줄을 H52가 옛 문구로 가리켰음. 대상을 고치고(db3653e) 60개 대상이 모두 한 번씩 있는지 확인한 뒤 처음부터 다시 실행. 첫 실행의 로그는 다시 실행할 때 덮어써 보존 안 됨(H1–H51 검출은 그 로그를 본 기록).

**D. 검증**
- 새 테스트와 관련 테스트 13개 통과. H52·H55–H60을 해당 테스트만으로 먼저 확인: 일곱 다 검출, 수정 안 한 사본 6 통과(출력 저장 안 함).
- 작업 폴더 149/149 (v7 venv, 출력 저장 안 함).
- 실제 데이터: 커밋된 `selection.json` 8개 정책의 끝 축 재계산, 정밀도 도구의 시험 폴더 거부와 validation 재실행(Cycle 4와 같음), protocol 전환별 plan 해시, 재현 비교 재실행(§22).
- 공식 변이 테스트를 커밋 1e2b35e 그대로: 공식 결과와 바이트 단위로 같음(17/17).
- 가드 변이 H1–H60을 커밋 db3653e 코드로 실행: 60/60, 수정 안 한 사본 63 통과 (`audit/cycle13/guard_mutants.json`).

**E. 커밋**
- cb22505: 코드, 테스트, 도구 (I-97 – I-100, H55–H60)
- db3653e: 가드 변이 H52 대상 수정
- 이 문서와 함께: 문서 정정(I-96), Cycle 13 기록과 증거

**F. 커밋 후 재검증**
- cb22505, db3653e 각각 새 worktree, lock venv: 148 통과, 1 건너뜀(원본 데이터 상대경로 없음). 결과 파일 SHA256 v6 60/60, v7 28/28. 출력은 `audit/cycle13/`에 보존.

## Cycle 14 (2026-10-22): Cycle 13 커밋 뒤 다시 한 독립 감사

**A. 분석**
- 코드 감사자(HEAD cb22505)와 문서 감사자(HEAD dcbfd0a)를 새로 두고 처음부터 읽게 함. 보고 항목은 코드·커밋된 파일·실제 실행으로 다시 확인한 뒤에만 고침.

**B. 식별 (P0·P1·P2 없음)**
- 코드 감사 P3 1건: I-101 plan이 Hybrid grid 값을 고정하지 않음.
- 코드 감사의 미확정 의심: IF ablation(`prepared_if`)의 누수 검사 없음(감사자가 cutoff 이후 이벤트·속성을 바꿔도 학습 쪽 파일 14개가 그대로임을 확인, 실제 누수 아님. IF를 쓴 결과는 v6 F1 사후 분석뿐이고 선택·paired 검정에 쓰이지 않음), GNN 상한 표시는 최고 epoch 390 이상만 봐서 350–380에서 400으로 잘린 학습을 표시하지 않음(v7에 해당 없음, v6 최대 290), `candidate_k`가 float로 들어오는 경우(호출부가 int 또는 `all`로 바꿈), 깨진 입력의 오류 메시지.
- 문서 감사 P3 4건 (I-102)과, 같이 보고된 의심 중 확인된 2건.

**C. 수정**
- 코드: plan의 `hybrid_grid_axes`를 `HYBRID_GRID`에서 뽑음 (v6·v7 plan 해시 그대로).
- 테스트: `test_cycle14_guards.py` 2개. 가드 변이 H61.
- 문서: I-102 항목들.
- 가드 변이를 처음 띄울 때 로그를 `/dev/null`로 보내는 명령을 써서, 그 프로세스를 바로 끝내고(시작 몇 초 뒤) 로그 파일로 다시 시작함.

**D. 검증**
- H61을 해당 테스트만으로 먼저 확인: 검출, 수정 안 한 사본 2 통과(출력 저장 안 함). 61개 대상이 모두 한 번씩 있음.
- 작업 폴더 151/151 (v7 venv, 출력 저장 안 함).
- 실제 데이터: v6·v7 plan 내용과 해시가 커밋된 `plan.json`과 같음 (`audit/cycle14/plan_hash_check.txt`).
- 가드 변이 H1–H61을 커밋 41ba20d 코드로 실행: 61/61, 수정 안 한 사본 65 통과 (`audit/cycle14/guard_mutants.json`).

**E. 커밋**
- 41ba20d: 코드, 테스트, 가드 변이 H61 (I-101)
- 이 문서와 함께: 문서 정정(I-102), Cycle 14 기록과 증거

**F. 커밋 후 재검증**
- 41ba20d: 새 worktree, lock venv 150 통과, 1 건너뜀. SHA256 v6 60/60, v7 28/28 (`audit/cycle14/postcommit_41ba20d_*.txt`).
- Cycle 13 문서 커밋 dcbfd0a: 148 통과, 1 건너뜀, 60/60, 28/28 (`audit/cycle14/postcommit_dcbfd0a_*.txt`).

## Cycle 15 (2026-10-22): Cycle 14 커밋 뒤 다시 한 독립 감사

**A. 분석**
- 코드 감사자와 문서 감사자(둘 다 HEAD 121b40a)를 새로 두고 처음부터 읽게 함. 보고 항목은 코드·커밋된 파일·실제 실행으로 다시 확인한 뒤에만 고침.

**B. 식별 (P0·P1·P2 없음)**
- 코드 감사 P3 1건: I-104 I-101 수정이 불완전(후보 깊이 값, 전략과 후보 깊이의 짝, grid 순서가 plan 해시에 들어가지 않음).
- 코드 감사의 미확정 의심: `load_factors`가 팩터 파일을 `als_artifacts.json`과 대조하지 않음(파일을 일부러 바꿔야 문제), 원자적이지 않은 JSON 기록(끊기면 실패하는 방향), 같은 root 동시 실행 잠금 없음, novel 정책의 Recent·RecentPop이 Holm 가족에 남음(보수적 방향, 공개됨), 행동 가중치가 plan에 없음(코드 해시가 보호), `recompute_test_metrics`가 접근 로그 없이 시험 라벨을 읽음(검증 전용).
- 문서 감사 P3 2건 (I-103). 문서 감사의 미확정 의심 중 F3 사용자 수 77,237(`v7_feasibility.md` §2)은 커밋된 파일로 확인되지 않음.

**C. 수정**
- 코드: grid가 봉인된 v6·v7 grid와 다르면 plan에 grid sha256 추가 (v6·v7 plan 해시 그대로).
- 테스트: `test_cycle15_guards.py` 6개, Cycle 14 테스트에 기록된 축 확인 추가. 가드 변이 H62.
- 문서: I-103 항목들, I-101 행에 불완전 표시.

**D. 검증**
- H61·H62를 해당 테스트만으로 먼저 확인: 처음에는 H61이 살아남아(새 sha256만으로 해시가 바뀜) 기록된 축 확인을 추가한 뒤 둘 다 검출, 수정 안 한 사본 8 통과(출력 저장 안 함). 62개 대상이 모두 한 번씩 있음.
- 작업 폴더 157/157 (v7 venv, 출력 저장 안 함).
- 실제 데이터: v6·v7 grid가 봉인된 grid와 같고 plan 내용과 해시가 커밋된 `plan.json`과 같음 (`audit/cycle15/plan_hash_check.txt`).
- 가드 변이 H1–H62를 커밋 4c1ddfb 코드로 실행: 62/62, 수정 안 한 사본 71 통과 (`audit/cycle15/guard_mutants.json`).

**E. 커밋**
- 4c1ddfb: 코드, 테스트, 가드 변이 H62 (I-104)
- 이 문서와 함께: 문서 정정(I-103), Cycle 15 기록과 증거

**F. 커밋 후 재검증**
- 4c1ddfb: 새 worktree, lock venv 156 통과, 1 건너뜀. SHA256 v6 60/60, v7 28/28 (`audit/cycle15/postcommit_4c1ddfb_*.txt`).
- Cycle 14 문서 커밋 121b40a: 150 통과, 1 건너뜀, 60/60, 28/28 (`audit/cycle15/postcommit_121b40a_*.txt`).
- Cycle 15 문서 커밋 c0b52c0: 156 통과, 1 건너뜀, 60/60, 28/28 (`audit/cycle16/postcommit_c0b52c0_*.txt`).

## Cycle 16 (2026-10-22): Cycle 15 커밋 뒤 다시 한 독립 감사

**A. 분석**
- 코드 감사자와 문서 감사자(둘 다 HEAD c0b52c0)를 새로 두고 처음부터 읽게 함. 보고 항목은 코드·커밋된 파일·실제 실행으로 다시 확인한 뒤에만 고침.

**B. 식별 (P0·P1 없음)**
- 문서 감사 P2 1건: I-105 v7을 "모든 모델군 grid 확장"으로 서술(Hybrid grid·ALS+BPR α는 v6와 같음).
- 코드 감사 P2 1건: I-106 F2 시험 구간을 "독립적인 확인"으로 서술(이벤트 80%가 v1–v5 내부 validation 구간).
- 문서 감사 P3 3건 (I-107), 코드 감사 P3 1건 (I-108 dirty 검사의 접두사 면제).
- 코드 감사는 4c1ddfb의 회귀가 없음을 확인(v6→v7→v6 전환 뒤에도 plan 내용·해시가 커밋된 값과 같음, plan payload를 따로 다시 계산하는 곳 없음). 미확정 의심(GNN 시험 경로 float32, `tie_safe_cutoff`와 `int(len·end)`의 반올림, 빈 줄 있는 접근 로그는 실패하는 방향 등)은 실제 데이터에서 차이 없음.

**C. 수정**
- 코드: dirty 검사 면제를 `audit_runs/` 폴더로 한정.
- 테스트: `test_cycle16_guards.py` 1개. 가드 변이 H63.
- 문서: I-105 – I-108.

**D. 검증**
- H63을 해당 테스트만으로 먼저 확인: 고친 코드 1 통과, H63을 넣은 사본 1 실패(출력 저장 안 함). 63개 대상이 모두 한 번씩 있음.
- 작업 폴더 전체 테스트: 158/158 (v7 venv, `audit/cycle16/precommit_pytest.txt`).
- 실제 저장소: `git_state()`가 커밋 안 된 코드·문서를 모두 나열하고 `audit_runs/`는 빠짐. v6·v7 plan 내용·해시가 커밋된 `plan.json`과 같음 (`audit/cycle16/git_state_and_plan_check.txt`).
- 가드 변이 H1–H63을 커밋 6ce754d 코드로 실행: 63/63, 수정 안 한 사본 72 통과 (`audit/cycle16/guard_mutants.json`).

**E. 커밋**
- 6ce754d: 코드, 테스트, 가드 변이 H63 (I-108)
- 이 문서와 함께: 문서 정정(I-105 – I-107), Cycle 16 기록과 증거

**F. 커밋 후 재검증**
- 6ce754d: 새 worktree, lock venv 157 통과, 1 건너뜀. SHA256 v6 60/60, v7 28/28 (`audit/cycle16/postcommit_6ce754d_*.txt`).
- Cycle 16 문서 커밋 0a0ca40: 157 통과, 1 건너뜀, 60/60, 28/28 (`audit/cycle17/postcommit_0a0ca40_*.txt`).

## Cycle 17 (2026-10-22): Cycle 16 커밋 뒤 다시 한 독립 감사

**A. 분석**
- 코드 감사자와 문서 감사자(둘 다 HEAD 0a0ca40)를 새로 두고 처음부터 읽게 함. 보고 항목은 코드·커밋된 파일·실제 실행으로 다시 확인한 뒤에만 고침.

**B. 식별 (P0·P1·P2 없음)**
- 코드 감사 P3 2건: I-110 I-108 수정이 불완전(git 설정·이름 변경에 따라 dirty 검사가 코드를 놓침), I-111 상대 `--root`에서 GNN spec 경로.
- 문서 감사 P3 6건 (I-109): F3 시험 구간도 v6 F2의 validation이었음, 머리말·docstring·§1 행 누락, 교수 자료 md 안내, v6 Hybrid backbone lr 끝.
- 미확정 의심: `audit_runs`가 junction·symlink일 때(제외 pathspec은 경로 이름으로 맞추므로 지금은 빠질 것으로 보지만 확인하지 않음), `assume-unchanged` 표시 파일(plan 코드 해시가 내용을 잡음), 가드 변이가 end-to-end 재현 말고 `test_cycle4_guards`의 한 테스트도 뺌.

**C. 수정**
- 코드: dirty 검사를 git 제외 pathspec과 `--untracked-files=normal`로, GNN spec 폴더를 절대 경로로.
- 테스트: `test_cycle16_guards.py`를 실제 git으로, `test_cycle17_guards.py` 2개, Cycle 5 상태 열 테스트의 가짜 출력을 지금 git 출력으로(제외 pathspec 요구). 가드 변이 H63 대상 변경, H64·H65.
- 문서: I-109 – I-111.

**D. 검증**
- H63–H65를 해당 테스트만으로 먼저 확인: 모두 검출, 수정 안 한 사본 3 통과(출력 저장 안 함). 처음 넣은 `--no-renames` 변이는 살아남아(제외 pathspec만으로 충분) 옵션과 변이를 뺌. 65개 대상이 모두 한 번씩 있음.
- 작업 폴더 전체 테스트: 첫 실행 159 통과 1 실패(Cycle 5 상태 열 테스트), 고친 뒤 160/160 (`audit/cycle17/precommit_pytest*.txt`).
- 실제 저장소: `git_state()` 목록이 일반 `git status`와 같고 `audit_runs/`는 빠짐. v6·v7 plan 내용·해시 그대로 (`audit/cycle17/git_state_and_plan_check.txt`).
- 가드 변이 H1–H65를 커밋 cf1a3fe 코드로 실행: 65/65, 수정 안 한 사본 74 통과 (`audit/cycle17/guard_mutants.json`).

**E. 커밋**
- cf1a3fe: 코드, 테스트, 가드 변이 H63–H65, docstring (I-110, I-111)
- 이 문서와 함께: 문서 정정(I-109), Cycle 17 기록과 증거

**F. 커밋 후 재검증**
- cf1a3fe: 새 worktree, lock venv 159 통과, 1 건너뜀. SHA256 v6 60/60, v7 28/28 (`audit/cycle17/postcommit_cf1a3fe_*.txt`).
- Cycle 17 문서 커밋 31b059c: 159 통과, 1 건너뜀, 60/60, 28/28 (`audit/cycle18/postcommit_31b059c_*.txt`).

## Cycle 18 (2026-10-23 KST): Cycle 17 커밋 뒤 다시 한 독립 감사

**A. 분석**
- 코드 감사자와 문서 감사자(둘 다 HEAD 31b059c)를 새로 두고 처음부터 읽게 함. 보고 항목은 코드·커밋된 파일·실제 실행으로 다시 확인한 뒤에만 고침.

**B. 식별 (P0·P1 없음)**
- 문서 감사 P2 1건: I-112 README_KO의 매니페스트 해시 서술이 틀림. P3: I-113 F3 한정 누락과 I-106 행의 고친 범위, I-114 H63 의미 변경과 docstring.
- 코드 감사 P3 3건: I-115 `git status`만 실패하면 깨끗한 트리로 처리(문서 감사도 미확정 의심으로 보고), I-116 `core.quotePath=false`에서 비ASCII 경로로 `git_state`가 멈춤, I-117 launcher의 경로 기준.
- 미확정 의심: `determinism_check`가 옮긴 root에서 실패할 가능성, 다른 드라이브의 원본에 hard link 실패(안전한 쪽), 재고 신호는 Hybrid만 받음(설계상 공정성 주의. 문서에는 γ 효과가 "Hybrid − ALS+BPR"에 섞인다는 말만 있어 §6.2에 baseline에는 없다는 점과 크기를 명시), 문서의 날짜 표기.

**C. 수정**
- 코드: git 상태를 읽지 못하면 거부, git 출력 UTF-8, launcher는 프로젝트 폴더에서만 실행하고 바로 끝난 명령을 알림.
- 테스트: `test_cycle18_guards.py` 5개. 가드 변이 H18 대상 이동, H66–H70.
- 문서: I-112 – I-117.

**D. 검증**
- 바뀌거나 새로 넣은 가드 변이(H14, H18, H63, H64, H66–H70)를 해당 테스트만으로 먼저 확인: 모두 검출, 수정 안 한 사본 27 통과(출력 저장 안 함). 70개 대상이 모두 한 번씩 있음.
- 작업 폴더 전체 테스트 165/165 (`audit/cycle18/precommit_pytest.txt`).
- 실제 launcher와 실제 저장소 확인 (`audit/cycle18/git_state_launch_check.txt`).
- 가드 변이 H1–H70을 커밋 0e80a80 코드로 실행: 70/70, 수정 안 한 사본 79 통과 (`audit/cycle18/guard_mutants.json`).

**E. 커밋**
- 0e80a80: 코드, 테스트, 가드 변이 (I-114 일부, I-115 – I-117)
- cf03fd7: 문서 정정(I-112 – I-114), Cycle 18 기록과 증거

**F. 커밋 후 재검증**
- 0e80a80: 새 worktree, lock venv 164 통과, 1 건너뜀. SHA256 v6 60/60, v7 28/28 (`audit/cycle18/postcommit_0e80a80_*.txt`).
- cf03fd7: 새 worktree, lock venv 164 통과, 1 건너뜀. SHA256 v6 60/60, v7 28/28 (`audit/cycle19/postcommit_cf03fd7_*.txt`).

## Cycle 19 (2026-10-23 KST): Cycle 18 커밋 뒤 다시 한 독립 감사

**A. 분석**
- 코드 감사자와 문서 감사자(둘 다 HEAD cf03fd7)를 새로 두고 처음부터 읽게 함. 보고 항목은 코드·커밋된 파일·실제 실행으로 다시 확인한 뒤에만 고침.

**B. 식별 (P0·P1 없음)**
- 코드 감사 P3 1건: I-118 GNN 선택 작업의 데이터 분할을 확인하는 장치가 없음(코드 두 줄에만 기댐. v6·v7 영향 없음).
- 문서 감사 P2 2건: I-119 §6.2의 재고 신호 효과를 NDCG로만 적음, I-120 재고 신호 비대칭이 원고 지침에 없음. P3 2건: I-121 재현 3단계의 dirty 검사 범위, I-122 날짜.
- 미확정 의심(결함 수에 넣지 않음): launcher의 10초 판정이 캐시가 빈 상태에서 넘칠 가능성, launcher의 10초 안 종료 코드 0 경로 테스트 없음, guard 도구의 git 출력 디코딩 오류 처리, 재고 효과는 F1 repeat·seed 42에서만 측정됨(§6.2에 한정으로 적음), H70은 launcher 폴더 검사만 덮음, Cycle 18 커밋 전 pytest 출력에 venv 기록 없음.

**C. 수정**
- 코드: `gnn_spec`의 선택 작업 역할 검사, `check_cached_runs`의 데이터 출처 비교.
- 테스트: `test_cycle19_guards.py` 3개, `test_cycle5_guards.py` 캐시 fixture에 manifest. 가드 변이 H71–H73.
- 문서: I-118 – I-122.

**D. 검증**
- 새 가드 변이 H71–H73을 `test_cycle19_guards.py`만으로, 상한 연장 줄 변이를 smoke 테스트로 먼저 확인: 모두 검출, 수정 안 한 사본 통과 (`audit/cycle19/quick_mutants.txt`). 73개 대상이 모두 한 번씩 있음.
- 작업 폴더 전체 테스트 168/168 (`audit/cycle19/precommit_pytest.txt`).
- 실제 v7 root에서 데이터 분할 검사와 음성 대조 (`audit/cycle19/gnn_data_split_check.txt`).
- 가드 변이 H1–H73을 커밋 d92de19 코드로 실행: 73/73, 수정 안 한 사본 82 통과 (`audit/cycle19/guard_mutants.json`).

**E. 커밋**
- d92de19: 코드, 테스트, 가드 변이 (I-118)
- 이 문서와 함께: 문서 정정(I-119 – I-122), Cycle 19 기록과 증거

**F. 커밋 후 재검증**
- d92de19: 새 worktree, lock venv 167 통과, 1 건너뜀. SHA256 v6 60/60, v7 28/28 (`audit/cycle19/postcommit_d92de19_*.txt`).
