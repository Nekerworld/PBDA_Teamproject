# 재현성 (Reproducibility)

## 1. 입력 데이터

원본 RetailRocket 파일 4개는 `<repo>/data/`에 있으며 저장소에는 포함되지 않음 (`.gitignore`). v6 `plan.json`의 `raw_input_sha256`에 해시를 기록함. 이 해시는 v1–v5 manifest의 `input_hashes`와 같음 (이 PC에서 확인).

| 파일 | 용도 |
|---|---|
| `events.csv` | 이벤트 (view / addtocart / transaction) |
| `item_properties_part1.csv`, `item_properties_part2.csv` | 시점별 상품 속성 (`available`, `categoryid` 등) |
| `category_tree.csv` | 정적 카테고리 트리 |

## 2. 환경

| 항목 | v6 실행 PC (2026-10-20) | v7 실행 PC (2026-10-21) | v1–v5 원래 실행 |
|---|---|---|---|
| Python | 3.11.9 | 3.11.9 | 3.12.3 |
| numpy | 2.4.6 | 2.4.6 | 2.1.3 |
| pandas | 3.0.6 | 3.0.6 | 2.3.0 |
| scikit-learn | 1.9.0 | 1.9.0 | 1.6.1 |
| scipy / implicit | 1.17.1 / 0.7.3 | 1.17.1 / 0.7.3 | v5 `experiment_plan.json` |
| torch / CUDA / GPU | 2.15.0.dev20260921+cu132 / 13.2 / RTX 5060 Ti | 2.11.0+cu128 / 12.8 / RTX 3090 Ti | v5 `experiment_plan.json` |

실행 환경 전체는 `audit/results_v6/plan.json`, `audit/results_v7/plan.json`의 `environment`에 기록됨.

- v6 PC의 torch는 개발 빌드(nightly)임. v7 PC는 정식 빌드(torch 2.11.0+cu128)를 저장소 밖에 새로 만든 venv에 설치함.
- GPU·torch 빌드가 다르면 BPR·GNN 결과가 소수점 아래에서 달라짐 (§4).
- **`requirements.txt`는 scikit-learn 1.9.1을 고정하지만 v6·v7은 모두 1.9.0으로 실행됨** (I-36). v6·v7 재현에는 v7 venv를 그대로 적은 `requirements-v7-lock.txt`를 씀.

venv 만들기:

```powershell
uv venv --python 3.11.9 <venv>
uv pip install --python <venv>\Scripts\python.exe --index-strategy unsafe-best-match -r requirements-v7-lock.txt
# 또는: python -m venv <venv>; <venv>\Scripts\python -m pip install -r requirements-v7-lock.txt
```

- 확인: 새 venv에 위 명령으로 설치한 뒤 `uv pip freeze` 결과가 v7 venv와 한 줄도 다르지 않음 (2026-10-21).
- uv는 기본적으로 첫 번째로 패키지를 찾은 index만 보므로, `--index-strategy unsafe-best-match`가 없으면 PyTorch index에서 `filelock` 버전을 못 찾아 실패함.

## 3. 처음부터 실행하는 방법

```powershell
cd Model/paper_revision_codex
# 1) 회귀 테스트 (pytest.ini가 수집 범위를 정함. 현재 82개)
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:MPLBACKEND='Agg'
python -m pytest -q -p no:cacheprovider
# 2) 누수 검사만 단독 실행 (전처리된 디렉터리 대상)
python audit_tools/leakage_check.py --processed <processed_dir> --raw ../../data --gnn --output leak.json
# 3) 사전등록 rolling-origin 실험 (git 작업 트리가 깨끗해야 시작됨). 셸과 분리해 실행
python report_tools/launch_detached.py --log audit_runs/v6_run.log -- python -u rolling_protocol.py run --root audit_runs/v6 --raw ../../data
python report_tools/launch_detached.py --log audit_runs/v7_run.log -- python -u rolling_protocol.py run --root audit_runs/v7 --raw ../../data --protocol v7
# 4) 커밋된 요약에서 그림과 표 다시 만들기 (모델 학습 없음, 바이트 단위로 같은 결과)
python report_tools/plot_rolling_results.py --results audit/results_v7 --output audit/figures_v7
# 5) 변이 테스트 (약 50분, gitignore 대상 사본에서 실행. 다른 GPU 작업이 없을 때 실행)
python report_tools/mutation_check.py
```

- 3단계는 중간에 멈춰도 같은 명령으로 이어서 실행할 수 있음. 완료된 산출물은 다시 만들지 않음.
- 코드나 plan이 바뀐 상태로 같은 `--root`를 쓰면 오류를 내고 멈춤.
  - 코드 해시는 `Model/paper_revision_codex/*.py`(테스트 파일 포함)와 `audit_tools/*.py`를 대상으로 함. `report_tools/`는 포함하지 않음.
  - 따라서 v6/v7을 처음부터 다시 실행하려면 각 plan의 commit(v6 `29ba891`, v7 `1170ea5`)을 checkout해야 함. 이후 commit에서는 코드와 테스트 파일이 바뀌어 해시가 다름. 새 `--root`로 실행하면 그 checkout 기준으로 새 `plan.json`이 만들어지고, plan 내용이 같으면 `plan_sha256`도 같음.
  - **줄바꿈 (I-38)**: v6/v7 plan은 작업 폴더의 원시 바이트를 해시함. 같은 commit이라도 checkout의 줄바꿈(`core.autocrlf`)이 다르면 해시가 달라, 기존 `audit_runs/v6`·`v7`의 plan으로 재개할 때 거부될 수 있음. 이후 만드는 plan은 줄바꿈을 정규화한 해시(`code_hash_scheme: sha256-lf`)를 씀.
  - 기존 plan의 코드가 commit 내용과 같은지는 줄바꿈과 무관하게 확인할 수 있음. v6 25개, v7 26개 모두 일치 (`audit/plan_code/`):

```powershell
python report_tools/verify_plan_code.py --plan audit/results_v7/plan.json --output audit/plan_code/plan_code_v7.json
```
- 이미 열린 fold의 시험을 다른 plan으로 다시 열 수 없음.
- `launch_detached.py`가 필요한 이유 (Windows): 셸의 Ctrl+C·창 닫기 이벤트가 v6 실행을 두 번 끊었고, `DETACHED_PROCESS`로 띄우면 BLAS 안의 Intel Fortran 런타임이 `forrtl: error (200)`으로 중단됨 (I-31). launcher는 별도 프로세스 그룹, job 분리, 숨은 자체 콘솔을 쓰고 `FOR_DISABLE_CONSOLE_CTRL_HANDLER=1`, BLAS 스레드 1을 설정함.

## 4. 결정성과 seed

- seed {42, 43, 44}를 ALS(`implicit` CPU), BPR(torch), GNN(torch CUDA), numpy RNG에 같은 값으로 전달함.
- 하이퍼파라미터 선택에는 seed 42만 씀 (`selection_seed`). 시험 보고에는 세 seed의 평균과 표준편차를 씀.
- BPR 체크포인트 콜백이 학습 결과를 바꾸지 않음을 확인함 (CPU, 비트 단위 동일: `test_bpr_checkpoints_do_not_change_training`).
- GPU의 sparse 행렬곱은 결정성이 보장되지 않음. 같은 seed로 두 번 학습한 결과 차이는 `validation_report.md`의 결정성 절에 기록함.
- **ALS는 BLAS 스레드 수에 따라 결과가 달라짐** (I-24). `rolling_protocol.py`와 `report_tools/*.py`는 numpy를 import하기 전에 `OPENBLAS/OMP/MKL/NUMEXPR_NUM_THREADS`를 1로 설정함. 이 변수가 이미 다른 값으로 설정돼 있으면 그 값이 우선하므로, 실행 전에 1로 맞춰야 함.
  - 1스레드: 같은 설정으로 다시 학습한 ALS 인자가 비트 단위로 같음.
  - 24스레드: validation 점수가 최대 0.0013 달라짐.

```powershell
python report_tools/determinism_check.py --root audit_runs/v6 --fold F3_early --output <dir>
```

- **다른 PC** (I-33, `validation_report.md` §9): 새 clone과 새 venv로 v6 F3 전처리·누수 검사·MF validation grid를 다시 실행함. 전처리와 ALS는 비트 단위로 같고, BPR(GPU)은 GPU·torch 빌드 차이로 선택 점수가 최대 0.0047 다름 (선택 결과는 같음).
- **CPU와 GPU** (I-34, §10): 같은 seed라도 장치가 다르면 GNN·BPR이 다른 모델로 학습됨 (목록 겹침 84% / 28%). v6·v7 결과는 모두 CUDA에서 얻음.

```powershell
python report_tools/cross_machine_repro.py --help
python report_tools/device_check.py --help
```

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
- 요약 표는 `audit/results_v6/`, `audit/results_v7/`에 복사해 추적하고, SHA256을 각 폴더의 `SHA256SUMS.txt`에 기록함. 두 폴더 모두 `.gitattributes`의 `* -text`로 줄바꿈 변환을 막음 (I-27).
- 확인: 각 폴더에서 `sha256sum -c SHA256SUMS.txt` (Git Bash) 또는 `Get-FileHash -Algorithm SHA256` 값을 `SHA256SUMS.txt`와 비교.
- v7 시험 지표는 저장된 추천 목록과 원본 `events.csv`로 다시 계산할 수 있음 (`audit_runs/v7`이 있어야 함):

```powershell
python report_tools/recompute_test_metrics.py --root audit_runs/v7 --fold F4_unseen --output audit/recompute/recompute_v7_F4_unseen.json
```

## 7. 알려진 재현성 한계

1. 라이브러리 버전이 바뀌면서 감사용 요약 `if_feature_distributions.csv`의 해시가 달라짐. 핵심 분할 파일은 바이트 단위로 같음 (I-17).
2. `selection.json`의 `job_dir`는 절대경로임 (I-20). 다른 PC에서는 `refits.json`의 상대경로를 쓰면 됨.
3. v1–v5 결과는 다른 PC의 절대경로를 포함하고, v5는 v3의 학습 산출물을 재사용함. 원래 환경과 산출물 없이는 v5를 비트 단위로 재현할 수 없음. v6가 이를 대체하는 일괄 재실행임.
4. BPR·GNN(GPU)은 GPU·torch 빌드·장치(CPU/GPU)에 따라 값이 달라짐 (I-19, I-33, I-34). 다른 GPU로 다시 실행하면 소수 셋째 자리 수준의 차이를 예상해야 함.
5. v6 실행 폴더(`audit_runs/v6`, 추천 목록 포함)는 v6 PC에만 있음. 이 PC에서는 v6 시험 지표를 추천 목록으로 다시 계산할 수 없음 (v6 지표 교차 계산은 Cycle 2에 v6 PC에서 함).
