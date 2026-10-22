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

- 확인: 새 venv에 위 명령으로 설치한 뒤 `uv pip freeze` 결과가 v7 venv와 한 줄도 다르지 않음 (2026-10-21. Cycle 6에 다시 확인: 두 venv 36개 패키지 동일, lock 파일과는 패키지 36개가 같고 주석 6줄과 index URL 두 줄만 다름, `audit/cycle6/freeze_pbda_v7.txt`, `freeze_pbda_lockcheck.txt`).
- uv는 기본적으로 첫 번째로 패키지를 찾은 index만 보므로, `--index-strategy unsafe-best-match`가 없으면 PyTorch index에서 `filelock` 버전을 못 찾아 실패함.

## 3. 처음부터 실행하는 방법

```powershell
cd Model/paper_revision_codex
# 1) 회귀 테스트 (pytest.ini가 수집 범위를 정함. 현재 160개)
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:MPLBACKEND='Agg'
python -m pytest -q -p no:cacheprovider
# 2) 누수 검사만 단독 실행 (전처리된 디렉터리 대상)
python audit_tools/leakage_check.py --processed <processed_dir> --raw ../../data --gnn --output audit_runs/repro/leak.json
#    rolling fold는 --raw에 그 fold의 fold_raw를 주거나, 원본 폴더와 함께 --end-fraction을 줌 (manifest에 끝 비율이 없음)
python audit_tools/leakage_check.py --processed audit_runs/v7/F4_unseen/prepared --raw ../../data --gnn --end-fraction 0.36
# 3) 사전등록 rolling-origin 실험 (git 작업 트리가 깨끗해야 시작됨). 셸과 분리해 실행
python report_tools/launch_detached.py --log audit_runs/v6_run.log -- python -u rolling_protocol.py run --root audit_runs/v6 --raw ../../data
python report_tools/launch_detached.py --log audit_runs/v7_run.log -- python -u rolling_protocol.py run --root audit_runs/v7 --raw ../../data --protocol v7
# 4)–11)의 출력은 gitignore 대상인 audit_runs/repro/에 쓰고, 표시한 커밋된 증거와 비교함 (커밋된 증거를 덮어쓰지 않도록. 5)·6) 도구의 기본 출력도 audit_runs/)
# 4) 커밋된 요약에서 그림과 표 다시 만들기 (모델 학습 없음. 비교: audit/figures_v7, 줄바꿈 외 같음)
python report_tools/plot_rolling_results.py --results audit/results_v7 --output audit_runs/repro/figures_v7
# 5) 변이 테스트 (변이마다 테스트 묶음 두 개를 다시 돌림. RTX 3090 Ti PC에서 약 50분. gitignore 대상 사본에서 실행. 비교: audit/mutation/mutation_results.json. Cycle 3 기준 실행은 70f5106에서 "약 50분, 다른 GPU 작업이 없을 때 실행"으로만 적혔고 시각 출력은 없다. 6364ad4부터 이 단계에 적힌 49분 38초는 Cycle 4 재실행(2026-10-21 15:42:12–16:31:50 UTC, audit/cycle4/mutation_cycle4_start.txt)의 시간이고, 그 재실행이 다른 작업과 겹쳤는지는 기록이 없다. 이후 재실행은 아래 작업과 겹쳤고(각 Cycle의 독립 감사 에이전트도 같은 시간대에 돌았을 수 있으나 그 실행 시각은 저장 안 됨) 결과는 모두 공식 결과와 바이트 단위로 같음: Cycle 8(343e75f, 20:14:17–21:08:52 UTC)·9(79b628d, 21:15:30–22:08:20)·10(ac9b865, 22:27:07–23:17:51)은 커밋 후 전체 테스트(b9e7134 20:38:42–20:44:29, ce97790 21:41:06–21:46:36, 4e166c5 23:06:45–23:13:08)와 같은 Cycle의 가드 변이, Cycle 11(8d8d0d5, 23:59:52–00:47:48)은 재현 실행(00:08:16–00:08:47, audit/cycle11/repro_run.txt), Cycle 12(2f1767c, 03:47:29–04:36:35)는 5026680 커밋 후 테스트(04:22:38–04:28:42)와 가드 변이(04:28:49 이후), Cycle 13(1e2b35e, 05:52:27–06:43:43)은 작업 폴더 전체 테스트(06:07–06:13, 셸 기록의 시각, 출력 저장 안 함), 가드 변이 첫 실행(06:06:59 이후, 셸 기록의 시각, 출력 저장 안 함), cb22505 커밋 후 테스트(06:15:38–06:22:35))
python report_tools/mutation_check.py --output audit_runs/repro/mutation_results.json
# 6) 가드 변이 테스트 (Cycle 5–17 가드의 옛 동작 65개를 사본에 하나씩 넣고 가드 테스트가 실패하는지 확인. 사본은 git 저장소 밖이라 git 이력을 읽는 verify_plan_code 테스트 하나와, 시간 때문에 끝까지 돌리는 재현 테스트 하나는 빼고 돌림. 출력 경로를 인자로 줌, 출력에 git HEAD와 커밋 안 된 파일 기록. Cycle 7에 셸 실행 시간 22분 14초, Cycle 8에 H1–H27을 36분 6초, Cycle 9에 H1–H35를 44분 53초, Cycle 10에 H1–H44를 52분 26초, Cycle 11에 H1–H48을 62분 55초, Cycle 12에 H1–H54를 76분 18초, Cycle 13에 H1–H60을 84분 18초, Cycle 14에 H1–H61을 85분 51초, Cycle 15에 H1–H62를 88분 46초, Cycle 16에 H1–H63을 91분 38초, Cycle 17에 H1–H65를 92분 44초(출력의 시작·종료 시각). Cycle 10까지는 공식 변이 테스트와(Cycle 8–10은 커밋 후 전체 테스트와도: b9e7134 20:38:42–20:44:29, ce97790 21:41:06–21:46:36, 4e166c5 23:06:45–23:13:08 UTC), Cycle 11은 커밋 후 전체 테스트·독립 코드 감사와, Cycle 12는 공식 변이 테스트(04:36 UTC까지)와, Cycle 13은 db3653e 커밋 후 전체 테스트(07:24:30–07:30:37 UTC)·Cycle 14 독립 코드 감사(cb22505 대상)와 동시 실행. Cycle 13의 첫 실행(06:06:59 UTC 시작(셸 기록의 시각, 출력 저장 안 함), H52에서 멈춤)은 커밋 전 작업 폴더 전체 테스트·공식 변이 테스트·cb22505 커밋 후 테스트와 겹쳤음. Cycle 14는 41ba20d 커밋 후 전체 테스트(09:03:38–09:09:53 UTC)·Cycle 14 독립 문서 감사와, Cycle 15는 4c1ddfb 커밋 후 전체 테스트(10:50:06–10:56:21 UTC)와, Cycle 16은 6ce754d 커밋 후 전체 테스트(12:40:38–12:46:37 UTC)와, Cycle 17은 cf1a3fe 커밋 후 전체 테스트(14:40:21–14:46:25 UTC)와 동시 실행. 비교: audit/cycle17/guard_mutants.json)
python report_tools/guard_mutation_check.py audit_runs/repro/guard_mutants.json
# 7) 끝난 root의 캐시가 plan·selection이 요청할 설정과 같은지 (학습·정답 읽기 없음.
#    상한 연장 대상은 validation_gnn.json에서 읽고 규칙으로 다시 유도하지 않음. F1 IF ablation 재학습은 대상 아님)
python report_tools/check_cached_runs.py --root audit_runs/v7 --output audit_runs/repro/cache_identity_v7.json   # 비교: audit/cycle5/cache_identity_v7.json
# 8) 학습 설정도 정책별로 골랐다면 validation 점수가 어떻게 되는지 (커밋된 선택 파일만 읽음, I-63)
python report_tools/policy_tuning_gap.py --root audit/results_v7 --output audit_runs/repro/policy_tuning_gap_v7.json   # 비교: audit/cycle8/policy_tuning_gap_v7.json (cycle7 판에는 note 필드가 없음)
# 9) 실제 원본으로 시험 열람 가드 확인 (학습·시험 읽기·로그 쓰기 없음)
python report_tools/guard_real_data_check.py --raw ../../data --output audit_runs/repro/guard_real_data.json   # 비교: audit/cycle10/guard_real_data.json (cycle9 판에는 decisions가 없음)
# 10) 산출물에 찍힌 학습 코드 해시가 plan과 같은지, 기록된 Hybrid grid를 다시 읽은 값이 선택과 같은지 (validation 파일만 읽음)
python report_tools/stamp_code_check.py --root audit_runs/v7 --output audit_runs/repro/stamp_code_v7.json   # 비교: audit/cycle9/stamp_code_v7.json
python report_tools/grid_cache_precision_check.py --root audit_runs/v7 --output audit_runs/repro/grid_cache_precision_v7.json   # 비교: audit/cycle8/grid_cache_precision_v7.json
# 11) 다시 만든 출력과 커밋된 증거 비교 (4, 7–10단계와 아래 plan 코드 확인·재계산. plan 코드 확인은 시간이 지나며 늘어나는 목록을 빼고 비교)
python report_tools/compare_outputs.py --output audit_runs/repro/compare.json audit_runs/repro/figures_v7=audit/figures_v7 audit_runs/repro/cache_identity_v7.json=audit/cycle5/cache_identity_v7.json audit_runs/repro/policy_tuning_gap_v7.json=audit/cycle8/policy_tuning_gap_v7.json audit_runs/repro/guard_real_data.json=audit/cycle10/guard_real_data.json audit_runs/repro/stamp_code_v7.json=audit/cycle9/stamp_code_v7.json audit_runs/repro/grid_cache_precision_v7.json=audit/cycle8/grid_cache_precision_v7.json audit_runs/repro/recompute_v7_F4_unseen.json=audit/recompute/recompute_v7_F4_unseen.json "audit_runs/repro/plan_code_v7.json=audit/plan_code/plan_code_v7.json#commit,scheme,files,all_plan_hashes_match_commit"
#    Cycle 11 실행(커밋 8d8d0d5의 도구): 29행(다시 만든 출력 28개. plan 코드 확인 출력 하나를 Cycle 3·Cycle 6 증거 두 곳과 비교했고, 11단계 명령은 Cycle 3 쪽만 비교해 28행) 모두 같음 (audit/cycle11/repro_compare.json, repro_run.txt)
#    Cycle 13 실행(cb22505로 커밋된 수정이 들어간 커밋 직전 작업 폴더의 도구(06:14:11 UTC, 커밋 06:14:59), Cycle 11에 다시 만든 같은 출력): 28행 중 I-89 전에 만든 v7 `pipeline.*` 3개만 다름, 커밋 파일이 없는 쌍 없음 (audit/cycle13/compare_repro.json). `#keys`의 키가 한쪽에라도 없거나 JSON 객체가 아니면 오류로 멈춤(I-98)
```

- 3단계는 중간에 멈춰도 같은 명령으로 이어서 실행할 수 있음. 완료된 산출물은 다시 만들지 않음.
- 코드나 plan이 바뀐 상태로 같은 `--root`를 쓰면 오류를 내고 멈춤. 코드는 실행 시작 때뿐 아니라 GNN 작업을 시작하기 직전과 새 시험 열람 직전에도 확인함(Cycle 8 I-70). 그래서 실행 중에 코드를 고치면 다음 GNN 작업이나 시험 열람에서 멈춤.
  - 코드 해시는 `Model/paper_revision_codex/*.py`(테스트 파일 포함)와 `audit_tools/*.py`를 대상으로 함. `report_tools/`는 포함하지 않음.
  - 따라서 v6/v7을 처음부터 다시 실행하려면 각 plan의 commit(v6 `29ba891`, v7 `1170ea5`)을 checkout해야 함. 이후 commit에서는 코드와 테스트 파일이 바뀌어 해시가 다름. 새 `--root`로 실행하면 그 checkout 기준으로 새 `plan.json`이 만들어지고, plan 내용이 같으면 `plan_sha256`도 같음.
  - plan commit의 코드는 Hybrid 끝 표시(`hybrid_at_grid_boundary`)를 I-97 이전 규칙(α 없음, 후보 수 끝은 500과 `all`)으로 쓰므로, 다시 실행한 `selection.json`의 이 플래그는 커밋된 파일과 같고 정정된 값과는 다르다. 정정된 값은 `audit/cycle13/hybrid_boundary_recheck.json`. 선택 결정(`selection_decision`)에는 이 플래그가 없다.
  - **줄바꿈 (I-38)**: v6/v7 plan은 작업 폴더의 원시 바이트를 해시함. 같은 commit이라도 checkout의 줄바꿈(`core.autocrlf`)이 다르면 해시가 달라, 기존 `audit_runs/v6`·`v7`의 plan으로 재개할 때 거부될 수 있음. 이후 만드는 plan은 줄바꿈을 정규화한 해시(`code_hash_scheme: sha256-lf`)를 씀.
  - 기존 plan의 코드가 commit 내용과 같은지는 줄바꿈과 무관하게 확인할 수 있음. v6 25개, v7 26개 모두 일치 (`audit/plan_code/`):

```powershell
python report_tools/verify_plan_code.py --plan audit/results_v7/plan.json --output audit_runs/repro/plan_code_v7.json
```
  - 종료 코드 0은 "plan 해시가 plan commit의 내용과 맞음"이라는 뜻이다. 그 뒤 코드가 바뀌었는지는 `changed_since_plan`에만 나온다. 바뀐 경우도 실패로 처리하려면 `--require-unchanged`를 붙인다 (v6·v7 모두 실패함. plan 뒤에 바뀐 파일은 `changed_since_plan`, 새 테스트 파일은 `added_since_plan`에 나오며 커밋마다 늘어남. 5026680 기준으로 v7은 `rolling_protocol.py`·`data_protocol.py`·`experiments.py`·`audit_tools/leakage_check.py`, v6은 여기에 `test_v6_smoke.py`가 바뀜. `audit/plan_code/`는 Cycle 3, `audit/cycle6/plan_code_v*.json`은 커밋 57dc03b 시점 결과다. 테스트 파일이 추가될 때마다 `added_since_plan`이 늘어나므로 현재 상태는 위 명령으로 다시 확인한다. 커밋된 파일과는 `commit`, `scheme`, `files`, `all_plan_hashes_match_commit`만 비교함: 위 11단계).
- 시험 열람 가드 (Cycle 4 I-40, Cycle 5 I-48 – I-52, Cycle 6 I-59, Cycle 7 I-64 – I-67, Cycle 8 I-70 – I-73, Cycle 9 I-75 – I-79, Cycle 10 I-81 – I-84):
  - 같은 `--root`에서는 plan이 바뀌었거나 `selection.json`이 다시 만들어졌으면(해시가 다르면) 이미 연 fold의 시험을 다시 열 수 없다. 시험 평가와 사후 분석 모두 같은 plan·selection이어야 하고, 이미 끝난 단계도 재개할 때 이 검사를 거친다. `selection.json`을 만든 plan이 현재 plan과 다르면 멈춘다.
  - 다른 `--root`에서도, `audit/`(커밋된 결과)와 `audit_runs/`(이 PC의 실행 폴더) 아래의 plan·접근 로그 중 같은 `events.csv`(시험 구간을 정하는 파일. 줄바꿈만 다른 사본도 같은 것으로 봄)로 겹치는 시험 구간(바깥 시험 구간 (end·(1−test_size), end)이 겹침, 맞닿기만 하면 아님. 시험 열람 직전에는 아래처럼 실제 구간으로 다시 판단)을 연 기록이 있으면 실행을 시작하지 않는다. 같은 검사를 시험을 열기 직전에도 다시 한다(Cycle 7 I-64, 병렬로 시작한 다른 root 대비). 예외는 plan, 코드, 원본 4개 파일이 모두 같은 재현뿐이다(I-65). 시험을 열기 직전에는 그 재현 기록이 같은 fold를 같은 결정의 selection으로 열었어야 한다(I-75, I-81). 결정은 `selection_decision`이 `selection.json`에서 뽑는 내용(MF·GNN·Hybrid 백본 설정과 epoch, 정책별 MF 가중과 Hybrid 추론 설정)이고, 봉인 시각·절대경로·학습 시간·validation 점수는 빠진다. 상대 기록의 결정은 그 fold 폴더의 `selection.json`에서 읽으며, 그 파일의 해시가 접근 로그에 적힌 열람 해시와 모두 같을 때만 쓴다. 결정이 다르면(예: GPU 비결정성으로 다른 설정이 선택됨) 학습을 마친 뒤 시험 열람 직전에 거부된다. 실행 시작 때는 selection이 아직 없어 미리 알 수 없다. 시험을 열기 직전의 검사에서는 실행 중인 root 자신은 제외한다(자기 열람은 접근 로그가 관리, I-71). 다른 root의 접근 로그에 읽을 수 없는 줄(쓰다 끊김)이 있으면 그 root의 모든 fold를 연 것으로 보고, 읽을 수 없는 `plan.json`은 파일을 지목하는 오류로 멈춘다(I-73). 구간 겹침은 실행 시작 때는 명목 비율(위 식)로, 시험을 열기 직전에는 기록된 실제 구간(이벤트 타임스탬프: `fold_raw/fold_raw.json`의 끝, `prepared/manifest.json` 또는 `leakage_check.json`의 분할 시점)으로 판단한다. 분할이 이벤트 index에서 잘려 명목상 맞닿는 구간도 경계 이벤트를 공유할 수 있기 때문이다(I-79). 실제 구간 기록이 없거나 그 값이 정수가 아닌 상대 root는 맞닿기만 해도 겹침으로 본다(I-82). 그래서 v6·v7은 현재 코드로는 다시 열 수 없고, 각 plan commit을 checkout해야 한다.
  - 가드가 보지 못하는 경우: 두 root가 거의 같은 순간에 확인하고 기록하는 경쟁(파일 잠금 없음), 다른 PC의 커밋되지 않은 기록, 프로젝트 폴더의 `audit/`·`audit_runs/` 바로 아래가 아닌 곳에 만든 root, 내용(행)이 바뀐 `events.csv`. 정확한 재현(같은 plan·코드)은 새 root에서 여러 번 열 수 있으므로, 결과를 고르지 않도록 재현 root도 접근 로그와 함께 보고해야 한다. 이 예외는 장치(CPU/GPU)와 라이브러리 버전을 보지 않으므로 다른 장치의 재현은 BPR·GNN 수치가 다를 수 있다(I-34, I-94). 장치는 각 root의 `plan.json` `environment`에 남고, 보고하는 수치는 처음 봉인된 실행(커밋본)이다.
  - GNN 작업은 MF 단계와 동시에 백그라운드에서 돈다. MF 단계가 실패하면 새 GNN 작업을 띄우지 않고 실행 중인 작업을 종료한 뒤 오류를 낸다(I-83). 종료된 작업은 `job_result.json`이 없어 재개 때 다시 학습한다. GNN 작업의 오류는 MF 단계가 끝난 뒤에 드러난다. Windows에서는 각 GNN 작업을 KILL_ON_JOB_CLOSE job object에 넣으므로, 실행 프로세스가 강제로 끝나도(`taskkill /F`, 작업 관리자) 작업이 함께 끝난다(I-87). 작업을 띄운 직후 job object에 넣기까지 약 10ms(Cycle 12 코드 감사자가 잰 값: 중앙값 9.6ms, 최대 16.9ms, 출력 저장 안 함) 사이에 실행이 강제로 끝나면 그 작업은 남을 수 있고, 넣기 전에 이미 끝난 작업은 종료 코드로 실패를 알린다(I-93). POSIX에는 이 보호가 없다. GNN 작업은 실행의 프로세스 그룹을 이어받고 `report_tools/launch_detached.py`는 POSIX에서 실행을 새 세션(그룹 리더)으로 띄우므로, 그룹 전체를 끝내면(`kill -- -<pid>`) 작업도 함께 끝난다. 실행 프로세스만 강제로 끝냈다면 재개 전에 남은 `rolling_protocol.py gnn-job` 프로세스를 직접 끝낸다. `--folds`에 이 protocol에 없는 이름을 주거나 이름 없이 주면 plan을 만들기 전에 멈춘다(I-84, I-91).
  - 전처리 폴더는 바깥 `manifest.json`이 마지막에 쓰이므로, 중간에 끊긴 전처리는 재개 때 "Incomplete preprocessing output"으로 멈춘다. 그 `prepared` 폴더를 지우고 다시 실행한다(I-90).
  - 공유 데이터의 hard link 복제(`experiments.clone_protocol`)는 `<이름>.partial`에 만든 뒤 한 번에 옮기므로, 끊긴 복제가 완성본으로 남지 않는다(I-86). 끊긴 `.partial` 폴더는 다음 실행이 지우고 다시 만든다.
  - 캐시 재사용 조건에 장치(CPU/GPU)는 들어가지 않는다. 다른 장치에서 재개하면 두 장치의 산출물이 섞일 수 있으므로(I-34) 같은 장치에서 재개한다. 장치는 plan `environment`와 각 GNN `job_result.json`에 남는다.
  - 재개할 때마다 원본 4개 파일을 다시 해시해 plan의 `raw_input_sha256`과 다르면 멈춘다. 캐시된 MF 학습과 GNN 작업은 설정(seed, 하이퍼파라미터, epoch)이 요청과 같을 때만 재사용한다.
  - 실행 중 누수 검사는 fold의 잘린 원본이 아니라 원본 전체 + fold `end_fraction`으로 한다 (fold 자르기도 독립적으로 확인).
- `launch_detached.py`가 필요한 이유 (Windows): 셸의 Ctrl+C·창 닫기 이벤트가 v6 실행을 두 번 끊었고, `DETACHED_PROCESS`로 띄우면 BLAS 안의 Intel Fortran 런타임이 `forrtl: error (200)`으로 중단됨 (I-31). launcher는 별도 프로세스 그룹, job 분리, 숨은 자체 콘솔을 쓰고 `FOR_DISABLE_CONSOLE_CTRL_HANDLER=1`, BLAS 스레드 1을 설정함.

## 4. 결정성과 seed

- seed {42, 43, 44}를 ALS(`implicit` CPU), BPR(torch), GNN(torch CUDA), numpy RNG에 같은 값으로 전달함.
- 하이퍼파라미터 선택에는 seed 42만 씀 (`selection_seed`). 시험 보고에는 세 seed의 평균과 표준편차를 씀.
- BPR 체크포인트 콜백이 학습 결과를 바꾸지 않음을 확인함 (CPU, 비트 단위 동일: `test_bpr_checkpoints_do_not_change_training`).
- GPU의 sparse 행렬곱은 결정성이 보장되지 않음. 같은 seed로 두 번 학습한 결과 차이는 `validation_report.md`의 결정성 절에 기록함.
- **ALS는 BLAS 스레드 수에 따라 결과가 달라짐** (I-24). `rolling_protocol.py`는 numpy를 import하기 전에 `OPENBLAS/OMP/MKL/NUMEXPR_NUM_THREADS`를 1로 강제한다.
  - 모델을 학습하거나 점수를 계산하는 보조 스크립트 9개(`benchmark_inference`, `check_fast_validation`, `check_score_precision`, `component_ablation`, `cross_machine_repro`, `determinism_check`, `device_check`, `mf_boundary_probe`, `plot_hybrid_results`)도 numpy보다 먼저 1로 강제한다 (I-55). `launch_detached`·`mutation_check`는 자식 프로세스에 1을 준다. 나머지(그림, 저장된 목록으로 지표 재계산, 해시 확인 등)는 ALS를 학습하지 않아 설정하지 않는다.
  - v6·v7 실행 시점의 `rolling_protocol.py`는 `setdefault`를 써서, 이 변수가 이미 다른 값이면 그 값이 우선했다. 나중에 import되는 `experiments.py`가 변수를 1로 다시 쓰지만 BLAS는 이미 로드된 뒤라, 변수 값만 보고는 실제 스레드 수를 알 수 없다 (I-41). v6·v7 모두 실제 스레드 수는 기록되지 않았다.
  - v7은 커밋되기 전의 분리 실행 launcher(gitignore 대상인 `audit_runs/`에만 있었고 지금은 없음)로 실행했다. `launch_detached.py`는 v7이 끝난 뒤 그 동작을 옮겨 커밋한 것이다 (e9538c8). 대신 결과로 확인했다: v7 F4 seed 42의 ALS를 같은 설정으로 다시 학습하면 BLAS 1스레드에서는 저장된 v7 인자와 비트 단위로 같고, 24스레드에서는 최대 차이 0.49, Top-50 목록 완전 일치 1.3%(평균 겹침 98.6%)로 다르다 (`audit/cycle5/determinism_v7_blas1.csv`, `determinism_v7_blas24.csv`). 따라서 v7 ALS는 1스레드 결과와 일치한다(2–23스레드는 검사 안 함).
  - v6의 실행 스레드도 기록되지 않았다. 다른 PC에서 1스레드로 다시 학습한 F3 ALS grid 24행이 v6 값과 비트 단위로 같아(`validation_report.md` §9) 1스레드였음을 뒷받침한다.
  - Cycle 4부터 `rolling_protocol.py`는 plan의 `environment`에 변수 값, 로드된 BLAS/OpenMP 라이브러리의 실제 스레드 수(`native_threads`), torch 스레드 수, 장치를 기록한다. Cycle 5부터는 실행을 시작할 때마다 `run_started` 로그에도 실제 스레드 수를 남긴다.
  - 1스레드: 같은 설정으로 다시 학습한 ALS 인자가 비트 단위로 같음.
  - 24스레드: validation 점수가 최대 0.0013 달라짐.

```powershell
python report_tools/determinism_check.py --root audit_runs/v6 --fold F3_early --output <dir>
# v7 ALS가 몇 스레드 결과와 맞는지 (GNN 제외, BLAS 스레드 지정)
python report_tools/determinism_check.py --root audit_runs/v7 --fold F4_unseen --skip-gnn --blas-threads 24 --output <dir>
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
python report_tools/recompute_test_metrics.py --root audit_runs/v7 --fold F4_unseen --output audit_runs/repro/recompute_v7_F4_unseen.json   # 비교: audit/recompute/recompute_v7_F4_unseen.json
```

## 7. 알려진 재현성 한계

1. 라이브러리 버전이 바뀌면서 감사용 요약 `if_feature_distributions.csv`의 해시가 달라짐. 핵심 분할 파일은 바이트 단위로 같음 (I-17).
2. `selection.json`의 `job_dir`는 절대경로임 (I-20). 다른 PC에서는 `refits.json`의 상대경로를 쓰면 됨.
3. v1–v5 결과는 다른 PC의 절대경로를 포함하고, v5는 v3의 학습 산출물을 재사용함. 원래 환경과 산출물 없이는 v5를 비트 단위로 재현할 수 없음. v6가 이를 대체하는 일괄 재실행임.
4. BPR·GNN(GPU)은 GPU·torch 빌드·장치(CPU/GPU)에 따라 값이 달라짐 (I-19, I-33, I-34). 다른 GPU로 다시 실행하면 소수 셋째 자리 수준의 차이를 예상해야 함.
5. v6 실행 폴더(`audit_runs/v6`, 추천 목록 포함)는 v6 PC에만 있음. 이 PC에서는 v6 시험 지표를 추천 목록으로 다시 계산할 수 없음 (v6 지표 교차 계산은 Cycle 2에 v6 PC에서 함).
