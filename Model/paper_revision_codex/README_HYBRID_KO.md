# Hybrid 개선 실험 사용 안내

> **감사 후 안내 (2026-10-21)**: 이 문서는 v3–v5 실험 경로(`improve_hybrid.py`, `robust_hybrid.py`)를 설명한다. 이 경로는 같은 시험 구간을 반복해서 열었다 (`audit/issue_register.md` I-01). 사전등록 평가(`rolling_protocol.py`, v6/v7)의 실행 방법은 `audit/reproducibility.md`를 본다.

새 PC에서 처음 설치하고 데이터부터 준비할 때는 [REPRODUCE_KO.md](REPRODUCE_KO.md)를 먼저 따른다. 아래 명령은 이미 준비된 데이터가 있는 환경의 실행 예시다.

## 먼저 볼 파일

- `EXECUTION_PLAN_KO.md`: 실제 실행에 앞서 고정한 계획과 목표
- `professor_materials/professor_briefing_ko.pdf`: 교수님 회의용 보고서
- `professor_materials/professor_briefing_ko.md`: 수정 가능한 설명 본문
- `professor_materials/figures`: 실제 수치 기반 고해상도 PNG 및 벡터 PDF SVG
- `results/hybrid_joint_8020_v5/result.json`: 성공 기준과 최종 판정
- `results/hybrid_joint_8020_v5/model_summary.csv`: 세 seed 평균과 표본 표준편차
- `professor_materials/verification.json`: 추천 CSV에서 독립적으로 재계산한 검증 결과

## 실행

기존의 `experiments.py`는 과거 90:10 실험의 진입점입니다. 공통 80:20 첫 개선 실험은 `improve_hybrid.py`, 세 seed validation으로 선택을 보강한 최종 실험은 `robust_hybrid.py`입니다. 과거 결과와 새 결과는 섞지 않습니다. 실제 실행 환경의 버전은 결과의 runtime_environment.json에 있습니다. reportlab·pypdfium2를 쓰는 문서 도구는 별도 문서 환경에서도 실행할 수 있습니다.

학습 환경에서 이 폴더로 이동한 뒤 다음과 같이 실행합니다.

```powershell
python improve_hybrid.py --source-dir ../../data/research_revision_paper/datasets/no_if_events --output-dir results/reproduce_v3 --max-hours 24
python robust_hybrid.py --parent results/reproduce_v3 --output-dir results/reproduce_v5 --max-hours 8
```

준비 데이터는 원시 데이터의 공통 시간순 80:20 분할과 내부 시간순 validation을 포함합니다. 처음 읽을 때 manifest와 데이터 해시를 검사합니다. 같은 데이터의 불변 파일은 하드링크로 연결하지만 원본을 수정하지 않습니다. 코드나 설정이 달라지면 기존 출력 디렉터리를 재사용하지 않습니다. 준비 데이터가 없으면 data_protocol.UnifiedPreprocessor에 data_dir, 새 processed_dir, test_size=0.2, validation_size=0.2, use_if=False, core_mode='events'를 전달해 run()합니다. min_user_events와 min_item_events는 각각 3입니다.

검사:

```powershell
python -m pytest test_revision.py test_hybrid_improvement.py test_codex_audit.py test_robust_hybrid.py test_fast_fusion.py -q
python report_tools/verify_hybrid_delivery.py --experiment results/hybrid_joint_8020_v5 --output professor_materials/verification.json
```

Windows에서 pytest의 `--basetemp`를 지정한다면 상위 폴더를 먼저 만드세요. 샌드박스에서 하드링크가 거부될 때는 정상 로컬 Python 환경에서 검사를 실행해야 합니다.

다른 위치로 코드를 옮긴 경우 PBDA_PROJECT_ROOT 환경 변수를 원시 data 폴더가 있는 프로젝트 루트로 지정하세요. 배포 ZIP에는 큰 모델 배열과 원시 데이터를 넣지 않았습니다. 이 기계의 전체 결과 폴더에는 모델·데이터·사용자별 추천 CSV가 남아 있으며, 다른 기계에서는 준비 데이터와 모델을 다시 생성해야 합니다.

그림과 보고서:

```powershell
python report_tools/benchmark_inference.py --experiment results/hybrid_joint_8020_v5
python report_tools/plot_hybrid_results.py --experiment results/hybrid_joint_8020_v5 --evidence professor_materials/evidence.json --output professor_materials/figures
python report_tools/build_professor_report.py --experiment results/hybrid_joint_8020_v5 --materials professor_materials
```

## 변경 내용

최종 MF 점수에서 BPR 신호를 보존했습니다. MF 후보 합집합은 ALS와 BPR의 할당 후보를 남긴 채 부족한 수만 채웁니다. MF 융합 비중, 후보 수, 재정렬 비중과 재고 가중은 내부 validation에서만 선택합니다. 선택 기준은 NDCG와 Recall의 상대 개선율 중 작은 값입니다.

기존 ReRanker의 als_weight와 CLI --als-weight 이름은 호환성을 위해 남겼습니다. 기본 mf_score_source='ensemble'에서는 최종 ALS+BPR 결합 점수의 비중이며 ALS 단독 비중이 아닙니다. 새 실험의 selection.json에서는 같은 역할을 omega로 기록합니다. Python API에서 mf_score_source='als'를 명시하면 과거 ALS-only 통제 조건을 실행합니다.

GNN은 64차원 이분그래프를 사용합니다. 균등 음성과 학습 MF 기반 혼합 음성을 비교합니다. 모든 표본에서 과거 관측 상품을 제외하고, MF 모델·그래프·속성 시점도 해당 학습 구간 안으로 제한합니다.

`source_before_hybrid_20261017`은 수정 전 코드이고 `source_v1_before_union_fix`는 첫 검증 학습의 소스입니다. `hybrid_joint_8020_v1`의 완료된 검증 체크포인트를 재사용하면서 후보 생성과 선택 로직을 수정한 버전이 v2입니다. 재사용 경로와 각 모델의 원래 생성 소스는 기록에 남아 있습니다.

v3는 동일한 수치 학습 코드와 데이터 해시를 확인한 뒤 v2의 완료된 학습 및 검증 지표를 재사용합니다. 모든 설정을 최종 공통 비교 기준으로 다시 평가하고, 상위 세 설정에 대해 후보 수 10,000과 20,000을 추가 확인합니다. 추가 설정 선택도 validation에서만 수행합니다. `source_v2_before_boundary_expansion`에 이전 소스를 보존했습니다.

v3는 NDCG는 가장 높았지만 Recall이 단독 LightGCN보다 낮아 목표를 달성하지 못했습니다. 실패 결과를 보존한 뒤 v4에서 validation seed 43·44를 추가했습니다. 세 backbone 각각의 576설정을 세 seed 평균으로 비교했습니다. epoch는 초기 seed 42에서 정한 값을 고정했습니다. 기존 3층 uniform 190epoch LightGCN 비교 모델은 그대로 유지했습니다. `ROBUST_SELECTION_KO.md`에 추가 검증의 이유와 범위를, `source_v3_before_robust_selection`에 당시 소스를 보존했습니다.

v5는 v4와 같은 세 seed·576설정·선택 기준을 사용하며 중복 정규화·순위 계산만 캐시합니다. 최종 Top-K는 경계 동점을 모두 포함한 뒤 기존 상품 ID 순서로 정렬합니다. 실제 576설정 전체 지표와 개별 추천 목록을 대조한 기록은 professor_materials/fast_validation_parity.json입니다. 최종 추천 CSV를 만드는 함수는 원래 구현입니다. `SCORING_OPTIMIZATION_KO.md`와 `source_v4_before_scoring_optimization`에 변경 근거와 이전 소스를 보존했습니다. v4의 완료된 검증 모델과 CSV를 재사용했으며 바깥 시험 전에 계산만 가속했습니다.

## 결과 해석

최종 비교는 original 평가 집합 전체를 분모로 사용합니다. 필터 때문에 추천하지 못한 사용자를 제외해 점수를 높이지 않습니다. 각 모델은 같은 원시 데이터 분할과 정답을 사용합니다.

이미 과거 시험 점수를 본 연구 과정에서 수행한 정정 평가입니다. 이번 탐색 자체에 outer test 라벨을 전달하지 않았더라도, 과거에 완전히 보지 않은 시험이었다고 주장할 수는 없습니다. 세 seed의 표준편차와 사용자 bootstrap 신뢰구간은 다른 데이터셋으로의 일반화 증거가 아닙니다.

v3 시험의 Recall 미달이 추가 연구의 계기가 되었으므로 연구 방향 결정까지 시험과 독립적이었다고 말하지 않습니다. v4의 수치 설정 선택은 내부 validation에서 수행했습니다. 보고서의 선택 산점도·gamma 곡선은 세 validation seed 평균이고, 학습 곡선은 초기 seed 42의 탐색 이력입니다. component_ablation의 validation 통제 비교는 seed 42 한 개입니다.

`Hybrid_ALS_only`는 동일한 후보·모델·최종 설정에서 MF 점수 중 BPR만 뺀 통제 실험입니다. 예전 논문의 점수 재현 결과와는 다릅니다.

교수님 자료는 로컬 파일로만 생성했습니다. 메일이나 카카오톡 전송, 논문 재투고는 하지 않았습니다.
