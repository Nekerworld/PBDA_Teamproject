# 코드 변경 검토 안내

실제 수정은 `Model/paper_revision_codex`의 실행 가능한 복사본에서 수행했다. 프로젝트의 원래 `Model/paper_revision`, 원시 데이터, 투고 PDF와 카카오톡 원본은 수정하지 않았다. 새 실험 진입점과 결과 폴더를 분리했다.

| 파일 | 변경과 목적 |
|---|---|
| model.py | ReRanker의 최종 MF 입력을 ALS 단독에서 ALS+BPR 결합으로 변경. 기존 비교를 위한 ALS-only 선택도 유지. MF 혼합 음성 및 검증 checkpoint 관찰 콜백 연결. |
| experiments.py | 검증 gamma·결합 탐색에서도 최종 MF 신호를 보존. 실제 추론과 설정 탐색의 수식 일치. 희소도는 이벤트 행 수가 아닌 고유 사용자·상품 쌍 기준. |
| learning.py | 학습 MF가 높게 평가한 미관측 후보와 uniform 표본을 섞는 음성 sampler. 저장 factor의 사용자·상품 ID를 그래프 순서에 맞춰 정렬하며 학습 이력 제외. 포화 사용자는 -1로 처리. |
| data_protocol.py | 작업 폴더 하위 출력 경로와 PBDA_PROJECT_ROOT 환경 변수 지원. 원시 데이터는 기존 프로젝트 data에서 읽는다. |
| improve_hybrid.py | 최초 실제 80:20 실험. 내부 검증 학습, MF 후보 합집합, 최종 융합·gamma 공동 선택, 선택 확정 뒤 세 seed 재학습 및 평가. |
| robust_hybrid.py | 첫 시험에서 Recall 미달을 확인한 후 추가한 세 seed validation 선택. 기존 3층 LightGCN baseline을 유지하며 정해진 세 backbone·576설정을 seed 평균으로 비교. |
| test_hybrid_improvement.py | 후보 합집합 할당 보존, BPR 신호 보존, 빠른 평가와 기준 평가 일치, 음성 ID 정렬·관측 이력 제외, 최종 공통 기준 재선택 검사. |
| test_robust_hybrid.py | 유리한 한 seed 대신 세 seed 평균으로 선택하는지, seed 누락을 거부하는지 검사. |
| fast_fusion.py / test_fast_fusion.py | 검증 설정 사이의 같은 정규화·순위를 캐시하고 경계 동점을 포함한 부분 Top-K로 계산량을 줄인다. 실제 576설정 지표와 추천 목록을 기존 구현과 대조했다. 최종 평가 추천 함수와 통계 선택 기준은 유지. |
| test_revision.py | 합성 IF 스트레스 자료에서 여러 미관측 쌍을 남기도록 fixture 조정. IF 뒤 완전 포화된 BPR 행렬에서 정상적으로 중단되어 후속 provenance 검증에 도달하지 못하던 문제를 해결. 실제 모델의 포화 사용자 guard는 완화하지 않았다. |
| report_tools | 원본 문제 근거 수집, IF 특징 제외 비교, 추천 CSV 독립 재계산, 동일 후보·동일 GNN 통제 비교, 실행시간 측정, 실제 지표 그림, 한국어 보고서와 PDF 렌더링·검수·패키징. |

`CODE_CHANGES.patch`는 수정 전 source_before_hybrid_20261017과 현재 루트 Python 파일 사이의 실제 diff다. `CODE_CHANGES_MANIFEST.json`에 각 파일의 전후 SHA-256을 기록했다. 이 diff는 원래 코드에 자동 적용하도록 검증한 설치 프로그램이 아니라 검토 자료다.

수치 학습 코드는 실험 계획의 해시로 고정했다. 이전 버전 source_v1_before_union_fix / source_v2_before_boundary_expansion / source_v3_before_robust_selection과 각 실행 계획을 보존했다. 보고서 생성 도구는 학습 소스와 분리해 그림 배치·설명 수정 때문에 학습을 재실행하지 않도록 했다.

최종 검사 결과는 regression_final.log, 최종 실제 데이터 성능은 results/hybrid_joint_8020_v5/result.json, 독립 재계산 결과는 professor_materials/verification.json을 확인한다. 평균 순위와 통계적 불확실성을 구분하며 실패한 v3 결과도 보존했다.
