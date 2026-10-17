# 의존성 및 GitHub 재실행 준비 점검 (2026-10-17)

- 기존 requirements는 하한만 지정하여 검증 당시와 다른 버전이 설치될 수 있었다.
- 최종 실험이 import하는 직접 의존성을 AST로 대조하여 누락 여부를 확인했다.
- 실제 v5 환경을 기준으로 numpy/pandas/scipy/scikit-learn/matplotlib/implicit/torch를 고정했다.
- requirements-cu128.txt로 CUDA 12.8 PyTorch 빌드를 명시했다. 설치 명령은 PyTorch 공식 previous-versions 문서를 확인했다.
- pytest, 보고서용 reportlab/pypdfium2/pypdf/Pillow를 별도 requirements로 분리했다.
- 원시 데이터 전처리 API, GPU 조건, 하드링크의 동일 볼륨 조건, 새 출력 폴더 사용법을 REPRODUCE_KO.md에 명시했다.
- 루트 requirements.txt와 README에서 최종 실험 환경으로 진입하도록 연결했다.
- Python 3.11 / Windows x86_64 대상 pip dry-run으로 CUDA 및 선택 의존성을 포함한 40개 패키지의 설치 해석을 통과했다. package 목록은 dependency_check.json에 있다.
- Git에서 제외되지 않는 파일 중 100 MiB 이상인 파일은 없었다. 새 requirements와 재실행 안내도 ignore되지 않는다.

검증 범위: pip dry-run은 실제 신규 가상환경 설치나 다른 PC에서의 전체 재학습을 대체하지 않는다. Linux 전체 실행, 최소 VRAM, 드라이버별 수치 동일성은 이번 검사에서 검증하지 않았다. 현재 PDF 작성기의 맑은 고딕 경로는 Windows 전용이며 문서에 명시했다. 기존 *.csv 제외 규칙으로 결과 지표/추천 CSV도 Git에 포함되지 않으므로 재실행하거나 별도로 전달해야 한다.

학습 및 평가 Python 소스는 변경하지 않았다. GitHub push는 수행하지 않았다.

회귀검사 재실행 결과: 46 passed, 1 warning in 101.62s. 기존 Python 3.11 실험 환경에서 OPENBLAS_NUM_THREADS/OMP_NUM_THREADS/MKL_NUM_THREADS=1 및 MPLBACKEND=Agg로 수행했다. 경고는 PyTorch sparse invariant 검사 관련 UserWarning이며 테스트 실패는 없었다.
