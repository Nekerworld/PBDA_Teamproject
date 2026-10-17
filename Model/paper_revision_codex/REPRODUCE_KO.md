# 새 컴퓨터에서 최종 Hybrid 실험 실행

대상은 이 폴더의 improve_hybrid.py → robust_hybrid.py 실험이다. 과거 Model/paper 또는 Model/paper_revision 실행 환경과 분리한다. 학습 소스는 이번 의존성 정리에서 변경하지 않았다.

## 환경 조건

- Python 3.11 64비트. 기존 실험은 Windows, Python 3.11.13, RTX 3090 Ti 24GB, torch 2.11.0+cu128에서 실행했다.
- 최종 두 실행기는 CUDA를 명시하므로 NVIDIA GPU 및 CUDA 12.8 PyTorch와 호환되는 드라이버가 필요하다. CPU 또는 Mac에서 이 두 명령을 그대로 실행할 수 없다. 24GB는 사용한 장비 사양이며 최소 VRAM 검증치는 아니다.
- Windows가 기존 검증 플랫폼이다. Linux의 신규 설치 및 전체 재학습은 별도 검증하지 않았다.
- 원시 데이터, 전처리 결과, 실험 출력은 하드링크를 지원하는 같은 로컬 볼륨에 둔다(Windows NTFS 등). 네트워크 공유나 서로 다른 드라이브에 나누면 os.link가 실패할 수 있다.
- requirements는 직접 의존성 버전을 고정했다. 전체 전이 의존성을 해시까지 고정한 lockfile은 아니다. 다른 GPU/드라이버에서는 동일 seed라도 수치의 완전 일치를 보장하지 않는다.

## 설치 (프로젝트 루트에서 PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r Model/paper_revision_codex/requirements-cu128.txt
python -m pip install -r Model/paper_revision_codex/requirements-dev.txt
python -m pip check
python -c "import torch,implicit,numpy,pandas,scipy,sklearn,matplotlib; print(torch.__version__); assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0)); x=torch.ones(1,device='cuda'); print(x+x)"
```

Linux에서는 `python3.11 -m venv .venv`, `source .venv/bin/activate`를 사용하고 나머지 pip 명령은 동일하다. PyTorch 공식 설치 출처: https://pytorch.org/get-started/previous-versions/#v2110 . torchvision/torchaudio는 이 실험에서 사용하지 않는다.

루트 requirements.txt는 공통 의존성 진입점이다. 최종 GPU 실험에는 위 requirements-cu128.txt를 사용하여 CUDA 빌드를 명시한다. optuna는 최종 실행 경로에서 사용하지 않아 제외했고, pytest와 보고서 패키지는 별도 파일로 분리했다. 과거 보관용 past_code.py의 seaborn 등은 이 실험의 의존성에 포함하지 않는다.

## 원시 데이터와 전처리

RetailRocket 원본을 내려받아 프로젝트 루트의 data 폴더에 아래 네 파일을 둔다. Git에서는 데이터와 가중치를 제외하므로 clone만으로는 실행되지 않는다.

- events.csv
- item_properties_part1.csv
- item_properties_part2.csv
- category_tree.csv

데이터 출처: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

이후 아래 명령은 모두 Model/paper_revision_codex를 작업 디렉터리로 사용한다. Git에 남은 이전 결과 JSON의 절대 경로는 새 PC에서 재사용하지 않는다. 비어 있는 새 출력 폴더를 사용하여 모델을 다시 학습한다.

```powershell
cd Model/paper_revision_codex
python -c "from pathlib import Path; from data_protocol import UnifiedPreprocessor; UnifiedPreprocessor(data_dir=Path('../../data').resolve(), processed_dir=Path('results/reproduce_data/no_if_events').resolve(), test_size=0.2, validation_size=0.2, use_if=False, core_mode='events', min_user_events=3, min_item_events=3, random_state=42).run()"
```

공통 시간순 outer 80:20, 내부 validation 80:20, no-IF, event core 3 조건이다. 위 명령은 기존 폴더 덮어쓰기를 허용하지 않는다. 재실행하려면 새 출력 폴더명을 사용한다. 프로젝트 구조를 유지하면 PBDA_PROJECT_ROOT를 설정할 필요가 없다. 구조가 다를 때만 원시 data 폴더를 포함하는 프로젝트 루트로 지정한다.

## 회귀검사와 전체 실행

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:MPLBACKEND='Agg'
python -m pytest test_revision.py test_hybrid_improvement.py test_codex_audit.py test_robust_hybrid.py test_fast_fusion.py -q
python improve_hybrid.py --source-dir results/reproduce_data/no_if_events --output-dir results/reproduce_v3 --max-hours 168
python robust_hybrid.py --parent results/reproduce_v3 --output-dir results/reproduce_v5 --max-hours 168
python report_tools/verify_hybrid_delivery.py --experiment results/reproduce_v5 --output results/reproduce_v5/independent_verification.json
```

Linux에서는 위 PowerShell 환경 변수 네 줄 대신 `export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg`를 실행한다. 수치 라이브러리의 과도한 스레드 생성과 화면 없는 환경에서의 그래프 표시를 피하기 위한 설정이다.

max-hours는 예상 소요 시간이 아니라 실행 중단 한도다. 컴퓨터 성능에 맞춰 조정한다. 위 명령은 이전 PC의 체크포인트 또는 validation CSV를 재사용하지 않는다. v3가 정상 완료되어야 robust_hybrid.py를 실행한다. 명령 사이에 작업 디렉터리를 변경하지 않는다.

결과의 result.json, selection.json, model_summary.csv, independent_verification.json을 확인한다. 기존 관측값은 Hybrid 세 seed 평균 NDCG@50 0.1171767043, Recall@50 0.1662600392다. 신규 실행의 결과를 이 값에 맞춰 조정하지 않는다. 기존 Recall 차이의 신뢰구간은 0을 포함하므로 통계적으로 확정된 보편적 우위로 해석하지 않는다.

## 선택 사항: 보고서

```powershell
python -m pip install -r requirements-report.txt
```

report_tools의 PDF 작성기는 현재 C:/Windows/Fonts/malgun.ttf 및 malgunbd.ttf를 사용한다. 따라서 Windows 맑은 고딕이 필요하다. Linux에서는 학습과 지표 검산을 진행할 수 있지만 PDF 작성기는 폰트 경로 수정이 추가로 필요하다. 보고서 작성에는 학습 외의 evidence/diagnosis/ablation 산출물도 필요하므로 requirements 설치만으로 보고서가 완성되지는 않는다.

## GitHub에 포함되는 범위

소스, requirements, 실행 문서를 함께 커밋한다. .gitignore의 data/, *.csv, 모델 배열, 체크포인트, 로그 제외 규칙 때문에 원본 데이터·추천 CSV·지표 CSV·가중치는 새 clone에 없다. 기존 보고서를 읽는 것과 실험을 재실행하는 것은 구분한다. 독립 검산 CSV가 필요하면 별도 데이터 전달 또는 재실행이 필요하다.
