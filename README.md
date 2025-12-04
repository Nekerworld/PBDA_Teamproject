# PBDA TeamProject

2025년 2학기 빅데이터분석실무 과목의 팀 프로젝트입니다.

이 레포지토리는 **RetailRocket 데이터셋**을 활용한 **온라인 쇼핑몰 고객 행동 분석 및 추천 시스템 구축** 프로젝트입니다.
## [데이터 링크 (Kaggle)](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

## 📋 목차

- [프로젝트 소개](#프로젝트-소개)
- [주요 기능](#주요-기능)
- [데이터 소개](#데이터-소개)
- [프로젝트 구조](#프로젝트-구조)
- [설치 및 실행 방법](#설치-및-실행-방법)
- [모델 아키텍처](#모델-아키텍처)
- [하이퍼파라미터 최적화](#하이퍼파라미터-최적화)
- [사용 예시](#사용-예시)
- [기술 스택](#기술-스택)
- [라이선스](#라이선스)

## 🎯 프로젝트 소개

이 프로젝트는 온라인 쇼핑몰의 고객 행동 데이터를 분석하고, 하이브리드 추천 시스템을 구축하는 것을 목표로 합니다. 

**RetailRocket 데이터셋**을 기반으로:
- 고객의 클릭, 장바구니 추가, 구매 행동을 분석
- ALS(Alternating Least Squares) 협업 필터링과 GNN(Graph Neural Network)을 결합한 하이브리드 추천 시스템 구현
- 이상치 탐지를 통한 데이터 품질 향상
- 다양한 평가 지표를 통한 모델 성능 측정

## ✨ 주요 기능

### 1. EDA
- **이벤트 분석**: 고객 행동 패턴, 전환율, 시간대별/요일별 트렌드 분석
- **세션 분석**: 30분 기준 세션 분할을 통한 고객 여정 분석
- **아이템 인기도 분석**: Long-tail 분포 및 인기 상품 분석
- **이상치 탐지**: IQR 방법을 활용한 이상 사용자 행동 패턴 식별
- **카테고리 분석**: 상품 카테고리별 고객 선호도 및 트렌드 분석
- **인터랙티브 시각화**: Plotly를 활용한 동적 차트와 HTML 리포트 생성

### 2. 추천 시스템 파이프라인
- **IsolationForestPreprocessor**: 이상치 탐지 및 데이터 전처리
- **ALSRecommender**: ALS 기반 협업 필터링 추천
- **GNNEmbeddingGenerator**: 그래프 신경망 기반 임베딩 생성
- **ReRanker**: ALS와 GNN 결과를 결합한 하이브리드 리랭킹
- **TestSetEvaluator**: 다양한 평가 지표를 통한 성능 측정

### 3. 하이퍼파라미터 최적화
- Grid Search, Bayesian Optimization, Hyperband, BOHB 지원
- F1 스코어, Precision, Recall, Hit Rate 등 다양한 메트릭 최적화

## 📊 데이터 소개

### 데이터셋
- **데이터셋명**: RetailRocket E-commerce Dataset
- **출처**: [Kaggle - RetailRocket E-commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
- **데이터 유형**: 이벤트 로그, 아이템 속성, 카테고리 트리

### 데이터 구성
프로젝트에서 사용하는 주요 데이터 파일:

- **events.csv**: 고객의 행동 이벤트 로그
  - 이벤트 타입: `view`, `addtocart`, `transaction`
  - 컬럼: `timestamp`, `visitorid`, `event`, `itemid`
  
- **item_properties_part1.csv & part2.csv**: 아이템의 속성 정보
  - 카테고리, 재고 상태 등 상품 메타데이터
  
- **category_tree.csv**: 상품 카테고리의 계층 구조

> **참고**: 정확한 데이터 크기 및 상세 통계는 EDA 리포트를 참고해주세요.

## 📁 프로젝트 구조

```
PBDA_Teamproject/
├── EDA/                                     # 탐색적 데이터 분석
│   ├── EDA.py                               # 메인 분석 스크립트
│   ├── requirements.txt                     # EDA에 필요한 패키지 목록
│   ├── retailrocket_recommendation_report.html  # 분석 리포트
│   └── *.pdf                                # 분석 관련 문서들
│
├── Model/                                   # 추천 시스템 모델
│   ├── model.py                             # 메인 모델 파이프라인
│   ├── hyperparameter_optimization.py      # 하이퍼파라미터 최적화 모듈
│   ├── example_hyperparameter_optimization.py  # 최적화 사용 예시
│   ├── requirements.txt                     # 모델 실행에 필요한 패키지 목록
│   └── HYPERPARAMETER_OPTIMIZATION_README.md  # 최적화 가이드
│
├── LICENSE                                  # 라이선스 파일
└── README.md                                # 프로젝트 설명서
```

## 🚀 설치 및 실행 방법

### 환경 요구사항

- **Python**: 3.8 이상
- **가상환경**: `venv` 또는 `conda` 환경 사용 권장

### 1. 레포지토리 클론

```bash
git clone https://github.com/Nekerworld/PBDA_Teamproject.git
cd PBDA_Teamproject
```

### 2. 패키지 설치

#### EDA 실행을 위한 패키지 설치
```bash
cd EDA
pip install -r requirements.txt
```

#### 모델 실행을 위한 패키지 설치
```bash
cd Model
pip install -r requirements.txt
```

> **참고**: `requirements.txt`에는 numpy, pandas, scipy, scikit-learn, matplotlib, torch, optuna가 포함되어 있습니다.

### 3. 데이터 준비

1. [Kaggle 데이터셋](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)에서 데이터 다운로드 (크기가 커서 레포에는 포함하지 않았습니다.)
2. `data/` 폴더에 다음 파일들을 배치:
   - `events.csv`
   - `item_properties_part1.csv`
   - `item_properties_part2.csv`
   - `category_tree.csv`

### 4. 실행

#### EDA 실행
```bash
cd EDA
python EDA.py
```

실행 후 `retailrocket_recommendation_report.html` 파일이 생성됩니다.

#### 추천 시스템 모델 실행
```bash
cd Model
python model.py
```

`model.py` 파일의 `if __name__ == "__main__"` 섹션에서 실행할 모듈을 주석 해제하여 사용할 수 있습니다.

## 🏗️ 모델 아키텍처

추천 시스템은 다음 5단계 파이프라인으로 구성됩니다:

### 1. IsolationForestPreprocessor
- **목적**: 이상치 탐지 및 데이터 전처리
- **기능**: 
  - 아이템 레벨 피처 테이블 생성
  - Isolation Forest를 사용한 이상치 탐지
  - 이상치 제거 후 학습/테스트 세트 분할

### 2. ALSRecommender
- **목적**: ALS 기반 협업 필터링 추천
- **기능**:
  - Implicit feedback 기반 사용자-아이템 행렬 생성
  - 이벤트 타입별 가중치 적용 (transaction > addtocart > view)
  - ALS 알고리즘을 통한 사용자/아이템 임베딩 학습
  - Top-K 추천 생성

### 3. GNNEmbeddingGenerator
- **목적**: 그래프 신경망 기반 임베딩 생성
- **기능**:
  - 사용자-아이템 상호작용 그래프 구성
  - GNN을 통한 노드 임베딩 학습
  - 임베딩 기반 유사도 계산

### 4. ReRanker
- **목적**: 하이브리드 리랭킹
- **기능**:
  - ALS와 GNN 결과를 가중치 결합
  - 최종 추천 리스트 생성 및 리랭킹

### 5. TestSetEvaluator
- **목적**: 모델 성능 평가
- **평가 지표**:
  - Precision, Recall, F1 Score
  - Hit Rate
  - ROC-AUC, Average Precision
- **평가 모드**: `strict`, `weighted`, `partial`, `score_based (최적)`, `rank_based`

## 🔧 하이퍼파라미터 최적화

프로젝트는 다양한 하이퍼파라미터 최적화 방법을 지원합니다.

### 지원하는 최적화 방법
- **Grid Search**: 지정된 파라미터 그리드에서 모든 조합 탐색
- **Bayesian Optimization**: 효율적인 탐색을 통한 최적 파라미터 발견 (Optuna 사용)
- **Hyperband**: Successive Halving 기반의 효율적인 최적화
- **BOHB**: Hyperband와 Bayesian Optimization을 결합한 최적화 기법

### 사용 예시

```python
from Model.hyperparameter_optimization import GridSearchOptimizer

# 파라미터 그리드 정의
param_grid = {
    "als_factors": [16, 32, 64],
    "als_regularization": [0.1, 0.5],
    "gnn_embedding_dim": [8, 16],
    "reranker_als_weight": [0.3, 0.5],
}

# 최적화 실행
optimizer = GridSearchOptimizer(
    param_grid=param_grid,
    metric="f1",
    cache_preprocessing=True,
)

best_config, best_score = optimizer.optimize()
optimizer.save_results("results.json")
```

자세한 사용법은 [`Model/README.md`](Model/README.md)를 참고하세요.

## 💻 사용 예시

### 기본 파이프라인 실행

```python
from Model.model import (
    IsolationForestPreprocessor,
    ALSRecommender,
    GNNEmbeddingGenerator,
    ReRanker,
    TestSetEvaluator
)

# 1. 데이터 전처리
preprocessor = IsolationForestPreprocessor()
preprocessor.run()

# 2. ALS 추천 생성
als_recommender = ALSRecommender()
als_recommender.run()

# 3. GNN 임베딩 생성
gnn_generator = GNNEmbeddingGenerator()
gnn_generator.run()

# 4. 리랭킹
reranker = ReRanker()
reranker.run()

# 5. 평가
evaluator = TestSetEvaluator(
    evaluation_mode="score_based",
    top_k=50,
    score_percentile=50.0
)
evaluator.run()
```

## 🛠️ 기술 스택

### 데이터 처리 및 분석
- `pandas` (>=1.5.0): 데이터프레임 처리 및 분석
- `numpy` (>=1.21.0): 수치 연산
- `scipy`: 희소 행렬 처리

### 시각화
- `matplotlib` (>=3.5.0): 정적 시각화
- `seaborn` (>=0.11.0): 통계적 시각화
- `plotly` (>=5.0.0): 인터랙티브 시각화
- `networkx` (>=2.8.0): 네트워크 분석 (카테고리 트리)

### 머신러닝
- `scikit-learn` (>=1.0.0): 머신러닝 모델 및 전처리
- `torch` (>=1.12.0): PyTorch - 그래프 신경망(GNN) 구현
- `optuna` (>=3.0.0): 하이퍼파라미터 최적화 (Bayesian Optimization, BOHB 등)

### 기타
- `datetime`: 시간 데이터 처리 (내장 모듈)
- `json`: JSON 데이터 처리 (내장 모듈)
- `collections`: 데이터 구조 처리 (내장 모듈)

## 📈 분석 결과

EDA를 실행하면 다음과 같은 분석 결과를 얻을 수 있습니다:

- **전환 퍼널 분석**: View → AddToCart → Transaction 전환율
- **시간대별 고객 행동 패턴**: 시간대와 요일별 이벤트 분포
- **아이템 인기도 분석**: Long-tail 분포 및 인기 상품 식별
- **세션 분석**: 고객 세션별 행동 패턴 및 지속 시간
- **이상치 탐지**: 비정상적인 사용자 행동 패턴 식별
- **카테고리별 트렌드**: 상품 카테고리별 고객 선호도 분석

> **참고**: 구체적인 성능 지표 및 분석 결과 수치는 프로젝트 팀에서 채워주세요.

## 📝 참고사항

### 모르는 부분 / 확인 필요 사항

다음 항목들은 프로젝트 팀에서 내용을 채워주시면 좋겠습니다:

1. **팀원 정보**: 프로젝트 참여자 및 역할
2. **프로젝트 기간**: 시작일 및 종료일
3. **성능 지표**: 모델의 구체적인 성능 수치 (F1, Precision, Recall 등)
4. **데이터 크기**: 실제 처리된 데이터의 정확한 크기 및 통계
5. **실행 시간**: 각 모듈별 예상 실행 시간
6. **하드웨어 요구사항**: GPU 필요 여부, 메모리 요구사항 등
7. **트러블슈팅**: 자주 발생하는 문제 및 해결 방법

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

**문의사항이나 버그 리포트는 이슈를 통해 제출해주세요.**
