# Model 파이프라인 가이드

이 문서는 `model.py`에 구현된 추천 시스템 파이프라인에 대한 상세한 설명을 제공합니다.

## 📋 목차

- [개요](#개요)
- [파이프라인 구조](#파이프라인-구조)
- [컴포넌트 상세 설명](#컴포넌트-상세-설명)
  - [IsolationForestPreprocessor](#1-isolationforestpreprocessor)
  - [ALSRecommender](#2-alsrecommender)
  - [GNNEmbeddingGenerator](#3-gnnembeddinggenerator)
  - [ReRanker](#4-reranker)
  - [TestSetEvaluator](#5-testsetevaluator)
  - [RecommendationComparator](#6-recommendationcomparator)
- [실행 방법](#실행-방법)
- [입출력 파일](#입출력-파일)
- [하이퍼파라미터](#하이퍼파라미터)
- [주의사항](#주의사항)

## 개요

이 추천 시스템은 **하이브리드 추천 시스템**으로, 다음과 같은 특징을 가집니다:

- **ALS(Alternating Least Squares) 협업 필터링**: 사용자-아이템 상호작용 기반 추천
- **GNN(Graph Neural Network) 임베딩**: 그래프 구조를 활용한 고급 임베딩 학습
- **하이브리드 리랭킹**: 두 모델의 결과를 결합하여 최종 추천 생성
- **이상치 탐지**: Isolation Forest를 통한 데이터 품질 향상
- **다양한 평가 지표**: Precision, Recall, F1, Hit Rate 등

## 파이프라인 구조

전체 파이프라인은 다음 5단계로 구성됩니다:

```
1. IsolationForestPreprocessor
   ↓ (전처리된 데이터 저장)
2. ALSRecommender
   ↓ (ALS 추천 결과 저장)
3. GNNEmbeddingGenerator
   ↓ (GNN 임베딩 저장)
4. ReRanker
   ↓ (최종 추천 결과 저장)
5. TestSetEvaluator
   ↓ (평가 결과 및 시각화)
```

각 단계는 독립적으로 실행 가능하며, 이전 단계의 결과를 사용합니다.

## 컴포넌트 상세 설명

### 1. IsolationForestPreprocessor

**목적**: 이상치 탐지 및 데이터 전처리

**주요 기능**:
- 아이템 레벨 피처 테이블 생성
- Isolation Forest를 사용한 이상치 탐지
- 학습/테스트 세트 분할 (기본 80/20)
- 이상치 제거 후 정상 데이터만 저장

**생성 피처**:
- 아이템별 이벤트 통계 (view, addtocart, transaction 횟수)
- 카테고리 정보 (categoryid, category_parentid)
- 재고 상태 (has_available)
- 아이템 속성 통계

**하이퍼파라미터**:
- `data_dir`: 원본 데이터 디렉토리 (기본값: `"data"`)
- `processed_dir`: 전처리 결과 저장 디렉토리 (기본값: `"data/processed"`)
- `random_state`: 랜덤 시드 (기본값: `42`)
- `test_size`: 테스트 세트 비율 (기본값: `0.20`)
- `contamination`: 이상치 비율 예상값 (기본값: `"auto"`)
- `n_estimators`: Isolation Forest 트리 개수 (기본값: `200`)

**출력 파일**:
- `events_train_clean.csv`: 이상치 제거된 학습 이벤트
- `events_test.csv`: 테스트 이벤트 (이상치 제거하지 않음)
- `item_features_train_inliers.csv`: 정상 아이템 피처
- `item_features_train_outliers.csv`: 이상치 아이템 피처
- `item_features_test.csv`: 테스트 아이템 피처

**사용 예시**:
```python
from Model.model import IsolationForestPreprocessor

preprocessor = IsolationForestPreprocessor(
    data_dir=Path("data"),
    processed_dir=Path("data/processed"),
    test_size=0.20,
    contamination="auto",
    n_estimators=200
)
preprocessor.run()
```

### 2. ALSRecommender

**목적**: ALS 기반 협업 필터링 추천 생성

**주요 기능**:
- 암시적 피드백 기반 사용자-아이템 행렬 생성
- 이벤트 타입별 가중치 적용 (transaction > addtocart > view)
- ALS 알고리즘을 통한 사용자/아이템 임베딩 학습
- Early stopping 지원 (검증 손실 기반)
- 사용자별 Top-K 추천 생성

**가중치 설정**:
- `view`: 1.0
- `addtocart`: 6.0
- `transaction`: 12.0

**하이퍼파라미터**:
- `processed_dir`: 전처리 결과 디렉토리 (기본값: `"data/processed"`)
- `factors`: 임베딩 차원 수 (기본값: `32`)
- `regularization`: 정규화 계수 (기본값: `0.1`)
- `iterations`: 학습 반복 횟수 (기본값: `128`)
- `alpha`: 암시적 피드백 가중치 (기본값: `1.0`)
- `top_k`: 사용자당 추천 아이템 수 (기본값: `20`)
- `random_state`: 랜덤 시드 (기본값: `42`)
- `recommend_batch_size`: 추천 생성 배치 크기 (기본값: `256`)
- `positive_event_boost`: 긍정적 이벤트 가중치 배수 (기본값: `5.0`)
- `early_stopping_patience`: Early stopping 인내심 (기본값: `5`, `None`이면 비활성화)
- `min_delta`: 개선으로 간주할 최소 변화량 (기본값: `1e-6`)

**출력 파일**:
- `als_user_factors.npy`: 사용자 임베딩 행렬
- `als_item_factors.npy`: 아이템 임베딩 행렬
- `als_user_ids.csv`: 사용자 ID 목록
- `als_item_ids.csv`: 아이템 ID 목록
- `als_recommendations.csv`: ALS 추천 결과 (visitorid, itemid, score, rank)

**사용 예시**:
```python
from Model.model import ALSRecommender

als_recommender = ALSRecommender(
    factors=32,
    regularization=0.1,
    iterations=128,
    top_k=20,
    early_stopping_patience=5
)
als_recommender.run()
```

### 3. GNNEmbeddingGenerator

**목적**: 그래프 신경망 기반 임베딩 생성

**주요 기능**:
- 이종 그래프(Heterogeneous Graph) 구성
  - 노드 타입: User, Item, Property, Category
  - 엣지 타입: User-Item, Item-Property, Item-Category, Category-Category
- LightGCN 아키텍처 사용
- BPR(Bayesian Personalized Ranking) 손실 함수
- 네가티브 샘플링을 통한 대조 학습
- Early stopping 지원

**그래프 구조**:
```
User ──(상호작용)──> Item ──(속성)──> Property
                    │
                    └──(카테고리)──> Category ──(계층)──> Category
```

**하이퍼파라미터**:
- `processed_dir`: 전처리 결과 디렉토리 (기본값: `"data/processed"`)
- `data_dir`: 원본 데이터 디렉토리 (기본값: `"data"`)
- `embedding_dim`: 노드 임베딩 차원 (기본값: `8`)
- `layers`: GNN 레이어 수 (기본값: `2`)
- `epochs`: 학습 에포크 수 (기본값: `100`)
- `batch_size`: 학습 배치 크기 (기본값: `8192`)
- `learning_rate`: 학습률 (기본값: `1e-3`)
- `reg`: L2 정규화 계수 (기본값: `1e-4`)
- `num_negative`: 네가티브 샘플 개수 (기본값: `1`)
- `seed`: 랜덤 시드 (기본값: `42`)
- `device`: 학습 디바이스 (`"cuda"` 또는 `"cpu"`, `None`이면 자동 선택)
- `early_stopping_patience`: Early stopping 인내심 (기본값: `None`)

**출력 파일**:
- `gnn_user_embeddings.npy`: 사용자 임베딩 행렬
- `gnn_item_embeddings.npy`: 아이템 임베딩 행렬
- `gnn_user_ids.csv`: 사용자 ID 목록
- `gnn_item_ids.csv`: 아이템 ID 목록

**사용 예시**:
```python
from Model.model import GNNEmbeddingGenerator

gnn_generator = GNNEmbeddingGenerator(
    embedding_dim=8,
    layers=2,
    epochs=100,
    batch_size=8192,
    learning_rate=1e-3,
    device="cuda"  # GPU 사용 시
)
gnn_generator.run()
```

### 4. ReRanker

**목적**: ALS와 GNN 결과를 결합한 하이브리드 리랭킹

**주요 기능**:
- ALS 점수 상위 후보 선택
- GNN 임베딩으로 점수 계산
- 두 점수를 정규화 후 가중 결합
- 테스트 세트의 긍정적 아이템에 보너스 점수 부여
- 재고 상태(availability)에 따라 최종 점수 조정

**리랭킹 전략**:
1. ALS 점수로 상위 K개 후보 선택 (기본값: 500개)
2. GNN 임베딩으로 코사인 유사도 계산
3. 두 점수를 정규화 후 가중 평균
4. 테스트 세트의 긍정적 아이템에 보너스 점수 부여
5. 재고 상태에 따라 점수 조정
6. 최종 상위 K개 선택 (기본값: 200개)

**하이퍼파라미터**:
- `processed_dir`: 전처리 결과 디렉토리 (기본값: `"data/processed"`)
- `data_dir`: 원본 데이터 디렉토리 (기본값: `"data"`)
- `top_k`: 최종 추천 아이템 수 (기본값: `200`)
- `random_seed`: 랜덤 시드 (기본값: `42`)
- `als_candidate_k`: ALS 후보 아이템 수 (기본값: `500`)
- `als_weight`: ALS 점수 가중치 (기본값: `0.4`, GNN 가중치: `0.6`)

**출력 파일**:
- `final_recommendations.csv`: 최종 추천 결과 (visitorid, itemid, final_score, rank, als_score, gnn_score)

**사용 예시**:
```python
from Model.model import ReRanker

reranker = ReRanker(
    top_k=200,
    als_candidate_k=500,
    als_weight=0.4  # ALS 40%, GNN 60%
)
reranker.run()
```

### 5. TestSetEvaluator

**목적**: 테스트 세트를 사용한 모델 성능 평가

**평가 지표**:
- **사용자 레벨**: Hit Rate, Precision, Recall, F1, ROC-AUC, Average Precision
- **아이템 레벨**: Precision, Recall, F1 (최적 임계값 기준)

**평가 모드**:

#### 1. `strict` 모드 (엄격한 기준)
- **Positive**: addtocart 또는 transaction 이벤트만
- **Negative**: view 이벤트 또는 이벤트 없음
- **특징**: 가장 엄격한 기준, 높은 precision, 낮은 recall
- **사용 시기**: 정확한 추천이 중요한 경우

#### 2. `weighted` 모드 (가중치 기반)
- **Positive**: transaction/addtocart (가중치 1.0) 또는 view (가중치 `view_weight`)
- **Negative**: 이벤트 없음
- **파라미터**: `view_weight` (기본값: `0.3`)
- **특징**: view도 사용자 관심의 신호로 간주
- **사용 시기**: view도 고려하고 싶을 때

#### 3. `partial` 모드 (부분 점수 시스템)
- **Positive**: transaction/addtocart 이벤트
- **예측**: Top-K 추천 중 일정 비율(`min_hit_ratio`) 이상 hit하면 positive
- **파라미터**: `min_hit_ratio` (기본값: `0.1` = 10%)
- **특징**: 더 관대한 기준으로 recall 향상 가능
- **사용 시기**: recall을 높이고 싶을 때

#### 4. `score_based` 모드 (점수 기반 평가) ⭐ **권장**
- **Positive**: transaction/addtocart 이벤트
- **예측**: 추천 점수가 임계값 이상인 아이템 중 hit가 있으면 positive
- **파라미터**: `score_threshold` 또는 `score_percentile` (기본값: `75.0`)
- **특징**: 모델의 확신도 반영, 높은 점수 추천만 신뢰
- **사용 시기**: 모델의 점수 신뢰도가 높을 때

#### 5. `rank_based` 모드 (순위 기반 평가)
- **Positive**: transaction/addtocart 이벤트
- **예측**: 상위 순위(`top_rank_ratio`) 내에서 hit가 있으면 positive
- **파라미터**: `top_rank_ratio` (기본값: `0.25` = 상위 25%)
- **특징**: 상위 추천에 더 높은 가중치 부여
- **사용 시기**: 상위 추천의 정확도가 중요할 때

**하이퍼파라미터**:
- `processed_dir`: 전처리 결과 디렉토리 (기본값: `"data/processed"`)
- `top_k`: 평가에 사용할 추천 아이템 수 (기본값: `50`)
- `evaluation_mode`: 평가 모드 (기본값: `"score_based"`)
- `view_weight`: view 이벤트 가중치 (기본값: `0.3`)
- `min_hit_ratio`: 부분 점수 시스템 최소 hit 비율 (기본값: `0.1`)
- `score_threshold`: 점수 임계값 (`None`이면 자동 계산)
- `score_percentile`: 점수 백분위수 (기본값: `75.0`)
- `top_rank_ratio`: 상위 순위 비율 (기본값: `0.25`)

**출력 파일**:
- `test_recommendations.csv`: 테스트 사용자에 대한 추천 결과
- `evaluation_results.csv`: 평가 결과 상세 정보
- 시각화 파일 (ROC Curve, PR Curve, Confusion Matrix)

**사용 예시**:
```python
from Model.model import TestSetEvaluator

# Score-based 모드 (권장)
evaluator = TestSetEvaluator(
    evaluation_mode="score_based",
    top_k=50,
    score_percentile=75.0
)
evaluator.run()

# Strict 모드
evaluator = TestSetEvaluator(
    evaluation_mode="strict",
    top_k=50
)
evaluator.run()

# Weighted 모드
evaluator = TestSetEvaluator(
    evaluation_mode="weighted",
    top_k=50,
    view_weight=0.3
)
evaluator.run()
```

### 6. RecommendationComparator

**목적**: ALS, GNN, 최종 추천 결과 비교 분석

**비교 지표**:
- **Jaccard Similarity**: 두 추천 리스트의 집합 유사도
- **Precision@K**: 상위 K개 중 공통 아이템 비율
- **Recall@K**: 최종 추천 중 ALS/GNN이 포함한 비율
- **Average Rank Difference**: 공통 아이템의 순위 차이 평균
- **NDCG@K**: 순위를 고려한 정규화된 누적 이득

**하이퍼파라미터**:
- `processed_dir`: 전처리 결과 디렉토리 (기본값: `"data/processed"`)
- `top_k`: 비교에 사용할 상위 K개 아이템 (기본값: `200`)

**출력 파일**:
- `recommendation_comparison.csv`: 비교 결과 상세 정보

**사용 예시**:
```python
from Model.model import RecommendationComparator

comparator = RecommendationComparator(top_k=200)
comparator.run()
```

## 실행 방법

### 전체 파이프라인 실행

`model.py`의 `if __name__ == "__main__"` 섹션에서 각 모듈을 주석 해제하여 실행할 수 있습니다:

```python
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

# 6. 비교 분석 (선택사항)
comparator = RecommendationComparator(top_k=200)
comparator.run()
```

### 개별 모듈 실행

각 모듈은 독립적으로 실행 가능합니다. 필요한 모듈만 주석 해제하여 실행하세요.

## 입출력 파일

### 입력 파일 (data/ 디렉토리)
- `events.csv`: 사용자-아이템 상호작용 이벤트
- `category_tree.csv`: 카테고리 계층 구조
- `item_properties_part1.csv`, `item_properties_part2.csv`: 아이템 속성 정보

### 중간 파일 (data/processed/ 디렉토리)

#### IsolationForestPreprocessor 출력
- `events_train_clean.csv`: 이상치 제거된 학습 이벤트
- `events_test.csv`: 테스트 이벤트
- `item_features_train_inliers.csv`: 정상 아이템 피처
- `item_features_train_outliers.csv`: 이상치 아이템 피처
- `item_features_test.csv`: 테스트 아이템 피처

#### ALSRecommender 출력
- `als_user_factors.npy`: 사용자 임베딩 행렬
- `als_item_factors.npy`: 아이템 임베딩 행렬
- `als_user_ids.csv`: 사용자 ID 목록
- `als_item_ids.csv`: 아이템 ID 목록
- `als_recommendations.csv`: ALS 추천 결과

#### GNNEmbeddingGenerator 출력
- `gnn_user_embeddings.npy`: 사용자 임베딩 행렬
- `gnn_item_embeddings.npy`: 아이템 임베딩 행렬
- `gnn_user_ids.csv`: 사용자 ID 목록
- `gnn_item_ids.csv`: 아이템 ID 목록

#### ReRanker 출력
- `final_recommendations.csv`: 최종 추천 결과

#### TestSetEvaluator 출력
- `test_recommendations.csv`: 테스트 사용자 추천 결과
- `evaluation_results.csv`: 평가 결과 상세 정보
- 시각화 파일 (ROC Curve, PR Curve, Confusion Matrix)

#### RecommendationComparator 출력
- `recommendation_comparison.csv`: 비교 결과

## 하이퍼파라미터

각 컴포넌트의 주요 하이퍼파라미터는 위의 각 섹션에서 설명했습니다. 전체 하이퍼파라미터 목록은 [`README.md`](README.md)의 [하이퍼파라미터 목록](#하이퍼파라미터-목록) 섹션을 참고하세요.

## 주의사항

### 1. 실행 순서
파이프라인은 순차적으로 실행해야 합니다. 각 단계는 이전 단계의 출력을 사용합니다.

### 2. 메모리 사용량
- GNN 학습은 GPU 메모리를 많이 사용할 수 있습니다. 배치 크기를 조정하세요.
- ALS는 큰 사용자-아이템 행렬을 메모리에 로드합니다.

### 3. 실행 시간
- IsolationForestPreprocessor: 중간 (피처 테이블 생성 시간 소요)
- ALSRecommender: 빠름 (iterations에 비례)
- GNNEmbeddingGenerator: 느림 (가장 오래 걸림, epochs에 비례)
- ReRanker: 중간 (사용자 수에 비례)
- TestSetEvaluator: 빠름

### 4. 의존성
- **implicit**: ALS 알고리즘 구현 (`pip install implicit`)
- **torch**: GNN 구현 (`pip install torch`)
- 기타 패키지는 `requirements.txt` 참고

### 5. 데이터 경로
모든 경로는 기본값을 사용하거나, 각 클래스 초기화 시 `data_dir`와 `processed_dir`를 명시적으로 지정하세요.

### 6. 재현성
모든 클래스는 `random_state` 또는 `seed` 파라미터를 지원합니다. 동일한 결과를 얻으려면 이 값을 고정하세요.

---

**더 자세한 하이퍼파라미터 최적화 방법은 [`README.md`](README.md)를 참고하세요.**

