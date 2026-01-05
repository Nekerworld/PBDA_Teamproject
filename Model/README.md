# 하이퍼파라미터 최적화 가이드

안녕하세요!
이곳은 추천 시스템의 하이퍼파라미터를 최적화하기 위한 도구를 제공합니다. 
Grid Search와 Bayesian Optimization을 지원하며, F1 스코어를 최대화하는 방향으로 최적화를 수행합니다.

## 📋 목차

- [주요 기능](#주요-기능)
- [설치 요구사항](#설치-요구사항)
- [빠른 시작](#빠른-시작)
- [최적화 방법 선택 가이드](#최적화-방법-선택-가이드)
- [하이퍼파라미터 목록](#하이퍼파라미터-목록)
- [최적화 메트릭](#최적화-메트릭)
- [사용 예시](#사용-예시)
- [성능 최적화 팁](#성능-최적화-팁)
- [결과 해석](#결과-해석)
- [주의사항](#주의사항)
- [문제 해결](#문제-해결)

## 주요 기능

- **Grid Search**: 지정된 파라미터 그리드에서 모든 조합을 탐색
- **Bayesian Optimization**: 효율적인 탐색을 통한 최적 파라미터 발견
- **Hyperband**: Successive Halving 기반의 효율적인 하이퍼파라미터 최적화
- **BOHB (Bayesian Optimization Hyperband)**: Hyperband와 Bayesian Optimization을 결합한 고급 최적화 기법
- **자동 평가**: 전체 파이프라인 실행 후 F1 스코어 자동 계산
- **결과 저장**: 최적화 결과를 JSON 파일로 저장

## 설치 요구사항

```bash
# Model 폴더의 requirements.txt를 사용하여 모든 패키지 설치
cd Model
pip install -r requirements.txt
```

또는 개별 설치:

```bash
# 기본 패키지
pip install numpy pandas scikit-learn scipy matplotlib

# 그래프 신경망 (GNN)을 위한 PyTorch
pip install torch

# Bayesian Optimization을 위한 optuna
pip install optuna
```

> **참고**: `Model/requirements.txt` 파일에 모든 필수 패키지가 정의되어 있습니다.

## 빠른 시작

### 1. Grid Search 사용

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
    metric="f1",  # 최적화할 메트릭
    cache_preprocessing=True,  # 전처리 결과 재사용
)

best_config, best_score = optimizer.optimize()
optimizer.save_results("results.json")
```

### 2. Bayesian Optimization 사용

```python
from Model.hyperparameter_optimization import BayesianOptimizer

# 파라미터 공간 정의
param_space = {
    "als_factors": (16, 128),  # 정수 범위
    "als_regularization": (0.01, 1.0),  # 실수 범위
    "gnn_embedding_dim": (8, 64),
    "reranker_als_weight": (0.0, 1.0),
}

# 최적화 실행
optimizer = BayesianOptimizer(
    param_space=param_space,
    metric="f1",
    cache_preprocessing=True,
)

best_config, best_score = optimizer.optimize(n_trials=50)
optimizer.save_results("results.json")
```

### 3. Hyperband 사용

```python
from Model.hyperparameter_optimization import HyperbandOptimizer

param_space = {
    "als_factors": (16, 128),
    "als_regularization": (0.01, 1.0),
    "gnn_embedding_dim": (8, 64),
}

optimizer = HyperbandOptimizer(
    param_space=param_space,
    metric="f1",
    max_resource=10,  # 최대 epochs
    eta=3,  # Successive Halving 감소 비율
    resource_param="gnn_epochs",  # resource로 사용할 파라미터
)

best_config, best_score = optimizer.optimize()
```

### 4. BOHB 사용

```python
from Model.hyperparameter_optimization import BOHBOptimizer

param_space = {
    "als_factors": (16, 128),
    "als_regularization": (0.01, 1.0),
    "gnn_embedding_dim": (8, 64),
}

optimizer = BOHBOptimizer(
    param_space=param_space,
    metric="f1",
    max_resource=10,
    eta=3,
    resource_param="gnn_epochs",
    min_samples=3,  # Bayesian Optimization 시작 전 최소 샘플 수
)

best_config, best_score = optimizer.optimize()
```

### 5. 커맨드라인에서 실행

```bash
# Grid Search
python -m Model.hyperparameter_optimization --method grid --metric f1

# Bayesian Optimization
python -m Model.hyperparameter_optimization --method bayesian --metric f1 --n_trials 30

# Hyperband
python -m Model.hyperparameter_optimization --method hyperband --metric f1 --max_resource 10 --eta 3

# BOHB
python -m Model.hyperparameter_optimization --method bohb --metric f1 --max_resource 10 --eta 3
```

## 최적화 방법 선택 가이드

다음 표를 참고하여 상황에 맞는 최적화 방법을 선택하세요:

| 방법 | 파라미터 수 | 실행 시간 | 정확도 | 추천 상황 |
|------|------------|----------|--------|----------|
| **Grid Search** | 적음 (2-4개) | 중간 | 높음 | 파라미터가 적고, 각 값의 범위가 명확할 때 |
| **Bayesian Optimization** | 많음 (5개 이상) | 빠름 | 높음 | 파라미터 공간이 크고, 효율적인 탐색이 필요할 때 |
| **Hyperband** | 많음 | 매우 빠름 | 중간 | 학습 시간이 오래 걸리고, 빠른 탐색이 필요할 때 |
| **BOHB** | 많음 | 빠름 | 매우 높음 | 정확도와 효율성을 모두 고려할 때 (권장) |

### 시나리오별 추천

#### 시나리오 1: 처음 시작하는 경우
```python
# 작은 Grid Search로 시작하여 대략적인 범위 파악
param_grid = {
    "als_factors": [16, 32, 64],
    "als_regularization": [0.1, 0.5],
}
optimizer = GridSearchOptimizer(param_grid=param_grid, metric="f1")
```

#### 시나리오 2: 시간이 부족한 경우
```python
# Hyperband로 빠르게 탐색
optimizer = HyperbandOptimizer(
    param_space={...},
    max_resource=5,  # 작은 값으로 시작
    eta=3
)
```

#### 시나리오 3: 최고 성능이 필요한 경우
```python
# BOHB로 정밀한 탐색
optimizer = BOHBOptimizer(
    param_space={...},
    max_resource=10,
    min_samples=5  # 더 많은 샘플로 시작
)
```

#### 시나리오 4: 파라미터가 많은 경우 (5개 이상)
```python
# Bayesian Optimization 사용
optimizer = BayesianOptimizer(
    param_space={...},
    metric="f1"
)
best_config, best_score = optimizer.optimize(n_trials=100)
```

## 하이퍼파라미터 목록

### IsolationForestPreprocessor
- `preprocessor_random_state`: 랜덤 시드 (기본값: 42)
- `preprocessor_test_size`: 테스트 세트 비율 (기본값: 0.20)
- `preprocessor_contamination`: 이상치 비율 예상값 (기본값: "auto")
- `preprocessor_n_estimators`: Isolation Forest 트리 개수 (기본값: 200)

### ALSRecommender
- `als_factors`: 사용자/아이템 임베딩 차원 (기본값: 32)
- `als_regularization`: 정규화 계수 (기본값: 0.1)
- `als_iterations`: 학습 반복 횟수 (기본값: 16)
- `als_alpha`: 암시적 피드백 가중치 (기본값: 1.0)
- `als_top_k`: 사용자당 추천 아이템 수 (기본값: 20)
- `als_random_state`: 랜덤 시드 (기본값: 42)
- `als_recommend_batch_size`: 추천 생성 배치 크기 (기본값: 256)
- `als_positive_event_boost`: 긍정적 이벤트 가중치 배수 (기본값: 5.0)

### GNNEmbeddingGenerator
- `gnn_embedding_dim`: 노드 임베딩 차원 (기본값: 8)
- `gnn_layers`: GNN 레이어 수 (기본값: 2)
- `gnn_epochs`: 학습 에포크 수 (기본값: 5)
- `gnn_batch_size`: 학습 배치 크기 (기본값: 8192)
- `gnn_learning_rate`: 학습률 (기본값: 1e-3)
- `gnn_reg`: L2 정규화 계수 (기본값: 1e-4)
- `gnn_num_negative`: 네가티브 샘플 수 (기본값: 1)
- `gnn_seed`: 랜덤 시드 (기본값: 42)
- `gnn_device`: 학습 디바이스 (기본값: None, 자동 선택)

### ReRanker
- `reranker_top_k`: 최종 추천 아이템 수 (기본값: 200)
- `reranker_random_seed`: 랜덤 시드 (기본값: 42)
- `reranker_als_candidate_k`: ALS 후보 아이템 수 (기본값: 500)
- `reranker_als_weight`: ALS 점수 가중치 (기본값: 0.4)

### TestSetEvaluator
- `evaluator_top_k`: 평가에 사용할 추천 아이템 수 (기본값: 50)

## 최적화 메트릭

다음 메트릭 중 하나를 선택하여 최적화할 수 있습니다:

- `f1`: 사용자 레벨 F1 스코어 (기본값, 권장)
- `item_f1`: 아이템 레벨 F1 스코어
- `precision`: 사용자 레벨 Precision
- `recall`: 사용자 레벨 Recall
- `hit_rate`: Hit Rate

## 사용 예시

더 자세한 예시는 `example_hyperparameter_optimization.py` 파일을 참고하세요.

```bash
# Grid Search 예시 실행
python Model/example_hyperparameter_optimization.py grid

# Bayesian Optimization 예시 실행
python Model/example_hyperparameter_optimization.py bayesian
```

## 성능 최적화 팁

1. **전처리 캐싱**: `cache_preprocessing=True`로 설정하면 전처리 단계를 한 번만 실행하고 재사용합니다. 하이퍼파라미터가 전처리에 영향을 주지 않는 경우 유용합니다.

2. **단계별 최적화**: 모든 파라미터를 한 번에 최적화하는 대신, 모듈별로 나누어 최적화할 수 있습니다:
   - 먼저 ALS 파라미터만 최적화
   - 그 다음 GNN 파라미터 최적화
   - 마지막으로 ReRanker 파라미터 최적화

3. **작은 그리드로 시작**: Grid Search의 경우, 먼저 작은 그리드로 빠르게 테스트한 후 좋은 영역을 찾아 세밀하게 탐색하세요.

4. **Bayesian Optimization 활용**: 파라미터 공간이 클 때는 Bayesian Optimization이 더 효율적입니다.

5. **Hyperband/BOHB 활용**: 학습 시간이 오래 걸리는 경우 (특히 GNN 학습), Hyperband나 BOHB를 사용하면 효율적으로 최적화할 수 있습니다:
   - **Hyperband**: 빠른 탐색이 필요할 때, 랜덤 샘플링과 Successive Halving을 결합
   - **BOHB**: 더 정확한 탐색이 필요할 때, Bayesian Optimization과 Hyperband를 결합
   - `max_resource`: 최대 resource 값 (예: 최대 epochs)
   - `eta`: Successive Halving의 감소 비율 (기본값: 3)
   - `resource_param`: resource로 사용할 파라미터명 (기본값: "gnn_epochs")

## 결과 해석

최적화 결과는 JSON 파일로 저장되며, 다음 정보를 포함합니다:

- `metric`: 최적화한 메트릭
- `param_grid` 또는 `param_space`: 탐색한 파라미터 공간
- `results`: 각 시도에 대한 상세 결과
  - `params`: 테스트한 파라미터 조합
  - `score`: 해당 조합의 스코어
  - `metrics`: 모든 평가 메트릭
  - `elapsed_time`: 실행 시간

최적 파라미터는 `*_best_config.json` 파일에도 별도로 저장됩니다.

### 결과 파일 예시

#### 최적화 결과 파일 (`results.json`)
```json
{
  "metric": "f1",
  "param_grid": {
    "als_factors": [16, 32, 64],
    "als_regularization": [0.1, 0.5]
  },
  "best_score": 0.4523,
  "best_params": {
    "als_factors": 32,
    "als_regularization": 0.1,
    "gnn_embedding_dim": 16,
    "reranker_als_weight": 0.4
  },
  "results": [
    {
      "params": {
        "als_factors": 16,
        "als_regularization": 0.1,
        "gnn_embedding_dim": 8,
        "reranker_als_weight": 0.3
      },
      "score": 0.4234,
      "metrics": {
        "f1": 0.4234,
        "precision": 0.4567,
        "recall": 0.3921,
        "hit_rate": 0.6789
      },
      "elapsed_time": 1234.56
    }
  ]
}
```

#### 최적 설정 파일 (`*_best_config.json`)
```json
{
  "als_factors": 32,
  "als_regularization": 0.1,
  "als_iterations": 16,
  "gnn_embedding_dim": 16,
  "gnn_layers": 2,
  "gnn_epochs": 5,
  "reranker_als_weight": 0.4
}
```

## 주의사항

1. **실행 시간**: 전체 파이프라인 실행은 시간이 오래 걸릴 수 있습니다. 특히 GNN 학습 단계가 가장 오래 걸립니다.

2. **메모리 사용량**: 큰 파라미터 그리드나 많은 시도 횟수는 메모리를 많이 사용할 수 있습니다.

3. **재현성**: `random_state` 파라미터를 고정하면 재현 가능한 결과를 얻을 수 있습니다.

4. **전처리 캐싱**: 전처리 단계의 하이퍼파라미터(`preprocessor_*`)를 변경할 때는 `cache_preprocessing=False`로 설정해야 합니다.

## 문제 해결

### 메모리가 부족하대요
- 파라미터 그리드 크기 줄이기
- `cache_preprocessing=True`로 설정하여 중간 결과를 재사용하는건 어떨까요
- 배치 크기를 줄여보세요 (`als_recommend_batch_size`, `gnn_batch_size`)

### 실행 시간이 너무 긴데요
- 작은 파라미터 그리드로 시작 ㄱㄱ
- Bayesian Optimization 써보십쇼 (더 효율적)
- `gnn_epochs`를 줄이세요
- `cache_preprocessing=True`로 설정해보세요

