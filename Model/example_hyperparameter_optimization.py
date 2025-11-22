"""
하이퍼파라미터 최적화 사용 예시

이 스크립트는 Grid Search와 Bayesian Optimization을 사용하여
추천 시스템의 하이퍼파라미터를 최적화하는 방법을 보여줍니다.
"""

from Model.hyperparameter_optimization import (
    BayesianOptimizer,
    BOHBOptimizer,
    GridSearchOptimizer,
    HyperbandOptimizer,
    HyperparameterConfig,
)


def example_grid_search():
    """Grid Search 예시"""
    print("=" * 60)
    print("Grid Search 예시")
    print("=" * 60)

    # 작은 파라미터 그리드로 빠른 테스트
    param_grid = {
        "als_factors": [16, 32],
        "als_regularization": [0.1, 0.5],
        "als_iterations": [10, 16],
        "gnn_embedding_dim": [8, 16],
        "gnn_layers": [1, 2],
        "reranker_als_weight": [0.3, 0.5],
    }

    optimizer = GridSearchOptimizer(
        param_grid=param_grid,
        metric="f1",  # F1 스코어 최적화
        cache_preprocessing=True,  # 전처리 결과 재사용
    )

    best_config, best_score = optimizer.optimize()
    optimizer.save_results("grid_search_results.json")

    print(f"\n최적 F1 스코어: {best_score:.4f}")
    print(f"최적 파라미터:")
    for key, value in best_config.to_dict().items():
        print(f"  {key}: {value}")


def example_bayesian_optimization():
    """Bayesian Optimization 예시"""
    print("=" * 60)
    print("Bayesian Optimization 예시")
    print("=" * 60)

    # 파라미터 공간 정의
    param_space = {
        "als_factors": (16, 64),  # 정수 범위
        "als_regularization": (0.01, 1.0),  # 실수 범위 (로그 스케일)
        "als_iterations": (10, 20),
        "gnn_embedding_dim": (8, 32),
        "gnn_layers": (1, 3),
        "gnn_epochs": (3, 7),
        "gnn_learning_rate": (1e-4, 1e-2),  # 로그 스케일
        "reranker_als_weight": (0.2, 0.8),
        "reranker_als_candidate_k": (300, 700),
    }

    optimizer = BayesianOptimizer(
        param_space=param_space,
        metric="f1",  # F1 스코어 최적화
        cache_preprocessing=True,
    )

    best_config, best_score = optimizer.optimize(n_trials=20)  # 20번 시도
    optimizer.save_results("bayesian_optimization_results.json")

    print(f"\n최적 F1 스코어: {best_score:.4f}")
    print(f"최적 파라미터:")
    for key, value in best_config.to_dict().items():
        print(f"  {key}: {value}")


def example_custom_config():
    """사용자 정의 설정으로 파이프라인 실행 예시"""
    print("=" * 60)
    print("사용자 정의 설정 예시")
    print("=" * 60)

    from Model.hyperparameter_optimization import run_pipeline

    # 특정 하이퍼파라미터로 설정 생성
    config = HyperparameterConfig(
        als_factors=64,
        als_regularization=0.2,
        als_iterations=20,
        gnn_embedding_dim=16,
        gnn_layers=2,
        gnn_epochs=5,
        reranker_als_weight=0.5,
    )

    # 파이프라인 실행 및 평가
    metrics = run_pipeline(config, cache_preprocessing=True)

    print("\n평가 결과:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")


def example_focused_optimization():
    """특정 모듈에 집중한 최적화 예시 (ALS만)"""
    print("=" * 60)
    print("ALS 모듈 집중 최적화 예시")
    print("=" * 60)

    # ALS 파라미터만 최적화
    param_grid = {
        "als_factors": [16, 32, 64, 128],
        "als_regularization": [0.01, 0.1, 0.5, 1.0],
        "als_iterations": [10, 16, 20, 24],
        "als_alpha": [0.5, 1.0, 2.0, 5.0],
    }

    optimizer = GridSearchOptimizer(
        param_grid=param_grid,
        metric="f1",
        cache_preprocessing=True,
    )

    best_config, best_score = optimizer.optimize()
    optimizer.save_results("als_optimization_results.json")

    print(f"\n최적 F1 스코어: {best_score:.4f}")
    print(f"최적 ALS 파라미터:")
    print(f"  factors: {best_config.als_factors}")
    print(f"  regularization: {best_config.als_regularization}")
    print(f"  iterations: {best_config.als_iterations}")
    print(f"  alpha: {best_config.als_alpha}")


def example_hyperband():
    """Hyperband 최적화 예시"""
    print("=" * 60)
    print("Hyperband 최적화 예시")
    print("=" * 60)

    # 파라미터 공간 정의
    param_space = {
        "als_factors": (16, 128),
        "als_regularization": (0.01, 1.0),
        "gnn_embedding_dim": (8, 64),
        "gnn_layers": (1, 4),
        "gnn_learning_rate": (1e-4, 1e-2),
        "reranker_als_weight": (0.2, 0.8),
    }

    optimizer = HyperbandOptimizer(
        param_space=param_space,
        metric="f1",
        cache_preprocessing=True,
        max_resource=10,  # 최대 epochs
        eta=3,  # Successive Halving 감소 비율
        resource_param="gnn_epochs",  # resource로 사용할 파라미터
    )

    best_config, best_score = optimizer.optimize()
    optimizer.save_results("hyperband_results.json")

    print(f"\n최적 F1 스코어: {best_score:.4f}")
    print(f"최적 파라미터:")
    for key, value in best_config.to_dict().items():
        print(f"  {key}: {value}")


def example_bohb():
    """BOHB (Bayesian Optimization Hyperband) 최적화 예시"""
    print("=" * 60)
    print("BOHB 최적화 예시")
    print("=" * 60)

    # 파라미터 공간 정의
    param_space = {
        "als_factors": (16, 128),
        "als_regularization": (0.01, 1.0),
        "gnn_embedding_dim": (8, 64),
        "gnn_layers": (1, 4),
        "gnn_learning_rate": (1e-4, 1e-2),
        "reranker_als_weight": (0.2, 0.8),
    }

    optimizer = BOHBOptimizer(
        param_space=param_space,
        metric="f1",
        cache_preprocessing=True,
        max_resource=10,  # 최대 epochs
        eta=3,  # Successive Halving 감소 비율
        resource_param="gnn_epochs",  # resource로 사용할 파라미터
        min_samples=3,  # Bayesian Optimization 시작 전 최소 샘플 수
    )

    best_config, best_score = optimizer.optimize()
    optimizer.save_results("bohb_results.json")

    print(f"\n최적 F1 스코어: {best_score:.4f}")
    print(f"최적 파라미터:")
    for key, value in best_config.to_dict().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        method = sys.argv[1]
        if method == "grid":
            example_grid_search()
        elif method == "bayesian":
            example_bayesian_optimization()
        elif method == "custom":
            example_custom_config()
        elif method == "focused":
            example_focused_optimization()
        elif method == "hyperband":
            example_hyperband()
        elif method == "bohb":
            example_bohb()
        else:
            print(f"Unknown method: {method}")
            print("Available methods: grid, bayesian, custom, focused, hyperband, bohb")
    else:
        print("사용법:")
        print("  python example_hyperparameter_optimization.py grid")
        print("  python example_hyperparameter_optimization.py bayesian")
        print("  python example_hyperparameter_optimization.py custom")
        print("  python example_hyperparameter_optimization.py focused")
        print("  python example_hyperparameter_optimization.py hyperband")
        print("  python example_hyperparameter_optimization.py bohb")

