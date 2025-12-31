"""
하이퍼파라미터 최적화 모듈

Grid Search, Bayesian Optimization, Hyperband, BOHB를 사용하여 추천 시스템의 하이퍼파라미터를 최적화합니다.
F1 스코어를 최대화하는 방향으로 최적화를 수행합니다.

사용 예시:
    # Grid Search
    optimizer = GridSearchOptimizer()
    best_params, best_score = optimizer.optimize()
    
    # Bayesian Optimization
    optimizer = BayesianOptimizer()
    best_params, best_score = optimizer.optimize(n_trials=50)
    
    # Hyperband
    optimizer = HyperbandOptimizer()
    best_params, best_score = optimizer.optimize()
    
    # BOHB (Bayesian Optimization Hyperband)
    optimizer = BOHBOptimizer()
    best_params, best_score = optimizer.optimize()
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from Model.model import (
    ALSRecommender,
    GNNEmbeddingGenerator,
    IsolationForestPreprocessor,
    ReRanker,
    TestSetEvaluator,
    set_global_seed,
    GLOBAL_SEED,
)

# 로깅 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# 전역 시드 고정 (재현가능성 보장)
set_global_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)


@dataclass
class HyperparameterConfig:
    """하이퍼파라미터 설정을 저장하는 클래스"""

    # IsolationForestPreprocessor
    preprocessor_random_state: int = 42
    preprocessor_test_size: float = 0.20
    preprocessor_contamination: str | float = "auto"
    preprocessor_n_estimators: int = 200

    # ALSRecommender
    als_factors: int = 32
    als_regularization: float = 0.1
    als_iterations: int = 16
    als_alpha: float = 1.0
    als_top_k: int = 20
    als_random_state: int = 42
    als_recommend_batch_size: int = 256
    als_positive_event_boost: float = 5.0

    # GNNEmbeddingGenerator
    gnn_embedding_dim: int = 8
    gnn_layers: int = 2
    gnn_epochs: int = 5
    gnn_batch_size: int = 8192
    gnn_learning_rate: float = 1e-3
    gnn_reg: float = 1e-4
    gnn_num_negative: int = 1
    gnn_seed: int = 42
    gnn_device: Optional[str] = None

    # ReRanker
    reranker_top_k: int = 200
    reranker_random_seed: int = 42
    reranker_als_candidate_k: int = 500
    reranker_als_weight: float = 0.4

    # TestSetEvaluator
    evaluator_top_k: int = 50

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> HyperparameterConfig:
        """딕셔너리에서 생성"""
        return cls(**config_dict)


def run_pipeline(config: HyperparameterConfig, cache_preprocessing: bool = True) -> Dict[str, float]:
    """
    전체 파이프라인을 실행하고 평가 메트릭을 반환합니다.

    Args:
        config: 하이퍼파라미터 설정
        cache_preprocessing: 전처리 결과를 캐시할지 여부 (True면 첫 실행 후 재사용)

    Returns:
        평가 메트릭 딕셔너리 (f1, item_f1, precision, recall, hit_rate 등)
    """
    processed_dir = Path("data/processed")
    preprocessor_output_exists = (
        (processed_dir / "events_train_clean.csv").exists()
        and (processed_dir / "events_test.csv").exists()
    )

    # 1. IsolationForestPreprocessor
    if not cache_preprocessing or not preprocessor_output_exists:
        logger.info("Running IsolationForestPreprocessor...")
        preprocessor = IsolationForestPreprocessor(
            random_state=config.preprocessor_random_state,
            test_size=config.preprocessor_test_size,
            contamination=config.preprocessor_contamination,
            n_estimators=config.preprocessor_n_estimators,
        )
        preprocessor.run()
    else:
        logger.info("Skipping IsolationForestPreprocessor (using cached results)")

    # 2. ALSRecommender
    logger.info("Running ALSRecommender...")
    als_recommender = ALSRecommender(
        factors=config.als_factors,
        regularization=config.als_regularization,
        iterations=config.als_iterations,
        alpha=config.als_alpha,
        top_k=config.als_top_k,
        random_state=config.als_random_state,
        recommend_batch_size=config.als_recommend_batch_size,
        positive_event_boost=config.als_positive_event_boost,
    )
    als_recommender.run()

    # 3. GNNEmbeddingGenerator
    logger.info("Running GNNEmbeddingGenerator...")
    gnn_generator = GNNEmbeddingGenerator(
        embedding_dim=config.gnn_embedding_dim,
        layers=config.gnn_layers,
        epochs=config.gnn_epochs,
        batch_size=config.gnn_batch_size,
        learning_rate=config.gnn_learning_rate,
        reg=config.gnn_reg,
        num_negative=config.gnn_num_negative,
        seed=config.gnn_seed,
        device=config.gnn_device,
    )
    gnn_generator.run()

    # 4. ReRanker
    logger.info("Running ReRanker...")
    reranker = ReRanker(
        top_k=config.reranker_top_k,
        random_seed=config.reranker_random_seed,
        als_candidate_k=config.reranker_als_candidate_k,
        als_weight=config.reranker_als_weight,
    )
    reranker.run()

    # 5. TestSetEvaluator (수정된 버전으로 F1 스코어 반환)
    logger.info("Running TestSetEvaluator...")
    metrics = evaluate_pipeline(config.evaluator_top_k)

    return metrics


def evaluate_pipeline(top_k: int = 50) -> Dict[str, float]:
    """
    파이프라인 실행 후 평가 메트릭을 계산합니다.

    Args:
        top_k: 평가에 사용할 추천 아이템 수

    Returns:
        평가 메트릭 딕셔너리
    """
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
    )

    processed_dir = Path("data/processed")
    final_rec_path = processed_dir / "final_recommendations.csv"
    events_test_path = processed_dir / "events_test.csv"

    if not final_rec_path.exists() or not events_test_path.exists():
        raise FileNotFoundError("필요한 파일이 없습니다. 파이프라인을 먼저 실행해 주세요.")

    recommendations = pd.read_csv(final_rec_path)
    events_test = pd.read_csv(events_test_path)

    test_users = events_test["visitorid"].dropna().astype(int).unique()
    test_recommendations = recommendations[recommendations["visitorid"].isin(test_users)]
    test_recommendations = (
        test_recommendations.sort_values(["visitorid", "rank"])
        .groupby("visitorid")
        .head(top_k)
    )

    positive_events = events_test[events_test["event"].isin(["addtocart", "transaction"])]
    positives: Dict[int, set[int]] = (
        positive_events.dropna(subset=["visitorid", "itemid"])
        .assign(
            visitorid=lambda df: df["visitorid"].astype(int),
            itemid=lambda df: df["itemid"].astype(int),
        )
        .groupby("visitorid")["itemid"]
        .apply(lambda items: set(items.tolist()))
        .to_dict()
    )

    y_true: List[int] = []
    y_pred: List[int] = []
    item_y_true: List[int] = []
    item_y_scores: List[float] = []

    for visitor_id, recs in test_recommendations.groupby("visitorid"):
        predicted_items = recs["itemid"].astype(int).tolist()
        relevant_items = positives.get(visitor_id, set())
        if not predicted_items:
            continue

        labels = [1 if item in relevant_items else 0 for item in predicted_items]
        item_y_true.extend(labels)
        item_y_scores.extend(recs["final_score"].astype(float).tolist())

        if not relevant_items:
            y_true.append(0)
            y_pred.append(0)
            continue

        hits = len(set(predicted_items) & relevant_items)
        y_true.append(1)
        y_pred.append(1 if hits > 0 else 0)

    # 사용자 레벨 메트릭
    user_f1 = f1_score(y_true, y_pred, zero_division=0) if y_true else 0.0
    user_precision = precision_score(y_true, y_pred, zero_division=0) if y_true else 0.0
    user_recall = recall_score(y_true, y_pred, zero_division=0) if y_true else 0.0

    # 아이템 레벨 메트릭 (최적 임계값 기준)
    item_f1 = 0.0
    item_precision = 0.0
    item_recall = 0.0

    if item_y_true:
        from sklearn.metrics import precision_recall_curve

        y_true_arr = np.array(item_y_true)
        y_score_arr = np.array(item_y_scores)
        precision_curve, recall_curve, thresholds = precision_recall_curve(
            y_true_arr, y_score_arr
        )
        f1_curve = (
            2
            * precision_curve
            * recall_curve
            / np.clip(precision_curve + recall_curve, 1e-12, None)
        )
        if f1_curve.size > 0:
            best_idx = int(np.nanargmax(f1_curve))
            if best_idx < thresholds.size:
                best_threshold = float(thresholds[best_idx])
            else:
                best_threshold = 1.0
            item_preds = (y_score_arr >= best_threshold).astype(int)
            item_precision = precision_score(y_true_arr, item_preds, zero_division=0)
            item_recall = recall_score(y_true_arr, item_preds, zero_division=0)
            item_f1 = f1_score(y_true_arr, item_preds, zero_division=0)

    # Hit Rate 계산
    hit_users = sum(1 for pred in y_pred if pred == 1)
    hit_rate = hit_users / len(y_true) if y_true else 0.0

    return {
        "f1": user_f1,  # 사용자 레벨 F1 (주요 메트릭)
        "item_f1": item_f1,  # 아이템 레벨 F1
        "precision": user_precision,
        "recall": user_recall,
        "item_precision": item_precision,
        "item_recall": item_recall,
        "hit_rate": hit_rate,
    }


class GridSearchOptimizer:
    """Grid Search를 사용한 하이퍼파라미터 최적화"""

    def __init__(
        self,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        metric: str = "f1",
        cache_preprocessing: bool = True,
    ):
        """
        Args:
            param_grid: 탐색할 하이퍼파라미터 그리드
            metric: 최적화할 메트릭 ("f1", "item_f1", "precision", "recall", "hit_rate")
            cache_preprocessing: 전처리 결과를 캐시할지 여부
        """
        self.metric = metric
        self.cache_preprocessing = cache_preprocessing

        # 기본 파라미터 그리드 (예시)
        if param_grid is None:
            self.param_grid = {
                # ALS 주요 파라미터
                "als_factors": [16, 32, 64],
                "als_regularization": [0.01, 0.1, 0.5],
                "als_iterations": [10, 16, 20],
                "als_alpha": [0.5, 1.0, 2.0],
                # GNN 주요 파라미터
                "gnn_embedding_dim": [8, 16, 32],
                "gnn_layers": [1, 2, 3],
                "gnn_epochs": [3, 5, 7],
                "gnn_learning_rate": [1e-4, 1e-3, 5e-3],
                # ReRanker 파라미터
                "reranker_als_weight": [0.3, 0.4, 0.5, 0.6],
                "reranker_als_candidate_k": [300, 500, 700],
            }
        else:
            self.param_grid = param_grid

        self.results: List[Dict[str, Any]] = []

    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """파라미터 조합 생성"""
        from itertools import product

        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combinations = []

        for combination in product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)

        return combinations

    def optimize(self) -> Tuple[HyperparameterConfig, float]:
        """
        Grid Search를 수행하여 최적의 하이퍼파라미터를 찾습니다.

        Returns:
            (최적 하이퍼파라미터 설정, 최적 스코어) 튜플
        """
        logger.info(f"Starting Grid Search with {self.metric} metric...")
        logger.info(f"Parameter grid: {self.param_grid}")
        param_combinations = self._generate_param_combinations()
        total_combinations = len(param_combinations)
        logger.info(f"Total combinations to test: {total_combinations}")

        best_score = -np.inf
        best_config = None
        self.results = []

        for idx, param_dict in enumerate(param_combinations, start=1):
            logger.info(
                f"\n[{idx}/{total_combinations}] Testing parameters: {param_dict}"
            )
            start_time = time.time()

            # 기본 설정 생성
            config = HyperparameterConfig()
            # 파라미터 업데이트
            for key, value in param_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            try:
                # 파이프라인 실행
                metrics = run_pipeline(config, cache_preprocessing=self.cache_preprocessing)
                score = metrics.get(self.metric, 0.0)
                elapsed_time = time.time() - start_time

                logger.info(
                    f"Score ({self.metric}): {score:.4f} | Time: {elapsed_time:.2f}s"
                )
                logger.info(f"All metrics: {metrics}")

                # 결과 저장
                result = {
                    "params": param_dict,
                    "score": score,
                    "metrics": metrics,
                    "elapsed_time": elapsed_time,
                }
                self.results.append(result)

                # 최적값 업데이트
                if score > best_score:
                    best_score = score
                    best_config = config
                    logger.info(f"*** New best score: {best_score:.4f} ***")

            except Exception as e:
                logger.error(f"Error during pipeline execution: {e}")
                elapsed_time = time.time() - start_time
                self.results.append(
                    {
                        "params": param_dict,
                        "score": -np.inf,
                        "error": str(e),
                        "elapsed_time": elapsed_time,
                    }
                )

        if best_config is None:
            raise ValueError("최적의 하이퍼파라미터를 찾지 못했습니다.")

        logger.info(f"\n=== Grid Search 완료 ===")
        logger.info(f"최적 스코어 ({self.metric}): {best_score:.4f}")
        logger.info(f"최적 파라미터: {best_config.to_dict()}")

        return best_config, best_score

    def save_results(self, filepath: str) -> None:
        """결과를 JSON 파일로 저장"""
        output = {
            "metric": self.metric,
            "param_grid": self.param_grid,
            "results": self.results,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Results saved to {filepath}")


class BayesianOptimizer:
    """Bayesian Optimization을 사용한 하이퍼파라미터 최적화"""

    def __init__(
        self,
        param_space: Optional[Dict[str, Tuple[Any, Any]]] = None,
        metric: str = "f1",
        cache_preprocessing: bool = True,
    ):
        """
        Args:
            param_space: 탐색할 하이퍼파라미터 공간 (파라미터명: (최소값, 최대값) 또는 리스트)
            metric: 최적화할 메트릭
            cache_preprocessing: 전처리 결과를 캐시할지 여부
        """
        self.metric = metric
        self.cache_preprocessing = cache_preprocessing

        # 기본 파라미터 공간 (예시)
        if param_space is None:
            self.param_space = {
                # ALS 파라미터 (연속값)
                "als_factors": (16, 128),
                "als_regularization": (0.01, 1.0),
                "als_iterations": (5, 30),
                "als_alpha": (0.1, 5.0),
                # GNN 파라미터
                "gnn_embedding_dim": (4, 64),
                "gnn_layers": (1, 4),
                "gnn_epochs": (3, 10),
                "gnn_learning_rate": (1e-5, 1e-2),
                # ReRanker 파라미터
                "reranker_als_weight": (0.0, 1.0),
                "reranker_als_candidate_k": (200, 1000),
            }
        else:
            self.param_space = param_space

        self.results: List[Dict[str, Any]] = []

    def _sample_params(self, trial: Any) -> Dict[str, Any]:
        """Trial 객체에서 파라미터 샘플링"""
        params = {}
        for param_name, param_range in self.param_space.items():
            if isinstance(param_range, tuple) and len(param_range) == 2:
                # 연속값 범위
                if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                    # 정수 범위
                    params[param_name] = trial.suggest_int(
                        param_name, param_range[0], param_range[1]
                    )
                else:
                    # 실수 범위
                    params[param_name] = trial.suggest_float(
                        param_name, param_range[0], param_range[1], log=True
                    )
            elif isinstance(param_range, list):
                # 이산값 리스트
                params[param_name] = trial.suggest_categorical(param_name, param_range)
            else:
                raise ValueError(f"Invalid parameter range for {param_name}")

        return params

    def optimize(self, n_trials: int = 50) -> Tuple[HyperparameterConfig, float]:
        """
        Bayesian Optimization을 수행하여 최적의 하이퍼파라미터를 찾습니다.

        Args:
            n_trials: 최적화 시도 횟수

        Returns:
            (최적 하이퍼파라미터 설정, 최적 스코어) 튜플
        """
        try:
            import optuna
        except ImportError:
            raise ImportError(
                "optuna가 필요합니다. `pip install optuna` 명령으로 설치해 주세요."
            )

        logger.info(f"Starting Bayesian Optimization with {self.metric} metric...")
        logger.info(f"Parameter space: {self.param_space}")
        logger.info(f"Number of trials: {n_trials}")

        def objective(trial: optuna.Trial) -> float:
            """최적화 목적 함수"""
            # 파라미터 샘플링
            param_dict = self._sample_params(trial)

            logger.info(f"\n[Trial {trial.number + 1}/{n_trials}] Testing: {param_dict}")
            start_time = time.time()

            # 기본 설정 생성
            config = HyperparameterConfig()
            # 파라미터 업데이트
            for key, value in param_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            try:
                # 파이프라인 실행
                metrics = run_pipeline(
                    config, cache_preprocessing=self.cache_preprocessing
                )
                score = metrics.get(self.metric, 0.0)
                elapsed_time = time.time() - start_time

                logger.info(
                    f"Score ({self.metric}): {score:.4f} | Time: {elapsed_time:.2f}s"
                )

                # 결과 저장
                result = {
                    "trial": trial.number,
                    "params": param_dict,
                    "score": score,
                    "metrics": metrics,
                    "elapsed_time": elapsed_time,
                }
                self.results.append(result)

                return score

            except Exception as e:
                logger.error(f"Error during pipeline execution: {e}")
                elapsed_time = time.time() - start_time
                self.results.append(
                    {
                        "trial": trial.number,
                        "params": param_dict,
                        "score": -np.inf,
                        "error": str(e),
                        "elapsed_time": elapsed_time,
                    }
                )
                return -np.inf

        # Optuna 스터디 생성 및 최적화 실행 (시드 고정)
        sampler = optuna.samplers.TPESampler(seed=GLOBAL_SEED)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # 최적 파라미터 추출
        best_params = study.best_params
        best_score = study.best_value

        # 최적 설정 생성
        best_config = HyperparameterConfig()
        for key, value in best_params.items():
            if hasattr(best_config, key):
                setattr(best_config, key, value)

        logger.info(f"\n=== Bayesian Optimization 완료 ===")
        logger.info(f"최적 스코어 ({self.metric}): {best_score:.4f}")
        logger.info(f"최적 파라미터: {best_config.to_dict()}")

        return best_config, best_score

    def save_results(self, filepath: str) -> None:
        """결과를 JSON 파일로 저장"""
        output = {
            "metric": self.metric,
            "param_space": self.param_space,
            "results": self.results,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Results saved to {filepath}")


class HyperbandOptimizer:
    """Hyperband 알고리즘을 사용한 하이퍼파라미터 최적화"""

    def __init__(
        self,
        param_space: Optional[Dict[str, Tuple[Any, Any]]] = None,
        metric: str = "f1",
        cache_preprocessing: bool = True,
        max_resource: int = 10,
        eta: int = 3,
        resource_param: str = "gnn_epochs",
    ):
        """
        Args:
            param_space: 탐색할 하이퍼파라미터 공간
            metric: 최적화할 메트릭
            cache_preprocessing: 전처리 결과를 캐시할지 여부
            max_resource: 최대 resource 값 (예: 최대 epochs)
            eta: Successive Halving의 감소 비율 (기본값: 3)
            resource_param: resource로 사용할 파라미터명 (기본값: "gnn_epochs")
        """
        self.metric = metric
        self.cache_preprocessing = cache_preprocessing
        self.max_resource = max_resource
        self.eta = eta
        self.resource_param = resource_param

        # 기본 파라미터 공간
        if param_space is None:
            self.param_space = {
                "als_factors": (16, 128),
                "als_regularization": (0.01, 1.0),
                "als_iterations": (5, 30),
                "gnn_embedding_dim": (4, 64),
                "gnn_layers": (1, 4),
                "gnn_learning_rate": (1e-5, 1e-2),
                "reranker_als_weight": (0.0, 1.0),
            }
        else:
            self.param_space = param_space

        # resource_param은 param_space에서 제외 (별도 관리)
        if self.resource_param in self.param_space:
            del self.param_space[self.resource_param]

        self.results: List[Dict[str, Any]] = []

    def _get_n_configs(self, s: int) -> int:
        """Bracket s에서의 초기 설정 개수 계산"""
        return int(np.ceil((self.max_resource / self.eta**s) * (self.eta**s) / (s + 1)))

    def _get_resource(self, s: int, i: int) -> int:
        """Bracket s, Round i에서의 resource 계산"""
        return int(self.max_resource / (self.eta**i))

    def _sample_random_params(self) -> Dict[str, Any]:
        """랜덤 파라미터 샘플링"""
        params = {}
        for param_name, param_range in self.param_space.items():
            if isinstance(param_range, tuple) and len(param_range) == 2:
                if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                    params[param_name] = np.random.randint(param_range[0], param_range[1] + 1)
                else:
                    params[param_name] = np.random.uniform(param_range[0], param_range[1])
            elif isinstance(param_range, list):
                params[param_name] = np.random.choice(param_range)
            else:
                raise ValueError(f"Invalid parameter range for {param_name}")
        return params

    def optimize(self) -> Tuple[HyperparameterConfig, float]:
        """
        Hyperband 알고리즘을 수행하여 최적의 하이퍼파라미터를 찾습니다.

        Returns:
            (최적 하이퍼파라미터 설정, 최적 스코어) 튜플
        """
        logger.info(f"Starting Hyperband Optimization with {self.metric} metric...")
        logger.info(f"Max resource: {self.max_resource}, Eta: {self.eta}")
        logger.info(f"Resource parameter: {self.resource_param}")

        s_max = int(np.floor(np.log(self.max_resource) / np.log(self.eta)))
        best_score = -np.inf
        best_config = None

        # 각 bracket에 대해 실행
        for s in range(s_max, -1, -1):
            logger.info(f"\n=== Bracket {s_max - s + 1}/{s_max + 1} (s={s}) ===")
            n_configs = self._get_n_configs(s)
            logger.info(f"Initial configurations: {n_configs}")

            # 초기 설정 생성
            configs = []
            for _ in range(n_configs):
                params = self._sample_random_params()
                configs.append(params)

            # Successive Halving 실행
            for i in range(s + 1):
                resource = self._get_resource(s, i)
                n_keep = max(1, int(n_configs / (self.eta**i)))
                logger.info(
                    f"Round {i + 1}/{s + 1}: Resource={resource}, "
                    f"Configs={len(configs)}, Keep top {n_keep}"
                )

                # 각 설정 평가
                scores = []
                evaluated_configs = []

                for idx, params in enumerate(configs):
                    logger.info(f"  Evaluating config {idx + 1}/{len(configs)}: {params}")

                    # 기본 설정 생성
                    config = HyperparameterConfig()
                    for key, value in params.items():
                        if hasattr(config, key):
                            setattr(config, key, value)
                    # Resource 파라미터 설정
                    setattr(config, self.resource_param, resource)

                    try:
                        start_time = time.time()
                        metrics = run_pipeline(
                            config, cache_preprocessing=self.cache_preprocessing
                        )
                        score = metrics.get(self.metric, 0.0)
                        elapsed_time = time.time() - start_time

                        logger.info(f"    Score ({self.metric}): {score:.4f} | Time: {elapsed_time:.2f}s")

                        scores.append(score)
                        evaluated_configs.append(params)

                        # 결과 저장
                        result = {
                            "bracket": s,
                            "round": i,
                            "resource": resource,
                            "params": {**params, self.resource_param: resource},
                            "score": score,
                            "metrics": metrics,
                            "elapsed_time": elapsed_time,
                        }
                        self.results.append(result)

                        # 최적값 업데이트
                        if score > best_score:
                            best_score = score
                            best_config = config
                            logger.info(f"    *** New best score: {best_score:.4f} ***")

                    except Exception as e:
                        logger.error(f"    Error during pipeline execution: {e}")
                        scores.append(-np.inf)
                        evaluated_configs.append(params)

                # 상위 n_keep개만 유지
                if len(configs) > n_keep:
                    sorted_indices = np.argsort(scores)[::-1]
                    configs = [evaluated_configs[idx] for idx in sorted_indices[:n_keep]]
                    logger.info(f"  Kept top {n_keep} configurations")

        if best_config is None:
            raise ValueError("최적의 하이퍼파라미터를 찾지 못했습니다.")

        logger.info(f"\n=== Hyperband 완료 ===")
        logger.info(f"최적 스코어 ({self.metric}): {best_score:.4f}")
        logger.info(f"최적 파라미터: {best_config.to_dict()}")

        return best_config, best_score

    def save_results(self, filepath: str) -> None:
        """결과를 JSON 파일로 저장"""
        output = {
            "method": "hyperband",
            "metric": self.metric,
            "max_resource": self.max_resource,
            "eta": self.eta,
            "resource_param": self.resource_param,
            "param_space": self.param_space,
            "results": self.results,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Results saved to {filepath}")


class BOHBOptimizer:
    """BOHB (Bayesian Optimization Hyperband) 알고리즘을 사용한 하이퍼파라미터 최적화"""

    def __init__(
        self,
        param_space: Optional[Dict[str, Tuple[Any, Any]]] = None,
        metric: str = "f1",
        cache_preprocessing: bool = True,
        max_resource: int = 10,
        eta: int = 3,
        resource_param: str = "gnn_epochs",
        min_samples: int = 3,
    ):
        """
        Args:
            param_space: 탐색할 하이퍼파라미터 공간
            metric: 최적화할 메트릭
            cache_preprocessing: 전처리 결과를 캐시할지 여부
            max_resource: 최대 resource 값
            eta: Successive Halving의 감소 비율
            resource_param: resource로 사용할 파라미터명
            min_samples: Bayesian Optimization 시작 전 필요한 최소 샘플 수
        """
        try:
            import optuna
        except ImportError:
            raise ImportError(
                "optuna가 필요합니다. `pip install optuna` 명령으로 설치해 주세요."
            )

        self.metric = metric
        self.cache_preprocessing = cache_preprocessing
        self.max_resource = max_resource
        self.eta = eta
        self.resource_param = resource_param
        self.min_samples = min_samples

        # 기본 파라미터 공간
        if param_space is None:
            self.param_space = {
                "als_factors": (16, 128),
                "als_regularization": (0.01, 1.0),
                "als_iterations": (5, 30),
                "gnn_embedding_dim": (4, 64),
                "gnn_layers": (1, 4),
                "gnn_learning_rate": (1e-5, 1e-2),
                "reranker_als_weight": (0.0, 1.0),
            }
        else:
            self.param_space = param_space

        # resource_param은 param_space에서 제외
        if self.resource_param in self.param_space:
            del self.param_space[self.resource_param]

        self.results: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []  # Bayesian Optimization을 위한 히스토리

    def _get_n_configs(self, s: int) -> int:
        """Bracket s에서의 초기 설정 개수 계산"""
        return int(np.ceil((self.max_resource / self.eta**s) * (self.eta**s) / (s + 1)))

    def _get_resource(self, s: int, i: int) -> int:
        """Bracket s, Round i에서의 resource 계산"""
        return int(self.max_resource / (self.eta**i))

    def _sample_random_params(self) -> Dict[str, Any]:
        """랜덤 파라미터 샘플링"""
        params = {}
        for param_name, param_range in self.param_space.items():
            if isinstance(param_range, tuple) and len(param_range) == 2:
                if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                    params[param_name] = np.random.randint(param_range[0], param_range[1] + 1)
                else:
                    params[param_name] = np.random.uniform(param_range[0], param_range[1])
            elif isinstance(param_range, list):
                params[param_name] = np.random.choice(param_range)
            else:
                raise ValueError(f"Invalid parameter range for {param_name}")
        return params

    def _sample_bayesian_params(self, resource: int) -> Dict[str, Any]:
        """Bayesian Optimization으로 파라미터 샘플링"""
        import optuna

        # 같은 resource 레벨 이하의 데이터만 사용
        resource_history = [
            h for h in self.history if h.get("resource", 0) <= resource
        ]

        if len(resource_history) < self.min_samples:
            return self._sample_random_params()

        # Optuna study 생성 (TPE sampler 사용, 시드 고정)
        sampler = optuna.samplers.TPESampler(seed=GLOBAL_SEED)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        # 히스토리 데이터를 study에 추가
        for h in resource_history:
            # enqueue_trial로 파라미터 추가
            study.enqueue_trial(h["params"])
            # 해당 trial을 평가
            trial = study.ask()
            # 파라미터 공간 정의를 위해 suggest 호출
            for param_name, param_range in self.param_space.items():
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                        trial.suggest_int(param_name, param_range[0], param_range[1])
                    else:
                        trial.suggest_float(param_name, param_range[0], param_range[1], log=True)
                elif isinstance(param_range, list):
                    trial.suggest_categorical(param_name, param_range)
            study.tell(trial, h["score"])

        # 새 파라미터 샘플링
        trial = study.ask()
        params = {}
        for param_name, param_range in self.param_space.items():
            if isinstance(param_range, tuple) and len(param_range) == 2:
                if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                    params[param_name] = trial.suggest_int(
                        param_name, param_range[0], param_range[1]
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        param_name, param_range[0], param_range[1], log=True
                    )
            elif isinstance(param_range, list):
                params[param_name] = trial.suggest_categorical(param_name, param_range)
            else:
                raise ValueError(f"Invalid parameter range for {param_name}")

        return params

    def optimize(self) -> Tuple[HyperparameterConfig, float]:
        """
        BOHB 알고리즘을 수행하여 최적의 하이퍼파라미터를 찾습니다.

        Returns:
            (최적 하이퍼파라미터 설정, 최적 스코어) 튜플
        """
        logger.info(f"Starting BOHB Optimization with {self.metric} metric...")
        logger.info(f"Max resource: {self.max_resource}, Eta: {self.eta}")
        logger.info(f"Resource parameter: {self.resource_param}")
        logger.info(f"Min samples for BO: {self.min_samples}")

        s_max = int(np.floor(np.log(self.max_resource) / np.log(self.eta)))
        best_score = -np.inf
        best_config = None

        # 각 bracket에 대해 실행
        for s in range(s_max, -1, -1):
            logger.info(f"\n=== Bracket {s_max - s + 1}/{s_max + 1} (s={s}) ===")
            n_configs = self._get_n_configs(s)
            logger.info(f"Initial configurations: {n_configs}")

            # 초기 설정 생성 (첫 round는 랜덤, 이후는 Bayesian)
            configs = []
            for _ in range(n_configs):
                params = self._sample_random_params()
                configs.append(params)

            # Successive Halving 실행
            for i in range(s + 1):
                resource = self._get_resource(s, i)
                n_keep = max(1, int(n_configs / (self.eta**i)))
                logger.info(
                    f"Round {i + 1}/{s + 1}: Resource={resource}, "
                    f"Configs={len(configs)}, Keep top {n_keep}"
                )

                # 각 설정 평가
                scores = []
                evaluated_configs = []

                for idx, params in enumerate(configs):
                    logger.info(f"  Evaluating config {idx + 1}/{len(configs)}: {params}")

                    # 기본 설정 생성
                    config = HyperparameterConfig()
                    for key, value in params.items():
                        if hasattr(config, key):
                            setattr(config, key, value)
                    # Resource 파라미터 설정
                    setattr(config, self.resource_param, resource)

                    try:
                        start_time = time.time()
                        metrics = run_pipeline(
                            config, cache_preprocessing=self.cache_preprocessing
                        )
                        score = metrics.get(self.metric, 0.0)
                        elapsed_time = time.time() - start_time

                        logger.info(f"    Score ({self.metric}): {score:.4f} | Time: {elapsed_time:.2f}s")

                        scores.append(score)
                        evaluated_configs.append(params)

                        # 히스토리에 추가
                        self.history.append(
                            {
                                "params": params,
                                "resource": resource,
                                "score": score,
                            }
                        )

                        # 결과 저장
                        result = {
                            "bracket": s,
                            "round": i,
                            "resource": resource,
                            "params": {**params, self.resource_param: resource},
                            "score": score,
                            "metrics": metrics,
                            "elapsed_time": elapsed_time,
                        }
                        self.results.append(result)

                        # 최적값 업데이트
                        if score > best_score:
                            best_score = score
                            best_config = config
                            logger.info(f"    *** New best score: {best_score:.4f} ***")

                    except Exception as e:
                        logger.error(f"    Error during pipeline execution: {e}")
                        scores.append(-np.inf)
                        evaluated_configs.append(params)

                # 상위 n_keep개만 유지
                if len(configs) > n_keep:
                    sorted_indices = np.argsort(scores)[::-1]
                    configs = [evaluated_configs[idx] for idx in sorted_indices[:n_keep]]
                    logger.info(f"  Kept top {n_keep} configurations")

                # 다음 round를 위한 새 설정 생성 (Bayesian Optimization 사용)
                if i < s:
                    n_new = n_configs // (self.eta ** (i + 1)) - n_keep
                    if n_new > 0:
                        logger.info(f"  Generating {n_new} new configurations using Bayesian Optimization...")
                        new_configs = []
                        for _ in range(n_new):
                            new_params = self._sample_bayesian_params(resource)
                            new_configs.append(new_params)
                        configs.extend(new_configs)

        if best_config is None:
            raise ValueError("최적의 하이퍼파라미터를 찾지 못했습니다.")

        logger.info(f"\n=== BOHB 완료 ===")
        logger.info(f"최적 스코어 ({self.metric}): {best_score:.4f}")
        logger.info(f"최적 파라미터: {best_config.to_dict()}")

        return best_config, best_score

    def save_results(self, filepath: str) -> None:
        """결과를 JSON 파일로 저장"""
        output = {
            "method": "bohb",
            "metric": self.metric,
            "max_resource": self.max_resource,
            "eta": self.eta,
            "resource_param": self.resource_param,
            "min_samples": self.min_samples,
            "param_space": self.param_space,
            "results": self.results,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Results saved to {filepath}")


if __name__ == "__main__":
    """
    하이퍼파라미터 최적화 실행 예시

    사용법:
        # Grid Search 실행
        python -m Model.hyperparameter_optimization --method grid --metric f1

        # Bayesian Optimization 실행
        python -m Model.hyperparameter_optimization --method bayesian --metric f1 --n_trials 30
    """
    import argparse

    parser = argparse.ArgumentParser(description="하이퍼파라미터 최적화")
    parser.add_argument(
        "--method",
        type=str,
        choices=["grid", "bayesian", "hyperband", "bohb"],
        default="grid",
        help="최적화 방법 (grid, bayesian, hyperband, 또는 bohb)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="f1",
        choices=["f1", "item_f1", "precision", "recall", "hit_rate"],
        help="최적화할 메트릭",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=30,
        help="Bayesian Optimization 시도 횟수",
    )
    parser.add_argument(
        "--max_resource",
        type=int,
        default=10,
        help="Hyperband/BOHB 최대 resource 값 (예: 최대 epochs)",
    )
    parser.add_argument(
        "--eta",
        type=int,
        default=3,
        help="Hyperband/BOHB Successive Halving 감소 비율",
    )
    parser.add_argument(
        "--resource_param",
        type=str,
        default="gnn_epochs",
        help="Hyperband/BOHB에서 resource로 사용할 파라미터명",
    )
    parser.add_argument(
        "--cache_preprocessing",
        action="store_true",
        default=True,
        help="전처리 결과 캐시 사용",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="hyperparameter_optimization_results.json",
        help="결과 저장 파일 경로",
    )

    args = parser.parse_args()

    if args.method == "grid":
        optimizer = GridSearchOptimizer(metric=args.metric, cache_preprocessing=args.cache_preprocessing)
        best_config, best_score = optimizer.optimize()
        optimizer.save_results(args.output)
    elif args.method == "bayesian":
        optimizer = BayesianOptimizer(metric=args.metric, cache_preprocessing=args.cache_preprocessing)
        best_config, best_score = optimizer.optimize(n_trials=args.n_trials)
        optimizer.save_results(args.output)
    elif args.method == "hyperband":
        optimizer = HyperbandOptimizer(
            metric=args.metric,
            cache_preprocessing=args.cache_preprocessing,
            max_resource=args.max_resource,
            eta=args.eta,
            resource_param=args.resource_param,
        )
        best_config, best_score = optimizer.optimize()
        optimizer.save_results(args.output)
    elif args.method == "bohb":
        optimizer = BOHBOptimizer(
            metric=args.metric,
            cache_preprocessing=args.cache_preprocessing,
            max_resource=args.max_resource,
            eta=args.eta,
            resource_param=args.resource_param,
        )
        best_config, best_score = optimizer.optimize()
        optimizer.save_results(args.output)

    # 최적 파라미터를 별도 파일로 저장
    best_config_path = args.output.replace(".json", "_best_config.json")
    with open(best_config_path, "w", encoding="utf-8") as f:
        json.dump(
            {"best_score": best_score, "best_config": best_config.to_dict()},
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(f"Best configuration saved to {best_config_path}")

