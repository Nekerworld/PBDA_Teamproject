"""
하이퍼파라미터 민감도 실험 스크립트

기존 실험 파라미터 + 제안 파라미터를 모두 사용해 one-at-a-time 민감도 실험을 수행하고
NDCG@K, Recall@K를 기록해 CSV 및 표로 출력합니다.

실험 파라미터:
- 기존: ALS factors, GNN embedding_dim, ALS weight, GNN layers
- 제안: ALS regularization, ALS alpha, GNN learning_rate, GNN reg,
        GNN num_negative, long_tail_bonus

사용 예:
    python -m Model.hyperparameter_sensitivity
    python -m Model.hyperparameter_sensitivity --output results_sensitivity.csv
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 프로젝트 루트를 path에 추가
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from Model.model import (
    ALSPreprocessor,
    ALSRecommender,
    GNNEmbeddingGenerator,
    GNNPreprocessor,
    ReRanker,
    _ndcg_at_k,
    _recall_at_k,
    set_global_seed,
    GLOBAL_SEED,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
set_global_seed(GLOBAL_SEED)


# ---------------------------------------------------------------------------
# 기본 설정 (als_debug / GNN / ReRanker 기본값과 동일하게 맞춤)
# ---------------------------------------------------------------------------

@dataclass
class SensitivityConfig:
    """민감도 실험용 통합 설정 (한 번에 하나의 파라미터만 변경)."""
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    data_dir: Path = field(default_factory=lambda: Path("data"))
    top_k: int = 50

    # ALS (ALSRecommender)
    als_factors: int = 128
    als_regularization: float = 0.03
    als_iterations: int = 256
    als_alpha: float = 1.5
    als_long_tail_bonus: float = 0.35
    als_long_tail_quantile: float = 0.5
    als_ensemble_alpha: float = 0.5

    # GNN
    gnn_embedding_dim: int = 8
    gnn_layers: int = 2
    gnn_epochs: int = 1024
    gnn_batch_size: int = 8192
    gnn_learning_rate: float = 1e-3
    gnn_reg: float = 1e-4
    gnn_num_negative: int = 1
    gnn_early_stopping_patience: Optional[int] = 50

    # ReRanker
    als_weight: float = 0.4

    def copy_with(self, **overrides: Any) -> "SensitivityConfig":
        d = self.__dict__.copy()
        d.update(overrides)
        return SensitivityConfig(**d)


# ---------------------------------------------------------------------------
# 파라미터 그리드 (기존 + 제안)
# ---------------------------------------------------------------------------

def get_parameter_grids() -> List[Tuple[str, str, List[Any]]]:
    """
    (파라미터 표시 이름, SensitivityConfig 필드명, 값 목록) 리스트 반환.
    한 번에 하나의 파라미터만 변경하는 one-at-a-time 그리드.
    """
    return [
        # 기존 실험 파라미터
        ("ALS factors", "als_factors", [16, 32, 64, 128]),
        ("GNN embedding_dim", "gnn_embedding_dim", [8, 16, 32, 64]),
        ("ALS weight", "als_weight", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        ("GNN layers", "gnn_layers", [1, 2, 3]),
        # 제안 파라미터
        ("ALS regularization", "als_regularization", [0.01, 0.03, 0.1, 0.3]),
        ("ALS alpha", "als_alpha", [0.5, 1.0, 1.5, 2.0]),
        ("GNN learning_rate", "gnn_learning_rate", [1e-4, 5e-4, 1e-3, 5e-3]),
        ("GNN reg", "gnn_reg", [1e-5, 1e-4, 1e-3]),
        ("GNN num_negative", "gnn_num_negative", [1, 3, 5]),
        ("long_tail_bonus", "als_long_tail_bonus", [0.0, 0.2, 0.35, 0.5]),
    ]


# ---------------------------------------------------------------------------
# 파이프라인 실행 및 평가
# ---------------------------------------------------------------------------

def load_positives_and_test_users(processed_dir: Path) -> Tuple[Dict[int, set], np.ndarray]:
    """events_test.csv(GNN 전처리) 또는 als_events_test.csv(ALS 전처리)에서 positives와 test_users 로드."""
    for name in ("events_test.csv", "als_events_test.csv"):
        path = processed_dir / name
        if not path.exists():
            continue
        events_test = pd.read_csv(path)
        if "visitorid" not in events_test.columns or "event" not in events_test.columns:
            continue
        test_users = events_test["visitorid"].dropna().astype(int).unique()
        pos = events_test[events_test["event"].isin(["addtocart", "transaction"])].dropna(
            subset=["visitorid", "itemid"]
        )
        pos = pos.assign(
            visitorid=pos["visitorid"].astype(int),
            itemid=pos["itemid"].astype(int),
        )
        positives: Dict[int, set] = (
            pos.groupby("visitorid")["itemid"]
            .apply(lambda s: set(s.tolist()))
            .to_dict()
        )
        return positives, test_users
    return {}, np.array([], dtype=np.int64)


def evaluate_recommendations(
    processed_dir: Path,
    top_k: int,
    score_col: str = "final_score",
) -> Tuple[float, float]:
    """final_recommendations.csv를 읽어 NDCG@K, Recall@K 계산."""
    path = processed_dir / "final_recommendations.csv"
    if not path.exists():
        return float("nan"), float("nan")
    rec = pd.read_csv(path)
    positives, test_users = load_positives_and_test_users(processed_dir)
    if not positives or rec.empty:
        return float("nan"), float("nan")
    rec["visitorid"] = rec["visitorid"].astype(int)
    rec["itemid"] = rec["itemid"].astype(int)
    rec = rec[rec["visitorid"].isin(test_users)]
    use_col = score_col if score_col in rec.columns else "rank"
    asc = use_col == "rank"
    rec = rec.sort_values(["visitorid", use_col], ascending=[True, asc])
    rec = rec.groupby("visitorid").head(top_k)
    ndcg = _ndcg_at_k(rec, positives, top_k, use_col, ascending=asc)
    recall = _recall_at_k(rec, positives, top_k, use_col, ascending=asc)
    return ndcg, recall


def run_als(config: SensitivityConfig) -> None:
    """ALS 단계만 실행 (config 반영)."""
    als = ALSRecommender(
        processed_dir=config.processed_dir,
        factors=config.als_factors,
        regularization=config.als_regularization,
        iterations=config.als_iterations,
        alpha=config.als_alpha,
        top_k=config.top_k,
        long_tail_bonus=config.als_long_tail_bonus,
        long_tail_quantile=config.als_long_tail_quantile,
        ensemble_alpha=config.als_ensemble_alpha,
    )
    als.run()


def _clear_gnn_memory() -> None:
    """GNN 실행 후 모델·인접행렬 등 대형 객체 참조 해제 및 GPU 캐시 비우기."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_gnn(config: SensitivityConfig) -> None:
    """GNN 단계만 실행 (config 반영). 실행 후 디스크에 저장된 임베딩만 남기고 메모리 비우기."""
    gnn = GNNEmbeddingGenerator(
        processed_dir=config.processed_dir,
        data_dir=config.data_dir,
        embedding_dim=config.gnn_embedding_dim,
        layers=config.gnn_layers,
        epochs=config.gnn_epochs,
        batch_size=config.gnn_batch_size,
        learning_rate=config.gnn_learning_rate,
        reg=config.gnn_reg,
        num_negative=config.gnn_num_negative,
        early_stopping_patience=config.gnn_early_stopping_patience,
        device=getattr(config, "gnn_device", "cpu"),
    )
    try:
        gnn.run()
    finally:
        del gnn
        _clear_gnn_memory()


def run_reranker(config: SensitivityConfig) -> None:
    """ReRanker 단계만 실행 (config 반영)."""
    reranker = ReRanker(
        processed_dir=config.processed_dir,
        data_dir=config.data_dir,
        top_k=200,
        als_weight=config.als_weight,
    )
    reranker.run()


# ---------------------------------------------------------------------------
# 민감도 실험 메인
# ---------------------------------------------------------------------------

def run_sensitivity_experiments(
    processed_dir: Path,
    data_dir: Path,
    top_k: int = 50,
    skip_als: bool = False,
    skip_gnn: bool = False,
) -> pd.DataFrame:
    """
    One-at-a-time 민감도 실험을 수행하고 결과 DataFrame 반환.

    skip_als: True면 ALS/ReRanker 관련 실험 생략 (이미 ALS 결과가 있을 때)
    skip_gnn: True면 GNN/ReRanker 관련 실험 생략
    """
    default_config = SensitivityConfig(
        processed_dir=processed_dir,
        data_dir=data_dir,
        top_k=top_k,
    )
    grids = get_parameter_grids()
    rows: List[Dict[str, Any]] = []

    # 0) 전처리 1회 (ALS용 als_events_*, GNN용 events_train_clean 등)
    als_events_train = processed_dir / "als_events_train.csv"
    events_train_clean = processed_dir / "events_train_clean.csv"
    if not als_events_train.exists():
        logger.info("Running ALSPreprocessor (als_events_*)...")
        ALSPreprocessor(data_dir=data_dir, processed_dir=processed_dir).run()
    if not events_train_clean.exists():
        logger.info("Running GNNPreprocessor (events_train_clean, item_properties_train_clean)...")
        GNNPreprocessor(data_dir=data_dir, processed_dir=processed_dir).run()

    # 1) 초기 파이프라인 1회 실행 (ALS 기본 + GNN 기본 + ReRanker 기본)
    logger.info("Initial pipeline: ALS(default) -> GNN(default) -> ReRanker(default)")
    if not skip_als:
        run_als(default_config)
    if not skip_gnn:
        run_gnn(default_config)
    run_reranker(default_config)
    ndcg0, recall0 = evaluate_recommendations(processed_dir, top_k)
    rows.append({
        "Parameter": "default",
        "Value": "-",
        "NDCG": ndcg0,
        "Recall": recall0,
    })
    logger.info("Default config -> NDCG=%.4f, Recall=%.4f", ndcg0, recall0)

    # 2) 파라미터별로 한 가지씩 변경하여 실험
    for param_label, field_name, values in grids:
        # ALS weight는 ReRanker만 다시 실행하면 됨
        if field_name == "als_weight":
            for v in values:
                cfg = default_config.copy_with(**{field_name: v})
                run_reranker(cfg)
                ndcg, recall = evaluate_recommendations(processed_dir, top_k)
                rows.append({"Parameter": param_label, "Value": v, "NDCG": ndcg, "Recall": recall})
                logger.info("%s = %s -> NDCG=%.4f, Recall=%.4f", param_label, v, ndcg, recall)
            continue

        # ALS 관련: ALS 다시 실행 후 ReRanker
        if field_name.startswith("als_"):
            if skip_als:
                continue
            for v in values:
                cfg = default_config.copy_with(**{field_name: v})
                run_als(cfg)
                run_reranker(cfg)
                ndcg, recall = evaluate_recommendations(processed_dir, top_k)
                rows.append({"Parameter": param_label, "Value": v, "NDCG": ndcg, "Recall": recall})
                logger.info("%s = %s -> NDCG=%.4f, Recall=%.4f", param_label, v, ndcg, recall)
            continue

        # GNN 관련: GNN 다시 실행 후 ReRanker
        if field_name.startswith("gnn_"):
            if skip_gnn:
                continue
            for v in values:
                cfg = default_config.copy_with(**{field_name: v})
                run_gnn(cfg)
                run_reranker(cfg)
                ndcg, recall = evaluate_recommendations(processed_dir, top_k)
                rows.append({"Parameter": param_label, "Value": v, "NDCG": ndcg, "Recall": recall})
                logger.info("%s = %s -> NDCG=%.4f, Recall=%.4f", param_label, v, ndcg, recall)

    return pd.DataFrame(rows)


def print_results_table(df: pd.DataFrame) -> None:
    """결과 DataFrame을 표 형태로 출력."""
    if df.empty:
        return
    w = 14
    sep = "-" * (20 + w * 3)
    print("\n" + sep)
    print(f"{'Parameter':20} | {'Value':^{w}} | {'NDCG':^{w}} | {'Recall':^{w}}")
    print(sep)
    for _, r in df.iterrows():
        val = r["Value"]
        if isinstance(val, float) and 0 < abs(val) < 1e-2:
            val_str = f"{val:.2e}"
        else:
            val_str = str(val)
        print(f"{r['Parameter']:20} | {val_str:^{w}} | {r['NDCG']:^{w}.4f} | {r['Recall']:^{w}.4f}")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter sensitivity experiments")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="hyperparameter_sensitivity_results.csv",
        help="Output CSV path (default: hyperparameter_sensitivity_results.csv)",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Processed data directory",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Raw data directory",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-K for NDCG/Recall",
    )
    parser.add_argument(
        "--skip-initial-als",
        action="store_true",
        help="Skip initial ALS run (use existing als_* artifacts)",
    )
    parser.add_argument(
        "--skip-initial-gnn",
        action="store_true",
        help="Skip initial GNN run (use existing gnn_* artifacts)",
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    data_dir = Path(args.data_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = run_sensitivity_experiments(
        processed_dir=processed_dir,
        data_dir=data_dir,
        top_k=args.top_k,
        skip_als=args.skip_initial_als,
        skip_gnn=args.skip_initial_gnn,
    )

    out_path = Path(args.output)
    df.to_csv(out_path, index=False)
    logger.info("Results saved to %s", out_path.resolve())

    print_results_table(df)


if __name__ == "__main__":
    main()
