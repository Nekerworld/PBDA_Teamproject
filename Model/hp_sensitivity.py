"""
우선순위 하이퍼파라미터 민감도 실험 (one-at-a-time, 기여도 기본 방식)

- model.py 기본값을 디폴트로 두고, 우선순위 8개 파라미터만 한 번에 하나씩 변경.
- 기여도: Parameter × Value별 NDCG@K, Recall@K 표로 기록 (나머지 파라미터는 고정).

실행:
  python -m Model.run_priority_sensitivity
  python -m Model.run_priority_sensitivity --output results_priority_sensitivity.csv
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from Model.model import (
    ALSPreprocessor,
    GNNPreprocessor,
    set_global_seed,
    GLOBAL_SEED,
)
from Model.hp_comp import (
    SensitivityConfig,
    evaluate_recommendations,
    run_als,
    run_gnn,
    run_reranker,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
set_global_seed(GLOBAL_SEED)


# ---------------------------------------------------------------------------
# model.py와 동일한 기본값 (디폴트)
# ---------------------------------------------------------------------------

def get_default_config(processed_dir: Path, data_dir: Path, top_k: int = 50) -> SensitivityConfig:
    """model.py / als_debug 기본값으로 SensitivityConfig 생성."""
    return SensitivityConfig(
        processed_dir=processed_dir,
        data_dir=data_dir,
        top_k=top_k,
        # ALSRecommender 기본값
        als_factors=128,
        als_regularization=0.03,
        als_iterations=256,
        als_alpha=1.5,
        als_long_tail_bonus=0.35,
        als_long_tail_quantile=0.5,
        als_ensemble_alpha=0.5,
        # GNNEmbeddingGenerator 기본값
        gnn_embedding_dim=8,
        gnn_layers=2,
        gnn_epochs=1024,
        gnn_batch_size=8192,
        gnn_learning_rate=1e-3,
        gnn_reg=1e-4,
        gnn_num_negative=1,
        gnn_early_stopping_patience=50,
        # ReRanker 기본값
        als_weight=0.4,
    )


# ---------------------------------------------------------------------------
# 우선순위 파라미터 그리드 (한 번에 하나만 변경)
# 순서: als_weight → factors → embedding_dim → layers → alpha → regularization → learning_rate → long_tail_bonus
# ---------------------------------------------------------------------------

def get_priority_parameter_grids() -> List[Tuple[str, str, List[Any]]]:
    """
    (파라미터 표시 이름, SensitivityConfig 필드명, 값 목록).
    기본값을 포함한 값 목록으로, 해당 파라미터만 바꿔가며 실험.
    """
    return [
        ("ALS weight", "als_weight", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        ("ALS factors", "als_factors", [16, 32, 64, 128]),
        ("GNN embedding_dim", "gnn_embedding_dim", [8, 16, 32, 64]),
        ("GNN layers", "gnn_layers", [1, 2, 3]),
        ("ALS alpha", "als_alpha", [0.5, 1.0, 1.5, 2.0]),
        ("ALS regularization", "als_regularization", [0.01, 0.03, 0.1, 0.3]),
        ("GNN learning_rate", "gnn_learning_rate", [1e-4, 5e-4, 1e-3, 5e-3]),
        ("long_tail_bonus", "als_long_tail_bonus", [0.0, 0.2, 0.35, 0.5]),
    ]


def _clear_gnn_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# 실험 실행
# ---------------------------------------------------------------------------

def run_priority_sensitivity(
    processed_dir: Path,
    data_dir: Path,
    top_k: int = 50,
) -> pd.DataFrame:
    """
    우선순위 8개 하이퍼파라미터에 대해 one-at-a-time 실험 수행.
    기여도는 기본 방식: (Parameter, Value, NDCG@K, Recall@K) 표로 기록.
    """
    default_config = get_default_config(processed_dir, data_dir, top_k)
    grids = get_priority_parameter_grids()
    rows: List[Dict[str, Any]] = []

    # 0) 전처리 1회
    if not (processed_dir / "als_events_train.csv").exists():
        logger.info("ALSPreprocessor 실행 (als_events_*)...")
        ALSPreprocessor(data_dir=data_dir, processed_dir=processed_dir).run()
    if not (processed_dir / "events_train_clean.csv").exists():
        logger.info("GNNPreprocessor 실행...")
        GNNPreprocessor(data_dir=data_dir, processed_dir=processed_dir).run()

    # 1) 디폴트 설정으로 전체 파이프라인 1회 → baseline 행
    logger.info("디폴트 설정으로 파이프라인 실행 (baseline)")
    run_als(default_config)
    run_gnn(default_config)
    run_reranker(default_config)
    ndcg0, recall0 = evaluate_recommendations(processed_dir, top_k)
    rows.append({
        "Parameter": "(baseline)",
        "Value": "default",
        "NDCG": ndcg0,
        "Recall": recall0,
    })
    logger.info("Baseline -> NDCG=%.4f, Recall=%.4f", ndcg0, recall0)

    # 2) als_weight: ReRanker만 재실행
    for param_label, field_name, values in grids:
        if field_name != "als_weight":
            continue
        for v in values:
            cfg = default_config.copy_with(**{field_name: v})
            run_reranker(cfg)
            ndcg, recall = evaluate_recommendations(processed_dir, top_k)
            rows.append({"Parameter": param_label, "Value": v, "NDCG": ndcg, "Recall": recall})
            logger.info("%s = %s -> NDCG=%.4f, Recall=%.4f", param_label, v, ndcg, recall)
        break

    # 3) ALS 관련 (factors, alpha, regularization, long_tail_bonus): ALS + ReRanker
    for param_label, field_name, values in grids:
        if field_name not in ("als_factors", "als_alpha", "als_regularization", "als_long_tail_bonus"):
            continue
        for v in values:
            cfg = default_config.copy_with(**{field_name: v})
            run_als(cfg)
            run_reranker(cfg)
            ndcg, recall = evaluate_recommendations(processed_dir, top_k)
            rows.append({"Parameter": param_label, "Value": v, "NDCG": ndcg, "Recall": recall})
            logger.info("%s = %s -> NDCG=%.4f, Recall=%.4f", param_label, v, ndcg, recall)

    # 4) ALS를 디폴트로 복구 후 GNN 관련 실험
    logger.info("ALS 디폴트 복구 후 GNN 파라미터 실험")
    run_als(default_config)
    run_reranker(default_config)

    for param_label, field_name, values in grids:
        if field_name not in ("gnn_embedding_dim", "gnn_layers", "gnn_learning_rate"):
            continue
        for v in values:
            cfg = default_config.copy_with(**{field_name: v})
            run_gnn(cfg)
            run_reranker(cfg)
            ndcg, recall = evaluate_recommendations(processed_dir, top_k)
            rows.append({"Parameter": param_label, "Value": v, "NDCG": ndcg, "Recall": recall})
            logger.info("%s = %s -> NDCG=%.4f, Recall=%.4f", param_label, v, ndcg, recall)
        try:
            del _
        except NameError:
            pass
        _clear_gnn_memory()

    return pd.DataFrame(rows)


def print_results_table(df: pd.DataFrame) -> None:
    """기여도 기본 방식: Parameter, Value, NDCG, Recall 표 출력."""
    if df.empty:
        return
    w = 14
    sep = "-" * (22 + w * 3)
    print("\n" + sep)
    print(f"{'Parameter':22} | {'Value':^{w}} | {'NDCG':^{w}} | {'Recall':^{w}}")
    print(sep)
    for _, r in df.iterrows():
        val = r["Value"]
        if isinstance(val, float) and 0 < abs(val) < 1e-2:
            val_str = f"{val:.2e}"
        else:
            val_str = str(val)
        ndcg = r["NDCG"]
        rec = r["Recall"]
        ndcg_str = f"{ndcg:.4f}" if pd.notna(ndcg) else "N/A"
        rec_str = f"{rec:.4f}" if pd.notna(rec) else "N/A"
        print(f"{r['Parameter']:22} | {val_str:^{w}} | {ndcg_str:^{w}} | {rec_str:^{w}}")
    print(sep)
    print("\n(나머지 파라미터는 모두 model.py 기본값으로 고정)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="우선순위 하이퍼파라미터 민감도 실험 (one-at-a-time, 기여도 기본)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="priority_sensitivity_results.csv",
        help="결과 CSV 경로",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="전처리 데이터 디렉토리",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="원본 데이터 디렉토리",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="NDCG/Recall 평가 시 K",
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    data_dir = Path(args.data_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = run_priority_sensitivity(
        processed_dir=processed_dir,
        data_dir=data_dir,
        top_k=args.top_k,
    )

    out_path = Path(args.output)
    df.to_csv(out_path, index=False)
    logger.info("결과 저장: %s", out_path.resolve())

    print_results_table(df)


if __name__ == "__main__":
    main()
