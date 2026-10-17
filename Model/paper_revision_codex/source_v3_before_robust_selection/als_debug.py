"""ALS/BPR helpers using the shared chronological revision protocol.
Repeat recommendations are allowed uniformly; test labels never change candidates.
"""
import pandas as pd
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from typing import Dict, List, Set, Tuple

from implicit.als import AlternatingLeastSquares
try:
    from .data_protocol import DEFAULT_DATA, DEFAULT_PROCESSED, load_shared_split
except ImportError:
    from data_protocol import DEFAULT_DATA, DEFAULT_PROCESSED, load_shared_split

# Legacy search constants. The revision experiment does not train with these:
# WEIGHT_MAP used 15/32 and implicit BPR ignored magnitudes. Locked weights are
# learning.BEHAVIOR_WEIGHTS (1/6/12) via model.ALSRecommender.


# ============== 설정 (실험으로 튜닝된 값 적용) ==============
DATA_DIR = DEFAULT_DATA
EVENTS_PATH = DATA_DIR / "events.csv"
RANDOM_STATE = 42

# 공통 전처리 설정 (실제 분할은 data_protocol에서 한 번만 생성)
TEST_SIZE = 0.20  # common 80/20 chronological partition
MIN_USER_EVENTS = 3
MIN_ITEM_EVENTS = 3

# 이벤트 가중치: transaction/addtocart 강한 신호 (논문·실험 기반)
WEIGHT_MAP = {"view": 1.0, "addtocart": 15.0, "transaction": 32.0}
# 집계 후 가중치 log 스케일: False 시 NDCG@10/Recall@10 더 높음 (als_hp_search 실험 결과)
USE_WEIGHT_LOG_SCALE = False

# 공통 평가: positive가 하나 이상인 모든 warm-start 사용자
MIN_TEST_POSITIVES = 1
TOP_K = 50
K_LIST = [10, 20, 50]

# ALS: alpha=confidence 스케일, factors/reg/iter 실험 반영
ALS_FACTORS = 128
ALS_REGULARIZATION = 0.03
ALS_ITERATIONS = 256
ALS_ALPHA = 1.5  # positive 신호 강조 (Hu et al. confidence)

# BPR
BPR_FACTORS = 128
BPR_REGULARIZATION = 0.03
BPR_ITERATIONS = 256

# 앙상블 비율 (실험으로 선택)
ENSEMBLE_ALPHA = 0.5


def load_raw_events() -> pd.DataFrame:
    """원본 이벤트 로드."""
    events = pd.read_csv(
        EVENTS_PATH,
        usecols=["timestamp", "visitorid", "event", "itemid", "transactionid"],
    )
    events = events.dropna(subset=["visitorid", "itemid"])
    events["visitorid"] = events["visitorid"].astype(int)
    events["itemid"] = events["itemid"].astype(int)
    return events


def preprocess(*args, **kwargs):
    raise RuntimeError("Independent ALS preprocessing was removed. Run model.py preprocess once and use load_shared_split(processed_dir).")


def build_user_item_matrix(
    events_train: pd.DataFrame,
    weight_map: Dict[str, float],
    use_log_scale: bool = USE_WEIGHT_LOG_SCALE,
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray, Dict, Dict, Dict[int, int]]:
    """
    User-Item 행렬 및 매핑 생성.
    use_log_scale=True 시 집계 가중치에 1+log(1+x) 적용 (반복 이벤트 신뢰도 완만 증가).
    """
    events_train = events_train.copy()
    events_train["weight"] = events_train["event"].map(weight_map).fillna(1.0)
    interactions = (
        events_train.groupby(["visitorid", "itemid"], as_index=False)["weight"].sum()
    )
    interactions = interactions[interactions["weight"] > 0]
    if use_log_scale:
        interactions["weight"] = 1.0 + np.log1p(interactions["weight"].values)

    user_ids, unique_users = pd.factorize(interactions["visitorid"], sort=True)
    item_ids, unique_items = pd.factorize(interactions["itemid"], sort=True)
    user_id_to_idx = {int(uid): idx for idx, uid in enumerate(unique_users)}
    item_id_to_idx = {int(iid): idx for idx, iid in enumerate(unique_items)}
    idx_to_item_id = {idx: int(iid) for idx, iid in enumerate(unique_items)}

    matrix = sp.coo_matrix(
        (interactions["weight"].astype(float), (user_ids, item_ids)),
        shape=(len(unique_users), len(unique_items)),
    ).tocsr()

    item_pop_by_id = interactions.groupby("itemid").size().to_dict()
    item_pop_by_id = {int(k): int(v) for k, v in item_pop_by_id.items()}

    return (
        matrix,
        unique_users.values,
        unique_items.values,
        user_id_to_idx,
        item_id_to_idx,
        idx_to_item_id,
        item_pop_by_id,
    )


def train_als(
    user_item_matrix: sp.csr_matrix,
    factors: int = ALS_FACTORS,
    regularization: float = ALS_REGULARIZATION,
    iterations: int = ALS_ITERATIONS,
    random_state: int = RANDOM_STATE,
    alpha: float = ALS_ALPHA,
    num_threads: int = 8,
    use_gpu: bool = False,
) -> AlternatingLeastSquares:
    """
    모든 사용자에 대해 ALS 학습.
    Hu et al. (2008) 신뢰도: c_ui = 1 + alpha * r_ui 로 변환 후 fit.
    (입력 행렬 값은 r_ui, 비관측은 라이브러리에서 C=1로 처리)
    신뢰도를 직접 넣으므로 모델 alpha=1로 두어 이중 스케일 방지.
    """
    # c_ui = 1 + alpha * r_ui (식 (3)); 관측된 (u,i)에만 적용
    confidence_matrix = user_item_matrix.copy()
    confidence_matrix.data = 1.0 + alpha * confidence_matrix.data.astype(np.float64)
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=random_state,
        alpha=1.0,  # 신뢰도는 위에서 이미 1+alpha*r_ui 로 반영
        num_threads=num_threads,
        use_gpu=use_gpu,
    )
    model.fit(confidence_matrix)
    return model


def train_bpr(*args, **kwargs):
    """Removed entry point. implicit BPR does not train on interaction magnitudes."""
    raise RuntimeError(
        "implicit BPR is not the revision trainer. Use learning.train_weighted_bpr "
        "through model.ALSRecommender or experiments.py.")


def get_per_user_scores(model, user_idx: int, n_items: int) -> np.ndarray:
    """한 유저에 대한 전체 아이템 선호도 점수 (user_factors[u] @ item_factors.T). ALS/BPR 공통."""
    return model.user_factors[user_idx] @ model.item_factors.T


def scores_to_recommendations_ensemble(
    eval_users: List[int],
    user_id_to_idx: Dict[int, int],
    item_id_to_idx: Dict[int, int],
    idx_to_item_id: Dict[int, int],
    train_item_set: Set[int],
    positives_filtered: Dict[int, Set[int]],
    events_train: pd.DataFrame,
    model_als: AlternatingLeastSquares,
    model_bpr,
    top_k: int,
    ensemble_alpha: float,
    return_per_user_scores: bool = False,
) -> Tuple[Dict[int, List[int]], Dict[int, np.ndarray]]:
    """
    ALS + BPR 앙상블: 유저별로 점수 min-max 정규화 후 alpha*ALS + (1-alpha)*BPR 결합,
    공통 후보 전체에서 Top-K 추천 생성.
    return_per_user_scores=False 이면 per_user_scores 미저장 (메모리 절약, 추천만 필요할 때).
    """
    n_items = model_als.item_factors.shape[0]
    recommendations: Dict[int, List[int]] = {}
    per_user_scores: Dict[int, np.ndarray] = {}
    item_id_set = set(item_id_to_idx.keys())

    for user_id in eval_users:
        user_idx = user_id_to_idx[user_id]
        als_scores = get_per_user_scores(model_als, user_idx, n_items).ravel().astype(np.float64)
        if model_bpr is not None and HAS_BPR:
            bpr_scores = get_per_user_scores(model_bpr, user_idx, n_items).ravel().astype(np.float64)
            def _norm(s: np.ndarray) -> np.ndarray:
                smin, smax = s.min(), s.max()
                return (s - smin) / (smax - smin + 1e-9)
            combined = ensemble_alpha * _norm(als_scores) + (1.0 - ensemble_alpha) * _norm(bpr_scores)
            scores = combined.copy()
        else:
            scores = als_scores.copy()

        if return_per_user_scores:
            per_user_scores[user_id] = scores.copy()

        # Repeat-item policy is fixed; positives_filtered is compatibility-only and unused.

        top_indices = np.lexsort((np.array([idx_to_item_id[i] for i in range(n_items)]), -scores))[:top_k]
        cand_ids = [idx_to_item_id[i] for i in top_indices if scores[i] > -np.inf]
        rec_list = cand_ids[:top_k]

        recommendations[user_id] = rec_list

    return recommendations, per_user_scores


def scores_to_recommendations(
    eval_users: List[int],
    user_id_to_idx: Dict[int, int],
    item_id_to_idx: Dict[int, int],
    idx_to_item_id: Dict[int, int],
    train_item_set: Set[int],
    positives_filtered: Dict[int, Set[int]],
    events_train: pd.DataFrame,
    model: AlternatingLeastSquares,
    top_k: int,
) -> Tuple[Dict[int, List[int]], Dict[int, np.ndarray]]:
    """
    각 유저별 선호도 점수를 구한 뒤, 공통 후보 전체에서 Top-K 추천 리스트 생성.
    동시에 유저별 점수 벡터 반환.
    """
    n_items = model.item_factors.shape[0]
    recommendations: Dict[int, List[int]] = {}
    per_user_scores: Dict[int, np.ndarray] = {}
    item_id_set = set(item_id_to_idx.keys())

    for user_id in eval_users:
        user_idx = user_id_to_idx[user_id]
        scores = get_per_user_scores(model, user_idx, n_items).ravel()
        per_user_scores[user_id] = scores.copy()

        # Repeat-item policy is fixed; positives_filtered is compatibility-only and unused.

        top_indices = np.lexsort((np.array([idx_to_item_id[i] for i in range(n_items)]), -scores))[:top_k]
        cand_ids = [idx_to_item_id[i] for i in top_indices if scores[i] > -np.inf]
        rec_list = cand_ids[:top_k]

        recommendations[user_id] = rec_list

    return recommendations, per_user_scores


def compute_ndcg_recall_at_k(
    eval_users: List[int],
    positives_filtered: Dict[int, Set[int]],
    recommendations: Dict[int, List[int]],
    k: int,
) -> Tuple[float, float]:
    """선호도 점수로 만든 추천 리스트로 NDCG@K, Recall@K 평균 계산."""
    ndcgs, recalls = [], []
    for user_id in eval_users:
        if user_id not in positives_filtered:
            continue
        rel = positives_filtered[user_id]
        rec = recommendations.get(user_id, [])[:k]
        rec_set = set(rec)
        hits = rec_set & rel
        hit_count = len(hits)
        recall = hit_count / len(rel) if rel else 0.0
        recalls.append(recall)
        dcg = sum(1.0 / np.log2(i + 2) for i, item_id in enumerate(rec) if item_id in rel)
        n = min(len(rel), k)
        idcg = sum(1.0 / np.log2(j + 2) for j in range(n))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)
    mean_ndcg = np.mean(ndcgs) if ndcgs else 0.0
    mean_recall = np.mean(recalls) if recalls else 0.0
    return mean_ndcg, mean_recall


def run_pipeline_return_metrics(verbose: bool = False, processed_dir=DEFAULT_PROCESSED / "validation") -> Dict[str, float]:
    """Hyperparameter search must not bypass the locked revision trainer."""
    raise RuntimeError(
        "ALS/BPR search via als_debug is disabled. The locked settings are in "
        "experiments.py; implicit BPR and WEIGHT_MAP 15/32 are not the revision protocol.")


def main():
    # Reuse the same training and evaluation classes; never make a separate split.
    try:
        from .model import ALSRecommender
    except ImportError:
        from model import ALSRecommender
    ALSRecommender(processed_dir=DEFAULT_PROCESSED, top_k=200).run()


if __name__ == "__main__":
    main()
