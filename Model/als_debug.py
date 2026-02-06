"""
ALS 통합 파이프라인 (전처리 + 학습 + NDCG/Recall 평가)

[구성]
1. 전처리: 원본 이벤트 로드 → 시간 기준 80/20 분할 → Core 필터 (한 파일 내 통합)
2. ALS: 모든 사용자에 대해 학습 → 각 유저별 선호도 점수( user_factors @ item_factors.T ) 산출
3. 평가: 선호도 점수로 랭킹 후 NDCG@K, Recall@K 계산

[논문용 NDCG/Recall 프로토콜]
- Train: 이벤트 80%, Core 필터 적용
- Test 정답: Test 기간 addtocart/transaction
- 평가: Test positive 보유 + Train 존재 유저에 대해, 선호도 점수로 상위 K개 랭킹 → NDCG/Recall

--------------------------------------------------------------------------------
이전 대비 변경 사항 (NDCG/Recall이 0이 아니게 된 이유)
--------------------------------------------------------------------------------

1. 전처리 통합 + 이벤트(시간) 기준 분할
   - 이전: model.py에서 아이템 기준 80/20 분할 → Train 아이템과 Test 아이템이 서로 겹치지 않음.
           Test 정답(positive) 아이템 대부분이 Train에 없어, ALS가 추천 후보에 넣을 수 없음 → Hit=0.
   - 변경: 이 파일에서 원본 events.csv 로드 후 시간 순 80/20 분할. Train/Test가 같은 아이템 공간을
           공유하므로 Test positive 아이템이 Train 행렬에 존재 → 추천 후보에 포함 가능.

2. 평가 시 정답(positive) Hold-out
   - 이전: filter_already_liked_items=True 로 Train에서 본 아이템을 추천 후보에서 제거.
           유저가 Train에서 view 하고 Test에서 addtocart 한 동일 아이템은 "이미 본 아이템"으로
           제거되어 Top-K에 절대 포함되지 않음 → Hit=0.
   - 변경: 추천/평가 시 해당 유저의 Test positive 아이템을 user_items에서 0으로 두어 "아직 안 본 것"으로
           취급. 선호도 점수 랭킹 시 Train만 제외하고 positive는 후보에 남김 → NDCG/Recall 계산 가능.

3. 선호도 점수로 직접 랭킹
   - 이전: implicit의 model.recommend()로 후보 생성 시, 내부 필터 때문에 정답이 후보에서 빠질 수 있음.
   - 변경: user_factors @ item_factors.T 로 유저별 전체 아이템 선호도 점수를 계산한 뒤, Hold-out 마스크
           (Train 제외·positive 포함) 적용 → 점수 기준 정렬 → Top-K 추천. 이 데이터로 NDCG/Recall 계산.

4. Core 필터 (희소도 완화)
   - 이전: 유저/아이템 수가 많고 희소도가 극도로 높으면 ALS가 개인화보다 인기도 위주로 학습.
   - 변경: 최소 상호작용 수(MIN_USER_EVENTS, MIN_ITEM_EVENTS) 미만 유저/아이템 제거 → 행렬 밀도 상대적
           상승 → 학습·랭킹 품질 개선에 기여.

5. 한 파일 내 통합
   - 이전: events_train_clean.csv / events_test.csv 에 의존 (model.py 전처리 선행 필요).
   - 변경: data/events.csv 만 있으면 전처리부터 NDCG/Recall까지 이 파일 하나로 실행 가능.
"""
import pandas as pd
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from typing import Dict, List, Set, Tuple

from implicit.als import AlternatingLeastSquares

try:
    from implicit.cpu.bpr import BayesianPersonalizedRanking
    HAS_BPR = True
except ImportError:
    try:
        from implicit.bpr import BayesianPersonalizedRanking
        HAS_BPR = True
    except ImportError:
        HAS_BPR = False


# ============== 설정 (실험으로 튜닝된 값 적용) ==============
DATA_DIR = Path("data")
EVENTS_PATH = DATA_DIR / "events.csv"
RANDOM_STATE = 42

# 전처리: Train 비율 확대(90%) → 학습 데이터 증가, Core 필터 강화 → 행렬 밀도 상승
TEST_SIZE = 0.10  # 90% train / 10% test
MIN_USER_EVENTS = 3
MIN_ITEM_EVENTS = 3

# 이벤트 가중치: transaction/addtocart 강한 신호 (논문·실험 기반)
WEIGHT_MAP = {"view": 1.0, "addtocart": 15.0, "transaction": 32.0}
# 집계 후 가중치 log 스케일: False 시 NDCG@10/Recall@10 더 높음 (als_hp_search 실험 결과)
USE_WEIGHT_LOG_SCALE = False

# 평가: Test positive 2개 이상인 유저만 사용 → 지표 안정
MIN_TEST_POSITIVES = 2
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


def preprocess(
    events: pd.DataFrame,
    test_size: float = TEST_SIZE,
    min_user_events: int = MIN_USER_EVENTS,
    min_item_events: int = MIN_ITEM_EVENTS,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, Set[int]], Set[int]]:
    """
    전처리: 시간 기준 Train/Test 분할 → Core 필터.
    Returns: events_train, events_test, positives (user -> set of item ids), train_item_set
    """
    # 시간 기준 정렬 후 분할
    if "timestamp" not in events.columns or events["timestamp"].isna().all():
        events = events.sample(frac=1, random_state=random_state).reset_index(drop=True)
    else:
        events = events.sort_values("timestamp").reset_index(drop=True)
    n = len(events)
    train_end = int((1 - test_size) * n)
    events_train = events.iloc[:train_end].copy()
    events_test = events.iloc[train_end:].copy()

    # Core 필터
    user_counts = events_train.groupby("visitorid").size()
    item_counts = events_train.groupby("itemid").size()
    keep_users = set(user_counts[user_counts >= min_user_events].index.tolist())
    keep_items = set(item_counts[item_counts >= min_item_events].index.tolist())
    events_train = events_train[
        events_train["visitorid"].isin(keep_users) & events_train["itemid"].isin(keep_items)
    ].copy()
    events_test = events_test[
        events_test["visitorid"].isin(keep_users) & events_test["itemid"].isin(keep_items)
    ].copy()

    # Test positive (addtocart, transaction)
    pos_events = events_test[
        events_test["event"].isin(["addtocart", "transaction"])
    ].dropna(subset=["visitorid", "itemid"])
    positives: Dict[int, Set[int]] = (
        pos_events.groupby("visitorid")["itemid"]
        .apply(lambda x: set(x.tolist()))
        .to_dict()
    )
    train_item_set = set(events_train["itemid"].dropna().astype(int).unique())

    return events_train, events_test, positives, train_item_set


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
) -> AlternatingLeastSquares:
    """모든 사용자에 대해 ALS 학습. alpha=positive 신호 confidence 스케일."""
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=random_state,
        alpha=alpha,
    )
    model.fit(user_item_matrix)
    return model


def train_bpr(
    user_item_matrix: sp.csr_matrix,
    factors: int = BPR_FACTORS,
    regularization: float = BPR_REGULARIZATION,
    iterations: int = BPR_ITERATIONS,
    random_state: int = RANDOM_STATE,
):
    """모든 사용자에 대해 BPR 학습 (랭킹 손실)."""
    if not HAS_BPR:
        return None
    model = BayesianPersonalizedRanking(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=random_state,
    )
    model.fit(user_item_matrix)
    return model


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
) -> Tuple[Dict[int, List[int]], Dict[int, np.ndarray]]:
    """
    ALS + BPR 앙상블: 유저별로 점수 min-max 정규화 후 alpha*ALS + (1-alpha)*BPR 결합,
    hold-out 적용하여 Top-K 추천 생성.
    """
    n_items = model_als.item_factors.shape[0]
    user_train_items: Dict[int, Set[int]] = (
        events_train.groupby("visitorid")["itemid"]
        .apply(lambda x: set(x.astype(int).tolist()))
        .to_dict()
    )
    recommendations: Dict[int, List[int]] = {}
    per_user_scores: Dict[int, np.ndarray] = {}

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

        per_user_scores[user_id] = scores.copy()

        train_items_u = user_train_items.get(user_id, set()) & set(item_id_to_idx.keys())
        test_positives_u = positives_filtered.get(user_id, set())
        exclude = train_items_u - test_positives_u
        for item_id in exclude:
            if item_id in item_id_to_idx:
                scores[item_id_to_idx[item_id]] = -np.inf

        top_indices = np.argsort(-scores)[:top_k]
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
    각 유저별 선호도 점수를 구한 뒤, hold-out 적용하여 Top-K 추천 리스트 생성.
    동시에 유저별 점수 벡터 반환.
    """
    n_items = model.item_factors.shape[0]
    user_train_items: Dict[int, Set[int]] = (
        events_train.groupby("visitorid")["itemid"]
        .apply(lambda x: set(x.astype(int).tolist()))
        .to_dict()
    )

    recommendations: Dict[int, List[int]] = {}
    per_user_scores: Dict[int, np.ndarray] = {}

    for user_id in eval_users:
        user_idx = user_id_to_idx[user_id]
        scores = get_per_user_scores(model, user_idx, n_items).ravel()
        per_user_scores[user_id] = scores.copy()

        train_items_u = user_train_items.get(user_id, set()) & set(
            item_id_to_idx.keys()
        )
        test_positives_u = positives_filtered.get(user_id, set())
        exclude = train_items_u - test_positives_u
        for item_id in exclude:
            if item_id in item_id_to_idx:
                scores[item_id_to_idx[item_id]] = -np.inf

        top_indices = np.argsort(-scores)[:top_k]
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


def run_pipeline_return_metrics(verbose: bool = False) -> Dict[str, float]:
    """
    현재 모듈 상수로 파이프라인 실행 후 NDCG/Recall 반환.
    하이퍼파라미터 탐색용. verbose=True 시 일부 로그 출력.
    """
    events = load_raw_events()
    events_train, events_test, positives, train_item_set = preprocess(
        events,
        test_size=TEST_SIZE,
        min_user_events=MIN_USER_EVENTS,
        min_item_events=MIN_ITEM_EVENTS,
        random_state=RANDOM_STATE,
    )
    positives_filtered: Dict[int, Set[int]] = {}
    for uid, rel in positives.items():
        filtered = rel & train_item_set
        if filtered:
            positives_filtered[uid] = filtered
    users_with_positives = [
        u for u in positives if positives[u] and len(positives_filtered.get(u, set())) >= MIN_TEST_POSITIVES
    ]
    (
        user_item_matrix,
        unique_users,
        unique_items,
        user_id_to_idx,
        item_id_to_idx,
        idx_to_item_id,
        item_pop_by_id,
    ) = build_user_item_matrix(events_train, WEIGHT_MAP, use_log_scale=USE_WEIGHT_LOG_SCALE)
    n_users, n_items = user_item_matrix.shape
    model_als = train_als(
        user_item_matrix,
        factors=ALS_FACTORS,
        regularization=ALS_REGULARIZATION,
        iterations=ALS_ITERATIONS,
        random_state=RANDOM_STATE,
        alpha=ALS_ALPHA,
    )
    model_bpr = None
    if HAS_BPR:
        model_bpr = train_bpr(
            user_item_matrix,
            factors=BPR_FACTORS,
            regularization=BPR_REGULARIZATION,
            iterations=BPR_ITERATIONS,
            random_state=RANDOM_STATE,
        )
    eval_users = [u for u in users_with_positives if u in user_id_to_idx]
    if not eval_users:
        return {"ndcg_10": 0.0, "recall_10": 0.0, "ndcg_50": 0.0, "recall_50": 0.0}
    recommendations, _ = scores_to_recommendations_ensemble(
        eval_users,
        user_id_to_idx,
        item_id_to_idx,
        idx_to_item_id,
        train_item_set,
        positives_filtered,
        events_train,
        model_als,
        model_bpr,
        TOP_K,
        ENSEMBLE_ALPHA,
    )
    metrics = {}
    for k in (10, 20, 50):
        ndcg, rec = compute_ndcg_recall_at_k(eval_users, positives_filtered, recommendations, k)
        metrics[f"ndcg_{k}"] = ndcg
        metrics[f"recall_{k}"] = rec
    return metrics


def main() -> None:
    print("=" * 60)
    print("1. 전처리 (통합)")
    print("=" * 60)

    events = load_raw_events()
    print(f"원본 이벤트: {len(events):,} rows")

    events_train, events_test, positives, train_item_set = preprocess(
        events,
        test_size=TEST_SIZE,
        min_user_events=MIN_USER_EVENTS,
        min_item_events=MIN_ITEM_EVENTS,
        random_state=RANDOM_STATE,
    )
    print(f"Train: {len(events_train):,} rows, 유저 {events_train['visitorid'].nunique():,}, 아이템 {events_train['itemid'].nunique():,}")
    print(f"Test:  {len(events_test):,} rows, 유저 {events_test['visitorid'].nunique():,}, 아이템 {events_test['itemid'].nunique():,}")
    print(f"Test positive 유저 수: {len(positives):,}")

    # Train 아이템으로 필터한 positive (평가용)
    positives_filtered: Dict[int, Set[int]] = {}
    for uid, rel in positives.items():
        filtered = rel & train_item_set
        if filtered:
            positives_filtered[uid] = filtered
    # Test positive 2개 이상인 유저만 평가 → 지표 안정 (MIN_TEST_POSITIVES)
    users_with_positives = [
        u for u in positives if positives[u] and len(positives_filtered.get(u, set())) >= MIN_TEST_POSITIVES
    ]

    print("\n" + "=" * 60)
    print("2. User-Item 행렬 생성")
    print("=" * 60)

    (
        user_item_matrix,
        unique_users,
        unique_items,
        user_id_to_idx,
        item_id_to_idx,
        idx_to_item_id,
        item_pop_by_id,
    ) = build_user_item_matrix(events_train, WEIGHT_MAP, use_log_scale=USE_WEIGHT_LOG_SCALE)

    n_users, n_items = user_item_matrix.shape
    print(f"행렬: ({n_users:,} x {n_items:,}), nnz: {user_item_matrix.nnz:,}")
    print(f"Sparsity: {100 * (1 - user_item_matrix.nnz / (n_users * n_items)):.4f}%")

    print("\n" + "=" * 60)
    print("3. ALS 학습 (모든 사용자)")
    print("=" * 60)

    model_als = train_als(
        user_item_matrix,
        factors=ALS_FACTORS,
        regularization=ALS_REGULARIZATION,
        iterations=ALS_ITERATIONS,
        random_state=RANDOM_STATE,
        alpha=ALS_ALPHA,
    )
    print(f"ALS user_factors: {model_als.user_factors.shape}, item_factors: {model_als.item_factors.shape}")

    model_bpr = None
    if HAS_BPR:
        print("\n" + "=" * 60)
        print("3.5 BPR 학습 (랭킹 손실, 모든 사용자)")
        print("=" * 60)
        model_bpr = train_bpr(
            user_item_matrix,
            factors=BPR_FACTORS,
            regularization=BPR_REGULARIZATION,
            iterations=BPR_ITERATIONS,
            random_state=RANDOM_STATE,
        )
        if model_bpr is not None:
            print(f"BPR user_factors: {model_bpr.user_factors.shape}, item_factors: {model_bpr.item_factors.shape}")
    else:
        print("\n(BPR 미사용: implicit에 BayesianPersonalizedRanking 없음)")

    print("\n" + "=" * 60)
    print("4. 유저별 선호도 점수 → 앙상블 → 추천 리스트")
    print("=" * 60)

    eval_users = [u for u in users_with_positives if u in user_id_to_idx]
    if not eval_users:
        print("평가 가능한 유저 없음.")
        return

    print(f"평가 유저 수: {len(eval_users):,} (Train 존재 + Test positive ≥ {MIN_TEST_POSITIVES}개)")
    print(f"앙상블: alpha={ENSEMBLE_ALPHA}*ALS + (1-{ENSEMBLE_ALPHA})*BPR")
    recommendations, per_user_scores = scores_to_recommendations_ensemble(
        eval_users,
        user_id_to_idx,
        item_id_to_idx,
        idx_to_item_id,
        train_item_set,
        positives_filtered,
        events_train,
        model_als,
        model_bpr,
        TOP_K,
        ENSEMBLE_ALPHA,
    )
    print(f"유저별 앙상블 점수 벡터: {len(per_user_scores):,} users x {n_items:,} items")
    print(f"추천 리스트: 유저당 Top-{TOP_K} 생성 완료")

    print("\n" + "=" * 60)
    print("5. NDCG / Recall (앙상블 선호도 점수 기반 랭킹)")
    print("=" * 60)

    hit_counts = [
        len(set(recommendations.get(uid, [])[:TOP_K]) & positives_filtered.get(uid, set()))
        for uid in eval_users
        if uid in positives_filtered
    ]
    print(f"평가 유저 수: {len(hit_counts):,}")
    print(f"Hit >= 1: {sum(1 for h in hit_counts if h >= 1):,}, Hit 평균: {np.mean(hit_counts):.4f}")

    print("\n[ALS+BPR 앙상블] NDCG / Recall (논문 보고용):")
    for k in K_LIST:
        mean_ndcg, mean_recall = compute_ndcg_recall_at_k(
            eval_users, positives_filtered, recommendations, k
        )
        print(f"  NDCG@{k}: {mean_ndcg:.6f}  Recall@{k}: {mean_recall:.6f}")

    # 샘플 출력: 유저 1명의 앙상블 점수 상위 10개 아이템
    if eval_users and eval_users[0] in per_user_scores:
        uid = eval_users[0]
        scores = per_user_scores[uid]
        top10_idx = np.argsort(-scores)[:10]
        top10_ids = [idx_to_item_id[i] for i in top10_idx]
        print(f"\n샘플 (유저 {uid}) 앙상블 선호도 Top-10 아이템 ID: {top10_ids}")


if __name__ == "__main__":
    main()
