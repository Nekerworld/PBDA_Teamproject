"""
ALS 단독 테스트 스크립트 (NDCG/Recall 평가용)

[논문용 NDCG/Recall 평가 프로토콜]
- Train: events_train_clean (이벤트 기반 80% 분할, Core 필터 적용 가능)
- Test 정답: Test 기간 중 addtocart/transaction 이벤트의 (user, item) 쌍
- 평가 대상: Test에 positive가 있고 Train에도 등장한 유저만 (추천 가능 유저)
- 추천: 모델 점수로 전체 아이템 정렬 후 상위 K개
- NDCG@K: 유저별 DCG/IDCG 평균 (정답이 상위에 있을수록 높음)
- Recall@K: 유저별 (상위 K개 중 Hit 수)/(해당 유저 정답 수) 평균
- 보고: NDCG@10, NDCG@50, Recall@10, Recall@50 (논문 표준)
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


def compute_ndcg_recall_at_k(
    eval_users: List[int],
    positives_filtered: Dict[int, Set[int]],
    recommendations: Dict[int, List[int]],
    k: int,
) -> Tuple[float, float]:
    """NDCG@K, Recall@K 평균 (논문용 프로토콜)."""
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


def main():
    processed_dir = Path("data/processed")
    top_k = 50
    # 논문용 NDCG/Recall 보고 K 값
    k_list = [10, 50]
    # NDCG/Recall 극대화용 설정
    MIN_USER_EVENTS = 2   # Core 필터: 최소 상호작용 수 (희소도 완화)
    MIN_ITEM_EVENTS = 2
    long_tail_bonus = 0.2   # 비인기 아이템 가산점 (정답이 롱테일일 때 Hit↑)
    long_tail_quantile = 0.5
    # 이벤트 가중치: transaction/addtocart 강조 (강한 positive 신호)
    weight_map = {"view": 1.0, "addtocart": 8.0, "transaction": 16.0}
    # ALS 하이퍼파라미터 (랭킹 품질 우선)
    als_factors = 64
    als_regularization = 0.08
    als_iterations = 80
    
    # ========================================
    # 1. 데이터 로드
    # ========================================
    print("=" * 60)
    print("1. 데이터 로드")
    print("=" * 60)
    
    events_train = pd.read_csv(processed_dir / "events_train_clean.csv")
    events_test = pd.read_csv(processed_dir / "events_test.csv")
    
    print(f"events_train_clean: {len(events_train):,} rows (원본)")
    print(f"events_test: {len(events_test):,} rows")
    
    # Core 필터: 희소도 완화 (최소 상호작용 수 미만 유저/아이템 제외)
    user_counts = events_train.groupby("visitorid").size()
    item_counts = events_train.groupby("itemid").size()
    keep_users = set(user_counts[user_counts >= MIN_USER_EVENTS].index.tolist())
    keep_items = set(item_counts[item_counts >= MIN_ITEM_EVENTS].index.tolist())
    events_train = events_train[
        events_train["visitorid"].isin(keep_users) & events_train["itemid"].isin(keep_items)
    ].copy()
    print(f"Core 필터 적용 (min_user={MIN_USER_EVENTS}, min_item={MIN_ITEM_EVENTS}): train {len(events_train):,} rows")
    
    # Train 데이터 통계
    train_users = events_train["visitorid"].dropna().astype(int).unique()
    train_items = events_train["itemid"].dropna().astype(int).unique()
    print(f"\nTrain 유저 수: {len(train_users):,}")
    print(f"Train 아이템 수: {len(train_items):,}")
    
    # Test 데이터 통계
    test_users = events_test["visitorid"].dropna().astype(int).unique()
    test_items = events_test["itemid"].dropna().astype(int).unique()
    print(f"\nTest 유저 수: {len(test_users):,}")
    print(f"Test 아이템 수: {len(test_items):,}")
    
    # Test에서 positive 이벤트 추출 (addtocart, transaction)
    pos_events = events_test[events_test["event"].isin(["addtocart", "transaction"])].copy()
    pos_events = pos_events.dropna(subset=["visitorid", "itemid"])
    pos_events["visitorid"] = pos_events["visitorid"].astype(int)
    pos_events["itemid"] = pos_events["itemid"].astype(int)
    
    positives: Dict[int, Set[int]] = (
        pos_events.groupby("visitorid")["itemid"]
        .apply(lambda x: set(x.tolist()))
        .to_dict()
    )
    
    users_with_positives = [u for u in test_users if u in positives and positives[u]]
    all_positive_items = set()
    for s in positives.values():
        all_positive_items |= s
    
    print(f"\nTest에서 positive 이벤트 수: {len(pos_events):,}")
    print(f"Test에서 positive 있는 유저 수: {len(users_with_positives):,}")
    print(f"Test에서 positive 아이템 (unique): {len(all_positive_items):,}")
    
    # ========================================
    # 2. Train/Test 아이템 겹침 확인
    # ========================================
    print("\n" + "=" * 60)
    print("2. Train/Test 아이템 겹침 확인")
    print("=" * 60)
    
    train_item_set = set(train_items)
    positive_in_train = all_positive_items & train_item_set
    
    print(f"Train 아이템 수: {len(train_item_set):,}")
    print(f"Test positive 아이템 수: {len(all_positive_items):,}")
    print(f"Test positive 중 Train에 있는 아이템 수: {len(positive_in_train):,} ({100*len(positive_in_train)/len(all_positive_items):.1f}%)")
    
    if len(positive_in_train) == 0:
        print("\n⚠️ 경고: Test positive 아이템이 Train에 하나도 없습니다!")
        print("   → ALS는 Train 아이템만 추천하므로 NDCG/Recall이 0이 됩니다.")
    elif len(positive_in_train) < 100:
        print(f"\n⚠️ 경고: Test positive 아이템 중 {len(positive_in_train)}개만 Train에 있습니다.")
        print(f"   Positive 아이템 ID (Train에 있는 것): {sorted(positive_in_train)[:20]}...")
    
    # ========================================
    # 3. User-Item 행렬 생성
    # ========================================
    print("\n" + "=" * 60)
    print("3. User-Item 행렬 생성")
    print("=" * 60)
    
    # 가중치 부여 (transaction/addtocart 강조 → NDCG/Recall 신호 강화)
    events_train["weight"] = events_train["event"].map(weight_map).fillna(1.0)
    
    interactions = (
        events_train.groupby(["visitorid", "itemid"], as_index=False)["weight"]
        .sum()
    )
    interactions = interactions[interactions["weight"] > 0]
    
    # user, item 인덱스 매핑
    user_ids, unique_users = pd.factorize(interactions["visitorid"], sort=True)
    item_ids, unique_items = pd.factorize(interactions["itemid"], sort=True)
    
    user_id_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item_id_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
    idx_to_item_id = {idx: iid for idx, iid in enumerate(unique_items)}
    
    # CSR 행렬 생성 (users x items)
    user_item_matrix = sp.coo_matrix(
        (interactions["weight"].astype(float), (user_ids, item_ids)),
        shape=(len(unique_users), len(unique_items)),
    ).tocsr()
    
    print(f"User-Item 행렬: {user_item_matrix.shape} (users x items)")
    print(f"Non-zero entries: {user_item_matrix.nnz:,}")
    print(f"Sparsity: {100 * (1 - user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1])):.4f}%")

    # 아이템 인기도 (Train 상호작용 수) — 롱테일 분석·리랭킹용
    item_pop_counts = interactions.groupby("itemid").size()
    item_pop_by_id = item_pop_counts.to_dict()

    # ========================================
    # 3.5 롱테일 분석
    # ========================================
    print("\n" + "=" * 60)
    print("3.5 롱테일 분석 (아이템 인기도)")
    print("=" * 60)
    percentiles = [0, 25, 50, 75, 100]
    perc_vals = np.percentile(item_pop_counts.values, percentiles)
    print("아이템당 Train 상호작용 수 분포 (percentile):")
    for p, v in zip(percentiles, perc_vals):
        print(f"  {p}%: {v:.1f}")
    long_tail_threshold = np.percentile(item_pop_counts.values, long_tail_quantile * 100)
    long_tail_item_ids = set(item_pop_counts[item_pop_counts <= long_tail_threshold].index.tolist())
    print(f"\n롱테일 정의: 상호작용 수 ≤ {long_tail_threshold:.1f} (하위 {int(long_tail_quantile*100)}%)")
    print(f"롱테일 아이템 수: {len(long_tail_item_ids):,} / {len(item_pop_counts):,} ({100*len(long_tail_item_ids)/len(item_pop_counts):.1f}%)")
    pos_in_train_list = list(positive_in_train)
    pos_long_tail = sum(1 for i in pos_in_train_list if i in long_tail_item_ids)
    print(f"Test positive 아이템 중 롱테일 비율: {pos_long_tail}/{len(pos_in_train_list)} ({100*pos_long_tail/max(len(pos_in_train_list),1):.1f}%)")
    if pos_long_tail >= len(pos_in_train_list) * 0.5:
        print("  → 정답(positive)이 대부분 롱테일이면 ALS가 인기 위주로 추천해 Hit이 낮을 수 있음.")
    
    # ========================================
    # 4. ALS 학습
    # ========================================
    print("\n" + "=" * 60)
    print("4. ALS 학습")
    print("=" * 60)
    
    model = AlternatingLeastSquares(
        factors=als_factors,
        regularization=als_regularization,
        iterations=als_iterations,
        random_state=42,
    )
    
    print(f"ALS 파라미터: factors={als_factors}, regularization={als_regularization}, iterations={als_iterations} (NDCG/Recall 극대화용)")
    print("학습 중...")
    model.fit(user_item_matrix)
    
    print(f"user_factors shape: {model.user_factors.shape}")
    print(f"item_factors shape: {model.item_factors.shape}")
    
    # ========================================
    # 5. 추천 생성 (테스트 유저 대상)
    # ========================================
    print("\n" + "=" * 60)
    print("5. 추천 생성 (테스트 유저 대상)")
    print("=" * 60)
    
    # 테스트 유저 중 Train에도 있는 유저만 추천 가능
    eval_users = [u for u in users_with_positives if u in user_id_to_idx]
    print(f"Positive 있는 테스트 유저 수: {len(users_with_positives):,}")
    print(f"이 중 Train에도 있는 유저 수 (추천 가능): {len(eval_users):,}")
    print(f"Train에 없는 유저 수 (추천 불가): {len(users_with_positives) - len(eval_users):,}")
    
    if len(eval_users) == 0:
        print("\n⚠️ 추천 가능한 유저가 없어 평가 불가!")
        return
    
    # 추천 생성 (N>top_k로 후보 뽑은 뒤 롱테일 보너스로 리랭킹)
    recommendations: Dict[int, List[int]] = {}
    recommend_n = top_k * 4 if long_tail_bonus > 0 else top_k  # 롱테일 리랭킹 여유 확보

    print(f"\n{len(eval_users)}명에 대해 Top-{top_k} 추천 생성 중... (롱테일 보너스={'%.2f' % long_tail_bonus if long_tail_bonus else '미적용'})")
    
    for i, user_id in enumerate(eval_users):
        user_idx = user_id_to_idx[user_id]
        user_items = user_item_matrix[user_idx:user_idx+1]
        
        item_indices, scores = model.recommend(
            np.array([user_idx]),
            user_items,
            N=recommend_n,
            filter_already_liked_items=True,
        )
        
        cand_ids = [idx_to_item_id[idx] for idx in item_indices[0] if idx in idx_to_item_id]
        cand_scores = scores[0][:len(cand_ids)].tolist()
        
        if long_tail_bonus > 0 and cand_ids:
            # 비인기(롱테일) 아이템에 가산점 부여 후 재정렬
            pop_list = [item_pop_by_id.get(iid, 0) for iid in cand_ids]
            new_scores = [
                s + long_tail_bonus * (1.0 / (p + 1.0))
                for s, p in zip(cand_scores, pop_list)
            ]
            sorted_pairs = sorted(zip(cand_ids, new_scores), key=lambda x: -x[1])
            rec_item_ids = [item_id for item_id, _ in sorted_pairs[:top_k]]
        else:
            rec_item_ids = cand_ids[:top_k]
        
        recommendations[user_id] = rec_item_ids
        
        if (i + 1) % 1000 == 0:
            print(f"  진행: {i+1}/{len(eval_users)}")
    
    print(f"추천 생성 완료: {len(recommendations)} 유저")
    # 추천 리스트 내 롱테일 비율
    rec_long_tail_ratios = []
    for uid in eval_users:
        rec_list = recommendations.get(uid, [])[:top_k]
        if not rec_list:
            continue
        n_lt = sum(1 for iid in rec_list if iid in long_tail_item_ids)
        rec_long_tail_ratios.append(n_lt / len(rec_list))
    if rec_long_tail_ratios:
        print(f"추천 Top-{top_k} 내 롱테일 아이템 비율 (평균): {np.mean(rec_long_tail_ratios)*100:.1f}%")
    
    # ========================================
    # 6. NDCG/Recall 계산
    # ========================================
    print("\n" + "=" * 60)
    print("6. NDCG/Recall 계산")
    print("=" * 60)
    
    # 먼저 positive 아이템을 Train 아이템으로 제한
    positives_filtered: Dict[int, Set[int]] = {}
    for uid, rel in positives.items():
        filtered = rel & train_item_set
        if filtered:
            positives_filtered[uid] = filtered
    
    print(f"원본 positives: {len(positives)} 유저")
    print(f"Train 아이템으로 필터링 후: {len(positives_filtered)} 유저")
    
    # 상세 분석: Hit 통계 (top_k 기준)
    hit_counts = []
    for user_id in eval_users:
        if user_id not in positives_filtered:
            continue
        rel = positives_filtered[user_id]
        rec_set = set(recommendations.get(user_id, [])[:top_k])
        hit_counts.append(len(rec_set & rel))
    
    print(f"\n평가 유저 수: {len(hit_counts)}")
    print(f"Hit >= 1 인 유저 수: {sum(1 for h in hit_counts if h >= 1)}")
    print(f"Hit 평균: {np.mean(hit_counts):.4f}")
    print(f"Hit 최대: {max(hit_counts) if hit_counts else 0}")
    
    # 논문용 NDCG/Recall 보고 (K=10, 50)
    print(f"\n[ALS] NDCG/Recall (논문 보고용):")
    for k in k_list:
        mean_ndcg, mean_recall = compute_ndcg_recall_at_k(
            eval_users, positives_filtered, recommendations, k
        )
        print(f"  NDCG@{k}: {mean_ndcg:.6f}  Recall@{k}: {mean_recall:.6f}")
    
    # BPR (랭킹 손실 기반) — NDCG/Recall에 맞춘 비교용
    if HAS_BPR:
        print("\n" + "=" * 60)
        print("BPR (ranking-oriented baseline, NDCG/Recall 동일 프로토콜)")
        print("=" * 60)
        model_bpr = BayesianPersonalizedRanking(
            factors=als_factors,
            regularization=0.01,
            iterations=als_iterations,
            random_state=42,
        )
        print("BPR 학습 중...")
        model_bpr.fit(user_item_matrix)
        rec_bpr: Dict[int, List[int]] = {}
        for user_id in eval_users:
            user_idx = user_id_to_idx.get(user_id)
            if user_idx is None:
                continue
            user_items = user_item_matrix[user_idx : user_idx + 1]
            item_indices, scores = model_bpr.recommend(
                np.array([user_idx]),
                user_items,
                N=recommend_n,
                filter_already_liked_items=True,
            )
            cand_ids = [idx_to_item_id[idx] for idx in item_indices[0] if idx in idx_to_item_id]
            cand_scores = scores[0][: len(cand_ids)].tolist()
            if long_tail_bonus > 0 and cand_ids:
                pop_list = [item_pop_by_id.get(iid, 0) for iid in cand_ids]
                new_scores = [
                    s + long_tail_bonus * (1.0 / (p + 1.0))
                    for s, p in zip(cand_scores, pop_list)
                ]
                sorted_pairs = sorted(zip(cand_ids, new_scores), key=lambda x: -x[1])
                rec_bpr[user_id] = [item_id for item_id, _ in sorted_pairs[:top_k]]
            else:
                rec_bpr[user_id] = cand_ids[:top_k]
        print("[BPR] NDCG/Recall (논문 보고용):")
        for k in k_list:
            mean_ndcg, mean_recall = compute_ndcg_recall_at_k(
                eval_users, positives_filtered, rec_bpr, k
            )
            print(f"  NDCG@{k}: {mean_ndcg:.6f}  Recall@{k}: {mean_recall:.6f}")
    else:
        print("\n(BPR 비교: implicit에 BayesianPersonalizedRanking 없음, pip install implicit 최신 버전)")
    
    # ========================================
    # 7. 샘플 유저 상세 분석
    # ========================================
    print("\n" + "=" * 60)
    print("7. 샘플 유저 상세 분석 (처음 5명)")
    print("=" * 60)
    
    sample_count = 0
    for user_id in eval_users:
        if user_id not in positives_filtered:
            continue
        
        rel = positives_filtered[user_id]
        rec = recommendations.get(user_id, [])[:10]  # Top 10만 표시
        hits = set(rec) & rel
        
        print(f"\n유저 {user_id}:")
        print(f"  Positive 아이템 (Train에 있는 것): {sorted(rel)[:10]}{'...' if len(rel) > 10 else ''} (총 {len(rel)}개)")
        print(f"  ALS 추천 Top-10: {rec}")
        print(f"  Hit: {hits} ({len(hits)}개)")
        
        sample_count += 1
        if sample_count >= 5:
            break


if __name__ == "__main__":
    main()
