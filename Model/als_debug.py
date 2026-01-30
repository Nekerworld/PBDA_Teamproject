"""
ALS 단독 테스트 스크립트
- 데이터 로드 → ALS 학습 → 추천 생성 → NDCG/Recall 평가
- 각 단계에서 상세 정보 출력
"""
import pandas as pd
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from typing import Dict, List, Set
from implicit.als import AlternatingLeastSquares


def main():
    processed_dir = Path("data/processed")
    top_k = 50
    
    # ========================================
    # 1. 데이터 로드
    # ========================================
    print("=" * 60)
    print("1. 데이터 로드")
    print("=" * 60)
    
    events_train = pd.read_csv(processed_dir / "events_train_clean.csv")
    events_test = pd.read_csv(processed_dir / "events_test.csv")
    
    print(f"events_train_clean: {len(events_train):,} rows")
    print(f"events_test: {len(events_test):,} rows")
    
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
    
    # 가중치 부여
    weight_map = {"view": 1.0, "addtocart": 6.0, "transaction": 12.0}
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
    
    # ========================================
    # 4. ALS 학습
    # ========================================
    print("\n" + "=" * 60)
    print("4. ALS 학습")
    print("=" * 60)
    
    model = AlternatingLeastSquares(
        factors=32,
        regularization=0.1,
        iterations=50,
        random_state=42,
    )
    
    print(f"ALS 파라미터: factors=32, regularization=0.1, iterations=50")
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
    
    # 추천 생성
    recommendations: Dict[int, List[int]] = {}
    
    print(f"\n{len(eval_users)}명에 대해 Top-{top_k} 추천 생성 중...")
    
    for i, user_id in enumerate(eval_users):
        user_idx = user_id_to_idx[user_id]
        user_items = user_item_matrix[user_idx:user_idx+1]
        
        item_indices, scores = model.recommend(
            np.array([user_idx]),
            user_items,
            N=top_k,
            filter_already_liked_items=True,
        )
        
        rec_item_ids = [idx_to_item_id[idx] for idx in item_indices[0] if idx in idx_to_item_id]
        recommendations[user_id] = rec_item_ids
        
        if (i + 1) % 1000 == 0:
            print(f"  진행: {i+1}/{len(eval_users)}")
    
    print(f"추천 생성 완료: {len(recommendations)} 유저")
    
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
    
    # 상세 분석: 몇 명이 hit이 있는지
    hit_counts = []
    ndcgs = []
    recalls = []
    
    for user_id in eval_users:
        if user_id not in positives_filtered:
            continue
        
        rel = positives_filtered[user_id]
        rec = recommendations.get(user_id, [])
        rec_set = set(rec[:top_k])
        
        hits = rec_set & rel
        hit_count = len(hits)
        hit_counts.append(hit_count)
        
        # Recall@K
        recall = hit_count / len(rel) if rel else 0.0
        recalls.append(recall)
        
        # NDCG@K
        dcg = 0.0
        for i, item_id in enumerate(rec[:top_k]):
            if item_id in rel:
                dcg += 1.0 / np.log2(i + 2)
        n = min(len(rel), top_k)
        idcg = sum(1.0 / np.log2(j + 2) for j in range(n))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)
    
    print(f"\n평가 유저 수: {len(hit_counts)}")
    print(f"Hit >= 1 인 유저 수: {sum(1 for h in hit_counts if h >= 1)}")
    print(f"Hit 평균: {np.mean(hit_counts):.4f}")
    print(f"Hit 최대: {max(hit_counts) if hit_counts else 0}")
    
    mean_ndcg = np.mean(ndcgs) if ndcgs else 0.0
    mean_recall = np.mean(recalls) if recalls else 0.0
    
    print(f"\n최종 결과:")
    print(f"  NDCG@{top_k}: {mean_ndcg:.6f}")
    print(f"  Recall@{top_k}: {mean_recall:.6f}")
    
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
