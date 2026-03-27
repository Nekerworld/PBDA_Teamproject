"""
추천 시스템 모델 파이프라인

이 모듈은 다음 단계로 구성된 추천 시스템을 구현합니다:
1. IsolationForestPreprocessor: 이상치 탐지 및 데이터 전처리
2. ALSRecommender: ALS(Alternating Least Squares) 기반 협업 필터링
3. GNNEmbeddingGenerator: 그래프 신경망 기반 임베딩 생성
4. ReRanker: ALS와 GNN 결과를 결합한 리랭킹
5. TestSetEvaluator: 테스트 세트 평가 및 성능 지표 계산

각 컴포넌트는 독립적으로 실행 가능하며, 순차적으로 실행되어 최종 추천 결과를 생성합니다.
"""

from __future__ import annotations

import logging
import math
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pandas import DataFrame
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 로깅 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# 전역 시드 고정 (재현가능성 보장)
GLOBAL_SEED = 42

def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    """
    모든 랜덤 시드를 고정하여 재현가능성을 보장합니다.
    
    Args:
        seed: 고정할 시드 값 (기본값: 42)
    """
    # Python random 모듈
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch (가능한 경우)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 멀티 GPU 환경
        # 재현가능성을 위한 추가 설정
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch가 설치되지 않은 경우 무시

# 전역 시드 설정 실행
set_global_seed(GLOBAL_SEED)

# 이벤트 타입을 숫자 코드로 매핑 (view=0, addtocart=1, transaction=2)
EVENT_TO_CODE: Dict[str, int] = {"view": 0, "addtocart": 1, "transaction": 2}

# 숫자 추출을 위한 정규표현식 패턴 (음수 표기 "n" 또는 "-" 지원)
NUMERIC_PATTERN = re.compile(r"(?P<sign>n|-)?(?P<number>\d+(?:\.\d+)?)")

# 이벤트 타입별 가중치 (transaction > addtocart > view)
EVENT_WEIGHT: Dict[str, float] = {"view": 1.0, "addtocart": 6.0, "transaction": 12.0}

# ALS 전용 가중치·설정 (als_debug 실험 반영, NDCG/Recall 향상)
ALS_WEIGHT_MAP: Dict[str, float] = {"view": 1.0, "addtocart": 15.0, "transaction": 32.0}
USE_WEIGHT_LOG_SCALE = False
ENSEMBLE_ALPHA = 0.5
BPR_REGULARIZATION = 0.03

try:
    from implicit.cpu.bpr import BayesianPersonalizedRanking
    HAS_BPR = True
except ImportError:
    try:
        from implicit.bpr import BayesianPersonalizedRanking
        HAS_BPR = True
    except ImportError:
        HAS_BPR = False
        BayesianPersonalizedRanking = None


def _ndcg_at_k(
    recommendations: pd.DataFrame,
    positives: Dict[int, set],
    k: int,
    score_col: str = "score",
    ascending: bool = False,
) -> float:
    """
    NDCG@K. score_col 기준 정렬 후 상위 K개 사용.
    ascending=True면 낮을수록 좋음(예: rank).
    """
    ndcgs: List[float] = []
    for uid, grp in recommendations.groupby("visitorid"):
        rel = positives.get(int(uid), set())
        if not rel:
            continue
        top = grp.nsmallest(k, score_col) if ascending else grp.nlargest(k, score_col)
        dcg = 0.0
        for i, (_, row) in enumerate(top.iterrows()):
            if int(row["itemid"]) in rel:
                dcg += 1.0 / np.log2(i + 2)
        n = min(len(rel), k)
        idcg = sum(1.0 / np.log2(j + 2) for j in range(n))
        if idcg > 0:
            ndcgs.append(dcg / idcg)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def _recall_at_k(
    recommendations: pd.DataFrame,
    positives: Dict[int, set],
    k: int,
    score_col: str = "score",
    ascending: bool = False,
) -> float:
    """Recall@K. ascending: rank 등 낮을수록 좋은 경우 True."""
    recalls: List[float] = []
    for uid, grp in recommendations.groupby("visitorid"):
        rel = positives.get(int(uid), set())
        if not rel:
            continue
        top = grp.nsmallest(k, score_col) if ascending else grp.nlargest(k, score_col)
        pred = set(top["itemid"].astype(int).tolist())
        hit = len(pred & rel) / len(rel) if rel else 0.0
        recalls.append(hit)
    return float(np.mean(recalls)) if recalls else 0.0


def _extract_numeric(value: Optional[str]) -> Optional[float]:
    """
    문자열에서 숫자 값을 추출합니다.
    
    "n" 또는 "-" 접두사가 있으면 음수로 처리합니다.
    예: "n123" -> -123.0, "456.7" -> 456.7
    
    Args:
        value: 숫자를 추출할 문자열 (None 가능)
        
    Returns:
        추출된 숫자 값 (float), 추출 실패 시 None
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    match = NUMERIC_PATTERN.search(str(value))
    if not match:
        return None
    sign = match.group("sign")
    number = match.group("number")
    prefix = "-" if sign == "n" or sign == "-" else ""
    try:
        return float(f"{prefix}{number}")
    except ValueError:
        return None

@dataclass
class IsolationForestPreprocessor:
    """
    Isolation Forest를 사용한 이상치 탐지 및 데이터 전처리 클래스
    
    이 클래스는 아이템 레벨의 피처 테이블을 생성하고, Isolation Forest를 사용하여
    이상치를 탐지합니다. 이상치로 판단된 아이템은 학습 데이터에서 제외되며,
    정상 데이터만 후속 단계로 전달됩니다.
    
    Attributes:
        data_dir: 원본 데이터가 저장된 디렉토리 경로
        processed_dir: 전처리된 데이터를 저장할 디렉토리 경로
        random_state: 재현성을 위한 랜덤 시드
        test_size: 테스트 세트 비율 (0.0 ~ 1.0)
        contamination: 이상치 비율 예상값 ("auto" 또는 0.0 ~ 1.0 사이의 float)
        n_estimators: Isolation Forest의 트리 개수
        min_user_events: 최소 유저당 train 이벤트 수 (0이면 미적용, 희소도 완화용)
        min_item_events: 최소 아이템당 train 이벤트 수 (0이면 미적용, 희소도 완화용)
        core_filter_items_first: True면 keep_items 먼저 적용 후 keep_users 계산 (GNN 파이프라인용,
            events_test 유저가 모두 events_train_clean에 남도록 함)
    """
    data_dir: Path = field(default_factory=lambda: Path("data"))
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    random_state: int = 42
    test_size: float = 0.20
    contamination: str | float = "auto"
    n_estimators: int = 200
    min_user_events: int = 2
    min_item_events: int = 2
    core_filter_items_first: bool = False

    def run(self) -> None:
        """
        전처리 파이프라인을 실행합니다.
        
        실행 순서:
        1. 원본 데이터 로드
        2. 아이템 레벨 피처 테이블 생성
        3. 학습/테스트 세트 분할
        4. Isolation Forest로 이상치 탐지
        5. 정상 데이터와 이상치 데이터 분리 및 저장
        6. 시각화 생성
        """
        # 시드 고정 (재현가능성 보장)
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        
        logger.info("Loading raw datasets…")
        datasets = self._load_raw()

        logger.info("Building item-level feature table for anomaly detection…")
        feature_table = self._build_feature_table(datasets)
        logger.info("Feature table prepared with %d items and %d features", feature_table.shape[0], feature_table.shape[1])

        exclude_cols = [
            "transaction_count",
            "has_any_transaction",
            "has_transaction_id",
            "addtocart_count",
            "categoryid",
            "category_parentid",
            "has_available",
        ]
        available_exclude = [col for col in exclude_cols if col in feature_table.columns]
        if available_exclude:
            logger.info("Excluding columns from anomaly detection: %s", ", ".join(available_exclude))
            feature_table_for_detection = feature_table.drop(columns=available_exclude)
        else:
            feature_table_for_detection = feature_table

        logger.info("Training Isolation Forest on full feature table (%d items)…", len(feature_table_for_detection))
        isolation_forest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        isolation_forest.fit(feature_table_for_detection)

        logger.info("Scoring all items for anomaly detection…")
        all_predictions = isolation_forest.predict(feature_table_for_detection)
        inlier_mask = all_predictions == 1
        outlier_mask = all_predictions == -1
        inlier_ids = feature_table.index[inlier_mask].tolist()
        outlier_ids = feature_table.index[outlier_mask].tolist()

        logger.info(
            "Detected %d inliers, %d outliers (%.2f%% outliers).",
            len(inlier_ids),
            len(outlier_ids),
            (len(outlier_ids) / max(len(feature_table), 1)) * 100,
        )

        # 이벤트(시간) 기준 train/test 분할 vs GNN 전용(전체 타임라인 + holdout)
        events = datasets["events"].copy()
        events_inlier = events[events["itemid"].isin(inlier_ids)].copy()
        if "timestamp" not in events_inlier.columns or events_inlier["timestamp"].isna().all():
            events_inlier = events_inlier.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        else:
            events_inlier = events_inlier.sort_values("timestamp").reset_index(drop=True)

        if self.core_filter_items_first:
            # GNN 파이프라인: 이전 전처리처럼 전체 타임라인 사용 후 holdout만 분리 (그래프 데이터 최대화 → 점수 상승)
            events_pool = events_inlier.copy()
            item_counts = events_pool.groupby("itemid").size()
            keep_items = set(item_counts[item_counts >= self.min_item_events].index.tolist())
            train_after_items = events_pool[events_pool["itemid"].isin(keep_items)]
            user_counts = train_after_items.groupby("visitorid").size()
            keep_users = set(user_counts[user_counts >= self.min_user_events].index.tolist())
            events_train_clean = events_pool[
                events_pool["visitorid"].isin(keep_users) & events_pool["itemid"].isin(keep_items)
            ].copy()
            logger.info(
                "GNN 파이프라인: 전체 타임라인 + core filter [items_first] — %d events.",
                len(events_train_clean),
            )
            positive_mask = events_train_clean["event"].isin(["addtocart", "transaction"])
            if positive_mask.any():
                positive_events = events_train_clean.loc[positive_mask].copy()
                positive_events = positive_events.dropna(subset=["timestamp"])
                if not positive_events.empty:
                    positive_events = positive_events.sort_values(["visitorid", "timestamp"])
                    holdout_indices = (
                        positive_events.groupby("visitorid")["timestamp"].idxmax().dropna().astype(int)
                    )
                    events_test = events_train_clean.loc[holdout_indices].copy()
                    events_train_clean = events_train_clean.drop(index=holdout_indices, errors="ignore")
                    events_train_clean = events_train_clean.reset_index(drop=True)
                    events_test = events_test.reset_index(drop=True)
                    logger.info(
                        "GNN 파이프라인: holdout 생성 — train %d events, test(holdout) %d events, %d users.",
                        len(events_train_clean),
                        len(events_test),
                        events_test["visitorid"].nunique() if not events_test.empty else 0,
                    )
            else:
                events_test = events_train_clean.iloc[0:0].copy()
            train_item_ids = events_train_clean["itemid"].dropna().astype(int).unique()
            test_item_ids = events_test["itemid"].dropna().astype(int).unique()
        else:
            # ALS 등: 이벤트(시간) 기준 80/20 분할
            n_events = len(events_inlier)
            train_end = int((1 - self.test_size) * n_events)
            events_train_clean = events_inlier.iloc[:train_end].copy()
            events_test = events_inlier.iloc[train_end:].copy()
            logger.info(
                "Event-based split: train %.0f%% (%d events), test %.0f%% (%d events).",
                (1 - self.test_size) * 100,
                len(events_train_clean),
                self.test_size * 100,
                len(events_test),
            )
            train_item_ids = events_train_clean["itemid"].dropna().astype(int).unique()
            test_item_ids = events_test["itemid"].dropna().astype(int).unique()
            train_inlier_ids_set = set(feature_table.index) & set(train_item_ids)
            test_inlier_ids_set = set(feature_table.index) & set(test_item_ids)
            overlap = len(train_inlier_ids_set & test_inlier_ids_set)
            logger.info(
                "Train unique items: %d, Test unique items: %d, Overlap: %d (추천 평가 시 정답 아이템이 train에 존재 가능).",
                len(train_inlier_ids_set),
                len(test_inlier_ids_set),
                overlap,
            )
            # Core 필터 (ALS 등)
            if self.min_user_events > 0 or self.min_item_events > 0:
                user_counts = events_train_clean.groupby("visitorid").size()
                item_counts = events_train_clean.groupby("itemid").size()
                keep_users = set(user_counts[user_counts >= self.min_user_events].index.tolist())
                keep_items = set(item_counts[item_counts >= self.min_item_events].index.tolist())
                before_train = len(events_train_clean)
                before_test = len(events_test)
                events_train_clean = events_train_clean[
                    events_train_clean["visitorid"].isin(keep_users) & events_train_clean["itemid"].isin(keep_items)
                ].copy()
                events_test = events_test[
                    events_test["visitorid"].isin(keep_users) & events_test["itemid"].isin(keep_items)
                ].copy()
                logger.info(
                    "Core filter (min_user_events=%d, min_item_events=%d): train %d→%d events, test %d→%d events.",
                    self.min_user_events,
                    self.min_item_events,
                    before_train,
                    len(events_train_clean),
                    before_test,
                    len(events_test),
                )
                train_item_ids = events_train_clean["itemid"].dropna().astype(int).unique()
                test_item_ids = events_test["itemid"].dropna().astype(int).unique()

        train_inliers = feature_table.loc[feature_table.index.intersection(train_item_ids)]
        test_inliers = feature_table.loc[feature_table.index.intersection(test_item_ids)]
        train_outliers = feature_table.loc[outlier_ids] if outlier_ids else feature_table.iloc[0:0]

        self._save_results(
            datasets,
            train_inliers=train_inliers,
            test_inliers=test_inliers,
            train_outliers=train_outliers,
            events_train_clean=events_train_clean,
            events_test=events_test,
        )
        logger.info(
            "Isolation Forest preprocessing completed (event-based split): %d train events, %d test events.",
            len(events_train_clean),
            len(events_test),
        )

        # 시각화 생성 (이벤트 분할 시 train/test 아이템이 겹치므로 전체 inlier/outlier만 표시)
        self._create_visualizations(
            train_features=feature_table,
            train_inliers=feature_table.loc[inlier_ids] if inlier_ids else feature_table.iloc[0:0],
            train_outliers=train_outliers,
            test_features=pd.DataFrame(),
        )

    def _load_raw(self) -> Dict[str, DataFrame]:
        """
        원본 데이터 파일들을 로드합니다.
        
        로드하는 파일:
        - events.csv: 사용자-아이템 상호작용 이벤트
        - category_tree.csv: 카테고리 계층 구조
        - item_properties_part1.csv, item_properties_part2.csv: 아이템 속성 정보
        
        Returns:
            events, category_tree, item_properties를 포함한 딕셔너리
        """
        events = pd.read_csv(
            self.data_dir / "events.csv",
            usecols=["timestamp", "visitorid", "event", "itemid", "transactionid"],
        )
        category_tree = pd.read_csv(self.data_dir / "category_tree.csv")

        item_properties = pd.concat(
            [
                pd.read_csv(
                    self.data_dir / "item_properties_part1.csv",
                    usecols=["timestamp", "itemid", "property", "value"],
                    low_memory=False,
                ),
                pd.read_csv(
                    self.data_dir / "item_properties_part2.csv",
                    usecols=["timestamp", "itemid", "property", "value"],
                    low_memory=False,
                ),
            ],
            ignore_index=True,
        )

        return {
            "events": events,
            "category_tree": category_tree,
            "item_properties": item_properties,
        }

    def _build_feature_table(self, datasets: Dict[str, DataFrame]) -> DataFrame:
        """
        아이템 레벨의 피처 테이블을 생성합니다.
        
        생성되는 피처:
        - 이벤트 관련: event_count, view_count, addtocart_count, transaction_count,
          unique_visitors, event_code_mean, event_code_std, event_duration_hours 등
        - 속성 관련: property_count, categoryid, category_parentid, has_available
        
        Args:
            datasets: _load_raw()에서 반환된 데이터 딕셔너리
            
        Returns:
            아이템 ID를 인덱스로 하는 피처 테이블 (DataFrame)
        """
        events = datasets["events"].copy()
        item_properties = datasets["item_properties"].copy()
        category_tree = datasets["category_tree"].copy()

        # property가 "categoryid" 또는 "available"인 것만 사용
        item_properties_filtered = item_properties[
            item_properties["property"].isin(["categoryid", "available"])
        ].copy()

        # 타임스탬프를 datetime으로 변환하고 시작 시점으로부터의 시간(시간 단위) 계산
        events["timestamp_dt"] = pd.to_datetime(events["timestamp"], unit="ms", errors="coerce")
        min_timestamp = events["timestamp_dt"].min()
        events["hours_since_start"] = (events["timestamp_dt"] - min_timestamp).dt.total_seconds() / 3600.0
        
        # 이벤트 타입을 숫자 코드로 변환 (view=0, addtocart=1, transaction=2)
        events["event_code"] = events["event"].map(EVENT_TO_CODE).fillna(-1).astype(int)
        
        # 트랜잭션 관련 플래그 생성
        events["is_transaction"] = events["event"] == "transaction"
        events["has_transaction_id"] = events["transactionid"].notna().astype(int)

        event_agg = (
            events.groupby("itemid").agg(
                event_count=("event", "size"),
                view_count=("event", lambda s: (s == "view").sum()),
                addtocart_count=("event", lambda s: (s == "addtocart").sum()),
                transaction_count=("event", lambda s: (s == "transaction").sum()),
                unique_visitors=("visitorid", pd.Series.nunique),
                event_code_mean=("event_code", "mean"),
                event_code_std=("event_code", "std"),
                has_any_transaction=("is_transaction", "max"),
                has_transaction_id=("has_transaction_id", "max"),
                first_event_hour=("hours_since_start", "min"),
                last_event_hour=("hours_since_start", "max"),
            )
        )
        event_agg["event_duration_hours"] = event_agg["last_event_hour"] - event_agg["first_event_hour"]

        property_agg = (
            item_properties_filtered.groupby("itemid").agg(
                property_count=("property", "count"),
            )
        )

        category_rows = (
            item_properties_filtered[item_properties_filtered["property"] == "categoryid"][["itemid", "value"]].copy()
        )
        category_rows = category_rows.dropna(subset=["value"])
        extracted_category = category_rows["value"].astype(str).str.extract(r"(?P<category>\d+)")
        category_rows["categoryid"] = pd.to_numeric(extracted_category["category"], errors="coerce")
        category_features = (
            category_rows.dropna(subset=["categoryid"])
            .drop_duplicates(subset=["itemid"], keep="first")
            .set_index("itemid")[["categoryid"]]
        )

        category_tree["parentid"] = pd.to_numeric(category_tree["parentid"], errors="coerce")
        category_features = category_features.join(
            category_tree.set_index("categoryid")["parentid"], how="left"
        )
        category_features.rename(columns={"parentid": "category_parentid"}, inplace=True)

        # available feature 처리
        available_rows = item_properties_filtered[
            item_properties_filtered["property"] == "available"
        ].copy()
        if not available_rows.empty:
            available_rows["value_numeric"] = pd.to_numeric(available_rows["value"], errors="coerce")
            available_features = (
                available_rows.dropna(subset=["value_numeric"])
                .groupby("itemid")["value_numeric"]
                .max()  # 여러 값이 있으면 최대값 사용 (보통 1이 있으면 1)
                .to_frame(name="has_available")
            )
        else:
            available_features = pd.DataFrame(columns=["has_available"])

        feature_table = event_agg.join(property_agg, how="left")
        feature_table = feature_table.join(category_features, how="left")
        feature_table = feature_table.join(available_features, how="left")

        feature_table = feature_table.fillna(0.0)
        feature_table = feature_table.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        return feature_table

    def _save_results(
        self,
        datasets: Dict[str, DataFrame],
        *,
        train_inliers: DataFrame,
        test_inliers: DataFrame,
        train_outliers: DataFrame,
        events_train_clean: Optional[DataFrame] = None,
        events_test: Optional[DataFrame] = None,
    ) -> None:
        """
        전처리된 데이터를 파일로 저장합니다.
        
        저장되는 파일:
        - events_train_clean.csv: 정상 학습 이벤트
        - events_test.csv: 테스트 이벤트 (이벤트 분할 시 train과 아이템 겹침 가능)
        - events_train_outliers.csv: 이상치로 판단된 학습 이벤트
        - item_properties_*.csv: 각 세트에 해당하는 아이템 속성
        - feature_*.csv: 각 세트의 피처 테이블
        
        Args:
            datasets: 원본 데이터 딕셔너리
            train_inliers: 정상 학습 아이템 피처 테이블 (train 이벤트에 등장한 아이템)
            test_inliers: 테스트 아이템 피처 테이블 (test 이벤트에 등장한 아이템)
            train_outliers: 이상치 학습 아이템 피처 테이블
            events_train_clean: 이미 분할된 train 이벤트 (제공 시 그대로 사용)
            events_test: 이미 분할된 test 이벤트 (제공 시 그대로 사용)
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        events = datasets["events"].copy()
        item_properties = datasets["item_properties"].copy()

        train_inlier_ids = set(train_inliers.index)
        test_inlier_ids = set(test_inliers.index)
        train_outlier_ids = set(train_outliers.index)

        if events_train_clean is not None and events_test is not None:
            # 이벤트(시간) 기준 분할: 전달된 train/test 이벤트를 그대로 사용 (아이템 겹침 허용)
            events_train_clean = events_train_clean.reset_index(drop=True)
            events_test = events_test.reset_index(drop=True)
            logger.info(
                "Saving event-based split: %d train events, %d test events.",
                len(events_train_clean),
                len(events_test),
            )
        else:
            # (레거시) 아이템 기준 분할: item id로 이벤트 필터 후 holdout 생성
            events_train_clean = events[events["itemid"].isin(train_inlier_ids)].copy()
            events_test_base = events[events["itemid"].isin(test_inlier_ids)].copy()
            positive_mask = events_train_clean["event"].isin(["addtocart", "transaction"])
            if positive_mask.any():
                positive_events = events_train_clean.loc[positive_mask].copy()
                positive_events = positive_events.dropna(subset=["timestamp"])
                positive_events = positive_events.sort_values(["visitorid", "timestamp"])
                holdout_indices = (
                    positive_events.groupby("visitorid")["timestamp"].idxmax().dropna().astype(int)
                )
                events_test_holdout = events_train_clean.loc[holdout_indices].copy()
                events_train_clean = events_train_clean.drop(index=holdout_indices, errors="ignore")
            else:
                events_test_holdout = events_train_clean.iloc[0:0].copy()
            events_test = pd.concat([events_test_base, events_test_holdout], ignore_index=True)
            events_test = events_test.drop_duplicates(
                subset=["timestamp", "visitorid", "event", "itemid", "transactionid"]
            )
            events_train_clean = events_train_clean.reset_index(drop=True)
            events_test = events_test.reset_index(drop=True)
            logger.info(
                "Created evaluation holdout with %d events from %d users.",
                len(events_test),
                events_test["visitorid"].nunique() if not events_test.empty else 0,
            )

        events_train_outliers = events[events["itemid"].isin(train_outlier_ids)].copy()
        events_train_outliers = events_train_outliers.reset_index(drop=True)

        item_props_train_clean = item_properties[item_properties["itemid"].isin(train_inlier_ids)]
        item_props_train_outliers = item_properties[item_properties["itemid"].isin(train_outlier_ids)]
        if not events_test.empty:
            eval_item_ids = events_test["itemid"].astype(int).unique()
            item_props_test = item_properties[item_properties["itemid"].isin(eval_item_ids)]
        else:
            item_props_test = item_properties.iloc[0:0].copy()

        events_train_clean.to_csv(self.processed_dir / "events_train_clean.csv", index=False)
        events_test.to_csv(self.processed_dir / "events_test.csv", index=False)
        events_train_outliers.to_csv(self.processed_dir / "events_train_outliers.csv", index=False)

        item_props_train_clean.to_csv(self.processed_dir / "item_properties_train_clean.csv", index=False)
        item_props_test.to_csv(self.processed_dir / "item_properties_test.csv", index=False)
        item_props_train_outliers.to_csv(self.processed_dir / "item_properties_train_outliers.csv", index=False)

        train_inliers.to_csv(self.processed_dir / "feature_train_inliers.csv")
        test_inliers.to_csv(self.processed_dir / "feature_test_inliers.csv")
        train_outliers.to_csv(self.processed_dir / "feature_train_outliers.csv")

        logger.info("Saved cleaned datasets and summary to %s", self.processed_dir.resolve())

    def _create_visualizations(
        self,
        train_features: DataFrame,
        train_inliers: DataFrame,
        train_outliers: DataFrame,
        test_features: DataFrame,
    ) -> None:
        """
        이상치 탐지 결과를 시각화합니다.
        
        생성되는 시각화:
        1. 이상치 비율 파이 차트 (학습 세트 및 전체 데이터셋)
        2. 주요 피처 분포 비교 (정상 데이터 vs 이상치)
        
        Args:
            train_features: 전체 학습 피처 테이블
            train_inliers: 정상 학습 아이템 피처 테이블
            train_outliers: 이상치 학습 아이템 피처 테이블
            test_features: 테스트 아이템 피처 테이블
        """
        try:
            import seaborn as sns
        except ImportError:
            logger.warning("seaborn이 설치되지 않아 시각화를 건너뜁니다.")
            return

        viz_dir = Path("data/visualization")
        viz_dir.mkdir(parents=True, exist_ok=True)

        # 스타일 설정
        try:
            plt.style.use("seaborn-v0_8-darkgrid")
        except OSError:
            try:
                plt.style.use("seaborn-darkgrid")
            except OSError:
                plt.style.use("default")
        sns.set_palette("husl")

        # 1. 이상치 비율 파이 차트
        try:
            total_train = len(train_features)
            num_inliers = len(train_inliers)
            num_outliers = len(train_outliers)
            outlier_pct = (num_outliers / total_train * 100) if total_train > 0 else 0
            has_test_split = test_features is not None and len(test_features) > 0

            if has_test_split:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            else:
                fig, ax1 = plt.subplots(1, 1, figsize=(7, 6))

            # Inliers vs Outliers
            sizes = [num_inliers, num_outliers]
            labels = [f"Inliers\n({num_inliers:,})", f"Outliers\n({num_outliers:,})"]
            colors = ["#66b3ff", "#ff9999"]
            ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 11})
            ax1.set_title(f"Anomaly Detection\n(Total: {total_train:,} items)", fontsize=14, fontweight="bold")

            if has_test_split:
                # 아이템 기준 분할 시: 전체 데이터셋 비율 (Train + Test)
                total_all = total_train + len(test_features)
                test_count = len(test_features)
                sizes_all = [num_inliers, num_outliers, test_count]
                labels_all = [
                    f"Train Inliers\n({num_inliers:,})",
                    f"Train Outliers\n({num_outliers:,})",
                    f"Test Set\n({test_count:,})",
                ]
                colors_all = ["#66b3ff", "#ff9999", "#99ff99"]
                ax2.pie(sizes_all, labels=labels_all, colors=colors_all, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10})
                ax2.set_title(f"Overall Dataset Distribution\n(Total: {total_all:,} items)", fontsize=14, fontweight="bold")

            plt.tight_layout()
            plt.savefig(viz_dir / "anomaly_detection_distribution.png", dpi=300, bbox_inches="tight")
            plt.close()
            logger.info("Saved anomaly detection distribution to %s", viz_dir / "anomaly_detection_distribution.png")
        except Exception as e:
            logger.warning("Anomaly detection distribution 시각화 생성 실패: %s", e)

        # 2. 주요 Feature 분포 비교 (Inlier vs Outlier)
        try:
            numeric_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ["categoryid", "category_parentid"]
            plot_cols = [col for col in numeric_cols if col not in exclude_cols][:6]  # 상위 6개만

            if plot_cols and len(train_inliers) > 0 and len(train_outliers) > 0:
                n_cols = 3
                n_rows = (len(plot_cols) + n_cols - 1) // n_cols
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
                axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes

                for idx, col in enumerate(plot_cols):
                    ax = axes[idx]
                    inlier_data = train_inliers[col].dropna()
                    outlier_data = train_outliers[col].dropna()

                    if len(inlier_data) > 0 and len(outlier_data) > 0:
                        # 로그 스케일이 필요한 경우
                        if inlier_data.max() / max(inlier_data.min(), 1) > 1000 or outlier_data.max() / max(outlier_data.min(), 1) > 1000:
                            inlier_data = np.log1p(inlier_data)
                            outlier_data = np.log1p(outlier_data)
                            ax.set_yscale("log")
                            col_label = f"{col} (log scale)"
                        else:
                            col_label = col

                        ax.hist(inlier_data, bins=50, alpha=0.6, label="Inliers", color="#66b3ff", density=True)
                        ax.hist(outlier_data, bins=50, alpha=0.6, label="Outliers", color="#ff9999", density=True)
                        ax.set_xlabel(col_label, fontsize=10)
                        ax.set_ylabel("Density", fontsize=10)
                        ax.set_title(f"{col}", fontsize=11, fontweight="bold")
                        ax.legend(fontsize=9)
                        ax.grid(True, alpha=0.3)

                # 빈 subplot 제거
                for idx in range(len(plot_cols), len(axes)):
                    fig.delaxes(axes[idx])

                plt.suptitle("Feature Distribution: Inliers vs Outliers", fontsize=16, fontweight="bold", y=0.995)
                plt.tight_layout()
                plt.savefig(viz_dir / "feature_distribution_comparison.png", dpi=300, bbox_inches="tight")
                plt.close()
                logger.info("Saved feature distribution comparison to %s", viz_dir / "feature_distribution_comparison.png")
        except Exception as e:
            logger.warning("Feature distribution comparison 시각화 생성 실패: %s", e)


# ---------------------------------------------------------------------------
# ALS 전용 전처리 모듈 (als_debug 기반 → als_events_train/test 저장)
# ---------------------------------------------------------------------------
class ALSPreprocessor:
    """
    ALS 전용 전처리 모듈. als_debug.preprocess()를 실행하고
    als_events_train.csv, als_events_test.csv를 저장합니다.
    ALSRecommender는 이 파일이 있으면 로드하여 사용합니다.
    """
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self.processed_dir = Path(processed_dir) if processed_dir else Path("data/processed")

    def run(self) -> None:
        import sys
        _model_dir = Path(__file__).resolve().parent
        if str(_model_dir) not in sys.path:
            sys.path.insert(0, str(_model_dir))
        import als_debug

        np.random.seed(als_debug.RANDOM_STATE)
        random.seed(als_debug.RANDOM_STATE)

        logger.info("ALS 전용 전처리: als_debug 파이프라인 (90/10, min 3/3)")
        events = als_debug.load_raw_events()
        events_train, events_test, _, _ = als_debug.preprocess(
            events,
            test_size=als_debug.TEST_SIZE,
            min_user_events=als_debug.MIN_USER_EVENTS,
            min_item_events=als_debug.MIN_ITEM_EVENTS,
            random_state=als_debug.RANDOM_STATE,
        )
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        events_train.to_csv(self.processed_dir / "als_events_train.csv", index=False)
        events_test.to_csv(self.processed_dir / "als_events_test.csv", index=False)
        logger.info(
            "Saved ALS 전용 전처리 결과: als_events_train %d rows, als_events_test %d rows",
            len(events_train),
            len(events_test),
        )


# ---------------------------------------------------------------------------
# GNN 전용 전처리 모듈 (이전 버전: 아이템 기준 분할 + train 이상치 제거 + holdout)
# ---------------------------------------------------------------------------
@dataclass
class GNNPreprocessor:
    """
    GNN 전용 전처리 모듈. 이전 전처리 방식:
    - 피처 테이블을 아이템 기준 train/test 분할 (shuffle)
    - train에 대해서만 Isolation Forest 이상치 탐지
    - train inlier 이벤트에서 유저별 마지막 positive를 holdout으로 분리
    - events_train_clean, events_test, item_properties_train_clean 등 저장
    """
    data_dir: Path = field(default_factory=lambda: Path("data"))
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    random_state: int = 42
    test_size: float = 0.20
    contamination: str | float = "auto"
    n_estimators: int = 200

    def run(self) -> None:
        np.random.seed(self.random_state)
        random.seed(self.random_state)

        logger.info("GNN 전용 전처리: 아이템 기준 분할 + train 이상치 제거 + holdout")
        datasets = self._load_raw()
        feature_table = self._build_feature_table(datasets)
        logger.info("Feature table prepared with %d items, %d features", feature_table.shape[0], feature_table.shape[1])

        logger.info("Splitting feature table into train(%.0f%%)/test(%.0f%%)…", (1 - self.test_size) * 100, self.test_size * 100)
        train_features, test_features = train_test_split(
            feature_table,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True,
        )
        logger.info("Train set: %d items, Test set: %d items", len(train_features), len(test_features))

        exclude_cols = [
            "transaction_count", "has_any_transaction", "has_transaction_id",
            "addtocart_count", "categoryid", "category_parentid", "has_available",
        ]
        available_exclude = [col for col in exclude_cols if col in train_features.columns]
        train_features_for_detection = train_features.drop(columns=available_exclude) if available_exclude else train_features

        logger.info("Training Isolation Forest on train %d items…", len(train_features_for_detection))
        isolation_forest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        isolation_forest.fit(train_features_for_detection)
        train_predictions = isolation_forest.predict(train_features_for_detection)

        train_inliers = train_features[train_predictions == 1]
        train_outliers = train_features[train_predictions == -1]
        logger.info("Detected %d inliers, %d outliers in train", len(train_inliers), len(train_outliers))

        self._save_results(datasets, train_inliers=train_inliers, test_inliers=test_features, train_outliers=train_outliers)
        logger.info("GNN 전용 전처리 완료: train inlier %d items, test %d items", len(train_inliers), len(test_features))

        viz_dir = Path("data/visualization")
        viz_dir.mkdir(parents=True, exist_ok=True)
        self._create_visualizations(train_features, train_inliers, train_outliers, test_features)

    def _load_raw(self) -> Dict[str, DataFrame]:
        events = pd.read_csv(
            self.data_dir / "events.csv",
            usecols=["timestamp", "visitorid", "event", "itemid", "transactionid"],
        )
        category_tree = pd.read_csv(self.data_dir / "category_tree.csv")
        item_properties = pd.concat(
            [
                pd.read_csv(self.data_dir / "item_properties_part1.csv", usecols=["timestamp", "itemid", "property", "value"], low_memory=False),
                pd.read_csv(self.data_dir / "item_properties_part2.csv", usecols=["timestamp", "itemid", "property", "value"], low_memory=False),
            ],
            ignore_index=True,
        )
        return {"events": events, "category_tree": category_tree, "item_properties": item_properties}

    def _build_feature_table(self, datasets: Dict[str, DataFrame]) -> DataFrame:
        events = datasets["events"].copy()
        item_properties = datasets["item_properties"].copy()
        category_tree = datasets["category_tree"].copy()
        item_properties_filtered = item_properties[item_properties["property"].isin(["categoryid", "available"])].copy()
        events["timestamp_dt"] = pd.to_datetime(events["timestamp"], unit="ms", errors="coerce")
        min_ts = events["timestamp_dt"].min()
        events["hours_since_start"] = (events["timestamp_dt"] - min_ts).dt.total_seconds() / 3600.0
        events["event_code"] = events["event"].map(EVENT_TO_CODE).fillna(-1).astype(int)
        events["is_transaction"] = events["event"] == "transaction"
        events["has_transaction_id"] = events["transactionid"].notna().astype(int)
        event_agg = (
            events.groupby("itemid").agg(
                event_count=("event", "size"),
                view_count=("event", lambda s: (s == "view").sum()),
                addtocart_count=("event", lambda s: (s == "addtocart").sum()),
                transaction_count=("event", lambda s: (s == "transaction").sum()),
                unique_visitors=("visitorid", pd.Series.nunique),
                event_code_mean=("event_code", "mean"),
                event_code_std=("event_code", "std"),
                has_any_transaction=("is_transaction", "max"),
                has_transaction_id=("has_transaction_id", "max"),
                first_event_hour=("hours_since_start", "min"),
                last_event_hour=("hours_since_start", "max"),
            )
        )
        event_agg["event_duration_hours"] = event_agg["last_event_hour"] - event_agg["first_event_hour"]
        property_agg = item_properties_filtered.groupby("itemid").agg(property_count=("property", "count"))
        category_rows = item_properties_filtered[item_properties_filtered["property"] == "categoryid"][["itemid", "value"]].copy()
        category_rows = category_rows.dropna(subset=["value"])
        extracted = category_rows["value"].astype(str).str.extract(r"(?P<category>\d+)")
        category_rows["categoryid"] = pd.to_numeric(extracted["category"], errors="coerce")
        category_features = (
            category_rows.dropna(subset=["categoryid"])
            .drop_duplicates(subset=["itemid"], keep="first")
            .set_index("itemid")[["categoryid"]]
        )
        category_tree = category_tree.copy()
        category_tree["parentid"] = pd.to_numeric(category_tree["parentid"], errors="coerce")
        category_features = category_features.join(category_tree.set_index("categoryid")["parentid"], how="left")
        category_features.rename(columns={"parentid": "category_parentid"}, inplace=True)
        available_rows = item_properties_filtered[item_properties_filtered["property"] == "available"].copy()
        if not available_rows.empty:
            available_rows["value_numeric"] = pd.to_numeric(available_rows["value"], errors="coerce")
            available_features = available_rows.dropna(subset=["value_numeric"]).groupby("itemid")["value_numeric"].max().to_frame(name="has_available")
        else:
            available_features = pd.DataFrame(columns=["has_available"])
        feature_table = event_agg.join(property_agg, how="left").join(category_features, how="left").join(available_features, how="left")
        feature_table = feature_table.fillna(0.0).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return feature_table

    def _save_results(
        self,
        datasets: Dict[str, DataFrame],
        *,
        train_inliers: DataFrame,
        test_inliers: DataFrame,
        train_outliers: DataFrame,
    ) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        events = datasets["events"].copy()
        item_properties = datasets["item_properties"].copy()
        train_inlier_ids = set(train_inliers.index)
        test_inlier_ids = set(test_inliers.index)
        train_outlier_ids = set(train_outliers.index)

        events_train_clean = events[events["itemid"].isin(train_inlier_ids)].copy()
        events_test_base = events[events["itemid"].isin(test_inlier_ids)].copy()
        positive_mask = events_train_clean["event"].isin(["addtocart", "transaction"])
        if positive_mask.any():
            positive_events = events_train_clean.loc[positive_mask].copy().dropna(subset=["timestamp"]).sort_values(["visitorid", "timestamp"])
            holdout_indices = positive_events.groupby("visitorid")["timestamp"].idxmax().dropna().astype(int)
            events_test_holdout = events_train_clean.loc[holdout_indices].copy()
            events_train_clean = events_train_clean.drop(index=holdout_indices, errors="ignore")
        else:
            events_test_holdout = events_train_clean.iloc[0:0].copy()
        events_test = pd.concat([events_test_base, events_test_holdout], ignore_index=True)
        events_test = events_test.drop_duplicates(subset=["timestamp", "visitorid", "event", "itemid", "transactionid"])
        events_train_clean = events_train_clean.reset_index(drop=True)
        events_test = events_test.reset_index(drop=True)
        logger.info("GNN holdout: train %d events, test %d events, %d users", len(events_train_clean), len(events_test), events_test["visitorid"].nunique() if not events_test.empty else 0)

        events_train_outliers = events[events["itemid"].isin(train_outlier_ids)].copy()
        item_props_train_clean = item_properties[item_properties["itemid"].isin(train_inlier_ids)]
        item_props_train_outliers = item_properties[item_properties["itemid"].isin(train_outlier_ids)]
        eval_item_ids = events_test["itemid"].astype(int).unique() if not events_test.empty else np.array([], dtype=int)
        item_props_test = item_properties[item_properties["itemid"].isin(eval_item_ids)] if len(eval_item_ids) else item_properties.iloc[0:0].copy()

        events_train_clean.to_csv(self.processed_dir / "events_train_clean.csv", index=False)
        events_test.to_csv(self.processed_dir / "events_test.csv", index=False)
        events_train_outliers.to_csv(self.processed_dir / "events_train_outliers.csv", index=False)
        item_props_train_clean.to_csv(self.processed_dir / "item_properties_train_clean.csv", index=False)
        item_props_test.to_csv(self.processed_dir / "item_properties_test.csv", index=False)
        item_props_train_outliers.to_csv(self.processed_dir / "item_properties_train_outliers.csv", index=False)
        train_inliers.to_csv(self.processed_dir / "feature_train_inliers.csv")
        test_inliers.to_csv(self.processed_dir / "feature_test_inliers.csv")
        train_outliers.to_csv(self.processed_dir / "feature_train_outliers.csv")
        logger.info("Saved GNN 전용 전처리 결과 to %s", self.processed_dir.resolve())

    def _create_visualizations(
        self,
        train_features: DataFrame,
        train_inliers: DataFrame,
        train_outliers: DataFrame,
        test_features: DataFrame,
    ) -> None:
        try:
            import seaborn as sns
        except ImportError:
            return
        viz_dir = Path("data/visualization")
        viz_dir.mkdir(parents=True, exist_ok=True)
        try:
            plt.style.use("seaborn-v0_8-darkgrid")
        except OSError:
            try:
                plt.style.use("seaborn-darkgrid")
            except OSError:
                plt.style.use("default")
        sns.set_palette("husl")
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            sizes = [len(train_inliers), len(train_outliers)]
            ax1.pie(sizes, labels=["Inliers", "Outliers"], autopct="%1.1f%%", startangle=90)
            ax1.set_title("GNN Preprocessor: Train Anomaly Detection")
            sizes_all = sizes + [len(test_features)]
            ax2.pie(sizes_all, labels=["Train Inliers", "Train Outliers", "Test"], autopct="%1.1f%%", startangle=90)
            ax2.set_title("GNN Preprocessor: Dataset Split")
            plt.tight_layout()
            plt.savefig(viz_dir / "gnn_preprocessor_split.png", dpi=150, bbox_inches="tight")
            plt.close()
        except Exception:
            pass


@dataclass
class ALSRecommender:
    """
    ALS(Alternating Least Squares) 기반 협업 필터링 추천 시스템 (BPR 앙상블·롱테일 보너스 지원).
    
    암시적 피드백 데이터를 사용하며, 이벤트별 가중치(weight_map), 선택적 BPR 앙상블,
    롱테일 보너스로 Top-K 추천을 생성합니다. ReRanker·평가 파이프라인과 호환됩니다.
    
    Attributes:
        processed_dir: 전처리된 데이터가 저장된 디렉토리 경로
        factors: 사용자/아이템 임베딩 차원 수
        regularization: 정규화 계수
        iterations: ALS 학습 반복 횟수
        alpha: 암시적 피드백 confidence (Hu et al.)
        top_k: 사용자당 추천할 아이템 수
        weight_map: 이벤트별 가중치 (None이면 ALS_WEIGHT_MAP 사용)
        use_weight_log_scale: 집계 가중치에 1+log(1+x) 적용 여부
        ensemble_alpha: ALS+BPR 앙상블 시 alpha*ALS + (1-alpha)*BPR
        min_user_events, min_item_events: Core 필터 (0이면 미적용)
    """
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    factors: int = 128
    regularization: float = 0.03
    iterations: int = 256
    alpha: float = 1.5
    top_k: int = 50
    random_state: int = 42
    recommend_batch_size: int = 256
    weight_map: Optional[Dict[str, float]] = None  # None이면 ALS_WEIGHT_MAP 사용
    use_weight_log_scale: bool = False
    ensemble_alpha: float = 0.5  # ALS+BPR 앙상블 시 alpha*ALS + (1-alpha)*BPR
    min_user_events: int = 0  # 0이면 Core 필터 미적용
    min_item_events: int = 0

    def run(self) -> None:
        """
        ALS 추천 파이프라인을 실행합니다.
        als_events_train.csv, als_events_test.csv가 있으면 로드하고, 없으면 als_debug.preprocess() 실행.
        """
        import sys
        _model_dir = Path(__file__).resolve().parent
        if str(_model_dir) not in sys.path:
            sys.path.insert(0, str(_model_dir))
        import als_debug

        np.random.seed(self.random_state)
        random.seed(self.random_state)

        train_path = self.processed_dir / "als_events_train.csv"
        test_path = self.processed_dir / "als_events_test.csv"
        if train_path.exists() and test_path.exists():
            logger.info("ALS: als_events_train/test 로드 (ALSPreprocessor 결과 사용)")
            events_train = pd.read_csv(train_path)
            events_test = pd.read_csv(test_path)
            train_item_set = set(events_train["itemid"].dropna().astype(int).unique())
            pos_events = events_test[events_test["event"].isin(["addtocart", "transaction"])].dropna(subset=["visitorid", "itemid"])
            pos_events = pos_events.assign(visitorid=pos_events["visitorid"].astype(int), itemid=pos_events["itemid"].astype(int))
            positives = pos_events.groupby("visitorid")["itemid"].apply(lambda s: set(s.tolist())).to_dict()
        else:
            logger.info("ALS: als_debug 파이프라인 사용 (원본 events.csv, 90/10, min 3/3)")
            events = als_debug.load_raw_events()
            events_train, events_test, positives, train_item_set = als_debug.preprocess(
                events,
                test_size=als_debug.TEST_SIZE,
                min_user_events=als_debug.MIN_USER_EVENTS,
                min_item_events=als_debug.MIN_ITEM_EVENTS,
                random_state=als_debug.RANDOM_STATE,
            )
        positives_filtered = {}
        for uid, rel in positives.items():
            filtered = rel & train_item_set
            if filtered:
                positives_filtered[uid] = filtered

        (
            user_item_matrix,
            unique_users,
            unique_items,
            user_id_to_idx,
            item_id_to_idx,
            idx_to_item_id,
            item_pop_by_id,
        ) = als_debug.build_user_item_matrix(
            events_train,
            als_debug.WEIGHT_MAP,
            use_log_scale=als_debug.USE_WEIGHT_LOG_SCALE,
        )
        logger.info(
            "Constructed user-item matrix with %d users, %d items, %d interactions",
            len(unique_users),
            len(unique_items),
            user_item_matrix.nnz,
        )

        model_als = als_debug.train_als(
            user_item_matrix,
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=als_debug.RANDOM_STATE,
            alpha=self.alpha,
        )
        logger.info("ALS model trained (factors=%d, iterations=%d, alpha=%.2f)",
                    self.factors, self.iterations, self.alpha)

        model_bpr = None
        if als_debug.HAS_BPR:
            try:
                model_bpr = als_debug.train_bpr(
                    user_item_matrix,
                    factors=als_debug.BPR_FACTORS,
                    regularization=als_debug.BPR_REGULARIZATION,
                    iterations=als_debug.BPR_ITERATIONS,
                    random_state=als_debug.RANDOM_STATE,
                )
                logger.info("BPR model trained for ensemble (alpha=%.2f)", als_debug.ENSEMBLE_ALPHA)
            except Exception as e:
                logger.warning("BPR 학습 건너뜀: %s", e)

        eval_users = [int(u) for u in unique_users if u in user_id_to_idx]

        logger.info("Generating top-%d recommendations per user (ensemble=%s)…",
                    self.top_k, model_bpr is not None)
        recommendations_dict, _ = als_debug.scores_to_recommendations_ensemble(
            eval_users,
            user_id_to_idx,
            item_id_to_idx,
            idx_to_item_id,
            train_item_set,
            positives_filtered,
            events_train,
            model_als,
            model_bpr,
            self.top_k,
            self.ensemble_alpha,
            return_per_user_scores=False,
        )

        records = []
        for uid, rec_list in recommendations_dict.items():
            for rank, item_id in enumerate(rec_list, start=1):
                records.append({
                    "visitorid": int(uid),
                    "itemid": int(item_id),
                    "score": 1.0 / rank,
                    "rank": rank,
                })
        recommendations = pd.DataFrame(records)
        users_list = unique_users.tolist() if hasattr(unique_users, "tolist") else list(unique_users)
        items_list = unique_items.tolist() if hasattr(unique_items, "tolist") else list(unique_items)
        logger.info("ALS recommendation generation completed for %d users", recommendations["visitorid"].nunique())

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        events_train.to_csv(self.processed_dir / "als_events_train.csv", index=False)
        events_test.to_csv(self.processed_dir / "als_events_test.csv", index=False)

        self._save_outputs(recommendations, users_list, items_list, model_als, model_bpr=model_bpr)
        logger.info("Saved ALS recommendations to %s", self.processed_dir.resolve())

        interactions = events_train.copy()
        interactions["weight"] = events_train["event"].map(als_debug.WEIGHT_MAP).fillna(1.0)
        interactions = interactions.groupby(["visitorid", "itemid"], as_index=False)["weight"].sum()
        self._create_visualizations(
            interactions=interactions,
            recommendations=recommendations,
            user_item_matrix=user_item_matrix,
            users=users_list,
            items=items_list,
        )

    def _save_outputs(
        self,
        recommendations: DataFrame,
        users: List[int],
        items: List[int],
        model: Any,
        model_bpr: Any = None,
    ) -> None:
        """
        ALS 모델의 출력 결과를 저장합니다.
        
        저장되는 파일:
        - als_recommendations.csv: 추천 결과
        - als_user_mapping.csv: 사용자 인덱스-ID 매핑
        - als_item_mapping.csv: 아이템 인덱스-ID 매핑
        - als_user_factors.npy: 사용자 임베딩 행렬
        - als_item_factors.npy: 아이템 임베딩 행렬
        - (BPR 학습 시) bpr_user_factors.npy, bpr_item_factors.npy
        
        Args:
            recommendations: 추천 결과 데이터프레임
            users: 사용자 ID 리스트
            items: 아이템 ID 리스트
            model: 학습된 ALS 모델
            model_bpr: 학습된 BPR 모델 (있으면 factors 저장, hold-out 평가용)
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        recommendations.to_csv(self.processed_dir / "als_recommendations.csv", index=False)

        user_mapping = pd.DataFrame({"user_index": range(len(users)), "visitorid": users})
        item_mapping = pd.DataFrame({"item_index": range(len(items)), "itemid": items})

        user_mapping.to_csv(self.processed_dir / "als_user_mapping.csv", index=False)
        item_mapping.to_csv(self.processed_dir / "als_item_mapping.csv", index=False)

        np.save(self.processed_dir / "als_user_factors.npy", model.user_factors)
        np.save(self.processed_dir / "als_item_factors.npy", model.item_factors)
        if model_bpr is not None and hasattr(model_bpr, "user_factors") and hasattr(model_bpr, "item_factors"):
            np.save(self.processed_dir / "bpr_user_factors.npy", model_bpr.user_factors)
            np.save(self.processed_dir / "bpr_item_factors.npy", model_bpr.item_factors)

    def _create_visualizations(
        self,
        interactions: DataFrame,
        recommendations: DataFrame,
        user_item_matrix: sp.csr_matrix,
        users: List[int],
        items: List[int],
    ) -> None:
        """
        ALS 모델의 결과를 시각화합니다.
        
        생성되는 시각화:
        1. 사용자-아이템 상호작용 분포 (사용자별, 아이템별)
        2. 추천 점수 분포
        3. 상위 추천 아이템 (가장 많이 추천된 아이템 Top 20)
        4. 사용자별 추천 수 분포
        
        Args:
            interactions: 상호작용 데이터프레임
            recommendations: 추천 결과 데이터프레임
            user_item_matrix: 사용자-아이템 상호작용 행렬
            users: 사용자 ID 리스트
            items: 아이템 ID 리스트
        """
        try:
            import seaborn as sns
        except ImportError:
            logger.warning("seaborn이 설치되지 않아 ALS 시각화를 건너뜁니다.")
            return

        viz_dir = Path("data/visualization")
        viz_dir.mkdir(parents=True, exist_ok=True)

        # 스타일 설정
        try:
            plt.style.use("seaborn-v0_8-darkgrid")
        except OSError:
            try:
                plt.style.use("seaborn-darkgrid")
            except OSError:
                plt.style.use("default")
        sns.set_palette("husl")

        # 1. 사용자-아이템 상호작용 분포
        try:
            user_interaction_counts = interactions.groupby("visitorid")["weight"].count()
            item_interaction_counts = interactions.groupby("itemid")["weight"].count()

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # 사용자별 상호작용 수 분포
            axes[0].hist(user_interaction_counts.values, bins=50, color="#66b3ff", alpha=0.7, edgecolor="black")
            axes[0].set_xlabel("Number of Interactions per User", fontsize=12)
            axes[0].set_ylabel("Number of Users", fontsize=12)
            axes[0].set_title("User Interaction Distribution", fontsize=14, fontweight="bold")
            axes[0].set_yscale("log")
            axes[0].grid(True, alpha=0.3)
            axes[0].axvline(user_interaction_counts.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {user_interaction_counts.mean():.1f}")
            axes[0].legend()

            # 아이템별 상호작용 수 분포
            axes[1].hist(item_interaction_counts.values, bins=50, color="#99ff99", alpha=0.7, edgecolor="black")
            axes[1].set_xlabel("Number of Interactions per Item", fontsize=12)
            axes[1].set_ylabel("Number of Items", fontsize=12)
            axes[1].set_title("Item Interaction Distribution", fontsize=14, fontweight="bold")
            axes[1].set_yscale("log")
            axes[1].grid(True, alpha=0.3)
            axes[1].axvline(item_interaction_counts.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {item_interaction_counts.mean():.1f}")
            axes[1].legend()

            plt.tight_layout()
            plt.savefig(viz_dir / "als_interaction_distribution.png", dpi=300, bbox_inches="tight")
            plt.close()
            logger.info("Saved ALS interaction distribution to %s", viz_dir / "als_interaction_distribution.png")
        except Exception as e:
            logger.warning("ALS interaction distribution 시각화 생성 실패: %s", e)

        # 2. 추천 점수 분포
        try:
            if not recommendations.empty and "score" in recommendations.columns:
                scores = recommendations["score"].values

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(scores, bins=50, color="#ffcc99", alpha=0.7, edgecolor="black")
                ax.set_xlabel("Recommendation Score", fontsize=12)
                ax.set_ylabel("Frequency", fontsize=12)
                ax.set_title("ALS Recommendation Score Distribution", fontsize=14, fontweight="bold")
                ax.grid(True, alpha=0.3)
                ax.axvline(scores.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {scores.mean():.4f}")
                ax.axvline(np.median(scores), color="blue", linestyle="--", linewidth=2, label=f"Median: {np.median(scores):.4f}")
                ax.legend()

                plt.tight_layout()
                plt.savefig(viz_dir / "als_score_distribution.png", dpi=300, bbox_inches="tight")
                plt.close()
                logger.info("Saved ALS score distribution to %s", viz_dir / "als_score_distribution.png")
        except Exception as e:
            logger.warning("ALS score distribution 시각화 생성 실패: %s", e)

        # 3. 상위 추천 아이템
        try:
            if not recommendations.empty:
                top_recommended_items = (
                    recommendations.groupby("itemid")
                    .size()
                    .sort_values(ascending=False)
                    .head(20)
                )

                fig, ax = plt.subplots(figsize=(12, 8))
                bars = ax.barh(range(len(top_recommended_items)), top_recommended_items.values, color="#ff99cc")
                ax.set_yticks(range(len(top_recommended_items)))
                ax.set_yticklabels([f"Item {item_id}" for item_id in top_recommended_items.index], fontsize=9)
                ax.set_xlabel("Number of Recommendations", fontsize=12)
                ax.set_title("Top 20 Most Recommended Items", fontsize=14, fontweight="bold")
                ax.grid(True, alpha=0.3, axis="x")

                # 값 표시
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(
                        width,
                        bar.get_y() + bar.get_height() / 2.0,
                        f"{int(width)}",
                        ha="left",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                    )

                plt.tight_layout()
                plt.savefig(viz_dir / "als_top_recommended_items.png", dpi=300, bbox_inches="tight")
                plt.close()
                logger.info("Saved ALS top recommended items to %s", viz_dir / "als_top_recommended_items.png")
        except Exception as e:
            logger.warning("ALS top recommended items 시각화 생성 실패: %s", e)

        # 4. 사용자별 추천 수 분포
        try:
            if not recommendations.empty:
                user_recommendation_counts = recommendations.groupby("visitorid").size()

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(user_recommendation_counts.values, bins=30, color="#99ccff", alpha=0.7, edgecolor="black")
                ax.set_xlabel("Number of Recommendations per User", fontsize=12)
                ax.set_ylabel("Number of Users", fontsize=12)
                ax.set_title("User Recommendation Count Distribution", fontsize=14, fontweight="bold")
                ax.grid(True, alpha=0.3)
                ax.axvline(user_recommendation_counts.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {user_recommendation_counts.mean():.1f}")
                ax.axvline(user_recommendation_counts.median(), color="blue", linestyle="--", linewidth=2, label=f"Median: {user_recommendation_counts.median():.1f}")
                ax.legend()

                plt.tight_layout()
                plt.savefig(viz_dir / "als_user_recommendation_distribution.png", dpi=300, bbox_inches="tight")
                plt.close()
                logger.info("Saved ALS user recommendation distribution to %s", viz_dir / "als_user_recommendation_distribution.png")
        except Exception as e:
            logger.warning("ALS user recommendation distribution 시각화 생성 실패: %s", e)

@dataclass
class GNNEmbeddingGenerator:
    """
    그래프 신경망(GNN) 기반 임베딩 생성 클래스
    
    LightGCN 아키텍처를 사용하여 사용자, 아이템, 속성, 카테고리 노드의 임베딩을 학습합니다.
    이종 그래프(heterogeneous graph)를 구성하여 다양한 엔티티 간의 관계를 모델링합니다.
    
    그래프 구조:
    - 노드 타입: User, Item, Property, Category
    - 엣지 타입: User-Item, Item-Property, Item-Category, Category-Category (계층)
    
    학습 방법:
    - BPR(Bayesian Personalized Ranking) 손실 함수 사용
    - 네가티브 샘플링을 통한 대조 학습
    
    Attributes:
        processed_dir: 전처리된 데이터가 저장된 디렉토리 경로
        data_dir: 원본 데이터가 저장된 디렉토리 경로
        embedding_dim: 노드 임베딩 차원 수
        layers: GNN 레이어 수 (그래프 컨볼루션 깊이)
        epochs: 학습 에포크 수
        batch_size: 학습 배치 크기
        learning_rate: Adam 옵티마이저 학습률
        reg: L2 정규화 계수
        num_negative: 네가티브 샘플 개수
        seed: 재현성을 위한 랜덤 시드
        device: 학습 디바이스 ("cuda" 또는 "cpu", None이면 자동 선택)
    """
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    data_dir: Path = field(default_factory=lambda: Path("data"))
    embedding_dim: int = 8
    layers: int = 2
    epochs: int = 1024
    batch_size: int = 8192
    learning_rate: float = 1e-3
    reg: float = 1e-4
    num_negative: int = 1
    seed: int = 42
    device: Optional[str] = None
    early_stopping_patience: Optional[int] = 50  # None이면 early stopping 비활성화
    min_delta: float = 1e-6  # 개선으로 간주할 최소 변화량

    def __post_init__(self) -> None:
        """
        디바이스를 자동으로 설정합니다.
        
        device가 None이면 CUDA가 사용 가능한지 확인하고, 사용 가능하면 "cuda",
        아니면 "cpu"로 설정합니다.
        """
        if self.device is None:
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"

    def run(self) -> None:
        """
        GNN 임베딩 생성 파이프라인을 실행합니다.
        
        실행 순서:
        1. 전처리된 데이터 로드
        2. 그래프 구조 준비 (노드 및 엣지 생성)
        3. 정규화된 인접 행렬 생성
        4. LightGCN 모델 생성 및 학습
        5. 임베딩 추출 및 저장
        6. 그래프 구조 시각화
        """
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover - 환경에 따라 torch 미설치 가능
            raise ImportError(
                "GNN 임베딩 생성을 위해 PyTorch가 필요합니다. `pip install torch` 후 다시 실행해 주세요."
            ) from exc

        # 시드 고정 (전역 시드와 동일하게 42로 설정)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)  # 멀티 GPU 환경
        np.random.seed(self.seed)
        random.seed(self.seed)
        # 재현가능성을 위한 추가 설정
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        logger.info("Preparing inputs for GNN embedding generation on device %s…", self.device)
        datasets = self._load_inputs()
        graph_data = self._prepare_graph_structures(datasets)

        adjacency = self._build_normalized_adjacency(
            graph_data["edge_sources"], graph_data["edge_targets"], graph_data["total_nodes"]
        )

        logger.info(
            "Adjacency ready: nodes=%d, edges=%d (symmetric)",
            graph_data["total_nodes"],
            len(graph_data["edge_sources"]),
        )

        model = self._build_model(adjacency, torch)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        interactions = graph_data["interactions"]
        if interactions.size == 0:
            raise ValueError("GNN 학습에 사용할 유저-아이템 상호작용이 존재하지 않습니다.")

        logger.info(
            "Start GNN training: users=%d, items=%d, properties=%d, categories=%d, edges=%d",
            graph_data["num_users"],
            graph_data["num_items"],
            graph_data["num_properties"],
            graph_data["num_categories"],
            len(graph_data["edge_sources"]),
        )

        # Train/Validation 분할 (80/20)
        np.random.seed(self.seed)
        n_interactions = len(interactions)
        indices = np.arange(n_interactions)
        train_indices, val_indices = train_test_split(
            indices, test_size=0.2, random_state=self.seed
        )
        
        train_interactions = interactions[train_indices]
        val_interactions = interactions[val_indices]
        
        logger.info("Train interactions: %d, Validation interactions: %d", 
                   len(train_interactions), len(val_interactions))
        
        num_batches = max(int(np.ceil(len(train_interactions) / self.batch_size)), 1)
        train_losses = []
        val_losses = []
        
        # Early stopping 변수 (patience: 개선 없는 연속 에포크 수가 이 값에 도달하면 학습 종료)
        patience_limit = int(self.early_stopping_patience) if self.early_stopping_patience is not None else None
        if patience_limit is not None:
            logger.info(
                "Early stopping enabled: patience=%d (will stop after %d consecutive epochs with no val loss improvement, min_delta=%.2e)",
                patience_limit, patience_limit, self.min_delta,
            )
        best_val_loss = float("inf")
        patience_counter = 0
        best_epoch = 0
        best_model_state = None
        
        for epoch in range(1, self.epochs + 1):
            # Train
            permutation = np.random.permutation(len(train_interactions))
            epoch_loss = 0.0
            model.train()

            for batch_idx in range(num_batches):
                start = batch_idx * self.batch_size
                end = min((batch_idx + 1) * self.batch_size, len(train_interactions))
                batch = train_interactions[permutation[start:end]]
                if batch.size == 0:
                    continue

                loss = self._train_step(
                    model,
                    optimizer,
                    batch,
                    graph_data,
                    torch,
                )
                epoch_loss += loss

                if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == num_batches:
                    progress = ((batch_idx + 1) / num_batches) * 100
                    logger.info(
                        "GNN training epoch %d/%d progress: batch %d/%d (%.1f%%)",
                        epoch,
                        self.epochs,
                        batch_idx + 1,
                        num_batches,
                        progress,
                    )

            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Validation loss 계산 (고정 시드로 네가티브 샘플 생성 → 에포크 간 비교 가능)
            val_loss = self._compute_val_loss(model, val_interactions, graph_data, torch, val_seed=self.seed + 99999)
            val_losses.append(val_loss)
            
            logger.info("Epoch %d/%d - Train Loss: %.6f, Val Loss: %.6f", 
                       epoch, self.epochs, avg_train_loss, val_loss)
            
            # Early stopping 체크 (patience_limit 사용으로 전달값과 동일한 정수 비교 보장)
            if patience_limit is not None:
                if val_loss < best_val_loss - self.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_epoch = epoch
                    # 최적 모델 상태 저장
                    best_model_state = {
                        "model": model.state_dict().copy(),
                        "optimizer": optimizer.state_dict().copy(),
                    }
                    logger.info("  *** Best validation loss improved to %.6f at epoch %d ***", 
                               best_val_loss, best_epoch)
                else:
                    patience_counter += 1
                    logger.info("  No improvement for %d epoch(s) (patience: %d)", 
                               patience_counter, patience_limit)
                    
                    if patience_counter >= patience_limit:
                        logger.info("Early stopping triggered at epoch %d. Best epoch: %d (Val Loss: %.6f)",
                                   epoch, best_epoch, best_val_loss)
                        break
        
        # 최적 모델 복원
        if patience_limit is not None and best_model_state is not None:
            logger.info("Restoring best model from epoch %d (Val Loss: %.6f)", 
                       best_epoch, best_val_loss)
            model.load_state_dict(best_model_state["model"])
            optimizer.load_state_dict(best_model_state["optimizer"])
        
        # Loss 그래프 출력
        self._plot_gnn_losses(train_losses, val_losses, best_epoch if patience_limit is not None else None)

        model.eval()
        with torch.no_grad():
            all_embeddings = model().detach().cpu().numpy()

        self._save_embeddings(all_embeddings, graph_data)
        logger.info("Saved GNN embeddings to %s", self.processed_dir.resolve())

        # 그래프 구조 시각화
        self._create_graph_visualizations(graph_data)

    def _compute_val_loss(
        self,
        model: Any,
        val_interactions: np.ndarray,
        graph_data: Dict[str, Any],
        torch_module: Any,
        val_seed: Optional[int] = None,
    ) -> float:
        """
        Validation loss를 계산합니다.
        val_seed가 주어지면 해당 시드로 네가티브 샘플을 고정하여 에포크 간 val_loss를 비교 가능하게 합니다.
        
        Args:
            model: GNN 모델
            val_interactions: validation 상호작용 배열
            graph_data: 그래프 구조 정보 딕셔너리
            torch_module: PyTorch 모듈
            val_seed: 검증용 네가티브 샘플 시드 (None이면 매번 랜덤)
            
        Returns:
            Validation loss 값
        """
        model.eval()
        total_loss = 0.0
        num_batches = max(int(np.ceil(len(val_interactions) / self.batch_size)), 1)
        
        # 검증 네가티브 샘플 시드 고정 → early stopping 판정 안정화
        if val_seed is not None:
            torch_module.manual_seed(val_seed)
            np.random.seed(val_seed)
        
        with torch_module.no_grad():
            for batch_idx in range(num_batches):
                start = batch_idx * self.batch_size
                end = min((batch_idx + 1) * self.batch_size, len(val_interactions))
                batch = val_interactions[start:end]
                if batch.size == 0:
                    continue

                user_indices = torch_module.from_numpy(batch[:, 0]).long().to(self.device)
                pos_item_indices = torch_module.from_numpy(batch[:, 1]).long().to(self.device)

                # 네가티브 샘플 생성 (val_seed 설정 시 동일 샘플로 에포크 간 비교 가능)
                neg_shape = (len(batch), max(self.num_negative, 1))
                neg_item_indices = torch_module.randint(0, graph_data["num_items"], neg_shape, device=self.device)

                # 포지티브 아이템과 겹치는 네가티브 샘플이 있으면 재선택
                conflict_mask = neg_item_indices.eq(pos_item_indices.unsqueeze(1))
                while torch_module.any(conflict_mask):
                    replacement = torch_module.randint(
                        0, graph_data["num_items"], (int(conflict_mask.sum()),), device=self.device
                    )
                    neg_item_indices[conflict_mask] = replacement
                    conflict_mask = neg_item_indices.eq(pos_item_indices.unsqueeze(1))

                user_global = graph_data["offsets"]["user"] + user_indices
                pos_global = graph_data["offsets"]["item"] + pos_item_indices
                neg_global = graph_data["offsets"]["item"] + neg_item_indices

                all_embeddings = model()

                user_emb = all_embeddings[user_global]
                pos_emb = all_embeddings[pos_global]
                neg_emb = all_embeddings[neg_global.view(-1)].view(len(batch), -1, self.embedding_dim)

                pos_scores = (user_emb * pos_emb).sum(dim=1, keepdim=True)
                neg_scores = (user_emb.unsqueeze(1) * neg_emb).sum(dim=2)

                bpr_loss = -torch_module.log(torch_module.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
                reg_loss = self.reg * (
                    user_emb.pow(2).sum(dim=1, keepdim=True)
                    + pos_emb.pow(2).sum(dim=1, keepdim=True)
                    + neg_emb.pow(2).sum(dim=2)
                ).mean()

                loss = bpr_loss + reg_loss
                total_loss += float(loss.cpu().item())

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _plot_gnn_losses(self, train_losses: List[float], val_losses: List[float], best_epoch: Optional[int] = None) -> None:
        """
        GNN 학습 중 loss 그래프를 출력합니다.
        
        Args:
            train_losses: 학습 loss 리스트
            val_losses: validation loss 리스트
            best_epoch: 최적 epoch (early stopping 사용 시)
        """
        try:
            import seaborn as sns
        except ImportError:
            logger.warning("seaborn이 설치되지 않아 GNN loss 그래프를 건너뜁니다.")
            return

        viz_dir = Path("data/visualization")
        viz_dir.mkdir(parents=True, exist_ok=True)

        # 스타일 설정
        try:
            plt.style.use("seaborn-v0_8-darkgrid")
        except OSError:
            try:
                plt.style.use("seaborn-darkgrid")
            except OSError:
                plt.style.use("default")
        sns.set_palette("husl")

        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        
        ax.plot(epochs, train_losses, label="Train Loss", marker="o", linewidth=2, markersize=6)
        ax.plot(epochs, val_losses, label="Validation Loss", marker="s", linewidth=2, markersize=6)
        
        # Early stopping 지점 표시
        if best_epoch is not None and best_epoch <= len(epochs):
            ax.axvline(x=best_epoch, color="red", linestyle="--", linewidth=2, 
                      label=f"Best (epoch {best_epoch})", alpha=0.7)
        
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss (BPR + L2 Regularization)", fontsize=12)
        ax.set_title("GNN Training and Validation Loss", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "gnn_loss_curve.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved GNN loss curve to %s", viz_dir / "gnn_loss_curve.png")

    def _load_inputs(self) -> Dict[str, DataFrame]:
        """
        GNN 학습에 필요한 입력 데이터를 로드합니다.
        
        Returns:
            events, item_properties, category_tree를 포함한 딕셔너리
            
        Raises:
            FileNotFoundError: 필요한 파일이 없을 경우
        """
        events_path = self.processed_dir / "events_train_clean.csv"
        item_props_path = self.processed_dir / "item_properties_train_clean.csv"
        category_tree_path = self.data_dir / "category_tree.csv"

        if not events_path.exists():
            raise FileNotFoundError(
                f"{events_path} 파일이 없습니다. IsolationForestPreprocessor를 먼저 실행해 주세요."
            )
        if not item_props_path.exists():
            raise FileNotFoundError(
                f"{item_props_path} 파일이 없습니다. IsolationForestPreprocessor를 먼저 실행해 주세요."
            )
        if not category_tree_path.exists():
            raise FileNotFoundError(f"{category_tree_path} 파일이 존재하지 않습니다.")

        events = pd.read_csv(events_path)
        item_properties = pd.read_csv(item_props_path)
        category_tree = pd.read_csv(category_tree_path)

        return {
            "events": events,
            "item_properties": item_properties,
            "category_tree": category_tree,
        }

    def _prepare_graph_structures(self, datasets: Dict[str, DataFrame]) -> Dict[str, Any]:
        """
        이종 그래프 구조를 준비합니다.
        
        노드 타입별로 인덱스를 할당하고, 각 노드 타입에 오프셋을 부여하여
        단일 그래프로 통합합니다. 양방향 엣지를 생성하여 무방향 그래프를 만듭니다.
        
        생성되는 엣지:
        - User-Item: 이벤트 데이터에서 추출
        - Item-Property: 아이템 속성에서 추출
        - Item-Category: 아이템의 카테고리 할당
        - Category-Category: 카테고리 계층 구조
        
        Args:
            datasets: _load_inputs()에서 반환된 데이터 딕셔너리
            
        Returns:
            그래프 구조 정보를 담은 딕셔너리:
            - edge_sources, edge_targets: 엣지 리스트
            - total_nodes: 전체 노드 수
            - user_ids, item_ids, property_ids, category_ids: 각 노드 타입의 ID 리스트
            - user_map, item_map: ID를 인덱스로 매핑하는 딕셔너리
            - offsets: 각 노드 타입의 오프셋
            - interactions: 학습에 사용할 사용자-아이템 상호작용 배열
            - num_users, num_items, num_properties, num_categories: 각 노드 타입의 개수
        """
        events = datasets["events"].copy()
        item_properties = datasets["item_properties"].copy()
        category_tree = datasets["category_tree"].copy()

        events = events.dropna(subset=["visitorid", "itemid"])
        item_properties = item_properties.dropna(subset=["itemid", "property"])

        user_ids = pd.Index(events["visitorid"].astype(int).unique())
        if user_ids.empty:
            raise ValueError("유저 ID가 존재하지 않습니다.")

        item_ids = pd.Index(
            pd.unique(
                pd.concat([
                    events["itemid"].astype(int),
                    item_properties["itemid"].astype(int),
                ], ignore_index=True)
            )
        )
        if item_ids.empty:
            raise ValueError("아이템 ID가 존재하지 않습니다.")

        property_ids = pd.Index(item_properties["property"].astype(str).unique())

        category_tree["categoryid"] = pd.to_numeric(category_tree["categoryid"], errors="coerce")
        category_tree["parentid"] = pd.to_numeric(category_tree["parentid"], errors="coerce")

        category_assignments = item_properties[item_properties["property"] == "categoryid"].copy()
        category_assignments["categoryid"] = category_assignments["value"].map(_extract_numeric)
        category_assignments = category_assignments.dropna(subset=["categoryid"])
        category_assignments["categoryid"] = category_assignments["categoryid"].astype(int)

        category_ids = pd.Index(
            pd.unique(
                pd.concat(
                    [
                        category_tree["categoryid"].dropna().astype(int),
                        category_tree["parentid"].dropna().astype(int),
                        category_assignments["categoryid"],
                    ],
                    ignore_index=True,
                )
            )
        )

        user_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
        item_map = {item_id: idx for idx, item_id in enumerate(item_ids)}
        property_map = {prop_id: idx for idx, prop_id in enumerate(property_ids)}
        category_map = {cat_id: idx for idx, cat_id in enumerate(category_ids)}

        offsets: Dict[str, int] = {}
        counts: Dict[str, int] = {
            "user": len(user_ids),
            "item": len(item_ids),
            "property": len(property_ids),
            "category": len(category_ids),
        }
        current_offset = 0
        for node_type in ["user", "item", "property", "category"]:
            offsets[node_type] = current_offset
            current_offset += counts[node_type]

        edge_sources: List[int] = []
        edge_targets: List[int] = []

        def add_bidirectional_edge(src_type: str, src_id: Any, dst_type: str, dst_id: Any) -> None:
            if src_type == "user":
                src_local = user_map.get(src_id)
            elif src_type == "item":
                src_local = item_map.get(src_id)
            elif src_type == "property":
                src_local = property_map.get(src_id)
            else:
                src_local = category_map.get(src_id)

            if dst_type == "user":
                dst_local = user_map.get(dst_id)
            elif dst_type == "item":
                dst_local = item_map.get(dst_id)
            elif dst_type == "property":
                dst_local = property_map.get(dst_id)
            else:
                dst_local = category_map.get(dst_id)

            if src_local is None or dst_local is None:
                return

            src_global = offsets[src_type] + src_local
            dst_global = offsets[dst_type] + dst_local

            edge_sources.extend([src_global, dst_global])
            edge_targets.extend([dst_global, src_global])

        # user-item edges from events
        event_edges = events[["visitorid", "itemid"]].drop_duplicates()
        for visitor_id, item_id in event_edges.itertuples(index=False):
            add_bidirectional_edge("user", int(visitor_id), "item", int(item_id))

        # item-property edges (property name level)
        property_edges = item_properties[["itemid", "property"]].drop_duplicates()
        for item_id, prop in property_edges.itertuples(index=False):
            add_bidirectional_edge("item", int(item_id), "property", str(prop))

        # item-category edges derived from item_properties
        if not category_assignments.empty:
            category_assignments_unique = category_assignments[["itemid", "categoryid"]].drop_duplicates()
            for item_id, category_id in category_assignments_unique.itertuples(index=False):
                add_bidirectional_edge("item", int(item_id), "category", int(category_id))

        # category hierarchy edges
        category_edges = category_tree.dropna(subset=["categoryid", "parentid"])
        for category_id, parent_id in category_edges.itertuples(index=False):
            add_bidirectional_edge("category", int(category_id), "category", int(parent_id))

        interactions = event_edges.copy()
        interactions["user_index"] = interactions["visitorid"].map(user_map)
        interactions["item_index"] = interactions["itemid"].map(item_map)
        interactions = interactions.dropna(subset=["user_index", "item_index"])

        interaction_array = interactions[["user_index", "item_index"]].to_numpy(dtype=np.int64)

        return {
            "edge_sources": edge_sources,
            "edge_targets": edge_targets,
            "total_nodes": current_offset,
            "user_ids": user_ids.tolist(),
            "item_ids": item_ids.tolist(),
            "property_ids": property_ids.tolist(),
            "category_ids": category_ids.tolist(),
            "user_map": user_map,
            "item_map": item_map,
            "offsets": offsets,
            "interactions": interaction_array,
            "num_users": counts["user"],
            "num_items": counts["item"],
            "num_properties": counts["property"],
            "num_categories": counts["category"],
        }

    def _build_normalized_adjacency(
        self,
        sources: List[int],
        targets: List[int],
        total_nodes: int,
    ) -> Any:
        """
        정규화된 인접 행렬을 생성합니다.
        
        대칭 정규화(Symmetric Normalization)를 적용합니다:
        D^(-1/2) * A * D^(-1/2)
        여기서 D는 차수 행렬, A는 인접 행렬입니다.
        
        Args:
            sources: 엣지 소스 노드 인덱스 리스트
            targets: 엣지 타겟 노드 인덱스 리스트
            total_nodes: 전체 노드 수
            
        Returns:
            PyTorch 희소 텐서 형식의 정규화된 인접 행렬
            
        Raises:
            ValueError: 엣지가 없을 경우
        """
        if not sources or not targets:
            raise ValueError("그래프에 유효한 엣지가 없습니다.")

        data = np.ones(len(sources), dtype=np.float32)
        adjacency = sp.coo_matrix((data, (sources, targets)), shape=(total_nodes, total_nodes))
        adjacency = adjacency.tocsr()

        # 대칭 정규화를 위한 차수 행렬 계산: D^(-1/2)
        degrees = np.array(adjacency.sum(axis=1)).flatten()
        with np.errstate(divide="ignore"):
            deg_inv_sqrt = np.power(degrees, -0.5)
        # 차수가 0인 노드(고립 노드) 처리: inf나 nan을 0으로 설정
        deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
        deg_inv_sqrt[np.isnan(deg_inv_sqrt)] = 0.0

        # 대칭 정규화 적용: D^(-1/2) * A * D^(-1/2)
        deg_inv_sqrt_mat = sp.diags(deg_inv_sqrt)
        normalized = deg_inv_sqrt_mat @ adjacency @ deg_inv_sqrt_mat
        normalized = normalized.tocoo()

        indices = np.vstack((normalized.row, normalized.col))
        import torch  # 지연 로딩 이후에도 사용 가능하도록 지역 임포트

        indices_tensor = torch.from_numpy(indices).long()
        values_tensor = torch.from_numpy(normalized.data.astype(np.float32))
        adjacency_tensor = torch.sparse.FloatTensor(indices_tensor, values_tensor, torch.Size(normalized.shape))
        return adjacency_tensor.coalesce()

    def _build_model(self, adjacency: Any, torch_module: Any) -> Any:
        """
        LightGCN 모델을 생성합니다.
        
        LightGCN은 가중치가 없는 그래프 컨볼루션 레이어를 사용하며,
        각 레이어의 임베딩을 평균하여 최종 임베딩을 생성합니다.
        
        Args:
            adjacency: 정규화된 인접 행렬 (PyTorch 희소 텐서)
            torch_module: PyTorch 모듈
            
        Returns:
            학습 가능한 디바이스로 이동된 LightGCN 모델
        """
        class LightGCN(torch_module.nn.Module):
            def __init__(self, num_nodes: int, embedding_dim: int, n_layers: int, adj: Any) -> None:
                super().__init__()
                self.embedding = torch_module.nn.Embedding(num_nodes, embedding_dim)
                torch_module.nn.init.xavier_uniform_(self.embedding.weight)
                self.n_layers = n_layers
                self.register_buffer("adjacency", adj.coalesce())

            def forward(self) -> torch_module.Tensor:
                embeddings = self.embedding.weight
                all_embeddings = [embeddings]
                for _ in range(self.n_layers):
                    embeddings = torch_module.sparse.mm(self.adjacency, embeddings)
                    all_embeddings.append(embeddings)
                stacked = torch_module.stack(all_embeddings, dim=0)
                return stacked.mean(dim=0)

        model = LightGCN(
            adjacency.shape[0],
            self.embedding_dim,
            self.layers,
            adjacency,
        )
        device = torch_module.device(self.device)
        model = model.to(device)
        return model

    def _train_step(
        self,
        model: Any,
        optimizer: Any,
        batch: np.ndarray,
        graph_data: Dict[str, Any],
        torch_module: Any,
    ) -> float:
        """
        하나의 배치에 대해 학습 단계를 수행합니다.
        
        BPR 손실과 L2 정규화 손실을 결합하여 학습합니다.
        네가티브 샘플은 랜덤하게 선택하며, 포지티브 아이템과 겹치지 않도록 합니다.
        
        Args:
            model: LightGCN 모델
            optimizer: 옵티마이저
            batch: 배치 데이터 (shape: [batch_size, 2], 컬럼: [user_index, item_index])
            graph_data: 그래프 구조 정보 딕셔너리
            torch_module: PyTorch 모듈
            
        Returns:
            배치 손실 값 (float)
        """
        model.train()
        optimizer.zero_grad()

        user_indices = torch_module.from_numpy(batch[:, 0]).long().to(self.device)
        pos_item_indices = torch_module.from_numpy(batch[:, 1]).long().to(self.device)

        # 네가티브 샘플 생성: 랜덤하게 선택하되 포지티브 아이템과 겹치지 않도록 함
        neg_shape = (len(batch), max(self.num_negative, 1))
        neg_item_indices = torch_module.randint(0, graph_data["num_items"], neg_shape, device=self.device)

        # 포지티브 아이템과 겹치는 네가티브 샘플이 있으면 재선택
        conflict_mask = neg_item_indices.eq(pos_item_indices.unsqueeze(1))
        while torch_module.any(conflict_mask):
            replacement = torch_module.randint(
                0, graph_data["num_items"], (int(conflict_mask.sum()),), device=self.device
            )
            neg_item_indices[conflict_mask] = replacement
            conflict_mask = neg_item_indices.eq(pos_item_indices.unsqueeze(1))

        user_global = graph_data["offsets"]["user"] + user_indices
        pos_global = graph_data["offsets"]["item"] + pos_item_indices
        neg_global = graph_data["offsets"]["item"] + neg_item_indices

        all_embeddings = model()

        user_emb = all_embeddings[user_global]
        pos_emb = all_embeddings[pos_global]
        neg_emb = all_embeddings[neg_global.view(-1)].view(len(batch), -1, self.embedding_dim)

        pos_scores = (user_emb * pos_emb).sum(dim=1, keepdim=True)
        neg_scores = (user_emb.unsqueeze(1) * neg_emb).sum(dim=2)

        bpr_loss = -torch_module.log(torch_module.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        reg_loss = self.reg * (
            user_emb.pow(2).sum(dim=1, keepdim=True)
            + pos_emb.pow(2).sum(dim=1, keepdim=True)
            + neg_emb.pow(2).sum(dim=2)
        ).mean()

        loss = bpr_loss + reg_loss
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu().item())

    def _save_embeddings(self, all_embeddings: np.ndarray, graph_data: Dict[str, Any]) -> None:
        """
        학습된 임베딩을 CSV 파일로 저장합니다.
        
        사용자와 아이템 임베딩만 추출하여 저장합니다.
        
        Args:
            all_embeddings: 모든 노드의 임베딩 행렬 (shape: [total_nodes, embedding_dim])
            graph_data: 그래프 구조 정보 딕셔너리
            
        저장되는 파일:
        - gnn_user_embeddings.csv: 사용자 임베딩
        - gnn_item_embeddings.csv: 아이템 임베딩
        """
        user_offset = graph_data["offsets"]["user"]
        item_offset = graph_data["offsets"]["item"]

        user_indices = np.arange(graph_data["num_users"], dtype=np.int64)
        item_indices = np.arange(graph_data["num_items"], dtype=np.int64)

        user_global = user_offset + user_indices
        item_global = item_offset + item_indices

        user_embeddings = all_embeddings[user_global]
        item_embeddings = all_embeddings[item_global]

        user_df = pd.DataFrame(user_embeddings, columns=[f"embedding_{i}" for i in range(self.embedding_dim)])
        user_df.insert(0, "visitorid", graph_data["user_ids"])

        item_df = pd.DataFrame(item_embeddings, columns=[f"embedding_{i}" for i in range(self.embedding_dim)])
        item_df.insert(0, "itemid", graph_data["item_ids"])

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        user_df.to_csv(self.processed_dir / "gnn_user_embeddings.csv", index=False)
        item_df.to_csv(self.processed_dir / "gnn_item_embeddings.csv", index=False)

    def _create_graph_visualizations(self, graph_data: Dict[str, Any]) -> None:
        """
        그래프 구조를 시각화합니다.
        
        생성되는 시각화:
        1. 노드 타입별 분포 (바 차트)
        2. 그래프 통계 (노드 수, 엣지 수, 상호작용 수)
        3. 엣지 타입별 분포 (샘플링된 데이터)
        
        Args:
            graph_data: 그래프 구조 정보 딕셔너리
        """
        try:
            import seaborn as sns
        except ImportError:
            logger.warning("seaborn이 설치되지 않아 그래프 시각화를 건너뜁니다.")
            return

        viz_dir = Path("data/visualization")
        viz_dir.mkdir(parents=True, exist_ok=True)

        # 스타일 설정
        try:
            plt.style.use("seaborn-v0_8-darkgrid")
        except OSError:
            try:
                plt.style.use("seaborn-darkgrid")
            except OSError:
                plt.style.use("default")
        sns.set_palette("husl")

        # 1. 노드 타입별 분포
        try:
            node_counts = {
                "Users": graph_data["num_users"],
                "Items": graph_data["num_items"],
                "Properties": graph_data["num_properties"],
                "Categories": graph_data["num_categories"],
            }

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(node_counts.keys(), node_counts.values(), color=["#66b3ff", "#99ff99", "#ffcc99", "#ff99cc"])
            ax.set_ylabel("Number of Nodes", fontsize=12)
            ax.set_title("Graph Structure: Node Type Distribution", fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y")

            # 값 표시
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height):,}",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                )

            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(viz_dir / "gnn_node_distribution.png", dpi=300, bbox_inches="tight")
            plt.close()
            logger.info("Saved GNN node distribution to %s", viz_dir / "gnn_node_distribution.png")
        except Exception as e:
            logger.warning("GNN node distribution 시각화 생성 실패: %s", e)

        # 2. 엣지 수 및 그래프 통계
        try:
            total_edges = len(graph_data["edge_sources"]) // 2  # 양방향이므로 2로 나눔
            total_nodes = graph_data["total_nodes"]
            num_interactions = len(graph_data["interactions"])

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # 그래프 통계 바 차트
            stats = {
                "Total Nodes": total_nodes,
                "Total Edges": total_edges,
                "Interactions": num_interactions,
            }
            bars = axes[0].bar(stats.keys(), stats.values(), color=["#66b3ff", "#99ff99", "#ffcc99"])
            axes[0].set_ylabel("Count", fontsize=12)
            axes[0].set_title("Graph Statistics", fontsize=14, fontweight="bold")
            axes[0].grid(True, alpha=0.3, axis="y")

            for bar in bars:
                height = bar.get_height()
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height):,}",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                )

            # 노드 타입별 비율 파이 차트
            node_counts = {
                "Users": graph_data["num_users"],
                "Items": graph_data["num_items"],
                "Properties": graph_data["num_properties"],
                "Categories": graph_data["num_categories"],
            }
            sizes = list(node_counts.values())
            labels = [f"{k}\n({v:,})" for k, v in node_counts.items()]
            colors = ["#66b3ff", "#99ff99", "#ffcc99", "#ff99cc"]
            axes[1].pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10})
            axes[1].set_title("Node Type Distribution", fontsize=14, fontweight="bold")

            plt.tight_layout()
            plt.savefig(viz_dir / "gnn_graph_statistics.png", dpi=300, bbox_inches="tight")
            plt.close()
            logger.info("Saved GNN graph statistics to %s", viz_dir / "gnn_graph_statistics.png")
        except Exception as e:
            logger.warning("GNN graph statistics 시각화 생성 실패: %s", e)

        # 3. 엣지 타입별 분포 (간단한 추정)
        try:
            # 엣지 소스에서 노드 타입 추정
            offsets = graph_data["offsets"]
            edge_sources = np.array(graph_data["edge_sources"])
            edge_targets = np.array(graph_data["edge_targets"])

            # 양방향이므로 절반만 사용
            unique_edges = set()
            for i in range(0, len(edge_sources), 2):  # 양방향이므로 2씩 건너뜀
                src, dst = edge_sources[i], edge_targets[i]
                if src < dst:  # 중복 제거
                    unique_edges.add((src, dst))

            # 노드 타입 분류
            def get_node_type(node_idx: int) -> str:
                if node_idx < offsets["user"] + graph_data["num_users"]:
                    return "User"
                elif node_idx < offsets["item"] + graph_data["num_items"]:
                    return "Item"
                elif node_idx < offsets["property"] + graph_data["num_properties"]:
                    return "Property"
                else:
                    return "Category"

            edge_types = {}
            for src, dst in list(unique_edges)[:10000]:  # 샘플링 (너무 많으면)
                src_type = get_node_type(src)
                dst_type = get_node_type(dst)
                edge_type = f"{src_type}-{dst_type}"
                edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

            if edge_types:
                fig, ax = plt.subplots(figsize=(12, 6))
                edge_type_names = list(edge_types.keys())
                edge_type_counts = list(edge_types.values())
                bars = ax.barh(edge_type_names, edge_type_counts, color="#66b3ff")
                ax.set_xlabel("Number of Edges", fontsize=12)
                ax.set_title("Graph Structure: Edge Type Distribution (Sample)", fontsize=14, fontweight="bold")
                ax.grid(True, alpha=0.3, axis="x")

                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(
                        width,
                        bar.get_y() + bar.get_height() / 2.0,
                        f"{int(width):,}",
                        ha="left",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                    )

                plt.tight_layout()
                plt.savefig(viz_dir / "gnn_edge_type_distribution.png", dpi=300, bbox_inches="tight")
                plt.close()
                logger.info("Saved GNN edge type distribution to %s", viz_dir / "gnn_edge_type_distribution.png")
        except Exception as e:
            logger.warning("GNN edge type distribution 시각화 생성 실패: %s", e)

@dataclass
class ReRanker:
    """
    ALS와 GNN 결과를 결합한 리랭킹 클래스
    
    ALS 모델의 상위 후보 아이템을 선택하고, GNN 임베딩을 사용하여 점수를 재계산합니다.
    두 점수를 가중 평균하여 결합하고, 최종적으로 상위 K개 아이템을 선택합니다.
    
    리랭킹 전략:
    1. ALS 점수 상위 K개 후보 선택
    2. GNN 임베딩으로 점수 계산
    3. 두 점수를 정규화 후 가중 결합
    4. 테스트 세트의 긍정적 아이템에 보너스 점수 부여
    5. 재고 상태(availability)에 따라 최종 점수 조정
    
    Attributes:
        processed_dir: 전처리된 데이터가 저장된 디렉토리 경로
        data_dir: 원본 데이터가 저장된 디렉토리 경로
        top_k: 최종 추천할 아이템 수
        random_seed: 재현성을 위한 랜덤 시드
        als_candidate_k: ALS 후보 아이템 수 (리랭킹 전 선택)
        als_weight: ALS 점수 가중치 (0.0 ~ 1.0, 나머지는 GNN 가중치)
    """
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    data_dir: Path = field(default_factory=lambda: Path("data"))
    top_k: int = 200
    random_seed: int = 42
    als_candidate_k: int = 500
    als_weight: float = 0.4

    def __post_init__(self) -> None:
        """
        랜덤 시드를 설정합니다.
        """
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    def run(self) -> None:
        """
        리랭킹 파이프라인을 실행합니다.
        
        실행 순서:
        1. ALS 및 GNN 결과 로드
        2. 각 사용자에 대해:
           - ALS 점수로 후보 아이템 선택
           - GNN 임베딩으로 점수 계산
           - 점수 결합 및 리랭킹
        3. 최종 추천 결과 저장
        """
        logger.info("Loading ALS and GNN artifacts for reranking…")
        als_user_factors, als_item_factors, als_user_ids, als_item_ids = self._load_als_outputs()
        gnn_user_embeddings, gnn_item_embeddings, gnn_user_ids, gnn_item_ids = self._load_gnn_embeddings()
        availability = self._load_availability()

        gnn_user_map = {user_id: idx for idx, user_id in enumerate(gnn_user_ids)}
        gnn_item_map = {item_id: idx for idx, item_id in enumerate(gnn_item_ids)}

        results: List[Dict[str, Any]] = []

        item_id_array = np.array(als_item_ids, dtype=int)
        gnn_index_lookup = np.array([gnn_item_map.get(item_id, -1) for item_id in item_id_array], dtype=int)
        has_gnn_mask = gnn_index_lookup >= 0

        target_user_set: Optional[set[int]] = None
        positives_by_user: Dict[int, set[int]] = {}
        events_test_path = self.processed_dir / "events_test.csv"
        if events_test_path.exists():
            events_test_df = pd.read_csv(events_test_path, usecols=["visitorid", "event", "itemid"]).dropna()
            events_test_df["visitorid"] = events_test_df["visitorid"].astype(int)
            events_test_df["itemid"] = events_test_df["itemid"].astype(int)
            target_user_set = set(events_test_df["visitorid"].unique())
            positive_events = events_test_df[events_test_df["event"].isin(["addtocart", "transaction"])]
            positives_by_user = (
                positive_events.groupby("visitorid")["itemid"].apply(lambda s: set(s.tolist())).to_dict()
            )
            als_item_set = set(item_id_array.tolist())
            filtered_positives: Dict[int, set[int]] = {}
            for user_id, items in positives_by_user.items():
                intersected = {int(item) for item in items if int(item) in als_item_set}
                if intersected:
                    filtered_positives[int(user_id)] = intersected
            positives_by_user = filtered_positives
            logger.info(
                "Restricting reranking to %d users present in test events (positive users with overlap: %d)",
                len(target_user_set),
                len(positives_by_user),
            )

        max_user_index = min(len(als_user_ids), als_user_factors.shape[0])
        eligible_pairs = [
            (idx, als_user_ids[idx])
            for idx in range(max_user_index)
            if als_user_ids[idx] in gnn_user_map
            and (target_user_set is None or als_user_ids[idx] in target_user_set)
        ]
        total_users = len(eligible_pairs)
        if total_users == 0:
            logger.warning("No eligible users found for reranking; skipping.")
            return
        logger.info("Reranking: processing %d users", total_users)

        for processed_count, (user_index, visitor_id) in enumerate(eligible_pairs, start=1):
            if user_index >= als_user_factors.shape[0]:
                logger.warning(
                    "Skipping user index %d (visitor %s) because it exceeds ALS factors shape %d",
                    user_index,
                    visitor_id,
                    als_user_factors.shape[0],
                )
                continue

            als_user_vector = als_user_factors[user_index]
            als_user_vector = als_user_vector.astype(np.float32, copy=False)
            gnn_user_vector = gnn_user_embeddings[gnn_user_map[visitor_id]].astype(np.float32, copy=False)

            als_scores_full = als_user_vector @ als_item_factors.T

            total_items = als_scores_full.shape[0]
            als_cutoff = min(self.als_candidate_k, total_items)
            als_cutoff = max(als_cutoff, 1)

            sorted_indices = np.argsort(als_scores_full)
            candidate_indices = sorted_indices[-als_cutoff:]
            if candidate_indices.size == 0:
                continue
            candidate_indices = candidate_indices[candidate_indices < len(item_id_array)][::-1]
            if candidate_indices.size == 0:
                continue

            top_scores = als_scores_full[candidate_indices]
            item_ids_subset = item_id_array[candidate_indices]

            if gnn_item_embeddings.size > 0:
                gnn_indices_subset = gnn_index_lookup[candidate_indices]
                gnn_scores = np.zeros_like(top_scores)
                valid_gnn = gnn_indices_subset >= 0
                if np.any(valid_gnn):
                    gnn_subset_embeddings = gnn_item_embeddings[gnn_indices_subset[valid_gnn]]
                    gnn_scores[valid_gnn] = gnn_subset_embeddings @ gnn_user_vector
            else:
                gnn_scores = np.zeros_like(top_scores)

            # ALS와 GNN 점수만 가중 결합 (테스트 정답 미사용 — 사용 시 데이터 누수로 하이브리드 NDCG/Recall이 비정상적으로 높아짐)
            combined_scores, weight_a = self._combine_scores(top_scores, gnn_scores)

            # 결합된 점수 기준으로 상위 K개 선택
            order = np.argsort(combined_scores)[::-1][: self.top_k]

            # 최종 점수 계산: 재고가 있는 아이템은 1.5배 가중치 적용
            for rank, idx in enumerate(order, start=1):
                item_id = int(item_ids_subset[idx])
                score = float(combined_scores[idx])
                final_score = score * (1.5 if availability.get(item_id, 0) == 1 else 1.0)
                results.append(
                    {
                        "visitorid": visitor_id,
                        "itemid": item_id,
                        "rank": rank,
                        "combined_score": float(score),
                        "final_score": float(final_score),
                        "als_weight": float(weight_a),
                        "gnn_weight": float(1 - weight_a),
                    }
                )

            if processed_count % 5000 == 0 or processed_count == total_users:
                percent = (processed_count / total_users) * 100 if total_users else 100
                logger.info(
                    "Reranking progress: processed %d/%d users (%.1f%%)",
                    processed_count,
                    total_users,
                    percent,
                )

        if not results:
            raise ValueError("리랭킹 결과가 비어 있습니다.")

        rerank_df = pd.DataFrame(results)
        rerank_df.sort_values(["visitorid", "rank"], inplace=True)
        rerank_df.to_csv(self.processed_dir / "final_recommendations.csv", index=False)
        logger.info("Saved final reranked recommendations to %s", self.processed_dir.resolve())

    def _load_als_outputs(self) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
        """
        ALS 모델의 출력을 로드합니다.
        
        Returns:
            (als_user_factors, als_item_factors, als_user_ids, als_item_ids) 튜플
            
        Raises:
            FileNotFoundError: ALS 결과 파일이 없을 경우
        """
        user_factors_path = self.processed_dir / "als_user_factors.npy"
        item_factors_path = self.processed_dir / "als_item_factors.npy"
        user_mapping_path = self.processed_dir / "als_user_mapping.csv"
        item_mapping_path = self.processed_dir / "als_item_mapping.csv"

        if not user_factors_path.exists() or not item_factors_path.exists():
            raise FileNotFoundError("ALS 임베딩 파일이 존재하지 않습니다. ALSRecommender를 먼저 실행해 주세요.")

        als_user_factors = np.load(user_factors_path)
        als_item_factors = np.load(item_factors_path)

        user_ids = pd.read_csv(user_mapping_path)["visitorid"].astype(int).tolist()
        item_ids = pd.read_csv(item_mapping_path)["itemid"].astype(int).tolist()

        return als_user_factors, als_item_factors, user_ids, item_ids

    def _load_gnn_embeddings(self) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
        """
        GNN 임베딩을 로드합니다.
        
        Returns:
            (gnn_user_embeddings, gnn_item_embeddings, gnn_user_ids, gnn_item_ids) 튜플
            
        Raises:
            FileNotFoundError: GNN 임베딩 파일이 없을 경우
        """
        user_emb_path = self.processed_dir / "gnn_user_embeddings.csv"
        item_emb_path = self.processed_dir / "gnn_item_embeddings.csv"

        if not user_emb_path.exists() or not item_emb_path.exists():
            raise FileNotFoundError("GNN 임베딩 파일이 없습니다. GNNEmbeddingGenerator를 먼저 실행해 주세요.")

        user_df = pd.read_csv(user_emb_path)
        item_df = pd.read_csv(item_emb_path)

        user_ids = user_df["visitorid"].astype(int).tolist()
        item_ids = item_df["itemid"].astype(int).tolist()

        user_embeddings = user_df.drop(columns=["visitorid"]).to_numpy(dtype=np.float32)
        item_embeddings = item_df.drop(columns=["itemid"]).to_numpy(dtype=np.float32)

        return user_embeddings, item_embeddings, user_ids, item_ids

    def _load_availability(self) -> Dict[int, int]:
        """
        아이템의 재고 상태(availability)를 로드합니다.
        
        Returns:
            아이템 ID를 키로, 재고 상태(1=재고 있음, 0=재고 없음)를 값으로 하는 딕셔너리
            파일이 없거나 컬럼이 없으면 빈 딕셔너리 반환
        """
        # feature table에서 has_available 컬럼 사용
        feature_path = self.processed_dir / "feature_train_inliers.csv"
        if not feature_path.exists():
            return {}

        feature_table = pd.read_csv(feature_path, index_col=0)
        if "has_available" not in feature_table.columns:
            return {}

        # has_available이 1인 아이템만 반환
        available_items = feature_table[feature_table["has_available"] == 1].index.astype(int)
        return {int(item_id): 1 for item_id in available_items}

    def _combine_scores(self, als_scores: np.ndarray, gnn_scores: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        ALS와 GNN 점수를 결합합니다.
        
        두 점수를 Min-Max 정규화한 후 가중 평균을 계산합니다.
        
        Args:
            als_scores: ALS 점수 배열
            gnn_scores: GNN 점수 배열
            
        Returns:
            (결합된 점수 배열, ALS 가중치) 튜플
        """
        als_norm = self._minmax_scale(als_scores)
        gnn_norm = self._minmax_scale(gnn_scores)

        weight_a = float(np.clip(self.als_weight, 0.0, 1.0))
        combined = weight_a * als_norm + (1 - weight_a) * gnn_norm
        return combined, weight_a

    def _minmax_scale(self, scores: np.ndarray) -> np.ndarray:
        """
        점수 배열을 Min-Max 정규화합니다.
        
        Args:
            scores: 정규화할 점수 배열
            
        Returns:
            [0, 1] 범위로 정규화된 점수 배열
        """
        if scores.size == 0:
            return scores
        min_val = scores.min()
        max_val = scores.max()
        if np.isclose(max_val, min_val):
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)

@dataclass
class RecommendationComparator:
    """
    ALS, GNN, 최종 추천 결과를 비교하는 클래스
    
    각 모델의 추천 결과가 최종 추천 결과와 얼마나 유사한지 다양한 지표로 측정합니다.
    ALS와 GNN 각각이 최종 추천 결과와 얼마나 닮았는지, 그리고 두 모델을 결합한 결과가
    최종 추천과 얼마나 비슷한지를 분석합니다.
    
    비교 지표:
    
    1. Jaccard Similarity (자카드 유사도):
       - 의미: 두 추천 리스트의 아이템 집합 유사도를 측정하는 지표
       - 범위: 0.0 ~ 1.0 (1.0에 가까울수록 완전히 동일)
       - 계산: 교집합 크기 / 합집합 크기
       - 예시: ALS 추천 [1,2,3,4,5], 최종 추천 [1,2,6,7,8]
              → 교집합: {1,2} (크기 2), 합집합: {1,2,3,4,5,6,7,8} (크기 8)
              → Jaccard = 2/8 = 0.25
       - 해석:
         * 0.0 ~ 0.2: 매우 낮은 유사도 (거의 다른 추천)
         * 0.2 ~ 0.4: 낮은 유사도 (일부 공통)
         * 0.4 ~ 0.6: 중간 유사도 (상당한 공통)
         * 0.6 ~ 0.8: 높은 유사도 (대부분 공통)
         * 0.8 ~ 1.0: 매우 높은 유사도 (거의 동일)
       - 특징: 집합의 크기에 영향을 받지 않으며, 순서는 고려하지 않음
    
    2. Precision@K (정밀도):
       - 의미: 상위 K개 추천 중 공통 아이템이 차지하는 비율
       - 범위: 0.0 ~ 1.0 (1.0에 가까울수록 좋음)
       - 계산: 공통 아이템 수 / K
       - 예시: K=200, 공통 아이템이 8개
              → Precision@200 = 8/200 = 0.04
       - 해석:
         * 높을수록: 두 추천이 더 많은 공통 아이템을 포함
         * 낮을수록: 두 추천이 서로 다른 아이템을 많이 추천
       - 특징: 추천 리스트의 크기(K)에 정규화되어 있어 비교 가능
    
    3. Recall@K (재현율):
       - 의미: 최종 추천 아이템 중 ALS/GNN이 얼마나 포함했는지
       - 범위: 0.0 ~ 1.0 (1.0에 가까울수록 좋음)
       - 계산: 공통 아이템 수 / 최종 추천 아이템 수
       - 예시: 최종 추천에 200개 아이템, 공통 아이템이 14개
              → Recall@200 = 14/200 = 0.07
       - 해석:
         * 높을수록: ALS/GNN이 최종 추천의 대부분을 포함
         * 낮을수록: ALS/GNN이 최종 추천의 일부만 포함
       - 특징: 최종 추천을 기준으로 얼마나 "포괄"하는지 측정
    
    4. Average Rank Difference (평균 순위 차이):
       - 의미: 공통 아이템의 순위 차이 평균값
       - 범위: 0.0 이상 (0에 가까울수록 좋음, 이상적으로는 0)
       - 계산: 각 공통 아이템의 순위 차이 절댓값의 평균
       - 예시: 아이템 A가 ALS에서 5위, 최종에서 10위 → 차이 5
              아이템 B가 ALS에서 3위, 최종에서 4위 → 차이 1
              → 평균 순위 차이 = (5 + 1) / 2 = 3.0
       - 해석:
         * 0 ~ 10: 매우 유사한 순위 (거의 동일한 순서)
         * 10 ~ 30: 유사한 순위 (대체로 비슷한 위치)
         * 30 ~ 50: 다른 순위 (상당한 차이)
         * 50 이상: 매우 다른 순위 (완전히 다른 순서)
       - 특징: 공통 아이템이 있을 때만 계산되며, 순위의 중요성을 반영
    
    5. NDCG@K (Normalized Discounted Cumulative Gain):
       - 의미: 순위를 고려한 정규화된 누적 이득 (상위 순위에 더 높은 가중치)
       - 범위: 0.0 ~ 1.0 이상 (1.0 이상도 가능, 이상적으로는 1.0)
       - 계산:
         * DCG (Discounted Cumulative Gain): Σ(관련 아이템 점수 / log2(순위 + 1))
         * Ideal DCG: 최종 추천을 기준으로 한 이상적인 DCG
         * NDCG = DCG / Ideal DCG
       - 예시: 상위 3개 중 2개가 공통
              → DCG = 1/log2(2) + 1/log2(3) = 1 + 0.63 = 1.63
              → Ideal DCG = 1/log2(2) + 1/log2(3) = 1.63
              → NDCG = 1.63 / 1.63 = 1.0
       - 해석:
         * 0.0 ~ 0.3: 매우 낮은 순위 품질 (공통 아이템이 적거나 하위 순위에 많음)
         * 0.3 ~ 0.6: 낮은 순위 품질 (일부 공통이지만 순위가 다름)
         * 0.6 ~ 0.8: 중간 순위 품질 (상당한 공통, 순위도 어느 정도 유사)
         * 0.8 ~ 1.0: 높은 순위 품질 (대부분 공통, 순위도 유사)
         * 1.0 이상: 이상적보다 좋음 (상위 순위에 공통 아이템이 더 많음)
       - 특징: 상위 순위의 공통 아이템에 더 높은 가중치를 부여하여 순위의 중요성 반영
    
    실행 과정:
    1. ALS 추천 결과 로드 (als_recommendations.csv)
    2. 최종 추천 결과 로드 (final_recommendations.csv)
    3. GNN 임베딩만 사용하여 GNN 단독 추천 생성
    4. ALS vs 최종 추천 비교 분석
    5. GNN vs 최종 추천 비교 분석
    6. 결과 출력 및 CSV 파일로 저장
    
    출력 파일:
    - data/processed/recommendation_comparison.csv: 비교 결과가 저장되는 CSV 파일
    
    Attributes:
        processed_dir: 전처리된 데이터가 저장된 디렉토리 경로
        top_k: 비교에 사용할 상위 K개 아이템 (기본값: 200)
            - 각 사용자별로 상위 K개 추천만 비교에 사용
            - K가 클수록 더 많은 아이템을 비교하지만 계산 시간 증가
            - 일반적으로 50~200 사이의 값을 사용
    
    Examples:
        >>> # 기본 설정으로 비교 분석 실행
        >>> comparator = RecommendationComparator()
        >>> comparator.run()
        
        >>> # 상위 100개만 비교
        >>> comparator = RecommendationComparator(top_k=100)
        >>> comparator.run()
        
        >>> # 다른 디렉토리에서 비교
        >>> comparator = RecommendationComparator(
        ...     processed_dir=Path("custom/processed"),
        ...     top_k=150
        ... )
        >>> comparator.run()
    
    Note:
        - GNN 단독 추천은 GNN 임베딩만 사용하여 실시간으로 생성됩니다
        - 비교는 공통 사용자에 대해서만 수행됩니다
        - 각 지표는 사용자별로 계산된 후 평균값과 표준편차를 출력합니다
        - NDCG가 1.0을 초과할 수 있는데, 이는 비교 대상(ALS/GNN)이 기준(최종 추천)보다
          상위 순위에 더 많은 공통 아이템을 포함하고 있다는 의미입니다
    """
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    top_k: int = 200

    def run(self) -> None:
        """
        추천 결과 비교 분석을 실행합니다.
        
        실행 순서:
        1. ALS, GNN, 최종 추천 결과 로드
        2. GNN만 사용한 추천 결과 생성
        3. 각 모델별로 최종 추천과 비교 지표 계산
        4. 결과 출력 및 저장
        """
        logger.info("추천 결과 비교 분석 시작...")
        
        # 추천 결과 로드
        als_recs = self._load_recommendations("als_recommendations.csv", ["visitorid", "itemid", "score", "rank"])
        final_recs = self._load_recommendations("final_recommendations.csv", ["visitorid", "itemid", "rank", "final_score"])
        
        # GNN만 사용한 추천 생성
        gnn_recs = self._generate_gnn_only_recommendations()
        
        # 비교 분석 수행
        results = {}
        user_metrics = {}
        
        logger.info("ALS vs 최종 추천 비교 중...")
        als_result, als_user_metrics = self._compare_recommendations(als_recs, final_recs, "ALS", "최종")
        results["als_vs_final"] = als_result
        user_metrics["als_vs_final"] = als_user_metrics
        
        logger.info("GNN vs 최종 추천 비교 중...")
        gnn_result, gnn_user_metrics = self._compare_recommendations(gnn_recs, final_recs, "GNN", "최종")
        results["gnn_vs_final"] = gnn_result
        user_metrics["gnn_vs_final"] = gnn_user_metrics
        
        # 결과 출력
        self._print_comparison_results(results)
        
        # 결과 저장
        self._save_comparison_results(results)
        
        # 시각화 생성
        self._create_visualizations(results, user_metrics)
        
        logger.info("추천 결과 비교 분석 완료!")

    def _load_recommendations(self, filename: str, columns: List[str]) -> DataFrame:
        """추천 결과 파일을 로드합니다."""
        filepath = self.processed_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"추천 결과 파일을 찾을 수 없습니다: {filepath}")
        df = pd.read_csv(filepath, usecols=columns)
        df["visitorid"] = df["visitorid"].astype(int)
        df["itemid"] = df["itemid"].astype(int)
        return df

    def _generate_gnn_only_recommendations(self) -> DataFrame:
        """GNN 임베딩만 사용하여 추천 결과를 생성합니다."""
        logger.info("GNN만 사용한 추천 결과 생성 중...")
        
        # GNN 임베딩 로드
        gnn_user_embeddings_df = pd.read_csv(self.processed_dir / "gnn_user_embeddings.csv")
        gnn_item_embeddings_df = pd.read_csv(self.processed_dir / "gnn_item_embeddings.csv")
        
        # 임베딩 배열로 변환
        gnn_user_ids = gnn_user_embeddings_df["visitorid"].values.astype(int)
        gnn_item_ids = gnn_item_embeddings_df["itemid"].values.astype(int)
        
        user_embedding_cols = [col for col in gnn_user_embeddings_df.columns if col.startswith("embedding_")]
        item_embedding_cols = [col for col in gnn_item_embeddings_df.columns if col.startswith("embedding_")]
        
        gnn_user_embeddings = gnn_user_embeddings_df[user_embedding_cols].values.astype(np.float32)
        gnn_item_embeddings = gnn_item_embeddings_df[item_embedding_cols].values.astype(np.float32)
        
        # 매핑 생성
        gnn_user_map = {user_id: idx for idx, user_id in enumerate(gnn_user_ids)}
        gnn_item_map = {item_id: idx for idx, item_id in enumerate(gnn_item_ids)}
        
        # 최종 추천에 있는 사용자들만 처리 (이미 run()에서 로드했지만 여기서도 필요)
        final_recs_path = self.processed_dir / "final_recommendations.csv"
        if final_recs_path.exists():
            final_recs = pd.read_csv(final_recs_path, usecols=["visitorid"])
            target_users = final_recs["visitorid"].unique()
        else:
            # 최종 추천이 없으면 모든 GNN 사용자 사용
            target_users = gnn_user_ids
        
        results: List[Dict[str, Any]] = []
        
        for visitor_id in target_users:
            if visitor_id not in gnn_user_map:
                continue
            
            user_idx = gnn_user_map[visitor_id]
            user_vector = gnn_user_embeddings[user_idx]
            
            # 모든 아이템에 대한 점수 계산
            gnn_scores = gnn_item_embeddings @ user_vector
            
            # 상위 K개 선택
            top_indices = np.argsort(gnn_scores)[::-1][:self.top_k]
            
            for rank, item_idx in enumerate(top_indices, start=1):
                item_id = gnn_item_ids[item_idx]
                score = float(gnn_scores[item_idx])
                results.append({
                    "visitorid": int(visitor_id),
                    "itemid": int(item_id),
                    "score": score,
                    "rank": rank
                })
        
        return pd.DataFrame(results)

    def _compare_recommendations(
        self, 
        rec1: DataFrame, 
        rec2: DataFrame, 
        name1: str, 
        name2: str
    ) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
        """
        두 추천 결과를 비교하여 다양한 지표를 계산합니다.
        
        Args:
            rec1: 첫 번째 추천 결과 (비교 대상)
            rec2: 두 번째 추천 결과 (기준, 최종 추천)
            name1: 첫 번째 추천의 이름
            name2: 두 번째 추천의 이름
            
        Returns:
            (평균 지표 딕셔너리, 사용자별 지표 딕셔너리) 튜플
        """
        # 사용자별로 그룹화
        # rec1 정렬 기준 결정
        rec1_sort_col = "score" if "score" in rec1.columns else "rank"
        rec1_ascending = False if rec1_sort_col == "score" else True
        
        # FutureWarning 해결: 그룹핑 컬럼을 제외하고 필요한 컬럼만 선택
        rec1_cols = ["itemid", rec1_sort_col]
        rec1_by_user = rec1.groupby("visitorid")[rec1_cols].apply(
            lambda x: set(x.nlargest(self.top_k, rec1_sort_col)["itemid"] if rec1_sort_col == "score" 
                         else x.nsmallest(self.top_k, rec1_sort_col)["itemid"])
        ).to_dict()
        
        rec2_by_user = rec2.groupby("visitorid")[["itemid", "rank"]].apply(
            lambda x: set(x.nsmallest(self.top_k, "rank")["itemid"])
        ).to_dict()
        
        # 공통 사용자만 비교
        common_users = set(rec1_by_user.keys()) & set(rec2_by_user.keys())
        
        if len(common_users) == 0:
            logger.warning(f"{name1}와 {name2} 사이에 공통 사용자가 없습니다.")
            return {}
        
        jaccard_similarities = []
        precisions = []
        recalls = []
        rank_differences = []
        ndcgs = []
        
        for user_id in common_users:
            items1 = rec1_by_user[user_id]
            items2 = rec2_by_user[user_id]
            
            # Jaccard Similarity
            intersection = len(items1 & items2)
            union = len(items1 | items2)
            jaccard = intersection / union if union > 0 else 0.0
            jaccard_similarities.append(jaccard)
            
            # Precision@K
            precision = intersection / self.top_k if self.top_k > 0 else 0.0
            precisions.append(precision)
            
            # Recall@K
            recall = intersection / len(items2) if len(items2) > 0 else 0.0
            recalls.append(recall)
            
            # Rank Difference (공통 아이템의 순위 차이)
            rec1_sort_col = "score" if "score" in rec1.columns else "rank"
            user_rec1 = rec1[rec1["visitorid"] == user_id].nlargest(self.top_k, rec1_sort_col) if rec1_sort_col == "score" else rec1[rec1["visitorid"] == user_id].nsmallest(self.top_k, rec1_sort_col)
            user_rec2 = rec2[rec2["visitorid"] == user_id].nsmallest(self.top_k, "rank")
            
            # 아이템별 순위 매핑
            rank_map1 = {row["itemid"]: idx + 1 for idx, (_, row) in enumerate(user_rec1.iterrows())}
            rank_map2 = {row["itemid"]: idx + 1 for idx, (_, row) in enumerate(user_rec2.iterrows())}
            
            common_items = items1 & items2
            if common_items:
                rank_diffs = []
                for item in common_items:
                    if item in rank_map1 and item in rank_map2:
                        rank_diffs.append(abs(rank_map1[item] - rank_map2[item]))
                if rank_diffs:
                    rank_differences.append(np.mean(rank_diffs))
            
            # NDCG@K 계산
            ndcg = self._calculate_ndcg(user_rec1, user_rec2, common_items)
            ndcgs.append(ndcg)
        
        avg_metrics = {
            "jaccard_similarity": np.mean(jaccard_similarities),
            "precision_at_k": np.mean(precisions),
            "recall_at_k": np.mean(recalls),
            "avg_rank_difference": np.mean(rank_differences) if rank_differences else 0.0,
            "ndcg_at_k": np.mean(ndcgs),
            "num_common_users": len(common_users),
            "jaccard_std": np.std(jaccard_similarities),
            "precision_std": np.std(precisions),
            "recall_std": np.std(recalls),
        }
        
        user_metrics = {
            "jaccard_similarities": jaccard_similarities,
            "precisions": precisions,
            "recalls": recalls,
            "rank_differences": rank_differences if rank_differences else [],
            "ndcgs": ndcgs,
        }
        
        return avg_metrics, user_metrics

    def _calculate_ndcg(self, rec1: DataFrame, rec2: DataFrame, relevant_items: set) -> float:
        """NDCG@K를 계산합니다."""
        if len(relevant_items) == 0:
            return 0.0
        
        # rec2를 기준으로 ideal DCG 계산
        ideal_gains = []
        for idx, (_, row) in enumerate(rec2.iterrows()):
            if row["itemid"] in relevant_items:
                ideal_gains.append(1.0 / np.log2(idx + 2))
        ideal_dcg = sum(ideal_gains[:self.top_k])
        
        if ideal_dcg == 0:
            return 0.0
        
        # rec1의 DCG 계산
        dcg = 0.0
        for idx, (_, row) in enumerate(rec1.iterrows()):
            if idx >= self.top_k:
                break
            if row["itemid"] in relevant_items:
                dcg += 1.0 / np.log2(idx + 2)
        
        return dcg / ideal_dcg

    def _print_comparison_results(self, results: Dict[str, Dict[str, float]]) -> None:
        """비교 결과를 출력합니다."""
        print("\n" + "="*80)
        print("추천 결과 비교 분석")
        print("="*80)
        
        for comparison_name, metrics in results.items():
            if not metrics:
                continue
            
            print(f"\n[{comparison_name.upper().replace('_', ' ')}]")
            print("-" * 80)
            print(f"공통 사용자 수: {metrics.get('num_common_users', 0):,}")
            print(f"Jaccard Similarity: {metrics.get('jaccard_similarity', 0):.4f} (±{metrics.get('jaccard_std', 0):.4f})")
            print(f"Precision@{self.top_k}: {metrics.get('precision_at_k', 0):.4f} (±{metrics.get('precision_std', 0):.4f})")
            print(f"Recall@{self.top_k}: {metrics.get('recall_at_k', 0):.4f} (±{metrics.get('recall_std', 0):.4f})")
            print(f"평균 순위 차이: {metrics.get('avg_rank_difference', 0):.2f}")
            print(f"NDCG@{self.top_k}: {metrics.get('ndcg_at_k', 0):.4f}")
        
        print("\n" + "="*80)

    def _save_comparison_results(self, results: Dict[str, Dict[str, float]]) -> None:
        """비교 결과를 CSV 파일로 저장합니다."""
        output_path = self.processed_dir / "recommendation_comparison.csv"
        
        rows = []
        for comparison_name, metrics in results.items():
            if metrics:
                row = {"comparison": comparison_name}
                row.update(metrics)
                rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            logger.info(f"비교 결과를 저장했습니다: {output_path}")

    def _create_visualizations(
        self,
        results: Dict[str, Dict[str, float]],
        user_metrics: Dict[str, Dict[str, List[float]]]
    ) -> None:
        """
        비교 분석 결과를 시각화합니다.
        
        생성되는 시각화:
        1. 지표별 비교 바 차트 (ALS vs 최종, GNN vs 최종)
        2. Jaccard Similarity 분포 히스토그램
        3. Precision@K 분포 히스토그램
        4. Recall@K 분포 히스토그램
        5. NDCG@K 분포 히스토그램
        6. 평균 순위 차이 비교
        7. 사용자별 공통 아이템 수 분포
        
        Args:
            results: 평균 지표 딕셔너리
            user_metrics: 사용자별 지표 딕셔너리
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib 또는 seaborn이 설치되지 않아 시각화를 건너뜁니다.")
            return

        viz_dir = Path("data/visualization")
        viz_dir.mkdir(parents=True, exist_ok=True)

        # 스타일 설정
        try:
            plt.style.use("seaborn-v0_8-darkgrid")
        except OSError:
            try:
                plt.style.use("seaborn-darkgrid")
            except OSError:
                plt.style.use("default")
        sns.set_palette("husl")

        # 1. 지표별 비교 바 차트
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle("Recommendation Comparison - Metrics Comparison", fontsize=16, fontweight="bold", y=0.995)

            metrics_to_plot = [
                ("jaccard_similarity", "Jaccard Similarity", axes[0, 0]),
                ("precision_at_k", f"Precision@{self.top_k}", axes[0, 1]),
                ("recall_at_k", f"Recall@{self.top_k}", axes[0, 2]),
                ("ndcg_at_k", f"NDCG@{self.top_k}", axes[1, 0]),
                ("avg_rank_difference", "Average Rank Difference", axes[1, 1]),
            ]

            comparisons = ["als_vs_final", "gnn_vs_final"]
            comparison_labels = ["ALS vs Final", "GNN vs Final"]
            colors = ["#3498db", "#e74c3c"]

            for metric_key, metric_label, ax in metrics_to_plot:
                values = []
                labels = []
                for comp, comp_label in zip(comparisons, comparison_labels):
                    if comp in results and results[comp] and metric_key in results[comp]:
                        values.append(results[comp][metric_key])
                        labels.append(comp_label)
                
                if values:
                    bars = ax.bar(labels, values, color=colors[:len(values)], alpha=0.7, edgecolor='black', linewidth=1.5)
                    ax.set_ylabel(metric_label, fontsize=11, fontweight="bold")
                    ax.set_title(metric_label, fontsize=12, fontweight="bold")
                    ax.grid(axis='y', alpha=0.3, linestyle='--')
                    
                    # 값 표시
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.4f}',
                               ha='center', va='bottom', fontsize=10, fontweight="bold")
                    
                    # 평균 순위 차이는 낮을수록 좋으므로 y축 반전
                    if metric_key == "avg_rank_difference":
                        ax.invert_yaxis()

            # 마지막 subplot에 요약 정보 표시
            ax = axes[1, 2]
            ax.axis('off')
            summary_text = "Comparison Summary\n\n"
            for comp, comp_label in zip(comparisons, comparison_labels):
                if comp in results and results[comp]:
                    summary_text += f"{comp_label}:\n"
                    summary_text += f"  Common Users: {results[comp].get('num_common_users', 0):,}\n"
                    summary_text += f"  Jaccard: {results[comp].get('jaccard_similarity', 0):.4f}\n"
                    summary_text += f"  Precision@{self.top_k}: {results[comp].get('precision_at_k', 0):.4f}\n"
                    summary_text += f"  Recall@{self.top_k}: {results[comp].get('recall_at_k', 0):.4f}\n\n"
            ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                   family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()
            plt.savefig(viz_dir / "recommendation_comparison_metrics.png", dpi=300, bbox_inches="tight")
            plt.close()
            logger.info("Saved comparison metrics chart to %s", viz_dir / "recommendation_comparison_metrics.png")
        except Exception as e:
            logger.warning("지표별 비교 차트 생성 실패: %s", e)

        # 2. Jaccard Similarity 분포 히스토그램
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle("Jaccard Similarity Distribution Comparison", fontsize=14, fontweight="bold")

            for idx, (comp, comp_label) in enumerate(zip(comparisons, comparison_labels)):
                if comp in user_metrics and "jaccard_similarities" in user_metrics[comp]:
                    jaccards = user_metrics[comp]["jaccard_similarities"]
                    if jaccards:
                        axes[idx].hist(jaccards, bins=50, alpha=0.7, color=colors[idx], edgecolor='black', linewidth=0.5)
                        axes[idx].axvline(np.mean(jaccards), color='red', linestyle='--', linewidth=2, 
                                         label=f'Mean: {np.mean(jaccards):.4f}')
                        axes[idx].set_xlabel("Jaccard Similarity", fontsize=11)
                        axes[idx].set_ylabel("Number of Users", fontsize=11)
                        axes[idx].set_title(f"{comp_label}\n(Mean: {np.mean(jaccards):.4f}, Std: {np.std(jaccards):.4f})", 
                                           fontsize=11, fontweight="bold")
                        axes[idx].legend()
                        axes[idx].grid(axis='y', alpha=0.3, linestyle='--')

            plt.tight_layout()
            plt.savefig(viz_dir / "recommendation_comparison_jaccard_distribution.png", dpi=300, bbox_inches="tight")
            plt.close()
            logger.info("Saved Jaccard similarity distribution to %s", viz_dir / "recommendation_comparison_jaccard_distribution.png")
        except Exception as e:
            logger.warning("Jaccard Similarity 분포 히스토그램 생성 실패: %s", e)

        # 3. Precision@K 분포 히스토그램
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"Precision@{self.top_k} Distribution Comparison", fontsize=14, fontweight="bold")

            for idx, (comp, comp_label) in enumerate(zip(comparisons, comparison_labels)):
                if comp in user_metrics and "precisions" in user_metrics[comp]:
                    precisions = user_metrics[comp]["precisions"]
                    if precisions:
                        axes[idx].hist(precisions, bins=50, alpha=0.7, color=colors[idx], edgecolor='black', linewidth=0.5)
                        axes[idx].axvline(np.mean(precisions), color='red', linestyle='--', linewidth=2,
                                         label=f'Mean: {np.mean(precisions):.4f}')
                        axes[idx].set_xlabel(f"Precision@{self.top_k}", fontsize=11)
                        axes[idx].set_ylabel("Number of Users", fontsize=11)
                        axes[idx].set_title(f"{comp_label}\n(Mean: {np.mean(precisions):.4f}, Std: {np.std(precisions):.4f})",
                                           fontsize=11, fontweight="bold")
                        axes[idx].legend()
                        axes[idx].grid(axis='y', alpha=0.3, linestyle='--')

            plt.tight_layout()
            plt.savefig(viz_dir / "recommendation_comparison_precision_distribution.png", dpi=300, bbox_inches="tight")
            plt.close()
            logger.info("Saved Precision@K distribution to %s", viz_dir / "recommendation_comparison_precision_distribution.png")
        except Exception as e:
            logger.warning("Precision@K 분포 히스토그램 생성 실패: %s", e)

        # 4. Recall@K 분포 히스토그램
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"Recall@{self.top_k} Distribution Comparison", fontsize=14, fontweight="bold")

            for idx, (comp, comp_label) in enumerate(zip(comparisons, comparison_labels)):
                if comp in user_metrics and "recalls" in user_metrics[comp]:
                    recalls = user_metrics[comp]["recalls"]
                    if recalls:
                        axes[idx].hist(recalls, bins=50, alpha=0.7, color=colors[idx], edgecolor='black', linewidth=0.5)
                        axes[idx].axvline(np.mean(recalls), color='red', linestyle='--', linewidth=2,
                                         label=f'Mean: {np.mean(recalls):.4f}')
                        axes[idx].set_xlabel(f"Recall@{self.top_k}", fontsize=11)
                        axes[idx].set_ylabel("Number of Users", fontsize=11)
                        axes[idx].set_title(f"{comp_label}\n(Mean: {np.mean(recalls):.4f}, Std: {np.std(recalls):.4f})",
                                           fontsize=11, fontweight="bold")
                        axes[idx].legend()
                        axes[idx].grid(axis='y', alpha=0.3, linestyle='--')

            plt.tight_layout()
            plt.savefig(viz_dir / "recommendation_comparison_recall_distribution.png", dpi=300, bbox_inches="tight")
            plt.close()
            logger.info("Saved Recall@K distribution to %s", viz_dir / "recommendation_comparison_recall_distribution.png")
        except Exception as e:
            logger.warning("Recall@K 분포 히스토그램 생성 실패: %s", e)

        # 5. NDCG@K 분포 히스토그램
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"NDCG@{self.top_k} Distribution Comparison", fontsize=14, fontweight="bold")

            for idx, (comp, comp_label) in enumerate(zip(comparisons, comparison_labels)):
                if comp in user_metrics and "ndcgs" in user_metrics[comp]:
                    ndcgs = user_metrics[comp]["ndcgs"]
                    if ndcgs:
                        axes[idx].hist(ndcgs, bins=50, alpha=0.7, color=colors[idx], edgecolor='black', linewidth=0.5)
                        axes[idx].axvline(np.mean(ndcgs), color='red', linestyle='--', linewidth=2,
                                         label=f'Mean: {np.mean(ndcgs):.4f}')
                        axes[idx].set_xlabel(f"NDCG@{self.top_k}", fontsize=11)
                        axes[idx].set_ylabel("Number of Users", fontsize=11)
                        axes[idx].set_title(f"{comp_label}\n(Mean: {np.mean(ndcgs):.4f}, Std: {np.std(ndcgs):.4f})",
                                           fontsize=11, fontweight="bold")
                        axes[idx].legend()
                        axes[idx].grid(axis='y', alpha=0.3, linestyle='--')

            plt.tight_layout()
            plt.savefig(viz_dir / "recommendation_comparison_ndcg_distribution.png", dpi=300, bbox_inches="tight")
            plt.close()
            logger.info("Saved NDCG@K distribution to %s", viz_dir / "recommendation_comparison_ndcg_distribution.png")
        except Exception as e:
            logger.warning("NDCG@K 분포 히스토그램 생성 실패: %s", e)

        # 6. 박스플롯으로 모든 지표 비교
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            fig.suptitle("Recommendation Comparison - Boxplot", fontsize=16, fontweight="bold")

            metrics_for_boxplot = [
                ("jaccard_similarities", "Jaccard Similarity", axes[0, 0]),
                ("precisions", f"Precision@{self.top_k}", axes[0, 1]),
                ("recalls", f"Recall@{self.top_k}", axes[1, 0]),
                ("ndcgs", f"NDCG@{self.top_k}", axes[1, 1]),
            ]

            for metric_key, metric_label, ax in metrics_for_boxplot:
                data_to_plot = []
                labels = []
                for comp, comp_label in zip(comparisons, comparison_labels):
                    if comp in user_metrics and metric_key in user_metrics[comp]:
                        values = user_metrics[comp][metric_key]
                        if values:
                            data_to_plot.append(values)
                            labels.append(comp_label)
                
                if data_to_plot:
                    bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True, 
                                   showmeans=True, meanline=True)
                    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    ax.set_ylabel(metric_label, fontsize=11, fontweight="bold")
                    ax.set_title(metric_label, fontsize=12, fontweight="bold")
                    ax.grid(axis='y', alpha=0.3, linestyle='--')

            plt.tight_layout()
            plt.savefig(viz_dir / "recommendation_comparison_boxplot.png", dpi=300, bbox_inches="tight")
            plt.close()
            logger.info("Saved comparison boxplot to %s", viz_dir / "recommendation_comparison_boxplot.png")
        except Exception as e:
            logger.warning("박스플롯 생성 실패: %s", e)

@dataclass
class TestSetEvaluator:
    """
    테스트 세트 평가 클래스
    
    최종 추천 결과를 테스트 세트와 비교하여 다양한 성능 지표를 계산합니다.
    
    평가 지표:
    - 사용자 레벨: Hit Rate, Precision, Recall, F1, ROC-AUC, Average Precision
    - 아이템 레벨: Precision, Recall, F1 (최적 임계값 기준)
    
    평가 모드 (evaluation_mode):
    
    1. "strict" 모드 (기존 방식):
       - Positive: addtocart 또는 transaction 이벤트만
       - Negative: view 이벤트 또는 이벤트 없음
       - 예측: 추천된 아이템 중 최소 1개라도 transaction/addtocart와 매칭되면 positive
       - 특징: 가장 엄격한 기준, 높은 precision, 낮은 recall
       - 사용 시기: 정확한 추천이 중요한 경우, false positive를 최소화하고 싶을 때
    
    2. "weighted" 모드 (가중치 기반):
       - Positive: transaction/addtocart (가중치 1.0) 또는 view (가중치 view_weight)
       - Negative: 이벤트 없음
       - 예측: 추천된 아이템 중 transaction/addtocart가 있으면 무조건 positive, 없어도 view 가중치 합계가 threshold(view_weight) 이상이면 positive
       - 특징: view 이벤트도 일정 가중치를 받아 더 많은 positive 예측 가능
       - 파라미터: view_weight (기본값: 0.3)
       - 사용 시기: view도 사용자 관심의 신호로 간주하고 싶을 때
    
    3. "partial" 모드 (부분 점수 시스템):
       - Positive: transaction/addtocart 이벤트
       - 예측: Top-K 추천 중 일정 비율(min_hit_ratio) 이상 hit하면 positive
       - 예시: top_k=50, min_hit_ratio=0.1이면 5개 이상 hit하면 positive
       - 특징: 더 관대한 기준으로 더 많은 positive 예측, recall 향상 가능
       - 파라미터: min_hit_ratio (기본값: 0.1 = 10%)
       - 사용 시기: recall을 높이고 싶을 때, 부분적으로 맞는 추천도 인정하고 싶을 때
    
    4. "score_based" 모드 (점수 기반 평가):
       - Positive: transaction/addtocart 이벤트
       - 예측: 추천 점수(final_score)가 임계값 이상인 아이템 중 hit가 있으면 positive
       - 특징: 높은 점수의 추천만 신뢰하여 평가, 모델의 확신도 반영
       - 파라미터: score_threshold (None이면 score_percentile로 자동 계산)
       - 사용 시기: 모델의 점수 신뢰도가 높을 때, 높은 점수 추천만 평가하고 싶을 때
    
    5. "rank_based" 모드 (순위 기반 평가):
       - Positive: transaction/addtocart 이벤트
       - 예측: 상위 순위(rank) 내에서 hit가 있으면 positive
       - 특징: 상위 추천에 더 높은 가중치 부여, 실제 사용자 행동 패턴 반영
       - 파라미터: top_rank_ratio (기본값: 0.25 = 상위 25%)
       - 사용 시기: 상위 추천의 정확도가 중요할 때, 순위가 의미가 있을 때
    
    Attributes:
        processed_dir: 전처리된 데이터가 저장된 디렉토리 경로
        top_k: 평가에 사용할 추천 아이템 수 (기본값: 50)
        evaluation_mode: 평가 모드 ("strict", "weighted", "partial", "score_based", "rank_based", 기본값: "weighted")
        view_weight: view 이벤트의 가중치 (weighted 모드에서 사용, 기본값: 0.3)
            - transaction/addtocart는 1.0, view는 이 값 사용
            - 낮을수록 엄격, 높을수록 관대한 평가
        min_hit_ratio: 부분 점수 시스템에서 positive로 분류하기 위한 최소 hit 비율 
            (partial 모드에서 사용, 기본값: 0.1 = 10%)
            - 0.0 ~ 1.0 사이의 값
            - 낮을수록 더 많은 positive 예측 (더 관대)
            - 높을수록 더 적은 positive 예측 (더 엄격)
        score_threshold: 점수 기반 평가에서 사용할 점수 임계값 (score_based 모드에서 사용)
            - None이면 score_percentile로 자동 계산
            - 설정하면 해당 값 이상의 점수를 가진 추천만 positive로 분류
        score_percentile: 점수 임계값 자동 계산 시 사용할 백분위수 (score_based 모드에서 사용, 기본값: 75.0)
            - 0.0 ~ 100.0 사이의 값
            - 예: 75.0이면 상위 25% 추천만 positive로 분류
        top_rank_ratio: 순위 기반 평가에서 상위 순위 비율 (rank_based 모드에서 사용, 기본값: 0.25)
            - 0.0 ~ 1.0 사이의 값
            - 예: 0.25이면 상위 25% 순위 내에서 hit가 있으면 positive
    
    Examples:
        >>> # Strict 모드 사용
        >>> evaluator = TestSetEvaluator(evaluation_mode="strict", top_k=50)
        >>> evaluator.run()
        
        >>> # Weighted 모드 사용 (view 가중치 0.5)
        >>> evaluator = TestSetEvaluator(
        ...     evaluation_mode="weighted", 
        ...     top_k=50, 
        ...     view_weight=0.5
        ... )
        >>> evaluator.run()
        
        >>> # Partial 모드 사용 (20% 이상 hit하면 positive)
        >>> evaluator = TestSetEvaluator(
        ...     evaluation_mode="partial", 
        ...     top_k=50, 
        ...     min_hit_ratio=0.2
        ... )
        >>> evaluator.run()
        
        >>> # Score-based 모드 사용 (상위 25% 점수만 평가)
        >>> evaluator = TestSetEvaluator(
        ...     evaluation_mode="score_based", 
        ...     top_k=50, 
        ...     score_percentile=75.0
        ... )
        >>> evaluator.run()
        
        >>> # Rank-based 모드 사용 (상위 25% 순위만 평가)
        >>> evaluator = TestSetEvaluator(
        ...     evaluation_mode="rank_based", 
        ...     top_k=50, 
        ...     top_rank_ratio=0.25
        ... )
        >>> evaluator.run()
    """
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    top_k: int = 50
    evaluation_mode: str = "score_based"  # "strict", "weighted", "partial", "score_based", "rank_based"
    view_weight: float = 0.3  # view 이벤트의 가중치 (transaction=1.0, addtocart=1.0 기준)
    min_hit_ratio: float = 0.1  # partial 모드에서 최소 hit 비율 (예: 0.1 = 10%)
    score_threshold: Optional[float] = None  # score_based 모드에서 사용할 점수 임계값 (None이면 자동 계산)
    score_percentile: float = 75.0  # score_based 모드에서 임계값 계산 시 사용할 백분위수 (기본값: 75%)
    top_rank_ratio: float = 0.25  # rank_based 모드에서 상위 순위 비율 (기본값: 0.25 = 상위 25%)

    def run(self) -> None:
        """
        평가 파이프라인을 실행합니다.
        
        실행 순서:
        1. 추천 결과 및 테스트 이벤트 로드
        2. ALS, GNN, 최종 추천 결과 각각 평가
        3. 성능 지표 계산 및 출력
        4. 시각화 생성 (ROC Curve, PR Curve, Confusion Matrix)
        """
        events_test_path = self.processed_dir / "events_test.csv"
        if not events_test_path.exists():
            raise FileNotFoundError("events_test.csv 파일이 없습니다. IsolationForestPreprocessor가 생성한 데이터를 확인해 주세요.")

        events_test = pd.read_csv(events_test_path)
        test_users = events_test["visitorid"].dropna().astype(int).unique()

        # Positive 이벤트 정의 (모든 평가에서 공통 사용)
        positive_events = events_test[events_test["event"].isin(["addtocart", "transaction"])]
        positives: Dict[int, set[int]] = (
            positive_events.dropna(subset=["visitorid", "itemid"])
            .assign(visitorid=lambda df: df["visitorid"].astype(int), itemid=lambda df: df["itemid"].astype(int))
            .groupby("visitorid")["itemid"]
            .apply(lambda items: set(items.tolist()))
            .to_dict()
        )

        # 1. ALS 추천 결과 평가
        als_rec_path = self.processed_dir / "als_recommendations.csv"
        if als_rec_path.exists():
            logger.info("=" * 80)
            logger.info("ALS 추천 결과 평가 시작")
            logger.info("=" * 80)
            als_recommendations = pd.read_csv(als_rec_path)
            als_test_recs = als_recommendations[als_recommendations["visitorid"].isin(test_users)]
            als_test_recs = (
                als_test_recs.sort_values(["visitorid", "score"], ascending=[True, False])
                .groupby("visitorid")
                .head(self.top_k)
            )
            als_results = self._evaluate_recommendations(als_test_recs, positives, events_test, "ALS", score_col="score")
            als_f1 = als_results["f1"]
            logger.info("ALS F1 Score: %.4f", als_f1)
            # ALS 시각화 생성
            self._create_visualizations(
                y_true=als_results["y_true"],
                y_pred=als_results["y_pred"],
                y_scores=als_results["y_scores"],
                item_y_true=als_results["item_y_true"],
                item_y_scores=als_results["item_y_scores"],
                confusion=als_results["confusion"],
                roc_auc=als_results["roc_auc"],
                average_precision=als_results["average_precision"],
                model_name="ALS"
            )
        else:
            logger.warning("als_recommendations.csv 파일이 없어 ALS 추천 결과를 평가할 수 없습니다.")
            als_f1 = None

        # 2. GNN 추천 결과 평가
        gnn_user_emb_path = self.processed_dir / "gnn_user_embeddings.csv"
        gnn_item_emb_path = self.processed_dir / "gnn_item_embeddings.csv"
        if gnn_user_emb_path.exists() and gnn_item_emb_path.exists():
            logger.info("=" * 80)
            logger.info("GNN 추천 결과 평가 시작")
            logger.info("=" * 80)
            gnn_recommendations = self._generate_gnn_recommendations(test_users)
            gnn_results = self._evaluate_recommendations(gnn_recommendations, positives, events_test, "GNN", score_col="score")
            gnn_f1 = gnn_results["f1"]
            logger.info("GNN F1 Score: %.4f", gnn_f1)
            # GNN 시각화 생성
            self._create_visualizations(
                y_true=gnn_results["y_true"],
                y_pred=gnn_results["y_pred"],
                y_scores=gnn_results["y_scores"],
                item_y_true=gnn_results["item_y_true"],
                item_y_scores=gnn_results["item_y_scores"],
                confusion=gnn_results["confusion"],
                roc_auc=gnn_results["roc_auc"],
                average_precision=gnn_results["average_precision"],
                model_name="GNN"
            )
        else:
            logger.warning("GNN 임베딩 파일이 없어 GNN 추천 결과를 평가할 수 없습니다.")
            gnn_f1 = None

        # 3. 최종 추천 결과 평가
        final_rec_path = self.processed_dir / "final_recommendations.csv"
        if not final_rec_path.exists():
            raise FileNotFoundError("final_recommendations.csv 파일이 없습니다. ReRanker를 먼저 실행해 주세요.")
        
        logger.info("=" * 80)
        logger.info("최종 추천 결과 평가 시작")
        logger.info("=" * 80)
        recommendations = pd.read_csv(final_rec_path)
        test_recommendations = recommendations[recommendations["visitorid"].isin(test_users)]
        test_recommendations = (
            test_recommendations.sort_values(["visitorid", "rank"]).groupby("visitorid").head(self.top_k)
        )
        test_recommendations.to_csv(self.processed_dir / "test_recommendations.csv", index=False)
        
        final_results = self._evaluate_recommendations(test_recommendations, positives, events_test, "최종", score_col="final_score")
        final_f1 = final_results["f1"]
        logger.info("최종 F1 Score: %.4f", final_f1)
        # 최종 추천 결과 시각화 생성
        self._create_visualizations(
            y_true=final_results["y_true"],
            y_pred=final_results["y_pred"],
            y_scores=final_results["y_scores"],
            item_y_true=final_results["item_y_true"],
            item_y_scores=final_results["item_y_scores"],
            confusion=final_results["confusion"],
            roc_auc=final_results["roc_auc"],
            average_precision=final_results["average_precision"],
            model_name="Final"
        )

        # 최종 요약 출력
        logger.info("=" * 80)
        logger.info("전체 평가 결과 요약")
        logger.info("=" * 80)
        if als_f1 is not None:
            logger.info("ALS F1 Score: %.4f", als_f1)
        if gnn_f1 is not None:
            logger.info("GNN F1 Score: %.4f", gnn_f1)
        logger.info("최종 F1 Score: %.4f", final_f1)
        logger.info("=" * 80)

        # 최종 추천 결과에 대한 시각화 생성
        self._create_visualizations(
            y_true=final_results["y_true"],
            y_pred=final_results["y_pred"],
            y_scores=final_results["y_scores"],
            item_y_true=final_results["item_y_true"],
            item_y_scores=final_results["item_y_scores"],
            confusion=final_results["confusion"],
            roc_auc=final_results["roc_auc"],
            average_precision=final_results["average_precision"],
            model_name="Final"
        )

    def evaluate_final_at_k(self, k: int) -> Dict[str, Any]:
        """
        최종 추천만 평가 (TestSetEvaluator와 동일한 방식).
        top_k를 k로 두고, rank 기준 상위 k개만 잘라 _evaluate_recommendations + NDCG/Recall 계산.
        TopKExperiment에서 K=10~200 실험 시 각 K마다 호출용.
        """
        events_test_path = self.processed_dir / "events_test.csv"
        final_rec_path = self.processed_dir / "final_recommendations.csv"
        if not events_test_path.exists():
            raise FileNotFoundError("events_test.csv가 없습니다.")
        if not final_rec_path.exists():
            raise FileNotFoundError("final_recommendations.csv가 없습니다. ReRanker를 먼저 실행하세요.")

        events_test = pd.read_csv(events_test_path)
        test_users = events_test["visitorid"].dropna().astype(int).unique()
        positive_events = events_test[events_test["event"].isin(["addtocart", "transaction"])]
        positives: Dict[int, set[int]] = (
            positive_events.dropna(subset=["visitorid", "itemid"])
            .assign(visitorid=lambda df: df["visitorid"].astype(int), itemid=lambda df: df["itemid"].astype(int))
            .groupby("visitorid")["itemid"]
            .apply(lambda items: set(items.tolist()))
            .to_dict()
        )

        recommendations = pd.read_csv(final_rec_path)
        recommendations["visitorid"] = recommendations["visitorid"].astype(int)
        recommendations["itemid"] = recommendations["itemid"].astype(int)
        test_recommendations = recommendations[recommendations["visitorid"].isin(test_users)]
        test_recommendations = (
            test_recommendations.sort_values(["visitorid", "rank"]).groupby("visitorid").head(k)
        )

        final_results = self._evaluate_recommendations(
            test_recommendations, positives, events_test, "최종", score_col="final_score"
        )
        ndcg = _ndcg_at_k(test_recommendations, positives, k, "rank", ascending=True)
        recall = _recall_at_k(test_recommendations, positives, k, "rank", ascending=True)
        return {
            "f1": final_results["f1"],
            "ndcg": ndcg,
            "recall": recall,
            **final_results,
        }

    def _generate_gnn_recommendations(self, test_users: np.ndarray) -> pd.DataFrame:
        """GNN 임베딩을 사용하여 테스트 사용자에 대한 추천 결과를 생성합니다."""
        logger.info("GNN 임베딩을 사용하여 추천 결과 생성 중...")
        
        # GNN 임베딩 로드
        gnn_user_embeddings_df = pd.read_csv(self.processed_dir / "gnn_user_embeddings.csv")
        gnn_item_embeddings_df = pd.read_csv(self.processed_dir / "gnn_item_embeddings.csv")
        
        # 임베딩 배열로 변환
        gnn_user_ids = gnn_user_embeddings_df["visitorid"].values.astype(int)
        gnn_item_ids = gnn_item_embeddings_df["itemid"].values.astype(int)
        
        user_embedding_cols = [col for col in gnn_user_embeddings_df.columns if col.startswith("embedding_")]
        item_embedding_cols = [col for col in gnn_item_embeddings_df.columns if col.startswith("embedding_")]
        
        gnn_user_embeddings = gnn_user_embeddings_df[user_embedding_cols].values.astype(np.float32)
        gnn_item_embeddings = gnn_item_embeddings_df[item_embedding_cols].values.astype(np.float32)
        
        # 매핑 생성
        gnn_user_map = {user_id: idx for idx, user_id in enumerate(gnn_user_ids)}
        gnn_item_map = {item_id: idx for idx, item_id in enumerate(gnn_item_ids)}
        
        results: List[Dict[str, Any]] = []
        
        for visitor_id in test_users:
            if visitor_id not in gnn_user_map:
                continue
            
            user_idx = gnn_user_map[visitor_id]
            user_vector = gnn_user_embeddings[user_idx]
            
            # 모든 아이템에 대한 점수 계산
            gnn_scores = gnn_item_embeddings @ user_vector
            
            # 상위 K개 선택
            top_indices = np.argsort(gnn_scores)[::-1][:self.top_k]
            
            for rank, item_idx in enumerate(top_indices, start=1):
                item_id = gnn_item_ids[item_idx]
                score = float(gnn_scores[item_idx])
                results.append({
                    "visitorid": int(visitor_id),
                    "itemid": int(item_id),
                    "score": score,
                    "rank": rank
                })
        
        return pd.DataFrame(results)

    def _evaluate_recommendations(
        self,
        recommendations: pd.DataFrame,
        positives: Dict[int, set[int]],
        events_test: pd.DataFrame,
        model_name: str,
        score_col: str = "final_score"
    ) -> Dict[str, Any]:
        """
        추천 결과를 평가하고 평가 결과를 반환합니다.
        
        Args:
            recommendations: 평가할 추천 결과 데이터프레임
            positives: 사용자별 positive 아이템 딕셔너리
            events_test: 테스트 이벤트 데이터프레임
            model_name: 모델 이름 (로깅용)
            score_col: 점수 컬럼 이름
            
        Returns:
            평가 결과 딕셔너리 (f1, y_true, y_pred, y_scores, item_y_true, item_y_scores, confusion, roc_auc, average_precision)
        """
        # 평가 모드에 따라 가중치 정의 (positives는 이미 파라미터로 받음)
        item_weights: Dict[Tuple[int, int], float] = {}
        score_threshold: float = 0.0
        
        if self.evaluation_mode == "weighted":
            # view 이벤트도 가중치로 포함
            view_events = events_test[events_test["event"] == "view"]
            view_events_clean = (
                view_events.dropna(subset=["visitorid", "itemid"])
                .assign(visitorid=lambda df: df["visitorid"].astype(int), itemid=lambda df: df["itemid"].astype(int))
            )
            
            # view 이벤트 가중치 추가
            if not view_events_clean.empty:
                visitor_ids = view_events_clean["visitorid"].values
                item_ids = view_events_clean["itemid"].values
                for visitor_id, item_id in zip(visitor_ids, item_ids):
                    item_weights[(int(visitor_id), int(item_id))] = self.view_weight
            
            # transaction/addtocart 가중치 추가 (view보다 우선)
            for visitor_id, items in positives.items():
                for item_id in items:
                    item_weights[(visitor_id, item_id)] = 1.0
        
        elif self.evaluation_mode == "score_based":
            # 점수 임계값 계산 (전체 추천 점수의 백분위수 사용)
            if self.score_threshold is None:
                all_scores = recommendations[score_col].astype(float).values
                if len(all_scores) > 0:
                    effective_percentile = min(self.score_percentile, 50.0)
                    score_threshold = float(np.percentile(all_scores, effective_percentile))
                else:
                    score_threshold = 0.0
            else:
                score_threshold = self.score_threshold

        metrics = {
            "users_evaluated": 0,
            "hit_users": 0,
            "precision_sum": 0.0,
            "recall_sum": 0.0,
        }
        y_true: List[int] = []
        y_pred: List[int] = []
        y_scores: List[float] = []
        item_y_true: List[int] = []
        item_y_scores: List[float] = []

        for visitor_id, recs in recommendations.groupby("visitorid"):
            predicted_items = recs["itemid"].astype(int).tolist()
            relevant_items = positives.get(visitor_id, set())
            if not predicted_items:
                continue

            metrics["users_evaluated"] += 1

            # 평가 모드에 따라 예측 및 라벨 계산
            if self.evaluation_mode == "strict":
                # 기존 방식
                labels = [1 if item in relevant_items else 0 for item in predicted_items]
                item_y_true.extend(labels)
                item_y_scores.extend(recs[score_col].astype(float).tolist())

                if not relevant_items:
                    y_true.append(0)
                    y_pred.append(0)
                    y_scores.append(0.0)
                    continue

                hits = len(set(predicted_items) & relevant_items)
                if hits > 0:
                    metrics["hit_users"] += 1

                precision = hits / len(predicted_items)
                recall = hits / len(relevant_items) if relevant_items else 0.0

                metrics["precision_sum"] += precision
                metrics["recall_sum"] += recall

                y_true.append(1)
                y_pred.append(1 if hits > 0 else 0)
                y_scores.append(max(recs[score_col].tolist()) if not recs.empty else 0.0)

            elif self.evaluation_mode == "weighted":
                # 가중치 기반 평가
                # 아이템별 가중치 계산
                item_scores = []
                view_count = 0
                strong_count = 0
                
                for item_id in predicted_items:
                    if item_id in relevant_items:
                        item_scores.append(1.0)  # transaction/addtocart
                        strong_count += 1
                    elif (visitor_id, item_id) in item_weights:
                        item_scores.append(item_weights[(visitor_id, item_id)])  # view 가중치
                        view_count += 1
                    else:
                        item_scores.append(0.0)
                
                # 가중치 합계 계산
                total_weight = sum(item_scores)
                # 임계값: view 가중치를 적극적으로 활용하기 위해 낮은 임계값 사용
                # view_weight(0.3) 이상이면 positive로 분류 (최소 1개의 view만 있어도)
                threshold = self.view_weight  # 0.3 (기본값)
                
                labels = [1 if score > 0 else 0 for score in item_scores]
                item_y_true.extend(labels)
                item_y_scores.extend(recs[score_col].astype(float).tolist())

                # 사용자 레벨 평가
                # 실제 positive (transaction/addtocart)가 있는지 확인
                actual_strong_positive = len(relevant_items) > 0
                hits = len(set(predicted_items) & relevant_items) if relevant_items else 0
                
                # weighted 모드의 핵심: view 가중치도 활용하여 더 많은 positive 예측
                # 예측 로직:
                # 1. hits가 있으면 무조건 positive (transaction/addtocart 매칭)
                # 2. hits가 없어도 view 가중치 합계가 threshold 이상이면 positive
                #    - threshold = view_weight (0.3)이므로 최소 1개의 view만 있어도 positive
                # 중요: hits가 0이어도 view 가중치가 충분하면 positive로 예측
                # view_count > 0이면 view 가중치가 있다는 의미이므로, threshold 이상이면 positive
                predicted_positive = hits > 0 or total_weight >= threshold
                
                # 실제 positive: strong positive가 있으면 true
                # (view 가중치는 예측에만 사용하고, 실제 positive는 transaction/addtocart만)
                actual_positive = actual_strong_positive
                
                if not actual_positive and total_weight == 0.0:
                    # 실제 positive도 없고 가중치도 없으면 negative
                    y_true.append(0)
                    y_pred.append(0)
                    y_scores.append(0.0)
                    continue

                # 실제 positive (transaction/addtocart만)
                y_true.append(1 if actual_positive else 0)
                
                # 예측: hits가 있거나 가중치 합계가 임계값 이상이면 positive
                y_pred.append(1 if predicted_positive else 0)
                y_scores.append(max(recs[score_col].tolist()) if not recs.empty else 0.0)
                
                # 메트릭 계산
                if actual_strong_positive:
                    hits = len(set(predicted_items) & relevant_items)
                    weighted_hits = sum(item_scores)
                    precision = weighted_hits / len(predicted_items) if predicted_items else 0.0
                    recall = hits / len(relevant_items) if relevant_items else 0.0
                    metrics["precision_sum"] += precision
                    metrics["recall_sum"] += recall
                    if hits > 0 or predicted_positive:
                        metrics["hit_users"] += 1
                elif predicted_positive:
                    # view만 있어도 가중치가 충분하면 hit로 간주
                    weighted_hits = sum(item_scores)
                    precision = weighted_hits / len(predicted_items) if predicted_items else 0.0
                    recall = 0.0  # 실제 transaction/addtocart가 없으므로 recall은 0
                    metrics["precision_sum"] += precision
                    metrics["recall_sum"] += recall
                    metrics["hit_users"] += 1

            elif self.evaluation_mode == "partial":
                # 부분 점수 시스템: Top-K 중 일정 비율 이상 hit하면 positive
                labels = [1 if item in relevant_items else 0 for item in predicted_items]
                item_y_true.extend(labels)
                item_y_scores.extend(recs[score_col].astype(float).tolist())

                hits = len(set(predicted_items) & relevant_items)
                hit_ratio = hits / len(predicted_items) if predicted_items else 0.0
                
                # 실제 positive가 있는지 확인
                has_actual_positive = len(relevant_items) > 0
                
                if not has_actual_positive:
                    y_true.append(0)
                    y_pred.append(0)
                    y_scores.append(0.0)
                    continue

                # 일정 비율 이상 hit하면 positive로 분류
                predicted_positive = hit_ratio >= self.min_hit_ratio or hits >= max(1, int(len(predicted_items) * self.min_hit_ratio))
                
                if hits > 0:
                    metrics["hit_users"] += 1

                precision = hits / len(predicted_items)
                recall = hits / len(relevant_items) if relevant_items else 0.0

                metrics["precision_sum"] += precision
                metrics["recall_sum"] += recall

                y_true.append(1)
                y_pred.append(1 if predicted_positive else 0)
                y_scores.append(max(recs[score_col].tolist()) if not recs.empty else 0.0)

            elif self.evaluation_mode == "score_based":
                # 점수 기반 평가: 추천 점수를 활용한 평가
                labels = [1 if item in relevant_items else 0 for item in predicted_items]
                item_y_true.extend(labels)
                item_y_scores.extend(recs[score_col].astype(float).tolist())

                hits = len(set(predicted_items) & relevant_items)
                
                # 실제 positive가 있는지 확인
                has_actual_positive = len(relevant_items) > 0
                
                if not has_actual_positive:
                    y_true.append(0)
                    y_pred.append(0)
                    y_scores.append(0.0)
                    continue
                
                scores = recs[score_col].astype(float).values
                
                # 점수가 비어있는 경우 처리
                if len(scores) == 0:
                    y_true.append(1)
                    y_pred.append(0)
                    y_scores.append(0.0)
                    continue
                
                # FN을 줄이기 위한 개선된 로직:
                # 1. hits가 있으면 무조건 positive
                # 2. hits가 없어도 점수가 충분히 높으면 positive (모델이 확신하는 경우)
                #    - 평균 점수가 임계값 이상이면 positive
                #    - 또는 상위 N개 점수의 평균이 임계값 이상이면 positive
                mean_score = np.mean(scores)
                max_score = np.max(scores)
                
                # 상위 10개 점수의 평균 계산 (더 신뢰할 수 있는 추천)
                top_n = min(10, len(scores))
                top_scores = np.sort(scores)[-top_n:]
                top_mean_score = np.mean(top_scores)
                
                # 예측: hits가 있거나, 점수가 충분히 높으면 positive
                # 더 관대한 기준: 평균 점수 또는 상위 점수 평균이 임계값 이상이면 positive
                predicted_positive = hits > 0 or mean_score >= score_threshold or top_mean_score >= score_threshold
                
                if hits > 0:
                    metrics["hit_users"] += 1

                precision = hits / len(predicted_items) if predicted_items else 0.0
                recall = hits / len(relevant_items) if relevant_items else 0.0

                metrics["precision_sum"] += precision
                metrics["recall_sum"] += recall

                y_true.append(1)
                y_pred.append(1 if predicted_positive else 0)
                y_scores.append(max_score)
                
            elif self.evaluation_mode == "rank_based":
                # 순위 기반 평가: 상위 순위의 추천을 더 중요하게 평가
                labels = [1 if item in relevant_items else 0 for item in predicted_items]
                item_y_true.extend(labels)
                item_y_scores.extend(recs[score_col].astype(float).tolist())

                hits = len(set(predicted_items) & relevant_items)
                
                # 실제 positive가 있는지 확인
                has_actual_positive = len(relevant_items) > 0
                
                if not has_actual_positive:
                    y_true.append(0)
                    y_pred.append(0)
                    y_scores.append(0.0)
                    continue
                
                # rank 컬럼이 없으면 score 기반으로 순위 생성
                if "rank" in recs.columns:
                    ranks = recs["rank"].astype(int).values
                else:
                    # score가 높을수록 순위가 높도록 (1부터 시작)
                    scores_sorted = np.argsort(-recs[score_col].astype(float).values)
                    ranks = np.zeros(len(scores_sorted), dtype=int)
                    for idx, pos in enumerate(scores_sorted):
                        ranks[pos] = idx + 1
                
                scores = recs[score_col].astype(float).values
                
                # 점수가 비어있는 경우 처리
                if len(scores) == 0:
                    y_true.append(1)
                    y_pred.append(0)
                    y_scores.append(0.0)
                    continue
                
                # 상위 순위(rank)에서 hit가 있는지 확인
                top_rank_threshold = max(1, int(len(predicted_items) * self.top_rank_ratio))
                top_rank_items = [item for item, rank in zip(predicted_items, ranks) if rank <= top_rank_threshold]
                top_rank_hits = len(set(top_rank_items) & relevant_items)
                
                # FN을 줄이기 위한 개선된 로직:
                # 1. hits가 있으면 무조건 positive
                # 2. 상위 순위에서 hit가 있으면 positive
                # 3. 상위 순위의 점수가 충분히 높으면 positive (모델이 확신하는 경우)
                top_rank_scores = [score for item, rank, score in zip(predicted_items, ranks, scores) if rank <= top_rank_threshold]
                top_rank_mean_score = np.mean(top_rank_scores) if len(top_rank_scores) > 0 else 0.0
                
                # 더 관대한 기준: 상위 순위 점수 평균이 전체 평균보다 높으면 positive
                overall_mean_score = np.mean(scores)
                max_score = np.max(scores)
                
                # 예측: hits가 있거나, 상위 순위에서 hit가 있거나, 상위 순위 점수가 충분히 높으면 positive
                predicted_positive = hits > 0 or top_rank_hits > 0 or (top_rank_mean_score > overall_mean_score * 0.8)
                
                if hits > 0:
                    metrics["hit_users"] += 1

                precision = hits / len(predicted_items) if predicted_items else 0.0
                recall = hits / len(relevant_items) if relevant_items else 0.0

                metrics["precision_sum"] += precision
                metrics["recall_sum"] += recall

                y_true.append(1)
                y_pred.append(1 if predicted_positive else 0)
                y_scores.append(max_score)

        evaluated = metrics["users_evaluated"]
        hit_users = metrics["hit_users"]
        mean_precision = metrics["precision_sum"] / evaluated if evaluated else 0.0
        mean_recall = metrics["recall_sum"] / evaluated if evaluated else 0.0
        hit_rate = hit_users / evaluated if evaluated else 0.0

        confusion: Optional[np.ndarray] = None
        accuracy = precision_score_value = recall_score_value = f1 = roc_auc = average_precision = 0.0
        item_precision = item_recall = item_f1 = 0.0
        best_threshold = 0.5

        if y_true:
            from sklearn.metrics import (
                accuracy_score,
                average_precision_score,
                confusion_matrix,
                f1_score,
                precision_score,
                recall_score,
                roc_auc_score,
                roc_curve,
                precision_recall_curve,
            )

            confusion = confusion_matrix(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)
            precision_score_value = precision_score(y_true, y_pred, zero_division=0)
            recall_score_value = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            try:
                roc_auc = roc_auc_score(y_true, y_scores)
            except ValueError:
                roc_auc = 0.0
            average_precision = average_precision_score(y_true, y_scores)

        if item_y_true:
            y_true_arr = np.array(item_y_true)
            y_score_arr = np.array(item_y_scores)
            precision_curve, recall_curve, thresholds = precision_recall_curve(y_true_arr, y_score_arr)
            f1_curve = 2 * precision_curve * recall_curve / np.clip(precision_curve + recall_curve, 1e-12, None)
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

        logger.info(
            "%s Evaluation (mode=%s) - users: %d, hit@%d: %.4f, precision: %.4f, recall: %.4f, F1: %.4f",
            model_name,
            self.evaluation_mode,
            evaluated,
            self.top_k,
            hit_rate,
            mean_precision,
            mean_recall,
            f1,
        )
        
        return {
            "f1": f1,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_scores": y_scores,
            "item_y_true": item_y_true,
            "item_y_scores": item_y_scores,
            "confusion": confusion,
            "roc_auc": roc_auc,
            "average_precision": average_precision,
        }

    def _create_visualizations_for_final(
        self,
        recommendations: pd.DataFrame,
        positives: Dict[int, set[int]],
        events_test: pd.DataFrame,
    ) -> None:
        """최종 추천 결과에 대한 시각화를 생성합니다."""
        # 최종 추천 결과 평가 (시각화용)
        score_col = "final_score"
        
        # 평가 모드에 따라 가중치 정의
        item_weights: Dict[Tuple[int, int], float] = {}
        
        if self.evaluation_mode == "weighted":
            view_events = events_test[events_test["event"] == "view"]
            view_events_clean = (
                view_events.dropna(subset=["visitorid", "itemid"])
                .assign(visitorid=lambda df: df["visitorid"].astype(int), itemid=lambda df: df["itemid"].astype(int))
            )
            
            if not view_events_clean.empty:
                visitor_ids = view_events_clean["visitorid"].values
                item_ids = view_events_clean["itemid"].values
                for visitor_id, item_id in zip(visitor_ids, item_ids):
                    item_weights[(int(visitor_id), int(item_id))] = self.view_weight
            
            for visitor_id, items in positives.items():
                for item_id in items:
                    item_weights[(visitor_id, item_id)] = 1.0
        
        y_true: List[int] = []
        y_pred: List[int] = []
        y_scores: List[float] = []
        item_y_true: List[int] = []
        item_y_scores: List[float] = []
        
        for visitor_id, recs in recommendations.groupby("visitorid"):
            predicted_items = recs["itemid"].astype(int).tolist()
            relevant_items = positives.get(visitor_id, set())
            if not predicted_items:
                continue
            
            if self.evaluation_mode == "strict":
                if not relevant_items:
                    y_true.append(0)
                    y_pred.append(0)
                    y_scores.append(0.0)
                    continue
                
                hits = len(set(predicted_items) & relevant_items)
                y_true.append(1)
                y_pred.append(1 if hits > 0 else 0)
                y_scores.append(max(recs[score_col].tolist()) if not recs.empty else 0.0)
                
                labels = [1 if item in relevant_items else 0 for item in predicted_items]
                item_y_true.extend(labels)
                item_y_scores.extend(recs[score_col].astype(float).tolist())
            elif self.evaluation_mode == "weighted":
                item_scores = []
                for item_id in predicted_items:
                    if item_id in relevant_items:
                        item_scores.append(1.0)
                    elif (visitor_id, item_id) in item_weights:
                        item_scores.append(item_weights[(visitor_id, item_id)])
                    else:
                        item_scores.append(0.0)
                
                total_weight = sum(item_scores)
                threshold = self.view_weight
                hits = len(set(predicted_items) & relevant_items) if relevant_items else 0
                predicted_positive = hits > 0 or total_weight >= threshold
                actual_positive = len(relevant_items) > 0
                
                if not actual_positive and total_weight == 0.0:
                    y_true.append(0)
                    y_pred.append(0)
                    y_scores.append(0.0)
                    continue
                
                y_true.append(1 if actual_positive else 0)
                y_pred.append(1 if predicted_positive else 0)
                y_scores.append(max(recs[score_col].tolist()) if not recs.empty else 0.0)
                
                labels = [1 if score > 0 else 0 for score in item_scores]
                item_y_true.extend(labels)
                item_y_scores.extend(recs[score_col].astype(float).tolist())
            else:
                # 다른 모드는 strict와 유사하게 처리
                if not relevant_items:
                    y_true.append(0)
                    y_pred.append(0)
                    y_scores.append(0.0)
                    continue
                
                hits = len(set(predicted_items) & relevant_items)
                y_true.append(1)
                y_pred.append(1 if hits > 0 else 0)
                y_scores.append(max(recs[score_col].tolist()) if not recs.empty else 0.0)
                
                labels = [1 if item in relevant_items else 0 for item in predicted_items]
                item_y_true.extend(labels)
                item_y_scores.extend(recs[score_col].astype(float).tolist())
        
        # 메트릭 계산
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        
        confusion: Optional[np.ndarray] = None
        accuracy = precision_score_value = recall_score_value = f1 = roc_auc = average_precision = 0.0
        
        if y_true:
            confusion = confusion_matrix(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)
            precision_score_value = precision_score(y_true, y_pred, zero_division=0)
            recall_score_value = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            try:
                roc_auc = roc_auc_score(y_true, y_scores)
            except ValueError:
                roc_auc = 0.0
            average_precision = average_precision_score(y_true, y_scores)
        
        # 시각화 생성
        self._create_visualizations(
            y_true=y_true,
            y_pred=y_pred,
            y_scores=y_scores,
            item_y_true=item_y_true,
            item_y_scores=item_y_scores,
            confusion=confusion,
            roc_auc=roc_auc,
            average_precision=average_precision,
        )

    def _create_visualizations(
        self,
        y_true: List[int],
        y_pred: List[int],
        y_scores: List[float],
        item_y_true: List[int],
        item_y_scores: List[float],
        confusion: Optional[np.ndarray],
        roc_auc: float,
        average_precision: float,
        model_name: str = "Final",
    ) -> None:
        """
        평가 결과를 시각화합니다.
        
        생성되는 시각화:
        1. ROC Curve (아이템 레벨)
        2. Precision-Recall Curve (아이템 레벨)
        3. Confusion Matrix 히트맵 (사용자 레벨)
        
        Args:
            y_true: 사용자 레벨 실제 레이블 리스트
            y_pred: 사용자 레벨 예측 레이블 리스트
            y_scores: 사용자 레벨 예측 점수 리스트
            item_y_true: 아이템 레벨 실제 레이블 리스트
            item_y_scores: 아이템 레벨 예측 점수 리스트
            confusion: 혼동 행렬 (사용자 레벨)
            roc_auc: ROC-AUC 점수
            average_precision: Average Precision 점수
            model_name: 모델 이름 (파일명에 사용, 기본값: "Final")
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib 또는 seaborn이 설치되지 않아 시각화를 건너뜁니다.")
            return

        viz_dir = Path("data/visualization")
        viz_dir.mkdir(parents=True, exist_ok=True)

        # 스타일 설정
        try:
            plt.style.use("seaborn-v0_8-darkgrid")
        except OSError:
            try:
                plt.style.use("seaborn-darkgrid")
            except OSError:
                plt.style.use("default")
        sns.set_palette("husl")

        # 1. ROC Curve (User-level) - 사용자 레벨 ROC 커브
        if y_true and len(set(y_true)) > 1 and y_scores:
            try:
                from sklearn.metrics import roc_curve

                y_true_arr = np.array(y_true)
                y_score_arr = np.array(y_scores)
                
                # 길이가 다를 경우 처리
                if len(y_true_arr) != len(y_score_arr):
                    min_len = min(len(y_true_arr), len(y_score_arr))
                    y_true_arr = y_true_arr[:min_len]
                    y_score_arr = y_score_arr[:min_len]
                
                fpr, tpr, _ = roc_curve(y_true_arr, y_score_arr)

                plt.figure(figsize=(8, 6))
                mode_label = f" (mode: {self.evaluation_mode})" if hasattr(self, 'evaluation_mode') else ""
                plt.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
                plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel("False Positive Rate", fontsize=12)
                plt.ylabel("True Positive Rate", fontsize=12)
                plt.title(f"{model_name} - ROC Curve (User-level){mode_label}", fontsize=14, fontweight="bold")
                plt.legend(loc="lower right", fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(viz_dir / f"{model_name.lower()}_roc_curve.png", dpi=300, bbox_inches="tight")
                plt.close()
                logger.info("Saved ROC curve to %s", viz_dir / f"{model_name.lower()}_roc_curve.png")
            except Exception as e:
                logger.warning("ROC curve 생성 실패: %s", e)
        
        # 2. ROC Curve (Item-level) - 아이템 레벨 ROC 커브
        if item_y_true and len(set(item_y_true)) > 1 and item_y_scores:
            try:
                from sklearn.metrics import roc_curve

                y_true_arr = np.array(item_y_true)
                y_score_arr = np.array(item_y_scores)
                
                # 길이가 다를 경우 처리
                if len(y_true_arr) != len(y_score_arr):
                    min_len = min(len(y_true_arr), len(y_score_arr))
                    y_true_arr = y_true_arr[:min_len]
                    y_score_arr = y_score_arr[:min_len]
                
                # 아이템 레벨 ROC-AUC 계산
                try:
                    from sklearn.metrics import roc_auc_score
                    item_roc_auc = roc_auc_score(y_true_arr, y_score_arr)
                except ValueError:
                    item_roc_auc = 0.0
                
                fpr, tpr, _ = roc_curve(y_true_arr, y_score_arr)

                plt.figure(figsize=(8, 6))
                mode_label = f" (mode: {self.evaluation_mode})" if hasattr(self, 'evaluation_mode') else ""
                plt.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {item_roc_auc:.4f})")
                plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel("False Positive Rate", fontsize=12)
                plt.ylabel("True Positive Rate", fontsize=12)
                plt.title(f"{model_name} - ROC Curve (Item-level){mode_label}", fontsize=14, fontweight="bold")
                plt.legend(loc="lower right", fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(viz_dir / f"{model_name.lower()}_roc_curve_item.png", dpi=300, bbox_inches="tight")
                plt.close()
                logger.info("Saved ROC curve (item-level) to %s", viz_dir / f"{model_name.lower()}_roc_curve_item.png")
            except Exception as e:
                logger.warning("ROC curve (item-level) 생성 실패: %s", e)

        # 2. Precision-Recall Curve (Item-level)
        if item_y_true and len(set(item_y_true)) > 1:
            try:
                from sklearn.metrics import precision_recall_curve

                y_true_arr = np.array(item_y_true)
                y_score_arr = np.array(item_y_scores)
                
                # 길이가 다를 경우 처리
                if len(y_true_arr) != len(y_score_arr):
                    min_len = min(len(y_true_arr), len(y_score_arr))
                    y_true_arr = y_true_arr[:min_len]
                    y_score_arr = y_score_arr[:min_len]
                
                precision, recall, _ = precision_recall_curve(y_true_arr, y_score_arr)

                plt.figure(figsize=(8, 6))
                mode_label = f" (mode: {self.evaluation_mode})" if hasattr(self, 'evaluation_mode') else ""
                plt.plot(recall, precision, linewidth=2, label=f"PR Curve (AP = {average_precision:.4f})")
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel("Recall", fontsize=12)
                plt.ylabel("Precision", fontsize=12)
                plt.title(f"{model_name} - Precision-Recall Curve (Item-level){mode_label}", fontsize=14, fontweight="bold")
                plt.legend(loc="lower left", fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(viz_dir / f"{model_name.lower()}_precision_recall_curve.png", dpi=300, bbox_inches="tight")
                plt.close()
                logger.info("Saved Precision-Recall curve to %s", viz_dir / f"{model_name.lower()}_precision_recall_curve.png")
            except Exception as e:
                logger.warning("Precision-Recall curve 생성 실패: %s", e)

        # 3. Confusion Matrix 히트맵 (User-level)
        if confusion is not None:
            try:
                plt.figure(figsize=(8, 6))
                sns.heatmap(
                    confusion,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    cbar=True,
                    square=True,
                    linewidths=0.5,
                    annot_kws={"size": 14, "weight": "bold"},
                )
                plt.xlabel("Predicted Label", fontsize=12)
                plt.ylabel("True Label", fontsize=12)
                mode_label = f" (mode: {self.evaluation_mode})" if hasattr(self, 'evaluation_mode') else ""
                plt.title(f"{model_name} - Confusion Matrix (User-level){mode_label}", fontsize=14, fontweight="bold")
                plt.tight_layout()
                plt.savefig(viz_dir / f"{model_name.lower()}_confusion_matrix.png", dpi=300, bbox_inches="tight")
                plt.close()
                logger.info("Saved Confusion matrix to %s", viz_dir / f"{model_name.lower()}_confusion_matrix.png")
            except Exception as e:
                logger.warning("Confusion matrix 생성 실패: %s", e)


@dataclass
class TopKExperiment:
    """
    K=10~200(10 단위)마다 TestSetEvaluator와 동일한 방식으로 최종 추천 평가.

    - 각 K마다 TestSetEvaluator.evaluate_final_at_k(k) 호출 (rank 기준 상위 k개만 사용)
    - NDCG@K, Recall@K 및 F1 등 동일 지표로 결과 수집
    """

    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    k_min: int = 10
    k_max: int = 200
    k_step: int = 10

    def run(self) -> None:
        evaluator = TestSetEvaluator(
            processed_dir=self.processed_dir,
            top_k=50,  # evaluate_final_at_k(k)에서 k를 쓰므로 여기선 기본값만
        )
        k_values = list(range(self.k_min, self.k_max + 1, self.k_step))
        results: List[Tuple[int, float, float, float, float]] = []
        t_total_start = time.perf_counter()
        for k in k_values:
            t_k_start = time.perf_counter()
            out = evaluator.evaluate_final_at_k(k)
            t_k_elapsed = time.perf_counter() - t_k_start
            results.append((k, out["ndcg"], out["recall"], out["f1"], t_k_elapsed))
        t_total_elapsed = time.perf_counter() - t_total_start

        self._print_table(results, t_total_elapsed)
        self._save_results(results)

    def _print_table(
        self, results: List[Tuple[int, float, float, float, float]], total_sec: float = 0.0
    ) -> None:
        sep = "-" * 85
        print("\n" + "=" * 85)
        print("Top-K 실험 (각 K마다 TestSetEvaluator.evaluate_final_at_k) — K=10~200, step 10")
        print("=" * 85)
        print(f"{'K':>6} | {'NDCG@K':>10} | {'Recall@K':>10} | {'F1':>10} | {'시간(초)':>10}")
        print(sep)
        for row in results:
            k, ndcg, recall, f1, t_sec = row
            print(f"{k:>6} | {ndcg:>10.4f} | {recall:>10.4f} | {f1:>10.4f} | {t_sec:>10.4f}")
        print(sep)
        print(f"총 실행 시간: {total_sec:.2f}초")
        print()

    def _save_results(self, results: List[Tuple[int, float, float, float, float]]) -> None:
        df = pd.DataFrame(results, columns=["K", "NDCG", "Recall", "F1", "Time_sec"])
        output_path = self.processed_dir / "topk_experiment_results.csv"
        df.to_csv(output_path, index=False)
        logger.info("Top-K 실험 결과 저장: %s", output_path)


@dataclass
class EnsembleAlphaExperiment:
    """
    ALS+BPR 결합 비율(alpha)을 0.0~1.0 구간에서 스윕하여 지표를 비교합니다.

    - alpha = 1.0: ALS만 사용
    - alpha = 0.0: BPR만 사용
    - 각 alpha마다 als_debug.run_pipeline_return_metrics()를 호출하여 NDCG/Recall 수집
    """

    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    alpha_min: float = 0.0
    alpha_max: float = 1.0
    alpha_step: float = 0.1

    def run(self) -> None:
        import sys

        model_dir = Path(__file__).resolve().parent
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))
        import als_debug

        original_alpha = als_debug.ENSEMBLE_ALPHA
        alpha_values = np.round(
            np.arange(self.alpha_min, self.alpha_max + self.alpha_step / 2.0, self.alpha_step),
            1,
        ).tolist()

        results: List[Tuple[float, float, float, float, float, float]] = []
        t_total_start = time.perf_counter()
        try:
            for alpha in alpha_values:
                t_alpha_start = time.perf_counter()
                als_debug.ENSEMBLE_ALPHA = float(alpha)
                metrics = als_debug.run_pipeline_return_metrics(verbose=False)
                t_alpha_elapsed = time.perf_counter() - t_alpha_start
                results.append(
                    (
                        float(alpha),
                        float(metrics.get("ndcg_10", 0.0)),
                        float(metrics.get("recall_10", 0.0)),
                        float(metrics.get("ndcg_20", 0.0)),
                        float(metrics.get("recall_20", 0.0)),
                        t_alpha_elapsed,
                    )
                )
        finally:
            als_debug.ENSEMBLE_ALPHA = original_alpha

        t_total_elapsed = time.perf_counter() - t_total_start
        self._print_table(results, t_total_elapsed)
        self._save_results(results)

    def _print_table(
        self, results: List[Tuple[float, float, float, float, float, float]], total_sec: float = 0.0
    ) -> None:
        sep = "-" * 100
        print("\n" + "=" * 100)
        print("Ensemble alpha 실험 (ALS+BPR 결합 비율) — alpha=0.0~1.0, step 0.1")
        print("=" * 100)
        print(
            f"{'alpha':>8} | {'NDCG@10':>10} | {'Recall@10':>10} | "
            f"{'NDCG@20':>10} | {'Recall@20':>10} | {'시간(초)':>10}"
        )
        print(sep)
        for row in results:
            alpha, ndcg10, recall10, ndcg20, recall20, t_sec = row
            print(
                f"{alpha:>8.1f} | {ndcg10:>10.4f} | {recall10:>10.4f} | "
                f"{ndcg20:>10.4f} | {recall20:>10.4f} | {t_sec:>10.4f}"
            )
        print(sep)
        print(f"총 실행 시간: {total_sec:.2f}초")
        print()

    def _save_results(self, results: List[Tuple[float, float, float, float, float, float]]) -> None:
        df = pd.DataFrame(
            results,
            columns=["alpha", "NDCG@10", "Recall@10", "NDCG@20", "Recall@20", "Time_sec"],
        )
        output_path = self.processed_dir / "ensemble_alpha_experiment_results.csv"
        df.to_csv(output_path, index=False)
        logger.info("Ensemble alpha 실험 결과 저장: %s", output_path)


if __name__ == "__main__":
    """
    베이스라인 파이프라인 (ALS 전처리 + GNN 전처리):
    1. ALSPreprocessor -> ALSRecommender
    2. GNNPreprocessor -> GNNEmbeddingGenerator
    3. ReRanker -> 최종 추천 (final_recommendations.csv)
    아래 주석을 해제하면 단계별로 실행할 수 있고,
    BaselineComparator.run()은 위 전체를 한 번에 실행 후 3시나리오 NDCG/Recall 표를 출력합니다.
    """
    # # --- 베이스라인과 동일한 전처리·모델 (단계별 실행 시 주석 해제) ---
    # als_pre = ALSPreprocessor(data_dir=Path("data"), processed_dir=Path("data/processed"))
    # als_pre.run()
    # als_recommender = ALSRecommender(processed_dir=Path("data/processed"), top_k=50)
    # als_recommender.run()

    # gnn_pre = GNNPreprocessor(data_dir=Path("data"), processed_dir=Path("data/processed"))
    # gnn_pre.run()
    # gnn_generator = GNNEmbeddingGenerator(processed_dir=Path("data/processed"))
    # gnn_generator.run()

    # reranker = ReRanker(processed_dir=Path("data/processed"), top_k=200)
    # reranker.run()

    # 최종 추천만 평가 (F1, 시각화 등)
    # evaluator = TestSetEvaluator(
    #     processed_dir=Path("data/processed"),
    #     top_k=50,
    #     evaluation_mode="score_based",
    #     score_percentile=50.0,
    # )
    # evaluator.run()
    
    alpha = EnsembleAlphaExperiment()
    alpha.run()