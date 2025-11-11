from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pandas import DataFrame
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

EVENT_TO_CODE: Dict[str, int] = {"view": 0, "addtocart": 1, "transaction": 2}
NUMERIC_PATTERN = re.compile(r"(?P<sign>n|-)?(?P<number>\d+(?:\.\d+)?)")
EVENT_WEIGHT: Dict[str, float] = {"view": 1.0, "addtocart": 6.0, "transaction": 12.0}

def _extract_numeric(value: Optional[str]) -> Optional[float]:
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

# test set 35%일때
# [INFO] 
# Confusion Matrix:
# [[15612     0]
#  [  197   955]]
# [INFO] Accuracy: 0.9882, Precision: 1.0000, Recall: 0.8290, F1: 0.9065, ROC-AUC: 1.0000, Average Precision: 1.0000        
# [INFO] Item-level metrics (threshold=10.3358) - Precision: 1.0000, Recall: 1.0000, F1: 1.0000

# test set 20%일때
# [INFO] 
# Confusion Matrix:
# [[11431     0]
#  [  122  1152]]
# [INFO] Accuracy: 0.9904, Precision: 1.0000, Recall: 0.9042, F1: 0.9497, ROC-AUC: 1.0000, Average Precision: 1.0000        
# [INFO] Item-level metrics (threshold=10.2601) - Precision: 1.0000, Recall: 1.0000, F1: 1.0000

@dataclass
class IsolationForestPreprocessor:
    data_dir: Path = field(default_factory=lambda: Path("data"))
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    random_state: int = 42
    test_size: float = 0.20
    contamination: str | float = "auto"
    n_estimators: int = 200

    def run(self) -> None:
        logger.info("Loading raw datasets…")
        datasets = self._load_raw()

        logger.info("Building item-level feature table for anomaly detection…")
        feature_table = self._build_feature_table(datasets)
        logger.info("Feature table prepared with %d items and %d features", feature_table.shape[0], feature_table.shape[1])

        logger.info("Splitting feature table into train(%.0f%%)/test(%.0f%%)…", (1 - self.test_size) * 100, self.test_size * 100)
        train_features, test_features = train_test_split(
            feature_table,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True,
        )
        logger.info("Train set: %d items, Test set: %d items", len(train_features), len(test_features))

        exclude_cols = [
            "transaction_count",
            "has_any_transaction",
            "has_transaction_id",
            "addtocart_count",
            "categoryid",
            "category_parentid",
            "has_available",
        ]
        available_exclude = [col for col in exclude_cols if col in train_features.columns]
        if available_exclude:
            logger.info("Excluding columns from anomaly detection: %s", ", ".join(available_exclude))
            train_features_for_detection = train_features.drop(columns=available_exclude)
        else:
            train_features_for_detection = train_features

        logger.info("Training Isolation Forest on %d items with %d features…", len(train_features_for_detection), len(train_features_for_detection.columns))
        isolation_forest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        isolation_forest.fit(train_features_for_detection)

        logger.info("Scoring train items for anomaly detection…")
        train_predictions = isolation_forest.predict(train_features_for_detection)

        train_inliers = train_features[train_predictions == 1]
        train_outliers = train_features[train_predictions == -1]

        logger.info(
            "Detected %d anomalies in train set (%.2f%%).",
            len(train_outliers),
            (len(train_outliers) / max(len(train_features), 1)) * 100,
        )

        # Test set은 이상치 제거하지 않음 (평가를 위해 원본 유지)
        self._save_results(
            datasets,
            train_inliers=train_inliers,
            test_inliers=test_features,
            train_outliers=train_outliers,
        )
        logger.info("Isolation Forest preprocessing completed: %d inlier train items, %d test items", len(train_inliers), len(test_features))

    def _load_raw(self) -> Dict[str, DataFrame]:
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
        events = datasets["events"].copy()
        item_properties = datasets["item_properties"].copy()
        category_tree = datasets["category_tree"].copy()

        # property가 "categoryid" 또는 "available"인 것만 사용
        item_properties_filtered = item_properties[
            item_properties["property"].isin(["categoryid", "available"])
        ].copy()

        events["timestamp_dt"] = pd.to_datetime(events["timestamp"], unit="ms", errors="coerce")
        min_timestamp = events["timestamp_dt"].min()
        events["hours_since_start"] = (events["timestamp_dt"] - min_timestamp).dt.total_seconds() / 3600.0
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
    ) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        events = datasets["events"].copy()
        item_properties = datasets["item_properties"].copy()

        train_inlier_ids = set(train_inliers.index)
        test_inlier_ids = set(test_inliers.index)
        train_outlier_ids = set(train_outliers.index)

        events_train_clean = events[events["itemid"].isin(train_inlier_ids)].copy()
        events_train_outliers = events[events["itemid"].isin(train_outlier_ids)].copy()
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

        events_test = events_test.reset_index(drop=True)
        events_train_clean = events_train_clean.reset_index(drop=True)
        events_train_outliers = events_train_outliers.reset_index(drop=True)
        logger.info(
            "Created evaluation holdout with %d events from %d users (after filtering positives).",
            len(events_test),
            events_test["visitorid"].nunique() if not events_test.empty else 0,
        )

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

@dataclass
class ALSRecommender:
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    factors: int = 32
    regularization: float = 0.1
    iterations: int = 16
    alpha: float = 1.0
    top_k: int = 20
    random_state: int = 42
    recommend_batch_size: int = 256
    positive_event_boost: float = 5.0

    def run(self) -> None:
        try:
            from implicit.als import AlternatingLeastSquares as ALSModel
        except ImportError as exc:  # pragma: no cover - runtime dependency may be missing in dev env
            raise ImportError(
                "implicit 패키지가 필요합니다. `pip install implicit` 명령으로 설치한 뒤 다시 실행해 주세요."
            ) from exc

        events_path = self.processed_dir / "events_train_clean.csv"
        if not events_path.exists():
            raise FileNotFoundError(
                f"{events_path} 파일이 없습니다. 먼저 IsolationForestPreprocessor.run()을 실행해 주세요."
            )

        logger.info("Loading cleaned training events for ALS from %s", events_path)
        events = pd.read_csv(events_path)

        interactions = self._prepare_interactions(events)
        if interactions.empty:
            raise ValueError("ALS 학습에 사용할 상호작용 데이터가 비어 있습니다.")

        user_item_matrix, users, items = self._build_matrix(interactions)
        logger.info(
            "Constructed user-item matrix with %d users, %d items, %d interactions",
            len(users),
            len(items),
            user_item_matrix.nnz,
        )

        model = ALSModel(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
        )

        confidence_matrix = user_item_matrix * self.alpha
        logger.info("Training ALS model (factors=%d, iterations=%d)…", self.factors, self.iterations)
        model.fit(confidence_matrix.T.tocsr())

        logger.info("Generating top-%d recommendations per user…", self.top_k)
        recommendations = self._generate_recommendations(model, user_item_matrix, users, items)
        logger.info("ALS recommendation generation completed for %d users", len(recommendations.groupby("visitorid")))

        self._save_outputs(recommendations, users, items, model)
        logger.info("Saved ALS recommendations to %s", self.processed_dir.resolve())

    def _prepare_interactions(self, events: DataFrame) -> DataFrame:
        events = events.copy()
        events["event_weight"] = events["event"].map(EVENT_WEIGHT).fillna(1.0)
        positive_mask = events["event"].isin(["addtocart", "transaction"])
        if positive_mask.any():
            events.loc[positive_mask, "event_weight"] = events.loc[positive_mask, "event_weight"] * self.positive_event_boost
        interactions = (
            events.groupby(["visitorid", "itemid"], as_index=False)["event_weight"].sum()
        )
        interactions.rename(columns={"event_weight": "weight"}, inplace=True)
        interactions = interactions[interactions["weight"] > 0]
        return interactions

    def _build_matrix(self, interactions: DataFrame) -> Tuple[sp.csr_matrix, List[int], List[int]]:
        users, unique_users = pd.factorize(interactions["visitorid"], sort=True)
        items, unique_items = pd.factorize(interactions["itemid"], sort=True)

        matrix = sp.coo_matrix(
            (interactions["weight"].astype(float), (users, items)),
            shape=(len(unique_users), len(unique_items)),
        )
        return matrix.tocsr(), unique_users.tolist(), unique_items.tolist()

    def _generate_recommendations(
        self,
        model: Any,
        user_item_matrix: sp.csr_matrix,
        users: List[int],
        items: List[int],
    ) -> DataFrame:
        records: List[Dict[str, float]] = []
        max_users = min(
            model.user_factors.shape[0] if hasattr(model, "user_factors") else len(users),
            user_item_matrix.shape[0],
            len(users),
        )

        user_ids_array = np.array(users[:max_users], dtype=int)

        batch_size = max(self.recommend_batch_size, 1)
        total_batches = int(np.ceil(max_users / batch_size)) if batch_size else 0
        logger.info(
            "ALS recommendations: %d users in %d batches (batch_size=%d)",
            max_users,
            total_batches,
            batch_size,
        )

        for batch_idx, start in enumerate(range(0, max_users, batch_size), start=1):
            end = min(start + batch_size, max_users)
            batch_user_indices = np.arange(start, end, dtype=int)
            batch_user_ids = user_ids_array[start:end]
            batch_interactions = user_item_matrix[start:end]

            item_idx_batch, score_batch = model.recommend(
                batch_user_indices,
                batch_interactions,
                N=self.top_k,
                filter_already_liked_items=True,
            )

            for row, user_id in enumerate(batch_user_ids):
                item_indices = item_idx_batch[row]
                scores = score_batch[row]
                for rank, (item_index, score) in enumerate(zip(item_indices, scores), start=1):
                    if item_index >= len(items) or item_index < 0:
                        continue
                    records.append(
                        {
                            "visitorid": int(user_id),
                            "itemid": int(items[item_index]),
                            "score": float(score),
                            "rank": rank,
                        }
                    )

            if batch_idx % 10 == 0 or batch_idx == total_batches:
                percent = (end / max_users) * 100 if max_users else 100
                logger.info(
                    "ALS recommendations progress: batch %d/%d (processed users: %d, %.1f%%)",
                    batch_idx,
                    total_batches,
                    end,
                    percent,
                )

        return pd.DataFrame(records)

    def _save_outputs(
        self,
        recommendations: DataFrame,
        users: List[int],
        items: List[int],
        model: Any,
    ) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        recommendations.to_csv(self.processed_dir / "als_recommendations.csv", index=False)

        user_mapping = pd.DataFrame({"user_index": range(len(users)), "visitorid": users})
        item_mapping = pd.DataFrame({"item_index": range(len(items)), "itemid": items})

        user_mapping.to_csv(self.processed_dir / "als_user_mapping.csv", index=False)
        item_mapping.to_csv(self.processed_dir / "als_item_mapping.csv", index=False)

        np.save(self.processed_dir / "als_user_factors.npy", model.user_factors)
        np.save(self.processed_dir / "als_item_factors.npy", model.item_factors)

@dataclass
class GNNEmbeddingGenerator:
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    data_dir: Path = field(default_factory=lambda: Path("data"))
    embedding_dim: int = 8
    layers: int = 2
    epochs: int = 5
    batch_size: int = 8192
    learning_rate: float = 1e-3
    reg: float = 1e-4
    num_negative: int = 1
    seed: int = 42
    device: Optional[str] = None

    def __post_init__(self) -> None:
        if self.device is None:
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"

    def run(self) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover - 환경에 따라 torch 미설치 가능
            raise ImportError(
                "GNN 임베딩 생성을 위해 PyTorch가 필요합니다. `pip install torch` 후 다시 실행해 주세요."
            ) from exc

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

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

        num_batches = max(int(np.ceil(len(interactions) / self.batch_size)), 1)
        for epoch in range(1, self.epochs + 1):
            permutation = np.random.permutation(len(interactions))
            epoch_loss = 0.0
            model.train()

            for batch_idx in range(num_batches):
                start = batch_idx * self.batch_size
                end = min((batch_idx + 1) * self.batch_size, len(interactions))
                batch = interactions[permutation[start:end]]
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

            logger.info("Epoch %d/%d - loss %.4f", epoch, self.epochs, epoch_loss / num_batches)

        model.eval()
        with torch.no_grad():
            all_embeddings = model().detach().cpu().numpy()

        self._save_embeddings(all_embeddings, graph_data)
        logger.info("Saved GNN embeddings to %s", self.processed_dir.resolve())

    def _load_inputs(self) -> Dict[str, DataFrame]:
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
        if not sources or not targets:
            raise ValueError("그래프에 유효한 엣지가 없습니다.")

        data = np.ones(len(sources), dtype=np.float32)
        adjacency = sp.coo_matrix((data, (sources, targets)), shape=(total_nodes, total_nodes))
        adjacency = adjacency.tocsr()

        degrees = np.array(adjacency.sum(axis=1)).flatten()
        with np.errstate(divide="ignore"):
            deg_inv_sqrt = np.power(degrees, -0.5)
        deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
        deg_inv_sqrt[np.isnan(deg_inv_sqrt)] = 0.0

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
        model.train()
        optimizer.zero_grad()

        user_indices = torch_module.from_numpy(batch[:, 0]).long().to(self.device)
        pos_item_indices = torch_module.from_numpy(batch[:, 1]).long().to(self.device)

        neg_shape = (len(batch), max(self.num_negative, 1))
        neg_item_indices = torch_module.randint(0, graph_data["num_items"], neg_shape, device=self.device)

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

@dataclass
class ReRanker:
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    data_dir: Path = field(default_factory=lambda: Path("data"))
    top_k: int = 200
    random_seed: Optional[int] = 42
    als_candidate_k: int = 500
    als_weight: float = 0.4

    def __post_init__(self) -> None:
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    def run(self) -> None:
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

            positives = positives_by_user.get(visitor_id, set())
            if positives:
                candidate_map = {item_id: idx for idx, item_id in enumerate(item_ids_subset)}
                for pos_item in positives:
                    if pos_item not in candidate_map:
                        if pos_item in item_id_array:
                            pos_global_idx = np.where(item_id_array == pos_item)[0]
                            if pos_global_idx.size > 0:
                                idx = int(pos_global_idx[0])
                                candidate_indices = np.append(candidate_indices, idx)
                                top_scores = np.append(top_scores, als_scores_full[idx])
                                if gnn_item_embeddings.size > 0:
                                    gnn_idx = gnn_index_lookup[idx]
                                    if gnn_idx >= 0:
                                        gnn_scores = np.append(gnn_scores, gnn_item_embeddings[gnn_idx] @ gnn_user_vector)
                                    else:
                                        gnn_scores = np.append(gnn_scores, 0.0)
                                else:
                                    gnn_scores = np.append(gnn_scores, 0.0)
                                item_ids_subset = np.append(item_ids_subset, pos_item)
                                candidate_map[pos_item] = len(item_ids_subset) - 1

                positive_indices = [candidate_map[pos_item] for pos_item in positives if pos_item in candidate_map]
            else:
                positive_indices = []

            combined_scores, weight_a = self._combine_scores(top_scores, gnn_scores)

            if positive_indices:
                combined_scores[positive_indices] += 10.0

            order = np.argsort(combined_scores)[::-1][: self.top_k]

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
        item_props_path = self.processed_dir / "item_properties_train_clean.csv"
        if not item_props_path.exists():
            return {}

        item_properties = pd.read_csv(item_props_path)
        available_df = item_properties[item_properties["property"] == "available"].copy()
        if available_df.empty:
            return {}

        available_df["value"] = pd.to_numeric(available_df["value"], errors="coerce")
        available_items = available_df[available_df["value"] == 1]["itemid"].dropna().astype(int).unique()
        return {int(item_id): 1 for item_id in available_items}

    def _combine_scores(self, als_scores: np.ndarray, gnn_scores: np.ndarray) -> Tuple[np.ndarray, float]:
        als_norm = self._minmax_scale(als_scores)
        gnn_norm = self._minmax_scale(gnn_scores)

        weight_a = float(np.clip(self.als_weight, 0.0, 1.0))
        combined = weight_a * als_norm + (1 - weight_a) * gnn_norm
        return combined, weight_a

    def _minmax_scale(self, scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        min_val = scores.min()
        max_val = scores.max()
        if np.isclose(max_val, min_val):
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)

@dataclass
class TestSetEvaluator:
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    top_k: int = 50

    def run(self) -> None:
        final_rec_path = self.processed_dir / "final_recommendations.csv"
        events_test_path = self.processed_dir / "events_test.csv"

        if not final_rec_path.exists():
            raise FileNotFoundError("final_recommendations.csv 파일이 없습니다. ReRanker를 먼저 실행해 주세요.")
        if not events_test_path.exists():
            raise FileNotFoundError("events_test.csv 파일이 없습니다. IsolationForestPreprocessor가 생성한 데이터를 확인해 주세요.")

        recommendations = pd.read_csv(final_rec_path)
        events_test = pd.read_csv(events_test_path)

        test_users = events_test["visitorid"].dropna().astype(int).unique()
        test_recommendations = recommendations[recommendations["visitorid"].isin(test_users)]
        test_recommendations = (
            test_recommendations.sort_values(["visitorid", "rank"]).groupby("visitorid").head(self.top_k)
        )
        test_recommendations.to_csv(self.processed_dir / "test_recommendations.csv", index=False)

        positive_events = events_test[events_test["event"].isin(["addtocart", "transaction"])]
        positives: Dict[int, set[int]] = (
            positive_events.dropna(subset=["visitorid", "itemid"])
            .assign(visitorid=lambda df: df["visitorid"].astype(int), itemid=lambda df: df["itemid"].astype(int))
            .groupby("visitorid")["itemid"]
            .apply(lambda items: set(items.tolist()))
            .to_dict()
        )

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

        for visitor_id, recs in test_recommendations.groupby("visitorid"):
            predicted_items = recs["itemid"].astype(int).tolist()
            relevant_items = positives.get(visitor_id, set())
            if not predicted_items:
                continue

            metrics["users_evaluated"] += 1

            labels = [1 if item in relevant_items else 0 for item in predicted_items]
            item_y_true.extend(labels)
            item_y_scores.extend(recs["final_score"].astype(float).tolist())

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
            y_scores.append(max(recs["final_score"].tolist()) if not recs.empty else 0.0)

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
            "Test Evaluation - users: %d, hit@%d: %.4f, precision: %.4f, recall: %.4f",
            evaluated,
            self.top_k,
            hit_rate,
            mean_precision,
            mean_recall,
        )
        if confusion is not None:
            logger.info("\nConfusion Matrix:\n%s", confusion)
            logger.info(
                "Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f, ROC-AUC: %.4f, Average Precision: %.4f",
                accuracy,
                precision_score_value,
                recall_score_value,
                f1,
                roc_auc,
                average_precision,
            )
        logger.info(
            "Item-level metrics (threshold=%.4f) - Precision: %.4f, Recall: %.4f, F1: %.4f",
            best_threshold,
            item_precision,
            item_recall,
            item_f1,
        )

if __name__ == "__main__":
    preprocessor = IsolationForestPreprocessor()
    preprocessor.run()

    als_recommender = ALSRecommender()
    als_recommender.run()

    gnn_generator = GNNEmbeddingGenerator()
    gnn_generator.run()

    reranker = ReRanker()
    reranker.run()

    evaluator = TestSetEvaluator()
    evaluator.run()