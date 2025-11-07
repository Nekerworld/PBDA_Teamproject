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
EVENT_WEIGHT: Dict[str, float] = {"view": 1.0, "addtocart": 3.0, "transaction": 5.0}


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


@dataclass
class IsolationForestPreprocessor:
    data_dir: Path = field(default_factory=lambda: Path("data"))
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    random_state: int = 42
    test_size: float = 0.2
    contamination: str | float = "auto"
    n_estimators: int = 200

    def run(self) -> None:
        logger.info("Loading raw datasets…")
        datasets = self._load_raw()

        logger.info("Building item-level feature table for anomaly detection…")
        feature_table = self._build_feature_table(datasets)

        logger.info("Splitting feature table into train(%.0f%%)/test(%.0f%%)…", (1 - self.test_size) * 100, self.test_size * 100)
        train_features, test_features = train_test_split(
            feature_table,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True,
        )

        logger.info("Training Isolation Forest on %d items…", len(train_features))
        isolation_forest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        isolation_forest.fit(train_features)

        logger.info("Scoring train/test items for anomaly detection…")
        train_predictions = isolation_forest.predict(train_features)
        test_predictions = isolation_forest.predict(test_features)

        train_inliers = train_features[train_predictions == 1]
        train_outliers = train_features[train_predictions == -1]
        test_inliers = test_features[test_predictions == 1]

        logger.info(
            "Detected %d anomalies in train set (%.2f%%).",
            len(train_outliers),
            (len(train_outliers) / max(len(train_features), 1)) * 100,
        )

        self._save_results(
            datasets,
            train_inliers=train_inliers,
            test_inliers=test_inliers,
            train_outliers=train_outliers,
        )

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

        item_properties["numeric_value"] = item_properties["value"].map(_extract_numeric)
        item_properties["value_length"] = item_properties["value"].astype(str).str.len()

        property_agg = (
            item_properties.groupby("itemid").agg(
                property_count=("property", "count"),
                unique_property_count=("property", pd.Series.nunique),
                numeric_value_count=("numeric_value", lambda s: s.notna().sum()),
                numeric_value_mean=("numeric_value", "mean"),
                numeric_value_std=("numeric_value", "std"),
                value_length_mean=("value_length", "mean"),
            )
        )

        category_rows = (
            item_properties[item_properties["property"] == "categoryid"]["itemid"].to_frame().copy()
        )
        category_rows["value"] = item_properties[item_properties["property"] == "categoryid"]["value"].values
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

        feature_table = event_agg.join(property_agg, how="left")
        feature_table = feature_table.join(category_features, how="left")

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

        events_train_clean = events[events["itemid"].isin(train_inlier_ids)]
        events_test = events[events["itemid"].isin(test_inlier_ids)]
        events_train_outliers = events[events["itemid"].isin(train_outlier_ids)]

        item_props_train_clean = item_properties[item_properties["itemid"].isin(train_inlier_ids)]
        item_props_test = item_properties[item_properties["itemid"].isin(test_inlier_ids)]
        item_props_train_outliers = item_properties[item_properties["itemid"].isin(train_outlier_ids)]

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
    factors: int = 64
    regularization: float = 0.1
    iterations: int = 20
    alpha: float = 1.0
    top_k: int = 20
    random_state: int = 42

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

        self._save_outputs(recommendations, users, items, model)
        logger.info("Saved ALS recommendations to %s", self.processed_dir.resolve())

    def _prepare_interactions(self, events: DataFrame) -> DataFrame:
        events = events.copy()
        events["event_weight"] = events["event"].map(EVENT_WEIGHT).fillna(1.0)
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
        for user_index, user_id in enumerate(users):
            user_interactions = user_item_matrix[user_index]
            item_indices, scores = model.recommend(
                user_index,
                user_interactions,
                N=self.top_k,
                filter_already_liked_items=True,
            )
            for rank, (item_index, score) in enumerate(zip(item_indices, scores), start=1):
                if item_index >= len(items) or item_index < 0:
                    continue
                records.append(
                    {
                        "visitorid": user_id,
                        "itemid": items[item_index],
                        "score": float(score),
                        "rank": rank,
                    }
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
    embedding_dim: int = 64
    layers: int = 2
    epochs: int = 5
    batch_size: int = 2048
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

        datasets = self._load_inputs()
        graph_data = self._prepare_graph_structures(datasets)

        adjacency = self._build_normalized_adjacency(
            graph_data["edge_sources"], graph_data["edge_targets"], graph_data["total_nodes"]
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
    top_k: int = 20
    candidate_k: int = 100
    mmr_lambda: float = 0.7
    random_seed: Optional[int] = 42

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

        item_id_array = np.array(als_item_ids)

        for user_index, visitor_id in enumerate(als_user_ids):
            if visitor_id not in gnn_user_map:
                continue

            als_user_vector = als_user_factors[user_index]
            gnn_user_vector = gnn_user_embeddings[gnn_user_map[visitor_id]]

            valid_pairs = [
                (als_idx, gnn_item_map[item_id])
                for als_idx, item_id in enumerate(item_id_array)
                if item_id in gnn_item_map
            ]
            if not valid_pairs:
                continue

            als_indices = np.array([pair[0] for pair in valid_pairs], dtype=np.int64)
            gnn_indices = np.array([pair[1] for pair in valid_pairs], dtype=np.int64)

            als_item_subset = als_item_factors[als_indices]
            gnn_item_subset = gnn_item_embeddings[gnn_indices]
            item_ids_valid = item_id_array[als_indices]

            als_scores = als_user_vector @ als_item_subset.T
            gnn_scores = gnn_item_subset @ gnn_user_vector

            combined_scores, weight_a = self._combine_scores(als_scores, gnn_scores)

            reranked_items = self._apply_mmr(
                combined_scores,
                gnn_item_subset,
                item_ids_valid,
            )

            for rank, (item_id, score) in enumerate(reranked_items, start=1):
                final_score = score * (1.5 if availability.get(item_id, 0) == 1 else 1.0)
                results.append(
                    {
                        "visitorid": visitor_id,
                        "itemid": item_id,
                        "rank": rank,
                        "mmr_score": float(score),
                        "final_score": float(final_score),
                        "als_weight": float(weight_a),
                        "gnn_weight": float(1 - weight_a),
                    }
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

        weight_a = float(np.random.rand())
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

    def _apply_mmr(
        self,
        scores: np.ndarray,
        item_embeddings: np.ndarray,
        item_ids: np.ndarray,
    ) -> List[Tuple[int, float]]:
        candidate_indices = np.argsort(scores)[::-1][: self.candidate_k]
        candidate_scores = scores[candidate_indices]
        candidate_embeddings = item_embeddings[candidate_indices]
        candidate_ids = item_ids[candidate_indices]

        selected: List[int] = []
        mmr_scores: Dict[int, float] = {}

        while len(selected) < min(self.top_k, len(candidate_indices)):
            mmr_values = []
            for idx, score in enumerate(candidate_scores):
                if idx in selected:
                    mmr_values.append(-np.inf)
                    continue

                candidate_embedding = candidate_embeddings[idx]
                if not selected:
                    diversity_penalty = 0.0
                else:
                    selected_embeddings = candidate_embeddings[selected]
                    cos_sim = self._cosine_similarity(candidate_embedding, selected_embeddings)
                    diversity_penalty = float(np.max(cos_sim))

                mmr_score = self.mmr_lambda * score - (1 - self.mmr_lambda) * diversity_penalty
                mmr_values.append(mmr_score)

            best_local_idx = int(np.argmax(mmr_values))
            if best_local_idx in selected:
                break

            selected.append(best_local_idx)
            mmr_scores[best_local_idx] = float(mmr_values[best_local_idx])

        final_items: List[Tuple[int, float]] = []
        for idx in selected:
            item_id = int(candidate_ids[idx])
            final_items.append((item_id, mmr_scores[idx]))

        final_items.sort(key=lambda x: x[1], reverse=True)
        return final_items

    def _cosine_similarity(self, vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        vector_norm = np.linalg.norm(vector) + 1e-8
        matrix_norm = np.linalg.norm(matrix, axis=1) + 1e-8
        return np.dot(matrix, vector) / (matrix_norm * vector_norm)


@dataclass
class TestSetEvaluator:
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    top_k: int = 20

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

        for visitor_id, recs in test_recommendations.groupby("visitorid"):
            predicted_items = recs["itemid"].astype(int).tolist()
            relevant_items = positives.get(visitor_id, set())
            if not predicted_items:
                continue

            metrics["users_evaluated"] += 1

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

        if y_true:
            from sklearn.metrics import (
                accuracy_score,
                average_precision_score,
                confusion_matrix,
                f1_score,
                precision_score,
                recall_score,
                roc_auc_score,
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

        logger.info(
            "Test Evaluation - users: %d, hit@%d: %.4f, precision: %.4f, recall: %.4f",
            evaluated,
            self.top_k,
            hit_rate,
            mean_precision,
            mean_recall,
        )
        if confusion is not None:
            logger.info("Confusion Matrix:\\n%s", confusion)
            logger.info(
                "Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f, ROC-AUC: %.4f, Average Precision: %.4f",
                accuracy,
                precision_score_value,
                recall_score_value,
                f1,
                roc_auc,
                average_precision,
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

