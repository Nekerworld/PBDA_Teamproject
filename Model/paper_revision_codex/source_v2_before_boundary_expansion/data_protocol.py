"""Shared chronological, warm-start protocol for revision comments 1, 2 and 6.

One event partition is used by ALS, BPR and LightGCN. Test labels are never
inputs to preprocessing or candidate masking. Repeat recommendations are allowed.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import os
import hashlib
import json
import platform
import logging

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest

# Model/paper_revision/*.py → project root is parents[2]
ROOT = Path(os.environ.get('PBDA_PROJECT_ROOT', Path(__file__).resolve().parents[2]))
DEFAULT_DATA = ROOT / "data"
DEFAULT_PROCESSED = Path(__file__).resolve().parent / "results" / "processed_revision"
VERSION = "chronological-fixed-reference-v3"
POSITIVE_EVENTS = ("addtocart", "transaction")
IF_FEATURES = [
    "event_count", "view_count", "unique_visitors", "event_code_mean",
    "event_code_std", "first_event_hour", "last_event_hour",
    "event_duration_hours", "property_count",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")


def chronological_split(events, test_size=0.2):
    """Keep equal timestamps together; do not fall back to a random split."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between zero and one")
    if len(events) < 2 or events["timestamp"].isna().any():
        raise ValueError("At least two valid timestamped events are required")
    ordered = events.sort_values(["timestamp", "event_id"], kind="stable")
    index = min(max(int(len(ordered) * (1 - test_size)), 1), len(ordered) - 1)
    cutoff = int(ordered.iloc[index]["timestamp"])
    train = ordered.loc[ordered.timestamp < cutoff].copy()
    test = ordered.loc[ordered.timestamp >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Timestamp ties prevent a nonempty chronological partition")
    return train, test, cutoff


def dataset_stats(events, stage):
    users = int(events.visitorid.nunique())
    items = int(events.itemid.nunique())
    pairs = int(len(events[["visitorid", "itemid"]].drop_duplicates()))
    return {
        "stage": stage, "events": len(events), "users": users, "items": items,
        "unique_user_item_pairs": pairs,
        "sparsity_percent": 100 * (1 - pairs / (users * items)) if users * items else None,
        "positive_events": int(events.event.isin(POSITIVE_EVENTS).sum()),
        **{f"{kind}_events": int((events.event == kind).sum())
           for kind in ("view", "addtocart", "transaction")},
        "min_timestamp": int(events.timestamp.min()) if len(events) else None,
        "max_timestamp": int(events.timestamp.max()) if len(events) else None,
    }


def build_item_features(events, properties, category_tree):
    """All inputs must already be restricted to the training period."""
    frame = events.copy()
    frame["event_code"] = frame.event.map({"view": 0, "addtocart": 1, "transaction": 2})
    for kind in ('view', 'addtocart', 'transaction'):
        frame[kind + '_flag'] = (frame.event == kind).astype(int)
    frame["hours"] = (frame.timestamp - frame.timestamp.min()) / 3_600_000
    result = frame.groupby("itemid").agg(
        event_count=("event", "size"),
        view_count=("view_flag", "sum"),
        addtocart_count=("addtocart_flag", "sum"),
        transaction_count=("transaction_flag", "sum"),
        unique_visitors=("visitorid", "nunique"),
        event_code_mean=("event_code", "mean"), event_code_std=("event_code", "std"),
        first_event_hour=("hours", "min"), last_event_hour=("hours", "max"),
    )
    result["event_duration_hours"] = result.last_event_hour - result.first_event_hour
    selected = properties.loc[properties.property.isin(["categoryid", "available"])]
    result["property_count"] = selected.groupby("itemid").size()
    # The latest known value, never a maximum over future inventory states.
    latest = selected.sort_values("timestamp", kind="stable").drop_duplicates(
        ["itemid", "property"], keep="last")
    for prop, column in [("categoryid", "categoryid"), ("available", "has_available")]:
        values = latest.loc[latest.property == prop].set_index("itemid")["value"]
        result[column] = pd.to_numeric(values, errors="coerce")
    parent = category_tree.set_index("categoryid")["parentid"]
    result["category_parentid"] = result.categoryid.map(parent)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0)


def iterative_core(events, min_user_events=3, min_item_events=3, mode='events'):
    """Thresholds count event rows, not distinct partners; repeat to convergence."""
    result = events.copy()
    rounds = 0
    while len(result):
        counted = result if mode == 'events' else result[['visitorid','itemid']].drop_duplicates()
        if mode not in ('events', 'neighbors'):
            raise ValueError('core mode must be events or neighbors')
        users = counted.groupby("visitorid").size()
        items = counted.groupby("itemid").size()
        keep = result.visitorid.isin(users[users >= min_user_events].index) & result.itemid.isin(
            items[items >= min_item_events].index)
        if keep.all():
            break
        result = result.loc[keep].copy()
        rounds += 1
    return result, rounds


def positive_sets(events):
    if events.empty or "event" not in events.columns:
        return {}
    return {int(uid): set(map(int, rows.itemid)) for uid, rows in
            events.loc[events.event.isin(POSITIVE_EVENTS)].groupby("visitorid")}


def load_source_tables(data_dir):
    """One raw-file reading shared by single runs and condition comparisons."""
    data_dir = Path(data_dir)
    events = pd.read_csv(data_dir / "events.csv")
    events["event_id"] = np.arange(len(events), dtype=np.int64)
    for col in ["timestamp", "visitorid", "itemid"]:
        events[col] = pd.to_numeric(events[col], errors="coerce")
    valid = np.isfinite(events[["timestamp", "visitorid", "itemid"]]).all(axis=1)
    valid &= events.event.isin(["view", *POSITIVE_EVENTS])
    valid &= (events[["timestamp", "visitorid", "itemid"]].fillna(0) % 1 == 0).all(axis=1)
    invalid = events.loc[~valid].copy()
    clean = events.loc[valid].copy()
    for col in ["timestamp", "visitorid", "itemid"]:
        clean[col] = clean[col].astype("int64")
    properties = pd.concat([pd.read_csv(data_dir / f"item_properties_part{i}.csv",
                                        dtype={"property": str, "value": str})
                            for i in (1, 2)], ignore_index=True)
    for col in ["timestamp", "itemid"]:
        properties[col] = pd.to_numeric(properties[col], errors="coerce")
    properties = properties.dropna(subset=["timestamp", "itemid", "property", "value"])
    categories = pd.read_csv(data_dir / "category_tree.csv")
    return events, clean, invalid, properties, categories


@dataclass
class UnifiedPreprocessor:
    data_dir: Path = DEFAULT_DATA
    processed_dir: Path = DEFAULT_PROCESSED
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 200
    contamination: str | float = "auto"
    min_user_events: int = 3
    min_item_events: int = 3
    validation_size: float = 0.2
    use_if: bool = False
    core_mode: str = 'events'

    def run(self):
        """Write a fresh run directory; never overwrite old model artifacts."""
        self.data_dir, self.processed_dir = Path(self.data_dir), Path(self.processed_dir)
        if self.processed_dir.exists() and any(self.processed_dir.iterdir()):
            raise FileExistsError(f"Use a new empty output directory: {self.processed_dir}")
        if self.min_user_events < 1 or self.min_item_events < 1:
            raise ValueError("Core thresholds must be positive")
        events, clean, invalid, properties, categories = load_source_tables(self.data_dir)
        train, test, cutoff = chronological_split(clean, self.test_size)
        self.processed_dir.mkdir(parents=True)
        invalid.to_csv(self.processed_dir / "invalid_events.csv", index=False)
        # Create validation from RAW outer-train events. Its anomaly detector and
        # core filter are fitted independently, without outer validation information.
        val_train, val_test, val_cutoff = chronological_split(train, self.validation_size)
        self._write_partition(val_train, val_test, properties, categories, val_cutoff,
                              self.processed_dir / "validation", "validation", len(train))
        manifest = self._write_partition(train, test, properties, categories, cutoff,
                                         self.processed_dir, "test", len(events))
        return self.finalize_outer(manifest, self.processed_dir, events, clean, invalid)

    def _write_partition(self, train, test, properties, categories, cutoff, out, role, raw_count):
        out.mkdir(parents=True, exist_ok=True)
        past_props = properties.loc[(properties.timestamp < cutoff) &
                                     properties.itemid.isin(train.itemid.unique())].copy()
        features = build_item_features(train, past_props, categories)
        forest = IsolationForest(n_estimators=self.n_estimators,
                                 contamination=self.contamination,
                                 random_state=self.random_state, n_jobs=1)
        if self.use_if:
            forest.fit(features[IF_FEATURES])
            labels = forest.predict(features[IF_FEATURES])
        else:
            labels = np.ones(len(features), dtype=int)
        outlier_ids = set(features.index[labels == -1])
        train_if = train.loc[~train.itemid.isin(outlier_ids)].copy()
        final_train, core_rounds = iterative_core(train_if, self.min_user_events, self.min_item_events, self.core_mode)
        if final_train.empty:
            raise ValueError(f"All training events removed in {role}; inspect filtering settings")
        users, items = set(final_train.visitorid), set(final_train.itemid)
        raw_users, raw_items = set(train.visitorid), set(train.itemid)
        audit = test[["event_id", "visitorid", "itemid", "event", "timestamp"]].copy()
        audit["user_status"] = np.where(test.visitorid.isin(users), "known",
                                        np.where(test.visitorid.isin(raw_users), "filtered", "new"))
        audit["item_status"] = np.select(
            [test.itemid.isin(items), test.itemid.isin(outlier_ids), test.itemid.isin(raw_items)],
            ["known", "if_filtered", "core_filtered"], default="new")
        eligible = (audit.user_status == "known") & (audit.item_status == "known")
        audit["reason"] = audit.user_status + "/" + audit.item_status
        audit.loc[eligible, "reason"] = "eligible"
        final_test = test.loc[eligible].copy()
        positives = positive_sets(final_test)
        # Fixed population and truth BEFORE variant-specific filtering. The same
        # IDs and labels are used in every IF/core condition, including zero lists.
        reference = test.loc[test.visitorid.isin(raw_users) & test.itemid.isin(raw_items)].copy()
        reference_users = sorted(positive_sets(reference))
        original = test.loc[test.visitorid.isin(reference_users)].copy()
        reference = reference.loc[reference.visitorid.isin(reference_users)]
        original.to_csv(out / 'events_test_original.csv', index=False)
        reference.to_csv(out / 'events_test_reference.csv', index=False)
        original_sets = positive_sets(original)
        reference_sets = positive_sets(reference)
        retention = pd.DataFrame([{'visitorid': uid, 'train_user_retained': uid in users,
            'original_positive_items': len(original_sets[uid]),
            'reference_positive_items': len(reference_sets[uid]),
            'conditional_positive_items': len(positives.get(uid, set())),
            'original_retention': len(positives.get(uid, set())) / len(original_sets[uid]),
            'reference_retention': len(positives.get(uid, set())) / len(reference_sets[uid])}
            for uid in reference_users])
        retention.to_csv(out / 'truth_retention.csv', index=False)
        removed_if = train.loc[train.itemid.isin(outlier_ids)]
        removed_core = train_if.loc[~train_if.event_id.isin(final_train.event_id)]
        train_audit = train[["event_id", "visitorid", "itemid", "event", "timestamp"]].copy()
        train_audit["reason"] = "retained"
        train_audit.loc[train_audit.event_id.isin(removed_if.event_id), "reason"] = "isolation_forest"
        train_audit.loc[train_audit.event_id.isin(removed_core.event_id), "reason"] = "core_filter"
        train_audit.to_csv(out / "train_event_audit.csv", index=False)
        audit.to_csv(out / "test_event_audit.csv", index=False)
        audit.groupby("reason").agg(events=("event_id", "size"), users=("visitorid", "nunique"),
                                    items=("itemid", "nunique")).to_csv(out / "test_exclusion_summary.csv")
        stats = [dataset_stats(df, name) for name, df in [
            ("train_before_filter", train), ("train_if_removed", removed_if),
            ("train_after_if", train_if), ("train_core_removed", removed_core),
            ("train_final", final_train), ("test_before_filter", test),
            ("test_eligible", final_test), ("test_excluded", test.loc[~eligible])]]
        pd.DataFrame(stats).to_csv(out / "dataset_statistics.csv", index=False)
        features["if_score"] = forest.score_samples(features[IF_FEATURES]) if self.use_if else np.nan
        features["if_decision"] = forest.decision_function(features[IF_FEATURES]) if self.use_if else np.nan
        features["if_label"] = labels
        features.to_csv(out / "item_anomaly_audit.csv")
        summaries = []
        for label, group in features.groupby("if_label"):
            desc = group[IF_FEATURES].describe(percentiles=[0.25, 0.5, 0.75]).T.reset_index()
            desc.insert(0, "if_label", int(label))
            summaries.append(desc)
        pd.concat(summaries).to_csv(out / "if_feature_distributions.csv", index=False)
        # Core files are canonical, shared by every model.
        final_train.to_csv(out / "events_train_clean.csv", index=False)
        final_test.to_csv(out / "events_test.csv", index=False)
        past_props.loc[past_props.itemid.isin(items)].to_csv(out / "item_properties_train_clean.csv", index=False)
        categories.to_csv(out / "category_tree.csv", index=False)
        features.loc[features.index.isin(items)].to_csv(out / "feature_train_inliers.csv")
        pd.DataFrame({"visitorid": reference_users}).to_csv(out / "evaluation_users.csv", index=False)
        pd.DataFrame({"itemid": sorted(items)}).to_csv(out / "candidate_items.csv", index=False)
        config = asdict(self)
        config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
        files = ["events_train_clean.csv", "events_test.csv", "item_properties_train_clean.csv",
                 "category_tree.csv", "feature_train_inliers.csv", "evaluation_users.csv", "candidate_items.csv",
                 "events_test_original.csv", "events_test_reference.csv", "truth_retention.csv"]
        hashes = {name: sha256(out / name) for name in files}
        protocol_id = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        manifest = {
            "version": VERSION, "protocol_id": protocol_id, "role": role, "config": config,
            "cutoff_timestamp": cutoff, "cutoff_utc": pd.to_datetime(cutoff, unit="ms", utc=True).isoformat(),
            "actual_prefilter_train_ratio": len(train) / (len(train) + len(test)),
            "repeat_recommendations": True, "positive_events": list(POSITIVE_EVENTS),
            "core_counts": self.core_mode, "core_rounds": core_rounds, "use_if": self.use_if,
            "if_features": IF_FEATURES, "if_parameters": forest.get_params(),
            "if_offset": float(forest.offset_) if self.use_if else None, "if_item_count": len(outlier_ids),
            "if_item_percent_of_train_items": 100 * len(outlier_ids) / len(features),
            "if_removed_events": len(removed_if),
            "if_event_percent_of_prefilter_train": 100 * len(removed_if) / len(train),
            "if_event_percent_of_raw_input": 100 * len(removed_if) / raw_count,
            "test_event_coverage": len(final_test) / len(test),
            "test_positive_coverage": int(final_test.event.isin(POSITIVE_EVENTS).sum()) /
                max(int(test.event.isin(POSITIVE_EVENTS).sum()), 1),
            "evaluation_users": len(reference_users), "conditional_evaluation_users": len(positives), "hashes": hashes,
            "mean_original_truth_retention": float(retention.original_retention.mean()) if len(retention) else 0.,
            "mean_reference_truth_retention": float(retention.reference_retention.mean()) if len(retention) else 0.,
            "if_n_jobs": 1,
            "reference_id": hashlib.sha256(json.dumps({str(u):sorted(reference_sets[u]) for u in reference_users}, sort_keys=True).encode()).hexdigest(),
            "original_truth_id": hashlib.sha256(json.dumps({str(u):sorted(original_sets[u]) for u in reference_users}, sort_keys=True).encode()).hexdigest(),
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "sklearn": sklearn.__version__},
            "preprocessing_source_hash": sha256(Path(__file__)),
            "audit_hashes": {name: sha256(out / name) for name in (
                'dataset_statistics.csv', 'train_event_audit.csv', 'test_event_audit.csv',
                'test_exclusion_summary.csv', 'item_anomaly_audit.csv', 'if_feature_distributions.csv')},
            "interpretation": "Item-level anomalies; not verified bot identities. Test exclusions are not anomaly detections.",
            "category_tree_assumption": "Static taxonomy supplied without timestamps.",
        }
        health = []
        for kind in POSITIVE_EVENTS:
            before, after = int((train.event == kind).sum()), int((final_train.event == kind).sum())
            if before > 0 and after == 0:
                health.append(f'All {before} training {kind} events were removed by IF/core filtering.')
        if not positives:
            health.append('No conditional warm-start users remain. Fixed reference users are still scored; a missing recommendation is zero.')
        if manifest['test_positive_coverage'] < .1:
            health.append('Fewer than 10% of original test positive events remain in the warm-start evaluation.')
        manifest['data_quality_warnings'] = health
        for warning in health:
            logging.warning('%s: %s', role, warning)
        dump_json(out / "manifest.json", manifest)
        return manifest

    def finalize_outer(self, manifest, out, events, clean, invalid):
        """Attach raw-input provenance after the outer partition has been written."""
        out = Path(out)
        manifest["invalid_raw_events"] = len(invalid)
        manifest["raw_events"] = len(events)
        manifest["raw_valid_stats"] = dataset_stats(clean, "raw_valid")
        stats = pd.read_csv(out / "dataset_statistics.csv")
        pd.concat([pd.DataFrame([manifest["raw_valid_stats"]]), stats], ignore_index=True).to_csv(
            out / "dataset_statistics.csv", index=False)
        manifest["audit_hashes"]["dataset_statistics.csv"] = sha256(out / "dataset_statistics.csv")
        manifest["input_hashes"] = {name: sha256(self.data_dir / name) for name in (
            "events.csv", "item_properties_part1.csv", "item_properties_part2.csv", "category_tree.csv")}
        dump_json(out / "manifest.json", manifest)
        return manifest


def require_protocol(processed_dir, verify=True):
    directory = Path(processed_dir)
    path = directory / "manifest.json"
    if not path.exists():
        raise ValueError(f"Shared revision manifest missing: {path}. Run preprocess into a new directory.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION:
        raise ValueError("Legacy or incompatible preprocessing protocol")
    if verify:
        for name, expected in (manifest["hashes"] | manifest['audit_hashes']).items():
            if sha256(directory / name) != expected:
                raise ValueError(f"Shared input changed after preprocessing: {name}")
    return manifest


def load_shared_split(processed_dir=DEFAULT_PROCESSED):
    directory = Path(processed_dir)
    require_protocol(directory)
    train = pd.read_csv(directory / "events_train_clean.csv")
    test = pd.read_csv(directory / "events_test.csv")
    if train.empty:
        raise ValueError("Training events are empty")
    if len(test):
        if train.timestamp.max() >= test.timestamp.min():
            raise ValueError("Training events must strictly precede evaluation events")
        if not set(test.visitorid) <= set(train.visitorid) or not set(test.itemid) <= set(train.itemid):
            raise ValueError("Evaluation contains users/items outside training vocabulary")
    return train, test, positive_sets(test), set(map(int, train.itemid.unique()))


def stamp_artifacts(processed_dir, stage, files, **metadata):
    directory = Path(processed_dir)
    manifest = require_protocol(directory, verify=False)
    payload = {"protocol_id": manifest["protocol_id"], "stage": stage,
               "manifest_hash": sha256(directory / 'manifest.json'),
               "source_hashes": {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')
                                 if not p.name.startswith('test_')},
               "files": {name: sha256(directory / name) for name in files}, **metadata}
    if stage == 'gnn' and metadata.get('validation_protocol_id'):
        payload['validation_manifest_hash'] = sha256(directory / 'validation' / 'manifest.json')
    if stage in ('rerank', 'evaluation'):
        payload["parents"] = {name: sha256(directory / f"{name}_artifacts.json")
                              for name in (('als', 'gnn') if stage == 'rerank' else ('als', 'gnn', 'rerank'))}
    dump_json(directory / f"{stage}_artifacts.json", payload)


def require_artifacts(processed_dir, stage):
    directory = Path(processed_dir)
    manifest = require_protocol(directory, verify=False)
    path = directory / f"{stage}_artifacts.json"
    if not path.exists():
        raise ValueError(f"Missing revision provenance for {stage}; retrain this stage")
    stamp = json.loads(path.read_text(encoding="utf-8"))
    if stamp["protocol_id"] != manifest["protocol_id"]:
        raise ValueError(f"{stage} belongs to a different data partition")
    if stamp.get('manifest_hash') != sha256(directory / 'manifest.json'):
        raise ValueError(f'{stage} manifest changed; rerun this stage')
    if stamp.get('validation_manifest_hash'):
        require_protocol(directory / 'validation')
        if stamp['validation_manifest_hash'] != sha256(directory / 'validation' / 'manifest.json'):
            raise ValueError('GNN validation data changed; retrain GNN')
    for name, expected in stamp.get("parents", {}).items():
        if sha256(directory / f"{name}_artifacts.json") != expected:
            raise ValueError(f"{stage} is stale after retraining {name}; rerun {stage}")
        require_artifacts(directory, name)
    for name, expected in stamp["files"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Stale or changed {stage} artifact: {name}")
    return stamp
