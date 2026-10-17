"""One macro ranking evaluation population for all revision models."""
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import pandas as pd

try:
    from .data_protocol import (DEFAULT_PROCESSED, load_shared_split, require_artifacts,
                                require_protocol, positive_sets, stamp_artifacts)
    from .recommendation_scoring import GNNScorer, append_ranking, recommendation_frame
except ImportError:
    from data_protocol import (DEFAULT_PROCESSED, load_shared_split, require_artifacts,
                               require_protocol, positive_sets, stamp_artifacts)
    from recommendation_scoring import GNNScorer, append_ranking, recommendation_frame


def ranking_metrics(recommendations, positives, k, score_col="rank", ascending=True):
    """Missing recommendation lists contribute zero; duplicate items are invalid."""
    if k < 1:
        raise ValueError("k must be positive")
    if recommendations.duplicated(["visitorid", "itemid"]).any():
        raise ValueError("Duplicate user-item recommendations")
    if len(recommendations) and not np.isfinite(recommendations[score_col]).all():
        raise ValueError("Non-finite recommendation scores/ranks")
    ordered = recommendations.sort_values(['visitorid',score_col,'itemid'],ascending=[True,ascending,True],kind='stable')
    grouped = ordered.groupby('visitorid',sort=False).head(k).groupby('visitorid').itemid.agg(list).to_dict()
    rows = []
    for uid, relevant in sorted(positives.items()):
        if not relevant:
            continue
        group = grouped.get(uid)
        items = [] if group is None else group
        hits = len(set(items) & relevant)
        dcg = sum(1 / np.log2(rank + 2) for rank, item in enumerate(items) if item in relevant)
        idcg = sum(1 / np.log2(rank + 2) for rank in range(min(k, len(relevant))))
        precision, recall = hits / k, hits / len(relevant)
        rows.append({"visitorid": uid, "ndcg": dcg / idcg, "recall": recall,
                     "precision": precision, "hit_rate": float(hits > 0),
                     "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
                     "recommendations": len(items)})
    per_user = pd.DataFrame(rows)
    if per_user.empty:
        raise ValueError("The common evaluation cohort has no positive users")
    result = {name: float(per_user[name].mean()) for name in
              ("ndcg", "recall", "precision", "hit_rate", "f1")}
    result.update(users_evaluated=len(rows), missing_recommendation_users=int(
        (per_user.recommendations == 0).sum()))
    return result, per_user


@dataclass
class TestSetEvaluator:
    processed_dir: Path = DEFAULT_PROCESSED
    top_k: int = 50
    evaluation_mode: str = 'ranking'

    def __post_init__(self):
        self.processed_dir = Path(self.processed_dir)
        if self.evaluation_mode != 'ranking':
            raise ValueError('Only ranking evaluation is supported')

    def _cohort(self):
        _, _, _, items = load_shared_split(self.processed_dir)
        positives = positive_sets(pd.read_csv(self.processed_dir / 'events_test_original.csv'))
        users = pd.read_csv(self.processed_dir / 'evaluation_users.csv').visitorid.astype(int).tolist()
        if set(users) != set(positives):
            raise ValueError('Evaluation users and positive labels disagree')
        return users, positives, items

    def _validate_recommendations(self, frame, items):
        if not set(frame.itemid.astype(int)) <= items:
            raise ValueError('Recommendations outside the shared catalog')

    def _generate_gnn_recommendations(self, test_users, depth=None):
        require_artifacts(self.processed_dir, 'gnn')
        scorer = GNNScorer(self.processed_dir)
        records = []
        start = time.perf_counter()
        for uid in test_users:
            if uid not in scorer.users:
                continue
            append_ranking(records, uid, scorer.item_ids, scorer.scores(uid), depth or self.top_k)
        self.gnn_recommendation_seconds = time.perf_counter() - start
        return recommendation_frame(records)

    def run(self, k_values=None):
        out = self.processed_dir
        users, positives, items = self._cohort()
        protocol = require_protocol(out, verify=False)
        als = require_artifacts(out, 'als')
        gnn = require_artifacts(out, 'gnn')
        hybrid = require_artifacts(out, 'rerank')
        ks = sorted(set([self.top_k, *(k_values or [])]))
        if min(ks) < 1 or max(ks) > min(als['top_k'], hybrid['top_k']):
            raise ValueError('Requested K exceeds saved recommendation depth; regenerate recommendations')
        frames = {name: pd.read_csv(out / filename) for name, filename in [
            ('ALS', 'als_recommendations.csv'), ('BPR', 'bpr_recommendations.csv'),
            ('ALS+BPR', 'als_bpr_recommendations.csv'), ('Hybrid', 'final_recommendations.csv')]}
        frames['GNN'] = self._generate_gnn_recommendations(users, max(ks))
        gnn_seconds = self.gnn_recommendation_seconds
        frames['GNN'].to_csv(out / 'gnn_recommendations.csv', index=False)
        results, files, eval_seconds = [], ['gnn_recommendations.csv'], {}
        scopes = {
            'original': positives,
            'reference_warm': positive_sets(pd.read_csv(out / 'events_test_reference.csv')),
            'conditional': positive_sets(pd.read_csv(out / 'events_test.csv')),
        }
        for name in ['ALS', 'BPR', 'ALS+BPR', 'GNN', 'Hybrid']:
            frame = frames[name]
            self._validate_recommendations(frame, items)
            start = time.perf_counter()
            for scope, truth in scopes.items():
                if not truth:
                    continue
                for k in ks:
                    metrics, per_user = ranking_metrics(frame, truth, k)
                    filename = f"metrics_{scope}_{name.replace('+', '_')}_at_{k}_users.csv"
                    per_user.to_csv(out / filename, index=False)
                    files.append(filename)
                    results.append({'protocol_id': protocol['protocol_id'], 'reference_id':protocol['reference_id'],
                        'original_truth_id':protocol['original_truth_id'], 'scope':scope,'model': name, 'k': k, **metrics})
            eval_seconds[name] = time.perf_counter() - start
        result = pd.DataFrame(results)
        result.to_csv(out / 'topk_metrics.csv', index=False)
        primary = result.loc[(result.k == self.top_k) & (result.scope == 'original')].reset_index(drop=True)
        primary.to_csv(out / 'ranking_metrics.csv', index=False)
        timing = []
        for name in ['ALS', 'BPR', 'ALS+BPR', 'GNN', 'Hybrid']:
            if name in als['training_seconds']:
                training = als['training_seconds'][name]
                retrieval = als['recommendation_seconds'][name]
            elif name == 'GNN':
                training, retrieval = gnn['training_seconds'], gnn_seconds
            else:
                training = als['training_seconds']['ALS+BPR'] + gnn['training_seconds']
                retrieval = hybrid['retrieval_seconds']
            rerank = hybrid['rerank_seconds'] if name == 'Hybrid' else 0.
            timing.append({'protocol_id': protocol['protocol_id'], 'model': name,
                'users': len(users), 'recommendation_depth': (max(ks) if name == 'GNN' else
                    hybrid['top_k'] if name == 'Hybrid' else als['top_k']), 'training_seconds': training,
                'retrieval_seconds': retrieval, 'rerank_seconds': rerank,
                'evaluation_seconds': eval_seconds[name],
                'recommendation_seconds': retrieval + rerank})
        pd.DataFrame(timing).to_csv(out / 'timing_metrics.csv', index=False)
        candidates = pd.read_csv(out / 'hybrid_candidates.csv')
        candidate_rows = []
        for scope, truth in scopes.items():
            if truth:
                metrics, detail = ranking_metrics(candidates, truth, hybrid['candidate_k'], score_col='rank', ascending=True)
                filename=f'candidate_{scope}_users.csv'; detail.to_csv(out/filename,index=False); files.append(filename)
                candidate_rows.append({'scope':scope,'candidate_k':hybrid['candidate_k'],**metrics})
        pd.DataFrame(candidate_rows).to_csv(out/'candidate_metrics.csv',index=False)
        files += ['topk_metrics.csv', 'ranking_metrics.csv', 'timing_metrics.csv','candidate_metrics.csv']
        stamp_artifacts(out, 'evaluation', files, primary_k=self.top_k, k_values=ks,
            models=list(frames), runtime_note='Batch measurements, not isolated online latency. '
            'Training includes GNN validation selection and refit. Evaluation includes CSV writes. '
            'Hybrid timing includes ALS+BPR retrieval and candidate reranking.')
        print(primary.to_string(index=False))
        return primary

    def evaluate_final_at_k(self, k):
        _, positives, items = self._cohort()
        meta = require_artifacts(self.processed_dir, 'rerank')
        if k < 1 or k > meta['top_k']:
            raise ValueError('Requested K exceeds saved recommendation depth')
        frame = pd.read_csv(self.processed_dir / 'final_recommendations.csv')
        self._validate_recommendations(frame, items)
        return ranking_metrics(frame, positives, k)[0]



