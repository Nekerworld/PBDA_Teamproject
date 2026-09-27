"""Regression tests for chronological leakage, catalog coverage and fair metrics."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_protocol import (UnifiedPreprocessor, chronological_split, iterative_core,
    build_item_features, load_shared_split, require_protocol, require_artifacts, IF_FEATURES,
    positive_sets)
from ranking_evaluation import ranking_metrics, TestSetEvaluator as Evaluator
from model import ALSRecommender, GNNEmbeddingGenerator, ReRanker
import als_debug


def make_raw(folder):
    folder.mkdir()
    records = []
    for cycle in range(20):
        for user in range(1, 9):
            for item in range(1, 13):
                if user == 1 and item == 1:
                    continue  # Leave one unobserved pair so pairwise training is possible.
                event = ["view", "addtocart", "transaction"][(user + item + cycle) % 3]
                records.append((1000 + len(records), user, event, item, None))
    events = pd.DataFrame(records, columns=["timestamp", "visitorid", "event", "itemid", "transactionid"])
    # True cold-start events only in the outer test window.
    events.loc[len(events)-6:len(events)-4, "itemid"] = 999
    events.loc[len(events)-3:, "visitorid"] = 999
    events.to_csv(folder / "events.csv", index=False)
    props = pd.DataFrame([(1, i, p, v) for i in range(1, 13)
                          for p, v in [("categoryid", "1"), ("available", "1"), ("color", "blue")]],
                         columns=["timestamp", "itemid", "property", "value"])
    props.iloc[:18].to_csv(folder / "item_properties_part1.csv", index=False)
    props.iloc[18:].to_csv(folder / "item_properties_part2.csv", index=False)
    pd.DataFrame({"categoryid": [1], "parentid": [np.nan]}).to_csv(folder / "category_tree.csv", index=False)
    return events


@pytest.fixture
def bundle(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    make_raw(raw)
    UnifiedPreprocessor(raw, out, n_estimators=20).run()
    return raw, out


def test_chronological_ties_are_not_split():
    frame = pd.DataFrame({"event_id": range(10), "timestamp": [1,2,3,4,5,6,7,8,8,9]})
    train, test, cutoff = chronological_split(frame)
    assert train.timestamp.max() < test.timestamp.min()
    assert set(train.event_id).isdisjoint(test.event_id)
    assert cutoff == 8


def test_invalid_or_degenerate_time_fails():
    with pytest.raises(ValueError):
        chronological_split(pd.DataFrame({"event_id": [0,1], "timestamp": [3,3]}))
    with pytest.raises(ValueError):
        chronological_split(pd.DataFrame({"event_id": [0,1], "timestamp": [1,np.nan]}))


def test_core_filter_repeats_until_both_degrees_hold():
    events = pd.DataFrame({"visitorid": [1,1,2,2,3], "itemid": [1,2,2,3,3]})
    filtered, rounds = iterative_core(events, 2, 2)
    assert filtered.empty
    assert rounds > 1


def test_shared_split_catalog_and_accounting(bundle):
    _, out = bundle
    train, test, positives, items = load_shared_split(out)
    assert set(test.itemid) <= items
    assert set(test.visitorid) <= set(train.visitorid)
    assert train.groupby('visitorid').size().min() >= 3
    assert train.groupby('itemid').size().min() >= 3
    assert train.timestamp.max() < test.timestamp.min()
    manifest = require_protocol(out)
    assert manifest['if_features'] == IF_FEATURES
    assert manifest['actual_prefilter_train_ratio'] == .8
    audit = pd.read_csv(out/'train_event_audit.csv')
    stat = pd.read_csv(out/'dataset_statistics.csv').set_index('stage')
    assert len(audit) == stat.loc['train_before_filter','events']
    assert (audit.reason == 'retained').sum() == len(train)
    assert manifest['if_removed_events'] == (audit.reason == 'isolation_forest').sum()
    excluded = pd.read_csv(out/'test_event_audit.csv')
    assert (excluded.item_status == 'new').sum() == 3
    assert (excluded.user_status == 'new').sum() == 3
    assert (excluded.reason == 'eligible').sum() == len(test)
    reference = set(pd.read_csv(out/'evaluation_users.csv').visitorid.astype(int))
    original = set(positive_sets(pd.read_csv(out/'events_test_original.csv')))
    warm = set(positive_sets(pd.read_csv(out/'events_test_reference.csv')))
    assert reference == original == warm
    assert set(positives) <= reference
    assert require_protocol(out)['version'] == 'chronological-fixed-reference-v3'


def test_sparse_stat_uses_unique_pairs(bundle):
    _, out = bundle
    train, _, _, _ = load_shared_split(out)
    row = pd.read_csv(out/'dataset_statistics.csv').set_index('stage').loc['train_final']
    expected = 100 * (1 - len(train[['visitorid','itemid']].drop_duplicates()) /
                      (train.visitorid.nunique() * train.itemid.nunique()))
    assert row.sparsity_percent == pytest.approx(expected)


def test_future_event_and_metadata_changes_do_not_change_train(bundle, tmp_path):
    raw, out = bundle
    other = tmp_path/'other'
    make_raw(other)
    cutoff = require_protocol(out)['cutoff_timestamp']
    events = pd.read_csv(other/'events.csv')
    events.loc[events.timestamp >= cutoff, 'event'] = 'transaction'
    events.to_csv(other/'events.csv', index=False)
    props = pd.read_csv(other/'item_properties_part2.csv')
    future = pd.DataFrame([(cutoff + 1, 1, 'available', '0'),
                           (cutoff + 1, 9999, 'categoryid', '99')], columns=props.columns)
    pd.concat([props,future]).to_csv(other/'item_properties_part2.csv', index=False)
    changed = tmp_path/'changed'
    UnifiedPreprocessor(other, changed, n_estimators=20).run()
    for name in ['events_train_clean.csv','item_anomaly_audit.csv','item_properties_train_clean.csv']:
        pd.testing.assert_frame_equal(pd.read_csv(out/name),pd.read_csv(changed/name))


def test_inner_validation_fitted_before_its_own_cutoff(bundle):
    _, out = bundle
    inner = out/'validation'
    manifest = require_protocol(inner)
    props = pd.read_csv(inner/'item_properties_train_clean.csv')
    train, val, _, _ = load_shared_split(inner)
    assert props.timestamp.max() < manifest['cutoff_timestamp']
    graph_builder = GNNEmbeddingGenerator(processed_dir=inner,device='cpu')
    graph = graph_builder._prepare_graph_structures(graph_builder._load_inputs())
    pairs = {(graph['user_ids'][u], graph['item_ids'][i]) for u,i in graph['interactions']}
    assert pairs == set(train[['visitorid','itemid']].itertuples(index=False,name=None))
    assert set(val.itemid) <= set(graph['item_ids'])


def test_new_graph_pair_in_validation_is_not_in_training_graph(bundle):
    _, out = bundle
    builder = GNNEmbeddingGenerator(processed_dir=out,device='cpu')
    inputs = builder._load_inputs()
    # Simulate withholding every occurrence of one known user-item pair.
    inputs['events'] = inputs['events'].loc[~((inputs['events'].visitorid == 1) & (inputs['events'].itemid == 1))]
    graph = builder._prepare_graph_structures(inputs)
    pairs = set(map(tuple,graph['interactions']))
    if 1 in graph['user_map'] and 1 in graph['item_map']:
        assert (graph['user_map'][1],graph['item_map'][1]) not in pairs


def test_missing_recommendations_count_as_zero():
    rec = pd.DataFrame({'visitorid':[1], 'itemid':[10], 'rank':[1]})
    metrics, rows = ranking_metrics(rec, {1:{10},2:{20}}, 1)
    assert metrics['ndcg'] == .5
    assert metrics['recall'] == .5
    assert metrics['users_evaluated'] == 2
    assert metrics['missing_recommendation_users'] == 1


def test_duplicate_recommendations_rejected():
    rec = pd.DataFrame({'visitorid':[1,1], 'itemid':[10,10], 'rank':[1,2]})
    with pytest.raises(ValueError,match='Duplicate'):
        ranking_metrics(rec, {1:{10}}, 2)


def test_test_labels_cannot_change_als_candidates():
    model = SimpleNamespace(user_factors=np.array([[1.]]),item_factors=np.array([[3.],[2.],[1.]]))
    events = pd.DataFrame({'visitorid':[1,1], 'itemid':[10,20]})
    args = ([1],{1:0},{10:0,20:1,30:2},{0:10,1:20,2:30},{10,20,30})
    a,_ = als_debug.scores_to_recommendations_ensemble(*args,{1:{10}},events,model,None,3,.5)
    b,_ = als_debug.scores_to_recommendations_ensemble(*args,{1:{20}},events,model,None,3,.5)
    assert a == b == {1:[10,20,30]}


def test_legacy_and_mutated_input_fail_closed(bundle,tmp_path):
    _, out = bundle
    with pytest.raises(ValueError,match='manifest'):
        require_protocol(tmp_path/'legacy')
    with (out/'events_train_clean.csv').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError,match='changed'):
        require_protocol(out)


def test_existing_output_is_not_overwritten(bundle):
    raw,out = bundle
    with pytest.raises(FileExistsError):
        UnifiedPreprocessor(raw,out).run()


def test_end_to_end_small_models(bundle):
    raw,out = bundle
    ALSRecommender(processed_dir=out,factors=4,iterations=2,top_k=10,bpr_epochs=2,device='cpu').run()
    gnn = GNNEmbeddingGenerator(processed_dir=out,data_dir=raw,embedding_dim=4,layers=1,
        epochs=2,batch_size=256,early_stopping_patience=1,device='cpu')
    gnn.run()
    ReRanker(processed_dir=out,data_dir=raw,top_k=10,als_candidate_k=12).run()
    summary = Evaluator(processed_dir=out,top_k=5).run()
    assert summary.users_evaluated.nunique() == 1
    assert summary.missing_recommendation_users.sum() == 0
    assert set(summary.model) == {'ALS', 'BPR', 'ALS+BPR', 'GNN', 'Hybrid'}
    # Re-ranker candidates must be exactly the ALS+BPR retrieval ordering.
    from recommendation_scoring import FactorScorer, top_indices
    scorer = FactorScorer(out, .5)
    candidates = pd.read_csv(out/'hybrid_candidates.csv')
    for uid, group in candidates.groupby('visitorid'):
        expected = scorer.item_ids[top_indices(scorer.scores(uid)[2], scorer.item_ids, 12)]
        assert group.sort_values('rank').itemid.tolist() == expected.tolist()
    assert require_artifacts(out, 'evaluation')['primary_k'] == 5
    from viz import generate_figures
    figures = generate_figures(out, out/'figures_test', dpi=72)
    assert sorted(path.stem for path in figures.glob('*.pdf')) == [
        'baseline_comparison', 'candidate_recall', 'filtering_behavior', 'filtering_counts',
        'measured_runtime', 'test_coverage', 'topk_comparison', 'truth_retention']
    assert set(summary.scope) == {'original'}
    scopes = set(pd.read_csv(out/'topk_metrics.csv').scope)
    assert {'original', 'reference_warm', 'conditional'} <= scopes
    pd.testing.assert_frame_equal(pd.read_csv(figures/'ranking_metrics.csv'), summary)
    history = json.loads((out/'gnn_training_history.json').read_text())
    assert history['mode'] == 'chronological_validation_then_refit'
    assert len(history['refit_history']) == history['chosen_epochs']
    assert require_artifacts(out,'gnn')['protocol_id'] == require_artifacts(out,'als')['protocol_id']
    with pytest.raises(ValueError,match='depth'):
        Evaluator(processed_dir=out).evaluate_final_at_k(11)
    # New upstream training must invalidate recommendations even on the same split.
    stamp_path = out/'als_artifacts.json'
    stamp = json.loads(stamp_path.read_text())
    stamp['iterations'] = 123
    stamp_path.write_text(json.dumps(stamp),encoding='utf-8')
    with pytest.raises(ValueError,match='stale'):
        require_artifacts(out,'rerank')
    with pytest.raises(ValueError,match='stale'):
        generate_figures(out, out/'must_not_draw_stale')


def test_inventory_changes_rank_before_topk():
    from recommendation_scoring import hybrid_scores, top_indices
    als = np.array([0., .8, 1.])
    available = np.array([False, True, False])
    original = hybrid_scores(als, als, available, .4, 1.)
    adjusted = hybrid_scores(als, als, available, .4, 1.5)
    ids = np.array([10, 20, 30])
    assert ids[top_indices(original, ids, 1)].tolist() == [30]
    assert ids[top_indices(adjusted, ids, 1)].tolist() == [20]


def test_zero_hits_have_zero_ranking_f1():
    rec = pd.DataFrame({'visitorid': [1], 'itemid': [99], 'rank': [1], 'score': [100.]})
    metrics, _ = ranking_metrics(rec, {1: {10}}, 1)
    assert metrics['f1'] == metrics['ndcg'] == metrics['recall'] == 0.


def test_visualization_requires_auditable_results(bundle):
    from viz import generate_figures
    _, out = bundle
    pd.DataFrame({'model': ['invented']}).to_csv(out/'ranking_metrics.csv', index=False)
    with pytest.raises(ValueError, match='provenance'):
        generate_figures(out, out/'invalid_figures')


def test_latest_inventory_uses_only_training_metadata():
    events = pd.DataFrame({'timestamp': [10, 20], 'visitorid': [1, 1], 'itemid': [1, 1],
                           'event': ['view', 'view']})
    props = pd.DataFrame({'timestamp': [1, 2], 'itemid': [1, 1], 'property': ['available']*2,
                          'value': ['1', '0']})
    features = build_item_features(events, props, pd.DataFrame({'categoryid': [1], 'parentid': [np.nan]}))
    assert features.loc[1, 'has_available'] == 0


def test_neighbor_core_is_not_the_event_core():
    events = pd.DataFrame({'visitorid': [1, 1, 1, 2, 2, 2], 'itemid': [1, 1, 1, 1, 1, 1]})
    kept, _ = iterative_core(events, 3, 3, 'events')
    removed, _ = iterative_core(events, 3, 3, 'neighbors')
    assert len(kept) == len(events)
    assert removed.empty


def test_negative_sampling_excludes_observed_and_stops_when_saturated():
    from learning import NegativeSampler, complement_indices
    forbidden = np.array([0, 2, 3])
    mapped = complement_indices(np.arange(2), forbidden, 5)
    assert mapped.tolist() == [1, 4]
    saturated = NegativeSampler(np.array([[0, 0], [0, 1], [0, 2], [1, 0]]), 3)
    draws = saturated.sample(np.array([0, 0, 1, 1]), np.random.default_rng(0))
    assert draws[:2].tolist() == [-1, -1]
    assert set(draws[2:]).isdisjoint({0})
    dense = NegativeSampler(np.column_stack([np.zeros(9, dtype=int), np.arange(9)]), 10)
    many = dense.sample(np.zeros(200, dtype=int), np.random.default_rng(1))
    assert set(many.tolist()) == {9}


def test_behavior_weights_change_weighted_bpr_but_not_binary_bpr():
    import scipy.sparse as sp
    from learning import train_weighted_bpr, weighted_pairwise_loss
    import torch
    rows, cols = np.array([0, 0, 1]), np.array([0, 1, 0])
    light = sp.csr_matrix((np.array([1., 1., 1.]), (rows, cols)), shape=(2, 3))
    heavy = sp.csr_matrix((np.array([1., 12., 1.]), (rows, cols)), shape=(2, 3))
    kwargs = dict(factors=4, regularization=0.01, epochs=6, seed=7, device='cpu')
    first = train_weighted_bpr(heavy, **kwargs)
    second = train_weighted_bpr(light, **kwargs)
    binary_heavy = train_weighted_bpr(heavy, weighted=False, **kwargs)
    binary_light = train_weighted_bpr(light, weighted=False, **kwargs)
    assert not np.allclose(first.user_factors, second.user_factors)
    assert np.allclose(binary_heavy.user_factors, binary_light.user_factors)
    user = torch.ones(2, 2)
    positive, negative = torch.ones(2, 2), torch.zeros(2, 2)
    low = weighted_pairwise_loss(user, positive, negative, torch.tensor([1., 1.]), 0.)
    high = weighted_pairwise_loss(user, positive, negative, torch.tensor([1., 5.]), 0.)
    assert not torch.allclose(low, high)


def test_property_values_latest_policy_and_negative_tokens():
    from learning import encode_properties, numeric_token
    assert numeric_token(pd.Series(['n10', '10', 'n277.200'])).tolist() == [-10., 10., -277.2]
    base = {'timestamp': [1, 2], 'itemid': [1, 1], 'property': ['color', 'color']}
    red = encode_properties(pd.DataFrame({**base, 'value': ['red blue', 'red blue']}))
    blue = encode_properties(pd.DataFrame({**base, 'value': ['red blue', 'green']}), metadata_policy='latest')
    history = encode_properties(pd.DataFrame({**base, 'value': ['red', 'green']}), metadata_policy='history')
    names = encode_properties(pd.DataFrame({**base, 'value': ['red', 'green']}), mode='names')
    assert set(red.token) == {'red', 'blue'}
    assert set(blue.token) == {'green'}
    assert set(history.token) == {'red', 'green'}
    assert set(names.node_key) == {'color'}
    rare = encode_properties(pd.DataFrame({
        'timestamp': [1, 1], 'itemid': [1, 2], 'property': ['brand', 'brand'], 'value': ['solo', 'solo']}))
    once = encode_properties(pd.DataFrame({
        'timestamp': [1], 'itemid': [1], 'property': ['brand'], 'value': ['solo']}))
    assert 'solo' in set(rare.label)
    assert set(once.label) == {'<rare>'}


def _graph(events, props, mode='values', policy='latest', weighted=True):
    tree = pd.DataFrame({'categoryid': [1, 2], 'parentid': [np.nan, 1]})
    generator = GNNEmbeddingGenerator(device='cpu', embedding_dim=4, layers=1, seed=3,
                                      behavior_weighted=weighted, property_mode=mode, metadata_policy=policy,
                                      graph_mode='heterogeneous')
    return generator._prepare_graph_structures({
        'events': events, 'item_properties': props, 'category_tree': tree})


def test_property_values_change_the_graph_and_available_does_not():
    events = pd.DataFrame({'visitorid': [1, 1, 2], 'itemid': [10, 11, 10], 'event': ['view', 'view', 'view']})
    red = pd.DataFrame({'timestamp': [1, 1], 'itemid': [10, 11], 'property': ['color', 'color'], 'value': ['red', 'red']})
    blue = red.copy()
    blue.loc[0, 'value'] = 'blue'
    stock = pd.concat([red, pd.DataFrame({'timestamp': [1], 'itemid': [10], 'property': ['available'], 'value': ['0']})])
    other = stock.copy()
    other.loc[other.property == 'available', 'value'] = '1'
    assert _graph(events, red)['property_ids'] != _graph(events, blue)['property_ids']
    assert _graph(events, red, mode='names')['property_ids'] == _graph(events, blue, mode='names')['property_ids']
    assert np.array_equal(_graph(events, stock)['edge_sources'], _graph(events, other)['edge_sources'])
    categories = pd.DataFrame({
        'timestamp': [1, 2], 'itemid': [10, 10], 'property': ['categoryid', 'categoryid'], 'value': ['1', '2']})
    latest, history = _graph(events, categories), _graph(events, categories, policy='history')
    assert len(history['edge_sources']) > len(latest['edge_sources'])


def test_event_weight_changes_gnn_training():
    import torch
    events = pd.DataFrame({'visitorid': [1, 1, 2], 'itemid': [10, 11, 10], 'event': ['view', 'view', 'view']})
    purchased = events.copy()
    purchased.loc[2, 'event'] = 'transaction'
    props = pd.DataFrame({'timestamp': [1], 'itemid': [10], 'property': ['color'], 'value': ['red']})
    light = _graph(events, props)
    heavy = _graph(purchased, props)
    assert np.allclose(light['weights'], 1)
    assert not np.allclose(light['weights'], heavy['weights'])
    generator = GNNEmbeddingGenerator(device='cpu', embedding_dim=4, layers=1, epochs=15, seed=5,
                                      learning_rate=0.05, early_stopping_patience=None, behavior_weighted=True)

    def fit(graph):
        model, _, history = generator._fit_graph(graph, 15, torch)
        with torch.no_grad():
            return history[0]['train_loss'], model(0.0).detach().cpu().numpy().copy()

    light_loss, light_vectors = fit(light)
    heavy_loss, heavy_vectors = fit(heavy)
    assert abs(light_loss - heavy_loss) > 1e-4
    assert np.max(np.abs(light_vectors - heavy_vectors)) > 1e-4
    binary_light, binary_heavy = _graph(events, props, weighted=False), _graph(purchased, props, weighted=False)
    assert np.allclose(binary_light['weights'], binary_heavy['weights'])


def test_filtered_reference_user_scores_zero(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    rows = []
    stamp = 1
    for rep in range(8):
        event = 'transaction' if rep >= 6 else 'view'
        for user in range(1, 5):
            for item in range(1, 6):
                if item == user:
                    continue
                rows.append((stamp, user, event, item, None))
                stamp += 1
    rows = [(0, 5, 'view', 1, None), *rows, (10**9, 5, 'transaction', 1, None)]
    pd.DataFrame(rows, columns=['timestamp', 'visitorid', 'event', 'itemid', 'transactionid']).to_csv(raw/'events.csv', index=False)
    props = pd.DataFrame([(1, item, prop, value) for item in range(1, 6)
                          for prop, value in [('categoryid', '1'), ('available', '1'), ('color', 'red')]],
                         columns=['timestamp', 'itemid', 'property', 'value'])
    props.iloc[:6].to_csv(raw/'item_properties_part1.csv', index=False)
    props.iloc[6:].to_csv(raw/'item_properties_part2.csv', index=False)
    pd.DataFrame({'categoryid': [1], 'parentid': [np.nan]}).to_csv(raw/'category_tree.csv', index=False)
    out = tmp_path / 'processed'
    UnifiedPreprocessor(raw, out, n_estimators=10, use_if=False).run()
    retention = pd.read_csv(out/'truth_retention.csv').set_index('visitorid')
    assert retention.loc[5, 'train_user_retained'] == False
    assert retention.loc[5, 'original_retention'] == 0
    ALSRecommender(out, factors=4, iterations=1, top_k=5, bpr_epochs=1, device='cpu').run()
    GNNEmbeddingGenerator(out, raw, embedding_dim=4, layers=1, epochs=1, batch_size=32,
                          early_stopping_patience=None, device='cpu').run()
    ReRanker(out, raw, top_k=5, als_candidate_k=5, inventory_gamma=1.5).run()
    Evaluator(out, top_k=5).run(k_values=[5])
    original = pd.read_csv(out/'metrics_original_Hybrid_at_5_users.csv').set_index('visitorid')
    conditional = pd.read_csv(out/'metrics_conditional_Hybrid_at_5_users.csv').set_index('visitorid')
    assert original.loc[5, ['ndcg', 'recall', 'recommendations']].tolist() == [0, 0, 0]
    assert 5 not in conditional.index
    assert len(original) > len(conditional)
    assert original.index.is_monotonic_increasing


def test_conditions_share_reference_truth(tmp_path):
    raw = tmp_path / 'raw'
    make_raw(raw)
    ids = []
    for use_if, mode, name in [(False, 'events', 'a'), (True, 'events', 'b'), (False, 'neighbors', 'c')]:
        out = tmp_path / name
        UnifiedPreprocessor(raw, out, n_estimators=12, contamination=0.25, use_if=use_if, core_mode=mode).run()
        manifest = require_protocol(out)
        ids.append((manifest['reference_id'], manifest['original_truth_id']))
        if name == 'a':
            original = pd.read_csv(out/'events_test_original.csv')
        else:
            pd.testing.assert_frame_equal(original, pd.read_csv(out/'events_test_original.csv'))
    assert ids[0] == ids[1] == ids[2]


def test_old_bpr_entry_points_are_closed():
    with pytest.raises(RuntimeError, match='implicit BPR'):
        als_debug.train_bpr()
    with pytest.raises(RuntimeError, match='als_debug'):
        als_debug.run_pipeline_return_metrics()


def test_plan_and_gamma_cannot_use_the_outer_test(tmp_path):
    from experiments import gamma_sweep, run_suite
    from viz import generate_experiment_figures
    (tmp_path/'locked'/'experiment_plan.json').parent.mkdir()
    (tmp_path/'locked'/'experiment_plan.json').write_text('{"different": true}', encoding='utf-8')
    with pytest.raises(ValueError, match='plan'):
        run_suite(tmp_path/'locked', tmp_path, seeds=(42,), iterations=1, bpr_epochs=1, epochs=1,
                  device='cpu', include_ablations=False, factors=4, embedding_dim=4)
    raw = tmp_path/'raw'
    make_raw(raw)
    root = tmp_path/'suite'
    run_suite(root, raw, seeds=(42,), iterations=1, bpr_epochs=1, epochs=2, device='cpu',
              include_ablations=False, factors=4, embedding_dim=4)
    run_suite(root, raw, seeds=(42,), iterations=1, bpr_epochs=1, epochs=2, device='cpu',
              include_ablations=False, factors=4, embedding_dim=4)
    plan = json.loads((root/'experiment_plan.json').read_text(encoding='utf-8'))
    assert plan['locked_before_test_evaluation'] is True
    assert plan['threads']['OPENBLAS_NUM_THREADS'] == '1'
    assert plan['als_use_gpu'] is False
    run = root/'runs'/'no_if_events__main__s42'
    gamma = json.loads((run/'validation'/'gamma_selection.json').read_text(encoding='utf-8'))
    assert gamma['protocol_id'] == require_protocol(run/'validation')['protocol_id']
    assert gamma['protocol_id'] != require_protocol(run)['protocol_id']
    with pytest.raises(ValueError, match='validation'):
        gamma_sweep(run, [1.0])
    history = json.loads((run/'gnn_training_history.json').read_text(encoding='utf-8'))
    min_epochs = min(int(plan.get('gnn_min_epochs', 1)), int(plan['gnn_max_epochs']))
    scored = [row for row in history['selection_history']
              if 'validation_original_ndcg' in row and row['epoch'] >= min_epochs]
    assert scored, 'expected at least one eligible validation epoch'
    best_epoch, best_score = scored[0]['epoch'], scored[0]['validation_original_ndcg']
    for row in scored[1:]:
        if row['validation_original_ndcg'] > best_score + 1e-6:
            best_score = row['validation_original_ndcg']
            best_epoch = row['epoch']
    assert history['chosen_epochs'] == best_epoch
    assert len(history['refit_history']) == history['chosen_epochs']
    users = [pd.read_csv(run/f'metrics_original_{name}_at_50_users.csv').visitorid.tolist()
             for name in ['ALS', 'BPR', 'ALS_BPR', 'GNN', 'Hybrid']]
    assert all(values == users[0] for values in users)
    figures = generate_experiment_figures(root, root/'experiment_figures', dpi=72)
    assert (figures/'gamma_validation.pdf').exists()
    assert (figures/'filter_comparison.pdf').exists()
