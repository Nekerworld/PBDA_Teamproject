"""Audit saved revision evidence without importing ML libraries or training models."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import statistics


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def locked_config(path):
    for node in ast.parse(path.read_text(encoding='utf-8-sig')).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'LOCKED' for t in node.targets):
            # Only inspect literal fields; never execute the model module.
            return {ast.literal_eval(k): ast.literal_eval(v) for k, v in zip(node.value.keys, node.value.values)
                    if isinstance(k, ast.Constant) and k.value in ('primary', 'test_size')}
    raise ValueError('LOCKED configuration not found')


def audit(directory, source):
    plan = read_json(directory / 'experiment_plan.json')
    manifest = read_json(directory / 'experiment_manifest.json')
    issues, sources = [], {}
    def issue(code, message):
        issues.append({'code': code, 'message': message})
    def verify(base, hashes):
        for name, expected in hashes.items():
            path = base / name
            if not path.is_file() or sha256(path) != expected:
                issue('HASH_MISMATCH', str(path))
            else:
                sources[str(path)] = expected
    verify(directory, manifest['files'])
    primary = plan.get('primary_condition', plan['primary'])
    dataset = directory / 'datasets' / primary
    protocol = read_json(dataset / 'manifest.json')
    verify(dataset, protocol['hashes'])
    verify(dataset, protocol.get('audit_hashes', {}))
    stats = {r['stage']: r for r in read_csv(dataset / 'dataset_statistics.csv')}
    config = locked_config(source / 'experiments.py')
    actual_split = float(protocol['config']['test_size'])
    if 'test_size' not in plan:
        issue('PLAN_SPLIT_MISSING', f'Saved plan omits test_size; dataset records {actual_split}.')
    elif float(plan['test_size']) != actual_split:
        issue('PLAN_SPLIT_MISMATCH', 'Saved plan and dataset split disagree.')
    if float(config['test_size']) != actual_split:
        issue('SOURCE_SPLIT_DRIFT', f'Current source test_size={config["test_size"]}; saved dataset={actual_split}.')
    if primary != config['primary']:
        issue('SOURCE_FILTER_DRIFT', 'Current primary filter differs from saved plan.')
    if int(stats['train_before_filter']['transaction_events']) and not int(stats['train_final']['transaction_events']):
        issue('ALL_TRAIN_PURCHASES_REMOVED', 'Filtering removed every training transaction.')
    table = read_csv(directory / 'table1_dataset_statistics.csv')
    published = next(r for r in table if r['Metric'] == 'Sparsity (%)')['4. Final Train Set']
    computed = float(stats['train_final']['sparsity_percent'])
    if abs(float(published.rstrip('%')) - computed) > 0.00011:
        issue('SPARSITY_TABLE_MISMATCH', f'Table={published}; unique-pair sparsity={computed:.6f}%.')
    rows = read_csv(directory / 'experiment_metrics.csv')
    groups, timing, completed = {}, [], set()
    for run in manifest['runs']:
        name = run['run']
        base = directory / 'runs' / name
        verify(base, {'run_config.json':run['config'], 'evaluation_artifacts.json':run['evaluation'],
                      'manifest.json':run['manifest']})
        for file in ('als_artifacts.json', 'gnn_artifacts.json', 'rerank_artifacts.json', 'evaluation_artifacts.json'):
            artifact = read_json(base / file)
            verify(base, artifact.get('files', artifact.get('hashes', {})))
        complete = read_json(base / 'complete.json')
        if complete['evaluation_hash'] != sha256(base / 'evaluation_artifacts.json'):
            issue('COMPLETION_HASH_MISMATCH', name)
        else:
            completed.add(name)
        run_protocol = read_json(base / 'manifest.json')
        if any(run_protocol[k] != protocol[k] for k in ('protocol_id','reference_id','original_truth_id')):
            issue('RUN_PROTOCOL_MISMATCH', name)
        timing.extend(dict(r, run=name) for r in read_csv(base / 'timing_metrics.csv'))
    for r in rows:
        if r['scope'] != 'original' or r['condition'] != primary or r['ablation'] != 'main':
            continue
        if any(r[k] != protocol[k] for k in ('protocol_id','reference_id','original_truth_id')):
            issue('METRIC_PROTOCOL_MISMATCH', r['run'])
        if int(r['users_evaluated']) != protocol['evaluation_users']:
            issue('COHORT_MISMATCH', r['run'])
        groups.setdefault((r['model'], int(r['k'])), []).append(r)
    summary = []
    for (model, k), entries in sorted(groups.items()):
        if sorted(int(r['seed']) for r in entries) != sorted(plan['seeds']):
            issue('INCOMPLETE_SEEDS', f'{model}@{k}')
        row = {'model':model, 'k':k, 'seeds':len(entries), 'users':entries[0]['users_evaluated'],
               'missing_users':entries[0]['missing_recommendation_users']}
        for metric in ('ndcg','recall','precision','hit_rate','f1'):
            values = [float(r[metric]) for r in entries]
            row[metric+'_mean'] = statistics.mean(values)
            row[metric+'_std'] = statistics.stdev(values) if len(values)>1 else 0.
        summary.append(row)
    progress = read_json(directory / 'progress.json')
    if set(progress['completed']) != completed:
        issue('STALE_PROGRESS', 'progress.json differs from verified completion markers; do not retrain solely from progress.json.')
    return {'experiment':str(directory), 'split_test_size':actual_split, 'source_defaults':config,
            'condition':primary, 'protocol_id':protocol['protocol_id'], 'issues':issues,
            'verified_files':len(sources), 'source_hashes':sources, 'summary':summary,
            'statistics':list(stats.values()), 'timing':timing,
            'if_features':protocol['if_features'], 'if_removed_percent_train':protocol['if_event_percent_of_prefilter_train'],
            'if_removed_percent_raw':protocol['if_event_percent_of_raw_input'],
            'gamma_grid':plan['gamma_grid'], 'completed_runs':sorted(completed)}


def write_csv(path, rows):
    if rows:
        with path.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, default=Path(__file__).parent)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'audit_results')
    parser.add_argument('--strict', action='store_true', help='Exit 2 when any audit issue exists')
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use an empty output directory to preserve previous evidence')
    result = audit(args.experiment_dir.resolve(), args.source_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for name, key in [('model_summary.csv','summary'),('dataset_statistics.csv','statistics'),('timing.csv','timing')]:
        write_csv(args.output_dir/name, result[key])
    lines = ['# Saved result audit', '', f'Condition: {result["condition"]}; test fraction: {result["split_test_size"]}',
             f'Verified files: {result["verified_files"]}', '', '| Finding | Evidence |', '|---|---|']
    lines.extend(f'| {i["code"]} | {i["message"]} |' for i in result['issues'])
    (args.output_dir/'AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'verified_files':result['verified_files'],'issues':result['issues']},ensure_ascii=False,indent=2))
    return 2 if args.strict and result['issues'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
