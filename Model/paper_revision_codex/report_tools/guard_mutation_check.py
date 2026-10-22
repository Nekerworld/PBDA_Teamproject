"""Put the pre-fix behaviour of each cycle-5..11 guard (H1-H48) back into a temporary copy, one at a time, and check
that test_cycle4/5/7/8/9/10/11_guards.py fail (the end-to-end reproduction run is left out for time). The output records the git HEAD and uncommitted files the copy was taken
from. Default output: audit_runs/guard_mutants.json (git-ignored, so a rerun never replaces committed evidence such
as audit/cycle5, cycle7, cycle8 or cycle9/guard_mutants.json).

Usage: python report_tools/guard_mutation_check.py [output.json]
"""
import json, shutil, subprocess, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
src = Path(__file__).resolve().parents[1]
M = {
 'H1 reproduction exempt without matching code': ('rolling_protocol.py', "\n                              and code_matches(stored.get('code_sha256', {}), stored.get('code_hash_scheme', 'sha256-raw')))", ")"),
 'H2 exact window match instead of overlap': ('rolling_protocol.py', "                    overlap = max(low, windows[name][0]) < min(high, windows[name][1]) - 1e-9", "                    overlap = (low, high) == windows[name]"),
 'H3 committed records only': ('rolling_protocol.py', "RESULT_RECORDS = (HERE / 'audit', HERE / 'audit_runs')", "RESULT_RECORDS = (HERE / 'audit',)"),
 'H4 raw input not re-verified': ('rolling_protocol.py', "    if 'raw_input_sha256' in stored and raw_hashes(raw) != stored['raw_input_sha256']:", "    if False:"),
 'H5 selection compared within a purpose only': ('rolling_protocol.py', "        if item['fold'] != entry['fold']:", "        if item['fold'] != entry['fold'] or item['purpose'] != entry['purpose']:"),
 'H6 finished fold skips the log check': ('rolling_protocol.py', "    if not open_test(root, fold['name'], 'sealed_test', plan_hash, (fold_dir / 'test_metrics.csv').exists()):", "    if (fold_dir / 'test_metrics.csv').exists() or not open_test(root, fold['name'], 'sealed_test', plan_hash, False):"),
 'H7 cached MF fit not checked': ('rolling_protocol.py', "        if {k: cached.get(k) for k in wanted} != wanted:", "        if False:"),
 'H8 cached GNN job not checked': ('rolling_protocol.py', "        if any(result['spec'].get(k) != spec[k] for k in GNN_JOB_IDENTITY):", "        if False:"),
 'H9 precision tool imports numpy first': ('report_tools/check_score_precision.py', "import argparse\n", "import argparse\nimport numpy\n"),
 'H10 leakage gate on fold_raw only': ('rolling_protocol.py', "        leakage_main(prepared, raw, gnn=True, output=path, end_fraction=end_fraction)", "        leakage_main(prepared, Path(fold_dir) / 'fold_raw', gnn=True, output=path, end_fraction=None)"),
 'H11 posthoc marker written first': ('rolling_protocol.py', "    frame['note'] = 'Post-hoc on sealed test, seed 42, repeat policy; never used for selection.'\n", "    frame['note'] = 'Post-hoc on sealed test, seed 42, repeat policy; never used for selection.'\n    frame.to_csv(out, index=False)\n"),
 'H12 selection plan hash not checked': ('rolling_protocol.py', "    if selection['plan_sha256'] != plan_hash:\n        raise", "    if False:\n        raise"),
 'H13 records matched on all raw file bytes': ('rolling_protocol.py', "        if (stored.get('raw_input_sha256') or {}).get('events.csv') not in forms:", "        if stored.get('raw_input_sha256') != raw_hashes:"),
 'H14 git output fully stripped': ('rolling_protocol.py', ".stdout.rstrip('\\n')", ".stdout.strip()"),
 'H15 other roots checked only at start': ('rolling_protocol.py', "        conflicts = committed_test_conflicts(stored.get('raw_input_sha256'), plan_hash, [fold], event_forms=RUN_EVENT_FORMS,\n                                             exclude=root, selections={fold_name: selection_decision(Path(root) / fold_name / 'selection.json')},\n                                             test_windows={fold_name: recorded_test_window(Path(root) / fold_name)})", "        conflicts = []"),
 'H16 reproduction exempt with other raw files': ('rolling_protocol.py', "stored['plan_sha256'] == plan_hash and stored.get('raw_input_sha256') == raw_hashes", "stored['plan_sha256'] == plan_hash"),
 'H17 hybrid grid cache trusted when it exists': ('rolling_protocol.py', "        if len(rows) == len(HYBRID_GRID):", "        if True:"),
 'H18 plan created without git': ('rolling_protocol.py', "        if not state['commit']:\n            raise", "        if False:\n            raise"),
 'H19 code not re-checked before a test opening': ('rolling_protocol.py', "        require_frozen_code(root)\n        stored = json.loads((Path(root)", "        stored = json.loads((Path(root)"),
 'H20 code not re-checked before a GNN job': ('rolling_protocol.py', "                require_frozen_code(root)\n                path = spec_dir", "                path = spec_dir"),
 'H21 own root not excluded': ('rolling_protocol.py', " or plan_path.parent.resolve() == skip:", ":"),
 'H22 cut-off log line raises': ('rolling_protocol.py', "            try:\n                entry = json.loads(line)\n                opened.setdefault(entry['fold'], set()).add(entry.get('selection_sha256'))\n            except (ValueError, KeyError, TypeError):\n                for name in windows:\n                    opened.setdefault(name, set()).add(None)", "            entry = json.loads(line)\n            opened.setdefault(entry['fold'], set()).add(entry.get('selection_sha256'))"),
 'H23 cut-off log line skipped': ('rolling_protocol.py', "                for name in windows:\n                    opened.setdefault(name, set()).add(None)", "                pass"),
 'H24 unreadable record plan not named': ('rolling_protocol.py', "        except ValueError as error:\n            raise RuntimeError(f'Test record", "        except KeyError as error:\n            raise RuntimeError(f'Test record"),
 'H25 unreadable own log not named': ('rolling_protocol.py', "    except ValueError as error:\n        raise RuntimeError(f'{path} has", "    except KeyError as error:\n        raise RuntimeError(f'{path} has"),
 'H26 hybrid grid cache read with the default float parser': ('rolling_protocol.py', "pd.read_csv(path, float_precision='round_trip')", "pd.read_csv(path)"),
 'H27 test-opening re-check ignores line-ending copies': ('rolling_protocol.py', "[fold], event_forms=RUN_EVENT_FORMS,\n", "[fold],\n"),
 'H28 exact reproduction exempt whatever its selection': ('rolling_protocol.py', "        if exact_reproduction and selections is None:\n            continue", "        if exact_reproduction:\n            continue"),
 'H29 recorded windows ignored': ('rolling_protocol.py', "                if own and theirs:", "                if False:"),
 'H30 touching counts as disjoint without a recorded window': ('rolling_protocol.py', "overlap = max(low, windows[name][0]) <= min(high, windows[name][1]) + 1e-9", "overlap = max(low, windows[name][0]) < min(high, windows[name][1]) - 1e-9"),
 'H31 test opening passes no selection': ('rolling_protocol.py', "exclude=root, selections={fold_name: selection_decision(Path(root) / fold_name / 'selection.json')},", "exclude=root,"),
 'H32 test opening passes no actual window': ('rolling_protocol.py', ",\n                                             test_windows={fold_name: recorded_test_window(Path(root) / fold_name)})", ")"),
 'H33 completion CSV written in place': ('rolling_protocol.py', "    partial = path.with_name(path.name + '.partial')\n    frame.to_csv(partial, index=False)\n    os.replace(partial, path)", "    frame.to_csv(path, index=False)"),
 'H34 background error replaced by a generic one': ('rolling_protocol.py', "        raise box['error']", "        raise RuntimeError('GNN jobs failed')"),
 'H35 non-object access log entry accepted': ('rolling_protocol.py', "        if not all(isinstance(item, dict) and 'fold' in item for item in previous):", "        if False:"),
 'H36 test opening compares selection file bytes': ('rolling_protocol.py', "selections={fold_name: selection_decision(Path(root)", "selections={fold_name: sha256(Path(root)"),
 'H37 record selection not tied to its logged openings': ('rolling_protocol.py', "    if not path.exists() or opened != {sha256(path)}:", "    if not path.exists():"),
 'H38 run-specific seal time in the decision': ('rolling_protocol.py', "        decision = {'plan_sha256': s['plan_sha256'], 'fold': s['fold'],\n", "        decision = {'plan_sha256': s['plan_sha256'], 'fold': s['fold'], 'locked_utc': s.get('locked_utc'),\n"),
 'H39 odd split fields converted unguarded': ('rolling_protocol.py', "    if type(start) is not int or not (end is None or type(end) is int):\n        return None\n    return start, (end if end is not None else float('inf'))", "    return int(start), (int(end) if end is not None else float('inf'))"),
 'H40 failed step does not stop the background jobs': ('rolling_protocol.py', "        stop.set()\n        thread.join()\n        raise", "        raise"),
 'H41 stopped GNN jobs left running': ('rolling_protocol.py', "            process.terminate()\n", ""),
 'H42 stop ignored by the GNN job loop': ('rolling_protocol.py', "            if stop is not None and stop.is_set():\n                raise", "            if False:\n                raise"),
 'H43 unknown fold name accepted': ('rolling_protocol.py', "    if unknown:\n        raise ValueError", "    if False:\n        raise ValueError"),
 'H44 background worker keeps only ordinary errors': ('rolling_protocol.py', "        except BaseException as error:  # surfaced below", "        except Exception as error:  # surfaced below"),
 'H45 clone links straight into the destination': ('experiments.py',
     "    partial = destination.with_name(destination.name + '.partial')\n    if partial.exists():\n"
     "        shutil.rmtree(partial)  # hard links left by an interrupted clone; the source files are untouched\n"
     "    partial.mkdir(parents=True)\n    for name in sorted({'manifest.json', *manifest['hashes'], *manifest['audit_hashes']}):\n"
     "        try:\n            os.link(source / name, partial / name)\n"
     "        except OSError as error:\n            raise OSError(f'Could not hard-link {name} from {source} to {destination}') from error\n"
     "    if (source / 'validation').exists():\n        clone_protocol(source / 'validation', partial / 'validation')\n"
     "    if destination.exists():\n        destination.rmdir()\n    os.replace(partial, destination)\n",
     "    destination.mkdir(parents=True, exist_ok=True)\n"
     "    for name in {'manifest.json', *manifest['hashes'], *manifest['audit_hashes']}:\n"
     "        try:\n            os.link(source / name, destination / name)\n"
     "        except OSError as error:\n            raise OSError(f'Could not hard-link {name} from {source} to {destination}') from error\n"
     "    if (source / 'validation').exists():\n        clone_protocol(source / 'validation', destination / 'validation')\n"),
 'H46 GNN job not tied to the run': ('rolling_protocol.py', "                running.append((process, spec, logfile))\n                kill_with_parent(process)\n",
                                     "                running.append((process, spec, logfile))\n"),
 'H47 job object without kill-on-close': ('rolling_protocol.py', "LimitFlags = 0x2000", "LimitFlags = 0"),
 'H48 failed launch leaves its log open': ('rolling_protocol.py', "                except BaseException:\n                    logfile.close()\n                    raise",
                                           "                except BaseException:\n                    raise"),
}
git = lambda *a: subprocess.run(['git', *a], cwd=src, capture_output=True, text=True).stdout.rstrip('\n')
run_info = {'head': git('rev-parse', 'HEAD'), 'dirty': git('status', '--porcelain', '--', '.').splitlines(),
            'started_utc': datetime.now(timezone.utc).isoformat()}
tmp = Path(tempfile.mkdtemp(prefix='h_mut_'))
def fresh():
    d = tmp / 'p'
    if d.exists(): shutil.rmtree(d)
    d.mkdir()
    for f in src.glob('*.py'): shutil.copy2(f, d)
    shutil.copy2(src / 'pytest.ini', d)
    for s in ('audit_tools', 'report_tools'): shutil.copytree(src / s, d / s, ignore=shutil.ignore_patterns('__pycache__'))
    for v in ('v6', 'v7'):
        (d / 'audit' / f'results_{v}').mkdir(parents=True)
        for n in ('plan.json', 'test_access_log.jsonl'): shutil.copy2(src / 'audit' / f'results_{v}' / n, d / 'audit' / f'results_{v}' / n)
    return d
args = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'test_cycle4_guards.py', 'test_cycle5_guards.py',
        'test_cycle7_guards.py', 'test_cycle8_guards.py', 'test_cycle9_guards.py', 'test_cycle10_guards.py',
        'test_cycle11_guards.py', '--deselect', 'test_cycle4_guards.py::test_verify_plan_code_reports_changes_and_fails_only_when_asked',
        '--deselect', 'test_cycle10_guards.py::test_exact_reproduction_from_raw_reaches_its_test_opening']
rows = []
d = fresh(); r = subprocess.run(args, cwd=d, capture_output=True, text=True); base = r.stdout.strip().splitlines()[-1]; print('unmutated', base, flush=True)
assert r.returncode == 0
for name, (f, old, new) in M.items():
    d = fresh(); t = (d / f).read_text(encoding='utf-8'); assert t.count(old) == 1, (name, t.count(old))
    (d / f).write_text(t.replace(old, new, 1), encoding='utf-8')
    r = subprocess.run(args, cwd=d, capture_output=True, text=True)
    rows.append({'mutant': name, 'file': f, 'result': 'killed' if r.returncode else 'survived', 'summary': r.stdout.strip().splitlines()[-1]})
    print(json.dumps(rows[-1]), flush=True)
shutil.rmtree(tmp)
out = Path(sys.argv[1]) if len(sys.argv) > 1 else src / 'audit_runs' / 'guard_mutants.json'
out.parent.mkdir(parents=True, exist_ok=True)
run_info['finished_utc'] = datetime.now(timezone.utc).isoformat()
out.write_text(json.dumps({**run_info, 'unmutated': base, 'tests': args[5:], 'mutants': rows}, indent=2), encoding='utf-8')
