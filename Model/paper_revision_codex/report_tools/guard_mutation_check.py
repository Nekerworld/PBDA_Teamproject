"""Put the pre-fix behaviour of each cycle-5..23 guard (H1-H95) back into a temporary copy, one at a time, and check
that test_cycle4/5/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22_guards.py or test_compare_outputs.py fail (the end-to-end reproduction run is left out for time, and
the verify_plan_code test because the copy is outside git). H63, H64 and H66 take out a part of a fix rather than put back
the exact pre-fix code; H67 puts back the I-108 prefix exemption in pathspec form; H72 is no old code
but the failure the H71 check refuses (selection jobs on the outer split); H74 puts back the d92de19 check (any fold's
validation accepted) and H75 takes out the role part of the a22f6ea check (its own-fold path check stays). H79, H80 and H82-H86 are no old code but failures the Cycle 22 tests and checks
refuse (MF selection on the outer split, test-time assembly from the wrong selection or refit, refit with early
stopping). H88-H93 put back how this tool and mutation_check.py read pytest's output (locale codec, console
encoding) and how mutation_check.py judged a run (any non-zero exit as a kill, aborted runs accepted). H94 puts back the
Cycle 22 refit check (recorded mode and epochs only, no per-epoch rows) and H95 the cache audit without the MF fit data
check. pytest's output is
read as UTF-8. A mutant counts as killed only when pytest exits with 1 and reports
failed or erroring tests; any other non-zero exit (interrupt, collection error, process killed from outside) is recorded
as aborted. The git HEAD, uncommitted files and start time are printed when the run starts and stored in the output;
the exit code is 0 only when every mutant is killed. Default output: audit_runs/guard_mutants.json (git-ignored, so a rerun never replaces committed evidence such
as audit/cycle5, cycle7, cycle8 or cycle9/guard_mutants.json).

Usage: python report_tools/guard_mutation_check.py [output.json]
"""
import json, os, re, shutil, subprocess, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
src = Path(__file__).resolve().parents[1]
REFIT_CHECK = ("        if (trained.get('mode') != 'fixed_epochs' or trained.get('chosen_epochs') != spec['epochs']\n"
               "                or [row.get('epoch') for row in rows] != list(range(1, spec['epochs'] + 1))\n"
               "                or any('validation_original_ndcg' in row for row in rows)):")
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
 'H18 plan created without git': ('rolling_protocol.py', "        if not state['commit'] or state['dirty_files'] is None:\n            raise", "        if False:\n            raise"),
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
 'H46 GNN job not tied to the run': ('rolling_protocol.py', "                try:\n                    kill_with_parent(process)\n",
                                     "                try:\n                    pass\n"),
 'H47 job object without kill-on-close': ('rolling_protocol.py', "LimitFlags = 0x2000", "LimitFlags = 0"),
 'H48 failed launch leaves its log open': ('rolling_protocol.py', "                except BaseException:\n                    logfile.close()\n                    raise",
                                           "                except BaseException:\n                    raise"),
 'H49 outer manifest written before finalize_outer': ('data_protocol.py', '"test", len(events), write_manifest=False)',
                                                      '"test", len(events), write_manifest=True)'),
 'H50 empty fold list runs every fold': ('rolling_protocol.py', "    if folds is not None and not folds:\n        raise", "    if False:\n        raise"),
 'H51 job object error hides an ended job': ('rolling_protocol.py', "if process.poll() is None:  # a job that already ended",
                                             "if True:  # a job that already ended"),
 'H52 zero compared files pass': ('report_tools/compare_outputs.py', "'all_equal': bool(rows) and not empty_pairs and all(", "'all_equal': all("),
 'H53 JSON compared without types': ('report_tools/compare_outputs.py', "    if type(x) is not type(y):\n        return False\n", ""),
 'H54 files only in NEW ignored': ('report_tools/compare_outputs.py', "for name in sorted(listed(new) - listed(committed))]",
                                   "for name in []]"),
 'H55 #keys reads missing keys as None': ('report_tools/compare_outputs.py',
                                          "            if not (isinstance(x, dict) and isinstance(y, dict)):\n"
                                          "                raise ValueError(f'#keys needs a JSON object: {committed}')\n"
                                          "            missing = [k for k in keys if k not in x or k not in y]\n"
                                          "            if missing:\n"
                                          "                raise ValueError(f'#keys not in both files ({committed}): {missing}')\n"
                                          "            row['compared_keys'] = keys\n"
                                          "            x, y = {k: x[k] for k in keys}, {k: y[k] for k in keys}",
                                          "            row['compared_keys'] = keys\n"
                                          "            x, y = {k: x.get(k) for k in keys}, {k: y.get(k) for k in keys}"),
 'H56 #keys on a non-object JSON': ('report_tools/compare_outputs.py',
                                    "            if not (isinstance(x, dict) and isinstance(y, dict)):\n"
                                    "                raise ValueError(f'#keys needs a JSON object: {committed}')\n", ""),
 'H57 empty committed folder hidden by another pair': ('report_tools/compare_outputs.py',
                                                       "            if not listed(committed):\n"
                                                       "                empty_pairs.append(pair)\n", ""),
 'H58 precision check reads an outer-test folder': ('report_tools/check_score_precision.py',
                                                    "    if role != 'validation':", "    if False:"),
 'H59 use_protocol v6 keeps v7 values': ('rolling_protocol.py',
                                         "PROTOCOLS = {'v6': {name: globals()[name] for name in V7}, 'v7': V7}",
                                         "PROTOCOLS = {'v6': {}, 'v7': V7}"),
 'H60 Hybrid grid-end flags without alpha, candidate ends of blend only': (
     'rolling_protocol.py', "    grid = [c for c in HYBRID_GRID if c['strategy'] == best['strategy']]\n",
     "    return {'candidate_k': str(best['candidate_k']) in (str(CANDIDATE_KS[0]), str(CANDIDATE_KS[-1])),\n"
     "            'omega': float(best['omega']) in (.1, .9), 'gamma': float(best['gamma']) in (1., 1.5)}\n"),
 'H61 plan records hand-written Hybrid axes': (
     'rolling_protocol.py', "        'hybrid_grid_axes': {key: list(dict.fromkeys(c[key] for c in HYBRID_GRID))\n"
                            "                             for key in ('alpha', 'strategy', 'fusion', 'omega', 'gamma')},",
     "        'hybrid_grid_axes': {'alpha': [.25, .5, .75], 'strategy': ['blend', 'balanced_union'],\n"
     "                             'fusion': ['minmax', 'rrf'], 'omega': [.1, .2, .3, .4, .5, .6, .7, .8, .9],\n"
     "                             'gamma': [1., 1.1, 1.25, 1.5]},"),
 'H62 plan without the Hybrid grid digest': (
     'rolling_protocol.py', "    grid_record = {} if digest == SEALED_HYBRID_GRID_SHA256 else {'hybrid_grid_sha256': digest}",
     "    grid_record = {}"),
 'H63 dirty check lists the run-output folder': (
     'rolling_protocol.py', "'--', '.', ':(exclude)audit_runs')", "'--', '.')"),
 'H64 dirty check follows status.showUntrackedFiles': (
     'rolling_protocol.py', "run('status', '--porcelain', '--untracked-files=normal', '--',",
     "run('status', '--porcelain', '--',"),
 'H65 GNN job spec path relative to the caller': (
     'rolling_protocol.py', "    spec_dir = Path(root).resolve() / 'job_specs'", "    spec_dir = Path(root) / 'job_specs'"),
 'H66 unreadable dirty files pass as clean': (
     'rolling_protocol.py', "if not state['commit'] or state['dirty_files'] is None:", "if not state['commit']:"),
 'H67 dirty check exempts any path starting with audit_runs': (
     'rolling_protocol.py', "'--', '.', ':(exclude)audit_runs')", "'--', '.', ':(exclude)audit_runs*')"),
 'H68 failed git status read as no dirty files': (
     'rolling_protocol.py', "None if status is None else [line for line in status.splitlines() if line.strip()]",
     "[line for line in (status or '').splitlines() if line.strip()]"),
 'H69 git output read with the locale codec': (
     'rolling_protocol.py', "capture_output=True, encoding='utf-8',\n                                  check=True)",
     "capture_output=True, text=True,\n                                  check=True)"),
 'H70 detached launch from any folder': (
     'report_tools/launch_detached.py', "    if Path.cwd().resolve() != HERE:", "    if False:"),
 'H71 GNN selection job on any data': (
     'rolling_protocol.py', "    if mode == 'select' and (Path(source)", "    if False and (Path(source)"),
 'H72 GNN selection grid on the outer split': (
     'rolling_protocol.py', "config_name('gnn', c), val_dir, c, SELECTION_SEED", "config_name('gnn', c), prepared, c, SELECTION_SEED"),
 'H73 cached GNN data not compared': (
     'report_tools/check_cached_runs.py', "            if protocol_id(source) is None or protocol_id(fold_dir / job) != protocol_id(source):", "            if False:"),
 'H74 GNN selection job on the validation of another fold': (
     'rolling_protocol.py', "(Path(source).resolve() != Path(job_dir).resolve().parents[1] / 'prepared' / 'validation'\n                             or json.loads(", "(json.loads("),
 'H75 GNN selection job data role not read': (
     'rolling_protocol.py', "or json.loads((Path(source) / 'manifest.json').read_text(encoding='utf-8')).get('role') != 'validation'):", "or False):"),
 'H76 guard run killed from outside counted as killed': (
     'report_tools/guard_mutation_check.py', "    if returncode == 1 and " + r"re.search(r'\b\d+ (failed|errors?)\b', summary(stdout)):",
     "    if returncode:"),
 'H77 guard run with no output crashes the tool': (
     'report_tools/guard_mutation_check.py', "    return lines[-1] " + "if lines else ''", "    return lines[-1]"),
 'H78 MF selection grid on any data': (
     'rolling_protocol.py', "    require_inner_validation(fold_dir, val_dir, 'The MF selection grid')", "    pass"),
 'H79 MF selection grid on the outer split': (
     'rolling_protocol.py', "lambda: mf_validation(fold_dir, val_dir, contexts))", "lambda: mf_validation(fold_dir, prepared, contexts))"),
 'H80 Hybrid selection MF fit cloned from the outer split': (
     'rolling_protocol.py', "        clone_protocol(val_dir, val_mf)", "        clone_protocol(prepared, val_mf)"),
 'H81 Hybrid selection MF fit data not checked': (
     'rolling_protocol.py', "    require_inner_validation(fold_dir, val_mf, 'The MF fit used for Hybrid selection')", "    pass"),
 'H82 Hybrid backbone recorded as the best standalone GNN': (
     'rolling_protocol.py', "'hybrid_backbone': next(r for r in gnn_rows if r['name'] == repeat_backbone),", "'hybrid_backbone': gnn_rows[0],"),
 'H83 test Hybrid scored with the standalone GNN': (
     'rolling_protocol.py', "    hyb = FactorScores(context, als, bpr, gnn_hybrid)", "    hyb = FactorScores(context, als, bpr, gnn_std)"),
 'H84 novel test run with the repeat settings': (
     'rolling_protocol.py', "gnn_std, gnn_hyb, selection['policies'][policy])", "gnn_std, gnn_hyb, selection['policies']['repeat'])"),
 'H85 test Hybrid vectors from the standalone GNN refit': (
     'rolling_protocol.py', "gnn_hyb = load_gnn(refits[seed]['hybrid_gnn'], context)", "gnn_hyb = load_gnn(refits[seed]['gnn'], context)"),
 'H86 GNN refit keeps early stopping': (
     'rolling_protocol.py', "        kwargs['early_stopping_patience'] = None", "        pass"),
 'H87 GNN refit epochs not checked': (
     'rolling_protocol.py', REFIT_CHECK, "        if False:"),
 'H88 guard run output read with the locale codec': (
     'report_tools/guard_mutation_check.py', "    return subprocess.run(args, cwd=d, capture_output=True, " + "encoding='utf-8', errors='replace',",
     "    return subprocess.run(args, cwd=d, capture_output=True, text=True,"),
 'H89 guard run output encoding left to the console': (
     'report_tools/guard_mutation_check.py', "env={**os.environ, " + "'PYTHONIOENCODING': 'utf-8'})", "env=dict(os.environ))"),
 'H90 mutation tool counts any non-zero exit as killed': (
     'report_tools/mutation_check.py', "    return classify(result.returncode, result.stdout), summary(result.stdout)",
     "    return ('killed' if result.returncode else 'survived'), summary(result.stdout)"),
 'H91 mutation tool output read with the locale codec': (
     'report_tools/mutation_check.py', "cwd=scratch, capture_output=True, encoding='utf-8', errors='replace', env=env)",
     "cwd=scratch, capture_output=True, text=True, env=env)"),
 'H92 mutation tool output encoding left to the console': (
     'report_tools/mutation_check.py', "'MPLBACKEND': 'Agg',\n           'PYTHONIOENCODING': 'utf-8'}", "'MPLBACKEND': 'Agg'}"),
 'H93 mutation tool passes with aborted runs': (
     'report_tools/mutation_check.py', "    if aborted:\n        raise RuntimeError", "    if False:\n        raise RuntimeError"),
 'H94 GNN refit judged by its recorded mode and epochs only': (
     'rolling_protocol.py', REFIT_CHECK,
     "        if trained.get('mode') != 'fixed_epochs' or trained.get('chosen_epochs') != spec['epochs']:"),
 'H95 cache audit does not compare MF fit data': (
     'report_tools/check_cached_runs.py',
     "            if protocol_id(source) is None or protocol_id(directory) != protocol_id(source):\n"
     "                differs.append('data')\n            mf.append", "            mf.append"),
}
git = lambda *a: subprocess.run(['git', *a], cwd=src, capture_output=True, encoding='utf-8').stdout.rstrip('\n')
def fresh(tmp):
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
        'test_cycle11_guards.py', 'test_cycle12_guards.py', 'test_cycle13_guards.py', 'test_cycle14_guards.py', 'test_cycle15_guards.py', 'test_cycle16_guards.py', 'test_cycle17_guards.py', 'test_cycle18_guards.py', 'test_cycle19_guards.py', 'test_cycle20_guards.py', 'test_cycle21_guards.py', 'test_cycle22_guards.py', 'test_compare_outputs.py', '--deselect', 'test_cycle4_guards.py::test_verify_plan_code_reports_changes_and_fails_only_when_asked',
        '--deselect', 'test_cycle10_guards.py::test_exact_reproduction_from_raw_reaches_its_test_opening']
def run_tests(d):
    # pytest's output is read as UTF-8 whatever the console code page: one byte the locale codec cannot decode in a
    # failure message would otherwise drop the whole output and turn a kill into an abort.
    return subprocess.run(args, cwd=d, capture_output=True, encoding='utf-8', errors='replace',
                          env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})


def summary(stdout):
    lines = (stdout or '').strip().splitlines()
    return lines[-1] if lines else ''


def classify(returncode, stdout):
    # pytest exits with 1 only when tests failed or errored; any other non-zero code (interrupt, collection error,
    # a process killed from outside) proves nothing about the mutant.
    if returncode == 0:
        return 'survived'
    if returncode == 1 and re.search(r'\b\d+ (failed|errors?)\b', summary(stdout)):
        return 'killed'
    return 'aborted'


def main():
    run_info = {'head': git('rev-parse', 'HEAD'), 'dirty': git('status', '--porcelain', '--', '.').splitlines(),
                'started_utc': datetime.now(timezone.utc).isoformat()}
    print('start', json.dumps(run_info), flush=True)
    tmp = Path(tempfile.mkdtemp(prefix='h_mut_'))
    rows = []
    d = fresh(tmp); r = run_tests(d); base = summary(r.stdout); print('unmutated', base, flush=True)
    assert r.returncode == 0 and re.search(r'\b\d+ passed\b', base), (r.returncode, base, r.stderr[-2000:])
    for name, (f, old, new) in M.items():
        d = fresh(tmp); t = (d / f).read_text(encoding='utf-8'); assert t.count(old) == 1, (name, t.count(old))
        (d / f).write_text(t.replace(old, new, 1), encoding='utf-8')
        r = run_tests(d)
        rows.append({'mutant': name, 'file': f, 'result': classify(r.returncode, r.stdout), 'returncode': r.returncode,
                     'summary': summary(r.stdout)})
        print(json.dumps(rows[-1]), flush=True)
    shutil.rmtree(tmp)
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else src / 'audit_runs' / 'guard_mutants.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    run_info['finished_utc'] = datetime.now(timezone.utc).isoformat()
    out.write_text(json.dumps({**run_info, 'unmutated': base, 'tests': args[5:], 'mutants': rows}, indent=2), encoding='utf-8')
    counts = {k: sum(row['result'] == k for row in rows) for k in ('killed', 'survived', 'aborted')}
    print('result', json.dumps(counts), flush=True)
    return 0 if counts['killed'] == len(rows) else 1


if __name__ == '__main__':
    sys.exit(main())
