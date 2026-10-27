#!/usr/bin/env python3
"""Add four geometric midpoint gaps to fixed-B=4 CIFAR-10 stability scans.

python3 run_gap9_midpoints.py --dry-run
python3 run_gap9_midpoints.py --max-new-trials 2
python3 run_gap9_midpoints.py
python3 run_gap9_midpoints.py --status
python3 run_gap9_midpoints.py --export

472 new Polyak/Nesterov trials. Reuse SGD B=4 observations for all nine gap
positions. Existing revised trainer and output/summary_round2 are required.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

EXPECTED_TRAINING_HASH = '388b650f262371ccfb627e5593bfcdbda9e4a66a813b9b0f6b0fe30e724cb7ca'
DEFAULT_OUTPUT = 'output/gap9_midpoints_20260921'
BASE_LRS = [0.00390625, 0.00552427, 0.0078125, 0.0110485, 0.015625,
            0.0220971, 0.03125, 0.0441942, 0.0625, 0.0883883, 0.125,
            0.176777, 0.25, 0.353553, 0.5, 0.707107, 1.0, 1.414214,
            2.0, 2.828427, 4.0, 5.656854, 8.0]
BOUNDARY_LRS = [0.25, 0.353553, 0.5, 0.707107, 1.0, 1.414214, 2.0, 2.828427, 4.0]
MIDPOINTS = [(d, 1/(d*math.sqrt(2))) for d in (32,16,8,4)]
GAP_GRID = sorted([1/64,1/32,1/16,1/8,1/4] + [g for d,g in MIDPOINTS])


def gap_label(value):
    for d,g in MIDPOINTS:
        if math.isclose(value,g,rel_tol=1e-12):return f'1/({d}*sqrt(2))'
    return f'1/{round(1/value)}'


def groups():
    result=[]
    for algorithm in ('polyak','nesterov'):
        for denominator,gap in MIDPOINTS:
            for suffix,lrs,seeds in [('full_seed0',BASE_LRS,[0]),
                                      ('boundary_seed1_4',BOUNDARY_LRS,[1,2,3,4])]:
                result.append(dict(name=f'{len(result)+1:02d}_{algorithm}_gap1over{denominator}sqrt2_{suffix}',
                    algorithm=algorithm,axis='gap',batch=4,gap=gap,steps=2000,lrs=lrs,seeds=seeds))
    return result


def save_sgd_reuse(args,root):
    """Replicate display positions, never count them as independent SGD trials."""
    snapshot=root/'_baseline'/'merged_cells.csv'
    if not snapshot.exists():
        source=args.baseline_summary.expanduser().resolve()
        if source.is_dir():source=source/'merged_cells.csv'
        if not source.is_file():
            raise ValueError('Missing old merged_cells.csv; pass --baseline-summary /path/to/summary_round2')
        snapshot.parent.mkdir(exist_ok=True)
        shutil.copy2(source,snapshot)
    payload=snapshot.read_bytes()
    all_rows=list(csv.DictReader(io.StringIO(payload.decode())))
    rows=[r for r in all_rows if r['algorithm']=='sgd' and r['axis']=='batch' and int(r['batch'])==4]
    if not rows:raise ValueError('Baseline summary lacks SGD B=4 batch-axis observations')
    if len({float(r['lr']) for r in rows})!=len(rows):raise ValueError('Duplicate baseline SGD LR cells')
    output=[]
    for row in rows:
        if float(row['gap'])!=1 or int(row['K'])!=2000:raise ValueError('Unexpected baseline SGD protocol')
        if int(row['n_unknown']) or int(row['n_assessed'])!=int(row['n_total']):
            raise ValueError('Baseline SGD has incomplete cells')
        for gap in GAP_GRID:
            output.append(dict(row,axis='gap',x=gap,axis_label=gap_label(gap),
                               reused_physical_configuration='sgd_B4_no_momentum',
                               is_independent_repeat=False))
    with (root/'sgd_gap9_reused_cells.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    info=dict(fixed_batch=4,gaps=GAP_GRID,gap_labels=[gap_label(g) for g in GAP_GRID],
              new_gaps=[g for d,g in MIDPOINTS],
              sgd_new_training_trials=0,sgd_reused_lr_cells=len(rows),
              sgd_display_cells=len(output),
              baseline_sha256=hashlib.sha256(payload).hexdigest(),
              sgd_semantics='Same physical B=4 trials shown at nine x positions; do not count nine independent repeats',
              gap128_included=False)
    (root/'gap9_views.json').write_text(json.dumps(info,indent=2)+'\n')
    return info


def load_trainer(path):
    spec = importlib.util.spec_from_file_location('cifar_gap9_training_backend', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.training_code_hash() != EXPECTED_TRAINING_HASH:
        raise ValueError(f'{path}: training code differs from the audited revised trainer')
    if '--parameter-safety-mode' not in module.build_parser().format_help():
        raise ValueError('The revised trainer with --parameter-safety-mode is required')
    return module


def choose_trainer(requested):
    candidates = [Path(requested)] if requested else [
        Path.cwd() / 'cifar_stability_revised.py', Path.cwd() / 'cifar_stability.py',
        Path(__file__).resolve().parent / 'cifar_stability_revised.py',
        Path(__file__).resolve().parent / 'cifar_stability.py']
    errors = []
    for path in dict.fromkeys(p.resolve() for p in candidates):
        if path.is_file():
            try:
                load_trainer(path)
                return path
            except Exception as exc:
                errors.append(str(exc))
    raise ValueError('Cannot find the revised training script. Pass --trainer /path/to/'
                     'cifar_stability_revised.py.\n' + '\n'.join(errors))


def csv_text(values):
    return ','.join(str(value) for value in values)


def command_args(g, args, root):
    return ['--preset', 'pilot', '--algorithms', g['algorithm'], '--axis', g['axis'],
            '--batches', str(g['batch']), '--fixed-batch', str(g['batch']),
            '--gaps', str(g['gap']), '--fixed-gap', str(g['gap']),
            '--lrs', csv_text(g['lrs']), '--seeds', csv_text(g['seeds']),
            '--steps', str(g['steps']), '--eval-every', '50', '--tail-evals', '10',
            '--probe-size', '1024', '--probe-batch', '256', '--probe-seed', '1729',
            '--stable-loss-ratio', '2', '--growth-ratio', '1.25',
            '--learning-ratio', '0.95', '--divergence-loss-ratio', '100',
            '--max-grad-norm', '1000000', '--max-parameter-ratio', '100',
            '--parameter-safety-mode', 'record', '--required-fraction', '1',
            '--model', 'resnet18', '--max-micro-batch', '64', '--workers', '4',
            '--threads', '4', '--devices', args.devices, '--data-root', str(args.data_root),
            '--output', str(root / g['name']), '--no-plot']


def prepare(args, root):
    snapshot = root / '_code' / 'cifar_stability.py'
    if not snapshot.exists():
        source = choose_trainer(args.trainer)
        snapshot.parent.mkdir(exist_ok=True)
        shutil.copy2(source, snapshot)
    module = load_trainer(snapshot)
    reuse = save_sgd_reuse(args, root)
    plan_groups = []
    rows = []
    seen = set()
    for g in groups():
        argv = command_args(g, args, root)
        manifest = module.make_manifest(module.parse_args(argv))
        for trial in manifest['jobs']:
            key = (g['steps'], trial['algorithm'], trial['batch'], trial['gap'],
                   trial['lr'], trial['seed'])
            if key in seen:
                raise ValueError(f'Duplicate trial in gap9: {key}')
            seen.add(key)
            rows.append(dict(group=g['name'], steps=g['steps'], **trial))
        plan_groups.append(dict(**g, argv=argv, count=len(manifest['jobs']),
                                manifest_fingerprint=manifest['fingerprint']))
    if len(rows) != 472:
        raise ValueError(f'Unexpected trial count: {len(rows)}')
    plan = dict(schema_version=1, round='gap9_midpoints_20260921',
                training_code_hash=EXPECTED_TRAINING_HASH,
                trainer_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                parameter_safety_mode='record', old_results_policy='retain_all',
                plotting_views=reuse,
                trials=len(rows), updates=sum(r['steps'] for r in rows),
                training_samples=sum(r['steps'] * r['batch'] for r in rows),
                groups=plan_groups)
    plan_path = root / 'gap9_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Saved plan differs. Keep the same arguments to resume, '
                         'or use a new --output directory.')
    # Persist the plan and all group manifests before starting any GPU jobs.
    module.atomic_json_dump(plan, plan_path)
    for g in plan_groups:
        subroot = root / g['name']
        with module.output_lock(subroot):
            requested = module.make_manifest(module.parse_args(g['argv']))
            if (subroot / 'manifest.json').exists():
                module.select_resume_manifest(module.read_manifest(subroot), requested)
            else:
                module.atomic_json_dump(requested, subroot / 'manifest.json')
    with (root / 'gap9_plan.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    launcher_snapshot = root / '_code' / 'run_gap9_midpoints.py'
    if not launcher_snapshot.exists():
        shutil.copy2(Path(__file__).resolve(), launcher_snapshot)
    return module, plan


def saved_plan(root):
    plan = json.loads((root / 'gap9_plan.json').read_text())
    snapshot = root / '_code' / 'cifar_stability.py'
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != plan['trainer_sha256']:
        raise ValueError('Saved training source was modified')
    return load_trainer(snapshot), plan


def progress(module, root, plan, show=True):
    total_done = 0
    for g in plan['groups']:
        subroot = root / g['name']
        manifest = module.read_manifest(subroot)
        if manifest['fingerprint'] != g['manifest_fingerprint']:
            raise ValueError(f'Manifest changed: {subroot}')
        done = sum(module.read_result(subroot, manifest, t) is not None
                   for t in manifest['jobs'])
        total_done += done
        if show:
            print(f"{g['name']}: {done}/{g['count']}", flush=True)
    if show:
        print(f"TOTAL: {total_done}/{plan['trials']} completed; "
              f"{plan['trials'] - total_done} remaining", flush=True)
    return total_done


def run_pending(module, root, plan, max_new_trials):
    baseline = progress(module, root, plan, show=False)
    for g in plan['groups']:
        completed = progress(module, root, plan, show=False)
        budget = None if max_new_trials is None else max_new_trials - (completed - baseline)
        if budget is not None and budget <= 0:
            break
        subroot = root / g['name']
        manifest = module.read_manifest(subroot)
        pending = sum(module.read_result(subroot, manifest, t) is None
                      for t in manifest['jobs'])
        if not pending:
            continue
        command = [sys.executable, '-u', str(root / '_code' / 'cifar_stability.py'), *g['argv']]
        if budget is not None:
            command += ['--max-new-trials', str(min(budget, pending))]
        print(f"\n===== {g['name']} ({pending} remaining) =====", flush=True)
        env = dict(os.environ, PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
        result = subprocess.run(command, env=env)
        if result.returncode:
            raise RuntimeError(f"{g['name']} exited with {result.returncode}. "
                               'Completed trials are saved. Inspect this group\'s logs and resume.')
        if budget is None and any(module.read_result(subroot, manifest, t) is None
                                  for t in manifest['jobs']):
            raise RuntimeError(f"{g['name']} returned success with unfinished trials")
    progress(module, root, plan)


def export_results(root, plan):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%fZ')
    destination = root.parent / f'cifar_gap9_results_{stamp}.zip'
    # The archive preserves output/<round>/<group>/trials/<id>/... for merging.
    # Full histories, manifests, CSVs, source snapshots, and logs are included.
    files = [p for p in root.rglob('*') if p.is_file()
             and '__pycache__' not in p.parts and not p.name.startswith('.')
             and p.suffix in {'.json', '.jsonl', '.csv', '.py', '.log', '.md'}]
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for path in sorted(files):
            archive.write(path, str(Path('output') / root.name / path.relative_to(root)))
    print(f'Exported: {destination}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--baseline-summary', type=Path, default=Path('output/summary_round2'),
                        help='Previous audited summary folder (used only to reuse SGD)')
    parser.add_argument('--trainer', help='Existing revised training .py; auto-detected by default')
    parser.add_argument('--output', type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument('--data-root', type=Path,
                        default=Path('/data/public/CIFAR-10/cifar-10-python'))
    parser.add_argument('--devices', default='cuda:0,cuda:1')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true', help='Save plans; no torch import or training')
    modes.add_argument('--status', action='store_true', help='Read validated completed-trial counts')
    modes.add_argument('--export', action='store_true', help='Export current round for later merging')
    parser.add_argument('--max-new-trials', type=int, help='Global cap across ALL sixteen groups')
    args = parser.parse_args()
    if args.max_new_trials is not None and args.max_new_trials < 1:
        parser.error('--max-new-trials must be positive')
    root = args.output.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    if args.status:
        module, plan = saved_plan(root)
        progress(module, root, plan)
        return 0
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.gap9.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another launcher is using this output. Use --status to check it.')
        if args.export:
            module, plan = saved_plan(root)
            progress(module, root, plan)
            export_results(root, plan)
            return 0
        module, plan = prepare(args, root)
        print(f"Plan: {plan['trials']} new trials; up to {plan['updates']:,} updates; "
              f"{plan['training_samples']:,} training samples (+ probes).", flush=True)
        if args.dry_run:
            progress(module, root, plan)
            print(f'No training performed. Plan: {root / "gap9_plan.csv"}', flush=True)
        else:
            run_pending(module, root, plan, args.max_new_trials)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nInterrupted; completed trials are saved. Rerun the same command.', file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
