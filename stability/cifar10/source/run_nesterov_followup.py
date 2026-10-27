#!/usr/bin/env python3
"""Add ten independent seeds (23..32) at 25 fixed-B=4 Nesterov settings.

python3 run_nesterov_followup.py --dry-run
python3 run_nesterov_followup.py --max-new-trials 2
python3 run_nesterov_followup.py
python3 run_nesterov_followup.py --status
python3 run_nesterov_followup.py --export

250 trials, unchanged K=2000 protocol, two GPUs, resume supported.
Existing revised trainer and output/summary_gap9/merged_cells.csv required.
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
DEFAULT_OUTPUT = 'output/nesterov_followup_20260921'
SEEDS = list(range(23,33))
TARGETS = [
    ('gap32sqrt2',1/(32*math.sqrt(2)),[1.,1.414214,2.,2.828427]),
    ('gap32',1/32,[.707107,.8408965453609618,1.,1.414214]),
    ('gap16sqrt2',1/(16*math.sqrt(2)),[.707107,1.,1.414214,2.,2.828427,4.]),
    ('gap16',1/16,[.707107,.8408965453609618,1.,1.189207,1.414214,2.]),
    ('gap4sqrt2',1/(4*math.sqrt(2)),[.707107,1.,1.414214,2.,2.828427]),
]


def groups():
    return [dict(name=f'{i:02d}_nesterov_{name}',algorithm='nesterov',axis='gap',
                 batch=4,gap=gap,steps=2000,lrs=lrs,seeds=SEEDS)
            for i,(name,gap,lrs) in enumerate(TARGETS,1)]


def save_baseline(args,root):
    snapshot=root/'_baseline'/'merged_cells.csv'
    if not snapshot.exists():
        source=args.baseline_summary.expanduser().resolve()
        if source.is_dir():source=source/'merged_cells.csv'
        if not source.is_file():
            raise ValueError('Missing output/summary_gap9/merged_cells.csv; run the previous gap9 plotting code first, or pass --baseline-summary')
        snapshot.parent.mkdir(exist_ok=True)
        shutil.copy2(source,snapshot)
    payload=snapshot.read_bytes()
    rows=list(csv.DictReader(io.StringIO(payload.decode())))
    selected=[]
    for g in groups():
        for lr in g['lrs']:
            matches=[r for r in rows if r['axis']=='gap' and r['algorithm']=='nesterov'
                     and math.isclose(float(r['x']),g['gap'],rel_tol=1e-10)
                     and math.isclose(float(r['lr']),lr,rel_tol=1e-8)]
            if len(matches)!=1:raise ValueError(f'Missing/duplicate baseline cell: gap={g["gap"]}, lr={lr}')
            r=matches[0]
            if int(r['K'])!=2000 or int(r['n_unknown']):raise ValueError('Baseline protocol/incomplete data')
            seeds={int(v) for v in r['seed_ids'].split(',')}
            if seeds & set(SEEDS):raise ValueError('New seeds overlap baseline; do not merge twice')
            selected.append(dict(gap=g['gap'],lr=lr,old_stable=r['n_stable'],old_total=r['n_total'],
                                 old_seed_ids=r['seed_ids'],new_seeds=','.join(map(str,SEEDS))))
    with (root/'target_cells_before.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(selected[0]));w.writeheader();w.writerows(selected)
    return dict(baseline_sha256=hashlib.sha256(payload).hexdigest(),
                policy='append disjoint seeds; never replace old outcomes',
                fixed_batch=4,steps=2000,target_cells=len(selected),seeds=SEEDS)


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
    reuse = save_baseline(args, root)
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
    if len(rows) != 250:
        raise ValueError(f'Unexpected trial count: {len(rows)}')
    plan = dict(schema_version=1, round='nesterov_followup_20260921',
                training_code_hash=EXPECTED_TRAINING_HASH,
                trainer_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                parameter_safety_mode='record', old_results_policy='retain_all',
                plotting_views=reuse,
                trials=len(rows), updates=sum(r['steps'] for r in rows),
                training_samples=sum(r['steps'] * r['batch'] for r in rows),
                groups=plan_groups)
    plan_path = root / 'followup_plan.json'
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
    with (root / 'followup_plan.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    launcher_snapshot = root / '_code' / 'run_nesterov_followup.py'
    if not launcher_snapshot.exists():
        shutil.copy2(Path(__file__).resolve(), launcher_snapshot)
    return module, plan


def saved_plan(root):
    plan = json.loads((root / 'followup_plan.json').read_text())
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
    destination = root.parent / f'cifar_nesterov_followup_results_{stamp}.zip'
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
    parser.add_argument('--baseline-summary', type=Path, default=Path('output/summary_gap9'),
                        help='Previous nine-point audited summary folder')
    parser.add_argument('--trainer', help='Existing revised training .py; auto-detected by default')
    parser.add_argument('--output', type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument('--data-root', type=Path,
                        default=Path('/data/public/CIFAR-10/cifar-10-python'))
    parser.add_argument('--devices', default='cuda:0,cuda:1')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true', help='Save plans; no torch import or training')
    modes.add_argument('--status', action='store_true', help='Read validated completed-trial counts')
    modes.add_argument('--export', action='store_true', help='Export current round for later merging')
    parser.add_argument('--max-new-trials', type=int, help='Global cap across ALL five groups')
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
    with (root / '.followup.lock').open('a') as lock:
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
            print(f'No training performed. Plan: {root / "followup_plan.csv"}', flush=True)
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
