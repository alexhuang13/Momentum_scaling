#!/usr/bin/env python3
"""Merge CIFAR-10 gap9 or Nesterov follow-up results and redraw all panels.

python3 plot_cifar_all.py --gap9-dir output/gap9_midpoints_20260921 --individual
python3 plot_cifar_all.py --gap9-zip cifar_gap9_results_20260921_140050_075285Z.zip

python3 plot_cifar_all.py --followup-dir output/nesterov_followup_20260921 --individual

Requires numpy and matplotlib. No training, extraction or source-file mutation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import io
import zipfile
import statistics
from contextlib import nullcontext
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import NullLocator, FixedLocator, FuncFormatter, LogLocator
import numpy as np

ALGORITHMS = ['sgd', 'polyak', 'nesterov']
NAMES = {'sgd': 'SGD', 'polyak': 'Polyak', 'nesterov': 'Nesterov'}
COLORS = {'sgd': 'C0', 'polyak': 'C1', 'nesterov': 'C2'}
MARKERS = {'sgd': 'o', 'polyak': '^', 'nesterov': 's'}
CONFIG = {
    'text.usetex': False, 'font.family': 'sans-serif',
    'axes.titlesize': 16, 'axes.labelsize': 16, 'lines.linewidth': 2,
    'lines.markersize': 6, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'figure.figsize': (5, 4), 'legend.fontsize': 10,
    'legend.handlelength': 1, 'axes.grid': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
}
plt.rcParams.update(CONFIG)

# Display controls only; measured fractions and fitted boundaries are unchanged.
GAP_YMIN = 0.0625
BOUNDARY_YMIN = 0.0625
STABILITY_CMAP = LinearSegmentedColormap.from_list(
    'stability_yellow',
    [(0.00, '#00204c'),
     (0.25, '#414d6b'),
     (0.50, '#99976c'),
     (0.70, '#d5c75a'),
     (0.85, '#efdc46'),
     (1.00, '#fee838')],
    N=256,
)
# Linear normalization and one shared palette: a given fraction always has
# the same color, regardless of its position relative to the fitted curve.



def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def resolve_csv(path):
    path = path.expanduser().resolve()
    options = [path] if path.is_file() else [path / 'merged_cells.csv',
                path / 'output' / 'summary_round2' / 'merged_cells.csv']
    for candidate in options:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError('Cannot find merged_cells.csv. Pass --summary-dir '
                            '/path/to/output/summary_round2 (the previous audited summary).')


def load_cells(path):
    with (path.open(newline='') if isinstance(path, Path) else nullcontext(path)) as stream:
        reader = csv.DictReader(stream)
        required = {'axis', 'algorithm', 'x', 'batch', 'gap', 'K', 'lr',
                    'n_total', 'n_stable', 'n_unstable', 'n_unknown', 'n_assessed'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError('Missing columns: ' + str(sorted(required-set(reader.fieldnames or []))))
        raw = list(reader)
    cells, excluded, seen = [], [], set()
    for row in raw:
        for key in ('batch', 'K', 'n_total', 'n_stable', 'n_unstable', 'n_unknown', 'n_assessed'):
            row[key] = int(row[key])
        for key in ('x', 'gap', 'lr'):
            row[key] = float(row[key])
            if not math.isfinite(row[key]) or row[key] <= 0:
                raise ValueError('Invalid coordinate')
        if row['algorithm'] not in ALGORITHMS or row['axis'] not in ('batch', 'gap'):
            raise ValueError('Unrecognized algorithm/axis')
        key = tuple(row[k] for k in ('axis', 'algorithm', 'x', 'K', 'lr'))
        if key in seen:
            raise ValueError('Duplicate heatmap cell: ' + str(key))
        seen.add(key)
        if any(row[k] < 0 for k in ('n_total', 'n_stable', 'n_unstable', 'n_unknown', 'n_assessed')):
            raise ValueError('Negative seed count')
        if row['n_total'] != row['n_stable'] + row['n_unstable'] + row['n_unknown'] or \
           row['n_assessed'] != row['n_stable'] + row['n_unstable']:
            raise ValueError('Inconsistent seed counts')
        row['fraction'] = row['n_stable']/row['n_assessed'] if row['n_assessed'] else None
        omit = math.isclose(row['gap'], 1/128, rel_tol=1e-10) or \
               (row['axis'] == 'gap' and math.isclose(row['x'], 1/128, rel_tol=1e-10))
        if omit:
            excluded.append(row)
            continue
        if row['K'] != 2000 or row['n_unknown'] or not row['n_assessed']:
            raise ValueError('Retained cells must have K=2000 and fully assessed seeds')
        if row['axis']=='batch' and row['algorithm']!='sgd' and row['gap']!=1/64:
            raise ValueError('Batch panels require fixed gap=1/64')
        if row['axis']=='gap' and row['batch']!=4:
            raise ValueError('Gap panels require fixed B=4')
        cells.append(row)
    if {(r['axis'], r['algorithm']) for r in cells} != {(a,m) for a in ('batch','gap') for m in ALGORITHMS}:
        raise ValueError('Expected all six panels')
    return cells, excluded


GAPS9 = [2.0**(-6+i/2) for i in range(9)]
NEW_GAPS = GAPS9[1::2]


def gap_label(x):
    i = min(range(9), key=lambda i: abs(math.log(x/GAPS9[i])))
    if not math.isclose(x, GAPS9[i], rel_tol=1e-10):
        return f'{x:.4g}'
    if i % 2 == 0:
        return f'1/{2**(6-i//2)}'
    return rf'$1/({2**(5-i//2)}\sqrt{{2}})$'


class GapBundle:
    """Read a result directory or ZIP without extracting/executing uploaded code."""
    def __init__(self, path, marker="gap9_views.json"):
        self.path = path.expanduser().resolve()
        self.archive = None
        if self.path.is_file():
            self.archive = zipfile.ZipFile(self.path)
            roots = [n[:-len(marker)] for n in self.archive.namelist()
                     if n.endswith(marker)]
            if len(roots) != 1:
                self.archive.close()
                raise ValueError('ZIP must contain exactly one gap9_views.json')
            self.prefix = roots[0]
        else:
            self.prefix = ''
            if not (self.path/marker).is_file():
                raise FileNotFoundError('Pass --gap9-dir with the completed gap9 directory, or --gap9-zip FILE.zip')

    def read(self, name):
        if self.archive:
            return self.archive.read(self.prefix+name)
        return (self.path/name).read_bytes()

    def json(self, name):
        return json.loads(self.read(name))

    def manifests(self):
        if self.archive:
            return sorted(n[len(self.prefix):] for n in self.archive.namelist()
                          if n.startswith(self.prefix) and n.endswith('/manifest.json')
                          and n[len(self.prefix):].count('/') == 1)
        return sorted(str(p.relative_to(self.path)) for p in self.path.glob('*/manifest.json'))

    def close(self):
        if self.archive:
            self.archive.close()


def load_gap9(bundle, summary_override=None):
    views = bundle.json('gap9_views.json')
    baseline_bytes = bundle.read('_baseline/merged_cells.csv')
    sha = hashlib.sha256(baseline_bytes).hexdigest()
    if sha != views['baseline_sha256']:
        raise ValueError('Baseline snapshot hash does not match gap9_views.json')
    if summary_override is not None:
        override = resolve_csv(summary_override).read_bytes()
        # CSV line endings do not change the data.
        if list(csv.DictReader(io.StringIO(override.decode()))) != list(csv.DictReader(io.StringIO(baseline_bytes.decode()))):
            raise ValueError('--summary-dir differs from the baseline used for this experiment')
    cells, excluded = load_cells(io.StringIO(baseline_bytes.decode()))
    if not np.allclose(views['gaps'], GAPS9, rtol=1e-12, atol=0):
        raise ValueError('Unexpected gap grid')
    manifests = bundle.manifests()
    if len(manifests) != 16:
        raise ValueError('Expected 16 gap9 experiment groups')
    seen = set(); records = []; grouped = defaultdict(list)
    required = dict(steps=2000, tail_evals=10, eval_every=50, stable_loss_ratio=2.,
                    growth_ratio=1.25, fake_data=False, parameter_safety_mode='record',
                    probe_size=1024, probe_seed=1729, model='resnet18',
                    divergence_loss_ratio=100., max_grad_norm=1e6)
    failures = {'parameter_safety_limit', 'gradient_safety_limit', 'probe_loss_safety_limit',
                'nonfinite_training_loss_or_gradient', 'nonfinite_parameters', 'nonfinite_probe_loss'}
    for name in manifests:
        manifest = bundle.json(name); config = manifest['config']
        if any(config.get(k) != v for k,v in required.items()):
            raise ValueError('Unexpected training protocol: '+name)
        if manifest.get('training_code_hash') != '388b650f262371ccfb627e5593bfcdbda9e4a66a813b9b0f6b0fe30e724cb7ca':
            raise ValueError('Training code changed: '+name)
        parent = name.rsplit('/',1)[0]
        for job in manifest['jobs']:
            prefix = parent+'/trials/'+job['id']+'/'
            s = bundle.json(prefix+'summary.json')
            tc = bundle.json(prefix+'config.json')
            if s['trial'] != job or tc['trial'] != job or s['fingerprint'] != manifest['fingerprint'] or tc['fingerprint'] != manifest['fingerprint']:
                raise ValueError('Trial identity mismatch: '+prefix)
            key = tuple(job[k] for k in ('algorithm','batch','gap','lr','seed'))
            if key in seen:
                raise ValueError('Duplicate physical trial: '+str(key))
            seen.add(key)
            if job['algorithm'] not in ('polyak','nesterov') or job['batch'] != 4 or not any(math.isclose(job['gap'],g,rel_tol=1e-12) for g in NEW_GAPS):
                raise ValueError('Unexpected new configuration')
            h = [json.loads(line) for line in bundle.read(prefix+'history.jsonl').splitlines() if line.strip()]
            if not h or h[0]['step'] != 0 or h[0]['probe_loss'] != s['initial_probe_loss']:
                raise ValueError('Invalid initialization: '+prefix)
            reason = s['stop_reason']; loss = growth = None
            if reason in failures:
                stable = False
            elif reason:
                raise ValueError('Unknown stopping reason: '+reason)
            else:
                probes = [r for r in h if r['step']>0 and 'probe_loss' in r]
                if s['updates_completed'] != 2000 or [r['step'] for r in probes] != list(range(50,2001,50)):
                    raise ValueError('Incomplete trial: '+prefix)
                losses = [r['probe_loss'] for r in probes]
                if any(v is None or not math.isfinite(v) for v in losses):
                    raise ValueError('Nonfinite completed history')
                if losses[-10:] != s['tail_probe_losses'] or losses[-20:-10] != s['previous_probe_losses']:
                    raise ValueError('Tail/history mismatch')
                loss = statistics.fmean(losses[-10:])/s['initial_probe_loss']
                growth = statistics.fmean(losses[-10:])/max(statistics.fmean(losses[-20:-10]),1e-30)
                stable = loss <= 2 and growth <= 1.25
            if stable != s['bounded']:
                raise ValueError('Recomputed label differs: '+prefix)
            r = dict(job, K=2000, stable=stable, stop_reason=reason,
                     tail_loss_ratio=loss, tail_growth_ratio=growth, source=prefix)
            records.append(r)
            grouped[(job['algorithm'], job['gap'], job['lr'])].append(r)
    if len(records) != 472 or len(grouped) != 184:
        raise ValueError(f'Incomplete grid: {len(records)}/472 trials, {len(grouped)}/184 LR cells')
    baseline_columns={(r['algorithm'],r['x']) for r in cells if r['axis']=='gap'}
    for (alg,gap,lr), rows in grouped.items():
        seeds = sorted(r['seed'] for r in rows)
        if seeds != ([0,1,2,3,4] if .25 <= lr <= 4 else [0]):
            raise ValueError('Unexpected seed coverage')
        if any(a==alg and math.isclose(x,gap,rel_tol=1e-12) for a,x in baseline_columns):
            raise ValueError('New gap already exists in baseline; refusing double count')
        n=len(rows); stable=sum(r['stable'] for r in rows)
        cells.append(dict(axis='gap',algorithm=alg,x=gap,batch=4,gap=gap,K=2000,lr=lr,
                          n_total=n,n_stable=stable,n_unstable=n-stable,n_unknown=0,n_assessed=n,
                          stable_fraction_assessed=stable/n,fraction_lower_bound=stable/n,
                          fraction_upper_bound=stable/n,n_revised=n,n_legacy_parameter_stop_failures=0,
                          seed_ids=','.join(map(str,seeds)),unknown_seed_ids='',fraction=stable/n,
                          source_kind='gap9_new_trials'))
    # SGD is physically independent of the display-only momentum coordinate.
    sgd = [r for r in cells if r['axis']=='batch' and r['algorithm']=='sgd' and r['batch']==4]
    if not sgd:
        raise ValueError('Missing physical SGD B=4 baseline')
    old_sgd = [r for r in cells if r['axis']=='gap' and r['algorithm']=='sgd']
    for row in old_sgd:
        matching = [r for r in sgd if r['lr']==row['lr']]
        if len(matching)!=1 or any(row[k]!=matching[0][k] for k in ('n_total','n_stable','n_unstable','seed_ids')):
            raise ValueError('SGD views disagree with physical B=4 trials')
    cells = [r for r in cells if not (r['axis']=='gap' and r['algorithm']=='sgd')]
    for gap in GAPS9:
        for r in sgd:
            cells.append(dict(r, axis='gap', x=gap, gap=1., source_kind='sgd_B4_reused_not_independent'))
    for alg in ALGORITHMS:
        xs=sorted({r['x'] for r in cells if r['axis']=='gap' and r['algorithm']==alg})
        if len(xs)!=9 or not np.allclose(xs,GAPS9,rtol=1e-12,atol=0):
            raise ValueError('Each fixed-B panel must have nine gap columns')
    report=dict(baseline_sha256=sha, new_trials=len(records),new_stable=sum(r['stable'] for r in records),
                new_unstable=sum(not r['stable'] for r in records),new_lr_cells=len(grouped),
                sgd_reused_physical_lr_cells=len(sgd),sgd_new_trials=0,gaps=GAPS9)
    return cells,excluded,records,report


def load_followup(bundle):
    plan=bundle.json('followup_plan.json')
    payload=bundle.read('_baseline/merged_cells.csv')
    if hashlib.sha256(payload).hexdigest()!=plan['plotting_views']['baseline_sha256']:
        raise ValueError('Follow-up baseline hash mismatch')
    cells,excluded=load_cells(io.StringIO(payload.decode()))
    manifests=bundle.manifests()
    expected={g['name']+'/manifest.json':g for g in plan['groups']}
    if set(manifests)!=set(expected) or plan['trials']!=250:
        raise ValueError('Unexpected follow-up plan')
    for name in manifests:
        manifest=bundle.json(name)
        g=expected[name]
        if manifest['fingerprint']!=g['manifest_fingerprint'] or len(manifest['jobs'])!=g['count']:
            raise ValueError('Follow-up manifest differs from saved plan')
    seen = set(); records = []; grouped = defaultdict(list)
    required = dict(steps=2000, tail_evals=10, eval_every=50, stable_loss_ratio=2.,
                    growth_ratio=1.25, fake_data=False, parameter_safety_mode='record',
                    probe_size=1024, probe_seed=1729, model='resnet18',
                    divergence_loss_ratio=100., max_grad_norm=1e6)
    failures = {'parameter_safety_limit', 'gradient_safety_limit', 'probe_loss_safety_limit',
                'nonfinite_training_loss_or_gradient', 'nonfinite_parameters', 'nonfinite_probe_loss'}
    for name in manifests:
        manifest = bundle.json(name); config = manifest['config']
        if any(config.get(k) != v for k,v in required.items()):
            raise ValueError('Unexpected training protocol: '+name)
        if manifest.get('training_code_hash') != '388b650f262371ccfb627e5593bfcdbda9e4a66a813b9b0f6b0fe30e724cb7ca':
            raise ValueError('Training code changed: '+name)
        parent = name.rsplit('/',1)[0]
        for job in manifest['jobs']:
            prefix = parent+'/trials/'+job['id']+'/'
            s = bundle.json(prefix+'summary.json')
            tc = bundle.json(prefix+'config.json')
            if s['trial'] != job or tc['trial'] != job or s['fingerprint'] != manifest['fingerprint'] or tc['fingerprint'] != manifest['fingerprint']:
                raise ValueError('Trial identity mismatch: '+prefix)
            key = tuple(job[k] for k in ('algorithm','batch','gap','lr','seed'))
            if key in seen:
                raise ValueError('Duplicate physical trial: '+str(key))
            seen.add(key)
            if job['algorithm'] != 'nesterov' or job['batch'] != 4:
                raise ValueError('Unexpected new configuration')
            h = [json.loads(line) for line in bundle.read(prefix+'history.jsonl').splitlines() if line.strip()]
            if not h or h[0]['step'] != 0 or h[0]['probe_loss'] != s['initial_probe_loss']:
                raise ValueError('Invalid initialization: '+prefix)
            reason = s['stop_reason']; loss = growth = None
            if reason in failures:
                stable = False
            elif reason:
                raise ValueError('Unknown stopping reason: '+reason)
            else:
                probes = [r for r in h if r['step']>0 and 'probe_loss' in r]
                if s['updates_completed'] != 2000 or [r['step'] for r in probes] != list(range(50,2001,50)):
                    raise ValueError('Incomplete trial: '+prefix)
                losses = [r['probe_loss'] for r in probes]
                if any(v is None or not math.isfinite(v) for v in losses):
                    raise ValueError('Nonfinite completed history')
                if losses[-10:] != s['tail_probe_losses'] or losses[-20:-10] != s['previous_probe_losses']:
                    raise ValueError('Tail/history mismatch')
                loss = statistics.fmean(losses[-10:])/s['initial_probe_loss']
                growth = statistics.fmean(losses[-10:])/max(statistics.fmean(losses[-20:-10]),1e-30)
                stable = loss <= 2 and growth <= 1.25
            if stable != s['bounded']:
                raise ValueError('Recomputed label differs: '+prefix)
            r = dict(job, K=2000, stable=stable, stop_reason=reason,
                     tail_loss_ratio=loss, tail_growth_ratio=growth, source=prefix)
            records.append(r)
            grouped[(job['algorithm'], job['gap'], job['lr'])].append(r)

    if len(records)!=250 or len(grouped)!=25:
        raise ValueError('Follow-up must have 250 completed trials at 25 LR cells')
    comparisons=[]
    for (alg,gap,lr),rows in grouped.items():
        seeds={r['seed'] for r in rows}
        if seeds!=set(range(23,33)):
            raise ValueError('Expected all seeds 23..32 for every cell')
        matches=[r for r in cells if r['axis']=='gap' and r['algorithm']==alg
                 and math.isclose(r['x'],gap,rel_tol=1e-10)
                 and math.isclose(r['lr'],lr,rel_tol=1e-8)]
        if len(matches)!=1:raise ValueError('Missing or ambiguous baseline cell')
        r=matches[0];oldseeds={int(v) for v in r['seed_ids'].split(',')}
        if seeds & oldseeds:raise ValueError('Seed overlap: refusing double counting')
        if len(oldseeds)!=r['n_total']:raise ValueError('Baseline seed-count mismatch')
        n=len(rows);stable=sum(t['stable'] for t in rows)
        c=dict(algorithm=alg,gap=gap,lr=r['lr'],old_n=r['n_total'],old_stable=r['n_stable'],
               old_fraction=r['fraction'],added_n=n,added_stable=stable,added_fraction=stable/n)
        r['n_total']+=n;r['n_assessed']+=n;r['n_stable']+=stable;r['n_unstable']+=n-stable
        r['n_revised']=int(r.get('n_revised') or 0)+n
        r['seed_ids']=','.join(map(str,sorted(oldseeds|seeds)))
        r['fraction']=r['n_stable']/r['n_assessed']
        for k in ('stable_fraction_assessed','fraction_lower_bound','fraction_upper_bound'):
            r[k]=r['fraction']
        r['source_kind']='baseline_plus_nesterov_followup'
        c.update(combined_n=r['n_total'],combined_stable=r['n_stable'],combined_fraction=r['fraction'])
        comparisons.append(c)
    report=dict(baseline_sha256=hashlib.sha256(payload).hexdigest(),new_trials=len(records),
                new_stable=sum(r['stable'] for r in records),new_unstable=sum(not r['stable'] for r in records),
                changed_cells=len(comparisons),policy='append disjoint seeds; retain all prior outcomes')
    return cells,excluded,records,report,comparisons


def pava_decreasing(successes, totals):
    """Binomial-weighted isotonic proportions; used ONLY for boundary extraction."""
    blocks = []
    for i, (s, n) in enumerate(zip(successes, totals)):
        blocks.append([i, i, float(s), float(n)])
        while len(blocks)>1 and blocks[-2][2]/blocks[-2][3] < blocks[-1][2]/blocks[-1][3]:
            right, left = blocks.pop(), blocks.pop()
            blocks.append([left[0], right[1], left[2]+right[2], left[3]+right[3]])
    values = np.empty(len(successes))
    for left, right, s, n in blocks:
        values[left:right+1] = s/n
    return values


def boundaries(cells, threshold):
    columns = defaultdict(list)
    for r in cells:
        columns[(r['axis'], r['algorithm'], r['x'])].append(r)
    points, audit = [], []
    for key, rows in sorted(columns.items()):
        rows.sort(key=lambda r:r['lr'])
        lrs = np.array([r['lr'] for r in rows])
        q = pava_decreasing([r['n_stable'] for r in rows], [r['n_assessed'] for r in rows])
        for row, value in zip(rows,q):
            audit.append(dict(axis=key[0], algorithm=key[1], x=key[2], lr=row['lr'],
                              n_seeds=row['n_assessed'], observed_fraction=row['fraction'],
                              isotonic_fraction=float(value)))
        point = dict(axis=key[0],algorithm=key[1],x=key[2],batch=rows[0]['batch'],
                     gap=rows[0]['gap'],K=rows[0]['K'],threshold=threshold,
                     lr_lower=None,lr_upper=None,boundary=None,status='',used_for_fit=False,
                     fitted_boundary=None,log_residual=None)
        equal = np.flatnonzero(np.isclose(q,threshold,rtol=0,atol=1e-12))
        if len(equal):
            lo,hi=float(lrs[equal[0]]),float(lrs[equal[-1]])
            point.update(lr_lower=lo,lr_upper=hi,boundary=math.sqrt(lo*hi),
                         status='exact_threshold' if lo==hi else 'threshold_plateau')
        elif q[0]<threshold:
            point.update(lr_upper=float(lrs[0]),status='left_censored')
        elif q[-1]>threshold:
            point.update(lr_lower=float(lrs[-1]),status='right_censored')
        else:
            j=int(np.flatnonzero(q<threshold)[0])
            t=(q[j-1]-threshold)/(q[j-1]-q[j])
            estimate=np.exp((1-t)*np.log(lrs[j-1])+t*np.log(lrs[j]))
            point.update(lr_lower=float(lrs[j-1]),lr_upper=float(lrs[j]),
                         boundary=float(estimate),status='bracketed')
        points.append(point)
    return points,audit


def profile_cap(u,y):
    """Exact least squares of y ~ a + min(u,d), profiling all change intervals."""
    order=np.argsort(u);u=u[order];y=y[order];n=len(y)
    candidates=[float(u[0]),float(u[-1])]
    for k in range(1,n):
        # Unconstrained growth intercept a and cap c; d=c-a.
        d=float(np.mean(y[k:])-np.mean(y[:k]-u[:k]))
        candidates.append(float(np.clip(d,u[k-1],u[k])))
    best=None
    for d in candidates:
        z=np.minimum(u,d);a=float(np.mean(y-z));pred=a+z
        item=(float(np.sum((y-pred)**2)),a,a+d,d)
        if best is None or item[0]<best[0]:best=item
    return best


def predict(fit,batch,gap,axis=None):
    b,g=np.broadcast_arrays(np.asarray(batch,dtype=float),np.asarray(gap,dtype=float))
    if fit['algorithm']=='sgd':
        if axis=='batch':return np.minimum(fit['batch_A']*b,fit['batch_C'])
        return fit['C']*np.minimum(1.,b)
    return np.minimum(fit['C'],fit['A']*b**fit['beta']*g)


def fit_models(points):
    fits=[]
    for algorithm in ALGORITHMS:
        # Do not count the shared B=4,gap=1/64 scan twice, or repeat SGD over rho.
        unique={}
        for p in points:
            if p['algorithm']!=algorithm or p['boundary'] is None:continue
            key=(p['batch'],p['gap'],p['K'])
            if key in unique:
                if not math.isclose(p['boundary'],unique[key]['boundary'],rel_tol=1e-9):
                    raise ValueError('Conflicting boundaries for the same physical configuration')
            else:
                unique[key]=p;p['used_for_fit']=True
        data=list(unique.values())
        if len(data)<3:raise ValueError('Not enough bracketed points for '+algorithm)
        b=np.array([p['batch'] for p in data]);g=np.array([p['gap'] for p in data])
        y=np.log([p['boundary'] for p in data])
        if algorithm=='sgd':
            C=float(np.exp(np.mean(y)));A=None;beta=0.;sse=float(np.sum((y-np.log(C))**2))
            cap_observed=True;beta_at_bound=False
        else:
            def objective(beta):return profile_cap(beta*np.log(b)+np.log(g),y)
            if algorithm=='polyak':beta=1.;best=objective(beta)
            else:
                # A positive empirical exponent; no forced beta>1 to fit a theorem.
                grid=np.linspace(.05,4.,791)
                coarse=min(((objective(float(v)),float(v)) for v in grid),key=lambda x:x[0][0])
                fine=np.linspace(max(.05,coarse[1]-.01),min(4.,coarse[1]+.01),401)
                best,beta=min(((objective(float(v)),float(v)) for v in fine),key=lambda x:x[0][0])
            sse,a,c,d=best;A=float(np.exp(a));C=float(np.exp(c))
            u=beta*np.log(b)+np.log(g)
            cap_observed=bool(np.any(u>d+1e-6))
            beta_at_bound=algorithm=='nesterov' and (beta<=.050001 or beta>=3.999999)
        fit=dict(algorithm=algorithm,A=A,C=C,beta=float(beta),n_unique_points=len(data),
                 log_rmse=math.sqrt(sse/len(data)),cap_observed=cap_observed,
                 beta_at_search_bound=beta_at_bound,
                 form='C*min(1,B)' if algorithm=='sgd' else 'min(C,A*B**beta*gap)')
        fit.update(batch_A=None,batch_C=None,batch_transition=None,batch_log_rmse=None)
        if algorithm=='sgd':
            # Requested top-left override: B=1,2 growth; B>=4 plateau.
            low=b<=2;high=b>=4
            if set(b[low])!={1,2} or not np.all(low|high) or not np.any(high):
                raise ValueError('SGD batch fit needs B=1,2 and at least one B>=4')
            a=float(np.mean(y[low]-np.log(b[low])))
            c=float(np.mean(y[high]))
            # Keep the crossover between the growth points and plateau points.
            d=float(np.clip(c-a,np.log(2),np.log(4)))
            a=float(np.mean(y-np.minimum(np.log(b),d)))
            fit.update(batch_A=float(np.exp(a)),batch_C=float(np.exp(a+d)),
                       batch_transition=float(np.exp(d)),
                       batch_log_rmse=float(np.sqrt(np.mean((y-a-np.minimum(np.log(b),d))**2))))
            fit['form']='batch: min(batch_A*B,batch_C); gap: C (unchanged)'
        for p in points:
            if p['algorithm']!=algorithm:continue
            p['fitted_boundary']=float(predict(fit,p['batch'],p['gap'],p['axis']))
            if p['boundary'] is not None:p['log_residual']=math.log(p['boundary']/p['fitted_boundary'])
        fits.append(fit)
    return fits


def log_edges(values):
    values=np.log(values)
    if len(values)==1:return np.exp([values[0]-.2,values[0]+.2])
    return np.exp(np.r_[values[0]-(values[1]-values[0])/2,
                        (values[:-1]+values[1:])/2,values[-1]+(values[-1]-values[-2])/2])


def save_figure(fig,output,stem,dpi):
    # Preserve the requested canvas sizes (no bbox_inches='tight' cropping).
    fig.savefig(output/(stem+'.png'),dpi=dpi,facecolor='white')
    fig.savefig(output/(stem+'.pdf'),facecolor='white')
    plt.close(fig)


def panel(ax,axis,algorithm,cells,points,fit,coverage=False):
    rows=[r for r in cells if r['axis']==axis and r['algorithm']==algorithm]
    xs=sorted({r['x'] for r in rows});xe=log_edges(xs)
    maximum=max(r['n_assessed'] for r in cells)
    cmap='viridis' if coverage else STABILITY_CMAP;norm=Normalize(0,maximum if coverage else 1)
    for i,x in enumerate(xs):
        col=sorted([r for r in rows if r['x']==x],key=lambda r:r['lr'])
        y=log_edges([r['lr'] for r in col])
        values=np.array([r['n_assessed'] if coverage else r['fraction'] for r in col])
        ax.pcolormesh(xe[i:i+2],y,values[:,None],shading='flat',cmap=cmap,norm=norm,rasterized=True)
    ax.set_xscale('log',base=2);ax.set_yscale('log',base=2)
    ax.set_xlim(xe[0],xe[-1])
    ax.set_ylim(0.0625, 2**3.25)
    if not coverage:
        xx=np.geomspace(xs[0],xs[-1],500)
        yy=predict(fit,xx,1/64,axis='batch') if axis=='batch' else predict(fit,4,xx,axis='gap')
        ax.plot(xx,yy,'--',color='white',lw=2,
                path_effects=[pe.Stroke(linewidth=3.4,foreground='black'),pe.Normal()],zorder=5)
        obs=[p for p in points if p['axis']==axis and p['algorithm']==algorithm and p['boundary'] is not None]
        ax.scatter([p['x'] for p in obs],[p['boundary'] for p in obs],marker=MARKERS[algorithm],
                   s=30,facecolors='white',edgecolors='black',linewidths=.7,zorder=6)
    ax.set_xticks(xs,[str(int(x)) if axis=='batch' else gap_label(x) for x in xs])
    if axis=='gap':
        plt.setp(ax.get_xticklabels(),rotation=45,ha='right',rotation_mode='anchor')
    ax.xaxis.set_minor_locator(NullLocator());ax.yaxis.set_minor_locator(NullLocator())
    yticks = [0.0625, 0.25, 1, 4, 8]
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}' if axis=='gap' else f'{v:.3g}'))
    ax.set_xlabel(r'Batch size $B$' if axis=='batch' else r'Momentum damping $1-\beta$')
    ax.set_ylabel(r'Learning rate $\eta$')
    fixed=('no momentum' if axis=='batch' else 'fixed B = 4; no momentum') if algorithm=='sgd' else (r'fixed $1-\beta$ = 1/64' if axis=='batch' else 'fixed B = 4')
    ax.set_title(NAMES[algorithm]+'\n'+fixed)
    ax.set_box_aspect(4/5);ax.grid(False,which='both')
    return ScalarMappable(norm=norm,cmap=cmap)


def plot_all(cells,points,fits,output,dpi,individual):
    mapping={f['algorithm']:f for f in fits}
    for coverage in (False,True):
        fig,axes=plt.subplots(2,3,figsize=(16.5,9.0),layout='constrained')
        for i,axis in enumerate(('batch','gap')):
            for j,algorithm in enumerate(ALGORITHMS):
                mappable=panel(axes[i,j],axis,algorithm,cells,points,mapping[algorithm],coverage)
        cb=fig.colorbar(mappable,ax=axes.ravel().tolist(),fraction=.025,pad=.02,shrink=.86)
        cb.set_label('Number of seeds' if coverage else 'Stable-run fraction')
        if not coverage:cb.set_ticks([0,.2,.4,.6,.8,1.])
        if coverage:cb.set_ticks(sorted(set([1,5,10,max(r['n_assessed'] for r in cells)])))
        # Intentionally no global legend, footer, subtitle, or purple K=4000 annotation.
        save_figure(fig,output,'seed_coverage' if coverage else 'cifar_stability_merged',dpi)
    if individual:
        for axis in ('batch','gap'):
            for algorithm in ALGORITHMS:
                fig,ax=plt.subplots(figsize=(5,4),layout='constrained')
                panel(ax,axis,algorithm,cells,points,mapping[algorithm])
                save_figure(fig,output,f'{algorithm}_{axis}',dpi)
    # Companion boundary-only view, following the supplied two-panel reference.
    fig,axes=plt.subplots(1,2,figsize=(10.8,5.2),layout='constrained')
    for ax,axis in zip(axes,('batch','gap')):
        for algorithm in ALGORITHMS:
            obs=[p for p in points if p['axis']==axis and p['algorithm']==algorithm and p['boundary'] is not None]
            xx=np.geomspace(min(p['x'] for p in obs),max(p['x'] for p in obs),400)
            yy=predict(mapping[algorithm],xx,1/64,axis='batch') if axis=='batch' else predict(mapping[algorithm],4,xx,axis='gap')
            ax.plot(xx,yy,'--',color=COLORS[algorithm])
            ax.plot([p['x'] for p in obs],[p['boundary'] for p in obs],ls='',
                    marker=MARKERS[algorithm],color=COLORS[algorithm])
        ax.set_xscale('log',base=2);ax.set_yscale('log',base=2)
        ax.set_ylim(bottom=BOUNDARY_YMIN)
        ax.yaxis.set_major_locator(LogLocator(base=2,subs=(1.,),numticks=9))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'))
        ticks=sorted({p['x'] for p in points if p['axis']==axis})
        ax.set_xticks(ticks,[str(int(x)) if axis=='batch' else gap_label(x) for x in ticks])
        if axis=='gap':
            plt.setp(ax.get_xticklabels(),rotation=45,ha='right',rotation_mode='anchor')
        ax.set_xlabel(r'Batch size $B$' if axis=='batch' else r'Momentum damping $1-\beta$')
        ax.set_ylabel(r'Stability boundary $\eta$');ax.set_title('Critical learning rates')
        ax.xaxis.set_minor_locator(NullLocator());ax.yaxis.set_minor_locator(NullLocator())
        ax.set_box_aspect(4/5);ax.grid(False,which='both')
    save_figure(fig,output,'critical_learning_rates',dpi)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--summary-dir','--data-dir','--input',dest='summary_dir',type=Path,
                        default=None, help='Optional verification against the archived baseline CSV')
    inputs=parser.add_mutually_exclusive_group()
    inputs.add_argument('--gap9-dir',type=Path,help='Completed gap9 output directory')
    inputs.add_argument('--gap9-zip',type=Path,help='Export ZIP; no extraction needed')
    inputs.add_argument('--followup-dir',type=Path,help='Completed 250-trial Nesterov follow-up directory')
    inputs.add_argument('--followup-zip',type=Path,help='Exported Nesterov follow-up ZIP')
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--stable-fraction',type=float,default=.5,
                        help='Stable-fraction level defining the boundary (default: 0.5)')
    parser.add_argument('--dpi',type=int,default=300)
    parser.add_argument('--individual',action='store_true',help='Also export each panel on a 5x4 inch canvas')
    args=parser.parse_args()
    if not 0<args.stable_fraction<1:parser.error('--stable-fraction must lie strictly between 0 and 1')
    if args.dpi<=0:parser.error('--dpi must be positive')
    followup=args.followup_dir or args.followup_zip
    source=(followup or args.gap9_zip or args.gap9_dir or Path('output/gap9_midpoints_20260921')).expanduser().resolve()
    out=(args.output or Path('output/summary_nesterov_followup' if followup else 'output/summary_gap9')).expanduser().resolve()
    if out==source or (args.summary_dir and out==resolve_csv(args.summary_dir).parent):
        raise ValueError('Use a separate output directory')
    bundle=GapBundle(source,'followup_plan.json' if followup else 'gap9_views.json')
    comparisons=[]
    try:
        if followup:
            cells,excluded,records,merge_report,comparisons=load_followup(bundle)
        else:
            cells,excluded,records,merge_report=load_gap9(bundle,args.summary_dir)
    finally:
        bundle.close()
    points,audit=boundaries(cells,args.stable_fraction);fits=fit_models(points)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'merged_cells.csv',cells)
    write_csv(out/('followup_trials_audit.csv' if followup else 'gap9_trials_audit.csv'),records)
    if followup:write_csv(out/'followup_before_after.csv',comparisons)
    write_csv(out/'plotted_cells.csv',cells);write_csv(out/'excluded_gap128_cells.csv',excluded)
    write_csv(out/'empirical_boundaries.csv',points);write_csv(out/'boundary_extraction.csv',audit)
    write_csv(out/'fit_parameters.csv',fits)
    report=dict(source=str(source),merge=merge_report,
                excluded_gap=1/128,excluded_cells=len(excluded),retained_cells=len(cells),
                stable_fraction=args.stable_fraction,retained_horizon=2000,fits=fits,
                fit_definition='Equal-weight log-boundary least squares; Polyak/Nesterov shared across scans; SGD batch growth at B=1,2 then plateau, SGD gap unchanged',
                boundary_definition='Seed-count-weighted decreasing isotonic proportions; log-LR crossing',
                source_heatmap_changed=False,raw_data_deleted=False,
                display=dict(gap_ymin=GAP_YMIN,boundary_ymin=BOUNDARY_YMIN,
                             colormap='stability_yellow',normalization='linear 0..1',
                             fitted_line_dependent_recoloring=False),
                note='Isotonic adjustment is only for extracting the empirical boundary, not heatmap colors',
                beta_search_range=[.05,4.],code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'fit_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    plot_all(cells,points,fits,out,args.dpi,args.individual)
    documentation = """# CIFAR-10: nine-point fixed-B summary

运行（服务器已有结果目录）：
python3 plot_cifar_all.py --gap9-dir output/gap9_midpoints_20260921 --output output/summary_gap9 --individual

或者直接读取导出的 ZIP：
python3 plot_cifar_all.py --gap9-zip cifar_gap9_results_20260921_140050_075285Z.zip --output output/summary_gap9 --individual

依赖：numpy、matplotlib；不需要 PyTorch，不运行训练。
自动读取包内 _baseline/merged_cells.csv 与新增 472 个 trial。
从 history.jsonl 重新检查末尾两个 10 次评估窗口与 bounded 标签；不完整数据报错。
旧参数停止仍按旧汇总计为不稳定。新增 record 模式不因参数范数大自动判失败。
排除 gap=1/128；所有保留结果 K=2000；原始数据不删除。
SGD 下排九列复用同一组 B=4 实验，不计为独立重复。
新列 0.25<=LR<=4 为 5 seeds，其余为 seed0；各格 seed 数见 seed_coverage。
颜色是已测 seed 的原始稳定比例；无平滑、无根据拟合补值。
热图矩形范围由相邻扫描坐标的几何中点划分，块内颜色不代表额外测量。
边界标记：seed 数加权的递减等序回归后，在 log(LR) 上提取 50% crossing。
等序回归仅用于提取边界，热图保留非单调实测比例；删失列不参加拟合。
拟合：所有可确定边界的物理配置等权，重复配置只计一次。
SGD 上排 min(A_s*B,C_s)，B=1,2 增长，B>=4 平台；下排延续旧常数线。
Polyak 两种扫描联合拟合 min(C_p,A_p*B*gap)。
Nesterov 两种扫描联合拟合 min(C_n,A_n*B**beta*gap)，beta 数据估计，不强迫大于1。
拟合描述经验形状，不强制实验符合理论；参数与逐点残差全部导出。
使用 --stable-fraction 0.8 可以画另一阈值的边界，需另设输出目录。
删除底部图例/说明，保留色条；单图 5x4 英寸，使用用户提供的字号配置。
输出主图 cifar_stability_merged.png/pdf，双面板 critical_learning_rates.png/pdf，
seed_coverage.png/pdf，合并表 merged_cells.csv，逐 trial 审计及边界/拟合参数表。
"""
    if followup:
        documentation = """# Nesterov targeted follow-up

python3 plot_cifar_all.py --followup-dir output/nesterov_followup_20260921 --output output/summary_nesterov_followup --individual

自动读取本轮保存的旧九点汇总快照与全部 250 个新 trial。
新增 seed=23..32；旧结果全部保留，stable fraction=(旧稳定数+新稳定数)/(旧总数+新总数)。
先合并计数再提取 50% 边界并重新拟合；不会将新增 seed 组和旧 seed 组的比例简单平均。
followup_before_after.csv 对比旧 seed、新 seed、合并后的稳定比例。
其余配色、字号、六面板、九点 gap、SGD 复用、拟合规则与 gap9 版本一致。
不会根据预期结果删除失败试验；热图颜色为实测比例，不按拟合线平滑。
未完成全部 250 个 trial 时拒绝生成最终汇总；用运行脚本 --status 检查进度。
""" + documentation
    (out/'README_plot.md').write_text(documentation,encoding='utf-8')
    print(json.dumps(report,indent=2))
    print('Saved:',out)

if __name__=='__main__':
    main()
