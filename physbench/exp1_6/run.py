"""Run EXP1.6 after a successful local EXP1.5 regression. See PROTOCOL.md."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback

from adapters import Adapter, ROOT, BASE, digest
from numerics import classify, roots_on_grid, dominant_pair


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def table(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class Experiment:
    def __init__(self, model, source, out, state_filter=None, forcing_filter=None):
        self.out = out.resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        self.model = model
        self.state_filter = state_filter
        self.forcing_filter = forcing_filter
        self.config = json.loads(Path(__file__).with_name('config.json').read_text())
        self.cache = {}
        self.reversal_cache = {}
        self.grid, self.reversals, self.transitions = [], [], []
        self.coverage, self.intersections, self.distances = [], [], []
        self.regressions = []
        self.evidence = gzip.open(self.out / 'evaluations.jsonl.gz', 'wt', encoding='utf-8')
        self.iterations = gzip.open(self.out / 'search_history.jsonl.gz', 'wt', encoding='utf-8')
        self.adapter = Adapter(model, source)
        self.pairs = list(itertools.combinations(self.adapter.terms, 2))
        self.started = datetime.now(timezone.utc).isoformat()
        protocol_files = ['config.json', 'PROTOCOL.md', 'frozen_inputs.json', 'adapters.py', 'numerics.py', 'run.py']
        for name in protocol_files:
            shutil.copy2(Path(__file__).with_name(name), self.out / name)
        self.manifest = {
            'model': model, 'base_commit': BASE, 'model_commit': self.adapter.source_commit,
            'started_utc': self.started, 'python': sys.version, 'platform': platform.platform(),
            'argv': sys.argv, 'config': self.config, 'frozen_inputs': self.adapter.input_records,
            'model_files': self.adapter.model_files,
            'protocol_source_sha256': {n: digest(self.out / n) for n in protocol_files},
            'external_historical_audit': 'PENDING_USER_UPLOAD; local rerun is a distinct evidence source',
        }
        dump(self.out / 'manifest_start.json', self.manifest)
        packages = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True) if __import__('importlib').util.find_spec('pip') else '\n'.join(sorted(f'{d.metadata["Name"]}=={d.version}' for d in __import__('importlib.metadata', fromlist=['distributions']).distributions()))
        (self.out / 'environment.txt').write_text(packages, encoding='utf-8')

    def evaluate(self, h, state, forcing, c):
        key = (int(h), state, int(forcing), round(float(c), 10))
        if key in self.cache:
            return self.cache[key]
        if len(self.cache) >= self.config['max_pair_evaluations_per_model']:
            raise RuntimeError('PREDECLARED_EVALUATION_BUDGET_EXHAUSTED')
        shard = f'{self.state_filter or "ALL"}_F{self.forcing_filter if self.forcing_filter is not None else "ALL"}'
        eid = f'{self.model}_{shard}_{len(self.cache):06d}'
        pair, trace = self.adapter.evaluate(*key, eid)
        v = pair['high_minus_low']['flux_contributions_ppm']
        r = pair['high_minus_low']['final_co2_ppm']
        eps = r - sum(v.values())
        row = {'eval_id': eid, 'model': self.model, 'horizon_s': int(h), 'state_id': state,
               'forcing_row': int(forcing), 'co2_ppm': float(c), 'response_ppm': r,
               'epsilon_ppm': eps, **classify(v, eps, self.model), **v}
        self.evidence.write(json.dumps({'evaluation': row, 'pair': pair, 'trace': trace}, allow_nan=False) + '\n')
        self.evidence.flush()
        self.cache[key] = row
        return row

    def record_search(self, kind, h, state, fr, root, pair=None):
        self.iterations.write(json.dumps({'kind': kind, 'h': h, 'state': state, 'forcing': fr,
                                          'pair': pair, **root}, allow_nan=False) + '\n')
        self.iterations.flush()

    def roots(self, h, state, fr, grid):
        key = (h, state, fr, tuple(grid))
        if key not in self.reversal_cache:
            fn = lambda c: self.evaluate(h, state, fr, c)['response_ppm']
            roots = roots_on_grid(fn, grid)
            for r in roots:
                self.record_search('REVERSAL', h, state, fr, r)
            self.reversal_cache[key] = roots
        return self.reversal_cache[key]

    def regression(self):
        path = ROOT / f'EXP1_5_{self.model}_BOUNDARY_MECHANISM_STABILITY.json'
        old = json.loads(path.read_text())
        if not old['run_pass']:
            raise RuntimeError('Legacy EXP1.5 did not pass')
        for p in ROOT.glob(f'EXP1_5_{self.model}_*'):
            if p.is_file():
                shutil.copy2(p, self.out / p.name)
        terms = list(csv.DictReader((ROOT / f'EXP1_5_{self.model}_TERM_CONTRIBUTIONS.csv').open(encoding='utf-8')))
        oldroots = list(csv.DictReader((ROOT / f'EXP1_5_{self.model}_ROOT_MECHANISMS.csv').open(encoding='utf-8')))
        for r in oldroots:
            e = self.evaluate(int(r['horizon_s']), r['state_id'], int(r['forcing_row']), float(r['initial_co2_ppm']))
            expected = {t['term']: float(t['contribution_ppm']) for t in terms
                        if t['condition_id'] == r['condition_id'] and t['point_type'] == r['point_type']
                        and abs(float(t['initial_co2_ppm']) - float(r['initial_co2_ppm'])) < 1e-7
                        and t['term'] in self.adapter.terms}
            if set(expected) != set(self.adapter.terms):
                raise RuntimeError('Incomplete legacy vector')
            error = max(abs(e[k] - v) for k, v in expected.items())
            response_error = abs(e['response_ppm'] - float(r['final_high_minus_low_ppm']))
            passed = error <= 1e-7 and response_error <= 1e-7 and e['dominant'] == r['dominant_term']
            self.regressions.append({'condition_id': r['condition_id'], 'eval_id': e['eval_id'],
                                     'max_term_error_ppm': error, 'response_error_ppm': response_error, 'pass': passed})
            if not passed:
                raise RuntimeError('Prefix-loader root/vector regression failed')
        dump(self.out / 'regression.json', {'legacy_run_pass': True, 'surface': old['surface_regression'],
                                           'prefix_loader': self.regressions})

    def scan(self):
        fine = [250. + 12.5 * k for k in range(29)]
        coarse = fine[::2]
        for state in self.adapter.states:
            if self.state_filter and state != self.state_filter:
                continue
            for fr in [0, 6, 10]:
                if self.forcing_filter is not None and fr != self.forcing_filter:
                    continue
                for h in range(300, 1801, 60):
                    evals = [self.evaluate(h, state, fr, c) for c in fine]
                    self.grid.extend(evals)
                    rr = self.roots(h, state, fr, fine)
                    coarse_roots = self.roots(h, state, fr, coarse)
                    self.coverage.append({'state': state, 'forcing': fr, 'h': h, 'kind': 'REVERSAL_COARSE_FINE',
                                          'coarse_count': len(coarse_roots), 'fine_count': len(rr),
                                          'status': 'MATCH' if len(rr) == len(coarse_roots) else 'COVERAGE_WARNING'})
                    for r in rr:
                        self.reversals.append({**self.evaluate(h, state, fr, r['x']),
                                               'root_lo': r['lo'], 'root_hi': r['hi'], 'root_status': r['status']})
                    found = []
                    for i, j in self.pairs:
                        fn = lambda c, i=i, j=j: abs(self.evaluate(h, state, fr, c)[i]) - abs(self.evaluate(h, state, fr, c)[j])
                        vals = [abs(e[i]) - abs(e[j]) for e in evals]
                        if all(abs(e[i]) + abs(e[j]) < 1e-12 for e in evals):
                            continue
                        candidates = []
                        for k, e in enumerate(evals):
                            if abs(vals[k]) <= 1e-7 and dominant_pair({t: e[t] for t in self.adapter.terms}, i, j):
                                candidates.extend(roots_on_grid(fn, [fine[k]]))
                        for k in range(len(fine) - 1):
                            if vals[k] * vals[k+1] >= 0:
                                continue
                            endpoints = [evals[k], evals[k+1]]
                            if not any(e['dominant'] in (i, j) for e in endpoints):
                                mid = self.evaluate(h, state, fr, (fine[k]+fine[k+1])/2)
                                if mid['dominant'] not in (i, j):
                                    continue
                            candidates.extend(roots_on_grid(fn, fine[k:k+2]))
                        unique = []
                        for r in sorted(candidates, key=lambda r: r['x']):
                            if not unique or r['x'] - unique[-1]['x'] > 1e-3:
                                unique.append(r)
                        for r in unique:
                            e = self.evaluate(h, state, fr, r['x'])
                            self.record_search('PAIR_EQUALITY', h, state, fr, r, (i, j))
                            if not dominant_pair({t:e[t] for t in self.adapter.terms}, i, j):
                                continue
                            left = self.evaluate(h, state, fr, max(250, r['x']-.1))
                            right = self.evaluate(h, state, fr, min(600, r['x']+.1))
                            row = {**e, 'term_i': i, 'term_j': j, 'gap_ppm': fn(r['x']),
                                   'root_lo': r['lo'], 'root_hi': r['hi'], 'root_status': r['status'],
                                   'left_eval_id': left['eval_id'], 'right_eval_id': right['eval_id'],
                                   'left_phase': left['phase'], 'right_phase': right['phase'],
                                   'resolved_phase_change': left['resolved_under_screen'] and right['resolved_under_screen'] and left['dominant'] != right['dominant'],
                                   'opposite_sign_pair': e[i]*e[j] < 0,
                                   'remaining_plus_epsilon_ppm': sum(e[t] for t in self.adapter.terms if t not in (i, j)) + e['epsilon_ppm']}
                            found.append(row)
                            self.transitions.append(row)
                        def count(indices):
                            return sum(vals[a]*vals[b] < 0 and any(evals[k]['dominant'] in (i,j) for k in (a,b))
                                       for a,b in zip(indices[:-1],indices[1:]))
                        nc, nf = count(list(range(0,29,2))), count(list(range(29)))
                        if nc or nf:
                            self.coverage.append({'state':state,'forcing':fr,'h':h,'kind':f'PAIR_COARSE_FINE:{i}|{j}',
                                                  'coarse_count':nc,'fine_count':nf,'status':'MATCH' if nc==nf else 'COVERAGE_WARNING'})
                        for k in range(1, 28):
                            if abs(vals[k]) < min(abs(vals[k-1]),abs(vals[k+1])) and vals[k-1]*vals[k+1] > 0 and evals[k]['dominant'] in (i,j):
                                self.coverage.append({'state':state,'forcing':fr,'h':h,'kind':f'GAP_MINIMUM:{i}|{j}',
                                                      'co2':fine[k],'value':vals[k],'status':'UNRESOLVED_FEATURE_CANDIDATE'})
                    for k in range(1,28):
                        ys = [evals[t]['response_ppm'] for t in (k-1,k,k+1)]
                        if abs(ys[1]) < min(abs(ys[0]),abs(ys[2])) and ys[0]*ys[2] > 0:
                            self.coverage.append({'state':state,'forcing':fr,'h':h,'kind':'RESPONSE_MINIMUM','co2':fine[k],
                                                  'value':ys[1],'status':'UNRESOLVED_FEATURE_CANDIDATE'})
                    for t in found:
                        for r in rr:
                            self.distances.append({'state':state,'forcing':fr,'h':h,'pair':t['term_i']+'|'+t['term_j'],
                                                   'transition_co2':t['co2_ppm'],'reversal_co2':r['x'],
                                                   'signed_distance_ppm':t['co2_ppm']-r['x'], 'transition_status':t['root_status']})
                    print(f'{self.model} {state} F{fr} H{h}: R={len(rr)} T={len(found)} evals={len(self.cache)}', flush=True)
                    self.save_tables()
                self.find_intersections(state, fr, fine)

    def find_intersections(self, state, fr, fine):
        def branch(h):
            rr = self.roots(h, state, fr, fine)
            if len(rr) != 1:
                return None
            return self.evaluate(h, state, fr, rr[0]['x'])
        for h0 in range(300,1800,60):
            a,b = branch(h0), branch(h0+60)
            if a is None or b is None:
                self.coverage.append({'state':state,'forcing':fr,'h':h0,'kind':'BRANCH_ASSOCIATION', 'status':'ABSENT_OR_AMBIGUOUS'})
                continue
            for i,j in self.pairs:
                ga,gb = abs(a[i])-abs(a[j]),abs(b[i])-abs(b[j])
                if ga*gb > 0 or (abs(ga)+abs(gb) < 1e-12):
                    continue
                if not any(e['dominant'] in (i,j) for e in [a,b]):
                    continue
                lo,hi = h0,h0+60
                aa,bb = a,b
                history = []
                status = 'HORIZON_BRACKET_ONLY'
                if self.model == 'M1':
                    while hi-lo > 1:
                        mid = (lo+hi)//2
                        e = branch(mid)
                        if e is None:
                            status = 'BRANCH_LOST_UNRESOLVED'
                            break
                        g = abs(e[i])-abs(e[j])
                        history.append({'h':mid,'eval_id':e['eval_id'],'gap':g,'R':e['response_ppm']})
                        if ga*g <= 0:
                            hi,bb,gb = mid,e,g
                        else:
                            lo,aa,ga = mid,e,g
                self.intersections.append({'state':state,'forcing':fr,'pair':i+'|'+j,
                    'h_lo':lo,'h_hi':hi,'c_at_h_lo':aa['co2_ppm'],'c_at_h_hi':bb['co2_ppm'],
                    'gap_lo':ga,'gap_hi':gb,'epsilon_lo':aa['epsilon_ppm'],'epsilon_hi':bb['epsilon_ppm'],
                    'eval_lo':aa['eval_id'],'eval_hi':bb['eval_id'], 'status':status,
                    'certified_intersection':False})
                self.iterations.write(json.dumps({'kind':'INTERSECTION_BRACKET','state':state,'fr':fr,'pair':[i,j],'history':history})+'\n')
                self.iterations.flush()
        self.save_tables()

    def save_tables(self):
        for name, rows in [('phase_grid',self.grid),('reversal_roots',self.reversals),('transition_roots',self.transitions),
                           ('coverage',self.coverage),('intersection_brackets',self.intersections),('signed_separation',self.distances)]:
            table(self.out / f'{name}.csv',rows)

    def finish(self, status, error=None):
        self.save_tables()
        self.evidence.close()
        self.iterations.close()
        self.adapter.close()
        result = {'status':status,'model':self.model,'paired_evaluations':len(self.cache),
                  'phase_grid_points':len(self.grid),'reversal_points':len(self.reversals),
                  'transition_points':len(self.transitions),'intersection_brackets':len(self.intersections),
                  'unresolved_grid_points':sum(not x['resolved_under_screen'] for x in self.grid),
                  'coverage_warnings':sum(r['status']=='COVERAGE_WARNING' for r in self.coverage),
                  'unresolved_feature_candidates':sum(r['status']=='UNRESOLVED_FEATURE_CANDIDATE' for r in self.coverage),
                  'transition_width_only':sum(r['root_status']=='WIDTH_ONLY' for r in self.transitions),
                  'error':error,'completed_utc':datetime.now(timezone.utc).isoformat(),
                  'interpretation':'Sampled transition set. No certified continuous manifold or globally exhaustive topology.'}
        dump(self.out / 'result.json',result)
        dump(self.out / 'output_hashes.json',{p.name:digest(p) for p in sorted(self.out.iterdir()) if p.is_file() and p.name!='output_hashes.json'})
        print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['M1','M2'],required=True)
    parser.add_argument('--model-source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--state',choices=['S_COOL_HUMID','S_NOMINAL','S_WARM_DRY'])
    parser.add_argument('--forcing',type=int,choices=[0,6,10])
    args=parser.parse_args()
    exp=Experiment(args.model,args.model_source,args.out,args.state,args.forcing)
    try:
        exp.regression()
        exp.scan()
    except BaseException:
        error=traceback.format_exc()
        exp.finish('FAILED',error)
        raise
    exp.finish('PASS_EXECUTION')


if __name__=='__main__':
    main()
