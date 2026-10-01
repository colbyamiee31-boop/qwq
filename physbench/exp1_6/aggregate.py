"""Aggregate all nine completed M2 strata while retaining original evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('shards',type=Path)
    p.add_argument('out',type=Path)
    a=p.parse_args()
    shards=[a.shards/f'{s}_F{f}' for s in ['S_COOL_HUMID','S_NOMINAL','S_WARM_DRY'] for f in [0,6,10]]
    results=[json.loads((s/'result.json').read_text()) for s in shards]
    if any(r['status']!='PASS_EXECUTION' for r in results):raise RuntimeError('Incomplete or failed shards')
    configs=[(s/'config.json').read_bytes() for s in shards]
    if any(c!=configs[0] for c in configs):raise RuntimeError('Mixed configurations')
    for name in ['adapters.py','numerics.py','run.py','frozen_inputs.json','PROTOCOL.md']:
        if len({sha(s/name) for s in shards})!=1:raise RuntimeError(f'Mixed sources: {name}')
    a.out.mkdir(parents=True,exist_ok=False)
    tables=['phase_grid','reversal_roots','transition_roots','coverage','intersection_brackets','signed_separation']
    if all((s/'transition_polish.csv').exists() for s in shards):tables.append('transition_polish')
    for name in tables:
        rows=[]
        for s in shards:
            with (s/f'{name}.csv').open(encoding='utf-8') as f:rows.extend(csv.DictReader(f))
        keys=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
        with (a.out/f'{name}.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    total={'model':'M2','status':'PASS_EXECUTION','shards':9,
           'interpretation':'Aggregated discrete-horizon phase map; intersection brackets are not certified continuous intersections.'}
    for k in ['paired_evaluations','phase_grid_points','reversal_points','transition_points','intersection_brackets',
              'unresolved_grid_points','coverage_warnings','unresolved_feature_candidates','transition_width_only']:
        total[k]=sum(r[k] for r in results)
    if 'transition_polish' in tables:
        audits=[json.loads((s/'identifiability_summary.json').read_text()) for s in shards]
        summary={k:sum(r[k] for r in audits) for k in ['candidates','residual_resolved','unresolved_equality']}
        (a.out/'identifiability_summary.json').write_text(json.dumps(summary,indent=2))
    (a.out/'result.json').write_text(json.dumps(total,indent=2))
    refs=[{'directory':str(Path('..')/a.shards.name/s.name),'output_hashes_sha256':sha(s/'output_hashes.json')} for s in shards]
    (a.out/'shard_references.json').write_text(json.dumps(refs,indent=2))
    for name in ['config.json','PROTOCOL.md','frozen_inputs.json','environment.txt']:
        shutil.copy2(shards[0]/name,a.out/name)
    (a.out/'output_hashes.json').write_text(json.dumps({p.name:sha(p) for p in a.out.iterdir() if p.is_file()},indent=2))
    print(json.dumps(total,indent=2))


if __name__=='__main__':main()
