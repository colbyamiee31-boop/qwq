"""Fail-closed evidence validation; does not infer scientific root completeness."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readrows(p):
    with p.open(encoding='utf-8') as f:return list(csv.DictReader(f))


def verify(root,full=True):
    result=json.loads((root/'result.json').read_text())
    assert result['status']=='PASS_EXECUTION',result
    hashes=json.loads((root/'output_hashes.json').read_text())
    for name,expected in hashes.items():assert sha(root/name)==expected,name
    ids=set()
    if (root/'shard_references.json').exists():
        for ref in json.loads((root/'shard_references.json').read_text()):
            shard=(root/ref['directory']).resolve()
            assert sha(shard/'output_hashes.json')==ref['output_hashes_sha256']
            report=verify(shard,full=False)
            assert not (ids&report['ids']),'Duplicate shard evaluation ID'
            ids.update(report['ids'])
    else:
        regression=json.loads((root/'regression.json').read_text())
        assert regression['legacy_run_pass'] and all(r['pass'] for r in regression['prefix_loader'])
        max_closure=0
        with gzip.open(root/'evaluations.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:
                x=json.loads(line);e=x['evaluation'];pair=x['pair']
                assert e['eval_id'] not in ids
                ids.add(e['eval_id'])
                vector=pair['high_minus_low']['flux_contributions_ppm']
                r=pair['high_minus_low']['final_co2_ppm']
                assert all(math.isfinite(v) for v in [r,*vector.values()])
                assert abs(e['epsilon_ppm']-(r-sum(vector.values())))<1e-10
                assert 250<=e['co2_ppm']<=600 and 300<=e['horizon_s']<=1800
                for k,v in vector.items():assert e[k]==v
                max_closure=max(max_closure,abs(e['epsilon_ppm']))
                if result['model']=='M1':
                    assert not x['trace']
                    assert 'native_final_states' in pair
                    for arm in ['LOW','HIGH']:
                        assert pair[arm]['augmented_vs_native_final_state_max_abs_diff']<=1e-7
                        assert abs(pair[arm]['named_vs_total_ode_closure_mg_m3'])<=1e-10
                else:
                    assert len(x['trace'])==2*e['horizon_s']//30
                    assert e['horizon_s']%60==0
                    assert abs(e['epsilon_ppm'])<=1e-8
        assert len(ids)==result['paired_evaluations']
        with gzip.open(root/'search_history.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line)
                if 'status' in r:assert r['status']!='ITERATION_LIMIT'
    grid=readrows(root/'phase_grid.csv')
    seen={(int(r['horizon_s']),r['state_id'],int(r['forcing_row']),float(r['co2_ppm'])) for r in grid}
    assert len(seen)==len(grid),'Duplicate grid points'
    if full:
        expected={(h,s,f,250+12.5*k) for h in range(300,1801,60)
                  for s in ['S_COOL_HUMID','S_NOMINAL','S_WARM_DRY'] for f in [0,6,10] for k in range(29)}
        assert seen==expected,'Missing/extra design points'
    else:
        assert len(seen)==26*29
    for name in ['phase_grid','reversal_roots','transition_roots']:
        for r in readrows(root/f'{name}.csv'):assert r['eval_id'] in ids
    return {'ids':ids,'grid_points':len(grid),'model':result['model'],'verified':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
    report=verify(a.directory)
    report['evaluations']=len(report.pop('ids'))
    (a.directory/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
