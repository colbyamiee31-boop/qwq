"""Prespecified coverage follow-up: seed pair searches with contribution zeros.

Reuse completed grid evidence. All new model calls remain in the audit stream.
The original run remains immutable and is an explicit parent.
"""
import argparse
import csv
import gzip
import itertools
import json
from pathlib import Path
import shutil
import traceback
from adapters import digest
from numerics import roots_on_grid, dominant_pair
from run import Experiment, dump


def readrows(p):
    with p.open(encoding='utf-8') as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--model-source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    parent=a.parent.resolve()
    old=json.loads((parent/'result.json').read_text())
    if old['status']!='PASS_EXECUTION':raise RuntimeError('Parent did not complete')
    priorgrid=readrows(parent/'phase_grid.csv')
    states=sorted({r['state_id'] for r in priorgrid})
    forcing=sorted({int(r['forcing_row']) for r in priorgrid})
    e=Experiment(old['model'],a.model_source,a.out,
                 states[0] if len(states)==1 else None,forcing[0] if len(forcing)==1 else None)
    for name in ['refine.py','COVERAGE_AMENDMENT.md']:
        shutil.copy2(Path(__file__).with_name(name),e.out/name)
    with gzip.open(parent/'evaluations.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            x=json.loads(line);r=x['evaluation']
            e.cache[(r['horizon_s'],r['state_id'],r['forcing_row'],round(r['co2_ppm'],10))]=r
            e.evidence.write(line)
    with gzip.open(parent/'search_history.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:e.iterations.write(line)
    e.grid=[e.cache[(int(r['horizon_s']),r['state_id'],int(r['forcing_row']),round(float(r['co2_ppm']),10))] for r in priorgrid]
    e.reversals=readrows(parent/'reversal_roots.csv')
    e.intersections=readrows(parent/'intersection_brackets.csv')
    e.coverage=readrows(parent/'coverage.csv')
    prior_t=readrows(parent/'transition_roots.csv')
    shutil.copy2(parent/'regression.json',e.out/'regression.json')
    for name in parent.glob('EXP1_5_*'):shutil.copy2(name,e.out/name.name)
    dump(e.out/'parent_evidence.json',{'directory':str(parent),'output_hashes_sha256':digest(parent/'output_hashes.json'),
                                     'reason':'Contribution-zero seeding for narrow dominant phases'})
    try:
        for state in states:
            for fr in forcing:
                for h in range(300,1801,60):
                    fine=[250.+12.5*k for k in range(29)]
                    old_r=[r for r in e.reversals if r['state_id']==state and int(r['forcing_row'])==fr and int(r['horizon_s'])==h]
                    seeds=set(fine+[float(r['co2_ppm']) for r in old_r])
                    for term in e.adapter.terms:
                        vals=[e.evaluate(h,state,fr,c)[term] for c in fine]
                        if max(abs(v) for v in vals)<1e-8:continue
                        roots=roots_on_grid(lambda c:e.evaluate(h,state,fr,c)[term],fine)
                        for r in roots:
                            seeds.add(r['x']);e.record_search('CONTRIBUTION_ZERO',h,state,fr,r,(term,))
                    seeds=sorted(seeds)
                    final=[]
                    for i,j in e.pairs:
                        fn=lambda c,i=i,j=j:abs(e.evaluate(h,state,fr,c)[i])-abs(e.evaluate(h,state,fr,c)[j])
                        vals=[fn(c) for c in seeds]
                        candidates=[]
                        for k,c in enumerate(seeds):
                            x=e.evaluate(h,state,fr,c)
                            if abs(vals[k])<=1e-7 and max(abs(x[i]),abs(x[j]))>1e-8 and dominant_pair({t:x[t] for t in e.adapter.terms},i,j):
                                candidates.extend(roots_on_grid(fn,[c]))
                        for k in range(len(seeds)-1):
                            if vals[k]*vals[k+1]>=0:continue
                            x,y=[e.evaluate(h,state,fr,c) for c in seeds[k:k+2]]
                            if not any(z['dominant'] in (i,j) for z in [x,y]):continue
                            candidates.extend(roots_on_grid(fn,seeds[k:k+2]))
                        seen=[]
                        for r in sorted(candidates,key=lambda r:r['x']):
                            if any(abs(r['x']-c)<1e-3 for c in seen):continue
                            seen.append(r['x'])
                            x=e.evaluate(h,state,fr,r['x']);v={t:x[t] for t in e.adapter.terms}
                            e.record_search('SEEDED_PAIR_EQUALITY',h,state,fr,r,(i,j))
                            if not dominant_pair(v,i,j):continue
                            l=e.evaluate(h,state,fr,max(250,r['x']-.1));rr=e.evaluate(h,state,fr,min(600,r['x']+.1))
                            row={**x,'term_i':i,'term_j':j,'gap_ppm':fn(r['x']), 'root_lo':r['lo'],'root_hi':r['hi'],'root_status':r['status'],
                                 'left_eval_id':l['eval_id'],'right_eval_id':rr['eval_id'],'left_phase':l['phase'],'right_phase':rr['phase'],
                                 'resolved_phase_change':l['resolved_under_screen'] and rr['resolved_under_screen'] and l['dominant']!=rr['dominant'],
                                 'opposite_sign_pair':x[i]*x[j]<0,
                                 'remaining_plus_epsilon_ppm':sum(x[t] for t in e.adapter.terms if t not in(i,j))+x['epsilon_ppm']}
                            final.append(row)
                    parent_cell=[r for r in prior_t if r['state_id']==state and int(r['forcing_row'])==fr and int(r['horizon_s'])==h]
                    for r in parent_cell:
                        if not any(r['term_i']==t['term_i'] and r['term_j']==t['term_j'] and abs(float(r['co2_ppm'])-float(t['co2_ppm']))<1e-3 for t in final):
                            r['co2_ppm']=float(r['co2_ppm']);r['response_ppm']=float(r['response_ppm'])
                            r['resolved_phase_change']=r['resolved_phase_change']=='True'
                            final.append(r)
                    e.transitions.extend(final)
                    e.coverage.append({'state':state,'forcing':fr,'h':h,'kind':'CONTRIBUTION_ZERO_SEEDED',
                                       'coarse_count':len(parent_cell),'fine_count':len(final),
                                       'status':'ADDITIONAL_CANDIDATES' if len(final)>len(parent_cell) else 'MATCH'})
                    for t in final:
                        for r in old_r:
                            e.distances.append({'state':state,'forcing':fr,'h':h,'pair':t['term_i']+'|'+t['term_j'],
                                                'transition_co2':float(t['co2_ppm']),'reversal_co2':float(r['co2_ppm']),
                                                'signed_distance_ppm':float(t['co2_ppm'])-float(r['co2_ppm']),'transition_status':t['root_status']})
                    e.save_tables()
                    print(old['model'],state,fr,h,'transitions',len(final),'evaluations',len(e.cache),flush=True)
    except BaseException:
        e.finish('FAILED',traceback.format_exc());raise
    e.finish('PASS_EXECUTION')


if __name__=='__main__':main()
