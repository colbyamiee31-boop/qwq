"""Audit whether narrow sign brackets contain a residual-resolved equality.

No scientific contribution or state solver is changed. Parent tables and all
parent evidence remain available. This is a numerical identifiability audit.
"""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import traceback
from adapters import digest
from numerics import bisect, dominant_pair
from run import Experiment, dump, table
from refine import readrows


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--model-source',type=Path,required=True)
    a=p.parse_args();parent=a.parent.resolve()
    result=json.loads((parent/'result.json').read_text())
    if result['status']!='PASS_EXECUTION':raise RuntimeError('Parent incomplete')
    grid=readrows(parent/'phase_grid.csv');states=sorted({r['state_id'] for r in grid});forcing=sorted({int(r['forcing_row']) for r in grid})
    e=Experiment(result['model'],a.model_source,a.out,states[0] if len(states)==1 else None,forcing[0] if len(forcing)==1 else None)
    for name in ['polish.py','IDENTIFIABILITY_AUDIT.md']:shutil.copy2(Path(__file__).with_name(name),e.out/name)
    with gzip.open(parent/'evaluations.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)['evaluation']
            e.cache[(row['horizon_s'],row['state_id'],row['forcing_row'],round(row['co2_ppm'],10))]=row
            e.evidence.write(line)
    with gzip.open(parent/'search_history.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:e.iterations.write(line)
    for attr,filename in [('grid','phase_grid'),('reversals','reversal_roots'),('transitions','transition_roots'),
                          ('coverage','coverage'),('intersections','intersection_brackets'),('distances','signed_separation')]:
        setattr(e,attr,readrows(parent/f'{filename}.csv'))
    for r in e.grid:r['resolved_under_screen']=r['resolved_under_screen']=='True'
    for r in e.transitions:r['resolved_phase_change']=r['resolved_phase_change']=='True'
    for f in parent.glob('EXP1_5_*'):shutil.copy2(f,e.out/f.name)
    shutil.copy2(parent/'regression.json',e.out/'regression.json')
    dump(e.out/'parent_evidence.json',{'directory':str(parent),'output_hashes_sha256':digest(parent/'output_hashes.json')})
    audit=[]
    try:
        for k,row in enumerate(e.transitions):
            h=int(row['horizon_s']);state=row['state_id'];fr=int(row['forcing_row'])
            i,j=row['term_i'],row['term_j'];lo=float(row['root_lo']);hi=float(row['root_hi'])
            fn=lambda c:abs(e.evaluate(h,state,fr,c)[i])-abs(e.evaluate(h,state,fr,c)[j])
            flo,fhi=fn(lo),fn(hi)
            if flo*fhi>0 and min(abs(flo),abs(fhi))>1e-7:
                r={'x':float(row['co2_ppm']),'f':fn(float(row['co2_ppm'])),'lo':lo,'hi':hi,'status':'BRACKET_NOT_REPRODUCED','history':[]}
            else:
                r=bisect(fn,lo,hi,width=1e-8,residual=1e-7,limit=60)
            e.record_search('IDENTIFIABILITY_POLISH',h,state,fr,r,(i,j))
            x=e.evaluate(h,state,fr,r['x']);v={t:x[t] for t in e.adapter.terms}
            eligible=dominant_pair(v,i,j)
            residual_ok=abs(r['f'])<=1e-7 and eligible
            left=e.evaluate(h,state,fr,r['lo']);right=e.evaluate(h,state,fr,r['hi'])
            audit.append({'candidate_eval_id':row['eval_id'],'polished_eval_id':x['eval_id'],
                          'state':state,'forcing':fr,'h':h,'pair':i+'|'+j,'co2_ppm':r['x'],
                          'gap_ppm':r['f'],'response_ppm':x['response_ppm'],'epsilon_ppm':x['epsilon_ppm'],
                          'lo_ppm':r['lo'],'hi_ppm':r['hi'],'bracket_width_ppm':r['hi']-r['lo'],
                          'gap_lo_ppm':abs(left[i])-abs(left[j]),'gap_hi_ppm':abs(right[i])-abs(right[j]),
                          'third_term_eligible':eligible,'residual_resolved':residual_ok,
                          'status':'RESIDUAL_RESOLVED' if residual_ok else 'UNRESOLVED_EQUALITY',
                          'solver_status':r['status'],'observed_closure_is_rigorous_error_bound':False})
            if k%50==0:print(result['model'],'polish',k,'/',len(e.transitions),'evals',len(e.cache),flush=True)
        table(e.out/'transition_polish.csv',audit)
        dump(e.out/'identifiability_summary.json',{'candidates':len(audit),
             'residual_resolved':sum(r['residual_resolved'] for r in audit),
             'unresolved_equality':sum(not r['residual_resolved'] for r in audit),
             'width_ppm':1e-8,'residual_ppm':1e-7,'changes_to_model_solver':False})
    except BaseException:
        table(e.out/'transition_polish.csv',audit);e.finish('FAILED',traceback.format_exc());raise
    e.finish('PASS_EXECUTION')


if __name__=='__main__':main()
