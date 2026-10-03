import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

Q_LEVELS=np.array([0.0,0.25,0.5,0.75,1.0],float)
U9=np.round(np.arange(0.1,1.0,0.1),1)
TARGET_SHA256='e07e64d8f8f7ac914a68b5cda73e0a76f5b9bd890a81473a42dd8ee8f6656292'
INPUT_SHA256='8a5890dc9fb6950ce3c3e7ffc8ff900133ede8ffde99dfa68d79435314b56023'
M1_COMMIT='2d3febb1ea002b24b452e32293e990beb78d3ce1'
M2_COMMIT='bea8c3b0a1324162a4b5487db578aa674c8b587c'
DV_ATOL=2e-5
DN_ATOL=5e-6
DOSE_RTOL=2e-7
MAX_INVERSION_ITER=24
RANGE_TOL=1e-10

COORD_COL={'DV':'specific_volume_dose_m3_m2','DN':'air_volume_equiv'}

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dose_tolerance(coord,target):
    atol=DV_ATOL if coord=='DV' else DN_ATOL
    return float(atol+DOSE_RTOL*abs(float(target)))

def validate_target_file(path):
    path=Path(path)
    got=sha256(path)
    if got!=TARGET_SHA256:
        raise AssertionError(f'target SHA mismatch: {got}')
    t=pd.read_csv(path)
    req={'event_id','team','event_date','daynight','horizon_min','coordinate','q','eligible_for_exp3_4b',
         'common_low','common_high','common_width','target_dose'}
    miss=req-set(t.columns)
    if miss: raise AssertionError(f'target columns missing: {sorted(miss)}')
    if set(np.round(t.q.unique(),12))!=set(Q_LEVELS): raise AssertionError('q grid changed')
    if len(t)!=1220 or t.duplicated(['event_id','horizon_min','coordinate','q']).any():
        raise AssertionError('target row identity changed')
    dv=t[(t.horizon_min==15)&(t.coordinate=='DV')&t.eligible_for_exp3_4b]
    dn=t[(t.horizon_min==15)&(t.coordinate=='DN')&t.eligible_for_exp3_4b]
    if dv.event_id.nunique()!=42 or len(dv)!=210: raise AssertionError('DV15 cohort changed')
    if dn.event_id.nunique()!=61 or len(dn)!=305: raise AssertionError('DN15 cohort changed')
    if not {27,86,89}.issubset(set(dv.event_id.astype(int))): raise AssertionError('locked primary events not DV-matchable')
    for x in (dv,dn):
        vals=x.target_dose.to_numpy(float)
        if not np.all(np.isfinite(vals)): raise AssertionError('nonfinite eligible target')
        if np.any(vals < x.common_low.to_numpy(float)-RANGE_TOL) or np.any(vals > x.common_high.to_numpy(float)+RANGE_TOL):
            raise AssertionError('target outside common interval')
    return t

def locked_bracket(grid_event,coord,target):
    col=COORD_COL[coord]
    g=grid_event.sort_values('action')
    a=g.action.to_numpy(float); d=g[col].to_numpy(float)
    if len(g)!=9 or not np.allclose(a,U9,rtol=0,atol=1e-12): raise AssertionError('locked U9 grid changed')
    if not np.all(np.isfinite(d)) or np.any(np.diff(d)<=0): raise AssertionError('dose grid not strictly increasing')
    target=float(target)
    if target < d[0]-RANGE_TOL or target > d[-1]+RANGE_TOL: raise AssertionError('target requires extrapolation')
    j=int(np.searchsorted(d,target,side='left'))
    if j==0: return float(a[0]),float(a[0]),float(d[0]),float(d[0])
    if j>=len(d): return float(a[-1]),float(a[-1]),float(d[-1]),float(d[-1])
    if abs(d[j]-target)<=RANGE_TOL:
        return float(a[j]),float(a[j]),float(d[j]),float(d[j])
    return float(a[j-1]),float(a[j]),float(d[j-1]),float(d[j])

def solve_target(grid_event,coord,target,simulate):
    tol=dose_tolerance(coord,target)
    ulo,uhi,dlo,dhi=locked_bracket(grid_event,coord,target)
    best=None; seen=set(); iterations=0

    def run(u):
        nonlocal best,iterations
        rec=simulate(float(u)); iterations+=1
        us=float(rec['native_command']); val=float(rec[COORD_COL[coord]])
        err=abs(val-float(target))
        if not (0.1-1e-12 <= us <= 0.9+1e-12): raise AssertionError('native command extrapolation')
        if not np.isfinite(val): raise AssertionError('nonfinite actual dose')
        if best is None or err<best[0]: best=(err,rec.copy())
        return us,val,err,rec

    # Exact locked endpoint first.
    if abs(uhi-ulo)<=1e-15:
        us,val,err,rec=run(ulo)
        if err<=tol:
            return rec,iterations,ulo,uhi,tol
        # If the locked endpoint shifted numerically, fail closed rather than extrapolate.
        raise AssertionError(f'endpoint target mismatch coord={coord} target={target} actual={val} err={err} tol={tol}')

    lo_u,hi_u,lo_v,hi_v=ulo,uhi,dlo,dhi
    for _ in range(MAX_INVERSION_ITER):
        if hi_v<=lo_v:
            raw=0.5*(lo_u+hi_u)
        else:
            raw=lo_u+(float(target)-lo_v)*(hi_u-lo_u)/(hi_v-lo_v)
            if not (lo_u < raw < hi_u): raw=0.5*(lo_u+hi_u)
        # Avoid stalling on a previously tested native command.
        key=round(float(raw),15)
        if key in seen: raw=0.5*(lo_u+hi_u)
        seen.add(round(float(raw),15))
        us,val,err,rec=run(raw)
        if err<=tol:
            return rec,iterations,ulo,uhi,tol
        if val < float(target):
            lo_u,lo_v=us,val
        else:
            hi_u,hi_v=us,val
        if hi_u-lo_u<=1e-12: break
    if best is not None and best[0]<=tol:
        return best[1],iterations,ulo,uhi,tol
    raise AssertionError(f'inversion failed coord={coord} target={target} best_err={None if best is None else best[0]} tol={tol} bracket=({ulo},{uhi})')
