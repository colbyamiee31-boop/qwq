import hashlib, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/'latent_history_config.json').read_text())

R=8.3144598
K=273.15
MCO2=44.01e-3
P=101325.0

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def state_sha(x):
    return hashlib.sha256(np.asarray(x,dtype=np.float64).tobytes()).hexdigest()

def sat_vp_pa(t):
    t=np.asarray(t,dtype=float)
    return 610.78*np.exp(17.2694*t/(t+238.3))

def ppm_to_mg_m3(t,ppm):
    return P*np.asarray(ppm)*MCO2/(R*(np.asarray(t)+K))

def mg_m3_to_ppm(t,mg):
    return R*(np.asarray(t)+K)*np.asarray(mg)/(P*MCO2)

def rh_from_t_vp(t,vp):
    return np.clip(100*np.asarray(vp)/sat_vp_pa(t),0,100)

def dedup_roots(roots,tol):
    good=[r for r in roots if np.isfinite(r['root_ppm'])]
    good.sort(key=lambda r:r['root_ppm'])
    out=[]
    for r in good:
        if not out or abs(r['root_ppm']-out[-1]['root_ppm'])>tol:
            out.append(r)
        elif abs(r['response_ppm'])<abs(out[-1]['response_ppm']):
            out[-1]=r
    return out
