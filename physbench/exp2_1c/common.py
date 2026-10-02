import hashlib
import numpy as np
from pathlib import Path
R=8.3144598; K=273.15; MCO2=44.01e-3; P=101325.0
def sat_vp_pa(t):
    t=np.asarray(t,dtype=float); return 610.78*np.exp(17.2694*t/(t+238.3))
def rh_from_t_vp(t,vp):
    return np.clip(100*np.asarray(vp,dtype=float)/sat_vp_pa(t),0,100)
def ah_from_t_vp(t,vp):
    return 216.7*(np.asarray(vp,dtype=float)/100.0)/(np.asarray(t,dtype=float)+273.15)
def ppm_to_mg_m3(t, ppm):
    return P*np.asarray(ppm,dtype=float)*MCO2/(R*(np.asarray(t,dtype=float)+K))
def mg_m3_to_ppm(t, mg):
    return R*(np.asarray(t,dtype=float)+K)*np.asarray(mg,dtype=float)/(P*MCO2)
def sign_model(delta, high, low):
    scale=max(1.0,abs(float(high)),abs(float(low))); eps=1e-8*scale
    return 1 if delta>eps else (-1 if delta<-eps else 0)
def sign_obs(delta):
    return 1 if delta>1e-12 else (-1 if delta<-1e-12 else 0)
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
