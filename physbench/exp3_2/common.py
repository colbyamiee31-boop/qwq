import hashlib
import numpy as np

R=8.3144598; K=273.15; MCO2=44.01e-3; P=101325.0

def sat_vp_pa(t):
    t=np.asarray(t,dtype=float)
    return 610.78*np.exp(17.2694*t/(t+238.3))

def ah_from_t_vp(t,vp):
    return 216.7*np.asarray(vp,dtype=float)/(np.asarray(t,dtype=float)+273.15)

def ppm_to_mg_m3(t, ppm):
    return P*np.asarray(ppm)*MCO2/(R*(np.asarray(t)+K))

def mg_m3_to_ppm(t, mg):
    return R*(np.asarray(t)+K)*np.asarray(mg)/(P*MCO2)

def sha256(path):
    return hashlib.sha256(open(path,'rb').read()).hexdigest()
