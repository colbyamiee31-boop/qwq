import numpy as np

H_TARGET=7.375
H_AIR_M1_TARGET=6.97
VENT_GROSS_RATIO=0.2515625
VENT_PROJECTED_RATIO=0.17474999631859142
STAGES=('H','HV')
ACTIONS=np.array([0.1,0.3,0.5,0.7,0.9],dtype=float)
R=8.3144598; K=273.15; MCO2=44.01e-3; P=101325.0

def sat_vp_pa(t):
    t=np.asarray(t,dtype=float)
    return 610.78*np.exp(17.2694*t/(t+238.3))

def ppm_to_mg_m3(t,ppm):
    return P*np.asarray(ppm,dtype=float)*MCO2/(R*(np.asarray(t,dtype=float)+K))

def mg_m3_to_ppm(t,mg):
    return R*(np.asarray(t,dtype=float)+K)*np.asarray(mg,dtype=float)/(P*MCO2)

def ah_from_t_vp(t,vp):
    return 216.7*(np.asarray(vp,dtype=float)/100.0)/(np.asarray(t,dtype=float)+273.15)

def sign_model(delta,high,low):
    scale=max(1.0,abs(float(high)),abs(float(low)))
    eps=1e-8*scale
    return 1 if delta>eps else (-1 if delta<-eps else 0)

def sign_obs(delta):
    return 1 if delta>1e-12 else (-1 if delta<-1e-12 else 0)

def harmonise_m1_p(p,stage):
    q=np.asarray(p,dtype=float).copy()
    if stage not in STAGES:
        raise ValueError(stage)
    q[48]=H_AIR_M1_TARGET
    q[49]=H_TARGET
    # dependent air heat capacities and CO2 capacities
    q[112]=q[48]*q[111]*q[23]
    q[120]=(q[49]-q[48])*q[111]*q[23]
    q[122]=q[48]
    q[123]=q[49]-q[48]
    if stage=='HV':
        q[55]=VENT_PROJECTED_RATIO*q[46]
    return q

def m1_geometry_audit(p):
    p=np.asarray(p,dtype=float)
    return {
        'hAir_m':float(p[48]),
        'hGh_m':float(p[49]),
        'effective_height_m':float(p[49]),
        'roof_aperture_ratio_m2_m2':float(p[55]/p[46]),
        'capAir':float(p[112]),
        'capTop':float(p[120]),
        'capCo2Air':float(p[122]),
        'capCo2Top':float(p[123])
    }
