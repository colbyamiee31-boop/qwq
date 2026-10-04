import numpy as np

H_TARGET=7.375
VENT_PROJECTED_RATIO=0.17474999631859142
CTIFL_FLOOR_AREA=1036.8
VENT_L=4.05
VENT_H=1.40
BANK_L=23
BANK_W=23
SEED=20261004
BOOT=2000
ACTIONS=np.array([0.1,0.3,0.5,0.7,0.9],dtype=float)

R=8.3144598
K=273.15
MCO2=44.01e-3
P=101325.0

def ppm_to_mg_m3(t,ppm):
    return P*np.asarray(ppm,dtype=float)*MCO2/(R*(np.asarray(t,dtype=float)+K))

def mg_m3_to_ppm(t,mg):
    return R*(np.asarray(t,dtype=float)+K)*np.asarray(mg,dtype=float)/(P*MCO2)

def sat_vp_pa(t):
    t=np.asarray(t,dtype=float)
    return 610.78*np.exp(17.2694*t/(t+238.3))

def ah_from_t_vp(t,vp):
    return 216.7*(np.asarray(vp,dtype=float)/100.0)/(np.asarray(t,dtype=float)+273.15)

def gl_fun(delta_deg):
    d=np.asarray(delta_deg,dtype=float)
    return 2.46e-2*(1.0-np.exp(-d/14.5))

def gw_fun(delta_deg):
    d=np.asarray(delta_deg,dtype=float)
    return -1.89e-5*d*d+2.23e-3*d

def q_common_pct(leeward_pct,windward_pct,wind_speed):
    vl=np.asarray(leeward_pct,dtype=float)/100.0
    vw=np.asarray(windward_pct,dtype=float)/100.0
    u=np.asarray(wind_speed,dtype=float)
    dl=44.0*vl
    dw=44.0*vw
    q=u*VENT_L*VENT_H/CTIFL_FLOOR_AREA*(BANK_L*gl_fun(dl)+BANK_W*gw_fun(dw))
    return q

def q_common_symmetric(action,wind_speed):
    a=np.asarray(action,dtype=float)
    return q_common_pct(100.0*a,100.0*a,wind_speed)

def sign_model(delta,high,low):
    scale=max(1.0,abs(float(high)),abs(float(low)))
    eps=1e-8*scale
    return 1 if delta>eps else (-1 if delta<-eps else 0)

def sign_obs(delta):
    return 1 if delta>1e-12 else (-1 if delta<-1e-12 else 0)

def harmonise_m1_hv(p):
    q=np.asarray(p,dtype=float).copy()
    q[48]=6.97
    q[49]=H_TARGET
    q[112]=q[48]*q[111]*q[23]
    q[120]=(q[49]-q[48])*q[111]*q[23]
    q[122]=q[48]
    q[123]=q[49]-q[48]
    q[55]=VENT_PROJECTED_RATIO*q[46]
    return q

def m1_native_leakage(p,wind):
    p=np.asarray(p,dtype=float)
    return float(max(float(p[205]),float(wind))*float(p[60]))

def m2_native_leakage(p,wind):
    return float(max(float(p['Cleakage'])*0.25,float(p['Cleakage'])*float(wind)))
