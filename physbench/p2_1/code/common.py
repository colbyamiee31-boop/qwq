from types import SimpleNamespace
import numpy as np, pandas as pd
M1_COMMIT='2d3febb1ea002b24b452e32293e990beb78d3ce1'
M2_COMMIT='bea8c3b0a1324162a4b5487db578aa674c8b587c'
SEED=20261004
FORCING_NODE_MIN=np.arange(0,361,15,dtype=float)
HOURS=(1,2,3,4,5,6)
ARMS={
 'V0E0':{'init':'I0','vent':0.0,'enclosure':0.0,'primary':1},
 'V50E0':{'init':'I0','vent':0.5,'enclosure':0.0,'primary':1},
 'V100E0':{'init':'I0','vent':1.0,'enclosure':0.0,'primary':1},
 'V0E1':{'init':'I0','vent':0.0,'enclosure':1.0,'primary':0},
 'I1_V0E0':{'init':'I1','vent':0.0,'enclosure':0.0,'primary':0},
}
MAPS={
 'greenhouse':['G1','G2','G3'], 'season':['DJF','MAM','JJA','SON'], 'daynight':['night','day'],
 'wind_bin_m_s':['<1','1-3','3-6','>=6'], 'Tin_minus_Tout_bin_C':['<=0','0-5','5-15','>15'],
 'T_stratification_bin_C':['<=1','1-3','>3'],
}

def load_compact(path,shard,nshards=8):
    z=np.load(path)
    idx=np.where(((z['atlas_num'].astype(int)-1)%nshards)==int(shard))[0]
    recs=[]
    scalar=['segment_id','Tin0_C','RHin0_pct','VP0_Pa','CO20_ppm','T06_0_C','T18_0_C','T30_0_C','RH06_0_pct','RH18_0_pct','RH30_0_pct','Tout0_C','Gout0_W_m2','WS2M0_m_s','dTio0_C','ST0_C']
    arrays=['forcing_Tout_C','forcing_RHout_pct','forcing_Gout_W_m2','forcing_WS2M_m_s','forcing_precip_mm']
    for i in idx:
        d={'atlas_id':f"ALAR-P2-{int(z['atlas_num'][i]):04d}"}
        d['timestamp_UTCplus6']=pd.Timestamp(int(z['timestamp_min'][i])*60,unit='s')
        for m,vals in MAPS.items(): d[m]=vals[int(z[m+'_code'][i])]
        for c in scalar: d[c]=float(z[c][i])
        d['segment_id']=int(z['segment_id'][i])
        for c in arrays: d[c]=np.asarray(z[c][i],dtype=float)
        d['obs_Tin']=np.asarray(z['obs_Tin'][i],dtype=float); d['obs_RHin']=np.asarray(z['obs_RHin'][i],dtype=float)
        d['obs_available']=np.asarray(z['obs_available'][i],dtype=np.uint8); d['f4_mask']=np.asarray(z['f4_mask'][i],dtype=np.uint8)
        recs.append(SimpleNamespace(**d))
    return recs

def sat_vp_pa(temp_c):
    t=np.asarray(temp_c,dtype=float); return 610.78*np.exp(17.2694*t/(t+238.3))
def rh_to_vp_pa(temp_c,rh_pct): return sat_vp_pa(temp_c)*np.asarray(rh_pct,dtype=float)/100.0
def ah_from_t_vp(temp_c,vp_pa): return 18.01528*np.asarray(vp_pa,dtype=float)/(8.3144598*(np.asarray(temp_c,dtype=float)+273.15))
def ppm_to_mg_m3(temp_c,ppm):
    R=8.3144598; P=101325.; M=44.01e-3; return P*1e-6*np.asarray(ppm,dtype=float)*M/(R*(np.asarray(temp_c,dtype=float)+273.15))*1e6
def mg_m3_to_ppm(temp_c,dens_mg_m3):
    R=8.3144598; P=101325.; M=44.01e-3; return 1e6*R*(np.asarray(temp_c,dtype=float)+273.15)*(np.asarray(dens_mg_m3,dtype=float)*1e-6)/(P*M)
def interp_fun(values):
    xs=FORCING_NODE_MIN*60.; ys=np.asarray(values,dtype=float); return lambda t: float(np.interp(float(t),xs,ys))
