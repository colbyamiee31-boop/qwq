import io,json,zipfile
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'CTIFL_Supplementary_Information_v0.zip'
GEN=ROOT/'physbench/r1_2/generated'
OUT=ROOT/'physbench/r2/generated'; OUT.mkdir(parents=True,exist_ok=True)
P97=GEN/'PRIMARY_97_MODEL_INPUT.csv'
S36=GEN/'STRICT_36_MODEL_INPUT.csv'
EXPECTED_P='236e6631f9f60b7adfa942e496f1caea36e2eef4ddfc0149cebebe7524024051'
EXPECTED_S='f144244dda17c88f7e1f2ed1eecc62810fb7e4dce449602f231d41b5254b84f0'
A=1036.8; LO=4.05; HO=1.4; HCT=7.375
def sha256(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def gl(d): return 2.46e-2*(1.0-np.exp(-np.asarray(d,dtype=float)/14.5))
def gw(d): return -1.89e-5*np.asarray(d,dtype=float)**2+2.23e-3*np.asarray(d,dtype=float)
def flux(vl,vw,u,nl,nw):
    dl=44.0*np.asarray(vl,dtype=float)/100.0
    dw=44.0*np.asarray(vw,dtype=float)/100.0
    return np.asarray(u,dtype=float)*LO*HO/A*(float(nl)*gl(dl)+float(nw)*gw(dw))
def augment(path,cohort):
    d=pd.read_csv(path)
    with zipfile.ZipFile(RAW) as z:
        name='Dataset/1_Measurements_dataset/4_WeatherData.csv'
        w=pd.read_csv(io.BytesIO(z.read(name)),skiprows=2)
    w=w[['Time','Wind direction','Wind speed']].copy()
    x=d.merge(w,left_on='Time_s',right_on='Time',how='left',validate='one_to_one')
    if x[['Wind direction','Wind speed']].isna().any().any(): raise AssertionError('missing weather match')
    if np.max(np.abs(x.event_Windsp-x['Wind speed']))>1e-12: raise AssertionError('wind speed mismatch')
    x['cohort']=cohort
    for nl,nw,label in [(23,23,'23_23'),(22,24,'22_24'),(24,22,'24_22')]:
        pre=flux(x.leeward_pre_pct,x.windward_pre_pct,x.event_Windsp,nl,nw)
        post=flux(x.leeward_post_pct,x.windward_post_pct,x.event_Windsp,nl,nw)
        x[f'emp_flux_pre_{label}']=pre; x[f'emp_flux_post_{label}']=post
        x[f'emp_delta_DV_{label}']=900.0*(post-pre)
        x[f'emp_delta_DN_{label}']=x[f'emp_delta_DV_{label}']/HCT
    x['anchor_primary_3_6']=((x.event_Windsp>=3.0)&(x.event_Windsp<=6.0))
    x['anchor_expanded_2_8']=((x.event_Windsp>=2.0)&(x.event_Windsp<=8.0))
    return x
if sha256(P97)!=EXPECTED_P: raise AssertionError(('P97 sha',sha256(P97)))
if sha256(S36)!=EXPECTED_S: raise AssertionError(('S36 sha',sha256(S36)))
p=augment(P97,'primary'); s=augment(S36,'strict')
p.to_csv(OUT/'PRIMARY_97_R2_INPUT.csv',index=False,float_format='%.12g')
s.to_csv(OUT/'STRICT_36_R2_INPUT.csv',index=False,float_format='%.12g')
summary={
 'primary_rows':len(p),'strict_rows':len(s),
 'primary_3_6':int(p.anchor_primary_3_6.sum()),
 'primary_2_8':int(p.anchor_expanded_2_8.sum()),
 'strict_3_6':int(s.anchor_primary_3_6.sum()),
 'primary_opening_3_6':int(((p.anchor_primary_3_6)&(p.direction=='opening')).sum()),
 'primary_closing_3_6':int(((p.anchor_primary_3_6)&(p.direction=='closing')).sum()),
 'ctifl_effective_height_m':HCT,'floor_area_m2':A,
 'principal_vent_m':[LO,HO],
 'primary_bank_counts':[23,23],
 'bracket_bank_counts':[[22,24],[24,22]],
 'all_empirical_anchor_finite':bool(np.isfinite(p.filter(regex='emp_').to_numpy(float)).all() and np.isfinite(s.filter(regex='emp_').to_numpy(float)).all())
}
(OUT/'PREPARE_SUMMARY.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
if not (summary['primary_3_6']==40 and summary['strict_3_6']==14 and summary['all_empirical_anchor_finite']):
    raise SystemExit('R2 anchor preparation gate failed')
