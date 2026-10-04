import json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'physbench/r4'))
from common import *

OUT=ROOT/'evidence/R4/FINAL'
OUT.mkdir(parents=True,exist_ok=True)

M1=pd.read_csv(ROOT/'evidence/R4/M1/ctifl_common_flux_responses.csv')
M2=pd.read_csv(ROOT/'evidence/R4/M2/ctifl_common_flux_responses.csv')
G1=pd.read_csv(ROOT/'evidence/R4/M1/d2_common_flux_grid.csv')
G2=pd.read_csv(ROOT/'evidence/R4/M2/d2_common_flux_grid.csv')
S1=json.loads((ROOT/'evidence/R4/M1/summary.json').read_text())
S2=json.loads((ROOT/'evidence/R4/M2/summary.json').read_text())
PIN=pd.read_csv(ROOT/'physbench/r4/generated/PRIMARY_97_R2_INPUT.csv')
SIN=pd.read_csv(ROOT/'physbench/r4/generated/STRICT_36_R2_INPUT.csv')

R3DIV=pd.read_csv(ROOT/'baseline_r3/evidence/R3/FINAL/03_cross_model_response_divergence_event.csv')
R3DE=pd.read_csv(ROOT/'baseline_r3/evidence/R3/FINAL/06_d2_harmonised_decision_event_metrics.csv')

B=BOOT

def kappa(a,b):
    a=np.asarray(a); b=np.asarray(b)
    if len(a)==0:return np.nan
    p0=np.mean(a==b)
    pa=np.mean(a==1); pb=np.mean(b==1)
    pe=pa*pb+(1-pa)*(1-pb)
    return np.nan if abs(1-pe)<1e-15 else float((p0-pe)/(1-pe))

def val_metric(f,var):
    ms=np.array([sign_model(d,hi,lo) for d,hi,lo in zip(f[f'model_{var}_delta'],f[f'post_{var}'],f[f'pre_{var}'])])
    os=np.array([sign_obs(x) for x in f[f'obs_{var}_delta']])
    keep=(ms!=0)&(os!=0)
    ff=f.loc[keep].copy(); ms=ms[keep]; os=os[keep]
    conc=float(np.mean(ms==os)) if len(ms) else np.nan
    kap=kappa(ms,os)
    gradcol='T_gradient_C' if var=='T' else 'AH_gradient_g_m3'
    den=np.abs(ff[gradcol].to_numpy(float))*np.abs(ff.delta_u.to_numpy(float))
    good=den>1e-12
    rho=float(spearmanr(
        ff.loc[good,f'model_{var}_delta'].to_numpy(float)/den[good],
        ff.loc[good,f'obs_{var}_delta'].to_numpy(float)/den[good]
    ).statistic) if good.sum()>=3 else np.nan
    return {'n_resolved':int(len(ms)),'concordance':conc,'kappa':kap,'spearman_norm':rho,'unresolved':int(len(f)-len(ms))}

def cluster_boot_val(f,var):
    dates=np.array(sorted(f.event_date.unique()))
    groups={d:f[f.event_date==d] for d in dates}
    rng=np.random.default_rng(SEED)
    vals=[]
    for _ in range(B):
        samp=rng.choice(dates,size=len(dates),replace=True)
        q=pd.concat([groups[d] for d in samp],ignore_index=True)
        m=val_metric(q,var)
        vals.append([m['concordance'],m['kappa'],m['spearman_norm']])
    a=np.asarray(vals,float)
    out={}
    for j,name in enumerate(['concordance','kappa','spearman_norm']):
        v=a[:,j]; v=v[np.isfinite(v)]
        out[name]=[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))] if len(v) else [np.nan,np.nan]
    return out

CUR=pd.concat([M1,M2],ignore_index=True)

# ---------- empirical metrics ----------
vrows=[]
for cohort in ['primary','strict']:
    for stage in ['CF0','CFN']:
        for model in ['M1','M2']:
            f=CUR[(CUR.cohort==cohort)&(CUR.stage==stage)&(CUR.model==model)]
            for var in ['T','AH']:
                m=val_metric(f,var); ci=cluster_boot_val(f,var)
                vrows.append({
                    'cohort':cohort,'stage':stage,'model':model,'variable':var,**m,
                    'concordance_ci_low':ci['concordance'][0],'concordance_ci_high':ci['concordance'][1],
                    'kappa_ci_low':ci['kappa'][0],'kappa_ci_high':ci['kappa'][1],
                    'spearman_ci_low':ci['spearman_norm'][0],'spearman_ci_high':ci['spearman_norm'][1]
                })
VAL=pd.DataFrame(vrows)
VAL.to_csv(OUT/'01_common_flux_empirical_validation.csv',index=False,float_format='%.12g')

# ---------- response divergence current ----------
def current_div(stage,cohort='primary'):
    a=CUR[(CUR.cohort==cohort)&(CUR.stage==stage)&(CUR.model=='M1')]
    b=CUR[(CUR.cohort==cohort)&(CUR.stage==stage)&(CUR.model=='M2')]
    x=a[['event_id','event_date','model_T_delta','model_AH_delta','pre_T','post_T','pre_AH','post_AH']].merge(
        b[['event_id','model_T_delta','model_AH_delta','pre_T','post_T','pre_AH','post_AH']],
        on='event_id',suffixes=('_M1','_M2'),validate='one_to_one'
    )
    x['stage']=stage
    return add_div(x)

def add_div(x):
    x=x.copy()
    for var in ['T','AH']:
        aa=x[f'model_{var}_delta_M1'].to_numpy(float)
        bb=x[f'model_{var}_delta_M2'].to_numpy(float)
        eps=1e-12*np.maximum.reduce([np.ones(len(x)),np.abs(aa),np.abs(bb)])
        x[f'D_{var}']=np.abs(aa-bb)/(np.abs(aa)+np.abs(bb)+eps)
        s1=np.array([sign_model(d,hi,lo) for d,hi,lo in zip(
            x[f'model_{var}_delta_M1'],x[f'post_{var}_M1'],x[f'pre_{var}_M1'])])
        s2=np.array([sign_model(d,hi,lo) for d,hi,lo in zip(
            x[f'model_{var}_delta_M2'],x[f'post_{var}_M2'],x[f'pre_{var}_M2'])])
        x[f'sign_M1_{var}']=s1
        x[f'sign_M2_{var}']=s2
        x[f'both_resolved_{var}']=(s1!=0)&(s2!=0)
        x[f'sign_disagree_{var}']=(s1!=0)&(s2!=0)&(s1!=s2)
    x['D_resp']=0.5*(x.D_T+x.D_AH)
    return x

ids=set(PIN.loc[PIN.anchor_primary_3_6.astype(bool),'event_id'])
base=R3DIV[(R3DIV.stage=='HV')&(R3DIV.event_id.isin(ids))].copy()
assert len(base)==40
base=base[['event_id','event_date','model_T_delta_M1','model_AH_delta_M1','pre_T_M1','post_T_M1','pre_AH_M1','post_AH_M1',
           'model_T_delta_M2','model_AH_delta_M2','pre_T_M2','post_T_M2','pre_AH_M2','post_AH_M2']].copy()
base['stage']='HV'
base=add_div(base)
cf0=current_div('CF0')
cfn=current_div('CFN')
DIV=pd.concat([base,cf0,cfn],ignore_index=True)
DIV.to_csv(OUT/'02_response_divergence_event.csv',index=False,float_format='%.12g')

drows=[]
for stage in ['HV','CF0','CFN']:
    q=DIV[DIV.stage==stage]
    drows.append({
        'stage':stage,'n_events':len(q),
        'mean_D_resp':float(q.D_resp.mean()),'median_D_resp':float(q.D_resp.median()),
        'T_sign_disagreement_fraction':float(q.loc[q.both_resolved_T,'sign_disagree_T'].mean()),
        'AH_sign_disagreement_fraction':float(q.loc[q.both_resolved_AH,'sign_disagree_AH'].mean())
    })
DS=pd.DataFrame(drows)

# paired date-cluster bootstrap vs HV
base2=base[['event_id','event_date','D_resp']].rename(columns={'D_resp':'D_HV'})
diffs=[]
for stage in ['CF0','CFN']:
    q=DIV[DIV.stage==stage][['event_id','D_resp']].rename(columns={'D_resp':f'D_{stage}'}).merge(base2,on='event_id',validate='one_to_one')
    q['diff']=q[f'D_{stage}']-q.D_HV
    dates=np.array(sorted(q.event_date.unique()))
    gd={d:q[q.event_date==d] for d in dates}
    rng=np.random.default_rng(SEED)
    vals=[]
    for _ in range(B):
        ss=rng.choice(dates,size=len(dates),replace=True)
        z=pd.concat([gd[d] for d in ss],ignore_index=True)
        vals.append(float(z['diff'].mean()))
    diffs.append({
        'stage':stage,'minus_HV_mean_D_resp':float(q['diff'].mean()),
        'ci_low':float(np.percentile(vals,2.5)),'ci_high':float(np.percentile(vals,97.5))
    })
DS=DS.merge(pd.DataFrame(diffs),on='stage',how='left')
DS.to_csv(OUT/'03_response_divergence_summary.csv',index=False,float_format='%.12g')

# ---------- exact common-flux identity between M1 and M2 ----------
idrows=[]
for cohort in ['primary','strict']:
    for stage in ['CF0','CFN']:
        a=M1[(M1.cohort==cohort)&(M1.stage==stage)]
        b=M2[(M2.cohort==cohort)&(M2.stage==stage)]
        z=a[['event_id','q_common_pre','q_common_post','delta_DV','delta_DN']].merge(
            b[['event_id','q_common_pre','q_common_post','delta_DV','delta_DN']],
            on='event_id',suffixes=('_M1','_M2'),validate='one_to_one'
        )
        idrows.append({
            'cohort':cohort,'stage':stage,'n':len(z),
            'max_q_pre_difference':float(np.max(np.abs(z.q_common_pre_M1-z.q_common_pre_M2))),
            'max_q_post_difference':float(np.max(np.abs(z.q_common_post_M1-z.q_common_post_M2))),
            'max_delta_DV_difference':float(np.max(np.abs(z.delta_DV_M1-z.delta_DV_M2))),
            'max_delta_DN_difference':float(np.max(np.abs(z.delta_DN_M1-z.delta_DN_M2)))
        })
ID=pd.DataFrame(idrows)
ID.to_csv(OUT/'04_common_flux_identity.csv',index=False,float_format='%.12g')

# ---------- D2 decision replay on 28 wind-dominated events ----------
COST=((ACTIONS-0.1)/0.8)**2
def benefit_table(df,eid):
    x=df[df.event_id==eid].sort_values('action')
    gt=float(x.T_gradient_C.iloc[0]); ga=float(x.AH_gradient_g_m3.iloc[0])
    T=x['T'].to_numpy(float); AH=x['AH'].to_numpy(float)
    bT=-np.sign(gt)*(T-T[0])/abs(gt)
    bA=-np.sign(ga)*(AH-AH[0])/abs(ga)
    return .5*(bT+bA),x
def breakpoints(Bx,lmax=1.0):
    v=[0.,float(lmax)]
    for i in range(len(ACTIONS)):
        for j in range(i+1,len(ACTIONS)):
            dc=COST[i]-COST[j]
            if abs(dc)>1e-15:
                z=(Bx[i]-Bx[j])/dc
                if 0<z<lmax:v.append(float(z))
    return sorted(set(round(x,14) for x in v))
def choose(Bx,l):
    q=Bx-l*COST
    mx=q.max()
    return int(np.where(q>=mx-1e-12)[0][0])
def decision_cf0():
    rows=[]
    for eid in sorted(G1.event_id.unique()):
        B1,x1=benefit_table(G1,eid)
        B2,x2=benefit_table(G2,eid)
        bp=sorted(set(breakpoints(B1)+breakpoints(B2)))
        dis=0.; gap=0.
        for lo,hi in zip(bp[:-1],bp[1:]):
            if hi<=lo:continue
            mid=(lo+hi)/2
            i1=choose(B1,mid); i2=choose(B2,mid); w=hi-lo
            if i1!=i2:dis+=w
            gap+=w*abs(ACTIONS[i1]-ACTIONS[i2])/0.8
        rows.append({
            'stage':'CF0','event_id':eid,'event_date':x1.event_date.iloc[0],
            'decision_disagreement_measure':dis,
            'normalized_action_gap_integral':gap,
            'any_disagreement':bool(dis>1e-12)
        })
    return pd.DataFrame(rows)
DCF=decision_cf0()
hv=R3DE[(R3DE.stage=='HV')&(R3DE.horizon_min==15)&(R3DE.lambda_max==1.0)&(R3DE.event_id.isin(DCF.event_id))].copy()
assert len(hv)==28
hv=hv[['event_id','event_date','decision_disagreement_measure','normalized_action_gap_integral','any_disagreement']].copy()
hv['stage']='HV'
DE=pd.concat([hv,DCF],ignore_index=True)
DE.to_csv(OUT/'05_d2_common_flux_decision_event.csv',index=False,float_format='%.12g')

ds=[]
for stage in ['HV','CF0']:
    q=DE[DE.stage==stage]
    dates=np.array(sorted(q.event_date.unique())); gd={d:q[q.event_date==d] for d in dates}
    rng=np.random.default_rng(SEED); vals=[]
    for _ in range(B):
        ss=rng.choice(dates,size=len(dates),replace=True)
        z=pd.concat([gd[d] for d in ss],ignore_index=True)
        vals.append(float(z.decision_disagreement_measure.mean()))
    ds.append({
        'stage':stage,'n_events':len(q),
        'mean_decision_disagreement':float(q.decision_disagreement_measure.mean()),
        'median_decision_disagreement':float(q.decision_disagreement_measure.median()),
        'fraction_any_disagreement':float(q.any_disagreement.mean()),
        'ci_low':float(np.percentile(vals,2.5)),'ci_high':float(np.percentile(vals,97.5))
    })
DSDEC=pd.DataFrame(ds)
DSDEC.to_csv(OUT/'06_d2_common_flux_decision_summary.csv',index=False,float_format='%.12g')

# ---------- machine summary ----------
hvrow=DS[DS.stage=='HV'].iloc[0]
cfrow=DS[DS.stage=='CF0'].iloc[0]
cfnrow=DS[DS.stage=='CFN'].iloc[0]
primary_identity=ID[(ID.cohort=='primary')&(ID.stage=='CF0')].iloc[0]
summary={
    'experiment':'PhysBench-GH R4 aerodynamic transport formulation isolation',
    'M1_gate':bool(S1['gate_pass']),'M2_gate':bool(S2['gate_pass']),
    'primary_events':40,'strict_events':14,'d2_events':28,
    'HV_mean_D_resp':float(hvrow.mean_D_resp),
    'CF0_mean_D_resp':float(cfrow.mean_D_resp),
    'CF0_minus_HV':float(cfrow.minus_HV_mean_D_resp),
    'CF0_minus_HV_CI95':[float(cfrow.ci_low),float(cfrow.ci_high)],
    'relative_D_resp_reduction':float((hvrow.mean_D_resp-cfrow.mean_D_resp)/hvrow.mean_D_resp),
    'HV_T_sign_disagreement':float(hvrow.T_sign_disagreement_fraction),
    'CF0_T_sign_disagreement':float(cfrow.T_sign_disagreement_fraction),
    'HV_AH_sign_disagreement':float(hvrow.AH_sign_disagreement_fraction),
    'CF0_AH_sign_disagreement':float(cfrow.AH_sign_disagreement_fraction),
    'CFN_mean_D_resp':float(cfnrow.mean_D_resp),
    'CFN_minus_HV':float(cfnrow.minus_HV_mean_D_resp),
    'primary_common_flux_identity':primary_identity.to_dict(),
    'D2_HV_mean':float(DSDEC[DSDEC.stage=='HV'].iloc[0].mean_decision_disagreement),
    'D2_CF0_mean':float(DSDEC[DSDEC.stage=='CF0'].iloc[0].mean_decision_disagreement),
    'locked_27_86_89_in_D2_subset':sorted(set([27,86,89]).intersection(set(DCF.event_id.astype(int)))),
    'scientific_result_is_not_runtime_gate':True
}
summary['gate_pass']=bool(
    S1['gate_pass'] and S2['gate_pass']
    and len(cf0)==40 and len(cfn)==40
    and primary_identity.max_q_pre_difference<=1e-12
    and primary_identity.max_q_post_difference<=1e-12
    and primary_identity.max_delta_DV_difference<=1e-10
    and primary_identity.max_delta_DN_difference<=1e-10
    and len(DCF)==28
)
(OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
if not summary['gate_pass']:
    raise SystemExit('R4 aggregate integrity gate failed')
