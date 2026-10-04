import sys, zipfile, hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd

P = 101325.0
PRIMARY_ROWS = [4368,6907,6936,6959,11817,12439,12453,12467,16239,18582,22302,25457,30454,31874,32972,33510,33798,33814,36163,36446,37992,38193,38205,38265,38280,38709,39145,39973,40256,40276,40429,40542,40982,41365,41387,41595,42796,42847,43000,43022,43457,43595,43700,44040,44859,45178,45947,46052,46193,46267,46308,46320,46340,46514,46625,47492,47695,47748,47775,48594,49188,49529,50053,50084,50377,50664,51227,51497,51540,51644,51656,51731,51769,51805,51828,52386,52398,52412,52688,52803,52822,52843,52939,52970,53521,53674,53712,53841,54125,54416,54579,54646,54708,54993,55038,55103,55157,55255,55274,55481,55576,55697,55736,55849,56008,56114,56145,56429,56546,56579,56721,57009,57296,57587,57874,58161,58282,58449,58739,58847,59213,59379,59588,59604,59746,59891,60031,60117,60200,60232,60296,60367,60547,60673,60698,60772,60887,61211,61327,61393,61632,61764,61876,61902,62208,62433,62445,62469,62687,62709,62745,62760,63014,63069,63362,63452,63501,63601,63613,63883,63897,63933,64088,64176,64224,64250,64269,64314,64400,64795,64882,64939,65090,65186,65367,65462,65668,65729,65751,65873,65976,65991,66013,66034,66163,66622,66768,66912,67048,67342,67623,67656,67793,67892,67939,67953,67967,68154,68181,68221,68403,68479,68508,68777,69061,69090,69230,69361,69387,69519,69652,69829,69854,70081,70102,70118,70268,70375,70430,70505,70522,70536,70668,70844,70937,71126,71377,71402,71415,71697,71710,71988,72409,72453,72615,72722,72920,72951,73112,73149,73170,73248,73273,73293,73358,73399,73549,73664,73931,73943,73976,74516,74545,74762,74827,74863,75159,75283,75452,75464,75720,75740,75803,75958,75973,75987,76004,76494,76514,76528,76545,76600,76830,76854,77073,77087,77291,77429,77462,77587,77832,78102,78121,78151,78164,78180,78199,78326,78562,78577,78593,78613,78862,79121,79477,79665,79677,79883,79998,80015,80298,80313,80574,80588,80784,80801,80918,81203,81445,81494,81614,81712,81728,81986,82001,82023,82036,82317,82475,82636,82677,82920,83139,83155,83182,83432,83464,83495,83753,84068,84291,84358,84877,84902,85073,85110,85140,85164,85189,85456,85480,85740,85774,85970,86033,86047,86061,86289,86316,86548,86596,86632,86871,86899,86919,87149,87164,87178,88317,88337,88350,88385,88533,88562,88595,88816,88905,89178,89471,89492,89749,89766,90015,90054,90324,90340,91760,92059,93313,93772,95515,95788,95830,96057,96119]
STRICT_ROWS = [33510,36163,38280,39145,42847,43460,45178,45947,46193,46320,50053,50084,50377,52390,52407,52688,52970,53841,54416,54708,54993,55038,55157,55849,56555,56721,57009,57587,57874,58449,58739,59588,59891,60232,60376,62208,62709,63069,63362,63902,65976,66037,66163,67656,67939,67967,68181,68486,70102,70522,70844,71991,72409,72615,73170,73399,73664,73976,75159,75452,75464,75740,75812,75998,76514,76528,76600,76854,77073,77092,77462,78326,80588,80784,80801,80918,82036,82636,83161,84071,84877,86052,86289,86871,88350,88385,88533,89492,89766,92063]

def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()
def satvp(t):
    a=np.asarray(t,float); return 610.78*np.exp(17.2694*a/(a+238.3))
def ahtrh(t,rh):
    t=np.asarray(t,float); rh=np.asarray(rh,float)
    return 216.7*((satvp(t)*rh/100)/100)/(t+273.15)
def cdiff(a,b,p):
    d=np.abs(a-b)%p; return np.minimum(d,p-d)
def circ_range(v):
    v=np.sort(np.asarray(v,float)%360)
    g=np.diff(np.r_[v,v[0]+360])
    return 360-g.max()

def load_ctifl(zp):
    zp=Path(zp)
    assert zp.stat().st_size==125986215
    assert md5(zp)=='d0e4486fa1041fac5e6e47673b6c95d3'
    with zipfile.ZipFile(zp) as z:
        main=pd.read_csv(z.open('Dataset/1_Measurements_dataset/2_MainData.csv'),skiprows=2)
        weather=pd.read_csv(z.open('Dataset/1_Measurements_dataset/4_WeatherData.csv'),skiprows=2)
    df=main.merge(weather,on='Time')
    assert len(df)==96192 and np.all(np.diff(df.Time)==300)
    t0=pd.Timestamp('2014-12-03')
    df['dt']=t0+pd.to_timedelta((df.Time-df.Time.iloc[0]).astype(int),unit='s')
    df['Tin_C']=df['Indoor process temperature']-273.15
    df['AHin']=ahtrh(df.Tin_C,df['Indoor process relative humidity'])
    df['vent_mean']=(df['Leeward roof vents opening']+df['Windward roof vents opening'])/200
    df['total_heat']=df[['Rails 51 injected heat','Forcas injected heat','PE injected heat']].sum(axis=1)
    return df,t0

def build_control_pool(df,t0):
    n=len(df)
    bad=np.zeros(n,bool)
    for a,b in [('2015-10-02 09:50','2015-10-05 09:15'),('2015-02-26 23:20','2015-02-27 08:35'),('2015-03-01 18:35','2015-03-02 08:10'),('2015-01-07 13:30','2015-01-07 17:00'),('2015-10-25 02:00','2015-10-25 02:55')]:
        bad|=((df.dt>=pd.Timestamp(a))&(df.dt<=pd.Timestamp(b))).to_numpy()
    offs=np.arange(-3,7)
    def W(c):
        a=df[c].to_numpy(); M=np.full((n,10),np.nan)
        for j,k in enumerate(offs):
            if k<0:M[-k:,j]=a[:n+k]
            elif k>0:M[:-k,j]=a[k:]
            else:M[:,j]=a
        return M
    full=np.ones(n,bool); full[:3]=False; full[-6:]=False
    BM=np.full((n,10),True)
    for j,k in enumerate(offs):
        if k<0:BM[-k:,j]=bad[:n+k]
        elif k>0:BM[:-k,j]=bad[k:]
        else:BM[:,j]=bad
    L,Wv=W('Leeward roof vents opening'),W('Windward roof vents opening')
    vent=(np.nanmax(L,1)-np.nanmin(L,1)<=2)&(np.nanmax(Wv,1)-np.nanmin(Wv,1)<=2)
    Th,Sh=W('Thermal screen position'),W('Shading screen position')
    screen=(np.nanmax(Th,1)==np.nanmin(Th,1))&(np.nanmax(Sh,1)==np.nanmin(Sh,1))&np.isin(df['Thermal screen position'],[0,100])&np.isin(df['Shading screen position'],[0,90,100])
    pump=np.ones(n,bool)
    for c in ['Rails 51 status','Forcas status','PE status']:
        X=W(c); pump&=(np.nanmax(X,1)==np.nanmin(X,1))
    H=W('total_heat'); heat=(np.nanmax(H,1)-np.nanmin(H,1)<=.005)&(np.nanmax(H,1)<=.020)
    To,Ho,Ws,G=W('Outdoor air temperature'),W('Outdoor humidity ratio'),W('Wind speed'),W('Global Horizontal Irradiance (measure)')
    weatherok=(np.nanmax(To,1)-np.nanmin(To,1)<=2)&(np.nanmax(Ho,1)-np.nanmin(Ho,1)<=.0015)&(np.nanmax(Ws,1)-np.nanmin(Ws,1)<=2.5)&(np.nanmax(G,1)-np.nanmin(G,1)<=200)
    wd=df['Wind direction'].to_numpy(float); ws=df['Wind speed'].to_numpy(float)
    wdok=np.zeros(n,bool)
    for i in range(3,n-6):
        wdok[i]=np.median(ws[i-3:i+7])<1 or circ_range(wd[i-3:i+7])<=60
    near=np.zeros(n,bool)
    for i in PRIMARY_ROWS: near[max(0,i-12):min(n,i+13)]=True
    ci=np.flatnonzero(full&~BM.any(1)&vent&screen&pump&heat&weatherok&wdok&~near)
    T=df.Tin_C.to_numpy(float); AH=df.AHin.to_numpy(float)
    Tp=np.full(n,np.nan); Ap=np.full(n,np.nan)
    Tp[3:]=1.5*(T[2:-1]-T[:-3]); Ap[3:]=1.5*(AH[2:-1]-AH[:-3])
    cdt=df.loc[ci,'dt']; cday=(cdt.dt.normalize()-t0).dt.days.to_numpy(); cclock=(cdt.dt.hour*60+cdt.dt.minute).to_numpy(float)
    def arr(c): return df.loc[ci,c].to_numpy()
    C={
      'ci':ci,'T':T,'AH':AH,'Tp':Tp,'Ap':Ap,'cday':cday,'cclock':cclock,
      'cv':df.loc[ci,'vent_mean'].to_numpy(float),'cT':T[ci],'cA':AH[ci],
      'cTo':arr('Outdoor air temperature').astype(float),'cHo':arr('Outdoor humidity ratio').astype(float),
      'cG':arr('Global Horizontal Irradiance (measure)').astype(float),'cWs':arr('Wind speed').astype(float),
      'cWd':arr('Wind direction').astype(float),'cth':arr('Thermal screen position'),'csh':arr('Shading screen position'),
      'cR':arr('Rails 51 status').astype(int),'cF':arr('Forcas status').astype(int),'cPE':arr('PE status').astype(int)
    }
    C['cD']=C['cG']>=20
    return C

STAGES=['exact_strata','calendar_30d','clock_120m','vent_pre_0p10','indoor_T_1p5C','indoor_AH_1p5','outdoor_T_2C','outdoor_humratio_0p0015','wind_speed_2','pretrend_T_0p5','pretrend_AH_0p75','day_GHI_100','wind_direction_45']

def event_candidates(df,t0,C,rows,prefix):
    out=[]; cands={}
    ci=C['ci']; T=C['T']; AH=C['AH']; Tp=C['Tp']; Ap=C['Ap']
    for k,i in enumerate(rows,1):
        eid=f'CTIFL-E5-{prefix}-{k:04d}'
        dt=df.at[i,'dt']; ed=(dt.normalize()-t0).days; ec=dt.hour*60+dt.minute
        ev=float(df.at[i-1,'vent_mean']); eT=T[i]; eA=AH[i]
        eTo=float(df.at[i,'Outdoor air temperature']); eHo=float(df.at[i,'Outdoor humidity ratio'])
        eG=float(df.at[i,'Global Horizontal Irradiance (measure)']); eWs=float(df.at[i,'Wind speed']); eWd=float(df.at[i,'Wind direction'])
        eth=float(df.at[i,'Thermal screen position']); esh=float(df.at[i,'Shading screen position'])
        eR=int(df.at[i,'Rails 51 status']); eF=int(df.at[i,'Forcas status']); ePE=int(df.at[i,'PE status']); eD=eG>=20
        dd=np.abs(C['cday']-ed); clock=cdiff(C['cclock'],ec,1440); wdd=cdiff(C['cWd'],eWd,360)
        masks=[]
        m=(C['cth']==eth)&(C['csh']==esh)&(C['cR']==eR)&(C['cF']==eF)&(C['cPE']==ePE)&(C['cD']==eD); masks.append(m.copy())
        m=m&(dd<=30); masks.append(m.copy())
        m=m&(clock<=120); masks.append(m.copy())
        m=m&(np.abs(C['cv']-ev)<=.10); masks.append(m.copy())
        m=m&(np.abs(C['cT']-eT)<=1.5); masks.append(m.copy())
        m=m&(np.abs(C['cA']-eA)<=1.5); masks.append(m.copy())
        m=m&(np.abs(C['cTo']-eTo)<=2); masks.append(m.copy())
        m=m&(np.abs(C['cHo']-eHo)<=.0015); masks.append(m.copy())
        m=m&(np.abs(C['cWs']-eWs)<=2); masks.append(m.copy())
        m=m&(np.abs(C['cTp']-Tp[i])<=.5); masks.append(m.copy())
        m=m&(np.abs(C['cAp']-Ap[i])<=.75); masks.append(m.copy())
        if eD: m=m&(np.abs(C['cG']-eG)<=100)
        masks.append(m.copy())
        both=(C['cWs']>=1)&(eWs>=1)
        m=m&((~both)|(wdd<=45)); masks.append(m.copy())
        ids=np.flatnonzero(m)
        if len(ids):
            d=(dd[ids]/30)**2+(clock[ids]/120)**2+((C['cv'][ids]-ev)/.1)**2+((C['cT'][ids]-eT)/1.5)**2+((C['cA'][ids]-eA)/1.5)**2+((C['cTo'][ids]-eTo)/2)**2+((C['cHo'][ids]-eHo)/.0015)**2+((C['cWs'][ids]-eWs)/2)**2+((C['cTp'][ids]-Tp[i])/.5)**2+((C['cAp'][ids]-Ap[i])/.75)**2
            if eD:d+=((C['cG'][ids]-eG)/100)**2
            d+=np.where(both[ids],(wdd[ids]/45)**2,0)
            o=np.argsort(d); cand_idx=ci[ids[o]]; cand_dist=np.sqrt(d[o])
        else:
            cand_idx=np.array([],int); cand_dist=np.array([],float)
        cands[eid]=(cand_idx,cand_dist)
        counts=[int(x.sum()) for x in masks]
        row={
          'event_id':eid,'row_index':i,'timestamp':str(dt),'direction':'opening' if df.at[i,'vent_mean']>df.at[i-1,'vent_mean'] else 'closing',
          'abs_delta_u':abs(float(df.at[i,'vent_mean']-df.at[i-1,'vent_mean'])),'u_pre':ev,
          'event_Tair_C':float(eT),'event_RH_pct':float(df.at[i,'Indoor process relative humidity']),'event_AH_g_m3':float(eA),
          'event_Tout_C':eTo,'event_out_humidity_ratio':eHo,'event_GHI_W_m2':eG,'daynight':'day' if eD else 'night',
          'event_wind_speed_m_s':eWs,'event_wind_direction_deg':eWd,'pretrend_T_C_per_15m':float(Tp[i]),'pretrend_AH_g_m3_per_15m':float(Ap[i]),
          'thermal_screen_pct':eth,'shading_screen_pct':esh,'pump_tuple':f'{eR}-{eF}-{ePE}',
          'total_heat_kWh_m2_5min':float(df.at[i,'total_heat']),'month':int(dt.month),'hour_decimal':float(dt.hour+dt.minute/60),
          'wind_dir_sin':float(np.sin(np.deg2rad(eWd))),'wind_dir_cos':float(np.cos(np.deg2rad(eWd))),
          'clock_sin':float(np.sin(2*np.pi*(dt.hour+dt.minute/60)/24)),'clock_cos':float(np.cos(2*np.pi*(dt.hour+dt.minute/60)/24)),
          'exact_controls':counts[0],'final_controls_pre_reuse':len(cand_idx)
        }
        for s,c in zip(STAGES,counts): row[f'n_after_{s}']=c
        first='none'
        prev=10**9
        for s,c in zip(STAGES,counts):
            if prev>=3 and c<3:
                first=s; break
            prev=c
        row['first_below3_stage']=first
        out.append(row)
    return pd.DataFrame(out),cands

def assign_with_reuse(event_df,cands):
    use={}; assign={}
    order=sorted((len(v[0]),eid) for eid,v in cands.items() if len(v[0])>=3)
    for _,eid in order:
        s=[]
        for j,d in zip(*cands[eid]):
            if use.get(int(j),0)<5:s.append((int(j),float(d)))
            if len(s)==5:break
        if len(s)>=3:
            assign[eid]=s
            for j,_ in s: use[j]=use.get(j,0)+1
    df=event_df.copy()
    df['assigned_controls']=df.event_id.map(lambda x:len(assign.get(x,[]))).astype(int)
    df['matched']=df.assigned_controls>=3
    def cat(r):
        if r['exact_controls']<3:return 'exact_strata_scarcity'
        if r['final_controls_pre_reuse']<3:return 'caliper_scarcity'
        if r['assigned_controls']<3:return 'reuse_cap_competition'
        return 'matched'
    df['final_category']=df.apply(cat,axis=1)
    return df,assign,use

def pooled_smd(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    a=a[np.isfinite(a)]; b=b[np.isfinite(b)]
    if len(a)<2 or len(b)<2:return np.nan
    sp=np.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2))
    return (np.mean(a)-np.mean(b))/sp if sp>0 else np.nan

def robust_shift(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    a=a[np.isfinite(a)]; b=b[np.isfinite(b)]
    if not len(a) or not len(b):return np.nan
    iqra=np.percentile(a,75)-np.percentile(a,25); iqrb=np.percentile(b,75)-np.percentile(b,25)
    piqr=(iqra+iqrb)/2
    return (np.median(a)-np.median(b))/piqr if piqr>0 else np.nan

def balance_tables(df):
    matched=df[df.matched]; unmatched=df[~df.matched]
    vars_=['abs_delta_u','u_pre','event_Tair_C','event_RH_pct','event_AH_g_m3','event_Tout_C','event_out_humidity_ratio','event_GHI_W_m2','event_wind_speed_m_s','pretrend_T_C_per_15m','pretrend_AH_g_m3_per_15m','total_heat_kWh_m2_5min','wind_dir_sin','wind_dir_cos','clock_sin','clock_cos']
    rows=[]
    for v in vars_:
        a=matched[v].to_numpy(float); b=unmatched[v].to_numpy(float); smd=pooled_smd(a,b)
        flag='small' if np.isfinite(smd) and abs(smd)<.10 else ('modest' if np.isfinite(smd) and abs(smd)<.20 else ('material' if np.isfinite(smd) else 'undefined'))
        rows.append({'variable':v,'matched_n':np.isfinite(a).sum(),'unmatched_n':np.isfinite(b).sum(),'matched_mean':np.nanmean(a),'unmatched_mean':np.nanmean(b),'matched_median':np.nanmedian(a),'unmatched_median':np.nanmedian(b),'matched_sd':np.nanstd(a,ddof=1),'unmatched_sd':np.nanstd(b,ddof=1),'smd_matched_minus_unmatched':smd,'robust_median_shift_over_pooled_iqr':robust_shift(a,b),'flag':flag})
    cont=pd.DataFrame(rows)
    cats=[]
    for v in ['direction','daynight','thermal_screen_pct','shading_screen_pct','pump_tuple','month']:
        levels=sorted(set(df[v].astype(str)))
        for level in levels:
            pm=(matched[v].astype(str)==level).mean(); pu=(unmatched[v].astype(str)==level).mean()
            cats.append({'variable':v,'level':level,'matched_prop':pm,'unmatched_prop':pu,'difference_pp':100*(pm-pu),'abs_difference_pp':100*abs(pm-pu)})
    return cont,pd.DataFrame(cats)

def stage_summary(df,label):
    rows=[]
    prev_col=None
    for s in STAGES:
        col=f'n_after_{s}'
        vals=df[col]
        crossed=0 if prev_col is None else int(((df[prev_col]>=3)&(df[col]<3)).sum())
        rows.append({'cohort':label,'stage':s,'events_total':len(df),'events_with_3plus_controls':int((vals>=3).sum()),'median_controls':float(vals.median()),'mean_controls':float(vals.mean()),'new_events_falling_below_3_at_stage':crossed})
        prev_col=col
    return pd.DataFrame(rows)

def write_report(out,primary,strict,cont,cat,stagep,stages):
    fc=primary.final_category.value_counts().to_dict()
    first=primary.loc[~primary.matched,'first_below3_stage'].value_counts().to_dict()
    topc=cont.assign(abs_smd=cont.smd_matched_minus_unmatched.abs()).sort_values('abs_smd',ascending=False).head(8)
    topcat=cat.sort_values('abs_difference_pp',ascending=False).head(8)
    lines=['# SPLIT-1.1 R1.2 ATTRITION / REPRESENTATIVENESS AUDIT','',
      '## Integrity reproduction',
      f'- Primary frozen match count: **{int(primary.matched.sum())}/{len(primary)} ({100*primary.matched.mean():.1f}%)**.',
      f'- Strict frozen match count: **{int(strict.matched.sum())}/{len(strict)} ({100*strict.matched.mean():.1f}%)**.',
      '- These counts must equal 97/385 and 36/90 respectively; the script fails otherwise.','',
      '## Primary unmatched-event categories']
    for k,v in fc.items(): lines.append(f'- {k}: **{v}**')
    lines += ['','## First frozen stage at which candidate support fell below three controls']
    for k,v in first.items(): lines.append(f'- {k}: **{v}**')
    lines += ['','## Largest continuous matched-versus-unmatched descriptive imbalances']
    for _,r in topc.iterrows():
        lines.append(f"- {r.variable}: SMD {r.smd_matched_minus_unmatched:+.3f} ({r.flag}); matched mean {r.matched_mean:.4g}, unmatched mean {r.unmatched_mean:.4g}.")
    lines += ['','## Largest categorical proportion differences']
    for _,r in topcat.iterrows():
        lines.append(f"- {r.variable}={r.level}: matched {100*r.matched_prop:.1f}% vs unmatched {100*r.unmatched_prop:.1f}% (difference {r.difference_pp:+.1f} pp).")
    mat=(cont.flag=='material').sum()
    lines += ['','## Interpretation lock',
      f'- Material continuous descriptive imbalances (|SMD| >= 0.20): **{int(mat)}** of {len(cont)} audited continuous features.',
      '- This is an outcome-blind selection/representativeness diagnostic, not a new efficacy analysis.',
      '- Matching failure is expected to be structured because exact strata and tight calipers deliberately restrict the control-support region.',
      '- The primary R1.2 inference therefore applies to the matchable 97-event subset. Claims of representativeness beyond the audited variables are not made.',
      '- No R1.2 outcome, model response or downstream R2-R4 result was used in this audit.'
    ]
    (out/'SPLIT_1_1_ATTRITION_AUDIT_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')

def run(zp,outdir):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    df,t0=load_ctifl(zp); C=build_control_pool(df,t0)
    p0,pc=event_candidates(df,t0,C,PRIMARY_ROWS,'P')
    primary,pa,pu=assign_with_reuse(p0,pc)
    s0,sc=event_candidates(df,t0,C,STRICT_ROWS,'S')
    strict,sa,su=assign_with_reuse(s0,sc)
    assert int(primary.matched.sum())==97, int(primary.matched.sum())
    assert int(strict.matched.sum())==36, int(strict.matched.sum())
    cont,cat=balance_tables(primary)
    stagep=stage_summary(primary,'primary_385')
    stages=stage_summary(strict,'strict_90')
    primary.to_csv(out/'SPLIT_1_1_EVENT_LEVEL_MATCHABILITY_PRIMARY_385.csv',index=False,float_format='%.12g')
    strict.to_csv(out/'SPLIT_1_1_EVENT_LEVEL_MATCHABILITY_STRICT_90.csv',index=False,float_format='%.12g')
    cont.to_csv(out/'SPLIT_1_1_REPRESENTATIVENESS_CONTINUOUS.csv',index=False,float_format='%.12g')
    cat.to_csv(out/'SPLIT_1_1_REPRESENTATIVENESS_CATEGORICAL.csv',index=False,float_format='%.12g')
    pd.concat([stagep,stages],ignore_index=True).to_csv(out/'SPLIT_1_1_SEQUENTIAL_MATCH_SUPPORT.csv',index=False,float_format='%.12g')
    summary=pd.DataFrame([
      {'cohort':'primary_385','events':len(primary),'matched':int(primary.matched.sum()),'unmatched':int((~primary.matched).sum()),'matched_pct':100*primary.matched.mean()},
      {'cohort':'strict_90','events':len(strict),'matched':int(strict.matched.sum()),'unmatched':int((~strict.matched).sum()),'matched_pct':100*strict.matched.mean()}
    ])
    summary.to_csv(out/'SPLIT_1_1_ATTRITION_SUMMARY.csv',index=False,float_format='%.12g')
    failure=pd.concat([
      primary.final_category.value_counts().rename_axis('category').reset_index(name='n').assign(cohort='primary_385'),
      strict.final_category.value_counts().rename_axis('category').reset_index(name='n').assign(cohort='strict_90')
    ],ignore_index=True)
    failure['pct']=100*failure['n']/failure['cohort'].map({'primary_385':385,'strict_90':90})
    failure.to_csv(out/'SPLIT_1_1_MATCH_FAILURE_BREAKDOWN.csv',index=False,float_format='%.12g')
    write_report(out,primary,strict,cont,cat,stagep,stages)
    meta={'ctifl_md5':md5(zp),'primary_matched':int(primary.matched.sum()),'strict_matched':int(strict.matched.sum()),'control_pool_n':int(len(C['ci']))}
    (out/'SPLIT_1_1_AUDIT_METADATA.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2))

if __name__=='__main__':
    run(sys.argv[1],sys.argv[2])
