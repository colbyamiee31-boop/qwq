import sys,zipfile,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
P=101325.0
PRIMARY_ROWS=[4368,6907,6936,6959,11817,12439,12453,12467,16239,18582,22302,25457,30454,31874,32972,33510,33798,33814,36163,36446,37992,38193,38205,38265,38280,38709,39145,39973,40256,40276,40429,40542,40982,41365,41387,41595,42796,42847,43000,43022,43457,43595,43700,44040,44859,45178,45947,46052,46193,46267,46308,46320,46340,46514,46625,47492,47695,47748,47775,48594,49188,49529,50053,50084,50377,50664,51227,51497,51540,51644,51656,51731,51769,51805,51828,52386,52398,52412,52688,52803,52822,52843,52939,52970,53521,53674,53712,53841,54125,54416,54579,54646,54708,54993,55038,55103,55157,55255,55274,55481,55576,55697,55736,55849,56008,56114,56145,56429,56546,56579,56721,57009,57296,57587,57874,58161,58282,58449,58739,58847,59213,59379,59588,59604,59746,59891,60031,60117,60200,60232,60296,60367,60547,60673,60698,60772,60887,61211,61327,61393,61632,61764,61876,61902,62208,62433,62445,62469,62687,62709,62745,62760,63014,63069,63362,63452,63501,63601,63613,63883,63897,63933,64088,64176,64224,64250,64269,64314,64400,64795,64882,64939,65090,65186,65367,65462,65668,65729,65751,65873,65976,65991,66013,66034,66163,66622,66768,66912,67048,67342,67623,67656,67793,67892,67939,67953,67967,68154,68181,68221,68403,68479,68508,68777,69061,69090,69230,69361,69387,69519,69652,69829,69854,70081,70102,70118,70268,70375,70430,70505,70522,70536,70668,70844,70937,71126,71377,71402,71415,71697,71710,71988,72409,72453,72615,72722,72920,72951,73112,73149,73170,73248,73273,73293,73358,73399,73549,73664,73931,73943,73976,74516,74545,74762,74827,74863,75159,75283,75452,75464,75720,75740,75803,75958,75973,75987,76004,76494,76514,76528,76545,76600,76830,76854,77073,77087,77291,77429,77462,77587,77832,78102,78121,78151,78164,78180,78199,78326,78562,78577,78593,78613,78862,79121,79477,79665,79677,79883,79998,80015,80298,80313,80574,80588,80784,80801,80918,81203,81445,81494,81614,81712,81728,81986,82001,82023,82036,82317,82475,82636,82677,82920,83139,83155,83182,83432,83464,83495,83753,84068,84291,84358,84877,84902,85073,85110,85140,85164,85189,85456,85480,85740,85774,85970,86033,86047,86061,86289,86316,86548,86596,86632,86871,86899,86919,87149,87164,87178,88317,88337,88350,88385,88533,88562,88595,88816,88905,89178,89471,89492,89749,89766,90015,90054,90324,90340,91760,92059,93313,93772,95515,95788,95830,96057,96119]
STRICT_ROWS=[33510,36163,38280,39145,42847,43460,45178,45947,46193,46320,50053,50084,50377,52390,52407,52688,52970,53841,54416,54708,54993,55038,55157,55849,56555,56721,57009,57587,57874,58449,58739,59588,59891,60232,60376,62208,62709,63069,63362,63902,65976,66037,66163,67656,67939,67967,68181,68486,70102,70522,70844,71991,72409,72615,73170,73399,73664,73976,75159,75452,75464,75740,75812,75998,76514,76528,76600,76854,77073,77092,77462,78326,80588,80784,80801,80918,82036,82636,83161,84071,84877,86052,86289,86871,88350,88385,88533,89492,89766,92063]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()
def satvp(t): return 610.78*np.exp(17.2694*np.asarray(t,float)/(np.asarray(t,float)+238.3))
def ahtrh(t,rh): return 216.7*((satvp(t)*np.asarray(rh,float)/100)/100)/(np.asarray(t,float)+273.15)
def vp_w(w): return np.asarray(w,float)*P/(0.62198+np.asarray(w,float))
def ahvp(t,vp): return 216.7*(np.asarray(vp,float)/100)/(np.asarray(t,float)+273.15)
def cdiff(a,b,p):
 d=np.abs(a-b)%p; return np.minimum(d,p-d)
def circ_range(v):
 v=np.sort(np.asarray(v,float)%360); g=np.diff(np.r_[v,v[0]+360]); return 360-g.max()
def main(zp,outdir):
 zp=Path(zp); out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
 assert zp.stat().st_size==125986215 and md5(zp)=='d0e4486fa1041fac5e6e47673b6c95d3'
 with zipfile.ZipFile(zp) as z:
  main=pd.read_csv(z.open('Dataset/1_Measurements_dataset/2_MainData.csv'),skiprows=2); weather=pd.read_csv(z.open('Dataset/1_Measurements_dataset/4_WeatherData.csv'),skiprows=2)
 df=main.merge(weather,on='Time'); assert len(df)==96192 and np.all(np.diff(df.Time)==300)
 t0=pd.Timestamp('2014-12-03'); df['dt']=t0+pd.to_timedelta((df.Time-df.Time.iloc[0]).astype(int),unit='s'); df['Tin_C']=df['Indoor process temperature']-273.15; df['AHin']=ahtrh(df.Tin_C,df['Indoor process relative humidity']); df['vent_mean']=(df['Leeward roof vents opening']+df['Windward roof vents opening'])/200; df['total_heat']=df[['Rails 51 injected heat','Forcas injected heat','PE injected heat']].sum(axis=1)
 n=len(df); bad=np.zeros(n,bool)
 for a,b in [('2015-10-02 09:50','2015-10-05 09:15'),('2015-02-26 23:20','2015-02-27 08:35'),('2015-03-01 18:35','2015-03-02 08:10'),('2015-01-07 13:30','2015-01-07 17:00'),('2015-10-25 02:00','2015-10-25 02:55')]: bad|=((df.dt>=pd.Timestamp(a))&(df.dt<=pd.Timestamp(b))).to_numpy()
 offs=np.arange(-3,7)
 def W(c):
  a=df[c].to_numpy(); M=np.full((n,10),np.nan)
  for j,k in enumerate(offs):
   if k<0:M[-k:,j]=a[:n+k]
   elif k>0:M[:-k,j]=a[k:]
   else:M[:,j]=a
  return M
 full=np.ones(n,bool);full[:3]=False;full[-6:]=False; BM=np.full((n,10),True)
 for j,k in enumerate(offs):
  if k<0:BM[-k:,j]=bad[:n+k]
  elif k>0:BM[:-k,j]=bad[k:]
  else:BM[:,j]=bad
 L,Wv=W('Leeward roof vents opening'),W('Windward roof vents opening'); vent=(np.nanmax(L,1)-np.nanmin(L,1)<=2)&(np.nanmax(Wv,1)-np.nanmin(Wv,1)<=2)
 Th,Sh=W('Thermal screen position'),W('Shading screen position'); screen=(np.nanmax(Th,1)==np.nanmin(Th,1))&(np.nanmax(Sh,1)==np.nanmin(Sh,1))&np.isin(df['Thermal screen position'],[0,100])&np.isin(df['Shading screen position'],[0,90,100])
 pump=np.ones(n,bool)
 for c in ['Rails 51 status','Forcas status','PE status']:
  X=W(c);pump&=(np.nanmax(X,1)==np.nanmin(X,1))
 H=W('total_heat'); heat=(np.nanmax(H,1)-np.nanmin(H,1)<=.005)&(np.nanmax(H,1)<=.020)
 To,Ho,Ws,G=W('Outdoor air temperature'),W('Outdoor humidity ratio'),W('Wind speed'),W('Global Horizontal Irradiance (measure)'); weatherok=(np.nanmax(To,1)-np.nanmin(To,1)<=2)&(np.nanmax(Ho,1)-np.nanmin(Ho,1)<=.0015)&(np.nanmax(Ws,1)-np.nanmin(Ws,1)<=2.5)&(np.nanmax(G,1)-np.nanmin(G,1)<=200)
 wd=df['Wind direction'].to_numpy(float);ws=df['Wind speed'].to_numpy(float);wdok=np.zeros(n,bool)
 for i in range(3,n-6): wdok[i]=np.median(ws[i-3:i+7])<1 or circ_range(wd[i-3:i+7])<=60
 near=np.zeros(n,bool)
 for i in PRIMARY_ROWS:near[max(0,i-12):min(n,i+13)]=True
 ci=np.flatnonzero(full&~BM.any(1)&vent&screen&pump&heat&weatherok&wdok&~near)
 T=df.Tin_C.to_numpy(float);AH=df.AHin.to_numpy(float);Tp=np.full(n,np.nan);Ap=np.full(n,np.nan);Tp[3:]=1.5*(T[2:-1]-T[:-3]);Ap[3:]=1.5*(AH[2:-1]-AH[:-3])
 cdt=df.loc[ci,'dt'];cday=(cdt.dt.normalize()-t0).dt.days.to_numpy();cclock=(cdt.dt.hour*60+cdt.dt.minute).to_numpy(float)
 def arr(c):return df.loc[ci,c].to_numpy()
 cv=df.loc[ci,'vent_mean'].to_numpy(float);cT=T[ci];cA=AH[ci];cTo=arr('Outdoor air temperature').astype(float);cHo=arr('Outdoor humidity ratio').astype(float);cG=arr('Global Horizontal Irradiance (measure)').astype(float);cWs=arr('Wind speed').astype(float);cWd=arr('Wind direction').astype(float);cth=arr('Thermal screen position');csh=arr('Shading screen position');cR=arr('Rails 51 status').astype(int);cF=arr('Forcas status').astype(int);cPE=arr('PE status').astype(int);cD=cG>=20;cTp=Tp[ci];cAp=Ap[ci]
 def match(rows,prefix):
  cmap={};meta={}
  for k,i in enumerate(rows,1):
   dt=df.at[i,'dt'];ed=(dt.normalize()-t0).days;ec=dt.hour*60+dt.minute;ev=float(df.at[i-1,'vent_mean']);eT=T[i];eA=AH[i];eTo=float(df.at[i,'Outdoor air temperature']);eHo=float(df.at[i,'Outdoor humidity ratio']);eG=float(df.at[i,'Global Horizontal Irradiance (measure)']);eWs=float(df.at[i,'Wind speed']);eWd=float(df.at[i,'Wind direction']);eth=float(df.at[i,'Thermal screen position']);esh=float(df.at[i,'Shading screen position']);eR=int(df.at[i,'Rails 51 status']);eF=int(df.at[i,'Forcas status']);ePE=int(df.at[i,'PE status']);eD=eG>=20
   dd=np.abs(cday-ed);clock=cdiff(cclock,ec,1440);wdd=cdiff(cWd,eWd,360);m=(cth==eth)&(csh==esh)&(cR==eR)&(cF==eF)&(cPE==ePE)&(cD==eD)&(dd<=30)&(clock<=120)&(np.abs(cv-ev)<=.10)&(np.abs(cT-eT)<=1.5)&(np.abs(cA-eA)<=1.5)&(np.abs(cTo-eTo)<=2)&(np.abs(cHo-eHo)<=.0015)&(np.abs(cWs-eWs)<=2)&(np.abs(cTp-Tp[i])<=.5)&(np.abs(cAp-Ap[i])<=.75)
   if eD:m&=np.abs(cG-eG)<=100
   both=(cWs>=1)&(eWs>=1);m&=(~both)|(wdd<=45);ids=np.flatnonzero(m)
   if len(ids):
    d=(dd[ids]/30)**2+(clock[ids]/120)**2+((cv[ids]-ev)/.1)**2+((cT[ids]-eT)/1.5)**2+((cA[ids]-eA)/1.5)**2+((cTo[ids]-eTo)/2)**2+((cHo[ids]-eHo)/.0015)**2+((cWs[ids]-eWs)/2)**2+((cTp[ids]-Tp[i])/.5)**2+((cAp[ids]-Ap[i])/.75)**2
    if eD:d+=((cG[ids]-eG)/100)**2
    d+=np.where(both[ids],(wdd[ids]/45)**2,0);o=np.argsort(d);cmap[f'CTIFL-E5-{prefix}-{k:04d}']=(ci[ids[o]],np.sqrt(d[o]));meta[f'CTIFL-E5-{prefix}-{k:04d}']=i
   else:cmap[f'CTIFL-E5-{prefix}-{k:04d}']=(np.array([],int),np.array([],float));meta[f'CTIFL-E5-{prefix}-{k:04d}']=i
  use={};assign={}
  for _,eid in sorted((len(v[0]),eid) for eid,v in cmap.items() if len(v[0])>=3):
   s=[]
   for j,d in zip(*cmap[eid]):
    if use.get(int(j),0)<5:s.append((int(j),float(d)))
    if len(s)==5:break
   if len(s)>=3:
    assign[eid]=s
    for j,_ in s:use[j]=use.get(j,0)+1
  return assign,meta
 def outputs(rows,prefix,expected,strict=False):
  A,M=match(rows,prefix);assert len(A)==expected,(prefix,len(A));obs=[];mi=[]
  for eid,S in A.items():
   i=M[eid];ctrl=[j for j,_ in S];dt=df.at[i,'dt'];Tin=float(T[i]);rh=float(df.at[i,'Indoor process relative humidity']);ivp=float(satvp(Tin)*rh/100);Tout=float(df.at[i,'Outdoor air temperature']);ovp=float(vp_w(df.at[i,'Outdoor humidity ratio']));AHout=float(ahvp(Tout,ovp))
   od={}
   for h,st in [(15,3),(30,6)]:
    eT=T[i+st]-T[i];eA=AH[i+st]-AH[i];cTc=np.array([T[j+st]-T[j] for j in ctrl]);cAc=np.array([AH[j+st]-AH[j] for j in ctrl]);od[h]=(float(eT-cTc.mean()),float(eA-cAc.mean()))
   mi.append({'event_id':eid,'row_index':i,'Time_s':int(df.at[i,'Time']),'timestamp':str(dt),'event_date':str(dt.date()),'month':dt.month,'day':dt.day,'day_of_year':dt.dayofyear,'hour_decimal':dt.hour+dt.minute/60,'daynight':'day' if float(df.at[i,'Global Horizontal Irradiance (measure)'])>=20 else 'night','direction':'opening' if df.at[i,'vent_mean']>df.at[i-1,'vent_mean'] else 'closing','u_pre':float(df.at[i-1,'vent_mean']),'u_post':float(df.at[i,'vent_mean']),'delta_u':float(df.at[i,'vent_mean']-df.at[i-1,'vent_mean']),'leeward_pre_pct':float(df.at[i-1,'Leeward roof vents opening']),'leeward_post_pct':float(df.at[i,'Leeward roof vents opening']),'windward_pre_pct':float(df.at[i-1,'Windward roof vents opening']),'windward_post_pct':float(df.at[i,'Windward roof vents opening']),'event_Tair':Tin,'event_RH':rh,'in_vp_pa':ivp,'CO2_pre_ppm':float(df.at[i,'Indoor CO2 concentration']),'canopy_temperature_c_closure':Tin,'event_Iglob':float(df.at[i,'Global Horizontal Irradiance (measure)']),'event_Tout':Tout,'out_vp_pa':ovp,'event_Windsp':float(df.at[i,'Wind speed']),'outdoor_co2_ppm_fixed':415.0,'sky_temperature_c_proxy':Tout,'soil_boundary_temperature_c_fixed':18.0,'T_gradient_C':Tin-Tout,'AH_gradient_g_m3':float(AH[i]-AHout),'obs_T_matched_15':od[15][0],'obs_AH_matched_15':od[15][1],'obs_T_matched_30':od[30][0],'obs_AH_matched_30':od[30][1],'n_controls':len(ctrl)})
  q=pd.DataFrame(mi).sort_values('event_id');fn='STRICT_36_MODEL_INPUT.csv' if strict else 'PRIMARY_97_MODEL_INPUT.csv';q.to_csv(out/fn,index=False,float_format='%.12g');return q,A
 qp,ap=outputs(PRIMARY_ROWS,'P',97,False);qs,as_=outputs(STRICT_ROWS,'S',36,True)
 assert sha(out/'PRIMARY_97_MODEL_INPUT.csv')=='236e6631f9f60b7adfa942e496f1caea36e2eef4ddfc0149cebebe7524024051'
 assert sha(out/'STRICT_36_MODEL_INPUT.csv')=='f144244dda17c88f7e1f2ed1eecc62810fb7e4dce449602f231d41b5254b84f0'
 print(json.dumps({'primary':len(qp),'strict':len(qs),'primary_sha':sha(out/'PRIMARY_97_MODEL_INPUT.csv'),'strict_sha':sha(out/'STRICT_36_MODEL_INPUT.csv')},indent=2))
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
