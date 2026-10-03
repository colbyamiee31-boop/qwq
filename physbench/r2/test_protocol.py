import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'physbench/r2/PROTOCOL.md'; G=ROOT/'physbench/r2/generated'
txt=P.read_text()
required=['3–6 m s-1','G_L(delta)','G_W(delta)','7.375 m','date-cluster bootstrap','2000','20261004','New-coordinate firewall','Scientific agreement/disagreement is never a runtime pass criterion']
for s in required:
    if s not in txt: raise AssertionError(('protocol_missing',s))
p=pd.read_csv(G/'PRIMARY_97_R2_INPUT.csv'); s=pd.read_csv(G/'STRICT_36_R2_INPUT.csv')
if len(p)!=97 or len(s)!=36: raise AssertionError(('input_rows',len(p),len(s)))
if int(p.anchor_primary_3_6.sum())!=40 or int(s.anchor_primary_3_6.sum())!=14: raise AssertionError('anchor counts')
if int(p.anchor_expanded_2_8.sum())!=74: raise AssertionError('expanded count')
if p.event_id.nunique()!=97 or s.event_id.nunique()!=36: raise AssertionError('duplicate event')
cols=['emp_delta_DV_23_23','emp_delta_DN_23_23','emp_delta_DV_22_24','emp_delta_DV_24_22']
if not np.isfinite(pd.concat([p[cols],s[cols]]).to_numpy(float)).all(): raise AssertionError('nonfinite anchor')
if not np.all(np.sign(p.loc[p.anchor_primary_3_6,'emp_delta_DV_23_23'])==np.sign(p.loc[p.anchor_primary_3_6,'delta_u'])): raise AssertionError('anchor/action sign')
print(json.dumps({'protocol_gate':'PASS','primary_rows':len(p),'strict_rows':len(s),'primary_3_6':int(p.anchor_primary_3_6.sum()),'strict_3_6':int(s.anchor_primary_3_6.sum()),'expanded_2_8':int(p.anchor_expanded_2_8.sum())},indent=2))
