import hashlib,json,py_compile
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'physbench/r3/PROTOCOL.md'; G=ROOT/'physbench/r3/generated'; D2=ROOT/'physbench/exp2_1c/D2_PRIMARY_61_MODEL_INPUT.csv'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
txt=P.read_text()
for s in ['7.375','0.17474999631859142','Stage H','Stage HV','2,000','20261004','Scientific improvement or deterioration is never a runtime pass criterion']:
    if s not in txt: raise AssertionError(('protocol_missing',s))
p=pd.read_csv(G/'PRIMARY_97_R2_INPUT.csv'); q=pd.read_csv(G/'STRICT_36_R2_INPUT.csv')
if len(p)!=97 or p.event_id.nunique()!=97: raise AssertionError('primary input')
if len(q)!=36 or q.event_id.nunique()!=36: raise AssertionError('strict input')
if int(p.anchor_primary_3_6.sum())!=40: raise AssertionError('anchor cohort')
if sha(D2)!='8a5890dc9fb6950ce3c3e7ffc8ff900133ede8ffde99dfa68d79435314b56023': raise AssertionError('D2 sha')
for f in ['common.py','m1_harmonised.py','m2_harmonised.py','aggregate.py']:
    py_compile.compile(str(ROOT/'physbench/r3'/f),doraise=True)
print(json.dumps({'protocol_gate':'PASS','primary_ctifl':97,'strict_ctifl':36,'anchor_3_6':40,'d2_events':61},indent=2))
