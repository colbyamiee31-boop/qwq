import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from common import q_common_pct

ROOT=Path(__file__).resolve().parents[2]
P97=ROOT/'physbench/r4/generated/PRIMARY_97_R2_INPUT.csv'
S36=ROOT/'physbench/r4/generated/STRICT_36_R2_INPUT.csv'
D2=ROOT/'physbench/exp2_1c/D2_PRIMARY_61_MODEL_INPUT.csv'
PROTOCOL=ROOT/'physbench/r4/PROTOCOL.md'

EXPECTED_P='236e6631f9f60b7adfa942e496f1caea36e2eef4ddfc0149cebebe7524024051'
EXPECTED_S='f144244dda17c88f7e1f2ed1eecc62810fb7e4dce449602f231d41b5254b84f0'
EXPECTED_D='8a5890dc9fb6950ce3c3e7ffc8ff900133ede8ffde99dfa68d79435314b56023'

def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

assert sha256(P97)==EXPECTED_P,(sha256(P97),EXPECTED_P)
assert sha256(S36)==EXPECTED_S,(sha256(S36),EXPECTED_S)
assert sha256(D2)==EXPECTED_D,(sha256(D2),EXPECTED_D)

p=pd.read_csv(P97); s=pd.read_csv(S36); d=pd.read_csv(D2)
assert len(p)==97 and p.event_id.nunique()==97
assert len(s)==36 and s.event_id.nunique()==36
assert len(d)==61 and d.event_id.nunique()==61
assert int(p.anchor_primary_3_6.sum())==40
assert int(s.anchor_primary_3_6.sum())==14
assert int(((d.event_Windsp>=3)&(d.event_Windsp<=6)).sum())==28

# frozen R2 empirical flux identity
for x in [p,s]:
    a=q_common_pct(x.leeward_pre_pct,x.windward_pre_pct,x.event_Windsp)
    b=q_common_pct(x.leeward_post_pct,x.windward_post_pct,x.event_Windsp)
    assert np.max(np.abs(a-x.emp_flux_pre_23_23.to_numpy(float)))<=2e-11
    assert np.max(np.abs(b-x.emp_flux_post_23_23.to_numpy(float)))<=2e-11

txt=PROTOCOL.read_text()
for token in [
    'CF0','CFN','3–6 m s-1','40 R1.2 matched CTIFL events',
    'q_common','2,000 resamples','No-tuning firewall',
    'Scientific improvement or deterioration is never a runtime PASS criterion'
]:
    assert token in txt,token

print(json.dumps({
    'protocol_gate':'PASS',
    'P97_sha256':sha256(P97),
    'S36_sha256':sha256(S36),
    'D2_sha256':sha256(D2),
    'primary_anchor_events':40,
    'strict_anchor_events':14,
    'D2_wind_dominated_events':28
},indent=2))
