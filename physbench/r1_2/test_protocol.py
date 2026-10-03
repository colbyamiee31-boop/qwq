from pathlib import Path
import hashlib,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
P=ROOT/'physbench/r1_2/generated/PRIMARY_97_MODEL_INPUT.csv'
S=ROOT/'physbench/r1_2/generated/STRICT_36_MODEL_INPUT.csv'
assert sha(P)=='236e6631f9f60b7adfa942e496f1caea36e2eef4ddfc0149cebebe7524024051'
assert sha(S)=='f144244dda17c88f7e1f2ed1eecc62810fb7e4dce449602f231d41b5254b84f0'
p=pd.read_csv(P); s=pd.read_csv(S)
assert p.event_id.nunique()==97 and len(p)==97
assert s.event_id.nunique()==36 and len(s)==36
for x in (p,s):
    assert ((x.u_pre>=0)&(x.u_pre<=1)&(x.u_post>=0)&(x.u_post<=1)).all()
    assert (x.n_controls>=3).all()
    assert (x.delta_u.abs()>0).all()
print('R1.2 protocol/input tests PASS')
