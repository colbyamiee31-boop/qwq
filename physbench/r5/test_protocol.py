import hashlib,json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
HANDOFF=ROOT/'physbench/r5'
P97=ROOT/'physbench/r5/generated/PRIMARY_97_R2_INPUT.csv'
EXPECTED={
    'field_flux.py':'b669a1209cc0e641b567bd5dbbc2f23c103ac3efbd91998a54f8a8a4939939cd',
    'field_ventilation_function.json':'4c7ae6aaab48097e522bb305b8353bae23ac59ffb0775508c077d6d1b130fc3b',
    'R5_event_list_97_v2.csv':'5721220e0ebb13750ae7b89950f7998e80836d3c52d0c8ee9309380ef82167b5',
    'PRIMARY_97_R2_INPUT.csv':'0127756518d8a481d8f79e27b0a0d57bd2b9e6d9d94232202109869f5a792d5b'
}
def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

actual={}
for name,expected in EXPECTED.items():
    p=P97 if name=='PRIMARY_97_R2_INPUT.csv' else HANDOFF/name
    actual[name]=sha256(p)
    assert actual[name]==expected,(name,actual[name],expected)

p=pd.read_csv(P97)
e=pd.read_csv(HANDOFF/'R5_event_list_97_v2.csv')
required={'row_index','event_id','in_R4_wind_3_6_subset'}
assert required.issubset(e.columns),sorted(required-set(e.columns))
assert len(p)==97 and p.event_id.nunique()==97 and p.row_index.nunique()==97
assert len(e)==97 and e.event_id.nunique()==97 and e.row_index.nunique()==97
j=e.merge(p[['row_index','event_id','event_Windsp','anchor_primary_3_6']],on='row_index',
          suffixes=('_list','_input'),how='outer',indicator=True,validate='one_to_one')
assert len(j)==97 and (j['_merge']=='both').all()
assert (j.event_id_list.astype(str)==j.event_id_input.astype(str)).all()
assert int(p.anchor_primary_3_6.astype(bool).sum())==40
assert int(e.in_R4_wind_3_6_subset.astype(bool).sum())==40
assert (j.in_R4_wind_3_6_subset.astype(bool)==j.anchor_primary_3_6.astype(bool)).all()
assert set(j.loc[j.anchor_primary_3_6.astype(bool),'event_id_input'].astype(str))==set(
       j.loc[j.in_R4_wind_3_6_subset.astype(bool),'event_id_list'].astype(str))
print(json.dumps({'protocol_input_gate':'PASS','sha256':actual,
                  'event_rows':len(j),'row_index_event_id_matches':97,
                  'anchor_primary_3_6_events':40},indent=2))
