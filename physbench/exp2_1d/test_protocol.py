import unittest, json, hashlib
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'physbench/exp2_1c/D2_PRIMARY_61_MODEL_INPUT.csv'; MED=ROOT/'physbench/exp2_1d/PRE15_FORCING_MEDIANS.csv'; CFG=ROOT/'physbench/exp2_1d/scenario_config.json'
EXPECTED='8a5890dc9fb6950ce3c3e7ffc8ff900133ede8ffde99dfa68d79435314b56023'; MEDSHA='177f6a9009bb60680fc1ccad438cbbcd233064a6f569caf48e04605f8914b578'
class T(unittest.TestCase):
 @classmethod
 def setUpClass(cls): cls.d=pd.read_csv(P); cls.m=pd.read_csv(MED); cls.c=json.loads(CFG.read_text())
 def test_hash(self): self.assertEqual(hashlib.sha256(P.read_bytes()).hexdigest(),EXPECTED)
 def test_cohort(self): self.assertEqual(len(self.d),61); self.assertEqual(self.d.daynight.value_counts().to_dict(),{'night':44,'day':17}); self.assertTrue(self.d.event_id.is_unique)
 def test_median_hash(self): self.assertEqual(hashlib.sha256(MED.read_bytes()).hexdigest(),MEDSHA)
 def test_pre15_medians(self): self.assertEqual(len(self.m),61); self.assertTrue(self.m.event_id.is_unique); self.assertTrue((self.m.pre15_weather_points==4).all()); self.assertEqual(set(self.m.event_id),set(self.d.event_id)); self.assertFalse(self.m.isna().any().any())
 def test_scenarios(self): self.assertEqual(len(self.c['mapping_scenarios'])*len(self.c['canopy_offsets_c'])*len(self.c['forcing_representations']),24)
 def test_mapping_bounds(self):
  for _,r in self.d.iterrows():
   for m in self.c['mapping_scenarios']:
    if m['kind']=='fixed': lo,hi=m['low'],m['high']
    elif m['kind']=='absolute_linear': lo,hi=r.status_before/100,r.status_after/100
    elif m['kind']=='absolute_compressed': lo,hi=m['offset']+m['scale']*r.status_before/100,m['offset']+m['scale']*r.status_after/100
    else: lo,hi=m['baseline'],min(1,m['baseline']+r.status_delta/100)
    self.assertTrue(0<=lo<hi<=1)
if __name__=='__main__': unittest.main()
