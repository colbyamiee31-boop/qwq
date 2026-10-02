import unittest, pandas as pd, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'physbench/exp2_1c/D2_PRIMARY_61_MODEL_INPUT.csv'
EXPECTED='8a5890dc9fb6950ce3c3e7ffc8ff900133ede8ffde99dfa68d79435314b56023'
class T(unittest.TestCase):
  @classmethod
  def setUpClass(cls): cls.df=pd.read_csv(P)
  def test_hash(self): self.assertEqual(hashlib.sha256(P.read_bytes()).hexdigest(),EXPECTED)
  def test_count(self): self.assertEqual(len(self.df),61); self.assertTrue(self.df.event_id.is_unique)
  def test_strata(self): self.assertEqual(self.df.daynight.value_counts().to_dict(),{'night':44,'day':17})
  def test_no_missing_primary(self): self.assertFalse(self.df[['event_Tair','event_AHin','event_Tout','event_AHout','event_Iglob','event_Windsp','CO2_pre_ppm','obs_matched_T_15','obs_matched_AH_15']].isna().any().any())
  def test_higher_status(self): self.assertTrue((self.df.status_delta>0).all())
if __name__=='__main__': unittest.main()
