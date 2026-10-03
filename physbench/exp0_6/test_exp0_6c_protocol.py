import json, unittest
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

class TestEXP06CProtocol(unittest.TestCase):
    def test_history_subset(self):
        cfg=json.loads((HERE/'latent_history_config.json').read_text())
        self.assertEqual(cfg['decision_primary_history_ids'],
            ['H00','H01','H04','H06','H07','H10','H11','H13','H16'])

    def test_event_input_lock(self):
        import hashlib
        p=ROOT/'physbench/exp2_1c/D2_PRIMARY_61_MODEL_INPUT.csv'
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        self.assertEqual(h,'8a5890dc9fb6950ce3c3e7ffc8ff900133ede8ffde99dfa68d79435314b56023')
        df=pd.read_csv(p)
        self.assertEqual(len(df),61)
        self.assertEqual(df.event_id.nunique(),61)
        self.assertEqual(set(df.daynight),{'day','night'})
        self.assertFalse((df.T_gradient_C==0).any())
        self.assertFalse((df.AH_gradient_g_m3==0).any())

    def test_locked_reference_values_declared(self):
        txt=(HERE/'EXP0_6C_EXECUTION_LOCK.md').read_text()
        for token in [
            '27, 86, 89','0.024552427907011506','0.006138106976752877',
            '0.001645164794050128','0.01691271675501554',
            '0.1, 0.3, 0.5, 0.7, 0.9'
        ]:
            self.assertIn(token,txt)

    def test_scripts_keep_locked_ah_representation(self):
        for name in ['m1_event_decision_latent.py','m2_event_decision_latent.py']:
            txt=(HERE/name).read_text()
            self.assertIn('216.7*np.asarray(vp,dtype=float)',txt)
            self.assertIn("DURATION_H=int(CFG['primary_prehistory_h'])",txt)

if __name__=='__main__':
    unittest.main()
