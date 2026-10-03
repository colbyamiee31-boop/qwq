import hashlib, json, unittest
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

class TestEXP34AProtocol(unittest.TestCase):
    def test_input_lock(self):
        p=ROOT/'physbench/exp2_1c/D2_PRIMARY_61_MODEL_INPUT.csv'
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),
            '8a5890dc9fb6950ce3c3e7ffc8ff900133ede8ffde99dfa68d79435314b56023')
        df=pd.read_csv(p)
        self.assertEqual(len(df),61)
        self.assertEqual(df.event_id.nunique(),61)

    def test_source_lock(self):
        txt=(HERE/'EXP3_4A_PHYSICAL_SOURCE_LOCK.md').read_text()
        for token in [
            'a[136] + a[137] + a[145]',
            'functions/csg_fun.py::ctl_csg1',
            'q_ext_M2 = Vent',
            'm3 m-2 s-1',
            'H_eff_M1 = p[49] = 6.2 m',
            'H_eff_M2 = D.Vair / D.area_floor'
        ]:
            self.assertIn(token,txt)

    def test_protocol_freezes_grid(self):
        txt=(HERE/'PROTOCOL.md').read_text()
        for token in [
            '0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9',
            'q = {0,0.25,0.50,0.75,1.00}',
            'D_V(m,i,u,h)',
            'D_N(m,i,u,h)',
            'No extrapolation'
        ]:
            self.assertIn(token,txt)

    def test_scripts_use_frozen_physical_terms(self):
        m1=(HERE/'m1_physical_dose.py').read_text()
        m2=(HERE/'m2_physical_dose.py').read_text()
        self.assertIn('a_sym[136]+a_sym[137]+a_sym[145]',m1)
        self.assertIn("vent_records.append(scalar(res[4]))",m2)
        self.assertIn("U9=np.round(np.arange(0.1,1.0,0.1),1)",m1)
        self.assertIn("U9=np.round(np.arange(0.1,1.0,0.1),1)",m2)

if __name__=='__main__':
    unittest.main()
