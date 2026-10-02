import hashlib, json, sys, unittest
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from generate_prehistory import generate_prehistory

CFG=json.loads((HERE/'latent_history_config.json').read_text())

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class TestFrozenLatentHistoryProtocol(unittest.TestCase):
    def test_history_counts(self):
        self.assertEqual(len(CFG['histories']['corners']),16)
        self.assertEqual(len(CFG['anchor_primary_history_ids']),17)
        self.assertEqual(len(CFG['resolution_IV_corner_ids']),8)
        self.assertEqual(len(CFG['decision_primary_history_ids']),9)
        self.assertNotIn('HNR',CFG['anchor_primary_history_ids'])
        self.assertNotIn('HNR',CFG['decision_primary_history_ids'])

    def test_full_factorial_unique(self):
        rows=[]
        for h in CFG['histories']['corners']:
            row=(h['A_thermal'],h['B_humidity'],h['C_radiation'],h['D_co2'])
            rows.append(row)
            self.assertTrue(all(v in (-1,1) for v in row))
        self.assertEqual(len(set(rows)),16)

    def test_resolution_iv_relation_and_main_effect_orthogonality(self):
        by_id={h['id']:h for h in CFG['histories']['corners']}
        X=[]
        for hid in CFG['resolution_IV_corner_ids']:
            h=by_id[hid]
            self.assertEqual(h['D_co2'],h['A_thermal']*h['B_humidity']*h['C_radiation'])
            X.append([h['A_thermal'],h['B_humidity'],h['C_radiation'],h['D_co2']])
        X=np.asarray(X,dtype=int)
        self.assertTrue(np.all(X.sum(axis=0)==0))
        gram=X.T@X
        self.assertTrue(np.array_equal(gram,np.eye(4,dtype=int)*8))

    def test_primary_and_slow_durations(self):
        self.assertEqual(CFG['primary_prehistory_h'],72)
        self.assertEqual(CFG['slow_memory_prehistory_h'],168)
        self.assertEqual(CFG['common_grid_s'],900)

    def test_history_generator_is_deterministic(self):
        a=generate_prehistory('H00',72,8.0)
        b=generate_prehistory('H00',72,8.0)
        pd.testing.assert_frame_equal(a,b,check_exact=True)
        self.assertEqual(len(a),72*4+1)
        self.assertAlmostEqual(float(a.local_hour.iloc[-1]),8.0)

    def test_forcing_bounds_all_primary_histories(self):
        for hid in CFG['anchor_primary_history_ids']:
            x=generate_prehistory(hid,72,8.0)
            self.assertTrue(np.isfinite(x.select_dtypes(include=[np.number]).to_numpy()).all())
            self.assertGreaterEqual(float(x.global_radiation_w_m2.min()),0.0)
            self.assertGreaterEqual(float(x.outdoor_rh_pct.min()),35.0)
            self.assertLessEqual(float(x.outdoor_rh_pct.max()),95.0)
            self.assertGreater(float(x.outdoor_vapor_pressure_pa.min()),0.0)
            self.assertGreater(float(x.outdoor_co2_ppm.min()),0.0)
            self.assertTrue(np.allclose(x.sky_temperature_c,x.outdoor_temperature_c-6.0))

    def test_center_history_has_no_factor_shift(self):
        x=generate_prehistory('H00',72,8.0)
        self.assertAlmostEqual(float(x.outdoor_co2_ppm.min()),415.0)
        self.assertAlmostEqual(float(x.outdoor_co2_ppm.max()),415.0)
        self.assertAlmostEqual(float(x.wind_speed_m_s.min()),1.5)
        self.assertAlmostEqual(float(x.wind_speed_m_s.max()),1.5)
        self.assertAlmostEqual(float(x.soil_boundary_temperature_c.min()),19.0)

    def test_reference_science_values_are_frozen_not_success_targets(self):
        r=CFG['hnr_reference']
        self.assertAlmostEqual(r['M1_root_ppm'],438.5370684042573)
        self.assertAlmostEqual(r['M2_root_ppm'],358.2789194211364)
        self.assertEqual(r['disagreement_event_ids'],[27,86,89])
        self.assertAlmostEqual(r['mean_integrated_disagreement'],0.024552427907011506)

    def test_locked_event_input_identity(self):
        p=ROOT/'physbench/exp2_1c/D2_PRIMARY_61_MODEL_INPUT.csv'
        self.assertEqual(sha256(p),'8a5890dc9fb6950ce3c3e7ffc8ff900133ede8ffde99dfa68d79435314b56023')
        df=pd.read_csv(p)
        self.assertEqual(df.event_id.nunique(),61)

    def test_action_and_root_definitions(self):
        self.assertEqual(CFG['action_grid'],[0.1,0.3,0.5,0.7,0.9])
        self.assertEqual(CFG['low_vent_command'],0.1)
        self.assertEqual(CFG['neutral_prehistory_vent_command'],0.3)
        self.assertEqual(CFG['high_vent_command'],0.9)
        r=CFG['root_scan']
        self.assertEqual((r['min_ppm'],r['max_ppm'],r['step_ppm']),(250.0,600.0,25.0))
        self.assertEqual(r['max_iterations'],50)

if __name__=='__main__':
    unittest.main()
