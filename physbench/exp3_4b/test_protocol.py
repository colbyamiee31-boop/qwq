import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent

class TestEXP34BProtocol(unittest.TestCase):
    def test_protocol_locks_primary_and_secondary(self):
        p=(HERE/'PROTOCOL.md').read_text()
        for token in [
            'exactly the 42 EXP3.4A cells',
            'DN` at 15 min: exactly 61/61',
            'q in {0,0.25,0.50,0.75,1.00}',
            'C(q)=q^2',
            'Events `27, 86, 89`',
            'No all-61 EXP3.1 summary may be contrasted numerically with the 42-event DV primary result'
        ]:
            self.assertIn(token,p)

    def test_target_and_ancestry_are_frozen(self):
        p=(HERE/'PROTOCOL.md').read_text()
        self.assertIn('e07e64d8f8f7ac914a68b5cda73e0a76f5b9bd890a81473a42dd8ee8f6656292',p)
        self.assertIn('37095887502',p)
        self.assertIn('37024060571',p)
        self.assertIn('2d3febb1ea002b24b452e32293e990beb78d3ce1',p)
        self.assertIn('bea8c3b0a1324162a4b5487db578aa674c8b587c',p)

    def test_decision_cost_uses_q_not_native_command(self):
        a=(HERE/'decision_analysis.py').read_text()
        self.assertIn('Q=np.array([0.0,0.25,0.5,0.75,1.0]',a)
        self.assertIn('COST=Q**2',a)
        self.assertIn("gap+=width*abs(Q[i1]-Q[i2])",a)

    def test_matching_is_bounded_and_fail_closed(self):
        for name in ['m1_matched_dose.py','m2_matched_dose.py']:
            s=(HERE/name).read_text()
            self.assertIn('for _ in range(30)',s)
            self.assertIn("tol=max(1e-8,1e-6*max(1.0,abs(target),abs(width)))",s)
            self.assertIn('command out of domain',s)
            self.assertIn('dose tolerance fail',s)

if __name__=='__main__':
    unittest.main()
