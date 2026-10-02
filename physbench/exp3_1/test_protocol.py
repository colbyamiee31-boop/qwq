import unittest, numpy as np

A=np.array([0.1,0.3,0.5,0.7,0.9],float)
C=((A-0.1)/0.8)**2

def choose(B,lam):
    q=np.asarray(B)-lam*C
    mx=q.max()
    return int(np.where(q>=mx-1e-12)[0][0])

class TestDecisionProtocol(unittest.TestCase):
    def test_cost_monotone(self):
        self.assertTrue(np.all(np.diff(C)>0))
        self.assertAlmostEqual(C[0],0.0)
        self.assertAlmostEqual(C[-1],1.0)
    def test_zero_benefit_chooses_low(self):
        self.assertEqual(choose(np.zeros(5),0.5),0)
    def test_tie_chooses_lower(self):
        B=np.zeros(5); B[0]=0.1; B[1]=0.1
        self.assertEqual(choose(B,0.0),0)
    def test_high_benefit_can_choose_high(self):
        B=np.array([0,0.1,0.2,0.3,1.0])
        self.assertEqual(choose(B,0.0),4)
    def test_penalty_can_reverse_decision(self):
        B=np.array([0,0.1,0.2,0.3,0.4])
        self.assertEqual(choose(B,0.0),4)
        self.assertEqual(choose(B,10.0),0)

if __name__=='__main__':
    unittest.main()
