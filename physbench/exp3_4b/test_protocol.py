import unittest
import numpy as np

Q=np.array([0.0,0.25,0.5,0.75,1.0],float)
TARGET_SHA='e07e64d8f8f7ac914a68b5cda73e0a76f5b9bd890a81473a42dd8ee8f6656292'
DV_ATOL=2e-5; DN_ATOL=5e-6; RTOL=2e-7

def cost(q,form):
    q=np.asarray(q,float)
    if form=='quadratic': return q*q
    if form=='linear': return q
    raise ValueError(form)

def choose(B,lam,form='quadratic'):
    util=np.asarray(B,float)-lam*cost(Q,form)
    mx=util.max()
    return int(np.where(util>=mx-1e-12)[0][0])

class TestProtocolFreeze(unittest.TestCase):
    def test_q_exact(self):
        self.assertEqual(Q.tolist(),[0.0,0.25,0.5,0.75,1.0])
    def test_target_sha_literal(self):
        self.assertEqual(TARGET_SHA,'e07e64d8f8f7ac914a68b5cda73e0a76f5b9bd890a81473a42dd8ee8f6656292')
    def test_primary_cost(self):
        self.assertTrue(np.allclose(cost(Q,'quadratic'),[0,0.0625,0.25,0.5625,1]))
    def test_linear_sensitivity(self):
        self.assertTrue(np.allclose(cost(Q,'linear'),Q))
    def test_tie_lower_q(self):
        self.assertEqual(choose(np.zeros(5),0.0),0)
    def test_penalty_can_change_choice(self):
        b=np.array([0,0.1,0.2,0.3,1.0])
        self.assertEqual(choose(b,0.0),4)
        self.assertEqual(choose(b,10.0),0)
    def test_tolerances_frozen_positive(self):
        self.assertEqual(DV_ATOL,2e-5); self.assertEqual(DN_ATOL,5e-6); self.assertEqual(RTOL,2e-7)

if __name__=='__main__': unittest.main()
