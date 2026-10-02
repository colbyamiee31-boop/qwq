import unittest, numpy as np

U3=np.array([0.1,0.5,0.9],float)
U5=np.array([0.1,0.3,0.5,0.7,0.9],float)
U9=np.round(np.arange(0.1,1.0,0.1),1)
WEIGHTS=[(0.75,0.25),(0.50,0.50),(0.25,0.75)]

def cost(actions,form):
    z=(np.asarray(actions,float)-0.1)/0.8
    if form=='quadratic': return z*z
    if form=='linear': return z
    raise ValueError(form)

def choose(B,C,lam):
    q=np.asarray(B,float)-lam*np.asarray(C,float)
    mx=q.max()
    return int(np.where(q>=mx-1e-12)[0][0])

class TestRobustnessProtocol(unittest.TestCase):
    def test_nested_grids(self):
        self.assertEqual(U3.tolist(),[0.1,0.5,0.9])
        self.assertEqual(U5.tolist(),[0.1,0.3,0.5,0.7,0.9])
        self.assertEqual(U9.tolist(),[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
        self.assertTrue(set(U3).issubset(set(U5)))
        self.assertTrue(set(U5).issubset(set(U9)))
    def test_weights(self):
        self.assertEqual(WEIGHTS,[(0.75,0.25),(0.5,0.5),(0.25,0.75)])
        for a,b in WEIGHTS: self.assertAlmostEqual(a+b,1.0)
    def test_costs(self):
        for form in ('quadratic','linear'):
            c=cost(U9,form)
            self.assertAlmostEqual(c[0],0.0)
            self.assertAlmostEqual(c[-1],1.0)
            self.assertTrue(np.all(np.diff(c)>0))
    def test_tie_chooses_lower(self):
        B=np.array([0.2,0.2,0.0])
        C=np.zeros(3)
        self.assertEqual(choose(B,C,0.0),0)
    def test_reference_spec_is_present(self):
        self.assertIn((0.5,0.5),WEIGHTS)
        self.assertEqual(cost(U5,'quadratic').tolist(),(((U5-0.1)/0.8)**2).tolist())

if __name__=='__main__':
    unittest.main()
