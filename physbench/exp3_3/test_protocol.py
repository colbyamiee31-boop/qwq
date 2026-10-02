import unittest, numpy as np

U3=np.array([0.1,0.5,0.9],float)
U5=np.array([0.1,0.3,0.5,0.7,0.9],float)
U9=np.round(np.arange(0.1,1.0,0.1),1)
WEIGHTS={'T75_AH25':(0.75,0.25),'equal':(0.5,0.5),'T25_AH75':(0.25,0.75)}

def cost(actions,form):
    z=(np.asarray(actions,float)-0.1)/0.8
    if form=='quadratic': return z*z
    if form=='linear': return z
    raise ValueError(form)

def choose(B,C,lam):
    q=np.asarray(B,float)-lam*np.asarray(C,float)
    mx=q.max()
    return int(np.where(q>=mx-1e-12)[0][0])

def negative_interval_stats(B,C,lo,hi):
    width=hi-lo
    if width<=0: return 0.0,0.0
    if abs(C)<=1e-15:
        return (width,-B*width) if B<0 else (0.0,0.0)
    root=B/C
    start=max(lo,root)
    if start>=hi: return 0.0,0.0
    start=max(start,lo)
    measure=hi-start
    area=0.5*C*(hi*hi-start*start)-B*(hi-start)
    return measure,max(0.0,area)

class TestModelSwapProtocol(unittest.TestCase):
    def test_reference_spec_frozen(self):
        self.assertEqual(U5.tolist(),[0.1,0.3,0.5,0.7,0.9])
        self.assertEqual(WEIGHTS['equal'],(0.5,0.5))
        self.assertEqual(cost(U5,'quadratic').tolist(),(((U5-0.1)/0.8)**2).tolist())
    def test_nested_grids(self):
        self.assertTrue(set(U3).issubset(set(U5)))
        self.assertTrue(set(U5).issubset(set(U9)))
    def test_baseline_is_zero_cost(self):
        for form in ('quadratic','linear'):
            self.assertAlmostEqual(cost(U9,form)[0],0.0)
    def test_negative_interval_exact(self):
        m,a=negative_interval_stats(0.2,0.4,0.0,1.0)
        self.assertAlmostEqual(m,0.5)
        self.assertAlmostEqual(a,0.05)
        m,a=negative_interval_stats(-0.1,0.0,0.0,1.0)
        self.assertAlmostEqual(m,1.0)
        self.assertAlmostEqual(a,0.1)
    def test_tie_chooses_lower(self):
        B=np.array([0.1,0.1,0.0]); C=np.zeros(3)
        self.assertEqual(choose(B,C,0.0),0)
    def test_directional_average_identity(self):
        r12=0.1; r21=0.3
        self.assertAlmostEqual(0.5*(r12+r21),0.2)

if __name__=='__main__':
    unittest.main()
