import math
import unittest
from numerics import bisect, roots_on_grid, classify, dominant_pair


class SearchTests(unittest.TestCase):
    def test_simple_root(self):
        r=bisect(lambda x:x-.3,0,1)
        self.assertLess(abs(r['x']-.3),1e-4)
        self.assertTrue(r['history'])

    def test_multiple_and_endpoint_roots(self):
        roots=roots_on_grid(lambda x:x*(x-1)*(x-2),[0,.5,1,1.5,2])
        self.assertEqual([r['x'] for r in roots],[0,1,2])

    def test_same_sign_is_not_certified_absence(self):
        self.assertEqual(roots_on_grid(lambda x:(x-.4)**2,[0,1]),[])
        with self.assertRaises(ValueError):
            bisect(lambda x:x*x+1,-1,1)

    def test_nonfinite_is_failure(self):
        with self.assertRaises(ValueError):
            roots_on_grid(lambda x:float('nan'),[0,1])
        with self.assertRaises(ValueError):
            classify({'a':1,'b':math.inf},0,'M1')

    def test_iteration_limit_is_not_success(self):
        self.assertEqual(bisect(lambda x:x-.123,0,1,limit=1)['status'],'ITERATION_LIMIT')

    def test_third_term_dominance(self):
        self.assertFalse(dominant_pair({'a':2,'b':-2,'c':3},'a','b'))
        self.assertTrue(dominant_pair({'a':2,'b':-2,'c':2},'a','b'))
        self.assertTrue(dominant_pair({'a':2.00001,'b':-2,'c':1},'a','b'))

    def test_closure_not_a_mechanism(self):
        r=classify({'a':2,'b':-1},1.1,'M1')
        self.assertFalse(r['resolved_under_screen'])
        self.assertEqual(r['dominant'],'a')

    def test_zero_tie_and_m2_floor(self):
        self.assertEqual(classify({'a':0,'b':0},0,'M1')['phase'],'UNRESOLVED')
        self.assertEqual(classify({'a':2,'b':-2,'c':2},0,'M1')['phase'],'UNRESOLVED')
        self.assertFalse(classify({'a':1,'b':-1+1e-10},0,'M2')['resolved_under_screen'])

    def test_discontinuity_width_does_not_mean_residual(self):
        r=bisect(lambda x:-1 if x<.123 else 1,0,1)
        self.assertEqual(r['status'],'WIDTH_ONLY')
        self.assertEqual(abs(r['f']),1)

    def test_contribution_zero_seed_recovers_narrow_phase(self):
        gap=lambda c:abs(c-.5)-.01
        self.assertEqual(roots_on_grid(gap,[0.,1.]),[])
        roots=roots_on_grid(gap,[0.,.5,1.])
        self.assertEqual(len(roots),2)
        self.assertAlmostEqual(roots[0]['x'],.49,delta=1e-4)
        self.assertAlmostEqual(roots[1]['x'],.51,delta=1e-4)

    def test_tangent_grid_zero_is_retained_without_claiming_phase_change(self):
        roots=roots_on_grid(lambda c:(c-.5)**2,[0.,.5,1.])
        self.assertEqual(len(roots),1)
        self.assertEqual(roots[0]['status'],'GRID_ZERO')


if __name__=='__main__':
    unittest.main()
