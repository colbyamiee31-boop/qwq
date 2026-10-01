"""EXP1.4 technical adapter for frozen CSGtom short-horizon solar geometry.

CSGtom's csg_shape.tran_cover() samples solar geometry every 600 s with
np.arange(0, end_t, 600) and then builds an slinear interp1d. A 300-s
simulation therefore supplies only one support point and SciPy rejects the
interpolator before dynamics are evaluated.

This adapter does NOT extend the dynamical rollout. It only guarantees a
900-s geometry-construction support window, yielding the native [0, 600] s
support points. The model is still integrated for the requested 300 s using
the frozen 30-s forward-Euler step.
"""
from pathlib import Path
from datetime import datetime, timedelta
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
CSG = ROOT / "CSGtom"
sys.path.insert(0, str(CSG))

from functions import csg_shape

_original_tran_cover = csg_shape.csg_shape.tran_cover


def _tran_cover_with_minimum_support(self):
    start = datetime.strptime(self.p['StartTime'], '%Y-%m-%dT%H:%M')
    end = datetime.strptime(self.p['EndTime'], '%Y-%m-%dT%H:%M')
    original_end = self.p['EndTime']
    # Need end_t > 600 because np.arange(0, end_t, 600) excludes end_t.
    if (end - start).total_seconds() <= 600:
        self.p['EndTime'] = (start + timedelta(seconds=900)).strftime('%Y-%m-%dT%H:%M')
        try:
            return _original_tran_cover(self)
        finally:
            self.p['EndTime'] = original_end
    return _original_tran_cover(self)


csg_shape.csg_shape.tran_cover = _tran_cover_with_minimum_support

runpy.run_path(str(ROOT / 'physbench/exp1_4/m2_boundary_robustness.py'), run_name='__main__')
