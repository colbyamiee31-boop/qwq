"""EXP1.5 read-only M1 flux integration adapter using the frozen native state path.

Purpose
-------
The first EXP1.5 audit showed that CasADi/CVODES quadratures with the frozen state
solver tolerances reconstruct the instantaneous named CO2 ODE exactly but can accumulate
non-negligible quadrature-vs-final-state mismatch over the expanded 300/900/1800-s
surface. Enabling CVODES quadrature error control changed the adaptive state trajectory,
which is unacceptable for a read-only attribution benchmark.

This adapter therefore leaves every state equation and native state-only CVODES setting
unchanged, asks the same state-only solver for a dense 1-s output grid, evaluates the
exact symbolic EXP1.3 flux-rate expressions on that frozen native state path, and
integrates those rates by the trapezoidal rule. The final state used by the attribution
is the final state of this state-only native-path integrator and is still checked against
the original frozen final-state integrator by the unchanged EXP1.5 gate.

No flux term, parameter, forcing, action, horizon, or root is modified.
"""
from pathlib import Path
import runpy
import numpy as np
import casadi as ca

ROOT = Path(__file__).resolve().parents[2]
_original_integrator = ca.integrator
SAMPLE_DT = 1.0


class _NativePathQuadrature:
    def __init__(self, name, solver, dae, t0, tf, opts):
        self.name = name
        self.t0 = float(t0)
        self.tf = float(tf)
        n = int(round((self.tf - self.t0) / SAMPLE_DT))
        self.grid = np.linspace(self.t0, self.tf, n + 1)
        state_dae = {k: dae[k] for k in ['x','u','p','ode']}
        self.path_integrator = _original_integrator(
            name + '_native_path', solver, state_dae, self.t0, self.grid, dict(opts)
        )
        self.qfun = ca.Function(
            name + '_qrate', [dae['x'], dae['u'], dae['p']], [dae['quad']]
        )
        self.qmap = self.qfun.map(len(self.grid))

    def __call__(self, **kwargs):
        r = self.path_integrator(x0=kwargs['x0'], u=kwargs['u'], p=kwargs['p'])
        states = r['xf']
        if states.size2() != len(self.grid):
            raise AssertionError(
                f'Unexpected native-path output columns: {states.size2()} != {len(self.grid)}'
            )
        u = ca.DM(kwargs['u'])
        p = ca.DM(kwargs['p'])
        umat = ca.repmat(u, 1, len(self.grid))
        pmat = ca.repmat(p, 1, len(self.grid))
        qvals = np.asarray(self.qmap(states, umat, pmat), dtype=float)
        qint = np.trapz(qvals, x=self.grid, axis=1)
        return {'xf': states[:, -1], 'qf': ca.DM(qint)}


def _integrator_native_path_quad(*args, **kwargs):
    args = list(args)
    dae = args[2] if len(args) > 2 else kwargs.get('dae')
    has_quad = isinstance(dae, dict) and 'quad' in dae
    if not has_quad:
        return _original_integrator(*args, **kwargs)

    if len(args) < 6:
        raise TypeError('EXP1.5 native-path adapter expects positional t0, tf, opts')
    name, solver, dae, t0, tf, opts = args[:6]
    return _NativePathQuadrature(name, solver, dae, t0, tf, opts)


ca.integrator = _integrator_native_path_quad
runpy.run_path(str(ROOT / 'physbench/exp1_5/m1_boundary_mechanism_stability.py'), run_name='__main__')
