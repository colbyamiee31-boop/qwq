"""EXP1.5 numerical adapter for GreenLight read-only flux quadratures.

The first EXP1.5 audit reproduced the EXP1.4 surface exactly and reconstructed the
instantaneous named CO2 ODE to machine precision, but CVODES quadrature integration
without quadrature error control accumulated up to ~0.256 ppm mismatch over the
expanded 300/900/1800-s root+edge set.

This adapter changes no GreenLight state equation, parameter, forcing, action, horizon,
root, or native solver tolerance. It only enables CVODES quadrature error control for
integrators that contain a `quad` field. Native state-only integrators remain untouched.
The EXP1.5 script still compares augmented final states against the frozen native
integrator and retains the original state/closure gates.
"""
from pathlib import Path
import runpy
import casadi as ca

ROOT = Path(__file__).resolve().parents[2]
_original_integrator = ca.integrator


def _integrator_with_quad_error_control(*args, **kwargs):
    args = list(args)
    dae = args[2] if len(args) > 2 else kwargs.get('dae')
    has_quad = isinstance(dae, dict) and 'quad' in dae
    if has_quad:
        if len(args) >= 6 and isinstance(args[5], dict):
            opts = dict(args[5])
            opts['quad_err_con'] = True
            args[5] = opts
        elif 'opts' in kwargs:
            opts = dict(kwargs['opts'])
            opts['quad_err_con'] = True
            kwargs['opts'] = opts
    return _original_integrator(*args, **kwargs)


ca.integrator = _integrator_with_quad_error_control
runpy.run_path(str(ROOT / 'physbench/exp1_5/m1_boundary_mechanism_stability.py'), run_name='__main__')
