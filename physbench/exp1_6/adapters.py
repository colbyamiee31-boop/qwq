"""Read-only reuse of the frozen EXP1.5 function definitions.

The legacy files execute experiments on import. Load their exact prefix before
the first cell_rows assignment, without copying or editing their physics.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = '92ad719b08b4412afd95919481065a8ee8f5c8e1'
COMMITS = {'M1': '2d3febb1ea002b24b452e32293e990beb78d3ce1',
           'M2': 'bea8c3b0a1324162a4b5487db578aa674c8b587c'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_inputs():
    lock = json.loads((Path(__file__).with_name('frozen_inputs.json')).read_text())
    records = []
    for rel, expected in lock.items():
        path = ROOT / rel
        normalized = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if normalized != expected:
            raise RuntimeError(f'Frozen input mismatch: {rel}')
        records.append({'path': rel, 'sha256_bytes': digest(path), 'sha256_lf': normalized})
    return records


class Adapter:
    def __init__(self, model, model_source):
        self.model = model
        self.input_records = verify_inputs()
        source = Path(model_source).resolve()
        commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        if commit != COMMITS[model]:
            raise RuntimeError(f'Incorrect {model} commit: {commit}')
        dirty = subprocess.check_output(['git', '-C', str(source), 'diff', 'HEAD', '--'], text=True)
        if dirty:
            raise RuntimeError('Frozen model has tracked modifications')
        self.source = source
        self.source_commit = commit
        self.model_files = []
        tracked = subprocess.check_output(['git', '-C', str(source), 'ls-files'], text=True).splitlines()
        for rel in tracked:
            p = source / rel
            if p.is_file():
                self.model_files.append({'path': rel, 'sha256': digest(p)})
        if model == 'M1':
            import gl_gym
            installed = Path(gl_gym.__file__).parent
            for p in (source / 'gl_gym').rglob('*.py'):
                counterpart = installed / p.relative_to(source / 'gl_gym')
                if p.read_bytes().replace(b'\r\n', b'\n') != counterpart.read_bytes().replace(b'\r\n', b'\n'):
                    raise RuntimeError(f'Installed M1 code mismatch: {p}')
        else:
            if source != (ROOT / 'CSGtom').resolve():
                raise RuntimeError('M2 source must be ROOT/CSGtom for the legacy adapter')
            os.chdir(source / 'models')
        script = ROOT / 'physbench/exp1_5' / f'{model.lower()}_boundary_mechanism_stability.py'
        tree = ast.parse(script.read_text(encoding='utf-8'))
        marker = next(i for i, node in enumerate(tree.body)
                      if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'cell_rows' for t in node.targets))
        prefix = ast.Module(body=tree.body[:marker], type_ignores=[])
        self.ns = {'__file__': str(script), '__name__': '_exp1_5_frozen_prefix'}
        exec(compile(prefix, str(script), 'exec'), self.ns)
        self.terms = list(self.ns['VECTOR_NAMES'])
        self.states = [s['state_id'] for s in self.ns['STATE_CASES']]

    def ensure_horizon(self, h):
        if int(h) != h or not 300 <= h <= 1800:
            raise ValueError('H must be an integer in [300,1800]')
        if self.model == 'M2':
            if h % 60:
                raise ValueError('M2 preserves minute timestamps and 30 s Euler: require H % 60 == 0')
            return
        n = self.ns
        if h in n['F_BY_H']:
            return
        # The same one-shot native and read-only quadrature options as EXP1.5.
        for name, quad in [('F_BY_H', False), ('FQ_BY_H', True)]:
            dae = {'x': n['x'], 'u': n['u'], 'p': n['ca'].vertcat(n['d'], n['p']), 'ode': n['dx']}
            if quad:
                dae['quad'] = n['q_terms']
            n[name][h] = n['ca'].integrator(f'EXP1_6_{name}_H{h}', 'cvodes', dae, 0., float(h),
                                          {'abstol': 1e-4, 'reltol': 1e-4, 'max_num_steps': 150000})

    def evaluate(self, h, state, forcing, c, eval_id):
        import numpy as np
        self.ensure_horizon(h)
        if not 250 <= c <= 600:
            raise ValueError('CO2 outside frozen domain')
        n = self.ns
        if self.model == 'M1':
            pair = n['paired_decomp'](h, state, forcing, c)
            for arm in ['LOW', 'HIGH']:
                a = pair[arm]
                if not a['finite'] or a['augmented_vs_native_final_state_max_abs_diff'] > 1e-7 or abs(a['named_vs_total_ode_closure_mg_m3']) > 1e-10:
                    raise RuntimeError(f'M1 trajectory/algebra gate failure: {eval_id}')
            # Store the exact one-shot endpoints: dense sampling would change scheduling.
            pair['native_initial_state'] = n['x0_for'](state, c).tolist()
            pair['native_final_states'] = {arm: n['native_arm'](h, state, forcing, c, vent)['xf'].tolist()
                                           for arm, vent in [('LOW', .1), ('HIGH', .9)]}
            trace = []
        else:
            n['trace_rows'].clear()
            pair = n['paired_decomp'](h, state, forcing, c, eval_id, 'EXP1_6')
            trace = list(n['trace_rows'])
            n['trace_rows'].clear()
            native = [n['native_arm'](h, state, forcing, c, vent) for vent in [.1, .9]]
            for arm, nat in zip(['LOW', 'HIGH'], native):
                a = pair[arm]
                if not a['finite'] or not nat['finite'] or a['max_requested_applied_error'] != 0 or nat['max_requested_applied_error'] != 0 or abs(a['total_ode_vs_actual_closure_ppm']) > 1e-8 or abs(a['final_co2_ppm'] - nat['final_co2_ppm']) > 1e-8:
                    raise RuntimeError(f'M2 native/attribution/action gate failure: {eval_id}')
        v = pair['high_minus_low']['flux_contributions_ppm']
        r = pair['high_minus_low']['final_co2_ppm']
        if not np.isfinite([r, *v.values()]).all():
            raise RuntimeError('Non-finite response/contribution')
        epsilon = r - sum(v.values())
        if self.model == 'M2' and abs(epsilon) > 1e-8:
            raise RuntimeError('M2 pair closure failure')
        return pair, trace

    def close(self):
        verify_inputs()
        for row in self.model_files:
            if digest(self.source / row['path']) != row['sha256']:
                raise RuntimeError(f'Model changed during run: {row["path"]}')
        if self.model == 'M1':
            self.ns['env'].close()
