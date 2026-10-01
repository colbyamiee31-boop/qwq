"""Hosted-CI compatibility wrapper for EXP1.6.

Only the independent-process EXP1.5 replay comparison tolerance is changed.
No greenhouse model, solver, root search, attribution, or EXP1.6 scientific
calculation is changed. See HOSTED_CI_AMENDMENT.md.
"""
import csv
import json
import shutil
import run as base

REPLAY_TOLERANCE_PPM = 1e-5


def hosted_regression(self):
    path = base.ROOT / f'EXP1_5_{self.model}_BOUNDARY_MECHANISM_STABILITY.json'
    old = json.loads(path.read_text())
    if not old['run_pass']:
        raise RuntimeError('Legacy EXP1.5 did not pass')
    for p in base.ROOT.glob(f'EXP1_5_{self.model}_*'):
        if p.is_file():
            shutil.copy2(p, self.out / p.name)
    terms = list(csv.DictReader((base.ROOT / f'EXP1_5_{self.model}_TERM_CONTRIBUTIONS.csv').open(encoding='utf-8')))
    oldroots = list(csv.DictReader((base.ROOT / f'EXP1_5_{self.model}_ROOT_MECHANISMS.csv').open(encoding='utf-8')))
    for r in oldroots:
        e = self.evaluate(int(r['horizon_s']), r['state_id'], int(r['forcing_row']), float(r['initial_co2_ppm']))
        expected = {t['term']: float(t['contribution_ppm']) for t in terms
                    if t['condition_id'] == r['condition_id'] and t['point_type'] == r['point_type']
                    and abs(float(t['initial_co2_ppm']) - float(r['initial_co2_ppm'])) < 1e-7
                    and t['term'] in self.adapter.terms}
        if set(expected) != set(self.adapter.terms):
            raise RuntimeError('Incomplete legacy vector')
        error = max(abs(e[k] - v) for k, v in expected.items())
        response_error = abs(e['response_ppm'] - float(r['final_high_minus_low_ppm']))
        passed = (error <= REPLAY_TOLERANCE_PPM and
                  response_error <= REPLAY_TOLERANCE_PPM and
                  e['dominant'] == r['dominant_term'])
        self.regressions.append({
            'condition_id': r['condition_id'], 'eval_id': e['eval_id'],
            'max_term_error_ppm': error, 'response_error_ppm': response_error,
            'hosted_replay_tolerance_ppm': REPLAY_TOLERANCE_PPM,
            'dominant_identity_match': e['dominant'] == r['dominant_term'],
            'pass': passed,
        })
        if not passed:
            base.dump(self.out / 'regression.json', {
                'legacy_run_pass': True, 'surface': old['surface_regression'],
                'hosted_replay_tolerance_ppm': REPLAY_TOLERANCE_PPM,
                'prefix_loader': self.regressions,
            })
            raise RuntimeError('Hosted independent-process root/vector regression failed')
    base.dump(self.out / 'regression.json', {
        'legacy_run_pass': True,
        'surface': old['surface_regression'],
        'hosted_replay_tolerance_ppm': REPLAY_TOLERANCE_PPM,
        'max_observed_term_error_ppm': max(r['max_term_error_ppm'] for r in self.regressions),
        'max_observed_response_error_ppm': max(r['response_error_ppm'] for r in self.regressions),
        'all_dominant_identities_match': all(r['dominant_identity_match'] for r in self.regressions),
        'prefix_loader': self.regressions,
    })


base.Experiment.regression = hosted_regression

if __name__ == '__main__':
    base.main()
