"""Pure search primitives. No greenhouse physics and no silent success states."""
import math


def classify(v, closure, model):
    ranked = sorted(v, key=lambda k: (-abs(v[k]), k))
    if not all(math.isfinite(x) for x in [closure, *v.values()]):
        raise ValueError('Non-finite mechanism vector')
    l1 = sum(abs(x) for x in v.values())
    margin = abs(v[ranked[0]]) - abs(v[ranked[1]])
    # Preserve EXP1.5 observed-closure audit for M1. M2 uses its frozen 1e-8 gate
    # conservatively as a near-tie screen; neither is a per-term error bound.
    screen = abs(closure) if model == 'M1' else max(abs(closure), 1e-8)
    resolved = l1 > 0 and margin > screen
    return {'dominant': ranked[0] if l1 else 'NONE', 'runner_up': ranked[1],
            'margin_ppm': margin, 'l1_ppm': l1, 'screen_ppm': screen,
            'resolved_under_screen': resolved,
            'phase': ranked[0] if resolved else 'UNRESOLVED'}


def bisect(fn, lo, hi, width=1e-4, residual=1e-7, limit=50):
    flo, fhi = fn(lo), fn(hi)
    history = []
    if not all(math.isfinite(x) for x in [flo, fhi]):
        raise ValueError('Non-finite bracket')
    if abs(flo) <= residual:
        return {'x': lo, 'f': flo, 'lo': lo, 'hi': lo, 'status': 'RESIDUAL', 'history': history}
    if abs(fhi) <= residual:
        return {'x': hi, 'f': fhi, 'lo': hi, 'hi': hi, 'status': 'RESIDUAL', 'history': history}
    if flo * fhi >= 0:
        raise ValueError('No sign-changing bracket')
    for _ in range(limit):
        mid = (lo + hi) / 2
        fm = fn(mid)
        history.append({'lo': lo, 'hi': hi, 'flo': flo, 'fhi': fhi, 'mid': mid, 'fm': fm})
        if not math.isfinite(fm):
            raise ValueError('Non-finite midpoint')
        if abs(fm) <= residual or hi - lo <= width:
            return {'x': mid, 'f': fm, 'lo': lo, 'hi': hi,
                    'status': 'RESIDUAL' if abs(fm) <= residual else 'WIDTH_ONLY', 'history': history}
        if flo * fm < 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return {'x': mid, 'f': fm, 'lo': lo, 'hi': hi, 'status': 'ITERATION_LIMIT', 'history': history}


def roots_on_grid(fn, grid, residual=1e-7):
    vals = [fn(c) for c in grid]
    out = []
    for c, y in zip(grid, vals):
        if not math.isfinite(y):
            raise ValueError('Non-finite scan')
        if abs(y) <= residual:
            out.append({'x': c, 'f': y, 'lo': c, 'hi': c, 'status': 'GRID_ZERO', 'history': []})
    for lo, hi, a, b in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if a * b < 0:
            out.append(bisect(fn, lo, hi, residual=residual))
    result = []
    for row in sorted(out, key=lambda r: r['x']):
        if not result or row['x'] - result[-1]['x'] > 1e-3:
            result.append(row)
    return result


def dominant_pair(v, i, j, tolerance=1e-6):
    # Equality residual is reported by the root solver separately. Here only
    # test whether a *third* term dominates this pair; do not reject a
    # width-converged candidate merely because i and j are not exactly equal.
    competitors = [abs(x) for k, x in v.items() if k not in (i, j)]
    return min(abs(v[i]), abs(v[j])) + tolerance >= max(competitors, default=0.)
