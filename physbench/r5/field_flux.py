"""Reference implementation of the field-identified ventilation flux (authoritative; do not re-derive).
q_field(vL, vW, U, Tin, Tout) -> m3 m-2 s-1 (roof-vent driven exchange; leakage NOT included)."""
import json, math, os
_P = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'field_ventilation_function.json')))
def _phi(v):
    s = _P['slope']; return [1 / (1 + math.exp(-s * (v - c))) - 1 / (1 + math.exp(s * c)) for c in _P['centres']]
def q_field(vL, vW, U, Tin, Tout):
    gL = sum(a * p for a, p in zip(_P['aL'], _phi(vL))); gW = sum(a * p for a, p in zip(_P['aW'], _phi(vW)))
    return U * (gL + gW) + _P['b'] * (vL + vW) * math.sqrt(max(Tin - Tout, 0.0))
if __name__ == '__main__':      # unit tests (values computed by the identification code)
    tests = [((0.0, 0.0, 4, 18, 10), 0.0), ((0.2, 0.0, 4, 18, 10), 3.205101e-03), ((0.5, 0.5, 4, 18, 10), 2.004732e-02), ((1.0, 1.0, 6, 25, 20), 3.266923e-02)]
    for args, ref in tests:
        v = q_field(*args); assert abs(v - ref) <= 1e-8 + 1e-5 * abs(ref), (args, v, ref)
    print('field_flux unit tests PASS')
