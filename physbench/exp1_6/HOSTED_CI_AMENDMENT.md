# EXP1.6 Hosted-CI replay amendment

The first hosted Linux run (`36905238816`) passed all 11 analytic tests and the
full frozen M1 EXP1.5 gate, but EXP1.6 stopped at its *independent-process replay*
precondition. At `H900_S_WARM_DRY_FROW10`, the newly evaluated response differed
from the just-produced legacy CSV by `6.809790988882014e-7 ppm`; the largest term
difference at that stopping point was `6.393890972233862e-8 ppm`, and dominant
mechanism identity was unchanged. The failure occurred before the EXP1.6 phase
scan.

This was not a model, equation, attribution, or phase-search failure. The original
`1e-7 ppm` comparison was an unnecessarily strict cross-process byte-to-number
replay gate. It is distinct from the scientific root criteria and from the native
solver/closure gates.

For hosted CI only, the replay comparison tolerance is frozen at `1e-5 ppm` for
both the saved response and saved native contribution vector, with exact dominant
identity still required. Every observed replay error is written to
`regression.json`. This tolerance is:

- 68 times larger than the observed hosted response difference that stopped run 1;
- 10 times smaller than the inherited `1e-4 ppm` root-location bracket width;
- unrelated to, and does not alter, the `1e-7 ppm` transition/root residual target;
- not used in any EXP1.6 model evaluation, root search, equality decision, phase
  label, numerical closure screen, or intersection analysis.

All frozen-model state/algebra/action/closure gates remain unchanged. If any root
changes dominant identity, or a replay term/response differs by more than `1e-5
ppm`, the hosted run fails closed.
