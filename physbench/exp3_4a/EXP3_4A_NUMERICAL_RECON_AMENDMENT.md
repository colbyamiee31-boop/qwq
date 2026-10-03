# EXP3.4A numerical reconstruction amendment

Frozen before any EXP3.4A physical-dose or overlap result is inspected.

## Trigger

The same-run reconstruction gate was initially written as a single absolute T/AH tolerance of 1e-9.
The untouched EXP3.1 reference writer serializes `action_grid_responses.csv` using `float_format='%.12g'`.

During fail-closed execution, both models reproduced temperature to about 5e-11 degC absolute error, while the
largest AH comparison was about 5e-9 in the ~10^3-valued AH coordinate. This is at the final-digit quantization scale
of a 12-significant-digit text representation.

The physical-dose values were not inspected before this amendment.

## Amended reconstruction gate

The same-run EXP3.1 reconstruction remains a hard gate, but its numerical comparison is made compatible with the
precision of the locked text artifact:

- temperature absolute error <= 1e-9 degC;
- scaled temperature error
  `|dT| / max(1, |T_ref|, |T_dose|) <= 1e-11`;
- scaled AH error
  `|dAH| / max(1, |AH_ref|, |AH_dose|) <= 1e-11`.

Absolute AH error is still reported, but is not used alone as a gate because the locked ~10^3-valued AH reference
is stored to 12 significant digits.

For M1, the augmented ventilation-dose quadrature is additionally required to reproduce the native trajectory state
with maximum absolute state error <= 1e-7 at every 900-s segment endpoint. The native trajectory, not the augmented
quadrature trajectory, supplies T/AH outcomes.

## Not changed

No model equation, parameter, event, forcing, physical ventilation-flux definition, command grid, horizon,
overlap rule, common-dose target grid, or decision definition is changed.

This amendment changes only the numerical comparison rule used to verify an already frozen, text-serialized
reference trajectory.
