# EXP3.4B numerical dose-matching amendment

Frozen before any EXP3.4B aggregate decision result is produced or inspected.

## Trigger

The original root-acceptance tolerance was:

`max(1e-8, 1e-6 * max(1, |D_target|, |H-L|))`.

Fail-closed hosted execution showed that this absolute/magnitude-scaled rule can be tighter than the numerical
resolution of the frozen model runners even after the command bracket has collapsed:

- M1 event 105, DV, 15 min, q=1:
  residual about `1.30e-5 m3 m-2` after 30 bisections, while the common span is about
  `0.23587 m3 m-2` (residual/common-span about `5.5e-5`);
- an earlier M2 implementation run at event 20, DN, 15 min, q=1 showed a residual about
  `2.66e-6` over a common span about `0.75993` (about `3.5e-6` of the common span).

No EXP3.4B decision aggregate had run at the time of this amendment. No decision result was available.

## Amended acceptance tolerance

For each frozen target:

`tol_match = max(1e-8, 1e-6 * max(1, |D_target|), 1e-4 * |H-L|)`.

Thus the tolerance includes a common-span numerical floor equal to **0.01% of the event-specific common physical-dose span**. Because the rule is a maximum of three terms, the pre-existing target-magnitude term can be larger than this floor for narrow common intervals.

The five decision levels remain spaced by 25% of that span, so the common-span floor itself is 1/2500 of an adjacent decision-level spacing. The achieved residual/common-span ratio is reported for every accepted row rather than promoted to an additional unregistered hard gate.

## Solver rule

The native-command bracket, target, q grid and no-extrapolation domain are unchanged.

The solver:
1. starts from the frozen EXP3.4A U9 bracket and linear inverse;
2. performs safeguarded bisection for at most 30 iterations;
3. retains the evaluated command with the smallest absolute target-dose residual;
4. accepts only if the best residual is <= `tol_match`.

The achieved dose and residual are stored for every row. Cross-model achieved-dose mismatch is audited separately.

## Not changed

This amendment does not change:
- any EXP3.4A target dose or target SHA;
- q={0,0.25,0.50,0.75,1.00};
- the 42-event primary DV15 cohort;
- any secondary cohort;
- model equations, parameters, forcing or initialization;
- the physical flux definitions;
- the decision benefit;
- `C(q)=q^2`;
- lambda domains;
- decision metrics;
- the locked EXP3.1 comparator.

This is strictly a numerical root-acceptance amendment made before decision outcomes exist.
