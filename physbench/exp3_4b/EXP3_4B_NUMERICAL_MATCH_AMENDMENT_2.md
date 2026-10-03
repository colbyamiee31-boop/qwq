# EXP3.4B numerical dose-matching amendment 2

Frozen before any EXP3.4B aggregate decision result is produced or inspected.

## Trigger

A fail-closed M1 matched-dose run on the already frozen target set reached event 158, DN, 30 min, q=0.75 with:
- target dose about 0.31962246;
- best residual about 2.26e-5;
- common physical-dose span about 0.17950177;
- residual/common-span about 1.26e-4.

At that point the safeguarded bisection bracket had collapsed to approximately one representable native-command increment
under the frozen M1 action path. No EXP3.4B decision aggregate had run and no matched-dose decision result had been inspected.

The first amendment used a common-span numerical floor of 1e-4 * |H-L|. The observed frozen-runner resolution can
slightly exceed that floor.

## Final numerical floor

For each target:

`tol_match = max(1e-8, 1e-6 * max(1, |D_target|), 2e-4 * |H-L|)`.

Thus the common-span numerical floor is **0.02% of the event-specific common physical-dose span**.

The five frozen decision levels remain separated by 25% of the common span. Therefore this floor is only
0.0008 of one adjacent q-level spacing, i.e. **0.08% of the adjacent decision-level separation**.

The achieved residual/common-span ratio is reported for every row.

## Not changed

No EXP3.4A target, target SHA, q level, eligible cohort, native command domain, model equation, model parameter,
forcing, initialization, physical ventilation definition, decision benefit, cost C(q)=q^2, lambda domain, metric,
or locked EXP3.1 comparator is changed.

This amendment changes only the numerical acceptance floor for an otherwise already solved physical target and is
frozen before any aggregate decision output exists.
