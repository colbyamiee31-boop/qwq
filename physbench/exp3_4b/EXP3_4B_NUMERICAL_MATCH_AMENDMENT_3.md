# EXP3.4B numerical dose-matching amendment 3 — representable-command floor

Frozen before any EXP3.4B aggregate decision result is produced or inspected.

## Trigger

The final fail-closed run under Amendment 2 reached M1 event 158, DV, 30 min, q=0.25 with:

- target dose: approximately `1.16938817776 m3 m-2`;
- best achieved dose: approximately `1.16947583618 m3 m-2`;
- best residual: approximately `8.77e-5 m3 m-2`;
- nominal Amendment-2 tolerance: approximately `1.79e-5 m3 m-2`;
- final native-command bracket:
  `[0.10216119140386581, 0.10216119885444641]`.

The bracket width is `7.4505806e-9`, i.e. one adjacent representable step in the frozen M1 `float32` action path.
The target therefore lies between physical doses produced by two adjacent representable native commands.

At this point no EXP3.4B aggregate job had run and no matched-dose decision result existed.

## Final quantization-aware fallback

The Amendment-2 continuous-root tolerance remains the primary acceptance rule:

`tol_match = max(1e-8, 1e-6 * max(1, |D_target|), 2e-4 * |H-L|)`.

Only if a target still fails this rule after the predeclared 30 safeguarded bisection iterations may M1 use the
following fallback. All conditions must hold:

1. the final lower and upper commands are adjacent `float32` values;
2. both adjacent commands are explicitly re-evaluated by the frozen native M1 runner;
3. their achieved physical doses bracket the frozen target;
4. the command with the smaller absolute physical-dose residual is retained;
5. its residual satisfies

   `|D_achieved-D_target| <= 0.5*|D_hi-D_lo| + eps_q`

   where

   `eps_q = 1e-10 * max(1, |D_target|, |D_lo|, |D_hi|)`.

If any condition fails, the run fails closed.

This is a nearest-representable-value rule, not a global tolerance increase. The nominal Amendment-2 tolerance,
the local command quantization gap, the local physical-dose quantization gap, the quantization acceptance bound,
and the acceptance mode are all stored per row.

For all non-quantized rows, the original Amendment-2 tolerance remains the sole hard acceptance threshold.

## Cross-model audit

Aggregate analysis uses each row's actual acceptance tolerance:
- Amendment-2 tolerance for ordinary continuous matches;
- the local quantization bound only for rows that pass the strict adjacent-representable fallback above.

The achieved M1/M2 physical-dose gap must remain no greater than the sum of the two row-specific acceptance tolerances.

## Not changed

This amendment does not change:
- any EXP3.4A target or its SHA256;
- q = {0,0.25,0.50,0.75,1.00};
- any eligible cohort;
- the native command domain [0.1,0.9];
- model equations, parameters, forcing or initialization;
- physical ventilation definitions;
- T/AH benefit definitions;
- C(q)=q^2;
- lambda domains or decision metrics;
- the locked EXP3.1 comparator.

It only defines how to represent a frozen physical target when that target falls strictly between two adjacent
commands that the frozen M1 runner can numerically represent.
