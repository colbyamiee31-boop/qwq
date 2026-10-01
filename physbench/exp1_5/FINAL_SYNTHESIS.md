# PhysBench-GH EXP1.5 — Final synthesis

Final CI run: `36891002335`

## Status
- M1 GreenLight-Gym2: PASS
- M2 CSGtom: PASS

## Surface regression
- M1: 21 unique-root cells, 6 no-crossing cells, 0 multiple roots; anchor 438.537073 ppm.
- M2: 18 unique-root cells, 9 no-crossing cells, 0 multiple roots; anchor 358.278894 ppm.

## M1 mechanism structure
Anchor active pair:
- `main_to_top`: positive, +2.446999 ppm;
- `temperature_conversion`: negative, -2.303717 ppm;
with a much smaller canopy term and negligible `main_to_outside` at the reversal root.

Across 21 unique-root cells:
- `main_to_top` dominant: 9;
- `temperature_conversion` dominant: 12;
- resolved dominant switches from anchor: 12;
- numerically unresolved roots under the explicit closure bound: 0.

The active sign structure remains stable: the core pair is `main_to_top:+` and `temperature_conversion:-`; six cells additionally activate `canopy_net:+` under the predeclared 10% L1 activity rule. No active-term sign flip was detected relative to the anchor.

`temperature_conversion` is a temperature-dependent density-to-ppm representation contribution, not a CO2 mass source/sink. Thus the M1 result is interpreted as switching between intercompartment transport dominance and representation/thermal-coupling dominance.

Direct `main_to_outside` absolute L1 share is <=0.002241 across the unique roots.

## M2 mechanism structure
Anchor:
- `ventilation_exchange`: +0.0229055 ppm;
- `photosynthesis_uptake`: -0.0107501 ppm;
- `organic_respiration`: -0.0121508 ppm;
- residual approximately machine zero.

Across 18 unique-root cells:
- `ventilation_exchange` dominant: 16;
- `photosynthesis_uptake` dominant: 1;
- `organic_respiration` dominant: 1;
- dominant switches from anchor: 2.

Both M2 switches occur at H=300 s and S_COOL_HUMID under stronger forcing:
- F_ROW6: photosynthesis uptake becomes dominant;
- F_ROW10: organic respiration becomes dominant, ventilation exchange changes sign, and cosine similarity to the anchor is negative.

## Main conclusion
EXP1.5 demonstrates that boundary location and boundary mechanism are separate conditional objects:

`C* = C*(H, x0, w)`

`M* = M*(H, x0, w)`

Boundary drift can occur with compositionally stable cancellation or with a resolved change in the dominant contribution. These are within-model structural results and are not a ranking of model families.

## M1 numerical-audit note
The original EXP1.3 0.005-ppm pair-closure gate does not remain a suitable single hard criterion over the expanded 300/900/1800-s root+edge surface because it mixes algebraic term completeness with adaptive quadrature accumulation. The final EXP1.5 M1 policy keeps the original one-shot state trajectory, retains machine-precision named-term-to-total-ODE closure, exposes the numerical closure remainder separately, and requires each root's top1-minus-top2 mechanism margin to exceed the observed numerical remainder before a dominant identity is accepted.

All 21 M1 unique-root cells satisfy this resolvability criterion.
