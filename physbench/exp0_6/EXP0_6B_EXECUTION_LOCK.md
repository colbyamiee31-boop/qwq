# PhysBench-GH EXP0.6B — Execution Lock

This file clarifies the already-frozen EXP0.6 protocol before EXP0.6B scientific results are inspected.

## Frozen subset
Use H00 plus the eight Resolution-IV corner histories:
H00, H01, H04, H06, H07, H10, H11, H13, H16.

HNR remains a reconstruction/reference condition and is not part of the history-derived distribution.

## 72-h horizon robustness
For each of the 9 history-derived 72-h latent states, run the frozen CO2 boundary procedure at:
- 300 s
- 900 s
- 1800 s

No search-domain expansion is allowed.

The regenerated 72-h native-state SHA256 values and the 900-s roots must reproduce the locked EXP0.6A values in
`exp0_6a_72h_state_hashes.json`.

## 168-h slow-memory extension
For the same 9 history IDs, regenerate model-native latent states using 168 h of the same deterministic prehistory design.
At 168 h, the primary slow-memory endpoint is the H900 CO2 boundary.

The 168-h result may move, overlap, lose roots, or alter topology without causing a scientific failure.

## Mechanism robustness
At every unique H900 root:
- run the frozen EXP1.5 native mechanism accounting;
- do this for both 72-h and 168-h histories;
- compare each history-derived mechanism with the HNR H900 mechanism from the same model;
- report dominant term, dominance share, normalized signed composition, active-term sign pattern, cosine similarity, and L1 distance.

For M1, follow the EXP1.5 numerical-audit amendment:
- native/augmented final-state equality and named instantaneous ODE-term closure are hard numerical gates;
- integrated numerical closure is reported separately;
- dominant identity is considered resolved only when the gap between the largest and second-largest physical/representation terms exceeds the absolute numerical closure correction.

For M2, retain the frozen 30-s Euler decomposition and closure gates from EXP1.5.

Mechanism switching is a scientific result, not a CI failure.

## HNR references
HNR is run at 300, 900 and 1800 s for root reconstruction context.
The 900-s HNR root must reproduce:
- M1: 438.5370684042573 ppm ± 0.02 ppm
- M2: 358.2789194211364 ppm ± 0.02 ppm

HNR H900 mechanism vectors are used only as within-model anchors.

## Explicit non-goals
EXP0.6B does not:
- match physical airflow dose between models;
- change utility scaling;
- compare raw mechanism magnitudes across model families;
- establish empirical ground truth;
- rank either model as physically correct.

