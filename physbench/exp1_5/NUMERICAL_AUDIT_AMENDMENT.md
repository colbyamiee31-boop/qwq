# EXP1.5 M1 numerical-audit amendment

## Why this amendment exists
The original EXP1.5 protocol inherited the EXP1.3 pair-closure thresholds (0.005 mg m^-3 and 0.005 ppm) as hard gates. That was appropriate for the original 900-s EXP1.3 point experiment, but the first EXP1.5 surface run showed that this criterion conflates two different questions when expanded to 300/900/1800 s and root+edge evaluations:

1. **algebraic completeness of the named native CO2 equation terms**, and
2. **numerical quadrature accumulation versus the final adaptive CVODES state**.

The first EXP1.5 audit reproduced the full EXP1.4 M1 topology (21 unique roots, 6 no-crossing cells, 0 multiple roots), reproduced the 900-s anchor to 4.73e-6 ppm, produced an augmented final state identical to the frozen native final state (max difference 0.0), and reconstructed the instantaneous named CO2 ODE to machine precision (maximum named-sum versus total-ODE mismatch 4.55e-13 mg m^-3). Nevertheless, integrated pair closure reached 0.256 ppm at the worst expanded root/edge point.

A subsequent diagnostic enabling CVODES quadrature error control changed the adaptive state trajectory (maximum final-state difference >4 state units) and was therefore rejected. A dense-output native-path diagnostic likewise changed integration scheduling and the final trajectory and was rejected. Neither diagnostic is used for final scientific attribution.

## Final M1 numerical policy for EXP1.5
The final analysis returns to the **original one-shot frozen state CVODES call plus the original EXP1.3 read-only quadrature expressions**. No state equation, model parameter, forcing, action, horizon, root, or physical flux contribution is modified.

Hard gates are separated by what they actually test:

### A. Algebraic / trajectory integrity — hard gates
- augmented attribution final state vs frozen native final state <= 1e-7 absolute state units;
- named native flux-rate sum vs the total native CO2 ODE <= 1e-10 mg m^-3;
- all states and contributions finite;
- EXP1.4 topology reproduced exactly;
- 900-s anchor root error <= 0.02 ppm;
- all detected roots remain inside their sign-changing brackets.

### B. Integrated numerical closure — reported audit, not a physical flux
For every attribution point define

`epsilon_num = actual HIGH-minus-LOW final CO2 - sum(reported physical/representation contributions)`.

The output records `numerical_closure_correction = epsilon_num` separately. It is never classified as a physical mechanism and is never included in the L1-normalized mechanism composition, cosine similarity, active-term sign pattern, or dominant-mechanism identity.

The original EXP1.3 0.005-ppm pair-closure result is still reported as a legacy comparison. EXP1.5 does not silently relax or overwrite it.

### C. Dominant-mechanism resolvability — exact conservative audit
For every unique-root cell, let `M1` and `M2` be the largest and second-largest absolute physical/representation contributions. The dominant identity is called **resolved under the observed numerical closure bound** only if

`|M1| - |M2| > |epsilon_num|`.

If this inequality fails, the cell is labeled `NUMERICALLY_UNRESOLVED_NEAR_TIE`; it cannot support a mechanism-switch claim.

This rule contains no fitted tolerance and does not redistribute numerical closure among physical terms. It asks whether the observed unassigned numerical remainder is large enough to erase the ordering of the two leading reported contributions.

## Scientific interpretation
Mechanism-switch counts in the final EXP1.5 report use only unique-root cells whose dominant identity is resolved by the criterion above. Edge audits remain descriptive and are never counted as reversal-mechanism switches.
