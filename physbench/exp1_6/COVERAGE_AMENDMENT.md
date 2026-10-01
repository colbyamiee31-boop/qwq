# Coverage amendment 2 — contribution-zero seeds

The uniform-grid run is retained as the parent coverage analysis. Absolute-value
gaps can contain narrow dips around a contribution's sign change, yielding two
transition roots within one 12.5 ppm grid interval. This is a search-coverage
issue, not a change to model equations, solver, root tolerances or the domain.

Before this additional search, freeze the following deterministic refinement:
for every existing horizon/stratum, add all detected zeros of native contributions
with sampled max magnitude >=1e-8 ppm and all existing reversal roots as scan
knots. Apply the same pairwise equality, third-term dominance and bisection rules.
The 1e-8 seed cutoff only skips roundoff/absent terms from auxiliary zero seeding;
all native terms still enter the primary dominant identity. Retain the original
transition candidates, merging by pair and the frozen 0.001 ppm dedup distance.

Reuse exact saved evaluations, copy their full evidence stream, append new calls,
and record the parent output hash. Report additions separately. This improves
coverage of narrow phases, but does not turn the finite search into a proof of
absence of tangencies or closed phase islands. No causal or smooth-manifold
claim is added.
