# EXP1.6 — Mechanism Transition Surface / Phase Map

Execution protocol v1, frozen before the first EXP1.6 grid evaluation. Source base:
`92ad719b08b4412afd95919481065a8ee8f5c8e1`. Numeric design is in `config.json`.
The preliminary workspace draft is superseded by this source-verified protocol.

## Inheritance and domain

M1 commit `2d3febb1ea002b24b452e32293e990beb78d3ce1`; M2 commit
`bea8c3b0a1324162a4b5487db578aa674c8b587c`. All EXP0.5/1.3/1.4/1.5 tracked
inputs are protected by LF-normalized SHA-256 in `frozen_inputs.json`; raw hashes
are also recorded. Model tracked files are checked before and after each run.

Reuse the exact EXP1.5 functions via an AST prefix loader, stopping before its
top-level experiment loop. No state, parameter, flux, action or conversion
formula is changed. The complete legacy scripts are run separately as regression
gates. The loader must also reproduce their saved root contribution vectors.

The three frozen states, forcing rows 0/6/10, LOW=0.1, HIGH=0.9 and CO2 domain
250–600 ppm are unchanged. All original 300/900/1800 s cells remain hard
regression anchors. New horizons are 300:60:1800 s (26 levels). These are added
EXP1.6 evaluations, not replacements of the original factorial design. States
and forcing rows remain categorical strata; no interpolation between them.

M1 uses the identical one-shot CVODES `abstol=reltol=1e-4`, max_num_steps=150000
and read-only quadratures. No dense-output rescheduling or quadrature error
control. M2 preserves 30 s Euler and minute-resolution timestamps; only integer
minute horizons are admissible. M1 horizon refinement stops at one second,
consistent with the inherited integer-horizon adapter. This study estimates
sections of a transition set; it cannot certify a smooth M2 continuous-horizon
manifold under the frozen implementation.

## Conditions

R = final CO2_HIGH − final CO2_LOW. Contributions a_j and all native terms are
exactly those in EXP1.5, including temperature_conversion in M1 and residual in
M2. epsilon = R − sum(a_j); M1 epsilon is excluded from the mechanism vector.

For every pair i,j, g_ij=|a_i|−|a_j|. A dominant transition requires g_ij=0 AND
both terms at the largest absolute magnitude among all native terms. Pair
equalities below another term are rejected. Multiway ties are retained.
Reversal is R=0. An intersection must satisfy both equations and the dominance
inequality; opposite phase labels at two old root cells alone are insufficient.

M1 phase resolution follows the final EXP1.5 amendment: largest magnitude minus
second-largest magnitude > |epsilon|. Call this **resolved under observed closure
screen**, not a certified per-term error bound: compensating integration errors
can cancel in the closure sum. M2 additionally screens gaps <=1e-8 ppm, its
inherited closure tolerance. Zero L1 or screened ties are UNRESOLVED. No closure
correction is redistributed to force equality or a desired phase.

## Search and coverage

1. Run legacy EXP1.5, require its final run_pass, exact expected topology and
   anchor error <=0.02 ppm. Re-evaluate all saved roots with the prefix loader,
   require every contribution and R within 1e-7 ppm of the just-rerun evidence.
   User-supplied historical audit files are an additional, separately reported
   provenance comparison and are never silently substituted by local reruns.
2. Evaluate both arms and attribution at every C=250:12.5:600 ppm for each new
   horizon/stratum. The 25 ppm subset is the coarse coverage check. Save all
   evaluated endpoints/flux traces, including search iterations.
3. Scan R and all native term pairs. Skip pairs whose magnitudes are everywhere
   below the leading envelope on an interval; check its midpoint before this
   screening. Refine sign-changing candidates and exact grid zeros with
   bisection: width <=1e-4 ppm or residual <=1e-7 ppm, maximum 50 iterations.
   Report WIDTH_ONLY separately from residual convergence. A width-only root
   does not establish a continuous zero. Deduplicate within 1e-3 ppm.
4. Record midpoint local minima in |R| and in leading pair gaps as tangency or
   unresolved-feature candidates, even without a sign change. These are not
   reported as solved roots. A coarse/fine count mismatch is a coverage warning,
   not a reason to retune or enlarge the domain.
5. Check transition left/right phase identity at ±0.1 ppm, clipped to the
   declared domain. Save R and signed contributions at every transition.
6. For each reversal branch available at adjacent horizons, test all pair gaps
   on that branch. Refine leading-pair sign changes along the reversal branch
   to 1 s (M1) or retain the 60 s bracket (M2). All branch-selection decisions
   and function values are logged. Branch-count changes or absent roots are
   coverage discontinuities, not interpolated intersections. Multiple roots
   are retained; ambiguous branch association is flagged rather than guessed.
7. Save fixed-horizon signed C_transition−C_reversal for every root pairing.
   Absence of detected intersections is only sampled separation. No global
   separation, completeness, transversality or smoothness certificate is claimed.

At leading opposite-sign ties, the two terms cancel and R equals the remaining
terms plus epsilon. Record that identity as a useful mechanism check; other
terms and numerical closure can keep transition and reversal apart.

## Integrity, stability, failure semantics

M1 state identity <=1e-7 and named/total quadrature closure <=1e-10; M2 applied
action error zero, native vs attributed CO2 and Euler/pair closure <=1e-8;
all outputs finite. Legacy M1 0.005 ppm closure remains an audit, not a new gate.

No solver retuning or unapproved step-size convergence experiment is performed.
Stability evidence consists of coarse/fine coverage, local phase perturbation,
root residual/width, observed closure screens and direct native-path checks.
This does not establish integration convergence or physical validity.

At most 60000 distinct paired evaluations/model. Any budget exhaustion, hard
gate failure or exception is written as FAILED with completed evidence retained;
it cannot produce a PASS. Numerical unresolved regions are valid scientific
outputs and are not erased to obtain a pass. PASS_EXECUTION means this declared
finite-grid analysis completed, not that a smooth manifold was certified.

## Evidence and deliverables

Append-only compressed JSONL contains every unique paired evaluation, complete
contribution vector and M1 one-shot initial/final states or M2 Euler flux records.
Evaluation IDs connect tables, bisection histories, regression and intersection
brackets. Failures retain traceback. Config/source/model/package/environment
hashes are recorded before execution and output hashes after completion.

Tables: phase grid, reversal roots, transition roots, signed separations,
intersection brackets, coverage/stability diagnostics and regression results.
Figures: nine H×C panels/model with native phase colors, unresolved region,
reversal points, accepted transition points and intersection intervals. Point
markers prevent plot interpolation from implying a certified curve. Summary
must report numerical qualification and missing external evidence explicitly.

## CI

Fast analytic tests cover search rejection, multiple roots, third-term dominance,
ties, nonfinite values and iteration limits. Separate real-model jobs reproduce
legacy gates then execute EXP1.6, upload evidence even on failure, and distinguish
workflow creation from an actual hosted CI run. Model installations use pinned
commits; exact installed distributions are exported. No push/deployment occurs
as part of a local run.
