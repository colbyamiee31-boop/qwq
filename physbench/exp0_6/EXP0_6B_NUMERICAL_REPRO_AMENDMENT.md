# EXP0.6B numerical-reproducibility amendment

## Trigger

The first hosted EXP0.6B execution failed before any boundary or mechanism result was evaluated because the regenerated
72-h latent-state **raw float-byte SHA256** differed from the SHA256 recorded in the earlier EXP0.6A hosted run.

This occurred in both M1 and M2 at H00. The underlying model commits, forcing generator, history design, package versions,
and intervention definitions were unchanged.

A byte-level SHA of a floating-point state vector is therefore not used as a cross-run scientific equality test across
different hosted runners.

## What is changed

The previously recorded EXP0.6A state hashes remain provenance records and are never overwritten.

The EXP0.6B reproducibility gate is split into two distinct checks:

1. **Same-run serialization/restoration — exact hard gate**
   - every generated latent state is saved as float64;
   - it is reloaded within the same hosted execution;
   - exact element-wise equality must hold for every state.

2. **Cross-run scientific/numerical regression — hard gate**
   - the regenerated 72-h H900 roots for the nine frozen histories must reproduce the locked EXP0.6A roots within
     the already-frozen 0.002 ppm tolerance;
   - HNR H900 must reproduce the original anchor within 0.02 ppm;
   - all requested/applied actions, finite-state, bracket, and mechanism numerical gates remain unchanged.

## Additional forensic audit

The final EXP0.6B lock will compare the regenerated 72-h state vectors against the archived EXP0.6A vectors and report
the maximum absolute and scaled element-wise differences. This comparison is descriptive and cannot be used to select
or delete histories.

## What is not changed

No forcing, history factor, duration, model equation, parameter, solver configuration, action, CO2 search domain,
root-refinement rule, mechanism definition, or scientific interpretation threshold is changed.

This amendment addresses portability of a binary floating-point fingerprint only.
