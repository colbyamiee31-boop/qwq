# Root identifiability audit

The inherited root criterion is bracket width <=1e-4 ppm OR residual <=1e-7 ppm.
A small bracket is a location statement and need not imply a small equality
residual. Before polishing candidates, freeze this additional audit:

- Re-evaluate saved brackets under the identical native model/attribution.
- Bisect to residual <=1e-7 ppm or width <=1e-8 ppm, maximum 60 iterations.
- Retain the full iteration history, both terminal gap values, actual R,
  numerical closure, and third-term qualification.
- If the equality residual remains large, label UNRESOLVED_EQUALITY. Do not
  manufacture a crossing by linearly interpolating the discontinuity.
- A residual-resolved equality is still only a numerical equality of the frozen
  reported contributions. It does not establish integration accuracy, smoothness,
  causality, or robust mechanism ordering at the tie itself.

This changes no model equation, parameter, native solver tolerance, timestep,
quadrature expression, phase threshold or primary scientific domain. It is a
separately recorded numerical audit, not a replacement of the inherited root gate.
