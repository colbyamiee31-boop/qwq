# EXP3.4B aggregate schema replay note

## Trigger

Hosted run `37103104462` completed both matched-dose model jobs successfully:

- M1 artifact `11266119958`, digest
  `cafadd5f61d5ff99d3126de7e51fe0252d0839dda9a35704cd81b97c1a71dd18`;
- M2 artifact `11266084966`, digest
  `833933bb5056da8e6bf54ff37456c3e984bd34e34249e986b1fc03306e2e5e8c`.

The aggregate job then failed before decision analysis because the Amendment-3 M1 output schema omitted the redundant
`common_width` column, while M2 retained it. Pandas therefore produced a single `common_width` column rather than
the expected `common_width_M1/common_width_M2` pair.

No model rerun or scientific-rule change is required.

## Replay rule

The aggregate replay:

1. reuses the exact PASS M1/M2 artifacts from run `37103104462`;
2. reconstructs each cell's common width independently as
   `max(target_dose)-min(target_dose)` across the frozen q grid;
3. if a stored `common_width` is present, requires agreement with that reconstruction within `2e-12`;
4. uses the verified width for the cross-model dose-gap audit;
5. leaves every target, achieved dose, model outcome, benefit, cost, lambda domain, cohort, and decision metric unchanged.

The schema repair is implementation-only. The locked M1/M2 model artifacts are not regenerated for this replay.
