# SPLIT-1.1 — R1.2 ATTRITION AUDIT + DECISION-LAYER COMPRESSION
## PROTOCOL LOCK — BEFORE ATTRITION OUTCOME INSPECTION

Status: FROZEN before executing the new attrition/representativeness analysis.

### Scope
This stage does not alter R1.1, R1.2, EXP3, R2, R3, or R4 scientific definitions, thresholds, cohorts, model commits, action mappings, or outcome endpoints. It adds one outcome-blind diagnostic of the 385 -> 97 R1.2 matching attrition and then compresses the manuscript's secondary decision layer.

### Frozen upstream inputs
- R1.2 execution commit: 5e3520bad4c31d73b2a93edc488253dfc127865c
- Primary E5 event rows: the 385 R1.1 events already hard-coded in physbench/r1_2/prepare_inputs.py
- Strict E5 sensitivity rows: the 90 R1.1 events already hard-coded in the same script
- CTIFL raw archive MD5: d0e4486fa1041fac5e6e47673b6c95d3
- R1.2 primary matched cohort definition: unchanged; at least 3 controls, up to 5, exact strata/calipers and reuse cap exactly as frozen in R1.2.

### Scientific question
Does the outcome-blind R1.2 matching step select a visibly restricted subset of the 385 E5-qualified realised roof-vent events, and what aspects of the frozen matching rules account for most non-matchability?

### No-outcome rule
The attrition audit must not use:
- post-event indoor response;
- model response;
- R1.2 concordance, kappa or magnitude-association outcomes;
- R2-R4 outcomes.
Only event-time or pre-event variables and the already-frozen matching/control-eligibility rules may be used.

### Primary attrition diagnostics
1. Reproduce the frozen R1.2 match assignments exactly:
   - primary: 97/385;
   - strict: 36/90.
   Failure to reproduce these counts is an execution FAIL.
2. For every primary E5 event, record:
   - exact-stratum control count;
   - final control count after all calipers but before reuse capping;
   - number assigned after the frozen reuse cap;
   - final matched/unmatched status.
3. Partition unmatched events into mutually exclusive frozen categories:
   - fewer than 3 controls after exact strata;
   - at least 3 after exact strata but fewer than 3 after all calipers;
   - at least 3 after all calipers but fewer than 3 after reuse-cap competition;
   - other/integrity anomaly.
4. Report sequential control-count retention under the frozen calipers in the exact protocol order. This is diagnostic and order-dependent; it must be labelled as such.
5. Compare matched (n=97) versus unmatched (n=288) primary E5 events using outcome-blind event/pre-event characteristics:
   - opening/closing direction;
   - absolute vent-position change;
   - pre-event mean vent opening;
   - event indoor T, RH and AH;
   - outdoor T and humidity ratio;
   - measured GHI and day/night;
   - wind speed and circular wind direction;
   - T and AH pretrends;
   - thermal/shading screen modes;
   - heating pump-status tuple;
   - month and local-clock time.
6. Continuous-variable balance is summarized by matched/unmatched mean, median, pooled-SD standardized mean difference (SMD) and robust median shift scaled by pooled IQR where defined. Categorical variables use proportions and absolute percentage-point differences. No null-hypothesis p-value will be used as a pass/fail criterion.
7. Descriptive flags are frozen before execution:
   - |SMD| < 0.10: small;
   - 0.10 <= |SMD| < 0.20: modest;
   - |SMD| >= 0.20: material descriptive imbalance.
These are interpretation aids, not statistical significance thresholds.

### Secondary diagnostics
- Repeat the match-count/failure-category summary for the frozen strict 90-event cohort.
- Report candidate-control count distributions.
- Do not create new matched cohorts, relax calipers, reweight events, or replace the primary R1.2 analysis.

### Manuscript actions after the audit
Regardless of audit direction:
- Abstract will identify CTIFL as a common Venlo external cohort and state that the paired M1-minus-M2 temperature comparison is inconclusive for global ranking.
- “External physical anchor/reference” will be narrowed to “semi-empirical external ventilation reference” where measurement status could be over-read.
- The ~60% residual discussion will explicitly state that the present experiments do not distinguish or rank the remaining candidate mechanisms.
- EXP3 will remain a secondary bridge analysis, not an independent primary research question.
- Main-text Figure 6 (decision-consequence bridge) will move to Supplementary Material; the structural-isolation figure will become main-text Figure 6.
- Main-text decision Results will be compressed without changing frozen numbers.

### Prohibited post-hoc responses
- no matching-threshold changes;
- no new subgroup promoted to primary;
- no exclusion of inconvenient unmatched events;
- no claim that unmatched events are comparable unless supported by this audit;
- no mechanistic ranking of the residual candidates from this attrition analysis.
