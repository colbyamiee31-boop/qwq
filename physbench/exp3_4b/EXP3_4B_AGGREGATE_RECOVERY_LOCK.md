# EXP3.4B aggregate recovery lock

Frozen after both matched-dose model jobs passed and before any successful EXP3.4B decision aggregate existed.

## Passed matched-dose artifacts

Source workflow run:
`37103104462`

M1:
- job: `111146366490`
- conclusion: PASS
- artifact: `exp3-4b-m1`
- artifact id: `11266119958`
- artifact SHA256: `cafadd5f61d5ff99d3126de7e51fe0252d0839dda9a35704cd81b97c1a71dd18`
- rows: 1060
- gate_pass: true

M2:
- job: `111146366468`
- conclusion: PASS
- artifact: `exp3-4b-m2`
- artifact id: `11266084966`
- artifact SHA256: `833933bb5056da8e6bf54ff37456c3e984bd34e34249e986b1fc03306e2e5e8c`
- rows: 1060
- gate_pass: true

Both artifacts use target SHA256:
`e07e64d8f8f7ac914a68b5cda73e0a76f5b9bd890a81473a42dd8ee8f6656292`.

## Aggregate failure

The aggregate job `111149170965` failed before producing any decision result because
`decision_analysis.py` expected both merged model tables to contain `common_width`, and therefore tried to access
`common_width_M1/common_width_M2`.

The locked M1 matched table does not store a `common_width` column; the locked M2 table does.
After the merge the valid column is therefore simply `common_width`.

This is an aggregate schema-access bug. It does not affect target inversion, matched commands, achieved physical doses,
T/AH trajectories, or model runtime gates.

## Recovery rule

The passed M1 and M2 artifacts above are frozen and MUST NOT be regenerated merely to repair the aggregate.

The corrected aggregate:
1. uses those exact run-37103104462 artifacts;
2. if both suffixed common-width columns exist, requires equality;
3. otherwise uses the single stored `common_width` only after independently verifying that it equals
   `max(target_dose)-min(target_dose)` for each frozen event/horizon/coordinate cell;
4. leaves all scientific definitions, cohorts, q levels, benefits, C(q)=q^2, lambda domains and paired comparison rules unchanged.

No successful EXP3.4B decision aggregate existed when this recovery lock was written.
