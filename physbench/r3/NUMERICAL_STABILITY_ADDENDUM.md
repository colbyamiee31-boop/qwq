# R3 numerical-stability addendum — frozen before rerun

## Trigger
The first hosted R3 run used the native CSGtom explicit integration step of 30 s.

M1 completed successfully.

M2 produced 875 finite runtime-audit units and exactly one non-finite unit:
- track: D2
- event: 27
- stage: HV
- native action: 0.9

The frozen physical targets were reached exactly:
- effective air height = 7.375 m
- projected roof-aperture ratio = 0.17474999631859142 m2 m-2

The failed M2 artifact is retained:
- workflow run: 37148237389
- artifact ID: 11282534563
- artifact SHA256: a748b4f2c5179793630c4b7ef5b86e370a1c47bdccce70dddee6a7d1e85fd446

No R3 aggregate/scientific result was produced because the fail-closed finite-state gate stopped the workflow.

## Frozen numerical remedy
No event, action, geometry target, vent target, model parameter, forcing, state initialisation, or scientific metric is changed.

For M2 only:
- integration step is reduced from 30 s to **10 s** for both H and HV stages;
- output horizons remain exactly 15 and 30 min;
- physical dose remains the time integral of the native `Vent` flux;
- all H/HV events/actions are rerun at 10 s so stage comparison does not mix solver steps.

## Numerical-sensitivity audit
The complete finite H-stage output from the failed 30-s run is retained as a solver-sensitivity reference.
The final aggregation will compare 10-s H with 30-s H on the same rows for:
- T;
- AH;
- DV;
- DN.

This numerical comparison is diagnostic and cannot be used to tune the physical envelope.

## Runtime rule
The final M2 run still fails closed for any non-finite state or missing row.
Scientific improvement/deterioration remains irrelevant to execution PASS.
