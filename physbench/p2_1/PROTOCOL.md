# PhysBench-GH P2.1 — ALAR NATIVE-DOMAIN PROCESS-FINGERPRINT MODEL REPLAY

## Protocol-freeze status
This protocol is frozen before any P2.1 M1/M2 outcome is inspected. It inherits P2.0 without changing any cohort, threshold, bin, greenhouse identity, or model version.

## Frozen ancestry
- Authoritative manuscript: `01_PhysBench_GH_CEA_MANUSCRIPT_RESCUE_v0_6_FINAL.docx`.
- P2.0 gate: `PASS FOR NATIVE-DOMAIN PROCESS FINGERPRINTING`.
- P2.0 frozen atlas: 1,195 states, SHA256 `69d8a888d9cb8390e53c21c9af24c364442a5e933f21d3d4a025b87453282555`.
- Canonical aligned Alar data SHA256 `d1f29af22ea51c209f486710064ddd36af821d52642ef885aa2431f6637d6461`.
- R4 final head: `6e687ba1b55c38d7fc184c94006a1d9fba10aa4d`.
- M1 GreenLight-Gym2: `2d3febb1ea002b24b452e32293e990beb78d3ce1`.
- M2 CSGtom: `bea8c3b0a1324162a4b5487db578aa674c8b587c`.

## Scientific role and firewall
Alar has no actuator log. P2.1 is therefore **not** realised-action validation and must not be interpreted as causal actuator-effect evidence. It is an Alar-native-domain process-structure replay designed to ask whether the frozen M1/M2 process behaviour is compatible with observed thermal/moisture fingerprints, and to localise the residual temperature-direction disagreement remaining after R3/R4.

No P2.1 result may be used to:
- infer actual Alar vent/blanket positions;
- relabel effective retention as ACH;
- change P2.0 thresholds, strata, atlas membership or greenhouse weighting;
- calibrate model-specific parameters to outcomes;
- create a global M1-versus-M2 winner score.

## Frozen 1,195-state cohort
Every P2.0 atlas state is replayed. No state may be added, removed or substituted after model outcomes are inspected. If a model runtime fails, the atlas row remains in the runtime audit with its atlas ID and failure status; scientific summaries use only predeclared finite-output availability masks and report failure counts explicitly.

All metrics are first calculated separately for G1/G2/G3 and only then macro-summarised with equal greenhouse weight.

## Common observed forcing and time grid
Each state is driven for 6 h from its atlas timestamp.

Model forcing is constructed from the canonical aligned Alar/NASA record only:
- global radiation `Gout_W_m2`;
- outdoor temperature `T2M_C`;
- outdoor RH `RH2M_pct` converted to vapour pressure;
- 2-m wind `WS2M_m_s`;
- precipitation retained only for the F4 observation mask.

Both models receive the same 15-min forcing nodes from t=0 to 6 h. Nodes that fall between canonical 10-min timestamps are deterministically linearly interpolated from the adjacent NASA-aligned forcing values. This does not upgrade forcing resolution: scientific forcing-response endpoints remain >=1 h, exactly as frozen in P2.0.

External closures inherited from the prior PhysBench replay are fixed before outcomes:
- outdoor CO2 = 415 ppm;
- sky temperature = outdoor temperature;
- deep-soil boundary = 18 degC.

## Observable-state initialization
At t=0 both models receive the same measured aggregate indoor state:
- Tair = Tin;
- indoor vapour pressure from Tin/RHin;
- CO2 = observed Alar CO2 only when 200 <= CO2 <= 2000 ppm; otherwise 415 ppm.

### I0 — neutral thermal closure (primary)
The primary replay suppresses arbitrary unobserved initial thermal gradients without fitting them:
- all non-soil greenhouse thermal states are initialized to Tin where a direct temperature state exists;
- five soil states are initialized by a deterministic linear gradient from Tin at the shallowest layer to 18 degC at the deepest layer;
- canopy 24-h temperature state is initialized to Tin where present;
- crop biomass/carbon/cumulative-development states retain each model's native defaults.

This is a deterministic closure, not a calibration.

### I1 — prior native-latent closure (sensitivity)
To test dependence on hidden-state initialization, the prior R1.2-style initialization is repeated only for the closed-vent/no-enclosure arm:
- overwrite observable Tair, VP, CO2 and Tcan=Tin;
- retain all other native reset/default latent states.

I1 cannot replace I0 as the primary replay.

## Actuator-uncertainty design
Because Alar actuator states are unknown, actual operation is never imputed. Instead, fixed standardized action sensitivities are replayed:

Primary neutral-initialization arms:
- `V0E0`: vent command 0.0, enclosure off;
- `V50E0`: vent command 0.5, enclosure off;
- `V100E0`: vent command 1.0, enclosure off;
- `V0E1`: vent command 0.0, maximum native thermal enclosure.

Initialization sensitivity:
- `I1_V0E0`: native-latent initialization, vent 0.0, enclosure off.

Other controllable inputs are fixed off (heating, CO2 injection, lighting, blackout where applicable).

`V0E1` is **within-model enclosure sensitivity only**. A GreenLight thermal screen and a CSGtom solar-greenhouse blanket are not asserted to be equivalent actuators, so their E1 magnitudes are not used for direct cross-model ranking.

## Native model topology is retained
P2.1 intentionally does **not** import the R3/R4 common-envelope geometry into Alar. It asks a native-domain fairness question and therefore retains each frozen model's native geometry, capacities, conductances and internal topology.

### M1
GreenLight-Gym2 is executed with its frozen native model and six-control interface. The thermal-screen command is `uThScr`; vent is `uVent`.

### M2
CSGtom is executed at its frozen commit with native geometry/thermal structure. Ventilation uses the native top-vent pathway; blanket command is `u_blanket`. Primary integration step = 30 s. If and only if a run fails because state/output becomes non-finite, that exact atlas-arm is retried at 10 s and tagged `NUMERICAL_RETRY_10S`. This is a predeclared numerical fallback and never depends on scientific outcome.

## Representation firewall for the three measured heights
Observed F1/F5 use SHT30 measurements at 0.6, 1.8 and 3.0 m. A direct sensor-height-to-state mapping is not semantically valid for either model:
- M2 has one indoor-air temperature state (`T_air`) and therefore cannot resolve vertical air stratification.
- M1 has `tAir` and `tTop`, but `tTop` is the upper compartment of the GreenLight screen/top-air topology, not the observed 3.0-m sensor height.

Therefore:
- no vertical-temperature RMSE or direct 0.6/1.8/3.0-m state score is computed;
- M2 receives `NOT_RESOLVED_1ZONE` for direct vertical-stratification representation;
- M1 `tTop-tAir` is retained only as a **secondary topology diagnostic**, not as observed `T3.0-T0.6`.

## Frozen endpoints

### P2.1-A — F2 thermal-envelope compatibility
At h={1,3,6} h calculate model `DeltaTio = Tair_model - Tout` and observational `DeltaTio` on the precomputed observation-availability mask. Summarise median and IQR by greenhouse, season, day/night and action arm. Report model-observation median differences, never a pooled global winner.

### P2.1-B — F3 solar-memory compatibility
For daytime atlas states with available observed future Tin, use h=1..6 h. Within each greenhouse, estimate Spearman association between initial Gout and future DeltaTio for observation and each model/arm. The lag of maximum association uses the frozen 0..6-h logic; if multiple lags tie numerically, take the smaller lag. Report model-observation lag difference and rho profiles.

### P2.1-C — F4 night effective-retention compatibility
Use only precomputed observation masks satisfying the P2.0 F4 rules: Gout<20 throughout, precipitation<0.1 mm in every hourly block, initial |DeltaTio|>=2 degC, same segment and complete observations. At h={1,3,6} h compute Rh and, only under the frozen validity rule, k_h. Compare greenhouse-specific medians/IQRs and macro differences. Never call k_h ACH.

### P2.1-D — F6 heat-moisture coupling
Convert model VP and T to absolute humidity with one fixed psychrometric formula. For h=1 h, estimate within-greenhouse Spearman association between DeltaT and DeltaAH for observation and each model/arm, then macro-summary. Vertical AH span remains observational only because direct model height mapping is not valid.

### P2.1-E — F1/F5 structural resolvability and localization
F1/F5 observed vertical stratification is not forced into a false model-state mapping. Instead:
- report model representational status as above;
- for M1 only, report `tTop-tAir` as a secondary topology diagnostic;
- use frozen observed ST and wind bins as axes for localising cross-model temperature-direction disagreement.

### P2.1-F — primary bridge to the R4 temperature-direction residual
For each atlas state, arm and h={1,3,6} h:
- compute sign of `Tair_model(h)-Tin(t0)` with a fixed deadband of exactly 0 degC (mathematical sign; no new empirical threshold);
- define M1/M2 disagreement when signs differ;
- localise disagreement on the already frozen P2.0 axes: ST bin (primary), day/night x ST, DeltaTio bin, wind bin and season x ST.

Also calculate each model's direction agreement with observed `Tin(h)-Tin(t0)` where that observation exists, but interpret this only as process compatibility because unknown actuation remains a confounder.

The key P2.1 inference is whether disagreement is concentrated in specific empirically observed solar-greenhouse process regimes or remains diffuse. No threshold is chosen after results.

## Statistical inference
- Greenhouse is the first-level independent unit for headline summaries.
- Bootstrap seed = 20261004.
- 2,000 bootstrap resamples for reported uncertainty intervals.
- Within-greenhouse state resampling is followed by equal-weight macro aggregation across G1/G2/G3.
- No raw-row-weighted pooled headline.
- A model may be called "more compatible" for a specific fingerprint only if the predeclared paired macro contrast has a 95% bootstrap interval excluding zero. Otherwise report `NOT_DISTINGUISHED`.
- No cross-fingerprint summed score and no global ranking.

## Runtime gates
Fail-closed engineering gates are limited to:
- upstream atlas/aligned-data identity mismatch;
- missing/duplicate atlas IDs;
- forcing-node construction failure;
- initial observable-state mismatch;
- requested/applied action mismatch where the API exposes applied controls;
- non-finite output after the predeclared M2 numerical retry;
- output row-count/arm/horizon mismatch;
- greenhouse/segment leakage in observation masks.

Scientific compatibility or incompatibility is never a runtime PASS criterion.
