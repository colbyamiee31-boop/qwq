# EXP1.3–EXP1.5 independent historical-audit cross-check

After the original EXP1.6 local delivery, the archived ChatGPT evidence packages
for EXP1.3, EXP1.4 and EXP1.5 were made available and compared independently with
the EXP1.6 embedded legacy reruns.

## Frozen inputs

The raw-byte hashes differed across Windows/Linux copies because of CRLF/LF line
endings. After LF normalization, both inherited inputs were exactly identical:

- `benchmark_forcing.csv`: `770cd596375cd18cd889decbb3d8520aa52f28bd6176fb898e7ddd242a2fcca0`
- `matched_initial_state.json`: `c72cd0fd189569ada6b8e17e8fecb85e203073893d0c8ea46910a78a9ba9a5d0`

## EXP1.3 / EXP1.4 anchor cross-check

Archived EXP1.3 refined 900-s roots:

- M1: `438.5370684042573 ppm`
- M2: `358.2789194211364 ppm`

EXP1.4/EXP1.5/EXP1.6 rerun anchors:

- M1: `438.537073135376 ppm`, absolute difference `4.731118679046631e-6 ppm`
- M2: `358.2788944244385 ppm`, absolute difference `2.4996697902679443e-5 ppm`

These are negligible solver-level differences and do not change topology or any
mechanism classification.

## EXP1.5 table cross-check

M1:
- root-mechanism rows: 21 vs 21;
- `condition_id`, `dominant_term`, `active_sign_pattern`, and
  `mechanism_classification` all identical;
- maximum numeric difference across common root-mechanism numeric fields:
  `4.085584580024171e-8`;
- maximum numeric difference in the long term-contribution table:
  `5.127254983969265e-5 ppm`.

M2:
- root-mechanism rows: 18 vs 18;
- `condition_id`, `dominant_term`, and `active_sign_pattern` all identical;
- maximum numeric root-mechanism difference: `1.1013412404281553e-13`;
- maximum numeric long-table term-contribution difference:
  `5.5011550870176507e-14`.

## Conclusion

Independent historical evidence comparison is complete. The frozen inputs match
after line-ending normalization; archived anchors, topology, and mechanism
identities are reproduced. Observed numerical differences are at solver/roundoff
scale and do not alter the EXP1.6 scientific interpretation.
