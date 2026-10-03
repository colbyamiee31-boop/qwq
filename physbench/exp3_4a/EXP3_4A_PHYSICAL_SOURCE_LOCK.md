# EXP3.4A — Physical ventilation-dose source lock

Frozen before EXP3.4A dose results are inspected.

## M1 — GreenLight-Gym2

Frozen source commit:
`2d3febb1ea002b24b452e32293e990beb78d3ce1`

Source:
`gl_gym/models/GreenLight/aux_states.py`

Native command:
- `u[3] = uVent`
- repository README describes this as roof ventilation opening.

Native external volumetric ventilation terms:
- `a[136] = fVentRoof` — total ventilation through roof, units `m3 m-2 s-1`;
- `a[137] = fVentSide` — total ventilation through side vents, units `m3 m-2 s-1`;
- `a[145] = fVentForced` — forced ventilation, zero in this frozen model.

The controlled roof-vent component is generated from:
- `a[123] = u[3] * p[55]` (roof aperture);
- `a[132]` (roof natural-ventilation rate);
- leakage `a[135]` is included in the external exchange through the frozen leakage split.

Therefore EXP3.4A defines the M1 whole-greenhouse external specific volumetric exchange flux as:

`q_ext_M1 = a[136] + a[137] + a[145]`

with units `m3 m-2 s-1`.

Frozen geometry/capacity parameters:
- `p[46] = aFlr = 144 m2`;
- `p[49] = hGh = 6.2 m`;
- `p[55] = aRoof = 52.2 m2`;
- `p[204] = cLeakTop = 0.9`.

The total greenhouse air-volume-per-floor-area coordinate used for the air-volume-equivalent sensitivity is
`H_eff_M1 = p[49] = 6.2 m`.

## M2 — CSGtom

Frozen source commit:
`bea8c3b0a1324162a4b5487db578aa674c8b587c`

Source:
`functions/csg_fun.py::ctl_csg1`

Native command:
- `u_vent`.

The routine computes:
- top/bottom vent flow;
- top vent flow;
- side vent flow;
- passive leakage;
and returns

`Vent = Leaching_air/(1+u_blanket) + (1-u_blanket)*u_vent*(...)`

with source-commented units `m3 m-2 s-1`.

Under the frozen EXP3 configuration:
- `u_blanket = 0`;
- `u_venttop = 1`;
- `u_ventside = 0`;
- `u_venttopbot = 0`.

Therefore EXP3.4A defines:

`q_ext_M2 = Vent`

with units `m3 m-2 s-1`.

Frozen geometry source:
`functions/csg_shape.py`

The air-volume-equivalent sensitivity uses:

`H_eff_M2 = D.Vair / D.area_floor`

with units m.

## Common physical dose coordinates

Primary:
`D_V(u,h) = integral_0^h q_ext(t;u) dt`

units: `m3 m-2`.

Secondary:
`D_N(u,h) = integral_0^h q_ext(t;u)/H_eff dt`

dimensionless air-volume equivalents.

Average ACH is reported as:
`ACH_bar = 3600 * D_N / h`.

These are physical model outputs. They are not inferred from command labels and are not calibrated from the observed D2 controller-status field.
