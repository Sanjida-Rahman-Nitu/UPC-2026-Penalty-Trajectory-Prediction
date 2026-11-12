# Synthetic penalty-kick dataset, v3 (UPC 2026 Problem B)

Regenerate: `python generate_dataset_v3.py --n 60000 --seed 2026` (bit-identical for the same seed).
Validate: `python validate_dataset_v3.py --dir .` → **65/65 checks pass** (`validation_report_v3.txt`).

## Files

| File | Rows | Use it for |
|---|---|---|
| `penalty_simple_v3.csv` | 60,000 | **Main file.** Simple physics (= the model you fit). Train / tune / main test. |
| `penalty_realistic_v3.csv` | 60,000 | **Same shots**, richer physics + wind + varying air density. Model-mismatch test (test split only). |
| `penalty_noise_v3.csv` | 70,000 | 10,000 test shots × 7 noise variants. Noise-sensitivity table. |
| `generator_settings_v3.json` | – | Every constant used. |
| `generate_dataset_v3.py`, `validate_dataset_v3.py` | – | Generator and independent validator. |

Train/val/test is the `split` column (70/15/15), identical in all files. Train and tune **only** on `simple` with `split` = train/val.

## Setup
Problem frame: penalty mark at origin, goal plane x = 11 m, posts y = ±3.66 m, bar z = 2.44 m.
Ball centre at (0, 0, 0.11) m at t = 0. Positions saved at t = 0.05, 0.10, 0.15, 0.20 s.
RK4, Δt = 0.5 ms; goal crossing and ground contact located within the step; flight stopped at x = 11 m or 1.5 s.

## Physics

**Simple model:** a = −g ẑ − k|v|v + s(ŵ × v); k = βC_D, s = βC_L v₀, β = ρA/(2m), ρ = 1.2; all constant.

**Realistic model:** relative air velocity v_rel = v − wind;
C_D = sigmoid(v_rel) (Goff, Asai & Hong 2014; Brazuca or Jabulani) + 0.55·Sp;
C_L = C_Lmax(1 − e^(−Sp/0.10)); Magnus = βC_L v_rel² in direction ŵ × v̂_rel;
spin decays ω = ω₀e^(−t/9.5 s); ρ random 1.14–1.29 kg/m³; wind 0–5 m/s, random horizontal direction.

**Bounce (both files, new in v3):** at ground contact (centre at 0.11 m) the velocity is mirrored (v_z → −v_z) and scaled by the coefficient of restitution COR ∈ U(0.56, 0.76). If the exit vertical speed is < 0.5 m/s the ball **rolls** (z fixed at 0.11 m; drag still acts; no rolling resistance).

## Priors

| Quantity | Distribution | Source |
|---|---|---|
| Speed | 50/50 mix N(26, 3) & N(32, 6) m/s, 18–40 | Club vs World Cup 2002 penalties |
| Azimuth | U(−22°, 22°) | Covers goal (±18.4°) + misses |
| Elevation | U(0°, 16°) (**v2: 3–16°**) | Low/rolling shots now included |
| Kick type | 70 % instep / 30 % curve | **Assumption** |
| Spin rate | instep N(3.25, 0.8), curve N(7.35, 1.0) rps, ≤ 10 | Tsukada & Sakurai; Asai 2007 |
| Spin axis | instep: back/top-dominant, side tilt N(0°, 20°), screw N(0, 0.1); curve: elevation N(62.6°, 10°), screw 0.38 ± 0.05 | Tsukada & Sakurai; Whiteside et al. |
| Top vs back, left vs right | 50/50 each | **Assumption** |
| Mass | U(0.410, 0.450) kg | FIFA Law 2 |
| C_D (no spin) | U(0.15, 0.20) | Asai 2007; Goff 2009/2012 |
| Spin drag | ΔC_D = 0.55·Sp | From Asai 2007 – **approximation** |
| C_Lmax | U(0.25, 0.35) | Carré 2002; Goff 2012 |
| COR (bounce) | U(0.56, 0.76) | Thomas (2024), angled rebound on turfgrass |
| Wind (realistic) | U(0, 5) m/s, direction U(0°, 360°) | **Assumption** (up to a gentle breeze) |
| Air density (realistic) | U(1.14, 1.29) kg/m³ | ≈ 0–35 °C at sea level |

## Columns

**Shot and spin** (both files): `shot_id`, `split`, `kick_type`, `speed_mps`, `azimuth_deg` (+ toward +y), `elevation_deg`, `spin_rps`, `spin_axis_x/y/z` (unit ŵ), `spin_axis_to_v_deg`, `topback` (top/back), `curve_dir` (left = toward +y, right = toward −y), `sp0`, `mass_kg`, `beta_per_m`, `cd0`, `clmax`, `cl0`, `cor`.
**Simple file only:** `cd_simple`, `k_per_m`, `s_per_s`.
**Realistic file only:** `ball_model`, `wind_speed_mps`, `wind_dir_deg` (direction the wind blows toward), `rho_kgm3`.

**Positions:** `{x,y,z}_tTTT_true` (exact), `{x,y,z}_tTTT_obs` (+ N(0, 5 mm), rounded to 0.01 m, like the problem). TTT = 005, 010, 015, 020. NaN after the ball crosses the goal plane.

**Labels:**
- `outcome` (v2-compatible, centre in the problem's rectangle): `on_target`, `wide`, `over`, `over_wide`, `timeout`.
- `outcome_frame` (**new**, ball radius + posts/bar up to 12 cm, IFAB Law 1): `goal` (whole ball clears the frame), `frame` (ball would hit a post/bar), `miss`, `timeout`.
- `y_goal`, `z_goal`, `t_goal_s`, `v_goal_mps`: where/when/how fast the centre crosses x = 11 m.
- `n_bounces`, `bounced_before_goal`, `t_bounce1_s`, `x_bounce1`, `y_bounce1`, `t_roll_start_s`; `t_ground_s` (= first ground contact, kept for v2 compatibility).

**Re-weighting for sensitivity analysis** (use as `weights=` in averages; mean ≈ 1):
- `w_kick5050` → 50/50 instep/curve instead of 70/30.
- `w_back70` → 70 % backspin instead of 50 %.
- `w_place_lower707` (on-target rows only) → 70.7 % of shots in the lower half of the goal (Danish top tier, 263 penalties). The unweighted data already give 69.1 %.

**Noise file:** `shot_id`, `noise_type`, `sigma_m`, observed positions (not rounded), labels.
`iid` σ = 2, 5, 10, 20 mm · `aniso_x3` (σx = 15 mm, σy = σz = 5 mm; depth error larger, **assumption**) · `jitter_1ms` (5 mm + 1 ms timing error per frame, **assumption**) · `outlier_5pct` (5 mm, 5 % of frames replaced by a 50 mm error, **assumption**).

## Changes from v2
1. **Bounce and roll** (COR from literature); elevation extended down to 0°. 29.6 % of shots bounce before the goal, 17.7 % roll.
2. **Frame-aware outcome** (`outcome_frame`): 9.6 % of shots would hit the post/bar.
3. **Wind and air density** in the realistic file. Median simple-vs-realistic goal-point difference: 2.5 cm in calm air, 13.6 cm with wind > 4 m/s.
4. **Seven noise variants** (v2: four iid levels).
5. **Re-weighting columns** for kick mix, back/top spin and literature shot placement.
6. **Spin class labels** (`topback`, `curve_dir`) and bounce bookkeeping.
7. All v2 column names kept with the same meaning.

## Validation summary
- Independent event-based re-simulation (scipy DOP853) of 1,200 shots incl. 528 bounced: 0 outcome or bounce-count mismatches; goal point within 0.04 mm; observation points within 0.16 mm; first-bounce time within 0.013 ms.
- Physical checks: no centre below 0.11 m; labels match coordinates; rolling balls cross at z = 0.11 m; spin labels have the right effect (backspin ends higher, "left" curves toward +y).
- Noise: SDs match each variant's specification.
- Literature benchmark (Goff, Asai & Hong 2014): t, z, v within 0.3 %, 1.3 cm and 0.6 %.
- Given scenario: 31 shots within 10 cm in observation space.
- Goal coverage: every 3×3 goal cell has ≥ 2,799 on-target shots. Lower cells are ~3× fuller than upper ones, matching real penalty placement (69.1 % vs 70.7 % in the lower half).

**Thresholds changed during validation:** the C_D range check uses 0.15–0.50 (Carré 2002 reports up to 0.5); the benchmark uses 1 % for time and speed and 2 cm for height; the goal-balance check (max/min < 3) was replaced by a coverage check (≥ 2,000 per cell) because the imbalance follows real placement data.

## Limitations (state in the paper)
- Bounce: mirror reflection with a single COR (angle ratio = 1); spin unchanged by the bounce; no rolling resistance.
- No knuckle-ball fluctuations, no spin-axis precession, no goalkeeper.
- Wind is steady and horizontal; no gusts.
- Kick mix, top/back split, ΔC_D slope, C_L(Sp) shape, wind range and noise variants are assumptions or approximations.
- Spin data come from small samples (2–4 players), not penalties specifically.
- Validation checks code correctness and literature consistency, **not** agreement with real penalty trajectories (no public 3D penalty trajectory data was found).

## References
Goff & Carré (2009) Am. J. Phys. 77, 1020 · Goff & Carré (2012) Procedia Eng. 34, 158 · Asai et al. (2007) Sports Eng. 10, 101 · Goff, Asai & Hong (2014) Proc. IMechE P 228, 188 · Carré et al. (2002) Sports Eng. 5, 193 · Tsukada & Sakurai (instep vs infront kick, conference abstract) · Whiteside, Alderson & Elliott (ISBS) · World Cup 2002 penalty analysis (Science and Football V) · Thomas, J. (2024) Soccer ball interactions on wet and dry turfgrasses at various mowing heights, MS thesis, Univ. of Tennessee · Executing penalty kicks in the Danish top tier (263 penalties, 2015/16–2019/20) · IFAB Laws of the Game, Laws 1 and 2.
