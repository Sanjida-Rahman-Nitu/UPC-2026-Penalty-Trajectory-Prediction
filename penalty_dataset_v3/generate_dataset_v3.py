"""
Synthetic penalty-kick dataset generator, v3 (UPC 2026 Problem B).

New in v3 (all v2 columns kept with the same names and meaning):
  * Bounce and roll on the pitch (COR 0.56-0.76, Thomas 2024), elevation now 0-16 deg.
  * Frame-aware outcome (`outcome_frame`): ball radius and post/bar thickness (<= 12 cm, Law 1).
  * Realistic file: random wind (0-5 m/s, horizontal) and air density (1.14-1.29 kg/m^3).
  * Noise file with 7 noise variants (iid sigma sweep, anisotropic, timing jitter, outliers).
  * Prior re-weighting columns for sensitivity analysis (kick mix, back/top spin, placement).
  * Spin class labels (top/back, left/right) and bounce bookkeeping.

Physics levels (same latent shots in both):
  simple    : a = -g z - k|v|v + s (w x v), k and s constant; no wind; rho = 1.2.
  realistic : C_D(v_rel) sigmoid (Goff, Asai & Hong 2014) + 0.55 Sp; C_L(Sp) saturating;
              Magnus ~ v_rel^2; spin decay; wind; random rho.
Bounce (both levels, same rule): speed ratio COR, mirror reflection (angle ratio 1, assumption);
if vertical exit speed < 0.5 m/s the ball rolls (z fixed at 0.11 m, no rolling resistance).

Run:  python generate_dataset_v3.py --n 60000 --seed 2026 --out .
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

G = 9.81
RHO0 = 1.2
R_BALL = 0.11
AREA = np.pi*R_BALL**2
X_GOAL, Y_POST, Z_BAR = 11.0, 3.66, 2.44
POST_W = 0.12                       # max post/bar width (IFAB Law 1)
Z0 = 0.11
OBS_TIMES = (0.05, 0.10, 0.15, 0.20)
DT = 5e-4
T_MAX = 1.5
SPIN_TAU = 9.5
SIGMA_MAIN = 0.005
ROUND_DP = 2
ROLL_VZ = 0.5                       # m/s, below this exit vertical speed the ball rolls
CD_FIT = {"brazuca": (0.187118, 0.260996, 12.9084, 0.717838),
          "jabulani": (0.163643, 0.306765, 17.3015, 1.61952)}
DCD_DSP, SP_C = 0.55, 0.10


def cl_of_sp(sp, clmax):
    return clmax*(1.0 - np.exp(-sp/SP_C))


def unit(v):
    return v/np.linalg.norm(v, axis=-1, keepdims=True)


def truncnorm(rng, mean, sd, lo, hi, n):
    mean = np.broadcast_to(mean, (n,)).astype(float); sd = np.broadcast_to(sd, (n,)).astype(float)
    out = rng.normal(mean, sd)
    bad = (out < lo) | (out > hi)
    while bad.any():
        out[bad] = rng.normal(mean[bad], sd[bad]); bad = (out < lo) | (out > hi)
    return out


def sample_latent(n, rng):
    p = {}
    curve = rng.random(n) < 0.30
    p["kick_type"] = np.where(curve, "curve", "instep")
    wc = rng.random(n) < 0.5
    p["speed_mps"] = truncnorm(rng, np.where(wc, 32.0, 26.0), np.where(wc, 6.0, 3.0), 18.0, 40.0, n)
    p["azimuth_deg"] = rng.uniform(-22.0, 22.0, n)
    p["elevation_deg"] = rng.uniform(0.0, 16.0, n)
    p["spin_rps"] = np.where(curve, truncnorm(rng, 7.35, 1.0, 0.0, 10.0, n),
                             truncnorm(rng, 3.25, 0.8, 0.0, 10.0, n))
    p["mass_kg"] = rng.uniform(0.410, 0.450, n)
    p["cd0"] = rng.uniform(0.15, 0.20, n)
    p["clmax"] = rng.uniform(0.25, 0.35, n)
    p["cor"] = rng.uniform(0.56, 0.76, n)                       # Thomas (2024) angled rebound COR
    # realistic-only environment
    p["ball_model"] = np.where(rng.random(n) < 0.5, "brazuca", "jabulani")
    p["wind_speed_mps"] = rng.uniform(0.0, 5.0, n)               # up to a gentle breeze (assumption)
    p["wind_dir_deg"] = rng.uniform(0.0, 360.0, n)              # direction the wind blows TOWARD
    p["rho_kgm3"] = rng.uniform(1.14, 1.29, n)                  # ~0-35 C at sea level

    az = np.radians(p["azimuth_deg"]); el = np.radians(p["elevation_deg"]); v0 = p["speed_mps"]
    vel = np.stack([v0*np.cos(el)*np.cos(az), v0*np.cos(el)*np.sin(az), v0*np.sin(el)], axis=1)

    e_s = unit(np.stack([np.cos(az), np.sin(az), np.zeros(n)], axis=1))
    e_z = np.tile([0.0, 0.0, 1.0], (n, 1))
    e_h = np.cross(e_z, e_s)
    side_sign = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    tb_sign = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    elev_i = np.radians(rng.normal(0.0, 20.0, n)); screw_i = rng.normal(0.0, 0.10, n)
    elev_c = np.radians(np.clip(rng.normal(62.6, 10.0, n), 30.0, 89.0)); screw_c = rng.normal(0.38, 0.05, n)
    elev = np.where(curve, elev_c, np.abs(elev_i)); screw = np.where(curve, screw_c, screw_i)
    zc = side_sign*np.sin(elev); horiz = np.cos(elev)
    screw = np.clip(screw, -0.95*horiz, 0.95*horiz)
    hc = tb_sign*np.sqrt(np.maximum(horiz**2 - screw**2, 0.0))
    w_hat = unit(hc[:, None]*e_h + screw[:, None]*e_s + zc[:, None]*e_z)
    # +e_h x v -> -z (topspin, dips); -e_h -> backspin (lifts); +z axis x v -> +y (curves left)
    p["topback"] = np.where(tb_sign > 0, "top", "back")
    p["curve_dir"] = np.where(side_sign > 0, "left", "right")

    omega = 2*np.pi*p["spin_rps"]
    p["sp0"] = R_BALL*omega/v0
    p["beta_per_m"] = RHO0*AREA/(2*p["mass_kg"])
    p["cd_simple"] = p["cd0"] + DCD_DSP*p["sp0"]
    p["cl0"] = cl_of_sp(p["sp0"], p["clmax"])
    p["k_per_m"] = p["beta_per_m"]*p["cd_simple"]
    p["s_per_s"] = p["beta_per_m"]*p["cl0"]*v0
    p["spin_axis_x"], p["spin_axis_y"], p["spin_axis_z"] = w_hat.T
    p["spin_axis_to_v_deg"] = np.degrees(np.arccos(np.clip(np.sum(w_hat*unit(vel), 1), -1, 1)))
    wd = np.radians(p["wind_dir_deg"])
    wind = np.stack([p["wind_speed_mps"]*np.cos(wd), p["wind_speed_mps"]*np.sin(wd), np.zeros(n)], 1)
    return p, vel, w_hat, omega, wind


def accel_simple(v, t, q, rolling):
    sp = np.linalg.norm(v, axis=1, keepdims=True)
    a = -q["k"][:, None]*sp*v + q["s"][:, None]*np.cross(q["w"], v)
    a[:, 2] -= G
    a[rolling, 2] = 0.0
    return a


def accel_real(v, t, q, rolling):
    vr = v - q["wind"]
    sp = np.linalg.norm(vr, axis=1)
    om = q["omega"]*np.exp(-t/SPIN_TAU)
    spar = R_BALL*om/sp
    cd = q["cda"] + q["cdb"]/(1+np.exp((sp-q["cdvc"])/q["cdvs"])) + DCD_DSP*spar
    cl = cl_of_sp(spar, q["clmax"])
    a = -(q["beta"]*cd*sp)[:, None]*vr + (q["beta"]*cl*sp)[:, None]*np.cross(q["w"], vr)
    a[:, 2] -= G
    a[rolling, 2] = 0.0
    return a


def integrate(vel0, q, accel, cor):
    n = len(vel0)
    pos = np.zeros((n, 3)); pos[:, 2] = Z0
    vel = vel0.copy()
    rolling = np.zeros(n, bool)
    obs = {tt: np.full((n, 3), np.nan) for tt in OBS_TIMES}
    obsv = {tt: np.full((n, 3), np.nan) for tt in OBS_TIMES}
    steps = {int(round(tt/DT)): tt for tt in OBS_TIMES}
    done = np.zeros(n, bool)
    out = {k: np.full(n, np.nan) for k in ("t_goal", "y_goal", "z_goal", "v_goal", "t_b1", "x_b1", "y_b1")}
    nb = np.zeros(n, int)
    t_roll = np.full(n, np.nan)
    for i in range(int(round(T_MAX/DT))):
        t = i*DT
        act = np.where(~done)[0]
        if len(act) == 0:
            break
        sub = {k: (v[act] if isinstance(v, np.ndarray) and v.shape[0] == n else v) for k, v in q.items()}
        rl = rolling[act]
        p0, v0 = pos[act], vel[act]
        k1 = accel(v0, t, sub, rl)
        k2 = accel(v0+0.5*DT*k1, t+0.5*DT, sub, rl)
        k3 = accel(v0+0.5*DT*k2, t+0.5*DT, sub, rl)
        k4 = accel(v0+DT*k3, t+DT, sub, rl)
        p1 = p0 + DT/6*(v0 + 2*(v0+0.5*DT*k1) + 2*(v0+0.5*DT*k2) + (v0+DT*k3))
        v1 = v0 + DT/6*(k1+2*k2+2*k3+k4)
        p1[rl, 2] = R_BALL; v1[rl, 2] = 0.0
        fx = np.where(p1[:, 0] >= X_GOAL, (X_GOAL-p0[:, 0])/(p1[:, 0]-p0[:, 0]), np.inf)
        fz = np.where((p1[:, 2] < R_BALL) & ~rl,
                      np.clip((p0[:, 2]-R_BALL)/np.maximum(p0[:, 2]-p1[:, 2], 1e-12), 0, 1), np.inf)
        gx = np.isfinite(fx) & (fx <= fz)
        gz = np.isfinite(fz) & ~gx
        if gx.any():
            f = fx[gx][:, None]; j = act[gx]
            pe = p0[gx]+f*(p1[gx]-p0[gx]); ve = v0[gx]+f*(v1[gx]-v0[gx])
            out["t_goal"][j] = t+f[:, 0]*DT; out["y_goal"][j] = pe[:, 1]; out["z_goal"][j] = pe[:, 2]
            out["v_goal"][j] = np.linalg.norm(ve, axis=1); done[j] = True
        if gz.any():
            f = fz[gz][:, None]; j = act[gz]
            pc = p0[gz]+f*(p1[gz]-p0[gz]); vc = v0[gz]+f*(v1[gz]-v0[gz])
            vb = cor[j][:, None]*vc*np.array([1, 1, -1])
            roll = vb[:, 2] < ROLL_VZ
            vb[roll, 2] = 0.0
            first = nb[j] == 0
            out["t_b1"][j[first]] = t+f[first, 0]*DT
            out["x_b1"][j[first]] = pc[first, 0]; out["y_b1"][j[first]] = pc[first, 1]
            nb[j] += 1
            newly = j[roll]; rolling[newly] = True; t_roll[newly] = t+f[roll, 0]*DT
            pc[:, 2] = R_BALL
            # finish the remainder of the step after the bounce with one RK4 sub-step
            h = ((1-f)*DT)[:, 0]
            subg = {k: (v[j] if isinstance(v, np.ndarray) and v.shape[0] == n else v) for k, v in q.items()}
            rg = rolling[j]
            tb = t + f[:, 0]*DT
            def acc_h(vv, tt):
                return accel(vv, tt, subg, rg)
            hk = h[:, None]
            c1 = acc_h(vb, tb)
            c2 = acc_h(vb+0.5*hk*c1, tb+0.5*h)
            c3 = acc_h(vb+0.5*hk*c2, tb+0.5*h)
            c4 = acc_h(vb+hk*c3, tb+h)
            pn = pc + hk/6*(vb + 2*(vb+0.5*hk*c1) + 2*(vb+0.5*hk*c2) + (vb+hk*c3))
            vn = vb + hk/6*(c1+2*c2+2*c3+c4)
            pn[rg, 2] = R_BALL; vn[rg, 2] = 0.0
            pn[:, 2] = np.maximum(pn[:, 2], R_BALL)
            p1[gz] = pn
            v1[gz] = vn
        keep = ~gx
        pos[act[keep]] = p1[keep]; vel[act[keep]] = v1[keep]
        if (i+1) in steps:
            tt = steps[i+1]; alive = ~done
            obs[tt][alive] = pos[alive]; obsv[tt][alive] = vel[alive]
    out.update(obs=obs, obsv=obsv, n_bounces=nb, t_roll=t_roll, timeout=~done)
    return out


def outcomes(res):
    y, z = res["y_goal"], res["z_goal"]; ay = np.abs(y)
    reached = np.isfinite(res["t_goal"])
    # v2-compatible centre-based outcome (problem's rectangle)
    o = np.full(len(y), "timeout", dtype=object)
    o[reached & (ay < Y_POST) & (z < Z_BAR)] = "on_target"
    o[reached & (ay >= Y_POST) & (z < Z_BAR)] = "wide"
    o[reached & (ay < Y_POST) & (z >= Z_BAR)] = "over"
    o[reached & (ay >= Y_POST) & (z >= Z_BAR)] = "over_wide"
    # frame-aware: ball (r = 0.11) vs posts |y| in [3.66, 3.78], bar z in [2.44, 2.56]
    f = np.full(len(y), "timeout", dtype=object)
    clear = reached & (ay <= Y_POST-R_BALL) & (z <= Z_BAR-R_BALL)
    hit_post = reached & (ay > Y_POST-R_BALL) & (ay < Y_POST+POST_W+R_BALL) & (z < Z_BAR+POST_W+R_BALL)
    hit_bar = reached & (z > Z_BAR-R_BALL) & (z < Z_BAR+POST_W+R_BALL) & (ay < Y_POST+POST_W+R_BALL)
    frame = (hit_post | hit_bar) & ~clear
    f[reached] = "miss"
    f[frame] = "frame"
    f[clear] = "goal"
    return o, f


def build(p, res, rng):
    df = pd.DataFrame(p)
    for tt in OBS_TIMES:
        tag = f"{int(round(tt*100)):03d}"
        tru = res["obs"][tt]
        noisy = np.round(tru + rng.normal(0.0, SIGMA_MAIN, tru.shape), ROUND_DP)
        for c, a in zip("xyz", range(3)):
            df[f"{c}_t{tag}_true"] = tru[:, a]; df[f"{c}_t{tag}_obs"] = noisy[:, a]
    o, f = outcomes(res)
    df["outcome"] = o; df["outcome_frame"] = f
    df["y_goal"] = res["y_goal"]; df["z_goal"] = res["z_goal"]
    df["t_goal_s"] = res["t_goal"]; df["v_goal_mps"] = res["v_goal"]
    df["n_bounces"] = res["n_bounces"]; df["bounced_before_goal"] = res["n_bounces"] > 0
    df["t_bounce1_s"] = res["t_b1"]; df["x_bounce1"] = res["x_b1"]; df["y_bounce1"] = res["y_b1"]
    df["t_roll_start_s"] = res["t_roll"]
    # v2 name kept: time of first ground contact
    df["t_ground_s"] = res["t_b1"]
    return df


def add_weights(df):
    # Sensitivity re-weighting (mean weight = 1 over the rows it applies to)
    df["w_kick5050"] = np.where(df.kick_type == "curve", 0.5/0.3, 0.5/0.7)
    df["w_back70"] = np.where(df.topback == "back", 0.7/0.5, 0.3/0.5)
    on = df.outcome == "on_target"
    low = on & (df.z_goal < Z_BAR/2); up = on & (df.z_goal >= Z_BAR/2)
    fl, fu = low.sum()/on.sum(), up.sum()/on.sum()
    w = np.full(len(df), np.nan)
    w[low.values] = 0.707/fl; w[up.values] = 0.293/fu          # Danish top tier: 70.7 % lower half
    df["w_place_lower707"] = w
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60000); ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="."); ap.add_argument("--sweep_n", type=int, default=10000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    p, vel, w_hat, omega, wind = sample_latent(a.n, rng)
    u = rng.random(a.n)
    split = np.where(u < 0.70, "train", np.where(u < 0.85, "val", "test"))

    res_s = integrate(vel, dict(k=p["k_per_m"], s=p["s_per_s"], w=w_hat), accel_simple, p["cor"])
    cd = np.array([CD_FIT[b] for b in p["ball_model"]])
    beta_r = p["rho_kgm3"]*AREA/(2*p["mass_kg"])
    q_r = dict(cda=cd[:, 0], cdb=cd[:, 1], cdvc=cd[:, 2], cdvs=cd[:, 3], clmax=p["clmax"],
               beta=beta_r, omega=omega, w=w_hat, wind=wind)
    res_r = integrate(vel, q_r, accel_real, p["cor"])

    S = build(p, res_s, np.random.default_rng(a.seed+1))
    Rl = build(p, res_r, np.random.default_rng(a.seed+2))
    for D in (S, Rl):
        D.insert(0, "shot_id", np.arange(a.n)); D.insert(1, "split", split)
    S = S.drop(columns=["ball_model", "wind_speed_mps", "wind_dir_deg", "rho_kgm3"])
    Rl = Rl.drop(columns=["k_per_m", "s_per_s", "cd_simple"])
    S = add_weights(S); Rl = add_weights(Rl)
    S.to_csv(os.path.join(a.out, "penalty_simple_v3.csv"), index=False, float_format="%.6g")
    Rl.to_csv(os.path.join(a.out, "penalty_realistic_v3.csv"), index=False, float_format="%.6g")

    # Noise variants on simple-physics TEST shots (not rounded)
    idx = np.where(split == "test")[0][:a.sweep_n]
    rs = np.random.default_rng(a.seed+3)
    variants = [("iid", 0.002, {}), ("iid", 0.005, {}), ("iid", 0.010, {}), ("iid", 0.020, {}),
                ("aniso_x3", 0.005, {}), ("jitter_1ms", 0.005, {}), ("outlier_5pct", 0.005, {})]
    rows = []
    for name, sig, _ in variants:
        d = pd.DataFrame({"shot_id": idx, "noise_type": name, "sigma_m": sig})
        for tt in OBS_TIMES:
            tag = f"{int(round(tt*100)):03d}"
            tru = res_s["obs"][tt][idx]; vv = res_s["obsv"][tt][idx]
            e = rs.normal(0.0, sig, tru.shape)
            if name == "aniso_x3":
                e[:, 0] *= 3.0
            if name == "jitter_1ms":
                e = e + vv*rs.normal(0.0, 1e-3, (len(idx), 1))
            if name == "outlier_5pct":
                m = rs.random(tru.shape[0]) < 0.05
                e[m] = rs.normal(0.0, 0.05, (m.sum(), 3))
            noisy = tru + e
            for c, ax in zip("xyz", range(3)):
                d[f"{c}_t{tag}_obs"] = noisy[:, ax]
        for c in ("outcome", "outcome_frame", "y_goal", "z_goal", "bounced_before_goal"):
            d[c] = S[c].values[idx]
        rows.append(d)
    pd.concat(rows).to_csv(os.path.join(a.out, "penalty_noise_v3.csv"), index=False, float_format="%.6g")

    meta = dict(version=3, n=a.n, seed=a.seed, dt=DT, t_max=T_MAX, obs_times=OBS_TIMES, sigma_main=SIGMA_MAIN,
                round_dp=ROUND_DP, g=G, rho_simple=RHO0, r_ball=R_BALL, post_width=POST_W, spin_tau=SPIN_TAU,
                dcd_dsp=DCD_DSP, sp_c=SP_C, roll_vz=ROLL_VZ, cd_fit=CD_FIT,
                noise_variants=[(v[0], v[1]) for v in variants])
    json.dump(meta, open(os.path.join(a.out, "generator_settings_v3.json"), "w"), indent=2)
    print("simple:", S.outcome.value_counts().to_dict())
    print("simple frame:", S.outcome_frame.value_counts().to_dict())
    print("realistic:", Rl.outcome.value_counts().to_dict())
    print("bounced frac:", S.bounced_before_goal.mean())


if __name__ == "__main__":
    main()
