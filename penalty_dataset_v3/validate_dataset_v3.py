"""
Independent validation of the v3 penalty datasets.
Physics re-implemented from scratch with scipy solve_ivp (DOP853) and event-based bounces,
not imported from the generator.

Run:  python validate_dataset_v3.py --dir .
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

G, R = 9.81, 0.11
A = np.pi*R**2
TAU = 9.5
CDF = {"brazuca": (0.187118, 0.260996, 12.9084, 0.717838),
       "jabulani": (0.163643, 0.306765, 17.3015, 1.61952)}
TAGS = ["005", "010", "015", "020"]; TIMES = [0.05, 0.10, 0.15, 0.20]
V2_COLS = ["shot_id", "split", "kick_type", "speed_mps", "azimuth_deg", "elevation_deg", "spin_rps", "mass_kg",
           "cd0", "clmax", "sp0", "beta_per_m", "spin_axis_x", "spin_axis_y", "spin_axis_z", "spin_axis_to_v_deg",
           "outcome", "y_goal", "z_goal", "t_goal_s", "v_goal_mps", "t_ground_s"] + \
          [f"{c}_t{t}_{k}" for t in TAGS for c in "xyz" for k in ("true", "obs")]
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def launch(r):
    az, el, v = np.radians(r.azimuth_deg), np.radians(r.elevation_deg), r.speed_mps
    return np.array([v*np.cos(el)*np.cos(az), v*np.cos(el)*np.sin(az), v*np.sin(el)])


def make_rhs(r, kind, g=G, beta=None, rho=None):
    w = np.array([r.spin_axis_x, r.spin_axis_y, r.spin_axis_z])
    if kind == "simple":
        def acc(t, v):
            return -r.k_per_m*np.linalg.norm(v)*v + r.s_per_s*np.cross(w, v) - [0, 0, g]
    else:
        a_, b_, vc, vs = CDF[r.ball_model]
        rho = r.rho_kgm3 if rho is None else rho
        beta = rho*A/(2*r.mass_kg) if beta is None else beta
        wd = np.radians(r.wind_dir_deg) if "wind_dir_deg" in r else 0.0
        ws = r.wind_speed_mps if "wind_speed_mps" in r else 0.0
        wind = np.array([ws*np.cos(wd), ws*np.sin(wd), 0.0])
        om0 = 2*np.pi*r.spin_rps
        def acc(t, v):
            vr = v - wind; s = np.linalg.norm(vr)
            sp = R*om0*np.exp(-t/TAU)/s
            cd = a_ + b_/(1+np.exp((s-vc)/vs)) + 0.55*sp
            cl = r.clmax*(1-np.exp(-sp/0.10))
            return -beta*cd*s*vr + beta*cl*s*np.cross(w, vr) - [0, 0, g]
    def fly(t, y):
        return np.r_[y[3:], acc(t, y[3:])]
    def roll(t, y):
        a = acc(t, y[3:]); a[2] = 0.0
        return np.r_[y[3:5], 0.0, a]
    return fly, roll


def simulate(r, kind, xg=11.0, z0=0.11, v0=None, zmin=0.11, bounce=True, tmax=1.5, **kw):
    fly, roll = make_rhs(r, kind, **kw)
    y = np.r_[0, 0, z0, launch(r) if v0 is None else v0]
    t, rolling, segs, nb, tb1 = 0.0, False, [], 0, np.nan
    ev_g = lambda t, y: y[0]-xg; ev_g.terminal = True; ev_g.direction = 1
    ev_z = lambda t, y: y[2]-zmin; ev_z.terminal = True; ev_z.direction = -1
    if y[5] <= 0 and bounce:  # starts at ground moving down/flat: will touch immediately
        pass
    while t < tmax:
        f = roll if rolling else fly
        evs = [ev_g] if (rolling or not bounce) else [ev_g, ev_z]
        sol = solve_ivp(f, [t, tmax], y, method="DOP853", events=evs, dense_output=True, rtol=1e-10, atol=1e-12)
        segs.append((t, sol.t[-1], sol.sol))
        if len(sol.t_events[0]):
            tg = sol.t_events[0][0]; yg = sol.y_events[0][0]
            return dict(segs=segs, reached=True, t_goal=tg, y_goal=yg[1], z_goal=yg[2], nb=nb, tb1=tb1)
        if not rolling and bounce and len(sol.t_events) > 1 and len(sol.t_events[1]):
            tc = sol.t_events[1][0]; yc = sol.y_events[1][0].copy()
            if nb == 0: tb1 = tc
            nb += 1
            vb = r.cor*yc[3:]*np.array([1, 1, -1])
            if vb[2] < 0.5:
                vb[2] = 0.0; rolling = True
            yc[2] = zmin; yc[3:] = vb
            t, y = tc, yc
            continue
        break
    return dict(segs=segs, reached=False, nb=nb, tb1=tb1)


def pos_at(res, tt):
    for a, b, s in res["segs"]:
        if a - 1e-12 <= tt <= b + 1e-12:
            return s(tt)[:3]
    return None


def outcome(y, z):
    if abs(y) < 3.66 and z < 2.44: return "on_target"
    if abs(y) >= 3.66 and z < 2.44: return "wide"
    if abs(y) < 3.66: return "over"
    return "over_wide"


def outcome_frame(y, z):
    ay = abs(y)
    if ay <= 3.55 and z <= 2.33: return "goal"
    post = 3.55 < ay < 3.89 and z < 2.67
    bar = 2.33 < z < 2.67 and ay < 3.89
    return "frame" if (post or bar) else "miss"


def resim(df, kind, n=500, seed=0):
    sub = df.sample(n, random_state=seed)
    mism, eg, eo, eb = [], [], [], []
    for _, r in sub.iterrows():
        res = simulate(r, kind)
        pred = outcome(res["y_goal"], res["z_goal"]) if res["reached"] else "timeout"
        if pred != r.outcome or res["nb"] != r.n_bounces:
            mism.append((r.shot_id, pred, r.outcome, res["nb"], r.n_bounces))
            continue
        if res["reached"]:
            eg.append([abs(res["y_goal"]-r.y_goal), abs(res["z_goal"]-r.z_goal), abs(res["t_goal"]-r.t_goal_s)])
        if res["nb"] > 0:
            eb.append(abs(res["tb1"]-r.t_bounce1_s))
        for tag, tt in zip(TAGS, TIMES):
            if np.isfinite(r[f"x_t{tag}_true"]):
                p = pos_at(res, tt)
                if p is not None:
                    eo.append(np.abs(p-[r[f"x_t{tag}_true"], r[f"y_t{tag}_true"], r[f"z_t{tag}_true"]]).max())
    eg, eo = np.array(eg), np.array(eo)
    nbounced = int((sub.n_bounces > 0).sum())
    check(f"{kind}: outcome and bounce count reproduced by independent event-based solver", len(mism) <= 3,
          f"{len(mism)}/{n} mismatches ({nbounced} bounced shots in sample) {mism[:3]}")
    check(f"{kind}: goal-plane point reproduced (max error < 2 mm)", eg[:, :2].max() < 2e-3,
          f"max |dy|={eg[:,0].max()*1000:.3f} mm, max |dz|={eg[:,1].max()*1000:.3f} mm, 99th pct={np.percentile(eg[:,:2].max(1),99)*1000:.3f} mm")
    check(f"{kind}: first-bounce time reproduced (max error < 0.5 ms)", (np.max(eb) if eb else 0) < 5e-4,
          f"max {np.max(eb)*1000:.4f} ms over {len(eb)} bounced shots")
    check(f"{kind}: true observation positions reproduced (max error < 1 mm)", eo.max() < 1e-3,
          f"max {eo.max()*1000:.4f} mm over {len(eo)} points")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="."); a = ap.parse_args()
    S = pd.read_csv(os.path.join(a.dir, "penalty_simple_v3.csv"))
    Rl = pd.read_csv(os.path.join(a.dir, "penalty_realistic_v3.csv"))
    W = pd.read_csv(os.path.join(a.dir, "penalty_noise_v3.csv"))

    print("== 1. Structure and v2 compatibility ==")
    check("60,000 rows in each main file", len(S) == 60000 and len(Rl) == 60000)
    miss = [c for c in V2_COLS if c not in S.columns]
    check("all v2 columns present in simple file", not miss, f"missing: {miss}")
    shared = ["speed_mps", "azimuth_deg", "elevation_deg", "spin_rps", "spin_axis_x", "spin_axis_y", "spin_axis_z", "mass_kg", "cor"]
    check("same latent shots in simple and realistic files", np.allclose(S[shared].values, Rl[shared].values) and (S.shot_id == Rl.shot_id).all())
    fr = S.split.value_counts(normalize=True)
    check("split ~70/15/15", abs(fr["train"]-0.70) < 0.01 and abs(fr["val"]-0.15) < 0.01, fr.round(3).to_dict())
    check("no duplicate rows", not S.duplicated().any())
    req = ["speed_mps", "azimuth_deg", "elevation_deg", "spin_rps", "k_per_m", "s_per_s", "cor", "outcome", "outcome_frame", "n_bounces"]
    check("no missing values in parameter/label columns", S[req].notna().all().all())

    print("\n== 2. Independent re-simulation (with bounces and wind) ==")
    resim(S, "simple"); resim(Rl, "realistic")
    # targeted: extra bounced shots
    print("   extra check on 200 simple-file shots with >= 2 bounces:")
    resim(S[S.n_bounces >= 2], "simple", n=200, seed=1)

    print("\n== 3. Physical consistency ==")
    for name, D in (("simple", S), ("realistic", Rl)):
        zmin = np.nanmin(D[[f"z_t{t}_true" for t in TAGS]].values)
        check(f"{name}: no true ball-centre height below 0.11 m", zmin >= 0.11-1e-6, f"min = {zmin:.5f}")
        r = D[D.t_goal_s.notna()]
        check(f"{name}: goal-plane crossing never below 0.11 m", (r.z_goal >= 0.11-1e-6).all(), f"min = {r.z_goal.min():.5f}")
        lab = r.apply(lambda q: outcome(q.y_goal, q.z_goal), axis=1)
        check(f"{name}: centre-based outcome matches y_goal/z_goal", (lab == r.outcome).all())
        labf = r.apply(lambda q: outcome_frame(q.y_goal, q.z_goal), axis=1)
        check(f"{name}: frame-aware outcome matches y_goal/z_goal", (labf == r.outcome_frame).all())
        check(f"{name}: every frame-aware 'goal' is also centre 'on_target'", (D[D.outcome_frame == "goal"].outcome == "on_target").all())
        bad = 0
        tend = D.t_goal_s.fillna(9)
        for tag, tt in zip(TAGS, TIMES):
            bad += (D[f"x_t{tag}_obs"].notna() & (tt > tend + 1e-9)).sum()
            bad += (D[f"x_t{tag}_obs"].isna() & (tt < tend - 1e-3)).sum()
        check(f"{name}: observations present only while the ball is in play", bad == 0, f"{bad} violations")
        b = D[D.n_bounces > 0]
        check(f"{name}: bounce bookkeeping consistent", b.t_bounce1_s.notna().all() and D[D.n_bounces == 0].t_bounce1_s.isna().all()
              and (b.t_bounce1_s < b.t_goal_s.fillna(9)).all() and (D.t_ground_s.equals(D.t_bounce1_s) or np.allclose(D.t_ground_s.fillna(-1), D.t_bounce1_s.fillna(-1))))
        rl = D[D.t_roll_start_s.notna()]
        check(f"{name}: rolling balls cross the goal at z = 0.11 m", np.allclose(rl.z_goal.dropna(), 0.11, atol=1e-6))
        print(f"      {name}: timeout (did not reach goal within 1.5 s): {(D.outcome=='timeout').mean()*100:.2f} %")
    check("spin axis is a unit vector", np.allclose(np.linalg.norm(S[["spin_axis_x", "spin_axis_y", "spin_axis_z"]].values, axis=1), 1, atol=1e-6))
    beta = 1.2*A/(2*S.mass_kg)
    check("k = beta*C_D and s = beta*C_L*v0", np.allclose(S.k_per_m, beta*S.cd_simple, rtol=1e-4) and np.allclose(S.s_per_s, beta*S.cl0*S.speed_mps, rtol=1e-4))
    nb = S[S.n_bounces == 0]
    dz = nb.groupby("topback").z_goal.mean()
    check("spin labels: backspin shots end higher than topspin shots", dz["back"] > dz["top"], dz.round(3).to_dict())
    dev = (nb.y_goal - 11*np.tan(np.radians(nb.azimuth_deg)))
    dy = dev.groupby(nb.curve_dir).mean()
    check("spin labels: 'left' sidespin curves toward +y, 'right' toward -y", dy["left"] > 0 > dy["right"], dy.round(3).to_dict())

    print("\n== 4. Measurement noise ==")
    for name, D in (("simple", S), ("realistic", Rl)):
        res = np.concatenate([(D[f"{c}_t{t}_obs"]-D[f"{c}_t{t}_true"]).dropna().values for c in "xyz" for t in TAGS])
        exp = np.sqrt(0.005**2 + 0.01**2/12)
        check(f"{name}: obs - true SD ~{exp*1000:.2f} mm, mean ~0", abs(res.mean()) < 2e-4 and abs(res.std()-exp) < 3e-4,
              f"mean={res.mean()*1000:.3f} mm, SD={res.std()*1000:.3f} mm")
    Si = S.set_index("shot_id")
    for (nt, sig), d in W.groupby(["noise_type", "sigma_m"], sort=False):
        tru = Si.loc[d.shot_id]
        e = {c: np.concatenate([d[f"{c}_t{t}_obs"].values - tru[f"{c}_t{t}_true"].values for t in TAGS]) for c in "xyz"}
        e = {c: v[np.isfinite(v)] for c, v in e.items()}
        if nt == "iid":
            sd = np.std(np.concatenate(list(e.values())))
            check(f"noise iid sigma={sig*1000:.0f} mm", abs(sd/sig-1) < 0.03, f"SD={sd*1000:.2f} mm")
        elif nt == "aniso_x3":
            check("noise aniso_x3: x SD ~15 mm, y/z ~5 mm", abs(e["x"].std()/0.015-1) < 0.05 and abs(e["y"].std()/0.005-1) < 0.05,
                  f"x={e['x'].std()*1000:.2f}, y={e['y'].std()*1000:.2f}, z={e['z'].std()*1000:.2f} mm")
        elif nt == "jitter_1ms":
            check("noise jitter_1ms: x SD ~ sqrt(5^2 + (v*1ms)^2) mm (~28 mm), y/z smaller",
                  20e-3 < e["x"].std() < 40e-3 and e["y"].std() < e["x"].std(),
                  f"x={e['x'].std()*1000:.1f}, y={e['y'].std()*1000:.1f}, z={e['z'].std()*1000:.1f} mm")
        elif nt == "outlier_5pct":
            big = np.mean(np.abs(e["x"]) > 0.025)
            check("noise outlier_5pct: ~5 % of points have large errors", 0.03 < big < 0.06, f"|err_x| > 25 mm in {big*100:.2f} %")
    check("noise file uses only test-split shots", set(Si.loc[W.shot_id.unique()].split) == {"test"})

    print("\n== 5. Priors vs literature ==")
    check("speed mean 26-32 m/s, range 18-40", 26 <= S.speed_mps.mean() <= 32 and S.speed_mps.between(18, 40).all(),
          f"mean={S.speed_mps.mean():.2f}, SD={S.speed_mps.std():.2f}")
    by = S.groupby("kick_type").spin_rps.mean()
    check("spin: instep ~3.25 rps, curve ~7.35 rps, max <= 10", abs(by["instep"]-3.25) < 0.1 and abs(by["curve"]-7.35) < 0.15 and S.spin_rps.max() <= 10, by.round(2).to_dict())
    check("C_D within 0.15-0.50 (Carre 2002: up to 0.5 with spin)", S.cd_simple.between(0.15, 0.5).all(), f"[{S.cd_simple.min():.3f}, {S.cd_simple.max():.3f}]")
    check("C_L at launch within 0-0.35", S.cl0.between(0, 0.35).all(), f"[{S.cl0.min():.3f}, {S.cl0.max():.3f}]")
    check("COR within 0.56-0.76 (Thomas 2024)", S.cor.between(0.56, 0.76).all())
    check("wind 0-5 m/s, rho 1.14-1.29 kg/m3 (realistic)", Rl.wind_speed_mps.between(0, 5).all() and Rl.rho_kgm3.between(1.14, 1.29).all())
    re0 = S.speed_mps*2*R/1.54e-5
    check("all launches supercritical (Re > 2.2e5)", (re0 > 2.2e5).all(), f"min Re = {re0.min():.3g}")

    print("\n== 6. Re-weighting columns ==")
    check("w_kick5050 gives 50/50 kick mix", abs(np.average(S.kick_type == "curve", weights=S.w_kick5050)-0.5) < 0.01)
    check("w_back70 gives 70 % backspin", abs(np.average(S.topback == "back", weights=S.w_back70)-0.7) < 0.01)
    on = S[S.outcome == "on_target"]
    check("w_place_lower707 gives 70.7 % of on-target shots in lower half", abs(np.average(on.z_goal < 1.22, weights=on.w_place_lower707)-0.707) < 0.002)
    check("w_place_lower707 defined only for on-target shots", S[S.outcome != "on_target"].w_place_lower707.isna().all() and on.w_place_lower707.notna().all())

    print("\n== 7. Literature benchmark: Goff, Asai & Hong (2014), no wind, rho = 1.2 ==")
    ref = {"brazuca": (0.751, 0.938, 24.21), "jabulani": (0.737, 1.006, 25.09)}
    for ball, b in (("brazuca", 0.0527), ("jabulani", 0.0521)):
        r = pd.Series(dict(spin_axis_x=0, spin_axis_y=0, spin_axis_z=1, ball_model=ball, spin_rps=0.0, clmax=0.3,
                           mass_kg=0.433, wind_speed_mps=0.0, wind_dir_deg=0.0, rho_kgm3=1.2, cor=0.6))
        v0 = 30*np.array([np.cos(np.radians(10)), 0, np.sin(np.radians(10))])
        res = simulate(r, "realistic", xg=20.0, z0=0.0, v0=v0, bounce=False, g=9.80, beta=b)
        tg = res["t_goal"]; zg = res["z_goal"]; vg = np.linalg.norm(res["segs"][-1][2](tg)[3:])
        t_ref, z_ref, v_ref = ref[ball]
        check(f"benchmark {ball}: time and speed within 1 %, height within 2 cm",
              abs(tg-t_ref)/t_ref < 0.01 and abs(zg-z_ref) < 0.02 and abs(vg-v_ref)/v_ref < 0.01,
              f"ours t={tg:.3f}, z={zg:.3f}, v={vg:.2f} | paper t={t_ref}, z={z_ref}, v={v_ref}")

    print("\n== 8. Model-mismatch size and wind effect ==")
    m = (S.outcome == "on_target") & (Rl.outcome == "on_target") & (S.n_bounces == 0) & (Rl.n_bounces == 0)
    d = np.hypot(S.y_goal[m]-Rl.y_goal[m], S.z_goal[m]-Rl.z_goal[m])
    print(f"      no-bounce on-target shots: median {d.median()*100:.1f} cm, 90th pct {d.quantile(0.9)*100:.1f} cm, max {d.max()*100:.1f} cm")
    check("realistic vs simple differ by a few cm (median 1-30 cm)", 0.01 < d.median() < 0.30)
    calm = Rl.wind_speed_mps < 1; windy = Rl.wind_speed_mps > 4
    print(f"      median difference, wind < 1 m/s: {d[calm[m]].median()*100:.1f} cm; wind > 4 m/s: {d[windy[m]].median()*100:.1f} cm")
    check("stronger wind gives larger simple-vs-realistic difference", d[windy[m]].median() > d[calm[m]].median())

    print("\n== 9. Coverage of the given scenario ==")
    gvec = np.array([1.50, 0.45, 0.44, 2.96, 0.86, 0.74])
    cols = [f"{c}_t{t}_obs" for t in ("005", "010") for c in "xyz"]
    dist = np.linalg.norm(S[cols].values-gvec, axis=1)
    print(f"      nearest {np.sort(dist)[:5].round(3)} m; within 5 cm: {(dist<0.05).sum()}, within 10 cm: {(dist<0.10).sum()}")
    check("given scenario inside dataset coverage (>= 20 shots within 10 cm)", (dist < 0.10).sum() >= 20)

    print("\n== 10. Outcome mix ==")
    for nm, D in (("simple", S), ("realistic", Rl)):
        print(f"      {nm} outcome: {D.outcome.value_counts(normalize=True).round(3).to_dict()}")
        print(f"      {nm} outcome_frame: {D.outcome_frame.value_counts(normalize=True).round(3).to_dict()}")
        print(f"      {nm} bounced before goal: {D.bounced_before_goal.mean()*100:.1f} %, rolling: {D.t_roll_start_s.notna().mean()*100:.1f} %")
    o = S[S.outcome == "on_target"]
    grid = pd.crosstab(pd.cut(o.z_goal, [0.10, 0.81, 1.62, 2.44]), pd.cut(o.y_goal, [-3.66, -1.22, 1.22, 3.66]))
    print(grid.to_string())
    low_share = (o.z_goal < 1.22).mean()
    print(f"      lower-half share of on-target shots: {low_share*100:.1f} % (real penalties, Danish top tier: 70.7 %)")
    check("goal-zone coverage: every 3x3 cell has >= 2,000 on-target shots", grid.values.min() >= 2000,
          f"min cell {grid.values.min()}, max/min ratio {grid.values.max()/grid.values.min():.2f}")

    n_pass = sum(r[1] for r in results)
    print(f"\nSUMMARY: {n_pass}/{len(results)} checks passed")
    pd.DataFrame(results, columns=["check", "passed", "detail"]).to_csv(os.path.join(a.dir, "validation_results_v3.csv"), index=False)


if __name__ == "__main__":
    main()
