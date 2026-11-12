# Bayesian Prediction of a Penalty Kick's Goal Entry Point from Two Early Observations

Solution to **Problem B: Predicting FIFA Penalty Trajectories** of the
**University Physics Competition (UPC) 2026**, submitted as Team 1194.

| | |
|---|---|
| 📄 Paper | [`1194B.pdf`](1194B.pdf) (34 pages) |
| 📝 LaTeX source | [`latex/`](latex/) |
| 📊 Data, code and results | [`penalty_dataset_v3/`](penalty_dataset_v3/) |

---

## Problem

A penalty kick is taken from the penalty mark, the origin (0, 0, 0). The goal lies in the
plane x = 11 m, with −3.66 < y < 3.66 m and 0 < z < 2.44 m, and the centre of the ball
starts at (0, 0, 0.11) m at t = 0.

Given only the position of the ball at t = 0.05 s and t = 0.10 s, the task is to develop and
test a general method that predicts the point where the ball is most likely to enter the
goal, and to estimate the uncertainty of that prediction. The method is then applied to a
shot observed at (1.50, 0.45, 0.44) m and (2.96, 0.86, 0.74) m.

## Answer for the given shot

| Quantity | Result |
|---|---|
| Entry point (posterior mean, 1σ) | **y = 2.65 ± 0.22 m, z = 1.95 ± 0.23 m** |
| Posterior mode | y = 2.55 m, z = 1.95 m |
| Sensitivity range of y (priors, noise) | 2.55 – 2.97 m |
| Probability that the ball centre enters the goal | 0.99 |
| Probability that the whole ball clears the frame | 0.95 |
| Posterior probability of a curve kick | 0.83 |

The height is well determined. The lateral coordinate depends on the unobserved sidespin.

## Method

1. **Layered physical model.** Gravity, then quadratic drag, then the Magnus force,
   integrated with fourth-order Runge–Kutta (Δt = 1 ms). For each set of aerodynamic
   parameters, the launch velocity is fitted to the two observations by least squares.
2. **Identifiability.** Two 3-D positions give six measurements, while the full model has
   seven unknowns (launch velocity, drag, spin rate, spin axis). The problem is therefore
   underdetermined, and different spins fit the data almost equally well.
3. **Bayesian ensemble (main method).** 2000 parameter sets are drawn from priors based on
   published soccer-ball aerodynamics and kick measurements. Each set is fitted, weighted by
   its likelihood e^(−χ²/2) and propagated to the goal plane. This gives a best estimate and
   68 % / 95 % uncertainty ellipses.
4. **Machine-learning cross-check.** A histogram gradient-boosting model with quantile
   intervals is trained on 23 features computed from the same six coordinates.
5. **Goalkeeper reach.** An illustrative estimate of whether a purely reactive goalkeeper
   can reach the predicted point.

## Testing

All methods are evaluated on the same 300 simulated test shots that reach the goal without
bouncing. The shots are generated twice: once with the simple flight model, and once with
richer physics (speed-dependent drag, spin decay, wind up to 5 m/s, varying air density).

| Method | Median error, simple | Median error, richer physics |
|---|---|---|
| A: straight line | 64.0 cm | 63.6 cm |
| B: gravity only | 48.3 cm | 44.8 cm |
| C: gravity + drag | 48.8 cm | 45.5 cm |
| **D: Bayesian ensemble** | **26.5 cm** | **25.4 cm** |
| ML: gradient boosting | 25.2 cm | 25.2 cm |

- The 68 % and 95 % regions of Method D contain the true entry point in 69–70 % and
  92–93 % of test shots.
- Method D and the ML model fail on the same shots (error correlation about 0.9). The
  remaining error of about 25 cm is therefore set by the information in two early
  observations, not by the choice of method.
- The paper also discusses the limitations: curve kicks, slow shots, bounced shots, and test
  shots that share the prior of the method.

## Repository structure

```
.
├── 1194B.pdf                        # submitted paper
├── latex/                           # LaTeX source of the paper
│   ├── main.tex
│   ├── nitu_spin.tex                # spin-posterior section
│   ├── nitu_compare.tex             # method comparison and noise tests
│   ├── appendix_code.tex            # code appendix
│   ├── ai_report.tex                # AI-use statement
│   ├── references.bib
│   ├── hellofig.png
│   └── Figure/                      # physical-model figures
└── penalty_dataset_v3/
    ├── generate_dataset_v3.py       # simulated-shot generator
    ├── validate_dataset_v3.py       # validation checks
    ├── generator_settings_v3.json
    ├── validation_report_v3.txt
    ├── penalty_simple_v3.csv        # 60 000 shots, simple physics
    ├── penalty_realistic_v3.csv     # the same shots, richer physics and wind
    ├── penalty_noise_v3.csv         # noise-level variants
    ├── Model-D/                     # Bayesian-ensemble results and figures
    └── ML/                          # gradient-boosting results and figures
```

## Reproducing

**Dataset** (Python 3 with `numpy`, `pandas` and `scipy`):

```bash
cd penalty_dataset_v3
python generate_dataset_v3.py --n 60000 --seed 2026   # identical output for the same seed
python validate_dataset_v3.py --dir .                 # 65/65 checks pass
```

**Paper:**

```bash
cd latex
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Some figures are read directly from `penalty_dataset_v3/`, so compile inside the full
repository.
