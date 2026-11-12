# Bayesian Prediction of a Penalty Kick's Goal Entry Point from Two Early Observations

Solution paper for **Problem B: Predicting FIFA Penalty Trajectories** of the
**University Physics Competition (UPC) 2026**, submitted as Team 1194.

📄 **Paper:** [`1194B.pdf`](1194B.pdf) (34 pages)
📝 **LaTeX source:** [`latex/`](latex/) (compile `main.tex` with `pdflatex` → `bibtex` → `pdflatex` ×2)
📊 **Code, data and results:** [`penalty_dataset_v3/`](penalty_dataset_v3/)

## The problem

A penalty kick starts with the ball centre at (0, 0, 0.11) m, and the goal lies in the
plane x = 11 m (−3.66 < y < 3.66, 0 < z < 2.44). Given only the ball position at
t = 0.05 s and t = 0.10 s, develop and test a general method that predicts where the
ball is most likely to enter the goal and estimates the uncertainty of that prediction.
Apply it to the shot observed at (1.50, 0.45, 0.44) m and (2.96, 0.86, 0.74) m.

## Approach

- **Layered physical model:** gravity → quadratic drag → Magnus force, integrated with
  fourth-order Runge–Kutta, with the launch velocity fitted to the two observations.
- **Identifiability:** two positions give six measurements, but the full model has seven
  unknowns, so the lateral entry point cannot be fixed uniquely.
- **Bayesian ensemble (main method):** 2000 draws of drag, lift and spin axis from
  literature-based priors; each draw is fitted, weighted by its likelihood and propagated
  to the goal plane, giving a best estimate with 68 % / 95 % uncertainty ellipses.
- **Testing:** 300 simulated test shots, including shots generated with richer physics
  and wind, with checks of accuracy and of the calibration of the uncertainty.
- **Cross-check:** an independent gradient-boosting model built on 23 features of the
  same six coordinates.
- **Goalkeeper reach:** an illustrative estimate of whether a purely reactive goalkeeper
  can reach the predicted point.

## Main results

| | Result |
|---|---|
| Entry point of the given shot | y = 2.65 ± 0.22 m, z = 1.95 ± 0.23 m (1σ) |
| Probability of a goal | 0.99 (whole ball clears the frame: 0.95) |
| Median error on 300 test shots | 26 cm (25 cm with richer physics) |
| Coverage of the 68 % / 95 % regions | 69–70 % / 92–93 % |
| Gradient-boosting cross-check | (2.74, 2.04) m |

The height of the entry point is robust, while the lateral coordinate depends on the
unobserved sidespin; the remaining error of about 25 cm is set by the information in two
early observations rather than by the choice of method.

## Repository structure

```
├── 1194B.pdf              # submitted paper
├── penalty_dataset_v3/
│   ├── generate_dataset_v3.py    # simulated-shot generator
│   ├── validate_dataset_v3.py    # validation checks (65/65 pass)
│   ├── penalty_simple_v3.csv     # 60 000 shots, simple physics
│   ├── penalty_realistic_v3.csv  # same shots, richer physics + wind
│   ├── penalty_noise_v3.csv      # noise-level variants
│   ├── Model-D/                  # Bayesian-ensemble results and figures
│   └── ML/                       # gradient-boosting results and figures
└── latex/
    ├── main.tex           # main document
    ├── nitu_spin.tex      # spin-posterior section
    ├── nitu_compare.tex   # method comparison and noise tests
    ├── appendix_code.tex  # code appendix
    ├── ai_report.tex      # AI-use statement
    ├── references.bib     # bibliography
    ├── hellofig.png       # model-hierarchy figure
    └── Figure/            # all other figures
```
