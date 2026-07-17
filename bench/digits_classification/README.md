# digits_classification

**Goal:** classify 8×8 handwritten digit images into the ten classes 0–9.

- **Kind:** supervised — `solve(X_train, y_train, X_test) -> y_pred`
- **Data:** the bundled `load_digits` set (1797 samples, 64 pixel features scaled to [0,1]), a stratified 70/30 split at `random_state=0`.
- **Metric:** accuracy (higher is better).
- **QA floor:** 0.90. A strong entry reaches ~0.98.

Digits is nearly linearly separable, so even logistic regression scores ~0.97; the interest is in squeezing out the last percent with kernels, forests, or a small MLP. See `SCOREBOARD.md` for the current ranking.
