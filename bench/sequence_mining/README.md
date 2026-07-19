# sequence_mining

**Goal:** classify symbol sequences by the ordered **pattern** planted in them.

- **Kind:** supervised — `solve(X_train, y_train, X_test) -> y_pred`, with `X` an object array of Python lists.
- **Data:** 300 sequences; class 0 hides the subsequence `A→B→C`, class 1 hides `D→E→F`, among random filler. 30% held out.
- **Metric:** accuracy (higher is better).
- **QA floor:** 0.80.

The signal is a **sequential** pattern. `prefixspan` mines discriminative subsequences and classifies on their presence; a bag-of-symbols baseline is the order-blind comparison.
