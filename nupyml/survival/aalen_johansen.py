"""Cumulative incidence under COMPETING RISKS (Aalen-Johansen estimator)."""
import numpy as np
from ..base import BaseEstimator


class AalenJohansen(BaseEstimator):
    """Cumulative incidence under COMPETING RISKS (Aalen-Johansen estimator).

    THE COMPETING-RISKS PROBLEM
    ---------------------------
    When a subject can fail from several MUTUALLY EXCLUSIVE causes (die of cancer
    OR heart disease), a naive Kaplan-Meier for one cause -- treating the other
    causes as censoring -- OVERSTATES that cause's incidence, because it pretends
    the subjects lost to the other cause could still have failed from this one.
    The Aalen-Johansen cumulative incidence function fixes this::

        CIF_k(t) = sum_{t_i <= t} S(t_{i-1}) * (d_{k,i} / n_i)

    -- each cause-k event is weighted by the OVERALL survival up to just before
    it, so a subject already dead from another cause cannot contribute. The CIFs
    of all causes sum to ``1 - S(t)``, as they must.

    ``events`` here carries the CAUSE (0 = censored, 1..K = cause of failure).
    ``cif_`` maps each cause to its (times, incidence) curve.
    """

    def fit(self, durations, events):
        durations = np.asarray(durations, float)
        events = np.asarray(events, int)
        causes = [c for c in np.unique(events) if c != 0]
        uniq_t = np.unique(durations[events != 0])
        n = len(durations)
        S = 1.0                                         # overall survival
        cif = {c: 0.0 for c in causes}
        self.cif_ = {c: {"time": [], "incidence": []} for c in causes}
        self.times_ = uniq_t
        for t in uniq_t:
            at_risk = (durations >= t).sum()
            if at_risk == 0:
                continue
            d_total = ((durations == t) & (events != 0)).sum()
            for c in causes:
                d_c = ((durations == t) & (events == c)).sum()
                cif[c] += S * (d_c / at_risk)           # weighted by prior survival
                self.cif_[c]["time"].append(t)
                self.cif_[c]["incidence"].append(cif[c])
            S *= (1 - d_total / at_risk)                # update overall survival
        for c in causes:
            self.cif_[c] = {k: np.array(v) for k, v in self.cif_[c].items()}
        return self

    def predict_cif(self, cause, t):
        c = self.cif_[cause]
        if len(c["time"]) == 0:
            return 0.0
        k = np.searchsorted(c["time"], t, side="right") - 1
        return 0.0 if k < 0 else float(c["incidence"][k])


__all__ = ["AalenJohansen"]
