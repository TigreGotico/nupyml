"""Time series and state-space models: when order is the signal.

WHAT MAKES THIS DIFFERENT
-------------------------
Everything else in this library assumes the rows are exchangeable -- shuffle them
and nothing changes. Here order IS the information. Yesterday predicts today;
shuffle the series and you have destroyed the very thing to be modelled. That
single change -- dependence through time -- is what the whole family is about, and
it forces different tools.

TWO WAYS TO THINK ABOUT A SEQUENCE
----------------------------------
* **State space.** There is a hidden state evolving over time (a position, a
  regime, a level and trend) that you observe only noisily. The job is to
  recover the state from the observations. This is the ``kalman`` family, and
  it is the same forward-backward idea as the HMM -- continuous state instead of
  discrete.
* **Autoregression.** The next value is a function of recent values and recent
  errors. No hidden state, just the series explaining itself. This is ``arima``
  and the smoothing methods.

They are not rivals so much as two languages for the same problems; exponential
smoothing turns out to be a state-space model in disguise, which is why the two
sections cross-reference each other.

THE MODULES
-----------
* ``kalman.py`` -- the Kalman filter and RTS smoother (linear-Gaussian), then the
  extended and unscented filters for when the dynamics are not linear, and the
  particle filter for when nothing is Gaussian at all.
* ``arima.py``  -- AR, MA, ARMA, ARIMA: autoregression, the differencing that
  makes a trending series stationary, and why stationarity is required.
* ``smoothing.py`` -- exponential smoothing through Holt-Winters, and STL
  decomposition into trend, season and remainder.

FILTER vs SMOOTH: THE ONE DISTINCTION TO GET RIGHT
--------------------------------------------------
Throughout, FILTERING estimates the state at time t from data up to t -- causal,
usable in real time. SMOOTHING estimates it from the WHOLE series, past and
future -- non-causal, only possible after the fact, and always at least as
accurate because it uses strictly more information. Which one you may use is
decided by whether the future is available, and confusing them is the classic
way to leak future information into a "real-time" system.
"""
from .kalman import (KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter,
                     ParticleFilter)
from .arima import AutoRegressive, ARMA, ARIMA
from .smoothing import (SimpleExponentialSmoothing, Holt, HoltWinters,
                        STL, seasonal_decompose)
from ._advanced import (Croston, Theta, SSA, sax, dtw_barycenter_averaging,
                        fourier_features)
from ._advanced2 import GARCH, VAR, AutoETS, ProphetForecaster, MSTL
from ._structural import BayesianStructuralTimeSeries
from ._advanced3 import SARIMA, TBATS, reconcile_forecasts, NBeats

__all__ = [
    "KalmanFilter", "ExtendedKalmanFilter", "UnscentedKalmanFilter",
    "ParticleFilter",
    "AutoRegressive", "ARMA", "ARIMA",
    "SimpleExponentialSmoothing", "Holt", "HoltWinters",
    "STL", "seasonal_decompose",
    "Croston", "Theta", "SSA", "sax", "dtw_barycenter_averaging",
    "fourier_features",
    "GARCH", "VAR", "AutoETS", "ProphetForecaster", "MSTL",
    "BayesianStructuralTimeSeries",
    "SARIMA", "TBATS", "reconcile_forecasts", "NBeats",
]
