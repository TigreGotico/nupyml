"""State-space filters, ARIMA, exponential smoothing, and sequence models.

The state-space tests plant a known trajectory and check the estimate recovers
it; the ARIMA/smoothing tests plant a known trend or season; the sequence tests
plant a dependency between neighbouring labels that only a joint model can learn.
"""
import numpy as np
import pytest

from nupyml.timeseries import (
    KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter, ParticleFilter,
    AutoRegressive, ARMA, ARIMA, SimpleExponentialSmoothing, Holt, HoltWinters,
    STL, seasonal_decompose,
)
from nupyml.timeseries.arima import difference, integrate
from nupyml.sequence import (
    LinearChainCRF, StructuredPerceptron, dtw_distance, dtw_path, levenshtein,
    damerau_levenshtein, hamming, jaro, jaro_winkler,
    longest_common_subsequence, needleman_wunsch,
)


# --- Kalman ---------------------------------------------------------------

@pytest.fixture
def tracking_data():
    """A constant-velocity walk observed with noise -- the canonical filter test."""
    rng = np.random.RandomState(0)
    T = 100
    true = np.cumsum(rng.normal(0, 1, T)) + np.arange(T) * 0.5
    obs = true + rng.normal(0, 2, T)
    return true, obs


def _cv_filter():
    """A constant-velocity Kalman filter: state is (position, velocity)."""
    return KalmanFilter(F=[[1, 1], [0, 1]], H=[[1, 0]],
                        Q=[[0.01, 0], [0, 0.01]], R=[[4.0]],
                        x0=[0, 0], P0=[[1, 0], [0, 1]])


def test_kalman_filter_beats_the_raw_measurements(tracking_data):
    true, obs = tracking_data
    means, _ = _cv_filter().filter(obs)
    filtered_rmse = np.sqrt(np.mean((means[:, 0] - true) ** 2))
    obs_rmse = np.sqrt(np.mean((obs - true) ** 2))
    assert filtered_rmse < obs_rmse


def test_smoother_beats_the_filter(tracking_data):
    """Smoothing uses the whole series, so it is always at least as accurate --
    the price is that it can only be done offline."""
    true, obs = tracking_data
    kf = _cv_filter()
    filtered, _ = kf.filter(obs)
    smoothed, _ = kf.smooth(obs)
    f_rmse = np.sqrt(np.mean((filtered[:, 0] - true) ** 2))
    s_rmse = np.sqrt(np.mean((smoothed[:, 0] - true) ** 2))
    assert s_rmse < f_rmse


def test_kalman_forecast_uncertainty_grows(tracking_data):
    """With no measurements to correct it, forecast variance only widens -- the
    honest statement that the model knows less the further out it looks."""
    _, obs = tracking_data
    _, cov = _cv_filter().predict(obs, n_ahead=10)
    position_var = cov[:, 0, 0]
    assert np.all(np.diff(position_var) > 0)


def test_kalman_gain_follows_the_noise_balance():
    """The core idea: trust the measurement when it is clean, the model when it
    is noisy. A tiny R should track the observations almost exactly."""
    rng = np.random.RandomState(0)
    obs = np.cumsum(rng.normal(0, 1, 50))
    trusting = KalmanFilter(F=[[1]], H=[[1]], Q=[[1.0]], R=[[1e-6]], x0=[0])
    means, _ = trusting.filter(obs)
    # near-zero measurement noise => gain ~1 => the estimate hugs the data
    assert np.allclose(means[:, 0], obs, atol=1e-2)


def test_extended_kalman_filter_tracks_a_nonlinear_system():
    """A state observed through a nonlinear map -- here its square."""
    rng = np.random.RandomState(0)
    T = 80
    true = np.zeros(T)
    x = 1.0
    for t in range(T):
        x = 0.95 * x + rng.normal(0, 0.1)
        true[t] = x
    obs = true ** 3 + rng.normal(0, 0.1, T)     # nonlinear observation

    ekf = ExtendedKalmanFilter(
        f=lambda s: 0.95 * s,
        h=lambda s: s ** 3,
        F_jac=lambda s: np.array([[0.95]]),
        H_jac=lambda s: np.array([[3 * s[0] ** 2]]),
        Q=[[0.01]], R=[[0.01]], x0=[1.0], P0=[[1.0]])
    means, _ = ekf.filter(obs)
    # the cubic observation is nearly flat near zero, so the state is genuinely
    # hard to see there; against a state ranging over ~2 units, this is good
    assert np.sqrt(np.mean((means[:, 0] - true) ** 2)) < 0.4


def test_unscented_kalman_filter_tracks_a_nonlinear_system():
    rng = np.random.RandomState(0)
    T = 80
    true = np.zeros(T)
    x = 1.0
    for t in range(T):
        x = 0.95 * x + rng.normal(0, 0.1)
        true[t] = x
    obs = true ** 3 + rng.normal(0, 0.1, T)

    ukf = UnscentedKalmanFilter(
        f=lambda s: 0.95 * s, h=lambda s: s ** 3,
        Q=[[0.01]], R=[[0.01]], x0=[1.0], P0=[[1.0]])
    means, _ = ukf.filter(obs)
    assert np.sqrt(np.mean((means[:, 0] - true) ** 2)) < 0.4


def test_particle_filter_tracks_a_state():
    """The most general filter: only needs to sample the dynamics and score the
    likelihood, so it works when nothing is Gaussian."""
    rng = np.random.RandomState(0)
    T = 60
    true = np.cumsum(rng.normal(0, 0.5, T))
    obs = true + rng.normal(0, 1.0, T)

    def transition(particles, r):
        return particles + r.normal(0, 0.5, particles.shape)

    def likelihood(particles, z):
        return np.exp(-0.5 * ((particles[:, 0] - z) / 1.0) ** 2)

    pf = ParticleFilter(transition, likelihood, n_particles=2000,
                        init_sampler=lambda n, r: r.normal(0, 1, (n, 1)),
                        random_state=0)
    means = pf.filter(obs)
    assert np.sqrt(np.mean((means[:, 0] - true) ** 2)) < 1.0


def test_particle_filter_resamples_when_it_degenerates():
    """Effective sample size tracks weight concentration; it must stay well above
    1, or the cloud has collapsed to a handful of particles."""
    rng = np.random.RandomState(0)
    obs = np.cumsum(rng.normal(0, 0.5, 40))

    def transition(particles, r):
        return particles + r.normal(0, 0.5, particles.shape)

    def likelihood(particles, z):
        return np.exp(-0.5 * ((particles[:, 0] - z) / 1.0) ** 2)

    pf = ParticleFilter(transition, likelihood, n_particles=1000,
                        init_sampler=lambda n, r: r.normal(0, 1, (n, 1)),
                        random_state=0)
    pf.filter(obs)
    assert min(pf.ess_) > 1.0


# --- ARIMA ----------------------------------------------------------------

def test_differencing_removes_a_linear_trend():
    """The I in ARIMA: a linear trend has a constant first difference, which is
    stationary. This is why differencing is how ARIMA handles trends."""
    x = np.arange(50) * 2.0 + 5.0
    d1 = difference(x, 1)
    assert np.allclose(d1, 2.0)               # constant => stationary


def test_integrate_reconstructs_from_the_last_value():
    """integrate is the forecast-side inverse of differencing: given the observed
    series and a run of future differences, it cumulatively sums them onto the
    last value to recover the forecast on the original scale."""
    rng = np.random.RandomState(0)
    x = np.cumsum(rng.normal(size=20))
    future_diffs = np.array([0.5, 0.5, 0.5])
    forecast = integrate(future_diffs, x.copy(), 1)
    assert np.allclose(forecast, x[-1] + np.cumsum(future_diffs))


def test_autoregressive_recovers_a_known_coefficient():
    """An AR(1) with phi=0.7 should be fit with a coefficient near 0.7."""
    rng = np.random.RandomState(0)
    n = 2000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + rng.normal(0, 1)
    ar = AutoRegressive(p=1).fit(x)
    assert ar.coef_[0] == pytest.approx(0.7, abs=0.05)


def test_ar_forecast_decays_toward_the_mean():
    """A stationary AR process forgets its start geometrically; the long-run
    forecast is just the series mean."""
    rng = np.random.RandomState(0)
    x = np.zeros(500)
    for t in range(1, 500):
        x[t] = 0.5 * x[t - 1] + rng.normal(0, 1)
    ar = AutoRegressive(p=1).fit(x)
    fc = ar.forecast(50)
    # the tail of the forecast sits near the mean, not near the last value
    assert abs(fc[-1] - x.mean()) < abs(fc[0] - x.mean()) + 1.0
    assert abs(fc[-1] - ar.mean_) < 0.5


def test_arma_fits_and_forecasts():
    rng = np.random.RandomState(0)
    n = 500
    e = rng.normal(0, 1, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + e[t] + 0.3 * e[t - 1]
    arma = ARMA(p=1, q=1).fit(x)
    assert arma.ar_coef_[0] == pytest.approx(0.6, abs=0.15)
    assert len(arma.forecast(5)) == 5


def test_arima_forecasts_a_trend():
    """ARIMA(p,1,q) should extend a trend, because differencing turns the trend
    into a stationary level it can model and then reintegrate."""
    rng = np.random.RandomState(0)
    x = np.arange(100) * 0.5 + rng.normal(0, 0.5, 100)
    arima = ARIMA(p=1, d=1, q=0).fit(x)
    fc = arima.forecast(10)
    # the forecast keeps climbing at roughly the trend slope
    assert fc[-1] > fc[0]
    assert fc[-1] > x[-1]


# --- exponential smoothing ------------------------------------------------

def test_simple_exponential_smoothing_forecasts_flat():
    """No trend term means it can only forecast a horizontal line."""
    rng = np.random.RandomState(0)
    x = 10 + rng.normal(0, 1, 100)
    ses = SimpleExponentialSmoothing(alpha=0.3).fit(x)
    fc = ses.forecast(5)
    assert np.allclose(fc, fc[0])             # a single flat level
    assert abs(fc[0] - 10) < 1.0


def test_holt_extends_a_trend():
    """Adding a trend term lets it forecast a slope, unlike simple smoothing."""
    x = np.arange(50) * 2.0 + 5.0
    holt = Holt(alpha=0.5, beta=0.3).fit(x)
    fc = holt.forecast(5)
    assert np.all(np.diff(fc) > 0)           # it keeps rising
    assert fc[0] == pytest.approx(x[-1] + 2.0, abs=1.0)


def test_damped_trend_flattens():
    """Damping stops an extrapolated trend running to absurdity: the forecast
    increments should shrink, not stay constant."""
    x = np.arange(40) * 2.0
    damped = Holt(alpha=0.5, beta=0.3, damped=True, phi=0.9).fit(x)
    fc = damped.forecast(20)
    increments = np.diff(fc)
    assert increments[-1] < increments[0]    # the slope tapers off


def test_holt_winters_captures_a_season():
    """The full method should forecast a repeating seasonal pattern."""
    rng = np.random.RandomState(0)
    t = np.arange(48)
    x = 50 + t * 0.5 + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, 48)
    hw = HoltWinters(season_length=12).fit(x)
    fc = hw.forecast(12)
    # a full season of forecast must swing, not sit flat
    assert fc.max() - fc.min() > 10


def test_holt_winters_needs_two_seasons():
    with pytest.raises(ValueError, match="two full seasons"):
        HoltWinters(season_length=12).fit(np.arange(20))


def test_seasonal_decompose_separates_the_parts():
    rng = np.random.RandomState(0)
    t = np.arange(60)
    season = 10 * np.sin(2 * np.pi * t / 12)
    x = 50 + t * 0.3 + season + rng.normal(0, 0.5, 60)
    d = seasonal_decompose(x, season_length=12)
    # the recovered seasonal component correlates with the true one
    valid = ~np.isnan(d["trend"])
    assert np.corrcoef(d["seasonal"], season)[0, 1] > 0.9


def test_stl_recovers_trend_and_season():
    """STL must put the trend in the trend and the season in the season -- the
    failure mode is the trend leaking into the seasonal subseries."""
    rng = np.random.RandomState(0)
    t = np.arange(48)
    x = 50 + t * 0.5 + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, 48)
    stl = STL(season_length=12).fit(x)
    # the trend genuinely rises rather than coming out flat
    assert stl.trend_[-6] - stl.trend_[6] > 10
    # the season is recovered and the remainder is small
    assert stl.seasonal_.max() - stl.seasonal_.min() > 12
    assert stl.remainder_.std() < 3


# --- sequence distances ---------------------------------------------------

def test_dtw_ignores_speed_differences():
    """The point of DTW: the same shape at different speeds is close, where
    Euclidean distance would call it far apart."""
    a = np.sin(np.linspace(0, 2 * np.pi, 50))
    b = np.sin(np.linspace(0, 2 * np.pi, 90))    # same wave, different sampling
    unrelated = np.cos(np.linspace(0, 2 * np.pi, 90))
    assert dtw_distance(a, b) < dtw_distance(a, unrelated)


def test_dtw_of_identical_sequences_is_zero():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert dtw_distance(a, a) == pytest.approx(0.0)


def test_dtw_path_aligns_the_endpoints():
    a = np.array([0.0, 1.0, 2.0, 3.0])
    b = np.array([0.0, 2.0, 3.0])
    path, cost = dtw_path(a, b)
    assert path[0] == (0, 0)
    assert path[-1] == (len(a) - 1, len(b) - 1)


def test_dtw_window_bounds_the_warp():
    a = np.random.RandomState(0).normal(size=40)
    b = np.random.RandomState(1).normal(size=40)
    # a band can only make the distance larger or equal (it forbids alignments)
    assert dtw_distance(a, b, window=5) >= dtw_distance(a, b) - 1e-9


def test_levenshtein_known_values():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("same", "same") == 0


def test_damerau_counts_a_transposition_as_one_edit():
    """The commonest typo: Levenshtein charges a swap as two edits, Damerau as
    one, which is why it is the better measure for typed text."""
    assert damerau_levenshtein("teh", "the") == 1
    assert levenshtein("teh", "the") == 2


def test_hamming_requires_equal_length():
    assert hamming("1011", "1001") == 1
    with pytest.raises(ValueError):
        hamming("abc", "ab")


def test_jaro_winkler_rewards_a_shared_prefix():
    """Winkler's boost: names sharing a start score higher, which suits record
    linkage."""
    base = jaro("MARTHA", "MARHTA")
    boosted = jaro_winkler("MARTHA", "MARHTA")
    assert boosted >= base
    assert jaro_winkler("MARTHA", "MARHTA") == pytest.approx(0.961, abs=0.01)


def test_lcs_finds_the_common_subsequence():
    assert longest_common_subsequence("ABCBDAB", "BDCAB") == 4
    assert longest_common_subsequence("abc", "xyz") == 0


def test_needleman_wunsch_scores_alignment():
    """Edit distance recast as a score to maximise, the bioinformatics framing."""
    assert needleman_wunsch("GATTACA", "GATTACA") == 7      # all matches, +1 each
    assert needleman_wunsch("AAAA", "TTTT") < 0             # all mismatches


# --- CRF and structured perceptron ----------------------------------------

def _tagging_data(n_seqs=80, seed=1):
    """Labels with a strong tendency to alternate -- a dependency between
    neighbouring tags that only a joint model can exploit."""
    rng = np.random.RandomState(seed)
    X, Y = [], []
    for _ in range(n_seqs):
        length = rng.randint(6, 12)
        feats, tags = [], []
        prev = 0
        for _ in range(length):
            tag = 1 - prev if rng.uniform() < 0.85 else prev
            f = np.zeros(3)
            f[tag] = 1.0                      # a weak per-token signal
            f[2] = rng.normal()               # plus noise
            feats.append(f)
            tags.append(tag)
            prev = tag
        X.append(np.array(feats))
        Y.append(tags)
    return X, Y


def test_crf_learns_to_tag():
    X, Y = _tagging_data()
    crf = LinearChainCRF(n_states=2, n_epochs=30, random_state=0).fit(X, Y)
    assert crf.score(X, Y) > 0.9


def test_crf_log_likelihood_increases():
    X, Y = _tagging_data()
    crf = LinearChainCRF(n_states=2, n_epochs=30, random_state=0).fit(X, Y)
    assert crf.log_likelihood_[-1] > crf.log_likelihood_[0]


def test_crf_learns_the_transition_structure():
    """The alternating tendency should show up as the off-diagonal transitions
    being preferred over staying in the same state."""
    X, Y = _tagging_data()
    crf = LinearChainCRF(n_states=2, n_epochs=40, random_state=0).fit(X, Y)
    trans = crf.transitions_
    # switching (0->1, 1->0) should outscore staying (0->0, 1->1)
    assert trans[0, 1] > trans[0, 0]
    assert trans[1, 0] > trans[1, 1]


def test_structured_perceptron_learns_to_tag():
    X, Y = _tagging_data()
    sp = StructuredPerceptron(n_states=2, n_epochs=20, random_state=0).fit(X, Y)
    assert sp.score(X, Y) > 0.85


def test_structured_perceptron_averaging_helps():
    """Averaging is what makes the structured perceptron generalise, not an
    optional extra -- the averaged weights should not be worse."""
    Xtr, Ytr = _tagging_data(n_seqs=60, seed=1)
    Xte, Yte = _tagging_data(n_seqs=40, seed=2)
    averaged = StructuredPerceptron(n_states=2, n_epochs=15, average=True,
                                    random_state=0).fit(Xtr, Ytr)
    plain = StructuredPerceptron(n_states=2, n_epochs=15, average=False,
                                 random_state=0).fit(Xtr, Ytr)
    assert averaged.score(Xte, Yte) >= plain.score(Xte, Yte) - 0.05
