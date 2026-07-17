"""VQ-VAE, normalizing flows, diffusion, and RBMs.

Each model here claims something specific -- an exact density, a discrete code, a
tractable free energy -- and each test tries to hold it to that claim rather than
just checking it trains.
"""
import numpy as np
import pytest

from nupyml.autograd import Tensor
from nupyml.datasets import load_digits
from nupyml.nn import (
    VQVAE, VectorQuantizer, RealNVP, AffineCoupling, DDPM, BernoulliRBM, Adam,
    linear_beta_schedule, cosine_beta_schedule, timestep_embedding,
)


@pytest.fixture(scope="module")
def digits():
    X, _ = load_digits(return_X_y=True)
    return X / 16.0


@pytest.fixture(scope="module")
def bimodal():
    rng = np.random.RandomState(0)
    return np.vstack([rng.normal([2, 2], 0.3, (1000, 2)),
                      rng.normal([-2, -2], 0.3, (1000, 2))])


def _batches(n, rng, size=32):
    idx = rng.permutation(n)
    return [idx[i:i + size] for i in range(0, n - size + 1, size)]


def _train(model, X, epochs, lr, rng=0):
    opt = Adam(model.parameters(), lr=lr)
    r = np.random.RandomState(rng)
    model.train()
    for _ in range(epochs):
        for b in _batches(len(X), r, 64):
            opt.zero_grad()
            model.loss(X[b]).backward()
            opt.step()
    return model


def _near_a_mode(s, tol=1.0):
    d = np.minimum(np.linalg.norm(s - [2, 2], axis=1),
                   np.linalg.norm(s - [-2, -2], axis=1))
    return (d < tol).mean()


# --- VQ-VAE: the discrete latent -----------------------------------------

def test_quantizer_output_is_exactly_a_codebook_entry():
    """Forward, the straight-through algebra must cancel to the quantized value."""
    q = VectorQuantizer(8, 3, rng=0)
    z_e = Tensor(np.random.RandomState(0).normal(size=(5, 3)))
    z_q, _, idx = q(z_e)
    assert np.allclose(z_q.data, q.codebook.data[idx])


def test_quantizer_picks_the_nearest_code():
    """The matmul distance shortcut must agree with the literal computation."""
    q = VectorQuantizer(8, 3, rng=0)
    z = np.random.RandomState(1).normal(size=(20, 3))
    brute = np.array([np.argmin(((q.codebook.data - row) ** 2).sum(1)) for row in z])
    assert np.array_equal(q.encode_indices(Tensor(z)), brute)


def test_straight_through_passes_gradient_unchanged():
    """The lie that makes VQ-VAE trainable: d z_q / d z_e is treated as identity."""
    q = VectorQuantizer(8, 3, beta=0.0, rng=0)
    z_e = Tensor(np.random.RandomState(0).normal(size=(4, 3)), requires_grad=True)
    z_q, _, _ = q(z_e)
    (z_q * Tensor(np.full((4, 3), 2.0))).sum().backward()
    # the gradient arrives exactly as it left the decoder -- argmin contributes
    # nothing, which is the entire point of the estimator
    assert np.allclose(z_e.grad, 2.0)


def test_argmin_alone_would_give_no_gradient():
    """Why the estimator is needed at all: quantizing without it kills the tape.

    A plain lookup severs the graph so completely that the result is not even
    connected to ``z_e`` -- there is nothing to differentiate, and the encoder
    could never learn.
    """
    q = VectorQuantizer(8, 3, rng=0)
    z_e = Tensor(np.random.RandomState(0).normal(size=(4, 3)), requires_grad=True)
    naive = Tensor(q.codebook.data[q.encode_indices(z_e)])  # a plain lookup
    assert not naive.requires_grad
    with pytest.raises(RuntimeError):
        naive.sum().backward()


def test_commitment_loss_pulls_the_encoder_not_the_codebook():
    q = VectorQuantizer(4, 2, beta=1.0, rng=0)
    z_e = Tensor(np.full((3, 2), 5.0), requires_grad=True)  # far from any code
    _, loss, _ = q(z_e)
    loss.backward()
    assert np.any(z_e.grad)          # the encoder is told to come closer
    assert np.any(q.codebook.grad)   # and the codes are told to move out


def test_vqvae_reconstructs_and_keeps_its_codebook_alive(digits):
    X = digits
    vq = _train(VQVAE(64, code_dim=8, n_codes=32, hidden=(32,), rng=0), X, 40, 1e-2)
    vq.eval()
    recon, _, idx = vq(X)
    assert np.mean((recon.data - X) ** 2) < np.mean((X.mean(axis=0) - X) ** 2) / 2
    # codebook collapse is this model's characteristic failure -- most codes must
    # actually be in use, or reconstruction is capped no matter the training
    assert vq.perplexity(X) > 16
    assert len(np.unique(idx)) > 16


def test_vqvae_decodes_from_symbols_alone(digits):
    """The payoff of a discrete latent: indices are enough to reconstruct."""
    X = digits
    vq = _train(VQVAE(64, code_dim=8, n_codes=32, hidden=(32,), rng=0), X, 20, 1e-2)
    vq.eval()
    recon, _, idx = vq(X[:16])
    assert np.allclose(vq.decode_indices(idx), recon.data)


def test_vqvae_latent_is_a_finite_vocabulary(digits):
    vq = VQVAE(64, code_dim=4, n_codes=8, hidden=(16,), rng=0)
    _, _, idx = vq(digits[:50])
    assert set(np.unique(idx)).issubset(set(range(8)))


# --- flows: the exact likelihood -----------------------------------------

def test_coupling_layer_is_exactly_invertible():
    layer = AffineCoupling(4, [1, 0, 1, 0], hidden=(16,), rng=0)
    x = np.random.RandomState(0).normal(size=(6, 4))
    y, ld = layer(x)
    back, ld_inv = layer.inverse(y)
    assert np.allclose(back.data, x, atol=1e-10)
    assert np.allclose(ld.data, -ld_inv.data, atol=1e-10)


def test_coupling_layer_leaves_the_frozen_half_untouched():
    mask = np.array([1, 0, 1, 0])
    layer = AffineCoupling(4, mask, hidden=(16,), rng=0)
    x = np.random.RandomState(0).normal(size=(6, 4))
    y, _ = layer(x)
    frozen = mask.astype(bool)
    assert np.allclose(y.data[:, frozen], x[:, frozen])
    assert not np.allclose(y.data[:, ~frozen], x[:, ~frozen])


def test_flow_round_trips_to_machine_precision():
    """A flow is a bijection or it is nothing -- the likelihood depends on it."""
    f = RealNVP(4, n_layers=4, hidden=(16,), rng=0)
    x = np.random.RandomState(0).normal(size=(8, 4))
    z, ld = f.inverse(x)
    back, ld_fwd = f(z)
    assert np.allclose(back.data, x, atol=1e-10)
    assert np.allclose(ld.data, -ld_fwd.data, atol=1e-10)


def test_flow_density_integrates_to_one():
    """The claim a flow makes that no other generative model here can: this is a
    normalized density, not a bound and not a score."""
    f = RealNVP(2, n_layers=4, hidden=(16,), rng=0)
    g = np.linspace(-16, 16, 700)
    step = g[1] - g[0]
    grid = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    assert np.exp(f.log_prob(grid).data).sum() * step ** 2 == pytest.approx(1.0, abs=1e-3)


def test_flow_log_det_matches_the_numerical_jacobian():
    """The triangular-Jacobian shortcut must equal the real determinant."""
    f = RealNVP(3, n_layers=2, hidden=(8,), rng=0)
    x = np.random.RandomState(0).normal(size=(1, 3))
    eps = 1e-6
    jac = np.zeros((3, 3))
    for j in range(3):
        step = np.zeros((1, 3))
        step[0, j] = eps
        jac[:, j] = ((f(x + step)[0].data - f(x - step)[0].data) / (2 * eps))[0]
    _, log_det = f(x)
    assert log_det.data[0] == pytest.approx(np.log(abs(np.linalg.det(jac))), abs=1e-5)


def test_flow_learns_a_density_and_scores_it(bimodal):
    f = _train(RealNVP(2, n_layers=6, hidden=(64,), rng=0), bimodal, 120, 1e-3)
    on = f.log_prob(bimodal).data.mean()
    off = f.log_prob(np.random.RandomState(3).normal(0, 4, (500, 2))).data.mean()
    assert on > off


def test_flow_covers_both_modes(bimodal):
    """Maximum likelihood cannot ignore a mode: missing one sends its log-prob to
    minus infinity. This is the structural reason flows do not mode-collapse."""
    f = _train(RealNVP(2, n_layers=6, hidden=(64,), rng=0), bimodal, 120, 1e-3)
    s = f.sample(1000, rng=1)
    assert _near_a_mode(s) > 0.8
    assert 0.3 < (s[:, 0] > 0).mean() < 0.7


# --- diffusion ------------------------------------------------------------

@pytest.mark.parametrize("T", [50, 100, 200, 1000])
def test_beta_schedules_destroy_the_signal_monotonically(T):
    for betas in (linear_beta_schedule(T), cosine_beta_schedule(T)):
        alpha_bar = np.cumprod(1 - betas)
        assert np.all(np.diff(alpha_bar) < 0)   # signal only ever fades
        assert alpha_bar[0] > 0.99              # step 1 barely touches the data
        assert alpha_bar[-1] < 0.01             # the end really is noise


@pytest.mark.parametrize("T", [10, 20, 50, 200, 1000])
def test_betas_stay_a_valid_variance(T):
    """beta >= 1 destroys the sample and makes the reverse step divide by zero.

    The linear rate is scaled by 1000/T, which reaches 1.0 at T=20 -- so short
    chains depend on the clip to stay valid rather than silently going NaN.
    """
    for betas in (linear_beta_schedule(T), cosine_beta_schedule(T)):
        assert betas.min() >= 0.0
        assert betas.max() < 1.0
        assert np.all(np.cumprod(1 - betas) > 0.0)


@pytest.mark.parametrize("T", [50, 200, 1000])
def test_cosine_schedule_preserves_signal_longer_than_linear(T):
    """The point of the cosine schedule: don't waste the middle of the chain.

    True at every T only because the linear rate is scaled -- unscaled at small
    T the linear schedule under-noises and this comparison inverts.
    """
    assert np.cumprod(1 - cosine_beta_schedule(T))[T // 2] > \
        np.cumprod(1 - linear_beta_schedule(T))[T // 2]


@pytest.mark.parametrize("T", [50, 200])
def test_forward_process_ends_where_sampling_begins(T):
    """The consistency the scaling exists to protect: sampling starts from
    N(0, I), so the forward chain must actually get there. If it stops short,
    the model is asked at generation time for a noise level it never trained on.
    """
    d = DDPM(2, T=T, rng=0)
    x_T, _ = d.q_sample(np.random.RandomState(0).normal(0, 1, (4000, 2)),
                        np.full(4000, T - 1))
    assert abs(x_T.mean()) < 0.1
    assert x_T.std() == pytest.approx(1.0, abs=0.1)


def test_timestep_embedding_distinguishes_timesteps():
    emb = timestep_embedding(np.arange(50), 16)
    assert emb.shape == (50, 16)
    assert np.abs(emb).max() <= 1.0
    # adjacent steps must be more alike than distant ones, or the network cannot
    # read "how noisy" off the embedding
    assert np.linalg.norm(emb[0] - emb[1]) < np.linalg.norm(emb[0] - emb[40])


def test_q_sample_keeps_variance_near_one(bimodal):
    """The coefficients are chosen so data dissolves rather than blows up."""
    d = DDPM(2, T=100, rng=0)
    x0 = bimodal / bimodal.std()
    for t in (0, 50, 99):
        x_t, _ = d.q_sample(x0, np.full(len(x0), t))
        assert 0.5 < x_t.std() < 2.0


def test_q_sample_at_the_end_is_essentially_noise(bimodal):
    d = DDPM(2, T=200, rng=0)
    x_t, _ = d.q_sample(bimodal, np.full(len(bimodal), 199))
    # the two clusters must no longer be distinguishable in the mean
    assert abs(x_t[:1000].mean() - x_t[1000:].mean()) < 0.5


def test_q_sample_is_the_closed_form_of_the_chain():
    """The one-shot jump must agree with simulating the forward chain step by
    step -- this is the identity that makes training O(1) instead of O(T)."""
    d = DDPM(3, T=50, rng=0)
    rng = np.random.RandomState(0)
    x0 = rng.normal(size=(20000, 3))

    stepwise = x0.copy()
    for t in range(10):
        stepwise = (np.sqrt(1 - d.betas[t]) * stepwise
                    + np.sqrt(d.betas[t]) * rng.normal(size=x0.shape))
    closed, _ = d.q_sample(x0, np.full(len(x0), 9))
    # both are random, so compare the distributions they define
    assert stepwise.std() == pytest.approx(closed.std(), abs=0.02)
    assert np.corrcoef(stepwise[:, 0], x0[:, 0])[0, 1] == pytest.approx(
        np.corrcoef(closed[:, 0], x0[:, 0])[0, 1], abs=0.02)


def test_ddpm_predicts_the_noise_it_added(bimodal):
    d = _train(DDPM(2, T=100, hidden=(64, 64), rng=0), bimodal, 80, 2e-3)
    d.eval()
    # a network that has learned nothing predicts ~0 and scores 1.0 (the noise
    # is unit variance), so beating that is the real bar
    assert d.loss(bimodal).item() < 0.8


def test_ddpm_covers_both_modes(bimodal):
    d = _train(DDPM(2, T=100, hidden=(64, 64), rng=0), bimodal, 120, 2e-3)
    s = d.sample(1000, rng=1)
    assert _near_a_mode(s) > 0.8
    assert 0.25 < (s[:, 0] > 0).mean() < 0.75


def test_reverse_process_is_stochastic():
    """Same starting noise, different trajectories: the re-noising term at work."""
    d = DDPM(2, T=20, hidden=(16,), rng=0)
    assert not np.allclose(d.sample(8, rng=1), d.sample(8, rng=2))


def test_sampling_trajectory_walks_from_noise_to_data(bimodal):
    d = _train(DDPM(2, T=100, hidden=(64, 64), rng=0), bimodal, 80, 2e-3)
    final, traj = d.sample(500, rng=1, return_trajectory=True)
    assert len(traj) == 101
    # it must end closer to the data than it started, which is the only thing the
    # reverse chain is for
    assert _near_a_mode(final) > _near_a_mode(traj[0])


# --- RBM ------------------------------------------------------------------

@pytest.fixture(scope="module")
def binary_digits(digits):
    return (digits > 0.3).astype(np.float64)


def test_rbm_gives_data_lower_free_energy_than_noise(binary_digits):
    """Training sculpts an energy landscape: down on data, up elsewhere."""
    rbm = BernoulliRBM(n_components=32, n_iter=20, random_state=0).fit(binary_digits)
    noise = np.random.RandomState(1).randint(0, 2, binary_digits.shape).astype(float)
    assert rbm.free_energy(binary_digits).mean() < rbm.free_energy(noise).mean()


def test_rbm_free_energy_matches_the_explicit_sum_over_hidden():
    """The closed form must equal literally summing out the hidden units.

    Only tractable because the hidden units are independent given v -- the
    restriction the model is named for.
    """
    rbm = BernoulliRBM(n_components=4, n_iter=1, random_state=0)
    rng = np.random.RandomState(0)
    rbm.components_ = rng.normal(0, 0.5, size=(4, 3))
    rbm.intercept_hidden_ = rng.normal(size=4)
    rbm.intercept_visible_ = rng.normal(size=3)

    v = np.array([[1.0, 0.0, 1.0]])
    total = sum(
        np.exp(v @ rbm.intercept_visible_
               + h @ rbm.intercept_hidden_
               + v @ rbm.components_.T @ h)
        for h in np.array(np.meshgrid(*[[0, 1]] * 4)).T.reshape(-1, 4).astype(float))
    assert rbm.free_energy(v)[0] == pytest.approx(-np.log(total)[0])


def test_rbm_score_samples_prefers_data(binary_digits):
    rbm = BernoulliRBM(n_components=32, n_iter=20, random_state=0).fit(binary_digits)
    noise = np.random.RandomState(2).randint(0, 2, binary_digits.shape).astype(float)
    assert rbm.score_samples(binary_digits).mean() > rbm.score_samples(noise).mean()


def test_rbm_transform_returns_probabilities(binary_digits):
    rbm = BernoulliRBM(n_components=16, n_iter=5, random_state=0).fit(binary_digits)
    h = rbm.transform(binary_digits)
    assert h.shape == (len(binary_digits), 16)
    assert h.min() >= 0.0 and h.max() <= 1.0


def test_rbm_sigmoid_does_not_overflow():
    """Saturated units are normal; the naive 1/(1+exp(-x)) form warns and lies."""
    from nupyml.nn.rbm import _sigmoid
    with np.errstate(over="raise"):
        out = _sigmoid(np.array([-800.0, -1.0, 0.0, 1.0, 800.0]))
    assert out[0] == pytest.approx(0.0)
    assert out[2] == pytest.approx(0.5)
    assert out[-1] == pytest.approx(1.0)


def test_rbm_reconstruction_beats_a_random_model(binary_digits):
    rbm = BernoulliRBM(n_components=64, n_iter=20, random_state=0).fit(binary_digits)
    trained = (rbm.gibbs(binary_digits, 1, random_state=0) == binary_digits).mean()
    untrained = BernoulliRBM(n_components=64, n_iter=0, random_state=0).fit(binary_digits)
    naive = (untrained.gibbs(binary_digits, 1, random_state=0) == binary_digits).mean()
    assert trained > naive


def test_rbm_gibbs_chain_stays_in_the_data_regime(binary_digits):
    """A long chain wanders toward the MODEL's distribution. If it has learned
    anything, that still looks broadly like data -- here, similarly sparse."""
    rbm = BernoulliRBM(n_components=64, n_iter=20, random_state=0).fit(binary_digits)
    far = rbm.gibbs(binary_digits[:200], n_steps=50, random_state=0)
    assert abs(far.mean() - binary_digits.mean()) < 0.15


def test_rbm_partial_fit_accumulates(binary_digits):
    rbm = BernoulliRBM(n_components=16, random_state=0)
    for chunk in np.array_split(binary_digits, 10):
        rbm.partial_fit(chunk)
    assert rbm.components_.shape == (16, 64)
    assert np.any(rbm.components_ != 0)
