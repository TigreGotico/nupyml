"""G8: variational inference, sequential Monte Carlo, ABC, and GPLVM.

VI and SMC must recover a conjugate-Gaussian posterior (known mean/std); ABC must
recover a parameter with no likelihood, only a simulator; GPLVM must recover a 1-D
latent that generated 3-D data (a nonlinear embedding PCA cannot fully unfold).
"""
import numpy as np
import pytest

from nupyml.inference import MeanFieldVI, SequentialMonteCarlo, ABC, GPLVM


@pytest.fixture
def gaussian_posterior():
    """Data ~ N(mu, 1), prior mu ~ N(0, 100). Posterior mean ~ sample mean,
    posterior std ~ 1/sqrt(n)."""
    rng = np.random.RandomState(0)
    data = rng.randn(50) + 3.0
    post_mean = data.sum() / (50 + 0.01)
    post_std = 1.0 / np.sqrt(50 + 0.01)
    log_post = lambda th: -0.5 * np.sum((data - th[0]) ** 2) - 0.5 * th[0] ** 2 / 100
    return data, log_post, post_mean, post_std


def test_mean_field_vi_recovers_gaussian_posterior(gaussian_posterior):
    data, log_post, pm, ps = gaussian_posterior
    vi = MeanFieldVI(log_post, dim=1, lr=0.1, n_iter=400, random_state=0).fit()
    assert abs(vi.mean_[0] - pm) < 0.15
    assert abs(vi.std_[0] - ps) < 0.05                # VI recovers the spread too


def test_smc_recovers_posterior_mean(gaussian_posterior):
    data, _, pm, _ = gaussian_posterior
    smc = SequentialMonteCarlo(
        log_prior=lambda t: -0.5 * t[0] ** 2 / 100,
        log_likelihood=lambda t: -0.5 * np.sum((data - t[0]) ** 2),
        dim=1, n_particles=500, random_state=0).fit()
    assert abs(smc.posterior_mean_[0] - pm) < 0.3
    assert smc.particles_.std() > 0                   # a genuine cloud, not collapsed


def test_abc_infers_without_a_likelihood():
    obs = [3.0]                                       # observed summary = mean

    def prior(rng):
        return rng.uniform(-5, 5)

    def simulator(theta, rng):                        # simulate, return the summary
        return [np.mean(rng.randn(50) + theta)]

    abc = ABC(prior, simulator, epsilon=0.15, n_accept=150, random_state=0).fit(obs)
    assert abs(abc.posterior_mean_[0] - 3.0) < 0.2
    assert 0 < abc.acceptance_rate_ < 1


def test_gplvm_recovers_nonlinear_latent():
    rng = np.random.RandomState(0)
    t = np.linspace(-2, 2, 80)
    # 3-D data generated from a single latent t via nonlinear coordinates
    Y = np.column_stack([t, t ** 3, np.tanh(t)]) + 0.02 * rng.randn(80, 3)
    gplvm = GPLVM(n_components=1, length_scale=1.5, lr=0.05, n_iter=80,
                  random_state=0)
    X = gplvm.fit_transform(Y)
    assert X.shape == (80, 1)
    # the recovered latent is (anti-)monotonic in the true generator
    assert abs(np.corrcoef(X[:, 0], t)[0, 1]) > 0.9
