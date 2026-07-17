"""Autoencoders, VAEs and GANs: reconstruction, the ELBO, and adversarial training."""
import numpy as np
import pytest

from nupyml.autograd import Tensor
from nupyml.datasets import load_digits
from nupyml.nn import (
    AutoEncoder, DenoisingAutoEncoder, SparseAutoEncoder, VAE, ConditionalVAE,
    VAELoss, kl_divergence_normal, Adam, MSELoss, GAN, Generator, Discriminator,
    GANLoss, NonSaturatingGANLoss, WGANLoss, Tanh,
)


@pytest.fixture(scope="module")
def digits():
    X, y = load_digits(return_X_y=True)
    return X / 16.0, y


def _batches(n, rng, size=32):
    idx = rng.permutation(n)
    return [idx[i:i + size] for i in range(0, n - size + 1, size)]


def _train_ae(model, X, epochs=30, lr=1e-2):
    """Plain reconstruction training, shared by every autoencoder variant."""
    opt, mse = Adam(model.parameters(), lr=lr), MSELoss()
    rng = np.random.RandomState(0)
    model.train()
    for _ in range(epochs):
        for b in _batches(len(X), rng):
            opt.zero_grad()
            if isinstance(model, SparseAutoEncoder):
                loss = model.loss(X[b], mse)
            else:
                loss = mse(model(X[b]), X[b])
            loss.backward()
            opt.step()
    return model


def _train_vae(vae, X, epochs=30, lr=1e-2, beta=1.0):
    opt, loss_fn = Adam(vae.parameters(), lr=lr), VAELoss(beta=beta)
    rng = np.random.RandomState(0)
    vae.train()
    for _ in range(epochs):
        for b in _batches(len(X), rng):
            opt.zero_grad()
            recon, mu, logvar = vae(X[b])
            loss_fn(recon, X[b], mu, logvar).backward()
            opt.step()
    return vae


def _recon_mse(model, X):
    model.eval()
    out = model(X)
    return np.mean(((out[0] if isinstance(out, tuple) else out).data - X) ** 2)


# --- autoencoders ---------------------------------------------------------

def test_autoencoder_beats_predicting_the_mean(digits):
    X, _ = digits
    ae = _train_ae(AutoEncoder(64, latent_dim=8, hidden=(32,), rng=0), X)
    assert _recon_mse(ae, X) < np.mean((X.mean(axis=0) - X) ** 2) / 2


def test_encode_decode_round_trip_shapes(digits):
    X, _ = digits
    ae = AutoEncoder(64, latent_dim=5, hidden=(32,), rng=0)
    z = ae.encode(X[:10])
    assert z.shape == (10, 5)
    assert ae.decode(z).shape == (10, 64)


def test_a_wider_bottleneck_reconstructs_better(digits):
    X, _ = digits
    errs = [_recon_mse(_train_ae(AutoEncoder(64, d, hidden=(32,), rng=0), X), X)
            for d in (2, 16)]
    assert errs[1] < errs[0]


def test_denoising_autoencoder_denoises_better_than_a_plain_one(digits):
    X, _ = digits
    noisy = np.clip(X + np.random.RandomState(0).normal(0, 0.3, X.shape), 0, 1)
    scores = []
    for model in (AutoEncoder(64, 8, hidden=(32,), rng=0),
                  DenoisingAutoEncoder(64, 8, hidden=(32,), rng=0)):
        _train_ae(model, X)
        model.eval()
        scores.append(np.mean((model(noisy).data - X) ** 2))
    plain, denoising = scores
    assert denoising < plain


def test_denoising_autoencoder_corrupts_only_while_training(digits):
    X, _ = digits
    dae = DenoisingAutoEncoder(64, 8, hidden=(32,), rng=0)
    dae.train()
    assert not np.allclose(dae.corrupt(Tensor(X[:4])).data, X[:4])
    dae.eval()
    assert np.allclose(dae.corrupt(Tensor(X[:4])).data, X[:4])


def test_sparse_autoencoder_activations_are_sparser(digits):
    X, _ = digits
    acts = []
    for model in (AutoEncoder(64, 16, hidden=(32,), rng=0),
                  SparseAutoEncoder(64, 16, hidden=(32,), rho=0.05, beta=1.0,
                                    rng=0)):
        _train_ae(model, X)
        model.eval()
        acts.append(np.abs(model.encode(X).data).mean())
    plain, sparse = acts
    assert sparse < plain


# --- the VAE --------------------------------------------------------------

def test_kl_closed_form_matches_monte_carlo():
    """The analytic KL is why the VAE needs no sampling for that half of the ELBO."""
    rng = np.random.RandomState(0)
    mu = rng.normal(size=(1, 4))
    logvar = rng.normal(size=(1, 4)) * 0.5
    closed = kl_divergence_normal(Tensor(mu), Tensor(logvar)).item()

    std = np.exp(0.5 * logvar)
    z = mu + std * rng.normal(size=(400_000, 4))
    log_q = (-0.5 * (np.log(2 * np.pi) + logvar + ((z - mu) / std) ** 2)).sum(1)
    log_p = (-0.5 * (np.log(2 * np.pi) + z ** 2)).sum(1)
    assert closed == pytest.approx(np.mean(log_q - log_p), abs=0.02)


def test_kl_is_zero_exactly_at_the_prior():
    assert kl_divergence_normal(Tensor(np.zeros((3, 5))),
                                Tensor(np.zeros((3, 5)))).item() == pytest.approx(0.0)


def test_kl_is_nonnegative():
    rng = np.random.RandomState(1)
    for _ in range(20):
        kl = kl_divergence_normal(Tensor(rng.normal(size=(4, 3)) * 2),
                                  Tensor(rng.normal(size=(4, 3))))
        assert kl.item() >= -1e-12


def test_reparameterization_lets_gradient_reach_the_encoder(digits):
    """The whole point of the trick: sampling stays on the tape."""
    X, _ = digits
    vae = VAE(64, latent_dim=4, hidden=(32,), rng=0)
    vae.train()
    recon, mu, logvar = vae(X[:16])
    VAELoss()(recon, X[:16], mu, logvar).backward()
    grads = [p.grad for p in vae.encoder_body.parameters() if p.grad is not None]
    assert grads and any(np.any(g != 0) for g in grads)


def test_vae_is_deterministic_in_eval_mode(digits):
    """eval() returns the mean rather than a sample, so predictions repeat."""
    X, _ = digits
    vae = VAE(64, latent_dim=4, hidden=(32,), rng=0)
    vae.eval()
    assert np.allclose(vae(X[:8])[0].data, vae(X[:8])[0].data)


def test_vae_samples_are_stochastic_in_train_mode(digits):
    X, _ = digits
    vae = VAE(64, latent_dim=4, hidden=(32,), rng=0)
    vae.train()
    assert not np.allclose(vae(X[:8])[0].data, vae(X[:8])[0].data)


def test_vae_reconstructs_and_its_prior_samples_look_like_data(digits):
    X, _ = digits
    vae = _train_vae(VAE(64, latent_dim=8, hidden=(32,), rng=0), X, epochs=40)
    assert _recon_mse(vae, X) < np.mean((X.mean(axis=0) - X) ** 2) / 2

    # a VAE is generative: decoding prior noise must give data-like output,
    # which is exactly what the KL term buys and a plain AE cannot do
    samples = vae.sample(500, rng=0)
    assert 0.0 <= samples.min() and samples.max() <= 1.0
    assert abs(samples.mean() - X.mean()) < 0.1


def test_elbo_improves_during_training(digits):
    X, _ = digits
    vae = VAE(64, latent_dim=8, hidden=(32,), rng=0)
    loss_fn = VAELoss()

    def elbo():
        vae.eval()
        recon, mu, logvar = vae(X)
        return loss_fn(recon, X, mu, logvar).item()

    before = elbo()
    _train_vae(vae, X, epochs=20)
    assert elbo() < before


def test_beta_trades_reconstruction_against_the_latent(digits):
    """beta > 1 buys a tighter latent at the cost of reconstruction."""
    X, _ = digits
    out = []
    for beta in (1.0, 10.0):
        vae = _train_vae(VAE(64, 8, hidden=(32,), rng=0), X, epochs=25, beta=beta)
        vae.eval()
        mu, logvar = vae.encode(X)
        out.append((_recon_mse(vae, X), kl_divergence_normal(mu, logvar).item()))
    (recon_lo, kl_lo), (recon_hi, kl_hi) = out
    assert kl_hi < kl_lo
    assert recon_hi > recon_lo


def test_conditional_vae_respects_its_label(digits):
    """Decoding under different labels must give different digits."""
    X, y = digits
    cvae = ConditionalVAE(64, latent_dim=8, n_classes=10, hidden=(32,), rng=0)
    opt, loss_fn = Adam(cvae.parameters(), lr=1e-2), VAELoss()
    rng = np.random.RandomState(0)
    cvae.train()
    for _ in range(40):
        for b in _batches(len(X), rng):
            opt.zero_grad()
            recon, mu, logvar = cvae(X[b], y[b])
            loss_fn(recon, X[b], mu, logvar).backward()
            opt.step()
    zeros = cvae.sample(np.zeros(200, dtype=int), rng=0)
    ones = cvae.sample(np.ones(200, dtype=int), rng=0)
    assert np.mean((zeros.mean(0) - ones.mean(0)) ** 2) > 1e-3


# --- GANs -----------------------------------------------------------------

def _gaussian(n=2000, seed=0):
    return np.random.RandomState(seed).normal([3, -2], [0.5, 0.5], size=(n, 2))


def _fit_gan(gan, real, epochs=60, n_critic=1, grad_clip=None, lr=1e-3):
    og = Adam(gan.generator.parameters(), lr=lr)
    od = Adam(gan.discriminator.parameters(), lr=lr)
    rng = np.random.RandomState(0)
    for _ in range(epochs):
        idx = rng.permutation(len(real))
        for i in range(0, len(real), 64):
            gan.step(real[idx[i:i + 64]], og, od,
                     n_critic=n_critic, grad_clip=grad_clip)
    return gan


def test_saturating_loss_gives_a_vanishing_gradient():
    """The reason nobody uses the textbook generator loss."""
    grads = []
    for loss in (GANLoss(), NonSaturatingGANLoss()):
        # a discriminator that confidently calls every fake a fake
        logits = Tensor(np.full(64, -8.0), requires_grad=True)
        loss.generator_loss(logits).backward()
        grads.append(np.abs(logits.grad).mean())
    saturating, non_saturating = grads
    assert saturating < non_saturating / 100


def test_the_two_generator_losses_agree_on_which_way_is_better():
    """Same fixed point, different gradient scale: both prefer a higher logit."""
    for loss in (GANLoss(), NonSaturatingGANLoss()):
        worse = loss.generator_loss(Tensor(np.full(8, -2.0))).item()
        better = loss.generator_loss(Tensor(np.full(8, 2.0))).item()
        assert better < worse


def test_discriminator_loss_is_lowest_when_it_is_right():
    loss = NonSaturatingGANLoss()
    right = loss.discriminator_loss(Tensor(np.full(8, 5.0)), Tensor(np.full(8, -5.0)))
    wrong = loss.discriminator_loss(Tensor(np.full(8, -5.0)), Tensor(np.full(8, 5.0)))
    assert right.item() < wrong.item()


def test_discriminator_step_leaves_the_generator_alone():
    """The detach in step(): the discriminator's loss must not move the generator."""
    gan = GAN(4, 2, hidden=(16, 16), rng=0)
    fake = gan.generator(gan.generator.sample_latent(8))
    gan.loss.discriminator_loss(
        gan.discriminator(_gaussian(8)), gan.discriminator(fake.detach())
    ).backward()
    assert all(p.grad is None or not np.any(p.grad)
               for p in gan.generator.parameters())


def test_generator_step_needs_gradient_through_the_discriminator():
    gan = GAN(4, 2, hidden=(16, 16), rng=0)
    fake = gan.generator(gan.generator.sample_latent(8))
    gan.loss.generator_loss(gan.discriminator(fake)).backward()
    assert any(p.grad is not None and np.any(p.grad)
               for p in gan.generator.parameters())


def test_gan_learns_a_2d_gaussian():
    real = _gaussian()
    gan = _fit_gan(GAN(4, 2, hidden=(32, 32), rng=0,
                       loss=NonSaturatingGANLoss()), real)
    assert np.allclose(gan.sample(2000, rng=0).mean(0), real.mean(0), atol=0.9)


def test_wgan_learns_a_2d_gaussian():
    real = _gaussian()
    gan = _fit_gan(GAN(4, 2, hidden=(32, 32), rng=0, loss=WGANLoss()),
                   real, n_critic=2, grad_clip=0.05)
    s = gan.sample(2000, rng=0)
    assert np.allclose(s.mean(0), real.mean(0), atol=0.6)
    assert np.allclose(s.std(0), real.std(0), atol=0.4)


def test_critic_loss_tracks_the_distance_it_estimates():
    """WGAN's selling point: unlike a GAN's, this loss actually means something."""
    real = _gaussian()
    gan = GAN(4, 2, hidden=(32, 32), rng=0, loss=WGANLoss())
    og = Adam(gan.generator.parameters(), lr=1e-3)
    od = Adam(gan.discriminator.parameters(), lr=1e-3)
    rng = np.random.RandomState(0)
    losses = []
    for _ in range(40):
        idx = rng.permutation(len(real))
        losses.append(np.mean([
            gan.step(real[idx[i:i + 64]], og, od, n_critic=2, grad_clip=0.05)[0]
            for i in range(0, len(real), 64)]))
    assert abs(np.mean(losses[-5:])) < abs(np.mean(losses[:5]))


def test_tanh_output_cannot_reach_data_outside_its_range():
    """A silent trap worth naming: this failure imitates mode collapse exactly."""
    gan = _fit_gan(GAN(4, 2, hidden=(32, 32), rng=0, output_activation=Tanh),
                   _gaussian())
    assert np.abs(gan.sample(500, rng=0)).max() <= 1.0


def test_generator_sampling_is_reproducible_and_shaped():
    g = Generator(4, 3, hidden=(8,), rng=0)
    assert g.sample(5, rng=0).shape == (5, 3)
    assert np.allclose(g.sample(5, rng=1), g.sample(5, rng=1))


def test_discriminator_emits_one_logit_per_sample():
    d = Discriminator(3, hidden=(8,), rng=0)
    assert d(np.zeros((7, 3))).shape == (7,)
