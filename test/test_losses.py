"""Losses: every one gradient-checked, and each pinned to the property it exists for."""
import itertools

import numpy as np
import pytest

from nupyml import nn
from nupyml.autograd import Tensor, gradcheck
from nupyml.autograd.functional import log_softmax
from nupyml.nn.ctc import _ctc_single

rng = np.random.RandomState(0)
LOGITS = Tensor(rng.normal(size=(6, 4)), requires_grad=True)
TARGET = np.array([0, 1, 2, 3, 1, 0])
PROBS = Tensor(rng.uniform(0.05, 0.95, size=(4, 8)), requires_grad=True)
SEG = (rng.uniform(size=(4, 8)) > 0.5).astype(float)
PRED = Tensor(rng.normal(size=10), requires_grad=True)
Y = rng.normal(size=10)


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,fn,inputs", [
    ("focal", lambda a: nn.FocalLoss()(a, TARGET), [LOGITS]),
    ("focal_gamma0", lambda a: nn.FocalLoss(gamma=0.0)(a, TARGET), [LOGITS]),
    ("label_smoothing", lambda a: nn.LabelSmoothingCrossEntropy()(a, TARGET), [LOGITS]),
    ("dice", lambda a: nn.DiceLoss()(a, SEG), [PROBS]),
    ("iou", lambda a: nn.IoULoss()(a, SEG), [PROBS]),
    ("tversky", lambda a: nn.TverskyLoss()(a, SEG), [PROBS]),
    ("logcosh", lambda a: nn.LogCoshLoss()(a, Y), [PRED]),
    ("smooth_l1", lambda a: nn.SmoothL1Loss()(a, Y), [PRED]),
    ("quantile", lambda a: nn.QuantileLoss(0.8)(a, Y), [PRED]),
    ("hinge", lambda a: nn.HingeEmbeddingLoss()(a, np.sign(Y)), [PRED]),
    ("squared_hinge", lambda a: nn.HingeEmbeddingLoss(squared=True)(a, np.sign(Y)), [PRED]),
])
def test_loss_gradient_matches_finite_differences(name, fn, inputs):
    assert gradcheck(fn, inputs, rtol=1e-4, atol=1e-6)


def test_tweedie_gradient():
    log_pred = Tensor(rng.normal(scale=0.3, size=10), requires_grad=True)
    y = np.abs(rng.normal(size=10)) + 0.1
    assert gradcheck(lambda a: nn.TweedieLoss(power=1.5)(a, y), [log_pred],
                     rtol=1e-4)


def test_kl_and_distillation_gradients():
    student = Tensor(rng.normal(size=(5, 3)), requires_grad=True)
    teacher = rng.normal(size=(5, 3))
    assert gradcheck(lambda a: nn.DistillationLoss()(a, teacher), [student],
                     rtol=1e-3, atol=1e-5)
    q = rng.dirichlet(np.ones(3), size=5)
    assert gradcheck(lambda a: nn.KLDivLoss()(log_softmax(a, axis=-1), q),
                     [student], rtol=1e-4)


# ---------------------------------------------------------------------------
# focal loss: down-weight the easy, keep the hard
# ---------------------------------------------------------------------------

def test_focal_reduces_to_cross_entropy_at_gamma_zero():
    ce = nn.CrossEntropyLoss()(LOGITS, TARGET).item()
    focal = nn.FocalLoss(gamma=0.0)(LOGITS, TARGET).item()
    assert focal == pytest.approx(ce, rel=1e-9)


def test_focal_downweights_easy_examples():
    """The whole point: a confident-correct sample must nearly vanish."""
    easy = Tensor(np.array([[10.0, 0.0]]))      # p_true ~ 0.9999
    hard = Tensor(np.array([[0.1, 0.0]]))       # p_true ~ 0.52
    t = np.array([0])
    ce_ratio = (nn.CrossEntropyLoss()(hard, t).item()
                / nn.CrossEntropyLoss()(easy, t).item())
    focal_ratio = (nn.FocalLoss(gamma=2.0)(hard, t).item()
                   / nn.FocalLoss(gamma=2.0)(easy, t).item())
    # focal widens the gap between hard and easy by orders of magnitude
    assert focal_ratio > ce_ratio * 100


def test_focal_gamma_controls_the_downweighting():
    easy = Tensor(np.array([[5.0, 0.0]]))
    t = np.array([0])
    losses = [nn.FocalLoss(gamma=g)(easy, t).item() for g in (0.0, 1.0, 2.0, 5.0)]
    assert losses == sorted(losses, reverse=True)   # larger gamma, smaller loss


def test_focal_alpha_weights_classes():
    weighted = nn.FocalLoss(gamma=2.0, alpha=[10.0, 1.0, 1.0, 1.0])(LOGITS, TARGET).item()
    plain = nn.FocalLoss(gamma=2.0, alpha=[1.0, 1.0, 1.0, 1.0])(LOGITS, TARGET).item()
    assert weighted > plain


def test_focal_alpha_must_cover_every_class():
    """A short alpha would silently index out of range."""
    with pytest.raises(ValueError, match="one weight per class"):
        nn.FocalLoss(alpha=[1.0, 1.0])(LOGITS, TARGET)


# ---------------------------------------------------------------------------
# label smoothing
# ---------------------------------------------------------------------------

def test_label_smoothing_reduces_to_cross_entropy_at_zero():
    ce = nn.CrossEntropyLoss()(LOGITS, TARGET).item()
    ls = nn.LabelSmoothingCrossEntropy(eps=0.0)(LOGITS, TARGET).item()
    assert ls == pytest.approx(ce, rel=1e-9)


def test_label_smoothing_penalises_extreme_confidence():
    """A one-hot target rewards infinite confidence; smoothing does not."""
    confident = Tensor(np.array([[20.0, 0.0, 0.0]]))
    moderate = Tensor(np.array([[3.0, 0.0, 0.0]]))
    t = np.array([0])
    ce_conf = nn.CrossEntropyLoss()(confident, t).item()
    ce_mod = nn.CrossEntropyLoss()(moderate, t).item()
    ls_conf = nn.LabelSmoothingCrossEntropy(eps=0.2)(confident, t).item()
    ls_mod = nn.LabelSmoothingCrossEntropy(eps=0.2)(moderate, t).item()
    assert ce_conf < ce_mod          # plain CE: more confidence is always better
    assert ls_conf > ls_mod          # smoothed: overconfidence now costs


# ---------------------------------------------------------------------------
# segmentation
# ---------------------------------------------------------------------------

def test_dice_is_zero_for_a_perfect_match():
    t = np.array([[1.0, 0.0, 1.0, 1.0]])
    assert nn.DiceLoss(smooth=0.0)(Tensor(t), t).item() == pytest.approx(0.0, abs=1e-9)


def test_dice_ignores_class_imbalance():
    """A tiny target is what Dice exists for: accuracy would call this a triumph."""
    target = np.zeros((1, 1000))
    target[0, :5] = 1.0                       # 0.5% positive
    all_background = Tensor(np.full((1, 1000), 1e-6))
    accuracy = 1.0 - target.mean()            # 99.5%
    assert accuracy > 0.99
    assert nn.DiceLoss()(all_background, target).item() > 0.8   # Dice is not fooled


def test_iou_is_stricter_than_dice():
    probs = Tensor(rng.uniform(size=(2, 20)))
    t = (rng.uniform(size=(2, 20)) > 0.5).astype(float)
    assert nn.IoULoss()(probs, t).item() >= nn.DiceLoss()(probs, t).item()


def test_tversky_reduces_to_dice_at_half():
    probs = Tensor(rng.uniform(size=(2, 20)))
    t = (rng.uniform(size=(2, 20)) > 0.5).astype(float)
    assert nn.TverskyLoss(alpha=0.5, beta=0.5)(probs, t).item() == pytest.approx(
        nn.DiceLoss()(probs, t).item(), rel=1e-9)


def test_tversky_beta_trades_precision_for_recall():
    """beta prices false negatives: raise it and misses hurt more."""
    target = np.array([[1.0, 1.0, 0.0, 0.0]])
    misses = Tensor(np.array([[0.1, 0.1, 0.0, 0.0]]))     # under-predicting
    false_alarms = Tensor(np.array([[1.0, 1.0, 0.9, 0.9]]))  # over-predicting
    recall_focused = nn.TverskyLoss(alpha=0.1, beta=0.9)
    precision_focused = nn.TverskyLoss(alpha=0.9, beta=0.1)
    assert recall_focused(misses, target).item() > precision_focused(misses, target).item()
    assert precision_focused(false_alarms, target).item() > recall_focused(false_alarms, target).item()


# ---------------------------------------------------------------------------
# regression
# ---------------------------------------------------------------------------

def test_logcosh_is_between_mse_and_mae_in_outlier_sensitivity():
    clean = np.zeros(10)
    outlier = np.zeros(10)
    outlier[0] = 100.0
    pred = Tensor(np.zeros(10), requires_grad=True)
    ratio = lambda L: L()(pred, outlier).item() / max(L()(pred, clean).item(), 1e-12)
    # MSE explodes on the outlier; log-cosh grows only linearly
    assert nn.LogCoshLoss()(pred, outlier).item() < nn.MSELoss()(pred, outlier).item()
    assert nn.LogCoshLoss()(pred, outlier).item() > nn.MAELoss()(pred, outlier).item() * 0.5


def test_logcosh_does_not_overflow_on_large_errors():
    """cosh(x) overflows past ~710; the stable form must not."""
    pred = Tensor(np.array([0.0]))
    assert np.isfinite(nn.LogCoshLoss()(pred, np.array([1e4])).item())


def test_quantile_loss_is_asymmetric():
    pred = Tensor(np.array([0.0]))
    over = nn.QuantileLoss(0.9)(pred, np.array([-1.0])).item()   # predicted too high
    under = nn.QuantileLoss(0.9)(pred, np.array([1.0])).item()   # predicted too low
    # at q=0.9, under-predicting should hurt 9x more than over-predicting
    assert under == pytest.approx(9 * over, rel=1e-6)


def test_quantile_at_half_is_mae():
    pred = Tensor(rng.normal(size=10), requires_grad=True)
    assert nn.QuantileLoss(0.5)(pred, Y).item() == pytest.approx(
        0.5 * nn.MAELoss()(pred, Y).item(), rel=1e-9)


def test_smooth_l1_matches_huber_shape():
    pred = Tensor(np.array([0.0]))
    assert nn.SmoothL1Loss(beta=1.0)(pred, np.array([0.5])).item() == pytest.approx(0.125)
    assert nn.SmoothL1Loss(beta=1.0)(pred, np.array([10.0])).item() == pytest.approx(9.5)


# ---------------------------------------------------------------------------
# distillation
# ---------------------------------------------------------------------------

def test_distillation_is_zero_when_student_matches_teacher():
    logits = rng.normal(size=(4, 5))
    loss = nn.DistillationLoss(temperature=3.0)(Tensor(logits), logits)
    assert loss.item() == pytest.approx(0.0, abs=1e-9)


def test_distillation_temperature_softens_the_target():
    """Higher T magnifies the small probabilities that carry the dark knowledge."""
    teacher = np.array([[10.0, 2.0, 1.0]])
    from nupyml.utils import softmax
    sharp = softmax(teacher / 1.0, axis=-1)
    soft = softmax(teacher / 5.0, axis=-1)
    assert soft.min() > sharp.min()          # the runner-up gets real mass
    assert soft.max() < sharp.max()


# ---------------------------------------------------------------------------
# adversarial
# ---------------------------------------------------------------------------

def test_wasserstein_rewards_separating_real_from_fake():
    real = Tensor(np.array([5.0, 5.0]))
    fake = Tensor(np.array([-5.0, -5.0]))
    separated = nn.WassersteinLoss()(real, fake).item()
    confused = nn.WassersteinLoss()(Tensor(np.array([0.0, 0.0])),
                                    Tensor(np.array([0.0, 0.0]))).item()
    assert separated < confused          # critic loss is lower when it can tell


def test_gradient_penalty_is_zero_for_a_unit_gradient_critic():
    """A critic with slope exactly 1 is already 1-Lipschitz: no penalty due."""
    critic = lambda x: x.sum(axis=1)
    real = np.ones((8, 1))
    fake = np.zeros((8, 1))
    assert nn.gradient_penalty(critic, real, fake, rng=0) == pytest.approx(0.0, abs=1e-6)


def test_gradient_penalty_punishes_a_steep_critic():
    steep = lambda x: (x * 10.0).sum(axis=1)
    real = np.ones((8, 1))
    fake = np.zeros((8, 1))
    assert nn.gradient_penalty(steep, real, fake, rng=0) == pytest.approx(81.0, rel=1e-4)


# ---------------------------------------------------------------------------
# CTC
# ---------------------------------------------------------------------------

def _brute_force_ctc(log_probs, labels, blank=0):
    """Enumerate every path and sum the ones that collapse to the target."""
    T, K = log_probs.shape

    def collapse(path):
        out, prev = [], -1
        for k in path:
            if k != prev and k != blank:
                out.append(k)
            prev = k
        return out

    total = -np.inf
    for path in itertools.product(range(K), repeat=T):
        if collapse(path) == list(labels):
            total = np.logaddexp(total, sum(log_probs[t, k]
                                            for t, k in enumerate(path)))
    return -total


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ctc_dp_equals_brute_force_enumeration(seed):
    """The dynamic program must sum EXACTLY the paths a brute force finds."""
    r = np.random.RandomState(seed)
    log_probs = np.log(r.dirichlet(np.ones(3), size=5))
    labels = np.array([1, 2])
    dp, _ = _ctc_single(log_probs, labels, blank=0)
    assert dp == pytest.approx(_brute_force_ctc(log_probs, labels), rel=1e-9)


def test_ctc_handles_repeated_labels():
    """'hello' needs a blank between the l's, or the merge rule eats one."""
    r = np.random.RandomState(3)
    log_probs = np.log(r.dirichlet(np.ones(3), size=6))
    labels = np.array([1, 1])            # a repeat: the hard case
    dp, _ = _ctc_single(log_probs, labels, blank=0)
    assert dp == pytest.approx(_brute_force_ctc(log_probs, labels), rel=1e-9)


def test_ctc_gradient_matches_finite_differences():
    T, N, K = 6, 1, 4
    logits = rng.normal(size=(T, N, K))
    targets = [np.array([1, 2, 3])]

    def loss_of(flat):
        return nn.CTCLoss()(log_softmax(Tensor(flat.reshape(T, N, K)), axis=-1),
                            targets).item()

    lg = Tensor(logits, requires_grad=True)
    nn.CTCLoss()(log_softmax(lg, axis=-1), targets).backward()
    numeric = np.zeros(logits.size)
    flat = logits.ravel().copy()
    for i in range(len(flat)):
        o = flat[i]
        flat[i] = o + 1e-6
        a = loss_of(flat.copy())
        flat[i] = o - 1e-6
        b = loss_of(flat.copy())
        flat[i] = o
        numeric[i] = (a - b) / 2e-6
    assert np.allclose(lg.grad.ravel(), numeric, atol=1e-5)


def test_ctc_learns_an_alignment():
    T, N, K = 12, 1, 4
    lg = Tensor(rng.normal(scale=0.1, size=(T, N, K)), requires_grad=True)
    opt = nn.Adam([lg], lr=0.1)
    targets = [np.array([1, 2, 3, 2])]     # includes a repeat of label 2
    for _ in range(300):
        opt.zero_grad()
        loss = nn.CTCLoss()(log_softmax(lg, axis=-1), targets)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
    assert nn.ctc_greedy_decode(log_softmax(lg, axis=-1))[0] == [1, 2, 3, 2]


def test_ctc_impossible_target_does_not_crash():
    """A target longer than the input can express has no valid alignment."""
    log_probs = np.log(np.full((2, 3), 1 / 3))
    loss, grad = _ctc_single(log_probs, np.array([1, 2, 1, 2]), blank=0)
    assert np.isfinite(loss)
    assert np.isfinite(grad).all()


def test_ctc_greedy_decode_collapses_correctly():
    # frames: blank, 1, 1, blank, 2, 2  ->  merge runs, drop blanks  ->  [1, 2]
    probs = np.zeros((6, 1, 3))
    for t, k in enumerate([0, 1, 1, 0, 2, 2]):
        probs[t, 0, k] = 1.0
    assert nn.ctc_greedy_decode(np.log(probs + 1e-12))[0] == [1, 2]


# ---------------------------------------------------------------------------
# metric learning
# ---------------------------------------------------------------------------

E1 = Tensor(rng.normal(size=(6, 4)), requires_grad=True)
E2 = Tensor(rng.normal(size=(6, 4)))
E3 = Tensor(rng.normal(size=(6, 4)))
PAIR_Y = np.array([1.0, 0, 1, 0, 1, 0])
LABELS = np.array([0, 0, 1, 1, 2, 2])


@pytest.mark.parametrize("name,fn", [
    ("contrastive", lambda a: nn.ContrastiveLoss()(a, E2, PAIR_Y)),
    ("triplet", lambda a: nn.TripletLoss()(a, E2, E3)),
    ("batch_hard", lambda a: nn.BatchHardTripletLoss()(a, LABELS)),
    ("npairs", lambda a: nn.NPairsLoss()(a, E2)),
    ("infonce", lambda a: nn.InfoNCELoss()(a, E2)),
])
def test_metric_loss_gradients(name, fn):
    assert gradcheck(fn, [E1], rtol=1e-3, atol=1e-5)


def test_margin_loss_gradients():
    for loss in (nn.ArcFaceLoss(4, 3, rng=0), nn.CosFaceLoss(4, 3, rng=0),
                 nn.CenterLoss(3, 4, rng=0)):
        assert gradcheck(lambda a: loss(a, LABELS), [E1], rtol=1e-3, atol=1e-5)


def test_contrastive_pulls_matches_and_pushes_non_matches():
    same = Tensor(np.array([[0.0, 0.0]]))
    near = Tensor(np.array([[0.1, 0.0]]))
    far = Tensor(np.array([[5.0, 0.0]]))
    # a matching pair should cost less when close
    assert (nn.ContrastiveLoss()(same, near, np.array([1.0])).item()
            < nn.ContrastiveLoss()(same, far, np.array([1.0])).item())
    # a non-matching pair should cost less when far
    assert (nn.ContrastiveLoss()(same, far, np.array([0.0])).item()
            < nn.ContrastiveLoss()(same, near, np.array([0.0])).item())


def test_contrastive_margin_caps_the_push():
    """Beyond the margin a non-match costs nothing: being distinguishable is enough."""
    a = Tensor(np.array([[0.0, 0.0]]))
    beyond = Tensor(np.array([[10.0, 0.0]]))
    assert nn.ContrastiveLoss(margin=1.0)(a, beyond,
                                          np.array([0.0])).item() == pytest.approx(0.0)


def test_triplet_is_zero_once_the_margin_is_satisfied():
    anchor = Tensor(np.array([[0.0, 0.0]]))
    positive = Tensor(np.array([[0.1, 0.0]]))
    negative = Tensor(np.array([[9.0, 0.0]]))
    assert nn.TripletLoss(margin=1.0)(anchor, positive,
                                      negative).item() == pytest.approx(0.0)


def test_triplet_penalises_a_violated_order():
    anchor = Tensor(np.array([[0.0, 0.0]]))
    positive = Tensor(np.array([[5.0, 0.0]]))    # positive is FAR
    negative = Tensor(np.array([[0.1, 0.0]]))    # negative is NEAR: violated
    assert nn.TripletLoss(margin=1.0)(anchor, positive, negative).item() > 0


def test_batch_hard_picks_the_hardest_pair():
    """It must select the furthest positive and nearest negative, not any pair."""
    emb = Tensor(np.array([[0.0, 0.0], [3.0, 0.0], [0.5, 0.0], [10.0, 0.0]]))
    labels = np.array([0, 0, 1, 1])
    # for anchor 0: hardest positive is idx1 (d=3), hardest negative is idx2 (d=0.5)
    loss = nn.BatchHardTripletLoss(margin=0.0)(emb, labels)
    assert loss.item() > 0            # 3 > 0.5: the constraint is violated


def test_infonce_temperature_sharpens():
    a = Tensor(rng.normal(size=(4, 8)))
    p = Tensor(rng.normal(size=(4, 8)))
    losses = [nn.InfoNCELoss(temperature=t)(a, p).item() for t in (1.0, 0.1)]
    assert losses[0] != losses[1]     # temperature must actually do something


def test_infonce_rewards_matching_views():
    """A positive that IS the anchor should give near-zero loss."""
    a = Tensor(rng.normal(size=(8, 16)))
    assert nn.InfoNCELoss(temperature=0.05)(a, a).item() < 0.1


def test_npairs_matches_diagonal():
    a = Tensor(np.eye(3) * 10.0)
    assert nn.NPairsLoss()(a, Tensor(np.eye(3) * 10.0)).item() < 0.01


def test_arcface_margin_makes_the_task_harder():
    """The margin handicaps the true class: same features, higher loss."""
    feats = Tensor(rng.normal(size=(6, 4)))
    no_margin = nn.ArcFaceLoss(4, 3, margin=0.0, rng=0)
    with_margin = nn.ArcFaceLoss(4, 3, margin=0.5, rng=0)
    assert with_margin(feats, LABELS).item() > no_margin(feats, LABELS).item()


def test_cosface_margin_makes_the_task_harder():
    feats = Tensor(rng.normal(size=(6, 4)))
    assert (nn.CosFaceLoss(4, 3, margin=0.35, rng=0)(feats, LABELS).item()
            > nn.CosFaceLoss(4, 3, margin=0.0, rng=0)(feats, LABELS).item())


def test_arcface_never_calls_arccos_and_stays_finite():
    """cos is clipped off +/-1 so the sin sqrt stays real."""
    feats = Tensor(np.array([[1e6, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]))
    loss = nn.ArcFaceLoss(4, 3, rng=0)(feats, np.array([0, 1]))
    assert np.isfinite(loss.item())


def test_center_loss_is_zero_at_its_centers():
    cl = nn.CenterLoss(3, 2, rng=0)
    feats = Tensor(cl.centers.data[np.array([0, 1, 2])])
    assert cl(feats, np.array([0, 1, 2])).item() == pytest.approx(0.0, abs=1e-12)


def test_metric_losses_train_an_embedding():
    """End to end: InfoNCE should pull augmented views of a point together."""
    r = np.random.RandomState(5)
    base = r.normal(size=(16, 3))
    model = nn.Sequential(nn.Linear(3, 8, rng=0), nn.ReLU(), nn.Linear(8, 4, rng=1))
    opt = nn.Adam(model.parameters(), lr=0.05)
    for _ in range(150):
        v1 = Tensor(base + r.normal(scale=0.05, size=base.shape))
        v2 = Tensor(base + r.normal(scale=0.05, size=base.shape))
        opt.zero_grad()
        loss = nn.InfoNCELoss(temperature=0.2)(model(v1), model(v2))
        loss.backward()
        opt.step()
    # views of the same point should now be each other's nearest neighbour
    e1 = model(Tensor(base + r.normal(scale=0.05, size=base.shape))).data
    e2 = model(Tensor(base + r.normal(scale=0.05, size=base.shape))).data
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True)
    matched = (e1 @ e2.T).argmax(axis=1) == np.arange(16)
    assert matched.mean() > 0.7
