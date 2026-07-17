"""Active-learning query strategies.

Each takes a fitted model (or committee) and an unlabelled pool ``X``, and
returns the INDICES of the ``n_instances`` most informative points to label next.
"""
import numpy as np

from ..utils import check_array, check_random_state


def _proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    # fall back to a hard one-hot if the model gives only labels
    pred = model.predict(X)
    classes = np.unique(pred)
    P = np.zeros((len(X), len(classes)))
    for i, c in enumerate(classes):
        P[pred == c, i] = 1.0
    return P


def uncertainty_sampling(model, X, n_instances=1):
    """Query where the model's TOP prediction is least confident.

    Rank points by ``1 - max_class_probability``: a point the model calls "class 3
    with probability 0.55" is near the decision boundary and its true label
    carries more information than one it calls "class 3 with 0.99". The simplest
    and most-used strategy. Its weakness: it tends to pick MANY points from the
    same confusing region, so it can under-sample the rest of the space -- which
    ``core_set`` addresses.
    """
    X = check_array(X)
    proba = _proba(model, X)
    uncertainty = 1 - proba.max(axis=1)
    return np.argsort(uncertainty)[::-1][:n_instances]


def margin_sampling(model, X, n_instances=1):
    """Query where the top TWO classes are closest -- the smallest margin.

    Uncertainty sampling looks only at the top class; margin sampling looks at the
    gap between the top TWO. A point split 0.5/0.45 between two classes is genuinely
    contested; one split 0.5/0.1/0.1/... over many classes is less so, even though
    both have the same max probability. So margin is the better uncertainty
    measure when there are more than two classes.
    """
    X = check_array(X)
    proba = _proba(model, X)
    part = np.sort(proba, axis=1)
    margin = part[:, -1] - part[:, -2]         # gap between the two best classes
    return np.argsort(margin)[:n_instances]    # smallest margin first


def entropy_sampling(model, X, n_instances=1):
    """Query where the full predictive distribution is most uncertain (highest
    entropy).

    Entropy uses the WHOLE probability vector, not just the top one or two
    classes, so it best captures uncertainty spread across many classes -- the
    natural generalisation of the other two to the multiclass case. A uniform
    distribution over classes is maximally uncertain and scores highest.
    """
    X = check_array(X)
    proba = np.clip(_proba(model, X), 1e-12, 1)
    entropy = -np.sum(proba * np.log(proba), axis=1)
    return np.argsort(entropy)[::-1][:n_instances]


def query_by_committee(models, X, n_instances=1):
    """Query where a COMMITTEE of models disagrees most (vote entropy).

    THE IDEA
    --------
    Train several models (on bootstraps, or with different inits). For each pool
    point, tally their votes over classes and measure the DISAGREEMENT via the
    entropy of that vote distribution. Points where the committee is unanimous are
    easy; points that split the committee are exactly the ambiguous ones whose
    labels resolve the models' disagreement.

    Disagreement can flag ambiguity that any single model's confidence hides -- a
    lone model might be confidently wrong, but a committee rarely agrees while
    being wrong. The cost is training and querying several models.
    """
    X = check_array(X)
    votes = np.array([m.predict(X) for m in models])       # (n_models, n_points)
    classes = np.unique(votes)
    disagreement = np.zeros(len(X))
    for i in range(len(X)):
        counts = np.array([(votes[:, i] == c).sum() for c in classes], float)
        p = counts / counts.sum()
        p = p[p > 0]
        disagreement[i] = -np.sum(p * np.log(p))           # vote entropy
    return np.argsort(disagreement)[::-1][:n_instances]


def expected_model_change(model, X, n_instances=1):
    """Query the point whose label would most CHANGE the model.

    THE IDEA
    --------
    The most useful label is the one that moves the model the most. As a tractable
    proxy, score each point by the expected magnitude of the gradient its label
    would induce -- large where the model is both uncertain AND the point sits in
    a direction it has not pinned down. Here that is approximated by
    ``uncertainty * ||x||``: uncertain points in high-leverage directions would
    perturb the parameters most. It targets learning IMPACT directly, rather than
    uncertainty as a stand-in for it.

    Settles, Craven & Ray (2008).
    """
    X = check_array(X)
    proba = _proba(model, X)
    uncertainty = 1 - proba.max(axis=1)
    # gradient magnitude of a logistic-style update scales with the input norm
    leverage = np.linalg.norm(X, axis=1)
    return np.argsort(uncertainty * leverage)[::-1][:n_instances]


def core_set(X_pool, X_labelled=None, n_instances=1):
    """Greedy k-center: pick points that best COVER the space, model-free.

    THE DIFFERENT PHILOSOPHY
    ------------------------
    The uncertainty methods chase the decision boundary and so tend to pile their
    queries into one ambiguous region, leaving the rest of the space unlabelled.
    Core-set ignores the model and instead picks a DIVERSE, representative subset:
    greedily choose the pool point FARTHEST from everything already labelled, add
    it, repeat. This guarantees coverage -- every unlabelled point is close to some
    labelled one -- which is a coverage bound the boundary-chasers cannot offer.

    It is the diversity end of the uncertainty-vs-diversity spectrum; in practice
    the two are combined. Selecting the farthest point each round is the same
    farthest-first heuristic that seeds k-means++ and k-center clustering.

    Sener & Savarese (2018).
    """
    X_pool = check_array(X_pool)
    from scipy.spatial.distance import cdist
    selected = []
    if X_labelled is not None and len(X_labelled):
        min_dist = cdist(X_pool, check_array(X_labelled)).min(axis=1)
    else:
        # nothing labelled yet: start from the point farthest from the centroid
        centroid = X_pool.mean(axis=0, keepdims=True)
        min_dist = cdist(X_pool, centroid).ravel()
    for _ in range(n_instances):
        # the point currently least covered by the labelled/selected set
        idx = int(np.argmax(min_dist))
        selected.append(idx)
        # every remaining point's coverage improves toward the newly selected one
        d = cdist(X_pool, X_pool[idx:idx + 1]).ravel()
        min_dist = np.minimum(min_dist, d)
        min_dist[idx] = -np.inf                # never pick the same point twice
    return np.array(selected)


__all__ = ["uncertainty_sampling", "margin_sampling", "entropy_sampling",
           "query_by_committee", "expected_model_change", "core_set"]
