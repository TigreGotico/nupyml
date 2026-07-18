"""I2: island-model GA, novelty search, and MAP-Elites.

The island model must find the global optimum of a multimodal function; novelty
search must fill the behaviour space with diverse individuals (ignoring fitness);
MAP-Elites must cover the behaviour grid with high-fitness elites.
"""
import numpy as np
import pytest

from nupyml.evolutionary import IslandModelGA, NoveltySearch, MAPElites


def test_island_model_finds_multimodal_optimum():
    # Rastrigin (many local optima); the global max of -Rastrigin is 0 at the origin
    f = lambda x: -(np.sum(x ** 2) - 10 * np.sum(np.cos(2 * np.pi * x)) + 20)
    im = IslandModelGA(f, ([-5, -5], [5, 5]), n_islands=4, pop_size=20,
                       n_gen=60, random_state=0).run()
    assert im.best_fitness_ > -1.0                    # near the global optimum
    assert np.linalg.norm(im.best_) < 0.5


def test_novelty_search_explores_diversely():
    # behaviour = position; novelty search should spread across the whole box
    ns = NoveltySearch(lambda x: x, ([-5, -5], [5, 5]), pop_size=30, n_gen=40,
                       random_state=0).run()
    assert len(ns.archive_) > 20
    # the archive covers a wide range in every dimension
    assert ns.archive_.std(axis=0).mean() > 1.5


def test_map_elites_covers_grid_with_good_elites():
    # fitness peaks at the origin; descriptor = position -> illuminate the space
    me = MAPElites(lambda x: -np.sum(x ** 2), lambda x: x,
                   ([-5, -5], [5, 5]), ([-5, -5], [5, 5]),
                   grid_shape=(8, 8), n_iter=3000, random_state=0).run()
    assert me.coverage() > 0.8                        # most cells filled
    # the global best elite is near the fitness peak
    best_genome, best_fit = me.best()
    assert best_fit > -0.1
    # every cell holds a genuine elite (a stored genome + fitness)
    assert all(len(v) == 2 for v in me.archive_.values())
