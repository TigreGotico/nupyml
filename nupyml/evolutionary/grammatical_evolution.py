"""Generate programs from a GRAMMAR via an integer genotype (O'Neill & Ryan)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_random_state


class GrammaticalEvolution(BaseEstimator):
    """Generate programs from a GRAMMAR via an integer genotype (O'Neill & Ryan).

    Genetic programming mutates trees directly and must repair invalid ones.
    Grammatical evolution instead evolves a string of INTEGERS and maps it through a
    BNF grammar: each integer, mod the number of choices for the current rule,
    picks a production. Every genotype therefore decodes to a SYNTACTICALLY VALID
    program, and the search operators stay simple integer mutation/crossover. Used
    here for symbolic regression from a small arithmetic grammar.
    """

    def __init__(self, grammar, fitness, genome_length=40, pop_size=200,
                 generations=60, max_wraps=3, random_state=None):
        self.grammar = grammar                            # dict: symbol -> list of productions
        self.fitness = fitness                            # callable(expr_string) -> score (higher better)
        self.genome_length = genome_length
        self.pop_size = pop_size
        self.generations = generations
        self.max_wraps = max_wraps
        self.random_state = random_state

    def decode(self, genome, start="expr"):
        out = []
        stack = [start]
        i = 0
        steps = 0
        limit = len(genome) * self.max_wraps
        while stack and steps < limit:
            sym = stack.pop()
            if sym not in self.grammar:
                out.append(sym); continue
            prods = self.grammar[sym]
            choice = prods[genome[i % len(genome)] % len(prods)]
            i += 1; steps += 1
            stack.extend(reversed(choice))
        return "".join(out)

    def fit(self):
        rng = check_random_state(self.random_state)
        pop = rng.randint(0, 256, (self.pop_size, self.genome_length))
        best, best_fit = None, -np.inf
        for _ in range(self.generations):
            fits = np.array([self.fitness(self.decode(g)) for g in pop])
            gi = fits.argmax()
            if fits[gi] > best_fit:
                best_fit, best = fits[gi], pop[gi].copy()
            order = fits.argsort()[::-1]
            elite = pop[order[:self.pop_size // 5]]
            new = [elite[i % len(elite)].copy() for i in range(2)]
            while len(new) < self.pop_size:
                p1, p2 = elite[rng.randint(len(elite))], elite[rng.randint(len(elite))]
                cut = rng.randint(1, self.genome_length)
                child = np.concatenate([p1[:cut], p2[cut:]])
                mask = rng.rand(self.genome_length) < 0.1
                child[mask] = rng.randint(0, 256, mask.sum())
                new.append(child)
            pop = np.array(new)
        self.best_genome_ = best
        self.best_expression_ = self.decode(best)
        self.best_fitness_ = best_fit
        return self


__all__ = ["GrammaticalEvolution"]
