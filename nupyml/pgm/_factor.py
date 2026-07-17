"""The discrete factor: the one data structure the whole module runs on.

A FACTOR is a function from an assignment of some variables to a non-negative
number, stored as a multidimensional array with one axis per variable. A CPD, a
clique potential, and an intermediate result of inference are all factors -- so
three operations on this one object (multiply, sum-out, condition) are the entire
algebra of graphical-model inference.
"""
import numpy as np


class Factor:
    """A non-negative function over a set of discrete variables.

    ``variables`` is an ordered tuple of names; ``values`` is an array whose axis
    ``i`` ranges over variable ``i``'s states (``cardinalities[i]`` of them).
    """

    def __init__(self, variables, cardinalities, values):
        self.variables = tuple(variables)
        self.cardinalities = tuple(cardinalities)
        self.values = np.asarray(values, dtype=float).reshape(cardinalities)

    def copy(self):
        return Factor(self.variables, self.cardinalities, self.values.copy())

    def multiply(self, other):
        """Pointwise product of two factors, aligning shared variables.

        This is how local pieces combine into a joint: two CPTs sharing a
        variable multiply into a factor over the UNION of their variables, with
        the shared axis lined up. Broadcasting does the alignment once both
        factors are reshaped to the common variable order -- no explicit loop
        over assignments.
        """
        # the combined variable set, in a stable order (self's, then other's new)
        new_vars = list(self.variables)
        for v in other.variables:
            if v not in new_vars:
                new_vars.append(v)
        card = {v: c for v, c in zip(self.variables, self.cardinalities)}
        card.update({v: c for v, c in zip(other.variables, other.cardinalities)})
        new_card = [card[v] for v in new_vars]

        a = self._align(new_vars)
        b = other._align(new_vars)
        return Factor(new_vars, new_card, a * b)

    def _align(self, target_vars):
        """This factor's values reshaped for broadcasting against ``target_vars``.

        The present variables are transposed into ``target_vars`` relative order,
        and a size-1 axis is inserted for every target variable this factor does
        not mention -- so numpy broadcasting then lines everything up.
        """
        present = [v for v in target_vars if v in self.variables]
        vals = np.transpose(self.values,
                            [self.variables.index(v) for v in present])
        bshape = [self.cardinalities[self.variables.index(v)]
                  if v in self.variables else 1 for v in target_vars]
        return vals.reshape(bshape)

    def marginalize(self, variables):
        """Sum OUT the named variables -- the core inference operation.

        "What is the distribution of the rest, ignoring these?" is answered by
        summing the factor over the eliminated axes. Every exact inference
        algorithm here is a schedule of marginalisations arranged to keep the
        intermediate factors small.
        """
        to_remove = [v for v in variables if v in self.variables]
        axes = tuple(self.variables.index(v) for v in to_remove)
        new_vars = [v for v in self.variables if v not in to_remove]
        new_card = [self.cardinalities[self.variables.index(v)] for v in new_vars]
        return Factor(new_vars, new_card, self.values.sum(axis=axes))

    def reduce(self, evidence):
        """Condition on observed values: slice the factor to the evidence.

        Fixing a variable to an observed state selects that slice of the array,
        dropping the axis. This is how evidence enters inference -- it shrinks the
        factors before the summing begins.
        """
        idx = [slice(None)] * len(self.variables)
        new_vars, new_card = [], []
        for i, v in enumerate(self.variables):
            if v in evidence:
                idx[i] = evidence[v]
            else:
                new_vars.append(v)
                new_card.append(self.cardinalities[i])
        return Factor(new_vars, new_card, self.values[tuple(idx)])

    def normalize(self):
        """Scale to sum to 1 -- turns an unnormalised factor into a distribution."""
        total = self.values.sum()
        if total > 0:
            self.values = self.values / total
        return self
