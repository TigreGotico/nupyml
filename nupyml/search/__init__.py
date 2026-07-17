"""Approximate search and sketches: trading exactness for scale.

THE COMMON BARGAIN
------------------
Everything here gives up an exact answer for one that is far cheaper in time or
memory, with a controllable error. That bargain is what makes web-scale work
possible: an exact answer you cannot afford to compute is worth less than an
approximate one you can.

Two families:

* **Approximate nearest neighbours** (``ann.py``) -- find points CLOSE to a query
  without scanning them all. Exact search is O(n) per query; these are sublinear,
  at the cost of occasionally missing the true nearest. LSH, IVF and product
  quantization each attack it differently.
* **Sketches** (``sketch.py``) -- answer questions about a huge stream in tiny,
  fixed memory: how many distinct items, how frequent is this one, is this one
  present, what are the quantiles. Each keeps a summary that never grows with the
  data, and pays in a bounded, quantifiable error.

The unifying idea is that RANDOMNESS buys you concentration: hash the data, and
guarantees about collisions and estimates follow from probability, not from
storing everything. That is why every structure here has a hash function at its
heart.
"""
from .ann import LSHForest, MinHashLSH, RandomProjectionLSH, IVFIndex, ProductQuantizer
from .sketch import (BloomFilter, CountMinSketch, HyperLogLog, ReservoirSampler,
                     TDigest, MinHash)

__all__ = [
    "LSHForest", "MinHashLSH", "RandomProjectionLSH", "IVFIndex",
    "ProductQuantizer",
    "BloomFilter", "CountMinSketch", "HyperLogLog", "ReservoirSampler",
    "TDigest", "MinHash",
]
