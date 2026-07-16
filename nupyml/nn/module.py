"""Module base class and containers."""
import numpy as np

from ..autograd import Tensor


class Parameter(Tensor):
    __slots__ = ()

    def __init__(self, data):
        super().__init__(data, requires_grad=True)


class Module:
    def __init__(self):
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def parameters(self):
        yield from self._parameters.values()
        for m in self._modules.values():
            yield from m.parameters()

    def named_parameters(self, prefix=""):
        for name, p in self._parameters.items():
            yield f"{prefix}{name}", p
        for mname, m in self._modules.items():
            yield from m.named_parameters(prefix=f"{prefix}{mname}.")

    def modules(self):
        yield self
        for m in self._modules.values():
            yield from m.modules()

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def train(self, mode=True):
        for m in self.modules():
            object.__setattr__(m, "training", mode)
        return self

    def eval(self):
        return self.train(False)

    def state_dict(self):
        return {name: p.data.copy() for name, p in self.named_parameters()}

    def load_state_dict(self, state):
        own = dict(self.named_parameters())
        missing = set(own) - set(state)
        unexpected = set(state) - set(own)
        if missing or unexpected:
            raise KeyError(f"state_dict mismatch: missing={sorted(missing)}, "
                           f"unexpected={sorted(unexpected)}")
        for name, p in own.items():
            if p.data.shape != state[name].shape:
                raise ValueError(f"shape mismatch for {name}: "
                                 f"{p.data.shape} vs {state[name].shape}")
            p.data = np.asarray(state[name], dtype=np.float64).copy()

    def save(self, path):
        np.savez(path, **self.state_dict())

    def load(self, path):
        with np.load(path) as data:
            self.load_state_dict({k: data[k] for k in data.files})

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)
        for i, layer in enumerate(layers):
            self._modules[str(i)] = layer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __getitem__(self, i):
        return self.layers[i]

    def __len__(self):
        return len(self.layers)


class ModuleList(Module):
    def __init__(self, modules=()):
        super().__init__()
        self._list = []
        for m in modules:
            self.append(m)

    def append(self, module):
        self._modules[str(len(self._list))] = module
        self._list.append(module)
        return self

    def __iter__(self):
        return iter(self._list)

    def __getitem__(self, i):
        return self._list[i]

    def __len__(self):
        return len(self._list)


__all__ = ["Parameter", "Module", "Sequential", "ModuleList"]
