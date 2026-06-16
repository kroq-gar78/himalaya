import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map_only

from himalaya.backend import ALL_BACKENDS, get_backend, set_backend
from himalaya.kernel_ridge import (
    KernelRidge,
    KernelRidgeCV,
    MultipleKernelRidgeCV,
    WeightedKernelRidge,
)
from himalaya.ridge import Ridge, RidgeCV

class SingleGPUEnforcer(TorchDispatchMode):
    """
    Raises RuntimeError if any CUDA tensor is allocated on or moved to
    a device other than `allowed_device`.
    """
    def __init__(self, allowed_device: int):
        super().__init__()
        self.allowed = allowed_device

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        # 1. Pre-op: catch explicit device= kwargs (e.g. torch.zeros(..., device='cuda:1'))
        device = kwargs.get("device")
        if device is not None:
            d = torch.device(device)
            if d.type == "cuda":
                idx = d.index if d.index is not None else torch.cuda.current_device()
                if idx != self.allowed:
                    raise RuntimeError(
                        f"[SingleGPUEnforcer] Attempted allocation on cuda:{idx} "
                        f"via {func.__name__}; only cuda:{self.allowed} is permitted."
                    )

        # 2. Pre-op: catch input tensors already on the wrong device
        def _check_input(t: torch.Tensor):
            if t.is_cuda and t.get_device() != self.allowed:
                raise RuntimeError(
                    f"[SingleGPUEnforcer] Input tensor on cuda:{t.get_device()} "
                    f"passed to {func.__name__}; only cuda:{self.allowed} is permitted."
                )

        tree_map_only(torch.Tensor, _check_input, args)

        result = func(*args, **kwargs)

        # 3. Post-op: safety net for any outputs that ended up on the wrong device
        def _check_output(t: torch.Tensor):
            if t.is_cuda and t.get_device() != self.allowed:
                raise RuntimeError(
                    f"[SingleGPUEnforcer] Output tensor on cuda:{t.get_device()} "
                    f"after {func.__name__}; only cuda:{self.allowed} is permitted."
                )

        tree_map_only(torch.Tensor, _check_output, result)
        return result

def _create_dataset(backend):
    n_samples, n_targets = 30, 3

    Xs = [
        backend.asarray(backend.randn(n_samples, n_features), backend.float64)
        for n_features in [100, 200]
    ]
    Ks = backend.stack([backend.matmul(X, X.T) for X in Xs])
    Y = backend.asarray(backend.randn(n_samples, n_targets), backend.float64)

    return Xs, Ks, Y


@pytest.mark.parametrize(
    'Estimator',
    [
        Ridge,
        RidgeCV,
        KernelRidge,
        KernelRidgeCV,
        # MultipleKernelRidgeCV,  # too long
        WeightedKernelRidge,
    ])
def test_get_kernel_preserves_input_device(Estimator):
    backend = set_backend('torch')
    import torch

    Xs, Ks, Y = _create_dataset(backend)

    #if torch.cuda.device_count() < 2:
    #    pytest.skip("Multiple CUDA devices required.")

    backend = set_backend('torch_cuda')
    device = torch.device('cuda:1')
    #device = torch.device('cuda:0')

    with SingleGPUEnforcer(device.index):
        #X = backend.asarray(backend.randn(10, 5), device=device)
        #Y = backend.asarray(backend.randn(10, 2), device=device)
        Xs = [backend.asarray(X, device=device) for X in Xs]
        Y = backend.asarray(Y, device=device)

        for solver in Estimator.ALL_SOLVERS.keys():
            model = Estimator(solver=solver)
            model.random_state = 0
            model.fit(Xs[0], Y)

            if hasattr(model, '_get_kernel'):
                K = model._get_kernel(Xs[0])
                assert K.device == device

            #for attribute in ["coef_", "dual_coef_", "deltas_"]:
            #    if hasattr(model, attribute):
            #        assert getattr(model, attribute).device == device

            #assert model.predict(Xs[0]).device == device
            #assert model.score(Xs[0], Y).device == device
