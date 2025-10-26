from __future__ import annotations
import math
import time
import numpy as np
from collections import defaultdict
from functools import partialmethod, reduce
from itertools import accumulate
from typing import Any, Callable, ClassVar, Iterable, List, Optional, Sequence, Set, Tuple, Type, Union

from teenygrad.helpers import argfix, make_pair, getenv, DEBUG, flatten, DType, dtypes, prod, all_int, round_up
from teenygrad.lazy import LazyBuffer
from teenygrad.ops import Device, LoadOps
from teenygrad.realize import run_schedule
from teenygrad.shape.symbolic import sint

# **** start with two base classes, Tensor and Function ****

class Function:
    def __init__(self, device: str, *tensors: Tensor):
        self.device = device
        self.needs_input_grad = [t.requires_grad for t in tensors]

        has_any_grad = any(self.needs_input_grad)
        has_none_grad = None in self.needs_input_grad
        self.requires_grad = True if has_any_grad else None if has_none_grad else False

        if self.requires_grad:
            self.parents = tensors
            
    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"forward not implemented for {type(self)}")
    
    def backward(self, *args, **kwargs): 
        raise RuntimeError(f"backward not implemented for {type(self)}")

    @classmethod
    def apply(fxn: Type[Function], *x: Tensor, **kwargs) -> Tensor:
        ctx = fxn(x[0].device, *x)
        ret = Tensor(ctx.forward(*[t.lazydata for t in x], **kwargs), device=ctx.device, requires_grad=ctx.requires_grad)
        if ctx.requires_grad and not Tensor.no_grad:
            ret._ctx = ctx  # used by autograd engine
        return ret

import teenygrad.mlops as mlops


class Tensor:
    __slots__ = "lazydata", "requires_grad", "grad", "_ctx"
    __deletable__ = ('_ctx',)
    training: ClassVar[bool] = False
    
    class train:
        def __init__(self, val=True):
            self.val = val
            
        def __enter__(self):
            self.prev = Tensor.training
            Tensor.training = self.val
            
        def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
            Tensor.training = self.prev
            
    no_grad: ClassVar[bool] = False
    default_type: ClassVar[DType] = dtypes.float32
    
    def __init__(self,
                 data: Union[None, int, float, list, LazyBuffer, np.ndarray, bytes],
                 device: Optional[str] = None,
                 dtype: Optional[DType] = None,
                 requires_grad: Optional[bool] = None):
        assert dtype is None or isinstance(dtype, DType), f"invalid dtype {dtype}"
        device = Device.canonicalize(device)
        self.grad: Optional[Tensor] = None  # tensors have gradients, buffers do not

        # NOTE: this can be in three states. False and None: no gradient, True: gradient
        # None (the default) will be updated to True if it's put in an optimizer
        self.requires_grad: Optional[bool] = requires_grad

        self._ctx: Optional[Function] = None  # internal variables used for autograd graph construction

        if isinstance(data, LazyBuffer):
            assert dtype is None or dtype == data.dtype, "dtype doesn't match, and casting isn't supported"
        elif isinstance(data, (int, float)):
            data = LazyBuffer.loadop(LoadOps.CONST, tuple(), dtype or Tensor.default_type, device, data)
        elif data is None or data.__class__ is list:
            assert dtype is None or dtype.np is not None, f"{dtype} doesn't have a numpy dtype"
            data = LazyBuffer.fromCPU(np.array([] if data is None else data, dtype=(dtype or Tensor.default_type).np))
        elif isinstance(data, bytes):
            data = LazyBuffer.fromCPU(np.frombuffer(data, np.uint8))
        elif isinstance(data, np.ndarray):
            assert dtype is None or dtype.np is not None, f"{dtype} doesn't have a numpy dtype"
            if data.shape == ():
                data = LazyBuffer.loadop(LoadOps.CONST, tuple(), dtype or dtypes.from_np(data.dtype), device, data.item())
            else:
                data = LazyBuffer.fromCPU(data.astype(dtype.np) if dtype is not None and dtype.np is not None else data)

        # data is a LazyBuffer, but it might be on the wrong device
        if not isinstance(data, LazyBuffer):
            raise RuntimeError(f"can't create Tensor from {data!r} with type {type(data)}")
        self.lazydata = data if data.device == device else data.copy_to_device(device)

    def __repr__(self):
        return f"<Tensor {self.lazydata!r} on {self.device} with grad {(self.grad.lazydata if self.grad else None)!r}>"

    def __hash__(self):
        return id(self)

    @property
    def device(self) -> str:
        return self.lazydata.device

    @property
    def shape(self) -> Tuple[sint, ...]:
        return self.lazydata.shape

    @property
    def dtype(self) -> DType:
        return self.lazydata.dtype

    # ***** data handlers ****

    @staticmethod
    def corealize(lst: Iterable[Tensor]):
        seen: Set[LazyBuffer] = set()
        sched = []
        for t in lst:
            sched += t.lazydata.schedule(seen)
        run_schedule(sched)

    def realize(self) -> Tensor:
        run_schedule(self.lazydata.schedule())
        return self

    def assign(self, x) -> Tensor:
        if self.device.startswith("DISK"):
            if x.__class__ is not Tensor:
                x = Tensor(x, device="CPU", dtype=self.dtype)
            self.contiguous().realize().lazydata.realized._copyin(x.numpy())
            return self

        if x.__class__ is not Tensor:
            x = Tensor(x, device=self.device, dtype=self.dtype)

        assert self.shape == x.shape and self.device == x.device, f"assign shape mismatch {self.shape} != {x.shape} or device mismatch {self.device} != {x.device}"
        assert not x.requires_grad  # self requires_grad is okay?

        if DEBUG >= 4:
            print(f"assign {self.lazydata} <- {x.lazydata}")
        if self.dtype == x.dtype and self.lazydata.realized is not None and not getenv("DISALLOW_ASSIGN"):
            x.lazydata.output_buffer = self.lazydata.realized

        self.lazydata = x.lazydata
        return self

    def detach(self) -> Tensor:
        return Tensor(self.lazydata, device=self.device, requires_grad=False)

    def numpy(self) -> np.ndarray:
        assert all_int(self.shape), f"no numpy if shape is symbolic, {self.shape=}"
        assert self.dtype.np is not None, f"no numpy dtype for {self.dtype}"

        detached = self.detach()
        dtype = dtypes.from_np(self.dtype.np)
        casted = detached.cast(dtype)
        contig = casted.contiguous()
        cpu = contig.to('CPU')
        realized = cpu.realize()
        buf = realized.lazydata.realized
        arr = buf.toCPU()
        return arr.reshape(self.shape)

    def item(self) -> Union[float, int]:
        return self.numpy().item()

    def to(self, device: Optional[str]) -> Tensor:
        if device is None or device == self.device:
            return self
        ret = Tensor(self.lazydata, device)
        if self.grad:
            ret.grad = self.grad.to(device)
        return ret

    def to_(self, device: Optional[str]):
        if device is None or device == self.device:
            return
        if self.grad:
            self.grad = self.grad.to_(device)
        _ret = Tensor(self.lazydata, device)
        self.lazydata = _ret.lazydata

    # ***** creation llop entrypoint *****

    @staticmethod
    def _loadop(op, sz, device: Optional[str] = None, dtype: Optional[DType] = None, arg=None, **kwargs):
        assert isinstance(sz, int), f"cannot create with symbolic size {sz}"
        return Tensor(LazyBuffer.loadop(op, (sz,), Tensor.default_type if dtype is None else dtype, Device.canonicalize(device), arg), dtype=dtype, device=device, **kwargs)

    @staticmethod
    def empty(*shape, **kwargs):
        shape = argfix(*shape)
        size = prod(shape)
        flat = Tensor._loadop(LoadOps.EMPTY, size, **kwargs)
        return flat.reshape(shape)

    _seed: int = int(time.time())

    @staticmethod
    def manual_seed(seed=0):
        Tensor._seed = seed

    @staticmethod
    def rand(*shape, **kwargs):
        Tensor._seed += 1
        shape = argfix(*shape)
        size = prod(shape)
        flat = Tensor._loadop(LoadOps.RAND, size, arg=Tensor._seed, **kwargs)
        return flat.reshape(shape)

    # ***** creation helper functions *****

    @staticmethod
    def full(shape: Tuple[sint, ...], fill_value, **kwargs):
        shape = argfix(shape)
        scalar = Tensor(fill_value, **kwargs)
        ones = [1] * len(shape)
        reshaped = scalar.reshape(ones)
        return reshaped.expand(shape)

    @staticmethod
    def zeros(*shape, **kwargs):
        return Tensor.full(argfix(*shape), 0, **kwargs)

    @staticmethod
    def ones(*shape, **kwargs):
        return Tensor.full(argfix(*shape), 1, **kwargs)

    @staticmethod
    def arange(start, stop=None, step=1, **kwargs):
        if stop is None:
            stop, start = start, 0

        n = math.ceil((stop - start) / step)
        steps = Tensor.full((n,), step, **kwargs)
        cumsum = steps.cumsum()
        offset = start - step
        return cumsum + offset

    @staticmethod
    def eye(dim: int, **kwargs):
        col = Tensor.full((dim, 1), 1, **kwargs)
        padded = col.pad(((0, 0), (0, dim)))
        pattern_sz = dim * (dim + 1)
        flat = padded.reshape(pattern_sz)
        trimmed = flat.shrink(((0, dim * dim),))
        return trimmed.reshape(dim, dim)

    def full_like(self, fill_value, **kwargs):
        return Tensor.full(self.shape, fill_value=fill_value, dtype=kwargs.pop("dtype", self.dtype), device=kwargs.pop("device", self.device), **kwargs)

    def zeros_like(self, **kwargs):
        return self.full_like(0, **kwargs)

    def ones_like(self, **kwargs):
        return self.full_like(1, **kwargs)

    # ***** rng hlops *****

    @staticmethod
    def randn(*shape, dtype: Optional[DType] = None, **kwargs) -> Tensor:
        # https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
        src = Tensor.rand(2, *shape, **kwargs)
        u1, u2 = src[0], src[1]

        angle = u1.mul(2 * math.pi)
        cos_val = angle.cos()

        log_term = (1 - u2).log()
        radius = log_term.mul(-2).sqrt()

        result = cos_val.mul(radius)
        dtype = Tensor.default_type if dtype is None else dtype
        return result.cast(dtype)

    @staticmethod
    def randint(*shape, low=0, high=10, **kwargs) -> Tensor:
        return (Tensor.rand(*shape, **kwargs) * (high - low) + low).cast(dtypes.int32)

    @staticmethod
    def normal(*shape, mean=0.0, std=1.0, **kwargs) -> Tensor:
        return (std * Tensor.randn(*shape, **kwargs)) + mean

    @staticmethod
    def uniform(*shape, low=0.0, high=1.0, **kwargs) -> Tensor:
        dtype = kwargs.pop("dtype", Tensor.default_type)
        return ((high - low) * Tensor.rand(*shape, **kwargs)).cast(dtype) + low

    @staticmethod
    def scaled_uniform(*shape, **kwargs) -> Tensor:
        return Tensor.uniform(*shape, low=-1.0, high=1.0, **kwargs).mul(prod(shape) ** -0.5)

    # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform
    @staticmethod
    def glorot_uniform(*shape, **kwargs) -> Tensor:
        return Tensor.uniform(*shape, low=-1.0, high=1.0, **kwargs).mul((6 / (shape[0] + prod(shape[1:]))) ** 0.5)

    # https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_
    @staticmethod
    def kaiming_uniform(*shape, a: float = 0.01, **kwargs) -> Tensor:
        bound = math.sqrt(3.0) * math.sqrt(2.0 / (1 + a ** 2)) / math.sqrt(prod(shape[1:]))
        return Tensor.uniform(*shape, low=-bound, high=bound, **kwargs)

    # https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_normal_
    @staticmethod
    def kaiming_normal(*shape, a: float = 0.01, **kwargs) -> Tensor:
        std = math.sqrt(2.0 / (1 + a ** 2)) / math.sqrt(prod(shape[1:]))
        return Tensor.normal(*shape, mean=0.0, std=std, **kwargs)

    def multinomial(self: Tensor, num_samples: int = 1, replacement: bool = False) -> Tensor:
        assert 1 <= self.ndim <= 2 and num_samples > 0, f"{self.ndim=} must be 1 or 2 dim, {num_samples=} must be positive"
        assert replacement or num_samples == 1, "no replacement only supports num_samples = 1"

        weight = self.unsqueeze(0) if self.ndim == 1 else self
        cw = weight.cumsum(1)
        total = cw[:, -1].unsqueeze(1)
        cdf = cw / total

        samples = Tensor.rand(num_samples, cdf.shape[0], 1)
        expanded = samples.expand((-1, -1, cdf.shape[1]))
        compare = expanded >= cdf
        indices = compare.sum(2).permute((1, 0))

        result = indices.squeeze(0) if self.ndim == 1 else indices
        return result.cast(dtypes.int32)

    # ***** toposort and backward pass *****

    def deepwalk(self):
        def _deepwalk(node, visited, nodes):
            visited.add(node)
            if getattr(node, "_ctx", None):
                for i in node._ctx.parents:
                    if i not in visited:
                        _deepwalk(i, visited, nodes)
                nodes.append(node)
            return nodes
        return _deepwalk(self, set(), [])

    def backward(self) -> Tensor:
        assert self.shape == tuple(), f"backward can only be called for scalar tensors, but it has shape {self.shape})"

        # fill in the first grad with one. don't use Tensor.ones because we don't need contiguous
        # this is "implicit gradient creation"
        self.grad = Tensor(1, device=self.device, requires_grad=False)

        for t0 in reversed(self.deepwalk()):
            assert (t0.grad is not None)
            grads = t0._ctx.backward(t0.grad.lazydata)
            grads = [Tensor(g, device=self.device, requires_grad=False) if g is not None else None
                     for g in ([grads] if len(t0._ctx.parents) == 1 else grads)]
            for t, g in zip(t0._ctx.parents, grads):
                if g is not None and t.requires_grad:
                    assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
                    t.grad = g if t.grad is None else (t.grad + g)
            del t0._ctx
        return self

    # ***** movement mlops *****

    def reshape(self, shape, *args) -> Tensor:
        new_shape = argfix(shape, *args)

        resolved = []
        for i, s in enumerate(new_shape):
            if s == -1:
                inferred = -prod(self.shape) // prod(new_shape)
                resolved.append(inferred)
            elif s is not None:
                resolved.append(s)
            else:
                resolved.append(self.shape[i])

        return mlops.Reshape.apply(self, shape=tuple(resolved))

    def expand(self, shape, *args) -> Tensor:
        target = argfix(shape, *args)
        resolved = []
        for cur, tgt in zip(self.shape, target):
            dim = tgt if tgt != -1 else cur
            resolved.append(dim)

        return mlops.Expand.apply(self, shape=tuple(resolved))

    def permute(self, order, *args) -> Tensor:
        return mlops.Permute.apply(self, order=argfix(order, *args))

    def flip(self, axis, *args) -> Tensor:
        return mlops.Flip.apply(self, axis=[x if x >= 0 else x+len(self.shape) for x in argfix(axis, *args)])

    def shrink(self, arg: Tuple[Optional[Tuple[sint, sint]], ...]) -> Tensor:
        if any(x is not None and x != (0, s) for x, s in zip(arg, self.shape)):
            return mlops.Shrink.apply(self, arg=tuple(x if x is not None else (0, s) for x, s in zip(arg, self.shape)))
        else:
            return self

    def pad(self, arg: Tuple[Optional[Tuple[int, int]], ...], value: float = 0.0) -> Tensor:
        if all(x is None or x == (0, 0) for x in arg):
            return self
        ret = mlops.Pad.apply(self, arg=(narg := tuple(x if x is not None else (0, 0) for x in arg)))
        return ret if 0 == value else ret + mlops.Pad.apply(Tensor.ones_like(self), arg=narg).where(0, value)

    # ***** movement hlops *****

    # - Negative indices are taken relative to the end of the sequence, so X[-2] returns the 2nd-to-last element
    # - A slice i:j returns the elements with indices in [i, j)
    #      - If omitted, i and j will default to 0 and N, respectively, where N is the length of the sequence
    #      - Negative values for i and j are taken relative to the end of the sequence
    #      - Both i and j will be clamped to the range (-N, N], where N in the length of the sequence
    # - Indexing with None on a given axis will add a new dimension of size one before that axis
    # - Empty slices are not allowed (tensors with 0s in shape have to be supported first, for all backends).
    # - For a slice [i:j:k] finding the correct indices is delegated to slice.indices(len).
    # - Strides > 1 and < 0 are now allowed!:
    #      - This works by applying Shrink -> [[Flip -> ] Pad -> Reshape -> Shrink] -> Reshape (ops in brackets are optional)
    #      - Idea of stride < 0 support:
    #          - Do the slice first, flip the axes were slice.step is negative, do slice.step -> -slice.step. Go to steps below.
    #      - Idea of stride `s` > 1 support (Pad -> Reshape -> Shrink):
    #          - Instead of doing [::s] on axis [dim_sz], do [:, 0] on axes [dim_sz_padded // s, s].
    #          - So pad dim_sz with as many zeros as needed (dim_sz -> dim_sz_padded) so that reshape to [dim_sz_padded // s, s]
    #            is possible.
    #          - Apply Shrink to do the slice [:, 0] on axes of shapes [dim_sz_padded // s, s].
    # - Fancy indexing and combined indexing is supported
    #      - Combined indexing works by letting regular slicing finish first -> computing the resulting dims w.r.t to Tensors passed in -> fancy indexing
    #      - Any Tensors passed in __getitem__ will perform (CMPEQ with arange -> MUL with self -> SUM_REDUCE) iteratively
    #          - The first iteration will expand the dim of self while consecutive iterations will reduce the dim
    #      - There's a special case where a permute is needed at the end:
    #          - if first Tensor passed in (expand dims) is not at dim 0
    #          - and following Tensors does not follow consecutively to the end of fancy indexing's dims

    def __getitem__(self, val) -> Tensor:  # val: Union[int, slice, Tensor, None, Ellipsis, Tuple[Union[int, slice, Tensor, None, Ellipsis], ...]]
        def normalize_int(e, i, dim_sz):
            if -dim_sz <= e < dim_sz:
                return e if e != -1 else dim_sz - 1
            raise IndexError(f"index {e} is out of bounds for dimension {i} with size {self.shape[i]}")

        orig_slices = list(val) if isinstance(val, tuple) else [val]
        count = defaultdict(list)
        for i, v in enumerate(orig_slices):
            count[type(v)].append(i)

        num_slices = len(count[int]) + len(count[slice]) + len(count[Tensor])
        if num_slices > len(self.shape):
            raise IndexError(f"too many indices for tensor of dimension {len(self.shape)}")

        ellipsis_found = count[type(Ellipsis)]
        if len(ellipsis_found) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")

        ellipsis_idx = ellipsis_found[0] if ellipsis_found else len(orig_slices)
        ell_fill = [slice(None)] * (len(self.shape) - num_slices)
        orig_slices[ellipsis_idx:ellipsis_idx + 1] = ell_fill

        valid_slices = [v for v in orig_slices if v is not None]
        norm_slices = []
        for i, (v, dim_sz) in enumerate(zip(valid_slices, self.shape)):
            if isinstance(v, slice):
                norm_slices.append(v)
            elif isinstance(v, int):
                norm_int = normalize_int(v, i, dim_sz)
                norm_slices.append(slice(norm_int, norm_int + 1))
            else:
                norm_slices.append(slice(None))
        valid_slices = norm_slices

        indices = [s.indices(dim_sz) for s, dim_sz in zip(valid_slices, self.shape)]
        if indices:
            start, stop, strides = zip(*indices)
        else:
            start, stop, strides = (), (), ()

        new_slice = []
        for s, e, st in zip(start, stop, strides):
            if st > 0:
                slc_range = (0, 0) if e < s else (s, e)
            else:
                slc_range = (0, 0) if e > s else (e + 1, s + 1)
            new_slice.append(slc_range)
        new_slice = tuple(new_slice)

        flip_axes = [i for i, s in enumerate(strides) if s < 0]
        sliced = self.shrink(new_slice).flip(axis=flip_axes)
        new_shape = sliced.shape

        if any(abs(s) != 1 for s in strides):
            strides = tuple(abs(s) for s in strides)
            # Pad: add pad at the end: [dim_sz] -> [dim_sz_padded]
            padding = []
            for s, dim_sz in zip(strides, sliced.shape):
                pad_right = s - (dim_sz % s) if dim_sz % s != 0 else 0
                padding.append((0, pad_right))
            padded = sliced.pad(tuple(padding))

            # Reshape: [dim_sz_padded] -> [dim_sz_padded // s, s]
            reshapes = [[sh // s, s] for sh, s in zip(padded.shape, strides)]
            reshaped = padded.reshape(flatten(reshapes))
            new_shape = reshaped.shape[::2]

            # Shrink: do [:, 0]
            shrinks = [((0, sh), (0, 1)) for sh in new_shape]
            sliced = reshaped.shrink(tuple(flatten(shrinks)))

        final_shape, it_shape, dim, tensors, dim_collapsed = [], iter(new_shape), [], [], 0
        for i, s in enumerate(orig_slices):
            if s is None:
                final_shape.append(1)
            else:
                dim_shape = next(it_shape)
                if isinstance(s, int):
                    dim_collapsed += 1
                else:
                    assert isinstance(dim_shape, int), f"does not support symbolic shape {dim_shape}"
                    final_shape.append(dim_shape)
                    if isinstance(s, Tensor):
                        tensors.append(s)
                        dim.append(i - dim_collapsed)
        ret = sliced.reshape(tuple(final_shape))

        if tensors:  # Fancy/tensor indexing
            # normalize idx
            # TODO: first contiguous fixes torch+cpu_only CI, but it causes llvm to fail. Second one fixes llvm
            idx = []
            for d, t in zip(dim, tensors):
                neg_mask = t.sign().contiguous().__neg__().contiguous().relu()
                norm_idx = neg_mask * ret.shape[d] + t
                idx.append(norm_idx)

            max_dim = max(i.ndim for i in idx)

            # compute sum_dim, arange, and idx
            sum_dim = [d if n == 0 else d + max_dim - n for n, d in enumerate(dim)]

            arange = []
            for n, (sd, d) in enumerate(zip(sum_dim, dim)):
                lead_ones = [1] * sd
                trail_ones = [1] * (ret.ndim + max_dim - n - sd - 1)
                ar_shape = (*lead_ones, ret.shape[d], *trail_ones)
                ar = Tensor.arange(ret.shape[d], dtype=dtypes.int32, requires_grad=False, device=self.device)
                arange.append(ar.reshape(*ar_shape))

            first_lead = [1] * dim[0]
            first_mid = [1] * (1 + max_dim - idx[0].ndim)
            first_trail = [1] * (ret.ndim - dim[0] - 1)
            first_shape = (*first_lead, *first_mid, *idx[0].shape, *first_trail)
            first_idx = [idx[0].reshape(*first_shape)]

            rest_idx = []
            for n, i in enumerate(idx[1:], 1):
                rest_lead = [1] * dim[0]
                rest_mid = [1] * (max_dim - i.ndim)
                rest_trail = [1] * (ret.ndim - dim[0] - n)
                rest_shape = (*rest_lead, *rest_mid, *i.shape, *rest_trail)
                rest_idx.append(i.reshape(*rest_shape))

            idx = first_idx + rest_idx

            ret_shape = (*ret.shape[:sum_dim[0] + 1], *[1] * max_dim, *ret.shape[sum_dim[0] + 1:])
            ret = ret.reshape(*ret_shape)

            # iteratively fancy index
            for a, i, sd in zip(arange, idx, sum_dim):
                ret = (a == i).mul(ret).sum(sd)

            # special permute case
            non_consec = dim != list(range(dim[0], dim[-1] + 1))
            if dim[0] != 0 and len(dim) != 1 and non_consec:
                ret_dims = list(range(ret.ndim))
                perm_order = ret_dims[dim[0]:dim[0] + max_dim] + ret_dims[:dim[0]] + ret_dims[dim[0] + max_dim:]
                ret = ret.permute(perm_order)

        return ret

    def __setitem__(self, s, v):
        return self.__getitem__(s).assign(v)

    # NOTE: using slice is discouraged and things should migrate to pad and shrink
    def slice(self, arg: Sequence[Optional[Tuple[int, sint]]], value: float = 0) -> Tensor:
        arg_ = tuple([a if a is not None else (0, s) for s, a in zip(self.shape, arg)])
        padding = tuple([(max(0, -p[0]), max(0, p[1] - self.shape[i])) for i, p in enumerate(arg_)])
        return self.pad(padding, value=value).shrink(tuple([(p[0] + padding[i][0], p[1] + padding[i][0])
                                                            for i, p in enumerate(arg_)]))

    def gather(self: Tensor, idx: Tensor, dim: int) -> Tensor:
        assert idx.ndim == self.ndim, "self.ndim must equal idx.ndim"
        assert all(s >= i for s, i in zip(self.shape, idx.shape)), "all dim of idx.shape must be smaller than self.shape"

        if dim < 0:
            dim += self.ndim

        idx = idx.transpose(ax1=dim, ax2=0).unsqueeze(-1)

        permarg = list(range(self.ndim))
        if dim != 0:
            permarg = permarg[1:dim] + [permarg[0]] + permarg[dim + 1:] + [permarg[dim]]
        else:
            permarg = permarg[1:] + [permarg[0]]

        oh_enc = (idx == Tensor.arange(self.shape[dim], dtype=dtypes.int32, requires_grad=False, device=self.device))
        slices = [*[(0, sh) for sh in idx.shape[1:-1]], (0, self.shape[dim])]
        aligned = self.permute(*permarg).shrink(tuple(slices)).unsqueeze(0)

        return (oh_enc * aligned).sum(-1).transpose(ax1=0, ax2=dim)

    def cat(self, *args, dim=0) -> Tensor:
        dim = (dim + len(self.shape)) if dim < 0 else dim
        assert all(len(y.shape) == len(self.shape) and all(y.shape[i] == s for i, s in enumerate(self.shape) if i != dim) for y in args)

        catargs = [self, *args]
        assert all(t.shape for t in catargs), "zero-dimensional tensor cannot be concatenated"

        shapes = [s.shape[dim] for s in catargs]
        shape_cumsum = [0, *accumulate(shapes)]
        slc = [[(0, 0) for _ in self.shape] for _ in catargs]
        for shp, k, s in zip(shapes, shape_cumsum[:-1], slc):
            s[dim] = (k, shape_cumsum[-1] - k - shp)
        return reduce(Tensor.__add__, [arg.pad(tuple(s)) for arg, s in zip(catargs, slc)])

    @staticmethod
    def stack(tensors, dim=0) -> Tensor:
        first = tensors[0].unsqueeze(dim)
        unsqueezed_tensors = [tensor.unsqueeze(dim) for tensor in tensors[1:]]
        # checks for shapes and number of dimensions delegated to cat
        return first.cat(*unsqueezed_tensors, dim=dim)

    def repeat(self, repeats) -> Tensor:
        base_shape = (1,) * (len(repeats) - self.ndim) + self.shape
        new_shape = [x for b in base_shape for x in [1, b]]
        expand_shape = [x for rs in zip(repeats, base_shape) for x in rs]
        final_shape = [r * s for r, s in zip(repeats, base_shape)]
        return self.reshape(new_shape).expand(expand_shape).reshape(final_shape)

    def chunk(self, num: int, dim: int = 0) -> List[Tensor]:
        assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
        dim, step = dim + self.ndim if dim < 0 else dim, math.ceil(self.shape[dim] / num)
        slice_params = [[slice(None)] * dim + [slice(k, k + step)] for k in range(0, self.shape[dim], step)]
        return [self[tuple(sl)] for sl in slice_params]

    def squeeze(self, dim=None) -> Tensor:
        if dim is None:
            return self if 1 not in self.shape else self.reshape(*[size for size in self.shape if size != 1])
        if dim <= 0 and self.ndim == 0:
            return self  # This is to match PyTorch behavior
        if not -self.ndim <= dim < self.ndim:
            raise IndexError(f"Dimension out of range (expected to be in range of [{-self.ndim if self.ndim > 0 else self.ndim - 1}, {self.ndim - 1 if self.ndim > 0 else self.ndim}], but got {dim})")
        if dim < 0:
            dim += self.ndim
        return self if self.shape[dim] != 1 else self.reshape(*[size for idx, size in enumerate(self.shape) if idx != dim])

    def unsqueeze(self, dim) -> Tensor:
        if dim < 0:
            dim = len(self.shape) + dim + 1
        return self.reshape(self.shape[:dim] + (1,) + self.shape[dim:])

    # (padding_left, padding_right, padding_top, padding_bottom)
    def pad2d(self, padding: Union[List[int], Tuple[int, ...]], value: float = 0) -> Tensor:
        slc = [(-p0, s + p1) for p0, p1, s in zip(padding[::2], padding[1::2], self.shape[::-1])][::-1]
        return self.slice([(0, s) for s in self.shape[:-(len(padding) // 2)]] + slc, value=value)

    @property
    def T(self) -> Tensor:
        return self.transpose()

    def transpose(self, ax1=1, ax2=0) -> Tensor:
        order = list(range(len(self.shape)))
        order[ax1], order[ax2] = order[ax2], order[ax1]
        return self.permute(order)

    def flatten(self, start_dim=0):
        return self.reshape(shape=self.shape[:start_dim] + (-1,))

    # ***** reduce ops *****
    def _reduce(self, fxn: Type[Function], axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdim=False) -> Tensor:
        if axis is None:
            axis_ = list(range(len(self.shape)))
        elif isinstance(axis, int):
            axis_ = [axis]
        else:
            axis_ = list(axis)

        axis_ = [x if x >= 0 else x + len(self.shape) for x in axis_]
        shape = tuple(s for i, s in enumerate(self.shape) if i not in axis_)

        has_zero_in = 0 in self.shape
        has_zero_out = 0 in shape
        if has_zero_in and not has_zero_out:
            defaults = {mlops.Sum: 0, mlops.Max: -float("inf")}
            fill = defaults[fxn]
            out_shape = tuple(1 if s == 0 else s for s in self.shape) if keepdim else shape
            return Tensor.full(out_shape, fill)

        new_shape = tuple([1 if i in axis_ else s for i, s in enumerate(self.shape)])
        ret = fxn.apply(self, new_shape=new_shape)
        return ret if keepdim else ret.reshape(shape=shape)

    def sum(self, axis=None, keepdim=False):
        return self._reduce(mlops.Sum, axis, keepdim)

    def max(self, axis=None, keepdim=False):
        return self._reduce(mlops.Max, axis, keepdim)

    def min(self, axis=None, keepdim=False):
        return -((-self).max(axis=axis, keepdim=keepdim))

    def mean(self, axis=None, keepdim=False):
        assert all_int(self.shape), "does not support symbolic shape"
        out = self.sum(axis=axis, keepdim=keepdim)
        return out.mul(prod(out.shape) / prod(self.shape)) if 0 not in self.shape else out

    def std(self, axis=None, keepdim=False, correction=1):
        assert all_int(self.shape), "does not support symbolic shape"
        square_sum = ((self - self.mean(axis=axis, keepdim=True)).square()).sum(axis=axis, keepdim=keepdim)
        return square_sum.div(prod(self.shape) / prod(square_sum.shape) - correction).sqrt()

    def _softmax(self, axis):
        m = self - self.max(axis=axis, keepdim=True)
        e = m.exp()
        return m, e, e.sum(axis=axis, keepdim=True)

    def softmax(self, axis=-1):
        _, e, ss = self._softmax(axis)
        return e.div(ss)

    def log_softmax(self, axis=-1):
        m, _, ss = self._softmax(axis)
        return m - ss.log()

    def argmax(self, axis=None, keepdim=False):
        if axis is None:
            mask = (self == self.max(axis))
            rev_idx = Tensor.arange(prod(self.shape) - 1, -1, -1, dtype=dtypes.int32, requires_grad=False, device=self.device)
            idx = mask * rev_idx.reshape(self.shape)
            max_idx = idx.max()
            return prod(self.shape) - max_idx - 1

        axis = axis + len(self.shape) if axis < 0 else axis
        max_val = self.max(axis=axis, keepdim=True)
        mask = self == max_val

        trailing = [1] * (self.ndim - axis - 1)
        rev_idx = Tensor.arange(self.shape[axis] - 1, -1, -1, dtype=dtypes.int32, requires_grad=False, device=self.device)
        rev_shaped = rev_idx.reshape(self.shape[axis], *trailing)

        idx = mask * rev_shaped
        max_idx = idx.max(axis=axis, keepdim=keepdim)
        return self.shape[axis] - max_idx - 1

    def argmin(self, axis=None, keepdim=False):
        return (-self).argmax(axis=axis, keepdim=keepdim)

    def dot(self, w: Tensor) -> Tensor:
        n1, n2 = len(self.shape), len(w.shape)
        assert n1 != 0 and n2 != 0, f"both arguments to matmul need to be at least 1D, but they are {n1}D and {n2}D"
        assert self.shape[-1] == w.shape[-min(n2, 2)], f"Input Tensor shapes {self.shape} and {w.shape} cannot be multiplied ({self.shape[-1]} != {w.shape[-min(n2, 2)]})"
        x = self.reshape(*self.shape[0:-1], *[1] * min(n1 - 1, n2 - 1, 1), self.shape[-1])
        w = w.reshape(*w.shape[0:-2], *[1] * min(n1 - 1, n2 - 1, 1), *w.shape[-min(n2, 2):]).transpose(-1, -min(n2, 2))
        return (x * w).sum(-1)

    def _cumsum(self, axis: int = 0, _first_zero=False) -> Tensor:
        return self.transpose(axis, -1).pad2d((self.shape[axis] - int(not _first_zero), 0))._pool((self.shape[axis],)).sum(-1).transpose(axis, -1)

    def cumsum(self, axis: int = 0) -> Tensor:
        # TODO: someday the optimizer will find this on it's own
        # for now this is a two stage cumsum
        SPLIT = 256
        if self.shape[axis] <= SPLIT * 2:
            return self._cumsum(axis)
        ret = self.transpose(axis, -1).pad2d((round_up(self.shape[axis], SPLIT) - self.shape[axis], 0))
        ret = ret.reshape(*ret.shape[0:-1], ret.shape[-1] // SPLIT, SPLIT)._cumsum(-1)
        base_add = ret[..., -1]._cumsum(-1, _first_zero=True)[..., :-1]
        base_add = base_add.unsqueeze(-1).expand(*base_add.shape, ret.shape[-1])

        def fix(x: Tensor):
            return x.reshape(*ret.shape[0:-2], ret.shape[-2] * ret.shape[-1])[..., -self.shape[axis]:].transpose(axis, -1)

        return fix(ret) + fix(base_add)

    @staticmethod
    def _tri(r: int, c: int, k: int = 0, **kwargs) -> Tensor:
        return Tensor.arange(r, **kwargs).unsqueeze(1).expand(r, c) <= Tensor.arange(-k, c - k, **kwargs).unsqueeze(0).expand(r, c)

    def triu(self, k: int = 0) -> Tensor:
        assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
        return Tensor._tri(self.shape[-2], self.shape[-1], k=k, dtype=self.dtype, device=self.device).where(self, Tensor.zeros_like(self))

    def tril(self, k: int = 0) -> Tensor:
        assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
        return Tensor._tri(self.shape[-2], self.shape[-1], k=k + 1, dtype=self.dtype, device=self.device).where(Tensor.zeros_like(self), self)

    # ***** mlops (unary) *****

    def neg(self):
        return mlops.Neg.apply(self)

    def contiguous(self):
        return mlops.Contiguous.apply(self)

    def contiguous_backward(self):
        return mlops.ContiguousBackward.apply(self)

    def log(self):
        return mlops.Log.apply(self)

    def log2(self):
        return mlops.Log.apply(self) / math.log(2)

    def exp(self):
        return mlops.Exp.apply(self)

    def exp2(self):
        return mlops.Exp.apply(self * math.log(2))

    def relu(self):
        return mlops.Relu.apply(self)

    def sigmoid(self):
        return mlops.Sigmoid.apply(self)

    def sin(self):
        return mlops.Sin.apply(self)

    def sqrt(self):
        return mlops.Sqrt.apply(self)

    def rsqrt(self):
        return (1 / self).sqrt()

    def cos(self):
        return ((math.pi / 2) - self).sin()

    def tan(self):
        return self.sin() / self.cos()

    # ***** math functions (unary) *****

    def trunc(self: Tensor) -> Tensor:
        return self.cast(dtypes.int32).contiguous().cast(self.dtype)

    def ceil(self: Tensor) -> Tensor:
        return (self > (b := self.trunc())).where(b + 1, b)

    def floor(self: Tensor) -> Tensor:
        return (self < (b := self.trunc())).where(b - 1, b)

    def square(self):
        return self * self

    def clip(self, min_, max_):
        return self.maximum(min_).minimum(max_)

    def abs(self):
        return self.relu() + (-self).relu()

    def sign(self):
        return self / (self.abs() + 1e-10)

    def reciprocal(self):
        return 1.0 / self

    # ***** activation functions (unary) *****

    def elu(self, alpha=1.0):
        return self.relu() - alpha * (1 - self.exp()).relu()

    def celu(self, alpha=1.0):
        return self.maximum(0) + (alpha * ((self / alpha).exp() - 1)).minimum(0)

    def swish(self):
        return self * self.sigmoid()

    def silu(self):
        return self.swish()  # The SiLU function is also known as the swish function.

    def relu6(self):
        return self.relu() - (self - 6).relu()

    def hardswish(self):
        return self * (self + 3).relu6() * (1 / 6)

    def tanh(self):
        return 2.0 * ((2.0 * self).sigmoid()) - 1.0

    def sinh(self):
        return (self.exp() - self.neg().exp()) / 2

    def cosh(self):
        return (self.exp() + self.neg().exp()) / 2

    def atanh(self):
        return ((1 + self) / (1 - self)).log() / 2

    def asinh(self):
        return (self + (self.square() + 1).sqrt()).log()

    def acosh(self):
        return (self + (self.square() - 1).sqrt()).log()

    def hardtanh(self, min_val=-1, max_val=1):
        return self.clip(min_val, max_val)

    def gelu(self):
        return 0.5 * self * (1 + (self * 0.7978845608 * (1 + 0.044715 * self * self)).tanh())

    def quick_gelu(self):
        return self * (self * 1.702).sigmoid()

    def leakyrelu(self, neg_slope=0.01):
        return self.relu() - (-neg_slope * self).relu()

    def mish(self):
        return self * self.softplus().tanh()

    def softplus(self, beta=1):
        return (1 / beta) * (1 + (self * beta).exp()).log()

    def softsign(self):
        return self / (1 + self.abs())

    # ***** broadcasted binary mlops *****

    def _broadcasted(self, y: Union[Tensor, float], reverse: bool = False) -> Tuple[Tensor, Tensor]:
        x: Tensor = self
        if not isinstance(y, Tensor):
            if 0 in x.shape:
                return x, x.full_like(y)
            y = Tensor(y, device=self.device, requires_grad=False, dtype=self.dtype if self.dtype != dtypes.bool and self.dtype.__class__ is not None else dtypes.float32)
        if reverse:
            x, y = y, x
        if (xshape := x.shape) == (yshape := y.shape):
            return (x, y)

        shape_delta = len(xshape) - len(yshape)
        if shape_delta > 0:
            y = y.reshape((1,) * shape_delta + yshape)
        elif shape_delta < 0:
            x = x.reshape((1,) * -shape_delta + xshape)
        if (xshape := x.shape) == (yshape := y.shape):
            return (x, y)

        shape_ret = tuple([max(x, y) for x, y in zip(xshape, yshape)])
        if xshape != shape_ret:
            x = x.expand(shape_ret)
        if yshape != shape_ret:
            y = y.expand(shape_ret)
        return (x, y)

    def _to_float(self, x: Union[Tensor, float]):
        if isinstance(x, Tensor):
            is_unrealized_const = x.lazydata.is_unrealized_contiguous_const()
            has_no_grad = not x.requires_grad
            broadcasted_x, _ = self._broadcasted(x)
            has_same_shape = broadcasted_x.shape == self.shape

            if is_unrealized_const and has_no_grad and has_same_shape:
                return x.lazydata.base.op.arg

        return x

    def add(self, x: Union[Tensor, float], reverse=False) -> Tensor:
        x = self._to_float(x)
        return mlops.Add.apply(*self._broadcasted(x, reverse)) if x.__class__ is Tensor or x else self

    def sub(self, x: Union[Tensor, float], reverse=False) -> Tensor:
        x = self._to_float(x)
        return mlops.Sub.apply(*self._broadcasted(x, reverse)) if x.__class__ is Tensor or x else (-self if reverse else self)

    def mul(self, x: Union[Tensor, float], reverse=False) -> Tensor:
        x = self._to_float(x)
        if x.__class__ is not Tensor and x == 0.0:
            return mlops.Zero.apply(self)
        if x.__class__ is not Tensor and x == -1.0:
            return -self
        return mlops.Mul.apply(*self._broadcasted(x, reverse)) if x.__class__ is Tensor or x != 1.0 else self

    def div(self, x: Union[Tensor, float], reverse=False) -> Tensor:
        x = self._to_float(x)
        return mlops.Div.apply(*self._broadcasted(x, reverse)) if x.__class__ is Tensor or reverse or not x or not dtypes.is_float(self.dtype) else self.mul(1 / x)

    def pow(self, x: Union[Tensor, float], reverse=False) -> Tensor:
        x = self._to_float(x)
        if x.__class__ is not Tensor and not reverse:
            # simple pow identities
            if x < 0:
                return self.reciprocal().pow(-x)
            if x == 3.0:
                return self * self * self
            if x == 2.0:
                return self * self
            if x == 1.0:
                return self
            if x == 0.5:
                return self.sqrt()

        if not isinstance(x, Tensor) and reverse and x > 0:
            return self.mul(math.log(x)).exp()
        ar = self.abs().log().mul(x).exp() if not reverse or isinstance(x, Tensor) else self.mul(math.log(abs(x))).exp()
        # correct sign of negative numbers raised to a power (cos has a period of 2pi so we use it here to get the oddness of the power)
        sign = (x * math.pi).cos() if isinstance(x, Tensor) else math.cos(x * math.pi) if not reverse else (self * math.pi).cos()
        # we only need to correct the sign if the base is negative
        base_sign = ((self.sign() if not reverse else x.sign() if isinstance(x, Tensor) else math.copysign(1, x)) - 1) / -2
        # we need 0 to be positive so we need to correct base_sign when the base is 0
        base_sign = base_sign - (1.5 * (1 - (self.sign().abs() if not reverse else x.sign().abs() if isinstance(x, Tensor) else abs(int(bool(x))))))
        # inject nan if the base is negative and the power is not an integer
        to_nan = (((x - x.trunc()) * 1e10).abs().clip(0, 1) if isinstance(x, Tensor) else int(bool(x - int(x))) if not reverse else ((self - self.trunc()) * 1e10).abs().clip(0, 1)) * base_sign
        inject_nan = ((((-to_nan) * 2) + 1)).log().add(1) if isinstance(to_nan, Tensor) else 1 if not to_nan else float("nan")
        return ar.mul(sign * base_sign + (1 - base_sign)).mul(inject_nan)

    def matmul(self, x: Tensor, reverse=False) -> Tensor:
        return x.dot(self) if reverse else self.dot(x)

    def maximum(self, x: Union[Tensor, float]) -> Tensor:
        return (self < x).detach().where(x, (self > x).detach().where(self, (self + x) / 2))

    def minimum(self, x: Union[Tensor, float]) -> Tensor:
        return -((-self).maximum(-x))

    def where(self: Tensor, input_: Union[Tensor, float], other: Union[Tensor, float]):
        x_, y = self._broadcasted(input_)
        x, z = x_._broadcasted(other)
        return mlops.Where.apply(x, *y._broadcasted(z))

    # ***** op wrappers *****

    def __neg__(self) -> Tensor:
        return self.neg()

    def __add__(self, x) -> Tensor:
        return self.add(x)

    def __sub__(self, x) -> Tensor:
        return self.sub(x)

    def __mul__(self, x) -> Tensor:
        return self.mul(x)

    def __pow__(self, x) -> Tensor:
        return self.pow(x)

    def __truediv__(self, x) -> Tensor:
        return self.div(x)

    def __matmul__(self, x) -> Tensor:
        return self.matmul(x)

    def __radd__(self, x) -> Tensor:
        return self.add(x, True)

    def __rsub__(self, x) -> Tensor:
        return self.sub(x, True)

    def __rmul__(self, x) -> Tensor:
        return self.mul(x, True)

    def __rpow__(self, x) -> Tensor:
        return self.pow(x, True)

    def __rtruediv__(self, x) -> Tensor:
        return self.div(x, True)

    def __rmatmul__(self, x) -> Tensor:
        return self.matmul(x, True)

    def __iadd__(self, x) -> Tensor:
        return self.assign(self.add(x))

    def __isub__(self, x) -> Tensor:
        return self.assign(self.sub(x))

    def __imul__(self, x) -> Tensor:
        return self.assign(self.mul(x))

    def __ipow__(self, x) -> Tensor:
        return self.assign(self.pow(x))

    def __itruediv__(self, x) -> Tensor:
        return self.assign(self.div(x))

    def __imatmul__(self, x) -> Tensor:
        return self.assign(self.matmul(x))

    def __lt__(self, x) -> Tensor:
        return mlops.Less.apply(*self._broadcasted(x, False))

    def __gt__(self, x) -> Tensor:
        return mlops.Less.apply(*self._broadcasted(x, True))

    def __ge__(self, x) -> Tensor:
        return 1.0 - (self < x)

    def __le__(self, x) -> Tensor:
        return 1.0 - (self > x)

    def __ne__(self, x) -> Tensor:
        return (self < x) + (self > x)  # type: ignore

    def __eq__(self, x) -> Tensor:
        return 1.0 - (self != x)  # type: ignore

    # ***** cast ops *****

    def cast(self, dtype: DType) -> Tensor:
        return mlops.Cast.apply(self, dtype=dtype) if self.dtype != dtype else self

    def bitcast(self, dtype: DType) -> Tensor:
        assert self.dtype.itemsize == dtype.itemsize, "can't bitcast mismatched dtype itemsizes"
        return mlops.Cast.apply(self, dtype=dtype, bitcast=True) if self.dtype != dtype else self

    def float(self) -> Tensor:
        return self.cast(dtypes.float32)

    def half(self) -> Tensor:
        return self.cast(dtypes.float16)

    # ***** convenience stuff *****

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def numel(self) -> sint:
        return prod(self.shape)

    def element_size(self) -> int:
        return self.dtype.itemsize

    def nbytes(self) -> int:
        return self.numel() * self.element_size()

    def is_floating_point(self) -> bool:
        return dtypes.is_float(self.dtype)

    # ***** functional nn ops *****

    def linear(self, weight: Tensor, bias: Optional[Tensor] = None):
        x = self.mul(weight) if len(weight.shape) == 1 else self.dot(weight)
        return x.add(bias) if bias is not None else x

    def sequential(self, ll: List[Callable[[Tensor], Tensor]]):
        return reduce(lambda x, f: f(x), ll, self)

    def layernorm(self, axis=-1, eps: float = 1e-5) -> Tensor:
        mean = self.mean(axis, keepdim=True)
        centered = self - mean

        var = (centered * centered).mean(axis, keepdim=True)
        inv_std = var.add(eps).rsqrt()

        return centered.mul(inv_std)

    def batchnorm(self, weight: Optional[Tensor], bias: Optional[Tensor], mean: Tensor, invstd: Tensor) -> Tensor:
        x = (self - mean.reshape(shape=[1, -1, 1, 1]))
        if weight: x = x * weight.reshape(shape=[1, -1, 1, 1])
        ret = x.mul(invstd.reshape(shape=[1, -1, 1, 1]) if len(invstd.shape) == 1 else invstd)
        return (ret + bias.reshape(shape=[1, -1, 1, 1])) if bias else ret

    def dropout(self, p=0.5) -> Tensor:
        if not Tensor.training or p == 0:
            return self
        mask = (Tensor.rand(*self.shape, requires_grad=False, device=self.device) >= p).cast(dtypes.bool)
        return self * mask * (1 / (1.0 - p))

    def scaled_dot_product_attention(self,
                                     key: Tensor,
                                     value: Tensor,
                                     attn_mask: Optional[Tensor] = None,
                                     dropout_p: float = 0.0,
                                     is_causal: bool = False) -> Tensor:
        # NOTE: it works if key, value have symbolic shape
        assert all_int(self.shape), f"does not support symbolic shape {self.shape}"

        if is_causal:
            ones = Tensor.ones(self.shape[-2], key.shape[-2], requires_grad=False, device=self.device)
            tril = ones.tril(0)
            attn_mask = tril.cast(dtypes.bool)

        if attn_mask is not None and attn_mask.dtype == dtypes.bool:
            attn_mask = (attn_mask == 0).where(-float("inf"), attn_mask)

        kt = key.transpose(-2, -1)
        scores = self @ kt
        scale = math.sqrt(self.shape[-1])
        scaled = scores / scale
        masked = scaled + attn_mask
        weights = masked.softmax(-1)
        dropped = weights.dropout(dropout_p)
        return dropped @ value

    # ***** processing ops *****

    def _pool(self,
              k_: Tuple[sint, ...],
              stride: Union[Tuple[int, ...], int] = 1,
              dilation: Union[Tuple[int, ...], int] = 1) -> Tensor:
        assert len(self.shape) >= len(k_), f"can't pool {self.shape} with {k_}"
        assert all_int(self.shape) and all_int(k_), f"does not support symbolic {self.shape=}, {k_=}"

        s_, d_ = make_pair(stride, len(k_)), make_pair(dilation, len(k_))
        assert len(k_) == len(s_) and len(k_) == len(
            d_), f"stride/dilation mismatch kernel:{k_} stride:{s_} dilation:{d_}"

        slc_prefix = [(0, x) for x in self.shape[0:-len(k_)]]
        prefix = self.shape[0:-len(k_)]
        i_ = self.shape[-len(k_):]

        if any(k > s for k, s in zip(k_, s_)) or any(d != 1 for d in d_):
            o_ = [(i - d * (k - 1) - 1) // s + 1 for i, d, k, s in zip(i_, d_, k_, s_)]
            e_ = [math.ceil(k * (i + d) / i) for k, i, d in
                  zip(k_, i_, d_)]  # expands such that we don't need padding

            reshape_dims = (*prefix, *flatten((1, i) for i in i_))
            expand_dims = (*prefix, *flatten((e, i) for e, i in zip(e_, i_)))
            final_dims = (*prefix, *[e * i for e, i in zip(e_, i_)])
            xup = self.reshape(*reshape_dims).expand(*expand_dims).reshape(*final_dims)

            # slide by dilation
            slice_dims = slc_prefix + [(0, k * (i + d)) for k, i, d in zip(k_, i_, d_)]
            xup = xup.slice(slice_dims)

            reshape_dims = (*prefix, *flatten((k, i + d) for k, i, d in zip(k_, i_, d_)))
            xup = xup.reshape(*reshape_dims)

            slice_dims = slc_prefix + flatten(((0, k), (0, o * s)) for k, o, s in zip(k_, o_, s_))
            xup = xup.slice(slice_dims)

            # handle stride, and permute to move reduce to the end
            reshape_dims = (*prefix, *flatten((k, o, s) for k, o, s in zip(k_, o_, s_)))
            xup = xup.reshape(*reshape_dims)

            slice_dims = slc_prefix + flatten(((0, k), (0, o), (0, 1)) for k, o in zip(k_, o_))
            xup = xup.slice(slice_dims)

            reshape_dims = (*prefix, *flatten((k, o) for k, o in zip(k_, o_)))
            xup = xup.reshape(*reshape_dims)

            prefix_indices = list(range(len(prefix)))
            output_indices = [len(prefix) + i * 2 + 1 for i in range(len(k_))]
            kernel_indices = [len(prefix) + i * 2 for i in range(len(k_))]
            return xup.permute(*prefix_indices, *output_indices, *kernel_indices)

        # TODO: once the shapetracker can optimize well, remove this alternative implementation. or not if the CPU implementation doesn't use ShapeTracker
        o_ = [(i + (s - k)) // s for i, s, k in zip(i_, s_, k_)]

        slice_dims = slc_prefix + [(0, o * s) for o, s in zip(o_, s_)]
        xup = self.slice(slice_dims)

        reshape_dims = (*prefix, *flatten(((o, s) for o, s in zip(o_, s_))))
        xup = xup.reshape(*reshape_dims)

        slice_dims = slc_prefix + flatten(((0, o), (0, k)) for o, k in zip(o_, k_))
        xup = xup.slice(slice_dims)

        prefix_indices = list(range(len(prefix)))
        output_indices = [len(prefix) + i * 2 for i in range(len(k_))]
        kernel_indices = [len(prefix) + i * 2 + 1 for i in range(len(k_))]
        return xup.permute(*prefix_indices, *output_indices, *kernel_indices)

        # NOTE: these work for more than 2D

    def avg_pool2d(self, kernel_size=(2, 2), stride=None, dilation=1):
        return self._pool(make_pair(kernel_size), stride if stride is not None else kernel_size, dilation).mean(
            axis=tuple(range(0 - len(make_pair(kernel_size)), 0)))

    def max_pool2d(self, kernel_size=(2, 2), stride=None, dilation=1):
        return self._pool(make_pair(kernel_size), stride if stride is not None else kernel_size, dilation).max(
            axis=tuple(range(0 - len(make_pair(kernel_size)), 0)))

    def conv_transpose2d(self,
                         weight: Tensor,
                         bias: Optional[Tensor] = None,
                         groups=1,
                         stride=1,
                         dilation=1,
                         padding=0,
                         output_padding=0) -> Tensor:
        HW = weight.shape[2:]
        trailing = list(range(3, len(weight.shape) + 1))

        w_grouped = weight.reshape(groups, weight.shape[0] // groups, weight.shape[1], *weight.shape[2:])
        w_permuted = w_grouped.permute(0, 2, 1, *trailing)
        w = w_permuted.flip(trailing)

        x = self
        stride = make_pair(stride, len(HW))

        if any(s > 1 for s in stride):
            x_expanded = x.reshape(*x.shape[:2], *flatten((k, 1) for k in x.shape[2:]))
            pad_spec = ((0, 0), (0, 0), *flatten(((0, 0), (0, s - 1)) for s in stride))
            x_padded = x_expanded.pad(pad_spec)
            new_dims = [k * s for k, s in zip(x_padded.shape[2::2], stride)]
            x_merged = x_padded.reshape(*x_padded.shape[:2], *new_dims)
            shrink_spec = ((0, x_merged.shape[0]), (0, x_merged.shape[1]),
                           *[(0, k - (s - 1)) for k, s in zip(x_merged.shape[2:], stride)])
            x = x_merged.shrink(shrink_spec)

        dilation_pair = make_pair(dilation, len(HW))
        padding_pair = make_pair(padding, len(HW))
        output_pad_pair = make_pair(output_padding, len(HW))
        padding_calc = (((k - 1) * d - p, (k - 1) * d - p + op)
                        for k, d, p, op in reversed(list(zip(HW, dilation_pair, padding_pair, output_pad_pair))))
        padding = flatten(padding_calc)

        w_merged = w.reshape(w.shape[0] * w.shape[1], *w.shape[2:])
        return x.conv2d(w_merged, groups=groups, bias=bias, dilation=dilation, padding=padding)

    wino = int(getenv("WINO", "0"))

    def conv2d(self,
               weight: Tensor,
               bias: Optional[Tensor] = None,
               groups=1,
               stride=1,
               dilation=1,
               padding=0) -> Tensor:
        bs, cin_ = self.shape[:2]
        cout, cin = weight.shape[:2]
        HW = weight.shape[2:]

        assert groups * cin == cin_ and len(self.shape) == len(
            weight.shape), f"Input Tensor shape {self.shape} does not match the shape of the weights {weight.shape}. ({groups * cin} vs. {cin_})"
        if isinstance(padding, (tuple, list)):
            assert len(padding) == 2 * len(HW) or len(padding) == len(
                HW), f"Expected padding of length {2 * len(HW)} or {len(HW)}, but got {len(padding)} for tensor of shape {self.shape}"

        if isinstance(padding, int):
            padding_ = [padding] * 2 * len(HW)
        elif len(padding) == 2 * len(HW):
            padding_ = padding
        else:
            padding_ = [p for p in padding for _ in range(2)][::-1]

        # conv2d is a pooling op (with padding)
        x_padded = self.pad2d(padding_)
        x = x_padded._pool(HW, stride, dilation)  # (bs, groups*cin, oy, ox, H, W)
        rcout = cout // groups
        oyx = x.shape[2:-len(HW)]

        use_winograd = all(k == 3 for k in HW) and stride == 1 and dilation == 1 and Tensor.wino
        if not use_winograd:
            # normal conv
            x_reshaped = x.reshape(bs, groups, cin, 1, *oyx, *HW)
            x_expanded = x_reshaped.expand(bs, groups, cin, rcout, *oyx, *HW)
            perm_order = [0, 1, 3, *[4 + i for i in range(len(oyx))], 2,
                          *[4 + len(oyx) + i for i in range(len(HW))]]
            x = x_expanded.permute(*perm_order)

            # conv! broadcasted to (bs, groups, rcout, *oyx, cin, *HW)
            w_reshaped = weight.reshape(1, groups, rcout, *[1] * len(oyx), cin, *HW)
            multiplied = x * w_reshaped
            sum_axes = [-1 - i for i in range(1 + len(oyx))]
            summed = multiplied.sum(sum_axes, keepdim=True)
            ret = summed.reshape(bs, cout, *oyx)

            if bias is None:
                return ret
            else:
                bias_reshaped = bias.reshape(1, -1, *[1] * len(HW))
                return ret.add(bias_reshaped)

        # winograd conv 3 kernel f(4x4,3x3) see: http://arxiv.org/abs/1509.09308

        def apply_matrix(mat, t, dim=0):
            if dim == len(HW):
                return t
            stacked = [apply_matrix(mat, sum(mm * t[j] for j, mm in enumerate(m) if mm), dim=dim + 1) for m in mat]
            return Tensor.stack(stacked)

        HWI, HWO = (6,) * len(HW), (4,) * len(HW)  # F(4x4,3x3) winograd tiles
        winograd_Bt = [
            [4, 0, -5, 0, 1, 0],
            [0, -4, -4, 1, 1, 0],
            [0, 4, -4, -1, 1, 0],
            [0, -2, -1, 2, 1, 0],
            [0, 2, -1, -2, 1, 0],
            [0, 4, 0, -5, 0, 1]
        ]
        winograd_G = [
            [1 / 4, 0, 0],
            [-1 / 6, -1 / 6, -1 / 6],
            [-1 / 6, 1 / 6, -1 / 6],
            [1 / 24, 1 / 12, 1 / 6],
            [1 / 24, -1 / 12, 1 / 6],
            [0, 0, 1]
        ]
        winograd_At = [
            [1, 1, 1, 1, 1, 0],
            [0, 1, -1, 2, -2, 0],
            [0, 1, 1, 4, 4, 0],
            [0, 1, -1, 8, -8, 1]
        ]  # applying At in pre-order almost doubles compilation time

        # todo: stride == dilation
        # use padding to round up to 4x4 output tiles
        tile_pad = []
        for i, dim in enumerate(self.shape[-len(HW):]):
            left_pad = padding_[i * 2]
            right_base = padding_[i * 2 + 1]
            pad_sum = sum(padding_[i * 2:(i + 1) * 2])
            extra_pad = (-(dim + pad_sum - 2) % 4)
            right_pad = right_base + extra_pad
            tile_pad.extend([left_pad, right_pad])

        d_padded = self.pad2d(tile_pad)
        d_pooled = d_padded._pool(HWI, HWO)  # (bs, cin_, tyx, HWI)
        perm_front = list(range(len(d_pooled.shape) - len(HW), len(d_pooled.shape)))
        perm_rest = list(range(len(d_pooled.shape) - len(HW)))
        d = d_pooled.permute(*perm_front,
                             *perm_rest).contiguous_backward()  # move HW to the front: (HWI, bs, cin_, tyx)
        tyx = d.shape[-len(HWI):]  # dim of tiling

        g_perm_front = list(range(len(weight.shape) - len(HW), len(weight.shape)))
        g_perm_rest = list(range(len(weight.shape) - len(HW)))
        g = weight.permute(*g_perm_front, *g_perm_rest)  # move HW to the front

        # compute 6x6 winograd tiles: GgGt, BtdB
        g_transformed = apply_matrix(winograd_G, g).contiguous()
        gfactors = g_transformed.reshape(*HWI, 1, groups, rcout, cin, *(
                    [1] * len(tyx)))  # (HWI, groups * rcout, cin) -> (HWI, bs=1, groups, rcout, cin, tyx=(1,1))

        d_transformed = apply_matrix(winograd_Bt, d).contiguous()
        dfactors = d_transformed.reshape(*HWI, bs, groups, 1, cin,
                                         *tyx)  # (HWI, bs, cin_, tyx) -> (HWI, bs, groups, 1 ,cin, *tyx)

        product = gfactors * dfactors
        sum_axis = -1 - len(HW)
        product_summed = product.sum(axis=sum_axis)
        ret = apply_matrix(winograd_At,
                           product_summed)  # matmul; sum across cin: (HWI, bs, groups, rcout, *tyx); then HWI -> HWO: (HWO, bs, groups, rcout, *tyx)

        interleave_order = [*range(len(HW), len(ret.shape) - len(HW)),
                            *[i + o for i in range(len(HW)) for o in [len(ret.shape) - len(HW), 0]]]
        ret_permuted = ret.permute(interleave_order)  # interleave tyx and HWO: (bs, groups, rcout, oy, HO, ox, WO)

        merged_shape = [bs, cout, *[c * HWO[i] for i, c in enumerate(tyx)]]
        ret_merged = ret_permuted.reshape(*merged_shape)
        shrink_spec = tuple((0, s) for s in [bs, cout, *oyx])
        ret = ret_merged.shrink(
            shrink_spec)  # merge groups and rcout, tyx and HWO: (bs, groups, cout, *yx), shrink to final

        if bias is not None:
            bias_reshaped = bias.reshape(1, -1, *[1 for _ in range(len(HW))])
            ret = ret.add(bias_reshaped)

        return ret.contiguous().contiguous_backward()

    # ***** loss functions *****

    def binary_crossentropy(self, y: Tensor) -> Tensor:
        return (-y * self.log() - (1 - y) * (1 - self).log()).mean()

    def binary_crossentropy_logits(self, y: Tensor) -> Tensor:
        return (self.maximum(0) - y * self + (1 + self.abs().__neg__().exp()).log()).mean()

    def sparse_categorical_crossentropy(self, Y, ignore_index=-1) -> Tensor:
        # NOTE: self is a logits input
        mask = Y != ignore_index

        n_classes = self.shape[-1]
        batch_sz = Y.numel()
        indices = Tensor.arange(n_classes, dtype=dtypes.int32, requires_grad=False, device=self.device)
        counter = indices.unsqueeze(0).expand(batch_sz, n_classes)

        y_flat = Y.flatten().reshape(-1, 1)
        one_hot = (counter == y_flat).where(-1.0, 0)
        masked = one_hot * mask.reshape(-1, 1)
        y = masked.reshape(*Y.shape, n_classes)

        log_probs = self.log_softmax()
        weighted = -log_probs.mul(y)
        total = weighted.sum()
        count = mask.sum()
        return total / count

# register functions to move between devices
for device in Device._buffers:
    setattr(Tensor, f"{device.lower()}", partialmethod(Tensor.to, device))
