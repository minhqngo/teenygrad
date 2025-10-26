import unittest
import numpy as np
from typing import Any, List

from teenygrad.helpers import CI, DTYPES_DICT, getenv, DType, DEBUG, OSX
from teenygrad.ops import Device
from teenygrad.tensor import Tensor, dtypes


def is_dtype_supported(dtype: DType):
    # for GPU, cl_khr_fp16 isn't supported (except now we don't need it!)
    # for LLVM, it segfaults because it can't link to the casting function
    if dtype == dtypes.half:
        return not (CI and Device.DEFAULT in ["GPU", "LLVM"]) and Device.DEFAULT != "WEBGPU" and getenv("CUDACPU") != 1
    if dtype == dtypes.float64:
        return Device.DEFAULT not in ["WEBGPU", "METAL"] and (not OSX and Device.DEFAULT == "GPU")
    if dtype in [dtypes.int8, dtypes.uint8]:
        return Device.DEFAULT not in ["WEBGPU"]
    if dtype in [dtypes.int16, dtypes.uint16]:
        return Device.DEFAULT not in ["WEBGPU", "TORCH"]
    if dtype == dtypes.uint32:
        return Device.DEFAULT not in ["TORCH"]
    if dtype in [dtypes.int64, dtypes.uint64]:
        return Device.DEFAULT not in ["WEBGPU", "TORCH"]
    if dtype == dtypes.bool:
        # host-shareablity is a requirement for storage buffers, but 'bool' type is not host-shareable
        if Device.DEFAULT == "WEBGPU":
            return False
    return True


def get_available_cast_dtypes(dtype: DType) -> List[DType]:
    return [v for k, v in DTYPES_DICT.items() if v != dtype and is_dtype_supported(v) and not k.startswith("_")]  # dont cast internal dtypes


def _test_to_np(a:Tensor, np_dtype, target):
    if DEBUG >= 2:
        print(a)
    na = a.numpy()
    if DEBUG >= 2:
        print(na, na.dtype, a.lazydata.realized)
    try:
        assert na.dtype == np_dtype
        np.testing.assert_allclose(na, target)
    except AssertionError as e:
        raise AssertionError(f"\ntensor {a.numpy()} does not match target {target} with np_dtype {np_dtype}") from e


def _assert_eq(tensor: Tensor, target_dtype: DType, target):
    if DEBUG >= 2:
        print(tensor.numpy())
    try:
        assert tensor.dtype == target_dtype
        np.testing.assert_allclose(tensor.numpy(), target)
    except AssertionError as e:
        raise AssertionError(f"\ntensor {tensor.numpy()} dtype {tensor.dtype} does not match target {target} with dtype {target_dtype}") from e


def _test_op(fxn, target_dtype: DType, target):
    _assert_eq(fxn(), target_dtype, target)



