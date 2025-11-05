"""
Simplified weight saving/loading for teenygrad.
Adapted from tinygrad, removing bf16 and other complex features to keep it educational.
"""
import json
import pathlib
import numpy as np
from typing import Any, Dict, Union
from collections import OrderedDict

from teenygrad.tensor import Tensor
from teenygrad.helpers import dtypes, round_up


safe_dtypes = {
    "BOOL": dtypes.bool,
    "I8": dtypes.int8,
    "U8": dtypes.uint8,
    "I16": dtypes.int16,
    "U16": dtypes.uint16,
    "I32": dtypes.int32,
    "U32": dtypes.uint32,
    "I64": dtypes.int64,
    "U64": dtypes.uint64,
    "F16": dtypes.float16,
    "F32": dtypes.float32,
    "F64": dtypes.float64,
}
inverse_safe_dtypes = {v: k for k, v in safe_dtypes.items()}


def safe_load(fn: Union[str, pathlib.Path]) -> Dict[str, Tensor]:
    """
    Loads a .safetensor file, returning the state_dict.

    Example:
        state_dict = safe_load("model.safetensor")

    Args:
        fn: Path to the .safetensor file

    Returns:
        Dictionary mapping tensor names to Tensor objects
    """
    with open(fn, 'rb') as f:
        data = f.read()

    header_size = int.from_bytes(data[0:8], "little")
    data_start = header_size + 8

    metadata = json.loads(data[8:data_start].decode('utf-8'))

    result = {}
    for k, v in metadata.items():
        if k == "__metadata__":
            continue

        dtype_str = v['dtype']
        target_dtype = safe_dtypes[dtype_str]
        shape = tuple(v['shape'])
        start_offset = v['data_offsets'][0]
        end_offset = v['data_offsets'][1]

        tensor_bytes = data[data_start + start_offset:data_start + end_offset]

        if target_dtype.np is not None:
            np_array = np.frombuffer(tensor_bytes, dtype=target_dtype.np).reshape(shape)
            result[k] = Tensor(np_array, dtype=target_dtype)
        else:
            raise ValueError(f"Dtype {target_dtype} has no numpy equivalent")

    return result


def safe_save(tensors: Dict[str, Tensor], fn: Union[str, pathlib.Path], metadata: Dict[str, Any] = None):
    """
    Saves a state_dict to disk in .safetensor format with optional metadata.

    Example:
        t = Tensor([1, 2, 3])
        safe_save({'t': t}, "model.safetensor")

    Args:
        tensors: Dictionary mapping names to Tensor objects
        fn: Path where the file should be saved
        metadata: Optional metadata dictionary
    """
    headers = {}
    if metadata:
        headers['__metadata__'] = metadata

    offset = 0
    for k, v in tensors.items():
        if v.dtype not in inverse_safe_dtypes:
            raise ValueError(f"Unsupported dtype {v.dtype} for tensor '{k}'. Teenygrad only supports standard types (no bf16).")

        headers[k] = {
            'dtype': inverse_safe_dtypes[v.dtype],
            'shape': list(v.shape),
            'data_offsets': [offset, offset + v.nbytes()]
        }
        offset += v.nbytes()

    header_json = json.dumps(headers, separators=(',', ':'))
    header_json += " " * (round_up(len(header_json), 8) - len(header_json))
    header_bytes = header_json.encode('utf-8')

    fn = pathlib.Path(fn)
    fn.unlink(missing_ok=True)

    with open(fn, 'wb') as f:
        f.write(len(header_bytes).to_bytes(8, 'little'))
        f.write(header_bytes)
        for k in tensors.keys():
            tensor_data = tensors[k].numpy().tobytes()
            f.write(tensor_data)


def get_state_dict(obj, prefix: str = '', tensor_type=Tensor) -> Dict[str, Tensor]:
    """
    Returns a state_dict of the object, with optional prefix.
    Recursively extracts all Tensor objects from the model.

    Example:
        class Net:
            def __init__(self):
                self.weight = Tensor.randn(10, 10)
                self.bias = Tensor.zeros(10)

        net = Net()
        state_dict = get_state_dict(net)
        # Returns: {'weight': <Tensor>, 'bias': <Tensor>}

    Args:
        obj: Object to extract tensors from (model, dict, list, etc.)
        prefix: Prefix for tensor names (used in recursion)
        tensor_type: Type to check for (default: Tensor)

    Returns:
        Dictionary mapping tensor names to Tensor objects
    """
    if isinstance(obj, tensor_type):
        return {prefix.strip('.'): obj}

    if hasattr(obj, '_asdict'):  # namedtuple
        return get_state_dict(obj._asdict(), prefix, tensor_type)

    if isinstance(obj, OrderedDict):
        return get_state_dict(dict(obj), prefix, tensor_type)

    if hasattr(obj, '__dict__'):
        return get_state_dict(obj.__dict__, prefix, tensor_type)

    state_dict = {}
    if isinstance(obj, (list, tuple)):
        for i, x in enumerate(obj):
            state_dict.update(get_state_dict(x, f"{prefix}{str(i)}.", tensor_type))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            state_dict.update(get_state_dict(v, f"{prefix}{str(k)}.", tensor_type))

    return state_dict


def get_parameters(obj) -> list[Tensor]:
    """
    Returns a list of all Tensor parameters in the object.

    Example:
        class Net:
            def __init__(self):
                self.weight = Tensor.randn(10, 10)
                self.bias = Tensor.zeros(10)

        net = Net()
        params = get_parameters(net)
        # Returns: [<Tensor weight>, <Tensor bias>]

    Args:
        obj: Object to extract parameters from

    Returns:
        List of Tensor objects
    """
    return list(get_state_dict(obj).values())


def load_state_dict(model, state_dict: Dict[str, Tensor], strict: bool = True):
    """
    Loads a state_dict into a model.

    Example:
        class Net:
            def __init__(self):
                self.weight = Tensor.randn(10, 10)
                self.bias = Tensor.zeros(10)

        net = Net()
        state_dict = get_state_dict(net)

        # Later, load the state_dict back
        new_net = Net()
        load_state_dict(new_net, state_dict)

    Args:
        model: Model object to load weights into
        state_dict: Dictionary mapping tensor names to Tensor objects
        strict: If True, raises error if shapes don't match or keys are missing
    """
    model_state_dict = get_state_dict(model)

    unused_keys = set(state_dict.keys()) - set(model_state_dict.keys())
    if unused_keys:
        print(f"Warning: unused weights in state_dict: {sorted(unused_keys)}")

    for k, v in model_state_dict.items():
        if k not in state_dict:
            if strict:
                raise KeyError(f"Missing key in state_dict: {k}")
            else:
                print(f"Warning: not loading {k} (missing in state_dict)")
                continue

        if v.shape != state_dict[k].shape:
            raise ValueError(
                f"Shape mismatch in layer '{k}': "
                f"Expected shape {v.shape}, but found {state_dict[k].shape} in state_dict."
            )

        v.assign(state_dict[k])
