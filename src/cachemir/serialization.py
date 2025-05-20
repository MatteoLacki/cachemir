import numpy as np
import pandas as pd

common_types = [
    str,
    int,
    float,
    bool,
    list,
    dict,
    tuple,
    set,
    type(None),
    bytes,
    complex,
    frozenset,
    bytearray,
    memoryview,
    range,
]

# Map type name to actual type
name_to_type = {typ.__name__: typ for typ in common_types}

# Reverse map: type to name
type_to_name = {v: k for k, v in name_to_type.items()}


numpy_to_python_type = {
    np.bool_: bool,
    np.int8: int,
    np.int16: int,
    np.int32: int,
    np.int64: int,
    np.uint8: int,
    np.uint16: int,
    np.uint32: int,
    np.uint64: int,
    np.float16: float,
    np.float32: float,
    np.float64: float,
    np.complex64: complex,
    np.complex128: complex,
    np.str_: str,
    np.bytes_: bytes,
    np.object_: object,
}


def derive_input_types(inputs_df: pd.DataFrame) -> dict[str, type]:
    return {
        col: numpy_to_python_type.get(type(val), type(val))
        for val, col in zip(inputs_df.iloc[0], inputs_df.columns)
    }


def enforce_types(
    inputs: tuple[int | float | str, ...],
    types: tuple[type, ...],
) -> tuple[int | float | str, ...]:
    return tuple(_type(_input) for _type, _input in zip(types, inputs))
