from multiprocessing import RawArray

import numpy as np
from numpy.typing import NDArray


def to_shared(arr: NDArray) -> NDArray:
    cdtype = np.ctypeslib.as_ctypes_type(arr.dtype)
    buffer = RawArray(cdtype, int(arr.dtype.itemsize * np.prod(arr.shape)))
    shared_arr = np.frombuffer(
        buffer, dtype=arr.dtype, count=np.prod(arr.shape)
    ).reshape(arr.shape)
    shared_arr[:] = arr[:]
    return shared_arr
