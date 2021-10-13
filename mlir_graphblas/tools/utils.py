import numpy as np
from mlir_graphblas.sparse_utils import MLIRSparseTensor

from typing import Sequence


def sparsify_array(
    input_array: np.ndarray, sparsity_values: Sequence[bool]
) -> MLIRSparseTensor:
    """Converts a numpy array into a MLIRSparseTensor."""

    indices = np.array(
        list(zip(*input_array.nonzero())),
        dtype=np.uint64,
    )
    values = np.array(
        [input_array[coordinate] for coordinate in map(tuple, indices)],
        dtype=input_array.dtype,
    )
    sizes = np.array(input_array.shape, dtype=np.uint64)
    sparsity = np.array(sparsity_values, dtype=np.bool8)

    sparse_tensor = MLIRSparseTensor(indices, values, sizes, sparsity)
    return sparse_tensor
