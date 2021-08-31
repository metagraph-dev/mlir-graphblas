import numpy as np
import scipy.sparse as ss
from mlir_graphblas.sparse_utils import MLIRSparseTensor

from typing import Sequence

GRAPHBLAS_PASSES = [
    "--graphblas-optimize",
    "--graphblas-lower",
    "--sparsification",
    "--sparse-tensor-conversion",
    "--linalg-bufferize",
    "--func-bufferize",
    "--tensor-bufferize",
    "--tensor-constant-bufferize",
    "--finalizing-bufferize",
    "--convert-linalg-to-loops",
    "--convert-scf-to-std",
    "--convert-memref-to-llvm",
    "--convert-std-to-llvm",
]

STANDARD_PASSES = [
    "--sparsification",
    "--sparse-tensor-conversion",
    "--linalg-bufferize",
    "--func-bufferize",
    "--tensor-bufferize",
    "--tensor-constant-bufferize",
    "--finalizing-bufferize",
    "--convert-linalg-to-loops",
    "--convert-scf-to-std",
    "--convert-memref-to-llvm",
    "--convert-std-to-llvm",
]

MLIR_TYPE_TO_NP_TYPE = {
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    # 'f16': np.float16, # 16-bit floats don't seem to be supported in ctypes
    "f32": np.float32,
    "f64": np.float64,
}


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


def densify_vector(sparse_vec: MLIRSparseTensor) -> np.ndarray:
    (vec_length,) = sparse_vec.shape
    dense_vec = np.zeros(vec_length, dtype=sparse_vec.values.dtype)
    dense_vec[sparse_vec.indices] = sparse_vec.values
    return dense_vec


def densify_csr(tensor_csr: MLIRSparseTensor) -> np.ndarray:
    ss_tensor_csr = ss.csr_matrix(
        (tensor_csr.values, tensor_csr.indices[1], tensor_csr.pointers[1]),
        shape=(tensor_csr.get_dimsize(0), tensor_csr.get_dimsize(1)),
    )
    return ss_tensor_csr.toarray()


def densify_csc(tensor_csc: MLIRSparseTensor) -> np.ndarray:
    ss_tensor_csc = ss.csc_matrix(
        (tensor_csc.values, tensor_csc.indices[1], tensor_csc.pointers[1]),
        shape=(tensor_csc.get_dimsize(0), tensor_csc.get_dimsize(1)),
    )
    return ss_tensor_csc.toarray()
