import pytest
import itertools
import numpy as np
import mlir_graphblas.sparse_utils


def test_mlirsparsetensor_dtypes():
    pointer_dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]
    index_dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]
    value_dtypes = [np.int8, np.int16, np.int32, np.float32, np.float64]

    sparsity = np.array([True, True, True], dtype=np.bool8)
    sizes = np.array([10, 20, 30], dtype=np.uint64)
    for pointer_dtype, index_dtype, value_dtype in itertools.product(
        pointer_dtypes, index_dtypes, value_dtypes
    ):
        indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=index_dtype)
        values = np.array([1.2, 3.4], dtype=value_dtype)
        a = mlir_graphblas.sparse_utils.MLIRSparseTensor(
            indices, values, sizes, sparsity, pointer_type=pointer_dtype
        )
        assert a.pointer_dtype == pointer_dtype
        assert a.index_dtype == index_dtype
        assert a.value_dtype == value_dtype

        i, j, k = a.indices
        assert i.dtype == j.dtype == k.dtype == index_dtype
        np.testing.assert_array_equal(i, [0, 1])
        np.testing.assert_array_equal(j, [0, 1])
        np.testing.assert_array_equal(k, [0, 1])

        i, j, k = a.pointers
        assert i.dtype == j.dtype == k.dtype == pointer_dtype
        np.testing.assert_array_equal(i, [0, 2])
        np.testing.assert_array_equal(j, [0, 1, 2])
        np.testing.assert_array_equal(k, [0, 1, 2])

        assert a.values.dtype == value_dtype
        np.testing.assert_array_almost_equal(a.values, values)


def test_mlirsparsetensor_bad_dtypes():
    pointer_dtype = np.uint64
    sparsity = np.array([True, True, True], dtype=np.bool8)
    sizes = np.array([10, 20, 30], dtype=np.uint64)
    indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.uint64)
    values = np.array([1, 3], dtype=np.int32)

    with pytest.raises(TypeError, match="Bad dtype for values"):
        mlir_graphblas.sparse_utils.MLIRSparseTensor(
            indices,
            values.astype(np.int64),
            sizes,
            sparsity,
            pointer_type=pointer_dtype,
        )
    with pytest.raises(TypeError, match="Bad dtype for values"):
        mlir_graphblas.sparse_utils.MLIRSparseTensor(
            indices,
            values.astype(np.uint32),
            sizes,
            sparsity,
            pointer_type=pointer_dtype,
        )
    with pytest.raises(TypeError, match="Bad dtype for indices"):
        mlir_graphblas.sparse_utils.MLIRSparseTensor(
            indices.astype(np.int32),
            values,
            sizes,
            sparsity,
            pointer_type=pointer_dtype,
        )
