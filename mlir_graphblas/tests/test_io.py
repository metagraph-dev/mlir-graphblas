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

    # Was bad, but now okay
    tensor = mlir_graphblas.sparse_utils.MLIRSparseTensor(
        indices,
        values.astype(np.int64),
        sizes,
        sparsity,
        pointer_type=pointer_dtype,
    )
    assert tensor.value_dtype == np.int64
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


def test_mlirsparsetensor_swap():
    sparsity = np.array([True, True, True], dtype=np.bool8)

    sizes1 = np.array([10, 20, 30], dtype=np.uint64)
    indices1 = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.uint64)
    values1 = np.array([1.2, 3.4], dtype=np.float64)
    a1 = mlir_graphblas.sparse_utils.MLIRSparseTensor(
        indices1, values1, sizes1, sparsity
    )

    sizes2 = np.array([100, 200, 300], dtype=np.uint64)
    indices2 = np.array([[2, 2, 2], [3, 3, 3]], dtype=np.uint64)
    values2 = np.array([12, 34], dtype=np.float64)
    a2 = mlir_graphblas.sparse_utils.MLIRSparseTensor(
        indices2, values2, sizes2, sparsity
    )

    i, j, k = a1.indices
    np.testing.assert_array_equal(i, [0, 1])
    np.testing.assert_array_equal(j, [0, 1])
    np.testing.assert_array_equal(k, [0, 1])

    a1.swap_indices(a2)

    np.testing.assert_array_equal(i, [0, 1])
    np.testing.assert_array_equal(j, [0, 1])
    np.testing.assert_array_equal(k, [0, 1])
    i, j, k = a1.indices
    np.testing.assert_array_equal(i, [2, 3])
    np.testing.assert_array_equal(j, [2, 3])
    np.testing.assert_array_equal(k, [2, 3])

    i, j, k = a2.indices
    np.testing.assert_array_equal(i, [0, 1])
    np.testing.assert_array_equal(j, [0, 1])
    np.testing.assert_array_equal(k, [0, 1])

    i1, j1, k1 = a1.pointers
    a1.swap_pointers(a2)
    i2, j2, k2 = a2.pointers
    np.testing.assert_array_equal(i1, i2)
    np.testing.assert_array_equal(j1, j2)
    np.testing.assert_array_equal(k1, k2)

    v1 = a1.values
    a1.swap_values(a2)
    v2 = a2.values
    np.testing.assert_array_equal(v1, v2)


def test_mlirsparsetensor_empty():
    # 1d
    sparsity = np.array([True], dtype=np.bool8)
    sizes = np.array([10], dtype=np.uint64)
    a1 = mlir_graphblas.sparse_utils.empty_mlir_sparse_tensor_safe(sizes, sparsity)

    (i,) = a1.pointers
    np.testing.assert_array_equal(i, [0, 0])
    (i,) = a1.indices
    assert i.shape == (0,)
    assert a1.values.shape == (0,)

    a2 = mlir_graphblas.sparse_utils.empty_mlir_sparse_tensor_fast(sizes)
    for x, y in zip(a1.pointers, a2.pointers):
        np.testing.assert_array_equal(x, y)
    for x, y in zip(a1.indices, a2.indices):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(a1.values, a2.values)

    # 2d
    sparsity = np.array([True, True], dtype=np.bool8)
    sizes = np.array([10, 20], dtype=np.uint64)
    a1 = mlir_graphblas.sparse_utils.empty_mlir_sparse_tensor_safe(sizes, sparsity)

    i, j = a1.pointers
    np.testing.assert_array_equal(i, [0, 0])
    assert j.shape == (0,)
    i, j = a1.indices
    assert i.shape == (0,)
    assert j.shape == (0,)
    assert a1.values.shape == (0,)

    a2 = mlir_graphblas.sparse_utils.empty_mlir_sparse_tensor_fast(sizes)
    for x, y in zip(a1.pointers, a2.pointers):
        np.testing.assert_array_equal(x, y)
    for x, y in zip(a1.indices, a2.indices):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(a1.values, a2.values)

    # 3d
    sparsity = np.array([True, True, True], dtype=np.bool8)
    sizes = np.array([10, 20, 30], dtype=np.uint64)
    a1 = mlir_graphblas.sparse_utils.empty_mlir_sparse_tensor_safe(sizes, sparsity)

    i, j, k = a1.pointers
    np.testing.assert_array_equal(i, [0, 0])
    assert j.shape == (0,)
    assert k.shape == (0,)
    i, j, k = a1.indices
    assert i.shape == (0,)
    assert j.shape == (0,)
    assert k.shape == (0,)
    assert a1.values.shape == (0,)

    a2 = mlir_graphblas.sparse_utils.empty_mlir_sparse_tensor_fast(sizes)
    for x, y in zip(a1.pointers, a2.pointers):
        np.testing.assert_array_equal(x, y)
    for x, y in zip(a1.indices, a2.indices):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(a1.values, a2.values)


def test_mlirsparsetensor_dup():
    sparsity = np.array([True, True, True], dtype=np.bool8)
    sizes = np.array([10, 20, 30], dtype=np.uint64)
    indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.uint64)
    values = np.array([1.2, 3.4], dtype=np.float64)

    a1 = mlir_graphblas.sparse_utils.MLIRSparseTensor(indices, values, sizes, sparsity)
    a2 = a1.dup()

    pointers = p1, p2, p3 = a1.pointers
    indices = i1, i2, i3 = a1.indices
    values = a1.values
    for x, y in zip(pointers, a2.pointers):
        np.testing.assert_array_equal(x, y)
    for x, y in zip(indices, a2.indices):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(values, a2.values)

    a1.resize_values(1)
    a1.resize_index(0, 1)
    a1.resize_index(1, 1)
    a1.resize_index(2, 1)

    for x, y in zip(pointers, a2.pointers):
        np.testing.assert_array_equal(x, y)
    for x, y in zip(indices, a2.indices):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(values, a2.values)

    for x, y in zip(pointers, a1.pointers):
        np.testing.assert_array_equal(x, y)
    for x, y in zip(indices, a1.indices):
        np.testing.assert_array_equal(x[:1], y)
    np.testing.assert_array_equal(values[:1], a1.values)


def test_empty_like():
    sparsity = np.array([True, True, True], dtype=np.bool8)
    sizes = np.array([10, 20, 30], dtype=np.uint64)
    indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.uint64)
    values = np.array([1.2, 3.4], dtype=np.float64)

    a = mlir_graphblas.sparse_utils.MLIRSparseTensor(indices, values, sizes, sparsity)
    b = a.empty_like()
    assert a.ndim == b.ndim
    assert a.shape == b.shape
    # Pointers are the same size as original, but containing all zeros
    for val, compare_val in zip(b.pointers, a.pointers):
        assert val.size == compare_val.size
        assert (val == 0).all()
    # Indices and Values are empty
    for val in b.indices:
        assert val.size == 0
    assert b.values.size == 0


def test_from_raw_pointer():
    sparsity = np.array([True, True, True], dtype=np.bool8)
    sizes = np.array([10, 20, 30], dtype=np.uint64)
    indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.uint64)
    values = np.array([1.2, 3.4], dtype=np.float64)

    a1 = mlir_graphblas.sparse_utils.MLIRSparseTensor(indices, values, sizes, sparsity)
    a2 = mlir_graphblas.sparse_utils.MLIRSparseTensor.from_raw_pointer(
        a1.data, a1.pointer_dtype, a1.index_dtype, a1.value_dtype
    )
    pointers = p1, p2, p3 = a1.pointers
    indices = i1, i2, i3 = a1.indices
    values = a1.values
    for x, y in zip(pointers, a2.pointers):
        np.testing.assert_array_equal(x, y)
    for x, y in zip(indices, a2.indices):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(values, a2.values)
    assert a1.pointer_dtype == a2.pointer_dtype
    assert a1.index_dtype == a2.index_dtype
    assert a1.value_dtype == a2.value_dtype

    a2.data = 0  # So we don't free the data twice


def test_views_have_parent():
    sparsity = np.array([True, True, True], dtype=np.bool8)
    sizes = np.array([10, 20, 30], dtype=np.uint64)
    indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.uint64)
    values = np.array([1.2, 3.4], dtype=np.float64)

    a1 = mlir_graphblas.sparse_utils.MLIRSparseTensor(indices, values, sizes, sparsity)
    v = a1.values
    assert v.base is a1
    del a1
    np.testing.assert_array_almost_equal(v, values)
