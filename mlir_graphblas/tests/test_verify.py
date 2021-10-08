import itertools

import grblas as gb
import numpy as np
import pytest

from mlir_graphblas.sparse_utils import MLIRSparseTensor


def issorted(arr):
    if arr.size > 1:
        prev = arr[0]
        for i in range(1, arr.size):
            cur = arr[i]
            if cur == prev:
                continue
            elif cur < prev:
                return False
            else:
                prev = cur
    return True


def isincreasing(arr):
    if arr.size > 1:
        prev = arr[0]
        for i in range(1, arr.size):
            cur = arr[i]
            if cur <= prev:
                return False
            else:
                prev = cur
    return True


def verify(mt):
    # Pure Python version to validate the correctness of a sparse tensor.
    # This captures the behavior we want to have in the C function.
    rev = mt.rev
    assert len(rev) == mt.ndim
    assert rev.min() == 0
    assert rev.max() == mt.ndim - 1
    check_rev = np.zeros(mt.ndim)
    check_rev[rev] = 1
    assert check_rev.all()

    was_dense = True
    cum_size = 1
    last_idx = None
    prev_ptr = None
    prev_idx = None
    for dim in range(mt.ndim):
        ptr = mt.get_pointers(dim)
        idx = mt.get_indices(dim)
        size = mt.get_dimsize(dim)
        assert size > 0
        cum_size *= size
        if len(ptr) == 0:
            assert len(idx) == 0
        else:
            if dim == 0:
                assert len(ptr) >= 2
            else:
                if was_dense:
                    assert len(ptr) == cum_size // size + 1
                assert len(ptr) >= 1
            assert len(ptr) <= cum_size + 1
            assert len(idx) <= cum_size
            assert issorted(ptr)
            assert ptr[0] == 0
            assert ptr[-1] == len(idx)
            assert (idx < size).all()
            if prev_ptr is not None:
                # These checks are probably redundant (will they work for higher rank?)
                assert len(prev_idx) < len(ptr)
                assert len(prev_idx) <= len(idx)
                assert len(prev_ptr) <= len(ptr) + 1
                assert len(prev_ptr) <= len(idx) + 2
            start = ptr[0]
            for end in ptr[1:]:
                view = idx[start:end]
                assert isincreasing(view)
                start = end
            last_idx = idx
            was_dense = False
            prev_ptr = ptr
            prev_idx = idx
    if last_idx is not None:
        assert len(last_idx) == len(mt.values)
    else:
        # fully dense
        len(mt.values) == cum_size


@pytest.mark.parametrize("nrows", range(1, 4))
@pytest.mark.parametrize("ncols", range(1, 4))
def test_verify(nrows, ncols):
    for indices in itertools.chain.from_iterable(
        itertools.combinations(list(range(nrows * ncols)), n)
        for n in range(nrows * ncols + 1)
    ):
        rows = [x // ncols for x in indices]
        cols = [x % ncols for x in indices]
        vals = np.arange(len(indices))
        M = gb.Matrix.from_values(rows, cols, vals, nrows=nrows, ncols=ncols)
        M.wait()

        # CSR
        d = M.ss.export("csr", sort=True)
        r, c, v = M.to_values()
        mt = MLIRSparseTensor(
            np.stack([r, c]).T.copy(),
            v,
            np.array(M.shape, dtype=np.uint64),
            np.array([False, True], dtype=np.bool8),
        )
        verify(mt)
        np.testing.assert_array_equal(mt.get_pointers(0), [])
        np.testing.assert_array_equal(mt.get_pointers(1), d["indptr"])
        np.testing.assert_array_equal(mt.get_indices(0), [])
        np.testing.assert_array_equal(mt.get_indices(1), d["col_indices"])
        np.testing.assert_array_equal(mt.values, d["values"])
        assert mt.shape == M.shape
        assert mt.sizes == M.shape

        # HyperCSR
        d = M.ss.export("hypercsr", sort=True)
        r, c, v = M.to_values()
        mt = MLIRSparseTensor(
            np.stack([r, c]).T.copy(),
            v,
            np.array(M.shape, dtype=np.uint64),
            np.array([True, True], dtype=np.bool8),
        )
        verify(mt)
        np.testing.assert_array_equal(mt.get_pointers(0), [0, len(d["rows"])])
        np.testing.assert_array_equal(mt.get_pointers(1), d["indptr"])
        np.testing.assert_array_equal(mt.get_indices(0), d["rows"])
        np.testing.assert_array_equal(mt.get_indices(1), d["col_indices"])
        np.testing.assert_array_equal(mt.values, d["values"])
        assert mt.shape == M.shape
        assert mt.sizes == M.shape

        # CSC
        d = M.ss.export("csc", sort=True)
        M2 = M.T.new()
        M2.wait()
        c, r, v = M2.to_values()
        mt = MLIRSparseTensor(
            np.stack([c, r]).T.copy(),
            v,
            np.array(M.shape[::-1], dtype=np.uint64),
            np.array([False, True], dtype=np.bool8),
            np.array([1, 0], dtype=np.uint64),
        )
        verify(mt)
        np.testing.assert_array_equal(mt.get_pointers(0), [])
        np.testing.assert_array_equal(mt.get_pointers(1), d["indptr"])
        np.testing.assert_array_equal(mt.get_indices(0), [])
        np.testing.assert_array_equal(mt.get_indices(1), d["row_indices"])
        np.testing.assert_array_equal(mt.values, d["values"])
        assert mt.shape == M.shape
        assert mt.sizes[::-1] == M.shape

        # HyperCSC
        d = M.ss.export("hypercsc", sort=True)
        M2 = M.T.new()
        M2.wait()
        c, r, v = M2.to_values()
        mt = MLIRSparseTensor(
            np.stack([c, r]).T.copy(),
            v,
            np.array(M.shape[::-1], dtype=np.uint64),
            np.array([True, True], dtype=np.bool8),
            np.array([1, 0], dtype=np.uint64),
        )
        verify(mt)
        np.testing.assert_array_equal(mt.get_pointers(0), [0, len(d["cols"])])
        np.testing.assert_array_equal(mt.get_pointers(1), d["indptr"])
        np.testing.assert_array_equal(mt.get_indices(0), d["cols"])
        np.testing.assert_array_equal(mt.get_indices(1), d["row_indices"])
        np.testing.assert_array_equal(mt.values, d["values"])
        assert mt.shape == M.shape
        assert mt.sizes[::-1] == M.shape

        if vals.size == nrows * ncols:
            # FullR
            d = M.ss.export("fullr", sort=True)
            r, c, v = M.to_values()
            mt = MLIRSparseTensor(
                np.stack([r, c]).T.copy(),
                v,
                np.array(M.shape, dtype=np.uint64),
                np.array([False, False], dtype=np.bool8),
            )
            verify(mt)
            np.testing.assert_array_equal(mt.values, d["values"].ravel())
            assert mt.shape == M.shape
            assert mt.sizes == M.shape

            # FullC
            d = M.ss.export("fullc", sort=True)
            M2 = M.T.new()
            M2.wait()
            c, r, v = M2.to_values()
            mt = MLIRSparseTensor(
                np.stack([c, r]).T.copy(),
                v,
                np.array(M.shape[::-1], dtype=np.uint64),
                np.array([False, False], dtype=np.bool8),
                np.array([1, 0], dtype=np.uint64),
            )
            verify(mt)
            np.testing.assert_array_equal(mt.values, d["values"].T.ravel())
            assert mt.shape == M.shape
            assert mt.sizes[::-1] == M.shape
