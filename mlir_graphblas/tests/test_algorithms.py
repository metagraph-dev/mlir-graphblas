import numpy as np
from mlir_graphblas.sparse_utils import MLIRSparseTensor
import mlir_graphblas.algorithms as mlalgo


def test_triangle_count():
    # 0 - 1    5 - 6
    # | X |    | /
    # 3 - 4 -- 2 - 7
    # fmt: off
    indices = np.array(
        [[0, 1], [0, 3], [0, 4],
         [1, 0], [1, 3], [1, 4],
         [2, 4], [2, 5], [2, 6], [2, 7],
         [3, 0], [3, 1], [3, 4],
         [4, 0], [4, 1], [4, 2], [4, 3],
         [5, 2], [5, 6],
         [6, 2], [6, 5],
         [7, 2]],
        dtype=np.uint64,
    )
    values = np.array([
        100, 200, 300, 100, 400, 500, 99, 50, 55, 75, 200,
        400, 600, 300, 500, 99, 600, 50, 60, 55, 60, 75],
        dtype=np.float64,
    )
    # fmt: on
    sizes = np.array([8, 8], dtype=np.uint64)
    sparsity = np.array([False, True], dtype=np.bool8)
    a = MLIRSparseTensor(indices, values, sizes, sparsity)

    num_triangles = mlalgo.triangle_count(a)
    assert num_triangles == 5, num_triangles

    num_triangles = mlalgo.triangle_count_combined(a)
    assert num_triangles == 5, num_triangles


def test_sssp():
    # This must be in sorted-for-CSR format. Random order breaks the constructor in strange ways.
    # fmt: off
    indices = np.array(
        [[0, 1], [0, 3],
         [1, 4], [1, 6],
         [2, 5],
         [3, 0], [3, 2],
         [4, 5],
         [5, 1],
         [6, 2], [6, 3], [6, 4]],
        dtype=np.uint64,
    )
    # fmt: on
    values = np.array([2, 3, 8, 4, 1, 3, 3, 7, 1, 5, 7, 3], dtype=np.float64)
    sizes = np.array([7, 7], dtype=np.uint64)
    sparsity = np.array([False, True], dtype=np.bool8)
    m = MLIRSparseTensor(indices, values, sizes, sparsity)

    indices = np.array([[1]], dtype=np.uint64)
    values = np.array([0], dtype=np.float64)
    sizes = np.array([7], dtype=np.uint64)
    sparsity = np.array([True], dtype=np.bool8)
    v = MLIRSparseTensor(indices, values, sizes, sparsity)

    # Compute SSSP from node #1 -- correct answer is [14, 0, 9, 11, 7, 10, 4]
    w = mlalgo.sssp(m, v)

    assert (w.indices[0] == np.arange(7)).all()
    assert (w.values == [14, 0, 9, 11, 7, 10, 4]).all()


def test_mssp():
    # This must be in sorted-for-CSR format. Random order breaks the constructor in strange ways.
    # fmt: off
    indices = np.array(
        [[0, 1], [0, 3],
         [1, 4], [1, 6],
         [2, 5],
         [3, 0], [3, 2],
         [4, 5],
         [5, 1],
         [6, 2], [6, 3], [6, 4]],
        dtype=np.uint64,
    )
    # fmt: on
    values = np.array([2, 3, 8, 4, 1, 3, 3, 7, 1, 5, 7, 3], dtype=np.float64)
    sizes = np.array([7, 7], dtype=np.uint64)
    sparsity = np.array([False, True], dtype=np.bool8)
    m = MLIRSparseTensor(indices, values, sizes, sparsity)

    indices = np.array([[0, 1], [1, 3]], dtype=np.uint64)
    values = np.array([0, 0], dtype=np.float64)
    sizes = np.array([2, 7], dtype=np.uint64)
    sparsity = np.array([False, True], dtype=np.bool8)
    v = MLIRSparseTensor(indices, values, sizes, sparsity)

    # Compute MSSP
    # correct answer from node #1 -- [14, 0, 9, 11, 7, 10, 4]
    # correct answer from node #3 -- [3,  5,  3,  0, 12,  4,  9]
    w = mlalgo.mssp(m, v)

    assert (w.indices[1] == [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6]).all()
    assert (w.values == [14, 0, 9, 11, 7, 10, 4, 3, 5, 3, 0, 12, 4, 9]).all()


def test_vertex_nomination():
    # fmt: off
    indices = np.array(
        [[0, 1], [0, 3],
         [1, 4], [1, 6],
         [2, 5],
         [3, 0], [3, 2],
         [4, 5],
         [5, 1],
         [6, 2], [6, 3], [6, 4]],
        dtype=np.uint64,
    )
    # fmt: on
    values = np.array([2, 3, 8, 4, 1, 3, 3, 7, 1, 5, 7, 3], dtype=np.float64)
    sizes = np.array([7, 7], dtype=np.uint64)
    sparsity = np.array([False, True], dtype=np.bool8)
    m = MLIRSparseTensor(indices, values, sizes, sparsity)

    indices = np.array([[6]], dtype=np.uint64)
    values = np.array([0], dtype=np.float64)
    sizes = np.array([7], dtype=np.uint64)
    sparsity = np.array([True], dtype=np.bool8)
    v = MLIRSparseTensor(indices, values, sizes, sparsity)

    # Compute Vertex Nomination
    # correct answer for node #6 is node #4
    w = mlalgo.vertex_nomination(m, v)
    assert w == 4

    # correct answer for nodes #0,1,5 is node #3
    indices = np.array([[0], [1], [5]], dtype=np.uint64)
    values = np.array([0, 0, 0], dtype=np.float64)
    v2 = MLIRSparseTensor(indices, values, sizes, sparsity)
    w2 = mlalgo.vertex_nomination(m, v2)
    assert w2 == 3


def test_pagerank():
    # fmt: off
    indices = np.array(
        [[0, 1], [0, 2], [1, 3], [2, 3], [2, 4], [3, 4], [4, 0]],
        dtype=np.uint64,
    )
    values = np.array([1.1, 9.8, 4.2, 7.1, 0.2, 6.9, 2.2], dtype=np.float64)
    sizes = np.array([5, 5], dtype=np.uint64)
    sparsity = np.array([False, True], dtype=np.bool8)
    m = MLIRSparseTensor(indices, values, sizes, sparsity)

    expected = np.array([0.2541917746, 0.1380315018, 0.1380315018, 0.2059901768, 0.2637550447])

    pr = mlalgo.pagerank(m)
    assert np.abs(pr.values - expected).all(), pr.values
