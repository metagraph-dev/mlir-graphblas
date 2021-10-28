import numpy as np
import pytest
from mlir_graphblas.sparse_utils import MLIRSparseTensor
import mlir_graphblas.algorithms as mlalgo
from mlir_graphblas.tools.utils import sparsify_array
from mlir_graphblas.mlir_builder import GRAPHBLAS_OPENMP_PASSES


@pytest.mark.parametrize("special_passes", [None, GRAPHBLAS_OPENMP_PASSES])
def test_triangle_count(special_passes):
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
    assert a.verify()

    num_triangles = mlalgo.triangle_count(a, compile_with_passes=special_passes)
    assert num_triangles == 5, num_triangles


@pytest.mark.parametrize("special_passes", [None, GRAPHBLAS_OPENMP_PASSES])
def test_sssp(special_passes):
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
    assert m.verify()

    indices = np.array([[1]], dtype=np.uint64)
    values = np.array([0], dtype=np.float64)
    sizes = np.array([7], dtype=np.uint64)
    sparsity = np.array([True], dtype=np.bool8)
    v = MLIRSparseTensor(indices, values, sizes, sparsity)
    assert v.verify()

    # Compute SSSP from node #1 -- correct answer is [14, 0, 9, 11, 7, 10, 4]
    w = mlalgo.sssp(m, v, compile_with_passes=special_passes)

    assert (w.indices[0] == np.arange(7)).all()
    assert (w.values == [14, 0, 9, 11, 7, 10, 4]).all()


@pytest.mark.parametrize("special_passes", [None, GRAPHBLAS_OPENMP_PASSES])
def test_mssp(special_passes):
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
    assert m.verify()

    indices = np.array([[0, 1], [1, 3]], dtype=np.uint64)
    values = np.array([0, 0], dtype=np.float64)
    sizes = np.array([2, 7], dtype=np.uint64)
    sparsity = np.array([False, True], dtype=np.bool8)
    v = MLIRSparseTensor(indices, values, sizes, sparsity)
    assert v.verify()

    # Compute MSSP
    # correct answer from node #1 -- [14, 0, 9, 11, 7, 10, 4]
    # correct answer from node #3 -- [3,  5,  3,  0, 12,  4,  9]
    w = mlalgo.mssp(m, v, compile_with_passes=special_passes)

    assert (w.indices[1] == [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6]).all()
    assert (w.values == [14, 0, 9, 11, 7, 10, 4, 3, 5, 3, 0, 12, 4, 9]).all()


@pytest.mark.parametrize("special_passes", [None, GRAPHBLAS_OPENMP_PASSES])
def test_bipartite_project_and_filter(special_passes):
    # Test Results
    r"""
    0  1  2  3
    |\ | /|\ |\
    | \|/ | \| \
    5  6  7  8  9
    """
    # fmt: off
    dense_input_tensor = np.array(
        [  #   0   1   2   3
            [  1,  0,  0,  0], # 5
            [ -9,  1,  1,  0], # 6
            [  0,  0,  1,  0], # 7
            [  0,  0,  1,  1], # 8
            [  0,  0,  0, -9], # 9
        ],
        dtype=np.float64,
    )
    # fmt: on
    input_tensor = sparsify_array(dense_input_tensor, [False, True])
    assert input_tensor.verify()

    # Test row projection
    result = mlalgo.bipartite_project_and_filter(input_tensor)
    assert result.verify()
    dense_result = result.toarray()

    expected_dense_result = dense_input_tensor @ dense_input_tensor.T
    expected_dense_result[expected_dense_result < 0] = 0

    assert np.all(dense_result == expected_dense_result)

    # Test column projection
    result2 = mlalgo.bipartite_project_and_filter(
        input_tensor, "column", cutoff=1.0, compile_with_passes=special_passes
    )
    assert result2.verify()
    dense_result2 = result2.toarray()

    expected_dense_result2 = dense_input_tensor.T @ dense_input_tensor
    expected_dense_result2[expected_dense_result2 < 1] = 0

    assert np.all(dense_result2 == expected_dense_result2)


@pytest.mark.parametrize("special_passes", [None, GRAPHBLAS_OPENMP_PASSES])
def test_vertex_nomination(special_passes):
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
    assert m.verify()

    indices = np.array([[6]], dtype=np.uint64)
    values = np.array([0], dtype=np.float64)
    sizes = np.array([7], dtype=np.uint64)
    sparsity = np.array([True], dtype=np.bool8)
    v = MLIRSparseTensor(indices, values, sizes, sparsity)
    assert v.verify()

    # Compute Vertex Nomination
    # correct answer for node #6 is node #4
    w = mlalgo.vertex_nomination(m, v, compile_with_passes=special_passes)
    assert w == 4

    # correct answer for nodes #0,1,5 is node #3
    indices = np.array([[0], [1], [5]], dtype=np.uint64)
    values = np.array([0, 0, 0], dtype=np.float64)
    v2 = MLIRSparseTensor(indices, values, sizes, sparsity)
    assert v2.verify()
    w2 = mlalgo.vertex_nomination(m, v2, compile_with_passes=special_passes)
    assert w2 == 3


@pytest.mark.parametrize("special_passes", [None, GRAPHBLAS_OPENMP_PASSES])
def test_scan_statistics(special_passes):
    # Test Results
    dense_input_tensor = np.array(
        [
            [0, 1, 0, 1, 1, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    input_tensor = sparsify_array(dense_input_tensor, [False, True])
    assert input_tensor.verify()

    result = mlalgo.scan_statistics(input_tensor, compile_with_passes=special_passes)

    # valid results are in {0, 1, 3, 4}, but we choose the lowest index
    expected_result = 0

    assert result == expected_result


@pytest.mark.parametrize("special_passes", [None, GRAPHBLAS_OPENMP_PASSES])
def test_pagerank(special_passes):
    # fmt: off
    indices = np.array(
        [[0, 1], [0, 2], [1, 3], [2, 3], [2, 4], [3, 4], [4, 0]],
        dtype=np.uint64,
    )
    # fmt: on
    values = np.array([1.1, 9.8, 4.2, 7.1, 0.2, 6.9, 2.2], dtype=np.float64)
    sizes = np.array([5, 5], dtype=np.uint64)
    sparsity = np.array([False, True], dtype=np.bool8)
    m = MLIRSparseTensor(indices, values, sizes, sparsity)
    assert m.verify()

    expected = np.array(
        [0.2541917746, 0.1380315018, 0.1380315018, 0.2059901768, 0.2637550447]
    )

    # Test success
    pr, niters = mlalgo.pagerank(m, tol=1e-7)
    assert np.abs(pr.values - expected).sum() < 1e-5, pr.values

    # Test maxiter reached, failed to converge
    pr, niters = mlalgo.pagerank(
        m, tol=1e-7, maxiter=6, compile_with_passes=special_passes
    )
    assert niters == 6
    assert (
        np.abs(pr.values - expected).sum() > 1e-5
    ), "Unexpectedly converged in 6 iterations"


# DO NOT RUN THIS ALGORITHM WITH OPENMP UNTIL WE HAVE THREAD SAFE RNG
def test_graph_search():
    # fmt: off
    indices = np.array(
        [[0, 1], [0, 2], [1, 0], [1, 3], [2, 0], [2, 4], [3, 2], [4, 4]],
        dtype=np.uint64,
    )
    values = np.array([100, 200, 300, 400, 175, 222, 333, 200], dtype=np.float64)
    # fmt: on
    sizes = np.array([5, 5], dtype=np.uint64)
    sparsity = np.array([False, True], dtype=np.bool8)
    graph = MLIRSparseTensor(indices, values, sizes, sparsity)
    assert graph.verify()

    # Random Uniform (no seed, so truly random)
    count = mlalgo.graph_search(graph, 3, [2, 4], "random")

    # Check for one of the possible solutions:
    # [0, 1, 4] or [0, 1, 3, 4] or [0, 2, 4] or [0, 2, 4] or [4]
    # [2, 1, 3]    [1, 1, 1, 3]    [2, 1, 3]    [1, 1, 4]    [6]
    for idx, vals in [
        ([0, 1, 4], [2, 1, 3]),
        ([0, 1, 3, 4], [1, 1, 1, 3]),
        ([0, 2, 4], [2, 1, 3]),
        ([0, 2, 4], [1, 1, 4]),
        ([4], [6]),
    ]:
        if len(count.indices[0]) == len(idx):
            if (count.indices[0] == idx).all() and (count.values == vals).all():
                break
    else:
        assert False, f"Invalid solution: idx={count.indices[0]}, vals={count.values}"

    # Random weighted
    count = mlalgo.graph_search(graph, 5, [0, 2], "random_weighted", rand_seed=14)
    assert (count.indices[0] == [0, 2, 4]).all()
    assert (count.values == [2, 3, 5]).all()

    # argmin
    count = mlalgo.graph_search(graph, 3, [0, 3], "argmin")
    assert (count.indices[0] == [0, 1, 2]).all()
    assert (count.values == [2, 3, 1]).all()

    # argmax
    count = mlalgo.graph_search(graph, 3, [0, 1], "argmax")
    assert (count.indices[0] == [2, 3, 4]).all()
    assert (count.values == [2, 1, 3]).all()


# DO NOT RUN THIS ALGORITHM WITH OPENMP UNTIL WE HAVE THREAD SAFE RNG
def test_random_walk():
    # fmt: off
    indices = np.array(
        [[0, 1], [0, 2], [1, 0], [1, 3], [2, 0], [2, 4], [3, 2]],
        dtype=np.uint64,
    )
    values = np.array([100, 200, 300, 400, 175, 222, 333], dtype=np.float64)
    # fmt: on
    sizes = np.array([5, 5], dtype=np.uint64)
    sparsity = np.array([False, True], dtype=np.bool8)
    graph = MLIRSparseTensor(indices, values, sizes, sparsity)
    assert graph.verify()

    paths = mlalgo.random_walk(graph, 20, [0, 1, 2, 3, 4, 4, 3, 2, 1, 0])

    # Perform validation for each row to verify correctness
    valid_steps = {}
    for start, end in indices:
        valid_steps.setdefault(start, set()).add(end)
    pointers = paths.pointers[-1]
    nodes = paths.values
    for irow, (istart, iend) in enumerate(zip(pointers[:-1], pointers[1:])):
        assert istart != iend, f"Initial node missing for row {irow}"
        for jstart, jend in zip(
            nodes[istart : int(iend - 1)], nodes[int(istart + 1) : iend]
        ):
            assert jstart in valid_steps, f"Row [{irow}] {jstart} is a terminator"
            assert (
                jend in valid_steps[jstart]
            ), f"Row [{irow}] {jstart}->{jend} is not a valid step"


@pytest.mark.parametrize("special_passes", [None, GRAPHBLAS_OPENMP_PASSES])
def test_bfs(special_passes):
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
    assert a.verify()

    parents, levels = mlalgo.bfs(0, a, compile_with_passes=special_passes)
    expected_parents = np.array([0, 0, 4, 0, 0, 2, 2, 2])
    expected_levels = np.array([0, 1, 2, 1, 1, 3, 3, 3])

    assert np.all(parents.toarray() == expected_parents)
    assert np.all(levels.toarray() == expected_levels)
