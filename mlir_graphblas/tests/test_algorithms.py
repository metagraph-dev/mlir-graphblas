import os
import math
import numpy as np
import pytest
import grblas as gb
import scipy.sparse as ss
import mlir_graphblas.algorithms as mlalgo
from mlir_graphblas.sparse_utils import MLIRSparseTensor
from mlir_graphblas.random_utils import ChooseUniformContext
from mlir_graphblas.tools.utils import sparsify_array
from mlir_graphblas.mlir_builder import GRAPHBLAS_OPENMP_PASSES


TEST_FOLDER = os.path.dirname(__file__)


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


# DO NOT RUN THIS ALGORITHM WITH OPENMP UNTIL WE HAVE THREAD SAFE RNG
def test_ties():
    # fmt: off
    indices = np.array(
        [[0, 1], [0, 2],
         [1, 0], [1, 3],
         [2, 0], [2, 4],
         [3, 2],
         [4, 4]],
         dtype=np.uint64,
    )
    values = np.array(
        [100, 200, 300, 400, 175, 222, 333, 200],
        dtype=np.float64
    )
    # fmt: on
    sizes = np.array([5, 5], dtype=np.uint64)
    sparsity = np.array([False, True], dtype=np.bool8)
    graph = MLIRSparseTensor(indices, values, sizes, sparsity)
    assert graph.verify()

    subgraph = mlalgo.totally_induced_edge_sampling(graph, 0.25, rand_seed=2021)
    assert np.all(subgraph.pointers[1] == [0, 1, 1, 3, 4, 5])
    assert np.all(subgraph.indices[1] == [2, 0, 4, 2, 4])


def test_graph_sage():
    # fmt: off
    A = np.array([
        [0, 1, 0, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
    ], dtype=np.float64)
    A = sparsify_array(A, [False, True])

    _num_nodes = A.shape[0]
    _dims = [7, 11, 13]
    # features = np.random.rand(_num_nodes, _dims[0])
    features = np.array([
        [0.39108972, 0.5012513, 0.89723081, 0.27099058, 0.32275578, 0.49175124, 0.92462542],
        [0.53783469, 0.9879627, 0.70174826, 0.93854928, 0.30000489, 0.63668296, 0.44688698],
        [0.69605861, 0.27255265, 0.35099912, 0.55982677, 0.4273765, 0.71363344, 0.06490641],
        [0.77244266, 0.88421098, 0.93914428, 0.0526303, 0.79569921, 0.86189287, 0.4751131],
        [0.02654776, 0.7612383, 0.51814711, 0.08852528, 0.33581081, 0.34977708, 0.80279946],
        [0.97218887, 0.8329417, 0.20214344, 0.39938109, 0.69338518, 0.37085116, 0.34679646],
        [0.96758692, 0.19503829, 0.72639853, 0.16756625, 0.76714647, 0.16902978, 0.32243089],
        [0.86073823, 0.75369185, 0.83310417, 0.97361141, 0.97555348, 0.07196352, 0.3626788],
    ], dtype=np.float64)
    features = sparsify_array(features, [False, True])

    # weight_matrices = [np.random.rand(2 * prev_dim, dim) for prev_dim, dim in zip(_dims, _dims[1:])]
    # weight_matrices = [weight_matrix - np.mean(weight_matrix) for weight_matrix in weight_matrices]
    weight_matrices = [
        np.array([
            0.44849969, -0.46665052,  0.38610724, -0.31954415, -0.31147248,  0.44301996, -0.17798382,  0.40529587,
            0.05003699,  0.34199498,  0.01262532, -0.09585784, -0.00561395, -0.00705517, -0.09003662,  0.1275429 ,
           -0.22295171, -0.10000134,  0.39103172,  0.44716378,  0.39686884, -0.34738009, -0.19950878,  0.29104661,
            0.11583818, -0.29612347, -0.39153596, -0.46010402,  0.52668431,  0.40351219,  0.22675305,  0.4982137 ,
           -0.07524554, -0.12946731,  0.44637269, -0.05820939, -0.30486384, -0.16480252,  0.03336091,  0.09315271,
           -0.05037402,  0.17064608, -0.13781243, -0.28559323, -0.19857246,  0.23892015, -0.00151305, -0.01174612,
           -0.43757934,  0.14200969,  0.08152132, -0.18876494, -0.25516716, -0.11913041, -0.34402115, -0.44478335,
           -0.03456947,  0.16198755,  0.22278569, -0.19652664, -0.08282499,  0.30166882, -0.01572978,  0.10788459,
            0.41259363,  0.1004477 , -0.38695633,  0.22499054, -0.42970007, -0.05014973,  0.16016598,  0.38971812,
           -0.22680452,  0.44080774, -0.2054676 , -0.44422513, -0.18831138,  0.18761742, -0.2546238 , -0.32898005,
           -0.30363691,  0.51678939,  0.34565312, -0.18908647, -0.28406102,  0.10069261,  0.13913278, -0.13931456,
            0.26840736, -0.26450718, -0.09599091,  0.18030395,  0.07797712, -0.39424769, -0.22933512,  0.12896401,
            0.21418343, -0.02390825, -0.43637196,  0.37204584, -0.08008971,  0.50665893,  0.15342747,  0.09842373,
           -0.25118726,  0.49232265, -0.25966891, -0.18288796,  0.51514014, -0.29988576, -0.28171506, -0.11972695,
           -0.1116426 ,  0.22407577,  0.07196866,  0.01189061, -0.39764054,  0.0846052 , -0.01040355,  0.50315026,
           -0.2239747 , -0.1650933 , -0.3489139 ,  0.4502217 ,  0.14315217, -0.13754185, -0.4372288 , -0.4174219 ,
            0.23195491, -0.31930873,  0.00634386, -0.30075998,  0.31291442, -0.15781314, -0.03668325, -0.01741184,
            0.51845429, -0.19204506, -0.25458102, -0.29754166,  0.48897955,  0.39761734,  0.50640932,  0.36440904,
           -0.13967416, -0.0590627 , -0.14178824, -0.38033831, -0.06993286, -0.03272309, -0.08665377, -0.37797494,
            0.22340132,  0.41555362]).reshape(14, 11),
        np.array([
           -0.22950825,  0.47356333, -0.23424293, -0.29770815, -0.34278594,  0.00926331, -0.15869387, -0.01557874,
            0.47950994, -0.11531171,  0.50282882,  0.15465202, -0.13093073,  0.07591561,  0.08070483, -0.2921942 ,
           -0.02377975,  0.37706882,  0.2608012 , -0.13428392, -0.08843151,  0.48539093,  0.19705366,  0.08918976,
            0.08232219, -0.16436965, -0.2884586 , -0.39956644,  0.50318162,  0.37740941, -0.01683036, -0.06273827,
            0.35301233,  0.04761192,  0.12480188,  0.42263395, -0.38179371,  0.1269861 ,  0.03290487, -0.01790114,
           -0.13740774,  0.11452844,  0.18328129,  0.28677584, -0.30561157, -0.24498321, -0.34226796,  0.12621682,
           -0.35308995, -0.43483672, -0.48333741,  0.41068243,  0.17499106,  0.4670842 , -0.42290461,  0.17553893,
           -0.47494292, -0.43169526,  0.14721555,  0.08425418, -0.18527037,  0.49794864,  0.19509773,  0.4071933 ,
            0.03311219,  0.00488919, -0.41919659,  0.36778391, -0.04703998,  0.0398842 , -0.28625282,  0.41089371,
           -0.31845561, -0.15968473, -0.37156018,  0.28574549,  0.22923943, -0.28373254,  0.39101841,  0.18762113,
            0.41920602, -0.13812974,  0.46620671, -0.36662375, -0.32503651, -0.33775304, -0.22632346, -0.1268525 ,
           -0.48116324,  0.10788509, -0.37015706,  0.0088477 , -0.45482819,  0.50917172,  0.44409586, -0.41619625,
            0.1047773 ,  0.3254928 , -0.18503631,  0.15787566,  0.40661986, -0.28784059, -0.40460377,  0.27253348,
           -0.27261771, -0.1719656 ,  0.02250423, -0.42453753,  0.14162989, -0.20185391, -0.27615418,  0.1258827 ,
           -0.36205247, -0.45835454, -0.30714659, -0.0173512 ,  0.41241587, -0.42408194,  0.29607361,  0.14064099,
            0.14000264, -0.34832738, -0.35163151,  0.16381133, -0.11631709, -0.21180209, -0.18789508,  0.04168156,
            0.33705247,  0.3608453 , -0.29178354, -0.26484368,  0.47700953, -0.06947477, -0.01467548, -0.3786903 ,
            0.43221556, -0.05880655,  0.47559545, -0.3807028 ,  0.19975792,  0.16071837, -0.38125686, -0.30645349,
            0.01190725,  0.29984574,  0.03564541, -0.12862994,  0.14654883, -0.48636503, -0.4102408 ,  0.48018389,
           -0.02150764,  0.34839799, -0.24232278,  0.23520199, -0.07027585,  0.00201372, -0.33555447, -0.03277968,
           -0.15786746,  0.49581866,  0.12140629,  0.0545428 , -0.06205085,  0.28415453, -0.05148018,  0.1314153 ,
            0.2758047 , -0.22779641,  0.21627012, -0.39168501,  0.10099208, -0.39645095, -0.16575748,  0.21427745,
           -0.48450068,  0.14938831, -0.45438177, -0.10204422, -0.01765876,  0.37052772, -0.18384067,  0.08190951,
            0.15128443,  0.06960123, -0.16301708,  0.03613785,  0.38059623,  0.05107006, -0.42448801,  0.2226469 ,
           -0.33395037,  0.39704235, -0.19176086,  0.13343548,  0.32719678,  0.43761432,  0.44786207, -0.25254014,
           -0.05925346,  0.40904021, -0.38186733,  0.31565752, -0.08247288,  0.30429126, -0.21306066, -0.12471994,
           -0.23898239,  0.08762914,  0.36601944,  0.12059623, -0.4629562 ,  0.4883836 ,  0.32175003,  0.0314467 ,
            0.20712602, -0.06729656,  0.35388646,  0.30054174,  0.32922542, -0.32104118,  0.08844141, -0.22243073,
           -0.09338935, -0.08056523, -0.34228469, -0.24185857,  0.0936732 ,  0.38480919,  0.49225913, -0.28794679,
            0.45533206,  0.22958076, -0.2348411 , -0.20642301,  0.23596382,  0.02872514, -0.33251817,  0.17174844,
           -0.03348297, -0.32859607,  0.10901442, -0.33836711, -0.16835414,  0.20867398, -0.19857043, -0.16663052,
            0.07825061, -0.37368182, -0.34076121,  0.40861486, -0.201939  , -0.44015233, -0.29720221,  0.16881501,
           -0.19469514,  0.0423145 ,  0.34634573,  0.05801634,  0.35696215,  0.05604903, -0.11479683, -0.45094242,
           -0.40379434,  0.50817091,  0.18003263, -0.24748827, -0.25429576,  0.09361117, -0.48118312,  0.32919184,
            0.1726551 ,  0.50909515,  0.2381023 ,  0.24317162, -0.39750715,  0.08456536,  0.43103512,  0.21821163,
           -0.16936465,  0.19771295,  0.06517018, -0.28561315, -0.05104375, -0.04098526]).reshape(22, 13)
    ]
    weight_matrices = [sparsify_array(weight_matrix, [False, True]) for weight_matrix in weight_matrices]

    expected = np.array([
        [0., 0.07379483, 0.26991188, 0., 0., 0.12508786, 0.34639293, 0., 0., 0., 0., 0.55470501, 0.69164241],
        [0., 0.07835575, 0.31145318, 0., 0., 0.0911438, 0.31115009, 0., 0., 0.,  0., 0.62758255, 0.63077402],
        [0., 0.16404006, 0.23516657, 0., 0., 0.10430948, 0.33573165, 0., 0., 0., 0., 0.69423177, 0.55877865],
        [0., 0.06955507, 0.34183008, 0., 0., 0.09311781, 0.33595258, 0., 0., 0., 0., 0.62122617, 0.60898052],
        [0., 0.2550802, 0.01022494, 0., 0., 0., 0.2638225, 0., 0., 0., 0., 0.64477487, 0.6704421],
        [0., 0.10571722, 0.24756088, 0., 0., 0.10241749, 0.34497212, 0., 0., 0., 0., 0.56359365, 0.69311223],
        [0., 0.18235096, 0.17561458, 0., 0., 0.12238869, 0.33438618, 0., 0., 0., 0., 0.55062202, 0.71128752],
        [0., 0.08043667, 0.16902664, 0., 0., 0.07887828, 0.30559865, 0., 0., 0., 0., 0.44908765, 0.81465815],
    ], dtype=np.float64)
    # fmt: on

    sample_count_per_layer = [999, 999]
    print("pre-RNG")
    rng_context = ChooseUniformContext(seed=1234)
    print("pre-call graph_sage")
    ans = mlalgo.graph_sage(
        A,
        features,
        weight_matrices,
        sample_count_per_layer,
        rng_context,
    ).toarray()

    print("pre-final comparison")
    assert np.isclose(expected, ans).all()


def test_geolocation():
    # fmt: off
    indices = np.array([
        [0, 1], [0, 2],
        [1, 0], [1, 3],
        [2, 0], [2, 4], [2, 5],
        [3, 1], [3, 4], [3, 6], [3, 7],
        [4, 2], [4, 3],
        [5, 2],
        [6, 3],
        [7, 3],
    ], dtype=np.uint64)
    values = np.array(
        [100, 200, 100, 300, 200, 400, 500, 300, 600, 700, 800, 400, 600, 500, 700, 800],
        dtype=np.float64)
    sizes = np.array([8, 8], dtype=np.uint64)
    sparsity = np.array([False, True], dtype=np.bool8)
    graph = MLIRSparseTensor(indices, values, sizes, sparsity)

    indices = np.array([[1], [2], [4], [6], [7]], dtype=np.uint64)
    values = np.array([37.7449063493, 37.8668048274, 9.4276164485, 33.9774659389, 39.2598884729], dtype=np.float64)
    sizes = np.array([8], dtype=np.uint64)
    sparsity = np.array([True], dtype=np.bool8)
    lat = MLIRSparseTensor(indices, values, sizes, sparsity)

    indices = np.array([[1], [2], [4], [6], [7]], dtype=np.uint64)
    values = np.array([-122.009432884, -122.257973253, -110.640705659, -114.886512278, -106.804662071],
                      dtype=np.float64)
    lon = MLIRSparseTensor(indices, values, sizes, sparsity)

    expected_lat = np.array(
        [37.80592086, 37.74490635, 37.86680483, 33.97769335,  9.42761645, 37.86680483, 33.97746594, 39.25988847])
    expected_lon = np.array(
        [-122.13360051, -122.00943288, -122.25797325, -114.88638873,
         -110.64070566, -122.25797325, -114.88651228, -106.80466207])
    # fmt: on
    lat, lon = mlalgo.geolocation(graph, lat, lon)

    np.testing.assert_allclose(lat.toarray(), expected_lat)
    np.testing.assert_allclose(lon.toarray(), expected_lon)


CONNECTED_COMPONENTS_TEST_CASES = [
    pytest.param(
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
        id="orphans",
    ),
    pytest.param(
        np.array(
            [
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            ]
        ),
        id="no_orphans",
    ),
]


@pytest.mark.parametrize("A_dense", CONNECTED_COMPONENTS_TEST_CASES)
def test_connected_components(A_dense):
    A = ss.csr_matrix(A_dense)
    num_connected_components, expected_ans = ss.csgraph.connected_components(A)

    A_sparse = sparsify_array(A_dense.astype(float), [False, True])
    ans = mlalgo.connected_components(A_sparse)
    ans = ans.toarray()

    assert num_connected_components == len(np.unique(ans))
    assert num_connected_components == len(set(zip(ans, expected_ans)))


def test_application_classification():
    with np.load(
        os.path.join(TEST_FOLDER, "data/application_classification.npz")
    ) as data:
        # Inputs
        data_vertex = data["data_vertex"]
        pattern_vertex = data["pattern_vertex"]
        data_edges_table = data["data_edges_table"]
        pattern_edges_table = data["pattern_edges_table"]
        de = data["data_edges"]
        pe = data["pattern_edges"]
        # Expected output
        expected_output = data["mu"]

    def sparse_from_dense(arr):
        nrows, ncols = arr.shape
        indices = np.array(
            [[i // ncols, i % ncols] for i in range(nrows * ncols)], dtype=np.uint64
        )
        values = arr.flatten()
        sizes = np.array(arr.shape, dtype=np.uint64)
        sparsity = np.array([False, True], dtype=np.bool8)
        return MLIRSparseTensor(indices, values, sizes, sparsity)

    dv = sparse_from_dense(data_vertex)
    pv = sparse_from_dense(pattern_vertex)
    det = sparse_from_dense(data_edges_table)
    pet = sparse_from_dense(pattern_edges_table)
    rv = mlalgo.application_classification(dv, det, de, pv, pet, pe)

    np.testing.assert_array_almost_equal(expected_output, rv.values.reshape(rv.shape))


def test_graph_wave():
    A = np.array(
        [
            [0, 1, 0, 1, 1, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
        ]
    )
    A_sparse = sparsify_array(A.astype(np.float64), [False, True])

    # fmt: off
    expected_ans = np.array([
        [1, 0, 0.87774496, 0.18258206, 0.73336003, 0.12847191, 0.81480474, 0.04483691, 0.92858125, 0.18957235, 1, 0, 0.88585881, 0.1907184, 0.73123909, 0.16473694, 0.77447817, 0.07729033, 0.89243752, 0.18318066],
        [1, 0, 0.87774496, 0.18258206, 0.73336003, 0.12847191, 0.81480474, 0.04483691, 0.92858125, 0.18957235, 1, 0, 0.88585881, 0.1907184, 0.73123909, 0.16473694, 0.77447817, 0.07729033, 0.89243752, 0.18318066],
        [1, 0, 0.87284429, 0.20949231, 0.72155559, 0.17678271, 0.79565621, 0.11923267, 0.88682803, 0.29206145, 1, 0, 0.87947901, 0.22133857, 0.71546402, 0.21813219, 0.75051113, 0.15777976, 0.84361165, 0.29583392],
        [1, 0, 0.87774496, 0.18258206, 0.73336003, 0.12847191, 0.81480474, 0.04483691, 0.92858125, 0.18957235, 1, 0, 0.88585881, 0.1907184, 0.73123909, 0.16473694, 0.77447817, 0.07729033, 0.89243752, 0.18318066],
        [1, 0, 0.87766593, 0.1933058, 0.73318096, 0.14998893, 0.81446235, 0.07760536, 0.92757334, 0.23415394, 1, 0, 0.88576381, 0.20334594, 0.73100612, 0.19011774, 0.77401683, 0.11605133, 0.8910841, 0.23619084],
        [1, 0, 0.87663525, 0.1762012, 0.73095019, 0.11394647, 0.81154927, 0.02262734, 0.92156405, 0.15929394, 1, 0, 0.88447675, 0.18333647, 0.72823205, 0.1474896, 0.77106314, 0.05022246, 0.88602628, 0.14634493],
        [1, 0, 0.87663525, 0.1762012, 0.73095019, 0.11394647, 0.81154927, 0.02262734, 0.92156405, 0.15929394, 1, 0, 0.88447675, 0.18333647, 0.72823205, 0.1474896, 0.77106314, 0.05022246, 0.88602628, 0.14634493],
        [1, 0, 0.8796893, 0.15611312, 0.73895622, 0.07680882, 0.82548327, -0.03400087, 0.95047066, 0.0820154, 1, 0, 0.88862453, 0.16028382, 0.7394665, 0.10587372, 0.78983456, -0.01201057, 0.92257343, 0.06079913]
    ])
    # fmt: on

    ans = mlalgo.graph_wave(
        A_sparse,
        [0.5, 0.6],
        1.7099629140148158,
        30,
        sparsify_array(np.arange(0, 10, 2, dtype=np.float64), [True]),
    )
    assert np.all(np.isclose(ans, expected_ans))
