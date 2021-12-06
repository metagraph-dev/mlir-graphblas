import math
import numpy as np
from mlir_graphblas import algo_utils, algorithms as mlalgo
from mlir_graphblas.sparse_utils import MLIRSparseTensor


def test_haversine_distance():
    # Sanity check
    # https://www.igismap.com/haversine-formula-calculate-geographic-distance-earth/
    # Nebraska
    # v1 = Vector.from_values([0], [41.507483])
    # w1 = Vector.from_values([0], [-99.436554])
    # Kansas
    # v2 = Vector.from_values([0], [38.504048])
    # w2 = Vector.from_values([0], [-98.315949])

    # Build a function to call the haversine_distance utility
    from mlir_graphblas.mlir_builder import MLIRFunctionBuilder

    irb = MLIRFunctionBuilder(
        "haversine_distance",
        input_types=[
            "tensor<?xf64, #CV64>",
            "tensor<?xf64, #CV64>",
            "tensor<?xf64, #CV64>",
            "tensor<?xf64, #CV64>",
        ],
        return_types=["tensor<?xf64, #CV64>"],
        aliases=mlalgo._build_common_aliases(),
    )
    v1, w1, v2, w2 = irb.inputs
    result = algo_utils.haversine_distance(irb, v1, w1, v2, w2)
    irb.return_vars(result)
    compiled_func = irb.compile()

    # haversine_distance(v1, w1, v2, w2)[0].new().isclose(347.3, abs_tol=0.1)
    v1 = MLIRSparseTensor(
        np.array([[0]], dtype=np.uint64),
        np.array([41.507483], dtype=np.float64),
        np.array([1], dtype=np.uint64),
        np.array([True], dtype=np.bool8),
    )
    w1 = MLIRSparseTensor(
        np.array([[0]], dtype=np.uint64),
        np.array([-99.436554], dtype=np.float64),
        np.array([1], dtype=np.uint64),
        np.array([True], dtype=np.bool8),
    )
    v2 = MLIRSparseTensor(
        np.array([[0]], dtype=np.uint64),
        np.array([38.504048], dtype=np.float64),
        np.array([1], dtype=np.uint64),
        np.array([True], dtype=np.bool8),
    )
    w2 = MLIRSparseTensor(
        np.array([[0]], dtype=np.uint64),
        np.array([-98.315949], dtype=np.float64),
        np.array([1], dtype=np.uint64),
        np.array([True], dtype=np.bool8),
    )

    dist = compiled_func(v1, w1, v2, w2)
    assert math.isclose(dist.values[0], 347.3, abs_tol=0.1)
