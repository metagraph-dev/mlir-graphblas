from typing import List
from .functions import (
    create_empty_matrix,
    Transpose,
    MatrixSelect,
    MatrixReduceToScalar,
    MatrixApply,
    MatrixMultiply,
)
from .sparse_utils import MLIRSparseTensor


transpose = Transpose().compile()
matrix_select_triu = MatrixSelect("TRIU").compile()
matrix_select_tril = MatrixSelect("TRIL").compile()
matrix_select_gt0 = MatrixSelect("gt0").compile()
matrix_reduce = MatrixReduceToScalar().compile()
matrix_apply_min = MatrixApply("min").compile()
mxm_plus_pair = MatrixMultiply("plus_pair", mask=True).compile()
mxm_plus_times = MatrixMultiply("plus_times").compile()
mxm_plus_plus = MatrixMultiply("plus_plus").compile()


def triangle_count(A: MLIRSparseTensor) -> int:
    # Create U and L matrices
    U = create_empty_matrix(*A.shape, nnz=A.pointers[1][-1])
    L = create_empty_matrix(*A.shape, nnz=A.pointers[1][-1])
    matrix_select_triu(U, A)
    matrix_select_tril(L, A)
    # Count Triangles
    UT = create_empty_matrix(*U.shape, nnz=U.pointers[1][-1])
    transpose(UT, U)
    C = create_empty_matrix(*A.shape, nnz=L.pointers[1][-1])
    mxm_plus_pair(C, L, UT, L)
    num_triangles = matrix_reduce(C)
    assert (
        int(num_triangles) == num_triangles
    ), f"{num_triangles} is unexpectedly not a whole number"
    return int(num_triangles)


def dense_neural_network(
    W: List[MLIRSparseTensor],
    Bias: List[MLIRSparseTensor],
    Y0: MLIRSparseTensor,
    ymax=32.0,
):
    nlayers = len(W)

    # We will swap between Y1 and Y2 in each calculation
    Y1 = create_empty_matrix(*Y0.shape, nnz=int(Y0.pointers[1][-1]) * 10)
    Y2 = create_empty_matrix(*Y0.shape, nnz=int(Y0.pointers[1][-1]) * 10)
    WT = create_empty_matrix(*W[0].shape, nnz=W[0].pointers[1][-1])
    for layer in range(nlayers):
        print(f"Layer {layer+1} of {nlayers}")
        transpose(WT, W[layer])
        if layer == 0:
            mxm_plus_times(Y1, Y0, WT)
        else:
            mxm_plus_times(Y1, Y2, WT)

        # Normally, I would need to transpose this, but I know these are purely diagonal matrices
        mxm_plus_plus(Y2, Y1, Bias[layer])

        matrix_select_gt0(Y1, Y2)
        matrix_apply_min(Y2, Y1, ymax)

    return Y2
