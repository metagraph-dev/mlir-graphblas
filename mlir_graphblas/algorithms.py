from typing import List
from .functions import (
    Transpose,
    MatrixSelect,
    MatrixReduceToScalar,
    MatrixApply,
    MatrixMultiply,
)
from .sparse_utils import MLIRSparseTensor


csr_to_csc = Transpose(swap_sizes=False).compile()
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
    U = matrix_select_triu(A)
    L = matrix_select_tril(A)
    # Count Triangles
    U_csc = csr_to_csc(U)
    C = mxm_plus_pair(L, U_csc, L)
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
    import time

    nlayers = len(W)

    Y = Y0.dup()
    very_start = now = time.time()
    for layer in range(nlayers):
        W_csc = csr_to_csc(W[layer])
        Y = mxm_plus_times(Y, W_csc)

        # Normally, I would need to transpose this, but I know these are purely diagonal matrices
        Y = mxm_plus_plus(Y, Bias[layer])

        Y = matrix_select_gt0(Y)
        Y = matrix_apply_min(Y, ymax)

        curr = time.time()
        diff, now = curr - now, curr
        print(f"Layer {layer+1} of {nlayers} took {diff:.2f} sec")

    print(f"\nTotal time = {(now - very_start):.2f} sec")
    return Y
