from .functions import create_empty_matrix, transpose, matrix_select, matrix_reduce, masked_spmm
from .sparse_utils import MLIRSparseTensor


def triangle_count(A: MLIRSparseTensor) -> int:
    # Create U and L matrices
    U = create_empty_matrix(*A.shape, A.pointers[1][-1])
    L = create_empty_matrix(*A.shape, A.pointers[1][-1])
    matrix_select(U, A, 'TRIU')
    matrix_select(L, A, 'TRIL')
    # Count Triangles
    UT = create_empty_matrix(*U.shape, U.pointers[1][-1])
    transpose(UT, U)
    C = create_empty_matrix(*A.shape, L.pointers[1][-1])
    masked_spmm(C, L, UT, mask=L, semiring='plus_pair')
    num_triangles = matrix_reduce(C)
    assert int(num_triangles) == num_triangles, f'{num_triangles} is unexpectedly not a whole number'
    return int(num_triangles)
