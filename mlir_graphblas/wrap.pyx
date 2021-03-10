cimport cython
import numpy as np
cimport numpy as np
from libc.stdint cimport uint64_t, uintptr_t
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "SparseUtils.cpp":
    cdef cppclass SparseTensor:
        SparseTensor(vector[uint64_t], uint64_t) except +
        void add(const vector[uint64_t], double)
        void sort()
        uint64_t getRank() const

    cdef cppclass SparseTensorStorage[P, I, V]:
        SparseTensorStorage(SparseTensor *, bool *) except +
        uint64_t getRank() const
        uint64_t getDimSize(uint64_t d)
        void getPointers(vector[P] **, uint64_t)
        void getIndices(vector[I] **, uint64_t)
        void getValues(vector[V] **)

    void delSparseTensor(void *)


def myfunc():
    cdef vector[uint64_t] v = vector[uint64_t]()
    v.push_back(1)
    cdef SparseTensor *st = new SparseTensor(v, 1)
    print(st.getRank())

    cdef bool b = True
    cdef SparseTensorStorage[uint64_t, uint64_t, double] *sts = new SparseTensorStorage[uint64_t, uint64_t, double](st, &b)
    print(sts.getRank(), sts.getDimSize(0))
    del st


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef uintptr_t build_sparse_tensor(
    uint64_t[:, :] indices, # N x D
    np.float64_t[:] values, # N
    uint64_t[:] sizes,      # D
    bool[:] sparsity,       # D
) except 0:
    cdef np.intp_t N = values.shape[0]  # number of values
    cdef np.intp_t D = sizes.shape[0]  # rank, number of dimensions
    if sparsity.shape[0] != D:
        raise ValueError(
            f'Length of sizes and sparsity arrays must match: {sizes.shape[0]} != {sparsity.shape[0]}'
        )
    if indices.shape[0] != N:
        raise ValueError(
            'First dimension of indices array must match length of values array: '
            f'{indices.shape[0]} != {values.shape[0]}'
        )
    if indices.shape[1] != D:
        raise ValueError(
            'Second dimension of indices array must match length of sparsity and sizes arrays: '
            f'{indices.shape[1]} != {D}'
        )

    cdef vector[uint64_t] sizes_vector = vector[uint64_t](D)
    for i in range(D):
        sizes_vector[i] = sizes[i]
    cdef SparseTensor *tensor = new SparseTensor(sizes_vector, N)

    cdef vector[uint64_t] *ind_ptr
    cdef vector[uint64_t] ind
    for i in range(N):
        ind_ptr = new vector[uint64_t](D)
        ind = ind_ptr[0]
        for j in range(D):
            ind[j] = indices[i, j]
        tensor.add(ind, values[i])

    cdef bool *sparsity_array = <bool*>malloc(sizeof(bool) * D)
    for i in range(D):
        sparsity_array[i] = sparsity[i]
    cdef SparseTensorStorage[uint64_t, uint64_t, double] *rv = (
        new SparseTensorStorage[uint64_t, uint64_t, double](tensor, sparsity_array)
    )
    free(sparsity_array)
    del tensor
    return <uintptr_t>rv


def run_example():
    cdef np.ndarray[uint64_t, ndim=2] indices = np.array([[0, 0], [1, 1]], dtype=np.uint64)
    cdef np.ndarray[np.float64_t, ndim=1] values = np.array([1.2, 3.4], dtype=np.float64)
    cdef np.ndarray[uint64_t, ndim=1] sizes = np.array([2, 2], dtype=np.uint64)
    cdef np.ndarray[bool, ndim=1] sparsity = np.array([True, True], dtype=np.bool8)
    cdef uintptr_t ptr = build_sparse_tensor(indices, values, sizes, sparsity)
    cdef SparseTensorStorage[uint64_t, uint64_t, double] *tensor = (
        <SparseTensorStorage[uint64_t, uint64_t, double]*>ptr
    )
    print('rank:', tensor.getRank())
    print('dim sizes:', tensor.getDimSize(0), tensor.getDimSize(1))
    print('pointer:', ptr)

    cdef vector[uint64_t] *ind_ptr
    print('pointers')
    tensor.getPointers(&ind_ptr, 0)
    print('  0:', ind_ptr[0])
    tensor.getPointers(&ind_ptr, 1)
    print('  1:', ind_ptr[0])

    print('indices')
    tensor.getIndices(&ind_ptr, 0)
    print('  0:', ind_ptr[0])
    tensor.getIndices(&ind_ptr, 1)
    print('  1:', ind_ptr[0])

    print('values')
    cdef vector[double] *val_ptr
    tensor.getValues(&val_ptr)
    print(' ', val_ptr[0])

    return ptr

