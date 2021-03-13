cimport cython
import numpy as np
cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t, uintptr_t
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp cimport bool
from numpy cimport float32_t, float64_t


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

    cdef struct MemRef1DF32:
        const float *base
        const float *data
        uint64_t off
        uint64_t sizes[1]
        uint64_t strides[1]

    cdef struct MemRef1DF64:
        const double *base
        const double *data
        uint64_t off
        uint64_t sizes[1]
        uint64_t strides[1]

    void delSparseTensor(void *)
    MemRef1DF32 sparseValuesF32(void *)
    MemRef1DF64 sparseValuesF64(void *)


def myfunc():
    cdef vector[uint64_t] v = vector[uint64_t]()
    v.push_back(1)
    cdef SparseTensor *st = new SparseTensor(v, 1)
    print(st.getRank())

    cdef bool b = True
    cdef SparseTensorStorage[uint64_t, uint64_t, double] *sts = new SparseTensorStorage[uint64_t, uint64_t, double](st, &b)
    print(sts.getRank(), sts.getDimSize(0))
    del st


# st for "sparse tensor"
ctypedef fused st_index_t:
    uint32_t
    uint64_t

ctypedef fused st_value_t:
    float32_t
    float64_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef uintptr_t build_sparse_tensor(
    st_index_t[:, :] indices,  # N x D
    st_value_t[:] values,      # N
    uint64_t[:] sizes,         # D
    bool[:] sparsity,          # D
    ptr_type=np.uint64,
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
    cdef int ptr_type_num = np.dtype(ptr_type).num
    if ptr_type_num != np.NPY_UINT32 and ptr_type_num != np.NPY_UINT64:
        raise TypeError(f"ptr_type must be np.uint32 or np.uint64, not: {ptr_type}")

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
    cdef uintptr_t rv
    if ptr_type_num == np.NPY_UINT32:
        rv = <uintptr_t>(new SparseTensorStorage[uint32_t, st_index_t, st_value_t](tensor, sparsity_array))
    else:
        rv = <uintptr_t>(new SparseTensorStorage[uint64_t, st_index_t, st_value_t](tensor, sparsity_array))
    free(sparsity_array)
    del tensor
    return rv


def run_example():
    cdef np.ndarray[uint64_t, ndim=2] indices = np.array([[0, 0], [1, 1]], dtype=np.uint64)
    cdef np.ndarray[float64_t, ndim=1] values = np.array([1.2, 3.4], dtype=np.float64)
    cdef np.ndarray[uint64_t, ndim=1] sizes = np.array([2, 2], dtype=np.uint64)
    cdef np.ndarray[bool, ndim=1] sparsity = np.array([True, True], dtype=np.bool8)
    cdef uintptr_t ptr = build_sparse_tensor[uint64_t, float64_t](indices, values, sizes, sparsity)
    cdef SparseTensorStorage[uint64_t, uint64_t, float64_t] *tensor = (
        <SparseTensorStorage[uint64_t, uint64_t, float64_t]*>ptr
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

    cdef MemRef1DF64 x64 = sparseValuesF64(tensor)
    print('sparseValuesF64:', x64.off, x64.sizes, x64.strides)

    ptr = build_sparse_tensor[uint64_t, float32_t](indices, values.astype(np.float32), sizes, sparsity)
    cdef SparseTensorStorage[uint64_t, uint64_t, float32_t] *tensor32 = (
        <SparseTensorStorage[uint64_t, uint64_t, float32_t]*>ptr
    )
    cdef MemRef1DF32 x32 = sparseValuesF32(tensor32)
    print('AA')
    print('sparseValuesF32:', x32.off, x32.sizes, x32.strides)

    # x64 = sparseValuesF64(tensor32)
    return ptr

