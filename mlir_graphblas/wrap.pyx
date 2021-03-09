from libc.stdint cimport uint64_t, uintptr_t
from libc.stdlib cimport malloc
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


def build_sparse_tensor(): #indices, values, sizes, sparsity):
    # R: rank, number of dimensions
    # N: number of values
    # indices[N, R]
    # values[N]
    # sparsity[R]
    # sizes[R]

    import numpy as np
    indices = np.array([[0, 0], [1, 1]])
    values = np.array([1.2, 3.4])
    sizes = np.array([2, 2])
    sparsity = np.array([True, True])

    cdef uint64_t N = values.size
    cdef uint64_t R = sizes.size
    cdef vector[uint64_t] sizes_vector = vector[uint64_t](R)
    for i in range(R):
        sizes_vector[i] = sizes[i]
    cdef SparseTensor *tensor = new SparseTensor(sizes_vector, N)
    print('tensor rank', tensor.getRank())

    cdef vector[vector[uint64_t]] indices_vector = vector[vector[uint64_t]](N)
    cdef vector[uint64_t] * ind_ptr
    for i in range(N):
        ind_ptr = new vector[uint64_t](R)
        indices_vector[i] = ind_ptr[0]

    cdef vector[uint64_t] ind
    for i in range(N):
        ind = indices_vector[i]
        for j in range(R):
            ind[j] = indices[i, j]
        tensor.add(ind, values[i])

    cdef bool *sparsity_array = <bool*>malloc(sizeof(bool) * R)
    for i in range(R):
        sparsity_array[i] = sparsity[i]
    cdef SparseTensorStorage[uint64_t, uint64_t, double] *rv = new SparseTensorStorage[uint64_t, uint64_t, double](tensor, sparsity_array)
    print('rank', rv.getRank())
    print('dim sizes', rv.getDimSize(0), rv.getDimSize(1))
    print(<uintptr_t>rv)
    return <uintptr_t>rv

