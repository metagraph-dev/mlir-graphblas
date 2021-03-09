from libc.stdint cimport uint64_t
from libcpp.vector cimport vector

cdef extern from "SparseUtils.cpp":
    cdef cppclass SparseTensor:
        SparseTensor(vector[uint64_t], uint64_t) except +
        void add(const vector[uint64_t], double)
        void sort()
        uint64_t getRank() const

    void delSparseTensor(void *)


def myfunc():
    cdef vector[uint64_t] v = vector[uint64_t]()
    v.push_back(1)
    cdef SparseTensor *rv = new SparseTensor(v, 1)
    print(rv.getRank())
    del rv

