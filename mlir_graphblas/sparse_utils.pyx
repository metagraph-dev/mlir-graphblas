""" This wraps https://github.com/llvm/llvm-project/blob/main/mlir/lib/ExecutionEngine/SparseUtils.cpp """

cimport cython
import numpy as np
cimport numpy as np
from libc.stdint cimport int8_t, int16_t, int32_t, uint8_t, uint16_t, uint32_t, uint64_t, uintptr_t
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp cimport bool
from numpy cimport float32_t, float64_t, intp_t, ndarray

np.import_array()

cdef extern from "numpy/arrayobject.h" nogil:
    void PyArray_ENABLEFLAGS(ndarray, int flags)
    void PyArray_CLEARFLAGS(ndarray, int flags)


cpdef ndarray claim_buffer(uintptr_t ptr, shape, strides, dtype):
    return _wrap_buffer(ptr, shape, strides, dtype, np.NPY_ARRAY_WRITEABLE, np.NPY_ARRAY_OWNDATA)


# Heh, there are likely better ways to create a read-only buffer, but
# it's convenient for us to use the same API as `claim_buffer` above.
cpdef ndarray view_buffer(uintptr_t ptr, shape, strides, dtype):
    return _wrap_buffer(ptr, shape, strides, dtype, 0, 0)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef ndarray _wrap_buffer(uintptr_t ptr, shape, strides, dtype, int init_flags, int post_flags):
    cdef intp_t[:] shape_array = np.ascontiguousarray(shape, dtype=np.intp)
    cdef intp_t D = shape_array.shape[0]  # number of dimensions
    cdef intp_t[:] strides_array
    cdef intp_t *strides_ptr
    if strides is not None:
        strides_array = np.ascontiguousarray(strides, dtype=np.intp)
        if strides_array.shape[0] != D:
            raise ValueError(
                f'Length of shape and strides arrays must match: {shape_array.shape[0]} != {strides_array.shape[0]}'
            )
        strides_ptr = &strides_array[0]
    else:
        # Defer to default: assume normal C-contiguous
        strides_ptr = NULL
    cdef int typenum = np.dtype(dtype).num  # is there a better way to do this?
    cdef ndarray array = np.PyArray_New(
        ndarray,
        D,
        &shape_array[0],
        typenum,
        strides_ptr,
        <void*>ptr,
        -1,  # itemsize; ignored if the type always has the same number of bytes
        init_flags,
        <object>NULL
    )
    PyArray_ENABLEFLAGS(array, post_flags)
    return array


cdef extern from "SparseUtils.cpp" nogil:
    cdef cppclass SparseTensor:
        SparseTensor(vector[uint64_t], uint64_t) except +
        void add(const vector[uint64_t], double)
        void sort()
        uint64_t getRank() const

    cdef cppclass SparseTensorStorage[P, I, V]:
        SparseTensorStorage(void *) except +  # HACKED IN
        SparseTensorStorage(vector[uint64_t]&, bint) except +  # HACKED IN
        # SparseTensorStorage(SparseTensorStorage[P, I, V]&) except +  # HACKED IN
        # SparseTensorStorage(vector[uint64_t]&, vector[P]&, vector[I]&, vector[V]&) except +  # HACKED IN

        SparseTensorStorage(SparseTensor *, bool *) except +
        uint64_t getRank() const
        uint64_t getDimSize(uint64_t d)
        void getPointers(vector[P] **, uint64_t)
        void getIndices(vector[I] **, uint64_t)
        void getValues(vector[V] **)

    # unsigned int
    cdef struct MemRef1DU8:
        const uint8_t *base
        const uint8_t *data
        uint64_t off
        uint64_t sizes[1]
        uint64_t strides[1]

    cdef struct MemRef1DU16:
        const uint16_t *base
        const uint16_t *data
        uint64_t off
        uint64_t sizes[1]
        uint64_t strides[1]

    cdef struct MemRef1DU32:
        const uint32_t *base
        const uint32_t *data
        uint64_t off
        uint64_t sizes[1]
        uint64_t strides[1]

    cdef struct MemRef1DU64:
        const uint64_t *base
        const uint64_t *data
        uint64_t off
        uint64_t sizes[1]
        uint64_t strides[1]

    # signed int
    # why no int64_t?
    cdef struct MemRef1DI8:
        const int8_t *base
        const int8_t *data
        uint64_t off
        uint64_t sizes[1]
        uint64_t strides[1]

    cdef struct MemRef1DI16:
        const int16_t *base
        const int16_t *data
        uint64_t off
        uint64_t sizes[1]
        uint64_t strides[1]

    cdef struct MemRef1DI32:
        const int32_t *base
        const int32_t *data
        uint64_t off
        uint64_t sizes[1]
        uint64_t strides[1]

    # floats
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

    uint64_t sparseDimSize(void *, uint64_t)

    MemRef1DU8 sparsePointers8(void *, uint64_t)
    MemRef1DU16 sparsePointers16(void *, uint64_t)
    MemRef1DU32 sparsePointers32(void *, uint64_t)
    MemRef1DU64 sparsePointers64(void *, uint64_t)

    MemRef1DU8 sparseIndices8(void *, uint64_t)
    MemRef1DU16 sparseIndices16(void *, uint64_t)
    MemRef1DU32 sparseIndices32(void *, uint64_t)
    MemRef1DU64 sparseIndices64(void *, uint64_t)

    MemRef1DI8 sparseValuesI8(void *)
    MemRef1DI16 sparseValuesI16(void *)
    MemRef1DI32 sparseValuesI32(void *)
    MemRef1DF32 sparseValuesF32(void *)
    MemRef1DF64 sparseValuesF64(void *)

    void delSparseTensor(void *)

    # HACKED IN
    # These return pointers to the vectors
    uint64_t get_rank(void *tensor)
    void *get_sizes_ptr(void *)
    void *get_pointers_ptr(void *)
    void *get_indices_ptr(void *)
    void *get_values_ptr(void *)

    void swap_sizes(void *tensor, void *new_sizes)
    void swap_pointers(void *tensor, void *new_pointers)
    void swap_indices(void *tensor, void *new_indices)
    void swap_values(void *tensor, void *new_values)

    void resize_pointers(void *tensor, uint64_t d, uint64_t size)
    void resize_index(void *tensor, uint64_t d, uint64_t size)
    void resize_values(void *tensor, uint64_t size)
    void resize_dim(void *tensor, uint64_t d, uint64_t size)

    void *dup_tensor(void *tensor)
    void *ptr8_to_tensor(void *tensor)
    void *tensor_to_ptr8(void *tensor)
    void *cast_csr_to_csc(void *tensor)
    void *cast_csc_to_csr(void *tensor)
    void *empty_like(void *tensor)
    void *empty(void *tensor, uint64_t ndims)


# st for "sparse tensor"
ctypedef fused st_index_t:
    uint8_t
    uint16_t
    uint32_t
    uint64_t

ctypedef fused st_value_t:
    int8_t
    int16_t
    int32_t
    float32_t
    float64_t


# XXX: Perhaps delete this in favor of `tensor.empty_like()` and `tensor.empty(d)`
cpdef empty_mlir_sparse_tensor_fast(uint64_t[:] sizes, bint is_sparse=True, pointer_type=np.uint64, index_type=np.uint64, value_type=np.float64):
    # Fast but unsafe, because we use a constructor of our creation
    pointer_dtype = np.dtype(pointer_type)
    index_dtype = np.dtype(index_type)
    value_dtype = np.dtype(value_type)

    cdef int pointer_type_num = pointer_dtype.num
    cdef int index_type_num = index_dtype.num
    cdef int value_type_num = value_dtype.num

    cdef intp_t D = sizes.shape[0]
    cdef vector[uint64_t] sizes_vector = vector[uint64_t](D)
    for i in range(D):
        sizes_vector[i] = sizes[i]

    cdef void *data
    if pointer_type_num == np.NPY_UINT8:
        if index_type_num == np.NPY_UINT8:
            if value_type_num == np.NPY_FLOAT32:
                data = (new SparseTensorStorage[uint8_t, uint8_t, float32_t](sizes_vector, is_sparse))
            elif value_type_num == np.NPY_FLOAT64:
                data = (new SparseTensorStorage[uint8_t, uint8_t, float64_t](sizes_vector, is_sparse))
            elif value_type_num == np.NPY_INT8:
                data = (new SparseTensorStorage[uint8_t, uint8_t, int8_t](sizes_vector, is_sparse))
            elif value_type_num == np.NPY_INT16:
                data = (new SparseTensorStorage[uint8_t, uint8_t, int16_t](sizes_vector, is_sparse))
            elif value_type_num == np.NPY_INT32:
                data = (new SparseTensorStorage[uint8_t, uint8_t, int32_t](sizes_vector, is_sparse))
            else:
                raise TypeError(f"Invalid type combo for pointers, indices, and values: {pointer_dtype}, {index_dtype}, {value_dtype}")
        else:
            raise TypeError(f"Invalid type combo for pointers, indices, and values: {pointer_dtype}, {index_dtype}, {value_dtype}")
    elif pointer_type_num == np.NPY_UINT16:
        if index_type_num == np.NPY_UINT16:
            if value_type_num == np.NPY_FLOAT32:
                data = (new SparseTensorStorage[uint16_t, uint16_t, float32_t](sizes_vector, is_sparse))
            elif value_type_num == np.NPY_FLOAT64:
                data = (new SparseTensorStorage[uint16_t, uint16_t, float64_t](sizes_vector, is_sparse))
            elif value_type_num == np.NPY_INT8:
                data = (new SparseTensorStorage[uint16_t, uint16_t, int8_t](sizes_vector, is_sparse))
            elif value_type_num == np.NPY_INT16:
                data = (new SparseTensorStorage[uint16_t, uint16_t, int16_t](sizes_vector, is_sparse))
            elif value_type_num == np.NPY_INT32:
                data = (new SparseTensorStorage[uint16_t, uint16_t, int32_t](sizes_vector, is_sparse))
            else:
                raise TypeError(f"Invalid type combo for pointers, indices, and values: {pointer_dtype}, {index_dtype}, {value_dtype}")
        else:
            raise TypeError(f"Invalid type combo for pointers, indices, and values: {pointer_dtype}, {index_dtype}, {value_dtype}")
    elif pointer_type_num == np.NPY_UINT32:
        if index_type_num == np.NPY_UINT32:
            if value_type_num == np.NPY_FLOAT32:
                data = (new SparseTensorStorage[uint32_t, uint32_t, float32_t](sizes_vector, is_sparse))
            elif value_type_num == np.NPY_FLOAT64:
                data = (new SparseTensorStorage[uint32_t, uint32_t, float64_t](sizes_vector, is_sparse))
            elif value_type_num == np.NPY_INT8:
                data = (new SparseTensorStorage[uint32_t, uint32_t, int8_t](sizes_vector, is_sparse))
            elif value_type_num == np.NPY_INT16:
                data = (new SparseTensorStorage[uint32_t, uint32_t, int16_t](sizes_vector, is_sparse))
            elif value_type_num == np.NPY_INT32:
                data = (new SparseTensorStorage[uint32_t, uint32_t, int32_t](sizes_vector, is_sparse))
            else:
                raise TypeError(f"Invalid type combo for pointers, indices, and values: {pointer_dtype}, {index_dtype}, {value_dtype}")
        elif index_type_num == np.NPY_UINT64:
            if value_type_num == np.NPY_FLOAT32:
                data = (new SparseTensorStorage[uint32_t, uint64_t, float32_t](sizes_vector, is_sparse))
            elif value_type_num == np.NPY_FLOAT64:
                data = (new SparseTensorStorage[uint32_t, uint64_t, float64_t](sizes_vector, is_sparse))
            else:
                raise TypeError(f"Invalid type combo for pointers, indices, and values: {pointer_dtype}, {index_dtype}, {value_dtype}")
        else:
            raise TypeError(f"Invalid type combo for pointers, indices, and values: {pointer_dtype}, {index_dtype}, {value_dtype}")
    elif pointer_type_num == np.NPY_UINT64:
        if index_type_num == np.NPY_UINT32:
            if value_type_num == np.NPY_FLOAT32:
                data = (new SparseTensorStorage[uint64_t, uint32_t, float32_t](sizes_vector, is_sparse))
            elif value_type_num == np.NPY_FLOAT64:
                data = (new SparseTensorStorage[uint64_t, uint32_t, float64_t](sizes_vector, is_sparse))
            else:
                raise TypeError(f"Invalid type combo for pointers, indices, and values: {pointer_dtype}, {index_dtype}, {value_dtype}")
        elif index_type_num == np.NPY_UINT64:
            if value_type_num == np.NPY_FLOAT32:
                data = (new SparseTensorStorage[uint64_t, uint64_t, float32_t](sizes_vector, is_sparse))
            elif value_type_num == np.NPY_FLOAT64:
                data = (new SparseTensorStorage[uint64_t, uint64_t, float64_t](sizes_vector, is_sparse))
            else:
                raise TypeError(f"Invalid type combo for pointers, indices, and values: {pointer_dtype}, {index_dtype}, {value_dtype}")
        else:
            raise TypeError(f"Invalid type combo for pointers, indices, and values: {pointer_dtype}, {index_dtype}, {value_dtype}")

    cdef MLIRSparseTensor rv = MLIRSparseTensor.__new__(MLIRSparseTensor)  # avoid __init__
    rv._data = data
    rv.ndim = D
    rv.pointer_dtype = pointer_dtype
    rv.index_dtype = index_dtype
    rv.value_dtype = value_dtype
    return rv


# XXX: Perhaps delete this in favor of `tensor.empty_like()` and `tensor.empty(d)`
cpdef empty_mlir_sparse_tensor_safe(uint64_t[:] sizes, bool[:] sparsity, pointer_type=np.uint64, index_type=np.uint64, value_type=np.float64):
    # Safe, because we use the standard initializer via SparseTensor
    cdef intp_t D = sizes.shape[0]  # rank, number of dimensions
    if sparsity.shape[0] != D:
        raise ValueError(
            f'Length of sizes and sparsity arrays must match: {sizes.shape[0]} != {sparsity.shape[0]}'
        )

    pointer_dtype = np.dtype(pointer_type)
    index_dtype = np.dtype(index_type)
    value_dtype = np.dtype(value_type)

    cdef int pointer_type_num = pointer_dtype.num
    cdef int index_type_num = index_dtype.num
    cdef int value_type_num = value_dtype.num

    cdef vector[uint64_t] sizes_vector = vector[uint64_t](D)
    for i in range(D):
        sizes_vector[i] = sizes[i]
    cdef SparseTensor *tensor = new SparseTensor(sizes_vector, 0)

    cdef bool *sparsity_array = <bool*>malloc(sizeof(bool) * D)
    for i in range(D):
        sparsity_array[i] = sparsity[i]

    cdef void *data
    try:
        if pointer_type_num == np.NPY_UINT8:
            if index_type_num == np.NPY_UINT8:
                if value_type_num == np.NPY_FLOAT32:
                    data = (new SparseTensorStorage[uint8_t, uint8_t, float32_t](tensor, sparsity_array))
                elif value_type_num == np.NPY_FLOAT64:
                    data = (new SparseTensorStorage[uint8_t, uint8_t, float64_t](tensor, sparsity_array))
                elif value_type_num == np.NPY_INT8:
                    data = (new SparseTensorStorage[uint8_t, uint8_t, int8_t](tensor, sparsity_array))
                elif value_type_num == np.NPY_INT16:
                    data = (new SparseTensorStorage[uint8_t, uint8_t, int16_t](tensor, sparsity_array))
                elif value_type_num == np.NPY_INT32:
                    data = (new SparseTensorStorage[uint8_t, uint8_t, int32_t](tensor, sparsity_array))
                else:
                    raise TypeError(f"Invalid type combo for pointers, indices, and values: {pointer_dtype}, {index_dtype}, {value_dtype}")
            else:
                raise TypeError(f"Invalid type combo for pointers, indices, and values: {pointer_dtype}, {index_dtype}, {value_dtype}")
        elif pointer_type_num == np.NPY_UINT16:
            if index_type_num == np.NPY_UINT16:
                if value_type_num == np.NPY_FLOAT32:
                    data = (new SparseTensorStorage[uint16_t, uint16_t, float32_t](tensor, sparsity_array))
                elif value_type_num == np.NPY_FLOAT64:
                    data = (new SparseTensorStorage[uint16_t, uint16_t, float64_t](tensor, sparsity_array))
                elif value_type_num == np.NPY_INT8:
                    data = (new SparseTensorStorage[uint16_t, uint16_t, int8_t](tensor, sparsity_array))
                elif value_type_num == np.NPY_INT16:
                    data = (new SparseTensorStorage[uint16_t, uint16_t, int16_t](tensor, sparsity_array))
                elif value_type_num == np.NPY_INT32:
                    data = (new SparseTensorStorage[uint16_t, uint16_t, int32_t](tensor, sparsity_array))
                else:
                    raise TypeError(f"Invalid type combo for pointers, indices, and values: {pointer_dtype}, {index_dtype}, {value_dtype}")
            else:
                raise TypeError(f"Invalid type combo for pointers, indices, and values: {pointer_dtype}, {index_dtype}, {value_dtype}")
        elif pointer_type_num == np.NPY_UINT32:
            if index_type_num == np.NPY_UINT32:
                if value_type_num == np.NPY_FLOAT32:
                    data = (new SparseTensorStorage[uint32_t, uint32_t, float32_t](tensor, sparsity_array))
                elif value_type_num == np.NPY_FLOAT64:
                    data = (new SparseTensorStorage[uint32_t, uint32_t, float64_t](tensor, sparsity_array))
                elif value_type_num == np.NPY_INT8:
                    data = (new SparseTensorStorage[uint32_t, uint32_t, int8_t](tensor, sparsity_array))
                elif value_type_num == np.NPY_INT16:
                    data = (new SparseTensorStorage[uint32_t, uint32_t, int16_t](tensor, sparsity_array))
                elif value_type_num == np.NPY_INT32:
                    data = (new SparseTensorStorage[uint32_t, uint32_t, int32_t](tensor, sparsity_array))
                else:
                    raise TypeError(f"Invalid type combo for pointers, indices, and values: {pointer_dtype}, {index_dtype}, {value_dtype}")
            elif index_type_num == np.NPY_UINT64:
                if value_type_num == np.NPY_FLOAT32:
                    data = (new SparseTensorStorage[uint32_t, uint64_t, float32_t](tensor, sparsity_array))
                elif value_type_num == np.NPY_FLOAT64:
                    data = (new SparseTensorStorage[uint32_t, uint64_t, float64_t](tensor, sparsity_array))
                else:
                    raise TypeError(f"Invalid type combo for pointers, indices, and values: {pointer_dtype}, {index_dtype}, {value_dtype}")
            else:
                raise TypeError(f"Invalid type combo for pointers, indices, and values: {pointer_dtype}, {index_dtype}, {value_dtype}")
        elif pointer_type_num == np.NPY_UINT64:
            if index_type_num == np.NPY_UINT32:
                if value_type_num == np.NPY_FLOAT32:
                    data = (new SparseTensorStorage[uint64_t, uint32_t, float32_t](tensor, sparsity_array))
                elif value_type_num == np.NPY_FLOAT64:
                    data = (new SparseTensorStorage[uint64_t, uint32_t, float64_t](tensor, sparsity_array))
                else:
                    raise TypeError(f"Invalid type combo for pointers, indices, and values: {pointer_dtype}, {index_dtype}, {value_dtype}")
            elif index_type_num == np.NPY_UINT64:
                if value_type_num == np.NPY_FLOAT32:
                    data = (new SparseTensorStorage[uint64_t, uint64_t, float32_t](tensor, sparsity_array))
                elif value_type_num == np.NPY_FLOAT64:
                    data = (new SparseTensorStorage[uint64_t, uint64_t, float64_t](tensor, sparsity_array))
                else:
                    raise TypeError(f"Invalid type combo for pointers, indices, and values: {pointer_dtype}, {index_dtype}, {value_dtype}")
            else:
                raise TypeError(f"Invalid type combo for pointers, indices, and values: {pointer_dtype}, {index_dtype}, {value_dtype}")
    finally:
        free(sparsity_array)
        del tensor

    cdef MLIRSparseTensor rv = MLIRSparseTensor.__new__(MLIRSparseTensor)  # avoid __init__
    rv._data = data
    rv.ndim = D
    rv.pointer_dtype = pointer_dtype
    rv.index_dtype = index_dtype
    rv.value_dtype = value_dtype
    return rv


cdef class MLIRSparseTensor:
    cdef void *_data
    cdef readonly uint64_t ndim
    cdef readonly object pointer_dtype
    cdef readonly object index_dtype
    cdef readonly object value_dtype

    def __init__(self, indices, values, uint64_t[:] sizes, bool[:] sparsity, pointer_type=np.uint64):
        indices = np.asarray(indices)
        if indices.ndim == 1:
            indices = indices[:, None]
        try:
            _build_sparse_tensor(self, indices, values, sizes, sparsity, pointer_type)
        except TypeError as exc:
            if "Function call with ambiguous argument types" in str(exc):
                if indices.dtype.type not in {np.uint8, np.uint16, np.uint32, np.uint64}:
                    raise TypeError(f"Bad dtype for indices: %s.  uint{8,16,32,64} expected." % indices.dtype)
                values = np.asarray(values)
                if values.dtype.type not in {np.int8, np.int16, np.int32, np.float32, np.float64}:
                    raise TypeError("Bad dtype for values: %s.  int{8,16,32} or float{32,64} expected." % values.dtype)
            raise

    def __dealloc__(self):
        delSparseTensor(self._data)

    @classmethod
    def from_raw_pointer(cls, uintptr_t data, pointer_dtype, index_dtype, value_dtype):
        cdef MLIRSparseTensor rv = MLIRSparseTensor.__new__(MLIRSparseTensor)  # avoid __init__
        rv._data = <void *>data
        rv.ndim = get_rank(rv._data)
        rv.pointer_dtype = np.dtype(pointer_dtype)
        rv.index_dtype = np.dtype(index_dtype)
        rv.value_dtype = np.dtype(value_dtype)
        return rv

    cpdef uint64_t get_dimsize(self, uint64_t d):
        if d >= self.ndim:
            raise IndexError(f'Bad dimension index: {d} >= {self.ndim}')
        return sparseDimSize(self._data, d)

    @property
    def data(self):
        return <uintptr_t>self._data

    # Use with caution!  Set to zero if data gets freed elsewhere
    @data.setter
    def data(self, uintptr_t data):
        self._data = <void*>data

    @property
    def shape(self):
        return tuple([sparseDimSize(self._data, i) for i in range(self.ndim)])

    cpdef ndarray get_pointers(self, uint64_t d):
        cdef MemRef1DU8 ref8
        cdef MemRef1DU16 ref16
        cdef MemRef1DU32 ref32
        cdef MemRef1DU64 ref64
        if d >= self.ndim:
            raise IndexError(f'Bad dimension index: {d} >= {self.ndim}')
        if self.pointer_dtype == np.uint8:
            ref8 = sparsePointers8(self._data, d)
            return view_buffer(<uintptr_t>ref8.data, ref8.sizes[0], ref8.strides[0], self.pointer_dtype)
        elif self.pointer_dtype == np.uint16:
            ref16 = sparsePointers16(self._data, d)
            return view_buffer(<uintptr_t>ref16.data, ref16.sizes[0], ref16.strides[0] * 2, self.pointer_dtype)
        elif self.pointer_dtype == np.uint32:
            ref32 = sparsePointers32(self._data, d)
            return view_buffer(<uintptr_t>ref32.data, ref32.sizes[0], ref32.strides[0] * 4, self.pointer_dtype)
        elif self.pointer_dtype == np.uint64:
            ref64 = sparsePointers64(self._data, d)
            return view_buffer(<uintptr_t>ref64.data, ref64.sizes[0], ref64.strides[0] * 8, self.pointer_dtype)
        else:
            raise RuntimeError(f'Bad dtype: {self.ptr_dtype}')

    @property
    def pointers(self):
        return tuple([self.get_pointers(i) for i in range(self.ndim)])

    cpdef ndarray get_indices(self, uint64_t d):
        cdef MemRef1DU8 ref8
        cdef MemRef1DU16 ref16
        cdef MemRef1DU32 ref32
        cdef MemRef1DU64 ref64
        if d >= self.ndim:
            raise IndexError(f'Bad dimension index: {d} >= {self.ndim}')
        if self.index_dtype == np.uint8:
            ref8 = sparseIndices8(self._data, d)
            return view_buffer(<uintptr_t>ref8.data, ref8.sizes[0], ref8.strides[0], self.index_dtype)
        elif self.index_dtype == np.uint16:
            ref16 = sparseIndices16(self._data, d)
            return view_buffer(<uintptr_t>ref16.data, ref16.sizes[0], ref16.strides[0] * 2, self.index_dtype)
        elif self.index_dtype == np.uint32:
            ref32 = sparseIndices32(self._data, d)
            return view_buffer(<uintptr_t>ref32.data, ref32.sizes[0], ref32.strides[0] * 4, self.index_dtype)
        elif self.index_dtype == np.uint64:
            ref64 = sparseIndices64(self._data, d)
            return view_buffer(<uintptr_t>ref64.data, ref64.sizes[0], ref64.strides[0] * 8, self.index_dtype)
        else:
            raise RuntimeError(f'Bad dtype: {self.index_dtype}')

    @property
    def indices(self):
        return tuple([self.get_indices(i) for i in range(self.ndim)])

    @property
    def values(self):
        cdef MemRef1DI8 ref8i
        cdef MemRef1DI16 ref16i
        cdef MemRef1DI32 ref32i
        cdef MemRef1DF32 ref32f
        cdef MemRef1DF64 ref64f
        if self.value_dtype == np.int8:
            ref8i = sparseValuesI8(self._data)
            return view_buffer(<uintptr_t>ref8i.data, ref8i.sizes[0], ref8i.strides[0], self.value_dtype)
        elif self.value_dtype == np.int16:
            ref16i = sparseValuesI16(self._data)
            return view_buffer(<uintptr_t>ref16i.data, ref16i.sizes[0], ref16i.strides[0] * 2, self.value_dtype)
        elif self.value_dtype == np.int32:
            ref32i = sparseValuesI32(self._data)
            return view_buffer(<uintptr_t>ref32i.data, ref32i.sizes[0], ref32i.strides[0] * 4, self.value_dtype)
        elif self.value_dtype == np.float32:
            ref32f = sparseValuesF32(self._data)
            return view_buffer(<uintptr_t>ref32f.data, ref32f.sizes[0], ref32f.strides[0] * 4, self.value_dtype)
        elif self.value_dtype == np.float64:
            ref64f = sparseValuesF64(self._data)
            return view_buffer(<uintptr_t>ref64f.data, ref64f.sizes[0], ref64f.strides[0] * 8, self.value_dtype)
            # ALT
            # cdef float64_t[:] view64
            # view64 = <float64_t[:ref64f.sizes[0]]>ref64f.data
            # return np.asarray(view64)
        else:
            raise RuntimeError(f'Bad dtype: {self.value_dtype}')

    cpdef swap_sizes(self, MLIRSparseTensor other):
        swap_sizes(self._data, get_sizes_ptr(other._data))

    cpdef swap_indices(self, MLIRSparseTensor other):
        swap_indices(self._data, get_indices_ptr(other._data))

    cpdef swap_pointers(self, MLIRSparseTensor other):
        swap_pointers(self._data, get_pointers_ptr(other._data))

    cpdef swap_values(self, MLIRSparseTensor other):
        swap_values(self._data, get_values_ptr(other._data))

    cpdef resize_pointers(self, uint64_t d, uint64_t size):
        resize_pointers(self._data, d, size)

    cpdef resize_index(self, uint64_t d, uint64_t size):
        resize_index(self._data, d, size)

    cpdef resize_values(self, uint64_t size):
        resize_values(self._data, size)

    cpdef resize_dim(self, uint64_t d, uint64_t size):
        resize_dim(self._data, d, size)

    cpdef MLIRSparseTensor dup(self):
        cdef MLIRSparseTensor rv = MLIRSparseTensor.__new__(MLIRSparseTensor)  # avoid __init__
        rv._data = dup_tensor(self._data)
        rv.ndim = self.ndim
        rv.pointer_dtype = self.pointer_dtype
        rv.index_dtype = self.index_dtype
        rv.value_dtype = self.value_dtype
        return rv

    cpdef MLIRSparseTensor empty_like(self):
        cdef MLIRSparseTensor rv = MLIRSparseTensor.__new__(MLIRSparseTensor)  # avoid __init__
        rv._data = empty_like(self._data)
        rv.ndim = self.ndim
        rv.pointer_dtype = self.pointer_dtype
        rv.index_dtype = self.index_dtype
        rv.value_dtype = self.value_dtype
        return rv

    cpdef MLIRSparseTensor empty(self, uint64_t ndims):
        cdef MLIRSparseTensor rv = MLIRSparseTensor.__new__(MLIRSparseTensor)  # avoid __init__
        rv._data = empty(self._data, ndims)
        rv.ndim = self.ndim
        rv.pointer_dtype = self.pointer_dtype
        rv.index_dtype = self.index_dtype
        rv.value_dtype = self.value_dtype
        return rv


# Use this to create `vector[vector[uint64_t]*]`, which isn't supported syntax
ctypedef vector[uint64_t]* v_ptr


@cython.boundscheck(False)
@cython.wraparound(False)
def _build_sparse_tensor(
    MLIRSparseTensor self,
    st_index_t[:, :] indices,  # N x D
    st_value_t[:] values,      # N
    uint64_t[:] sizes,         # D
    bool[:] sparsity,          # D
    pointer_type,
):
    cdef intp_t N = values.shape[0]  # number of values
    cdef intp_t D = sizes.shape[0]  # rank, number of dimensions
    self.ndim = D
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
    self.pointer_dtype = np.dtype(pointer_type)
    cdef int pointer_type_num = self.pointer_dtype.num
    if (
        pointer_type_num != np.NPY_UINT8
        and pointer_type_num != np.NPY_UINT16
        and pointer_type_num != np.NPY_UINT32
        and pointer_type_num != np.NPY_UINT64
    ):
        raise TypeError(f"pointer_type must be np.uint8, np.uint16, np.uint32 or np.uint64, not: {pointer_type}")

    cdef vector[uint64_t] sizes_vector = vector[uint64_t](D)
    for i in range(D):
        sizes_vector[i] = sizes[i]
    cdef SparseTensor *tensor = new SparseTensor(sizes_vector, N)

    cdef vector[v_ptr] index_vectors = vector[v_ptr](N)
    cdef v_ptr ind_ptr
    cdef vector[uint64_t] ind
    for i in range(N):
        index_vectors[i] = new vector[uint64_t](D)
        ind = index_vectors[i][0]
        for j in range(D):
            ind[j] = indices[i, j]
        tensor.add(ind, values[i])

    cdef bool *sparsity_array = <bool*>malloc(sizeof(bool) * D)
    for i in range(D):
        sparsity_array[i] = sparsity[i]

    if pointer_type_num == np.NPY_UINT8:
        self._data = (new SparseTensorStorage[uint8_t, st_index_t, st_value_t](tensor, sparsity_array))
    elif pointer_type_num == np.NPY_UINT16:
        self._data = (new SparseTensorStorage[uint16_t, st_index_t, st_value_t](tensor, sparsity_array))
    elif pointer_type_num == np.NPY_UINT32:
        self._data = (new SparseTensorStorage[uint32_t, st_index_t, st_value_t](tensor, sparsity_array))
    else:
        self._data = (new SparseTensorStorage[uint64_t, st_index_t, st_value_t](tensor, sparsity_array))

    free(sparsity_array)
    del tensor
    for i in range(N):
        del index_vectors[i]

    if st_index_t is uint8_t:
        self.index_dtype = np.dtype(np.uint8)
    elif st_index_t is uint16_t:
        self.index_dtype = np.dtype(np.uint16)
    elif st_index_t is uint32_t:
        self.index_dtype = np.dtype(np.uint32)
    else:
        self.index_dtype = np.dtype(np.uint64)

    if st_value_t is int8_t:
        self.value_dtype = np.dtype(np.int8)
    elif st_value_t is int16_t:
        self.value_dtype = np.dtype(np.int16)
    elif st_value_t is int32_t:
        self.value_dtype = np.dtype(np.int32)
    elif st_value_t is float32_t:
        self.value_dtype = np.dtype(np.float32)
    else:
        self.value_dtype = np.dtype(np.float64)
