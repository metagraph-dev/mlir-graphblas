//===- SparseUtils.cpp - Sparse Utils for MLIR execution ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a light-weight runtime support library that is useful
// for sparse tensor manipulations. The functionality provided in this library
// is meant to simplify benchmarking, testing, and debugging MLIR code that
// operates on sparse tensors. The provided functionality is **not** part
// of core MLIR, however.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/CRunnerUtils.h"

#ifdef MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

//// -> MODIFIED
// forward declaration
extern "C" {
void *openTensorC(char *filename, uint64_t *idata);
}
//// <- MODIFIED

//===----------------------------------------------------------------------===//
//
// Internal support for storing and reading sparse tensors.
//
// The following memory-resident sparse storage schemes are supported:
//
// (a) A coordinate scheme for temporarily storing and lexicographically
//     sorting a sparse tensor by index.
//
// (b) A "one-size-fits-all" sparse storage scheme defined by per-rank
//     sparse/dense annnotations to be used by generated MLIR code.
//
// The following external formats are supported:
//
// (1) Matrix Market Exchange (MME): *.mtx
//     https://math.nist.gov/MatrixMarket/formats.html
//
// (2) Formidable Repository of Open Sparse Tensors and Tools (FROSTT): *.tns
//     http://frostt.io/tensors/file-formats.html
//
//===----------------------------------------------------------------------===//

namespace {

/// A sparse tensor element in coordinate scheme (value and indices).
/// For example, a rank-1 vector element would look like
///   ({i}, a[i])
/// and a rank-5 tensor element like
///   ({i,j,k,l,m}, a[i,j,k,l,m])
template <typename V>
struct Element {
  Element(const std::vector<uint64_t> &ind, V val) : indices(ind), value(val){};
  std::vector<uint64_t> indices;
  V value;
};

/// A memory-resident sparse tensor in coordinate scheme (collection of
/// elements). This data structure is used to read a sparse tensor from
/// any external format into memory and sort the elements lexicographically
/// by indices before passing it back to the client (most packed storage
/// formats require the elements to appear in lexicographic index order).
template <typename V>
struct SparseTensor {
public:
  SparseTensor(const std::vector<uint64_t> &szs, uint64_t capacity = 0)
      : sizes(szs), pos(0) {
    if (capacity)
      elements.reserve(capacity);
  }
  /// Adds element as indices and value.
  void add(const std::vector<uint64_t> &ind, V val) {
    assert(getRank() == ind.size());
    for (uint64_t r = 0, rank = getRank(); r < rank; r++)
      assert(ind[r] < sizes[r]); // within bounds
    elements.emplace_back(Element<V>(ind, val));
  }
  /// Sorts elements lexicographically by index.
  void sort() { std::sort(elements.begin(), elements.end(), lexOrder); }
  /// Primitive one-time iteration.
  const Element<V> &next() { return elements[pos++]; }
  /// Returns rank.
  uint64_t getRank() const { return sizes.size(); }
  /// Getter for sizes array.
  const std::vector<uint64_t> &getSizes() const { return sizes; }
  /// Getter for elements array.
  const std::vector<Element<V>> &getElements() const { return elements; }

private:
  /// Returns true if indices of e1 < indices of e2.
  static bool lexOrder(const Element<V> &e1, const Element<V> &e2) {
    assert(e1.indices.size() == e2.indices.size());
    for (uint64_t r = 0, rank = e1.indices.size(); r < rank; r++) {
      if (e1.indices[r] == e2.indices[r])
        continue;
      return e1.indices[r] < e2.indices[r];
    }
    return false;
  }
  std::vector<uint64_t> sizes; // per-rank dimension sizes
  std::vector<Element<V>> elements;
  uint64_t pos;
};

/// Abstract base class of sparse tensor storage. Note that we use
/// function overloading to implement "partial" method specialization.
class SparseTensorStorageBase {
public:
  enum DimLevelType : uint8_t { kDense = 0, kCompressed = 1, kSingleton = 2 };

  virtual uint64_t getDimSize(uint64_t) = 0;

  // Overhead storage.
  virtual void getPointers(std::vector<uint64_t> **, uint64_t) { fatal("p64"); }
  virtual void getPointers(std::vector<uint32_t> **, uint64_t) { fatal("p32"); }
  virtual void getPointers(std::vector<uint16_t> **, uint64_t) { fatal("p16"); }
  virtual void getPointers(std::vector<uint8_t> **, uint64_t) { fatal("p8"); }
  virtual void getIndices(std::vector<uint64_t> **, uint64_t) { fatal("i64"); }
  virtual void getIndices(std::vector<uint32_t> **, uint64_t) { fatal("i32"); }
  virtual void getIndices(std::vector<uint16_t> **, uint64_t) { fatal("i16"); }
  virtual void getIndices(std::vector<uint8_t> **, uint64_t) { fatal("i8"); }

  // Primary storage.
  virtual void getValues(std::vector<double> **) { fatal("valf64"); }
  virtual void getValues(std::vector<float> **) { fatal("valf32"); }
  virtual void getValues(std::vector<int64_t> **) { fatal("vali64"); }
  virtual void getValues(std::vector<int32_t> **) { fatal("vali32"); }
  virtual void getValues(std::vector<int16_t> **) { fatal("vali16"); }
  virtual void getValues(std::vector<int8_t> **) { fatal("vali8"); }

  virtual ~SparseTensorStorageBase() {}

    //// -> MODIFIED
    virtual uint64_t getRank() const { return 0; }
    virtual void *get_sizes_ptr() { fatal("get_sizes_ptr"); return 0; }
    virtual void *get_pointers_ptr() { fatal("get_pointers_ptr"); return 0; }
    virtual void *get_indices_ptr() { fatal("get_indices_ptr"); return 0; }
    virtual void *get_values_ptr() { fatal("get_values_ptr"); return 0; }

    virtual void swap_sizes(void *new_sizes) { fatal("swap_sizes"); }
    virtual void swap_pointers(void *new_pointers) { fatal("swap_pointers"); }
    virtual void swap_indices(void *new_indices) { fatal("swap_indices"); }
    virtual void swap_values(void *new_values) { fatal("swap_values"); }

    virtual void resize_pointers(uint64_t d, uint64_t size) { fatal("resize_pointers"); }
    virtual void resize_index(uint64_t d, uint64_t size) { fatal("resize_index"); }
    virtual void resize_values(uint64_t size) { fatal("resize_values"); }
    virtual void resize_dim(uint64_t d, uint64_t size) { fatal("resize_dim"); }

    virtual void *dup() { fatal("dup"); return NULL; }
    virtual void *empty_like() { fatal("empty_like"); return NULL; }
    virtual void *empty(uint64_t ndims) { fatal("empty"); return NULL; }
    //// <- MODIFIED

private:
  void fatal(const char *tp) {
    fprintf(stderr, "unsupported %s\n", tp);
    exit(1);
  }
};

/// A memory-resident sparse tensor using a storage scheme based on per-rank
/// annotations on dense/sparse. This data structure provides a bufferized
/// form of an imaginary SparseTensorType, until such a type becomes a
/// first-class citizen of MLIR. In contrast to generating setup methods for
/// each differently annotated sparse tensor, this method provides a convenient
/// "one-size-fits-all" solution that simply takes an input tensor and
/// annotations to implement all required setup in a general manner.
template <typename P, typename I, typename V>
class SparseTensorStorage : public SparseTensorStorageBase {
public:
  /// Constructs sparse tensor storage scheme following the given
  /// per-rank dimension dense/sparse annotations.
  SparseTensorStorage(SparseTensor<V> *tensor, uint8_t *sparsity)
      : sizes(tensor->getSizes()), pointers(getRank()), indices(getRank()) {
    // Provide hints on capacity.
    // TODO: needs fine-tuning based on sparsity
    uint64_t nnz = tensor->getElements().size();
    values.reserve(nnz);
    for (uint64_t d = 0, s = 1, rank = getRank(); d < rank; d++) {
      s *= sizes[d];
      if (sparsity[d] == kCompressed) {
        pointers[d].reserve(s + 1);
        indices[d].reserve(s);
        s = 1;
      } else {
        assert(sparsity[d] == kDense && "singleton not yet supported");
      }
    }
    // Then setup the tensor.
    traverse(tensor, sparsity, 0, nnz, 0);
  }

  virtual ~SparseTensorStorage() {}

  uint64_t getRank() const override { return sizes.size(); }  //// MODIFIED: Added override

  uint64_t getDimSize(uint64_t d) override { return sizes[d]; }

  // Partially specialize these three methods based on template types.
  void getPointers(std::vector<P> **out, uint64_t d) override {
    *out = &pointers[d];
  }
  void getIndices(std::vector<I> **out, uint64_t d) override {
    *out = &indices[d];
  }
  void getValues(std::vector<V> **out) override { *out = &values; }

  // Factory method.
  static SparseTensorStorage<P, I, V> *newSparseTensor(SparseTensor<V> *t,
                                                       uint8_t *s) {
    t->sort(); // sort lexicographically
    SparseTensorStorage<P, I, V> *n = new SparseTensorStorage<P, I, V>(t, s);
    delete t;
    return n;
  }

private:
  /// Initializes sparse tensor storage scheme from a memory-resident
  /// representation of an external sparse tensor. This method prepares
  /// the pointers and indices arrays under the given per-rank dimension
  /// dense/sparse annotations.
  void traverse(SparseTensor<V> *tensor, uint8_t *sparsity, uint64_t lo,
                uint64_t hi, uint64_t d) {
    const std::vector<Element<V>> &elements = tensor->getElements();
    // Once dimensions are exhausted, insert the numerical values.
    if (d == getRank()) {
      values.push_back(lo < hi ? elements[lo].value : 0);
      return;
    }
    // Prepare a sparse pointer structure at this dimension.
    if (sparsity[d] == kCompressed && pointers[d].empty())
      pointers[d].push_back(0);
    // Visit all elements in this interval.
    uint64_t full = 0;
    while (lo < hi) {
      // Find segment in interval with same index elements in this dimension.
      unsigned idx = elements[lo].indices[d];
      unsigned seg = lo + 1;
      while (seg < hi && elements[seg].indices[d] == idx)
        seg++;
      // Handle segment in interval for sparse or dense dimension.
      if (sparsity[d] == kCompressed) {
        indices[d].push_back(idx);
      } else {
        for (; full < idx; full++)
          traverse(tensor, sparsity, 0, 0, d + 1); // pass empty
        full++;
      }
      traverse(tensor, sparsity, lo, seg, d + 1);
      // And move on to next segment in interval.
      lo = seg;
    }
    // Finalize the sparse pointer structure at this dimension.
    if (sparsity[d] == kCompressed) {
      pointers[d].push_back(indices[d].size());
    } else {
      for (uint64_t sz = tensor->getSizes()[d]; full < sz; full++)
        traverse(tensor, sparsity, 0, 0, d + 1); // pass empty
    }
  }

private:
  std::vector<uint64_t> sizes; // per-rank dimension sizes
  std::vector<std::vector<P>> pointers;
  std::vector<std::vector<I>> indices;
  std::vector<V> values;

//// -> MODIFIED
public:
    /*
    SparseTensorStorage(
        const std::vector<uint64_t> &other_sizes,
        const std::vector<std::vector<P>> &other_pointers,
        const std::vector<std::vector<I>> &other_indices,
        const std::vector<V> &other_values)
    :
        sizes(other_sizes),
        pointers(other_pointers),
        indices(other_indices),
        values(other_values)
    {}

    SparseTensorStorage(const SparseTensorStorage<P, I, V> &other)
    :
        sizes(other.sizes),
        pointers(other.pointers),
        indices(other.indices),
        values(other.values)
    {}
    */

    // Used by `empty_like`
    SparseTensorStorage(const std::vector<uint64_t> &other_sizes, void *other)
        : sizes(other_sizes), pointers(other_sizes.size()), indices(other_sizes.size())
    {
        // Update pointers to have same size as original tensor, but filled with zeros
        SparseTensorStorage<P, I, V> *tensor = static_cast<SparseTensorStorage<P, I, V>*>(other);
        for (uint64_t dim = 0; dim < other_sizes.size(); dim++) {
            pointers[dim].resize(tensor->pointers[dim].size());
        }
    }

    // Used by `empty`
    // Note that `len(pointers[0]) == 0`!
    SparseTensorStorage(uint64_t ndims)
        : sizes(ndims), pointers(ndims), indices(ndims)
    {}

    // Used by `dup`
    SparseTensorStorage(void *other)
    :
        sizes(static_cast<SparseTensorStorage<P, I, V>*>(other)->sizes),
        pointers(static_cast<SparseTensorStorage<P, I, V>*>(other)->pointers),
        indices(static_cast<SparseTensorStorage<P, I, V>*>(other)->indices),
        values(static_cast<SparseTensorStorage<P, I, V>*>(other)->values)
    {}

    SparseTensorStorage(const std::vector<uint64_t> &other_sizes, bool is_sparse)
        : sizes(other_sizes)
    {
        pointers.resize(sizes.size());
        if (is_sparse) {
            pointers[0].resize(2);
        }
        indices.resize(sizes.size());
    }

    void *get_sizes_ptr() override {
        return &sizes;
    }
    void *get_pointers_ptr() override {
        return &pointers;
    }
    void *get_indices_ptr() override {
        return &indices;
    }
    void *get_values_ptr() override {
        return &values;
    }
    void swap_sizes(void *new_sizes) override {
        sizes.swap(*(std::vector<uint64_t>*)new_sizes);
    }
    void swap_pointers(void *new_pointers) override {
        pointers.swap(*(std::vector<std::vector<P>>*)new_pointers);
    }
    void swap_indices(void *new_indices) override {
        indices.swap(*(std::vector<std::vector<I>>*)new_indices);
    }
    void swap_values(void *new_values) override {
        values.swap(*(std::vector<V>*)new_values);
    }
    void resize_pointers(uint64_t d, uint64_t size) override {
        pointers[d].resize(size);
    }
    void resize_index(uint64_t d, uint64_t size) override {
        indices[d].resize(size);
    }
    void resize_values(uint64_t size) override {
        values.resize(size);
    }
    void resize_dim(uint64_t d, uint64_t size) override {
        sizes[d] = size;
    }
    // New tensor of same type with same data
    void *dup() override {
        SparseTensorStorageBase *tensor = new SparseTensorStorage<P, I, V>(this);
        return tensor;
    }
    // New tensor of same type with same shape
    void *empty_like() override {
        SparseTensorStorageBase *tensor = new SparseTensorStorage<P, I, V>(sizes, this);
        return tensor;
    }
    // New tensor of dimensions `ndims` (no shape; must use `resize_dim`)
    void *empty(uint64_t ndims) override {
        SparseTensorStorageBase *tensor = new SparseTensorStorage<P, I, V>(ndims);
        return tensor;
    }
//// <- MODIFIED
};

/// Helper to convert string to lower case.
static char *toLower(char *token) {
  for (char *c = token; *c; c++)
    *c = tolower(*c);
  return token;
}

/// Read the MME header of a general sparse matrix of type real.
static void readMMEHeader(FILE *file, char *name, uint64_t *idata) {
  char line[1025];
  char header[64];
  char object[64];
  char format[64];
  char field[64];
  char symmetry[64];
  // Read header line.
  if (fscanf(file, "%63s %63s %63s %63s %63s\n", header, object, format, field,
             symmetry) != 5) {
    fprintf(stderr, "Corrupt header in %s\n", name);
    exit(1);
  }
  // Make sure this is a general sparse matrix.
  if (strcmp(toLower(header), "%%matrixmarket") ||
      strcmp(toLower(object), "matrix") ||
      strcmp(toLower(format), "coordinate") || strcmp(toLower(field), "real") ||
      strcmp(toLower(symmetry), "general")) {
    fprintf(stderr,
            "Cannot find a general sparse matrix with type real in %s\n", name);
    exit(1);
  }
  // Skip comments.
  while (1) {
    if (!fgets(line, 1025, file)) {
      fprintf(stderr, "Cannot find data in %s\n", name);
      exit(1);
    }
    if (line[0] != '%')
      break;
  }
  // Next line contains M N NNZ.
  idata[0] = 2; // rank
  if (sscanf(line, "%" PRIu64 "%" PRIu64 "%" PRIu64 "\n", idata + 2, idata + 3,
             idata + 1) != 3) {
    fprintf(stderr, "Cannot find size in %s\n", name);
    exit(1);
  }
}

/// Read the "extended" FROSTT header. Although not part of the documented
/// format, we assume that the file starts with optional comments followed
/// by two lines that define the rank, the number of nonzeros, and the
/// dimensions sizes (one per rank) of the sparse tensor.
static void readExtFROSTTHeader(FILE *file, char *name, uint64_t *idata) {
  char line[1025];
  // Skip comments.
  while (1) {
    if (!fgets(line, 1025, file)) {
      fprintf(stderr, "Cannot find data in %s\n", name);
      exit(1);
    }
    if (line[0] != '#')
      break;
  }
  // Next line contains RANK and NNZ.
  if (sscanf(line, "%" PRIu64 "%" PRIu64 "\n", idata, idata + 1) != 2) {
    fprintf(stderr, "Cannot find metadata in %s\n", name);
    exit(1);
  }
  // Followed by a line with the dimension sizes (one per rank).
  for (uint64_t r = 0; r < idata[0]; r++) {
    if (fscanf(file, "%" PRIu64, idata + 2 + r) != 1) {
      fprintf(stderr, "Cannot find dimension size %s\n", name);
      exit(1);
    }
  }
}

/// Reads a sparse tensor with the given filename into a memory-resident
/// sparse tensor in coordinate scheme.
template <typename V>
static SparseTensor<V> *openTensor(char *filename, uint64_t size,
                                   uint64_t *sizes, uint64_t *perm) {
  // Open the file.
  FILE *file = fopen(filename, "r");
  if (!file) {
    fprintf(stderr, "Cannot find %s\n", filename);
    exit(1);
  }
  // Perform some file format dependent set up.
  uint64_t idata[512];
  if (strstr(filename, ".mtx")) {
    readMMEHeader(file, filename, idata);
  } else if (strstr(filename, ".tns")) {
    readExtFROSTTHeader(file, filename, idata);
  } else {
    fprintf(stderr, "Unknown format %s\n", filename);
    exit(1);
  }
  // Prepare sparse tensor object with per-rank dimension sizes
  // and the number of nonzeros as initial capacity.
  assert(size == idata[0] && "rank mismatch");
  uint64_t nnz = idata[1];
  std::vector<uint64_t> indices(size);
  for (uint64_t r = 0; r < size; r++) {
    uint64_t sz = idata[2 + r];
    assert((sizes[r] == 0 || sizes[r] == sz) && "dimension size mismatch");
    indices[perm[r]] = sz;
  }
  SparseTensor<V> *tensor = new SparseTensor<V>(indices, nnz);
  // Read all nonzero elements.
  for (uint64_t k = 0; k < nnz; k++) {
    uint64_t idx = -1;
    for (uint64_t r = 0; r < size; r++) {
      if (fscanf(file, "%" PRIu64, &idx) != 1) {
        fprintf(stderr, "Cannot find next index in %s\n", filename);
        exit(1);
      }
      // Add 0-based index.
      indices[perm[r]] = idx - 1;
    }
    // The external formats always store the numerical values with the type
    // double, but we cast these values to the sparse tensor object type.
    double value;
    if (fscanf(file, "%lg\n", &value) != 1) {
      fprintf(stderr, "Cannot find next value in %s\n", filename);
      exit(1);
    }
    tensor->add(indices, value);
  }
  // Close the file and return tensor.
  fclose(file);
  return tensor;
}

/// Helper to copy a linearized dense tensor.
template <typename V>
static V *copyTensorTraverse(SparseTensor<V> *tensor,
                             std::vector<uint64_t> &indices, uint64_t r,
                             uint64_t rank, uint64_t *sizes, uint64_t *perm,
                             V *data) {
  for (uint64_t i = 0, sz = sizes[r]; i < sz; i++) {
    indices[perm[r]] = i;
    if (r + 1 == rank) {
      V d = *data++;
      if (d)
        tensor->add(indices, d);
    } else {
      data =
          copyTensorTraverse(tensor, indices, r + 1, rank, sizes, perm, data);
    }
  }
  return data;
}

/// Copies the nonzeros of a linearized dense tensor into a memory-resident
/// sparse tensor in coordinate scheme.
template <typename V>
static SparseTensor<V> *copyTensor(uint64_t size, uint64_t *sizes,
                                   uint64_t *perm, V *data) {
  std::vector<uint64_t> indices(size);
  for (uint64_t r = 0; r < size; r++)
    indices[perm[r]] = sizes[r];
  SparseTensor<V> *tensor = new SparseTensor<V>(indices);
  copyTensorTraverse<V>(tensor, indices, 0, size, sizes, perm, data);
  return tensor;
}

} // anonymous namespace

extern "C" {

/// Helper method to read a sparse tensor filename from the environment,
/// defined with the naming convention ${TENSOR0}, ${TENSOR1}, etc.
char *getTensorFilename(uint64_t id) {
  char var[80];
  sprintf(var, "TENSOR%" PRIu64, id);
  char *env = getenv(var);
  return env;
}

//===----------------------------------------------------------------------===//
//
// Public API of the sparse runtime support library that support an opaque
// implementation of a bufferized SparseTensor in MLIR. This could be replaced
// by actual codegen in MLIR.
//
// Because we cannot use C++ templates with C linkage, some macro magic is used
// to generate implementations for all required type combinations that can be
// called from MLIR generated code.
//
//===----------------------------------------------------------------------===//

struct UnrankedMemRef {
  uint64_t rank;
  void *descriptor;
};

#define TEMPLATE(NAME, TYPE)                                                   \
  struct NAME {                                                                \
    const TYPE *base;                                                          \
    const TYPE *data;                                                          \
    uint64_t off;                                                              \
    uint64_t sizes[1];                                                         \
    uint64_t strides[1];                                                       \
  }

#define CASE(p, i, v, P, I, V)                                                 \
  if (ptrTp == (p) && indTp == (i) && valTp == (v)) {                          \
    SparseTensor<V> *tensor;                                                   \
    if (action == 0)                                                           \
      tensor = openTensor<V>(static_cast<char *>(ptr), asize, sizes, perm);    \
    else                                                                       \
      tensor = copyTensor<V>(asize, sizes, perm, static_cast<V *>(ptr));       \
    return SparseTensorStorage<P, I, V>::newSparseTensor(tensor, sparsity);    \
  }

#define IMPL1(RET, NAME, TYPE, LIB)                                            \
  RET NAME(void *tensor) {                                                     \
    std::vector<TYPE> *v;                                                      \
    static_cast<SparseTensorStorageBase *>(tensor)->LIB(&v);                   \
    return {v->data(), v->data(), 0, {v->size()}, {1}};                        \
  }

#define IMPL2(RET, NAME, TYPE, LIB)                                            \
  RET NAME(void *tensor, uint64_t d) {                                         \
    std::vector<TYPE> *v;                                                      \
    static_cast<SparseTensorStorageBase *>(tensor)->LIB(&v, d);                \
    return {v->data(), v->data(), 0, {v->size()}, {1}};                        \
  }

#define PTR(NAME)                                                              \
  const void *NAME(int64_t sz, UnrankedMemRef *m) { return m->descriptor; }

TEMPLATE(MemRef1DU64, uint64_t);
TEMPLATE(MemRef1DU32, uint32_t);
TEMPLATE(MemRef1DU16, uint16_t);
TEMPLATE(MemRef1DU8, uint8_t);
TEMPLATE(MemRef1DI64, int64_t);
TEMPLATE(MemRef1DI32, int32_t);
TEMPLATE(MemRef1DI16, int16_t);
TEMPLATE(MemRef1DI8, int8_t);
TEMPLATE(MemRef1DF64, double);
TEMPLATE(MemRef1DF32, float);

enum OverheadTypeEnum : uint64_t { kU64 = 1, kU32 = 2, kU16 = 3, kU8 = 4 };

enum PrimaryTypeEnum : uint64_t {
  kF64 = 1,
  kF32 = 2,
  kI64 = 3,
  kI32 = 4,
  kI16 = 5,
  kI8 = 6
};

/// Constructs a new sparse tensor. This is the "swiss army knife"
/// method for materializing sparse tensors into the computation.
void *newSparseTensor(uint8_t *abase, uint8_t *adata, uint64_t aoff,
                      uint64_t asize, uint64_t astride, uint64_t *sbase,
                      uint64_t *sdata, uint64_t soff, uint64_t ssize,
                      uint64_t sstride, uint64_t *pbase, uint64_t *pdata,
                      uint64_t poff, uint64_t psize, uint64_t pstride,
                      uint64_t ptrTp, uint64_t indTp, uint64_t valTp,
                      uint32_t action, void *ptr) {
  assert(astride == 1 && sstride == 1 && pstride == 1);
  uint8_t *sparsity = adata + aoff;
  uint64_t *sizes = sdata + soff;
  uint64_t *perm = pdata + poff;

  // Double matrices with all combinations of overhead storage.
  CASE(kU64, kU64, kF64, uint64_t, uint64_t, double);
  CASE(kU64, kU32, kF64, uint64_t, uint32_t, double);
  CASE(kU64, kU16, kF64, uint64_t, uint16_t, double);
  CASE(kU64, kU8, kF64, uint64_t, uint8_t, double);
  CASE(kU32, kU64, kF64, uint32_t, uint64_t, double);
  CASE(kU32, kU32, kF64, uint32_t, uint32_t, double);
  CASE(kU32, kU16, kF64, uint32_t, uint16_t, double);
  CASE(kU32, kU8, kF64, uint32_t, uint8_t, double);
  CASE(kU16, kU64, kF64, uint16_t, uint64_t, double);
  CASE(kU16, kU32, kF64, uint16_t, uint32_t, double);
  CASE(kU16, kU16, kF64, uint16_t, uint16_t, double);
  CASE(kU16, kU8, kF64, uint16_t, uint8_t, double);
  CASE(kU8, kU64, kF64, uint8_t, uint64_t, double);
  CASE(kU8, kU32, kF64, uint8_t, uint32_t, double);
  CASE(kU8, kU16, kF64, uint8_t, uint16_t, double);
  CASE(kU8, kU8, kF64, uint8_t, uint8_t, double);

  // Float matrices with all combinations of overhead storage.
  CASE(kU64, kU64, kF32, uint64_t, uint64_t, float);
  CASE(kU64, kU32, kF32, uint64_t, uint32_t, float);
  CASE(kU64, kU16, kF32, uint64_t, uint16_t, float);
  CASE(kU64, kU8, kF32, uint64_t, uint8_t, float);
  CASE(kU32, kU64, kF32, uint32_t, uint64_t, float);
  CASE(kU32, kU32, kF32, uint32_t, uint32_t, float);
  CASE(kU32, kU16, kF32, uint32_t, uint16_t, float);
  CASE(kU32, kU8, kF32, uint32_t, uint8_t, float);
  CASE(kU16, kU64, kF32, uint16_t, uint64_t, float);
  CASE(kU16, kU32, kF32, uint16_t, uint32_t, float);
  CASE(kU16, kU16, kF32, uint16_t, uint16_t, float);
  CASE(kU16, kU8, kF32, uint16_t, uint8_t, float);
  CASE(kU8, kU64, kF32, uint8_t, uint64_t, float);
  CASE(kU8, kU32, kF32, uint8_t, uint32_t, float);
  CASE(kU8, kU16, kF32, uint8_t, uint16_t, float);
  CASE(kU8, kU8, kF32, uint8_t, uint8_t, float);

  // Integral matrices with same overhead storage.
  CASE(kU64, kU64, kI64, uint64_t, uint64_t, int64_t);
  CASE(kU64, kU64, kI32, uint64_t, uint64_t, int32_t);
  CASE(kU64, kU64, kI16, uint64_t, uint64_t, int16_t);
  CASE(kU64, kU64, kI8, uint64_t, uint64_t, int8_t);
  CASE(kU32, kU32, kI32, uint32_t, uint32_t, int32_t);
  CASE(kU32, kU32, kI16, uint32_t, uint32_t, int16_t);
  CASE(kU32, kU32, kI8, uint32_t, uint32_t, int8_t);
  CASE(kU16, kU16, kI32, uint16_t, uint16_t, int32_t);
  CASE(kU16, kU16, kI16, uint16_t, uint16_t, int16_t);
  CASE(kU16, kU16, kI8, uint16_t, uint16_t, int8_t);
  CASE(kU8, kU8, kI32, uint8_t, uint8_t, int32_t);
  CASE(kU8, kU8, kI16, uint8_t, uint8_t, int16_t);
  CASE(kU8, kU8, kI8, uint8_t, uint8_t, int8_t);

  // Unsupported case (add above if needed).
  fputs("unsupported combination of types\n", stderr);
  exit(1);
}

/// Returns size of sparse tensor in given dimension.
uint64_t sparseDimSize(void *tensor, uint64_t d) {
  return static_cast<SparseTensorStorageBase *>(tensor)->getDimSize(d);
}

/// Methods that provide direct access to pointers, indices, and values.
IMPL2(MemRef1DU64, sparsePointers, uint64_t, getPointers)
IMPL2(MemRef1DU64, sparsePointers64, uint64_t, getPointers)
IMPL2(MemRef1DU32, sparsePointers32, uint32_t, getPointers)
IMPL2(MemRef1DU16, sparsePointers16, uint16_t, getPointers)
IMPL2(MemRef1DU8, sparsePointers8, uint8_t, getPointers)
IMPL2(MemRef1DU64, sparseIndices, uint64_t, getIndices)
IMPL2(MemRef1DU64, sparseIndices64, uint64_t, getIndices)
IMPL2(MemRef1DU32, sparseIndices32, uint32_t, getIndices)
IMPL2(MemRef1DU16, sparseIndices16, uint16_t, getIndices)
IMPL2(MemRef1DU8, sparseIndices8, uint8_t, getIndices)
IMPL1(MemRef1DF64, sparseValuesF64, double, getValues)
IMPL1(MemRef1DF32, sparseValuesF32, float, getValues)
IMPL1(MemRef1DI64, sparseValuesI64, int64_t, getValues)
IMPL1(MemRef1DI32, sparseValuesI32, int32_t, getValues)
IMPL1(MemRef1DI16, sparseValuesI16, int16_t, getValues)
IMPL1(MemRef1DI8, sparseValuesI8, int8_t, getValues)

/// Releases sparse tensor storage.
void delSparseTensor(void *tensor)
{
  delete static_cast<SparseTensorStorageBase *>(tensor);
}

/// Helper to get pointer, one per value type.
PTR(getPtrF64)
PTR(getPtrF32)
PTR(getPtrI64)
PTR(getPtrI32)
PTR(getPtrI16)
PTR(getPtrI8)

#undef TEMPLATE
#undef CASE
#undef IMPL1
#undef IMPL2

//// --> MODIFIED
uint64_t get_rank(void *tensor) {
    return static_cast<SparseTensorStorageBase *>(tensor)->getRank();
}
void *get_sizes_ptr(void *tensor) {
    return static_cast<SparseTensorStorageBase *>(tensor)->get_sizes_ptr();
}
void *get_pointers_ptr(void *tensor) {
    return static_cast<SparseTensorStorageBase *>(tensor)->get_pointers_ptr();
}
void *get_indices_ptr(void *tensor) {
    return static_cast<SparseTensorStorageBase *>(tensor)->get_indices_ptr();
}
void *get_values_ptr(void *tensor) {
    return static_cast<SparseTensorStorageBase *>(tensor)->get_values_ptr();
}
void swap_sizes(void *tensor, void *new_sizes) {
    static_cast<SparseTensorStorageBase *>(tensor)->swap_sizes(new_sizes);
}
void swap_pointers(void *tensor, void *new_pointers) {
    static_cast<SparseTensorStorageBase *>(tensor)->swap_pointers(new_pointers);
}
void swap_indices(void *tensor, void *new_indices) {
    static_cast<SparseTensorStorageBase *>(tensor)->swap_indices(new_indices);
}
void swap_values(void *tensor, void *new_values) {
    static_cast<SparseTensorStorageBase *>(tensor)->swap_values(new_values);
}
void resize_pointers(void *tensor, uint64_t d, uint64_t size) {
    static_cast<SparseTensorStorageBase *>(tensor)->resize_pointers(d, size);
}
void resize_index(void *tensor, uint64_t d, uint64_t size) {
    static_cast<SparseTensorStorageBase *>(tensor)->resize_index(d, size);
}
void resize_values(void *tensor, uint64_t size) {
    static_cast<SparseTensorStorageBase *>(tensor)->resize_values(size);
}
void resize_dim(void *tensor, uint64_t d, uint64_t size) {
    static_cast<SparseTensorStorageBase *>(tensor)->resize_dim(d, size);
}
void *dup_tensor(void *tensor) {
    return static_cast<SparseTensorStorageBase *>(tensor)->dup();
}
void *empty_like(void *tensor) {
    return static_cast<SparseTensorStorageBase *>(tensor)->empty_like();
}
void *empty(void *tensor, uint64_t ndims) {
    return static_cast<SparseTensorStorageBase *>(tensor)->empty(ndims);
}
// Combinations of real types to !llvm.ptr<i8>
void *matrix_csr_f64_p64i64_to_ptr8(void *tensor) {
    return tensor;
}
void *matrix_csc_f64_p64i64_to_ptr8(void *tensor) {
    return tensor;
}
void *matrix_csr_f32_p64i64_to_ptr8(void *tensor) {
    return tensor;
}
void *matrix_csc_f32_p64i64_to_ptr8(void *tensor) {
    return tensor;
}
void *matrix_csr_i64_p64i64_to_ptr8(void *tensor) {
    return tensor;
}
void *matrix_csc_i64_p64i64_to_ptr8(void *tensor) {
    return tensor;
}
void *vector_f64_p64i64_to_ptr8(void *tensor) {
    return tensor;
}
void *vector_f32_p64i64_to_ptr8(void *tensor) {
    return tensor;
}
void *vector_i64_p64i64_to_ptr8(void *tensor) {
    return tensor;
}
// Combinations of !llvm.ptr<i8> to real types
void *ptr8_to_matrix_csr_f64_p64i64(void *tensor) {
    return tensor;
}
void *ptr8_to_matrix_csc_f64_p64i64(void *tensor) {
    return tensor;
}
void *ptr8_to_matrix_csr_f32_p64i64(void *tensor) {
    return tensor;
}
void *ptr8_to_matrix_csc_f32_p64i64(void *tensor) {
    return tensor;
}
void *ptr8_to_matrix_csr_i64_p64i64(void *tensor) {
    return tensor;
}
void *ptr8_to_matrix_csc_i64_p64i64(void *tensor) {
    return tensor;
}
void *ptr8_to_vector_f64_p64i64(void *tensor) {
    return tensor;
}
void *ptr8_to_vector_f32_p64i64(void *tensor) {
    return tensor;
}
void *ptr8_to_vector_i64_p64i64(void *tensor) {
    return tensor;
}
// New tensor generic constructors
void *matrix_prep_size(SparseTensorStorageBase *matrix, uint64_t nrows, uint64_t ncols, bool columnOriented) {
    matrix->resize_dim(0, nrows);
    matrix->resize_dim(1, ncols);
    matrix->resize_pointers(1, (columnOriented ? ncols : nrows) + 1);
    return matrix;
}
void *vector_prep_size(SparseTensorStorageBase *vector, uint64_t size) {
    vector->resize_dim(0, size);
    vector->resize_pointers(0, 2);
    return vector;
}
// New matrix specialized constructors
void *new_matrix_csr_f64_p64i64(uint64_t nrows, uint64_t ncols) {
    SparseTensorStorageBase *tensor = new SparseTensorStorage<uint64_t, uint64_t, double>(2);
    return matrix_prep_size(tensor, nrows, ncols, false);
}
void *new_matrix_csc_f64_p64i64(uint64_t nrows, uint64_t ncols) {
    SparseTensorStorageBase *tensor = new SparseTensorStorage<uint64_t, uint64_t, double>(2);
    return matrix_prep_size(tensor, nrows, ncols, true);
}
void *new_matrix_csr_f32_p64i64(uint64_t nrows, uint64_t ncols) {
    SparseTensorStorageBase *tensor = new SparseTensorStorage<uint64_t, uint64_t, float>(2);
    return matrix_prep_size(tensor, nrows, ncols, false);
}
void *new_matrix_csc_f32_p64i64(uint64_t nrows, uint64_t ncols) {
    SparseTensorStorageBase *tensor = new SparseTensorStorage<uint64_t, uint64_t, float>(2);
    return matrix_prep_size(tensor, nrows, ncols, true);
}
void *new_matrix_csr_i64_p64i64(uint64_t nrows, uint64_t ncols) {
    SparseTensorStorageBase *tensor = new SparseTensorStorage<uint64_t, uint64_t, int64_t>(2);
    return matrix_prep_size(tensor, nrows, ncols, false);
}
void *new_matrix_csc_i64_p64i64(uint64_t nrows, uint64_t ncols) {
    SparseTensorStorageBase *tensor = new SparseTensorStorage<uint64_t, uint64_t, int64_t>(2);
    return matrix_prep_size(tensor, nrows, ncols, true);
}
// New vector specialized constructors
void *new_vector_f64_p64i64(uint64_t size) {
    SparseTensorStorageBase *tensor = new SparseTensorStorage<uint64_t, uint64_t, double>(1);
    return vector_prep_size(tensor, size);
}
void *new_vector_f32_p64i64(uint64_t size) {
    SparseTensorStorageBase *tensor = new SparseTensorStorage<uint64_t, uint64_t, float>(1);
    return vector_prep_size(tensor, size);
}
void *new_vector_i64_p64i64(uint64_t size) {
    SparseTensorStorageBase *tensor = new SparseTensorStorage<uint64_t, uint64_t, int64_t>(1);
    return vector_prep_size(tensor, size);
}

//// <- MODIFIED
} // extern "C"

//// -> MODIFIED
extern "C" MLIR_CRUNNERUTILS_EXPORT void
memrefCopy(int64_t elemSize, UnrankedMemRefType<char> *srcArg,
           UnrankedMemRefType<char> *dstArg)
{
  DynamicMemRefType<char> src(*srcArg);
  DynamicMemRefType<char> dst(*dstArg);

  int64_t rank = src.rank;
  // Handle empty shapes -> nothing to copy.
  for (int rankp = 0; rankp < rank; ++rankp)
    if (src.sizes[rankp] == 0)
      return;

  char *srcPtr = src.data + src.offset * elemSize;
  char *dstPtr = dst.data + dst.offset * elemSize;

  if (rank == 0)
  {
    memcpy(dstPtr, srcPtr, elemSize);
    return;
  }

  int64_t *indices = static_cast<int64_t *>(alloca(sizeof(int64_t) * rank));
  int64_t *srcStrides = static_cast<int64_t *>(alloca(sizeof(int64_t) * rank));
  int64_t *dstStrides = static_cast<int64_t *>(alloca(sizeof(int64_t) * rank));

  // Initialize index and scale strides.
  for (int rankp = 0; rankp < rank; ++rankp)
  {
    indices[rankp] = 0;
    srcStrides[rankp] = src.strides[rankp] * elemSize;
    dstStrides[rankp] = dst.strides[rankp] * elemSize;
  }

  int64_t readIndex = 0, writeIndex = 0;
  for (;;)
  {
    // Copy over the element, byte by byte.
    memcpy(dstPtr + writeIndex, srcPtr + readIndex, elemSize);
    // Advance index and read position.
    for (int64_t axis = rank - 1; axis >= 0; --axis)
    {
      // Advance at current axis.
      auto newIndex = ++indices[axis];
      readIndex += srcStrides[axis];
      writeIndex += dstStrides[axis];
      // If this is a valid index, we have our next index, so continue copying.
      if (src.sizes[axis] != newIndex)
        break;
      // We reached the end of this axis. If this is axis 0, we are done.
      if (axis == 0)
        return;
      // Else, reset to 0 and undo the advancement of the linear index that
      // this axis had. Then continue with the axis one outer.
      indices[axis] = 0;
      readIndex -= src.sizes[axis] * srcStrides[axis];
      writeIndex -= dst.sizes[axis] * dstStrides[axis];
    }
  }
}
//// <- MODIFIED

#endif // MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS
