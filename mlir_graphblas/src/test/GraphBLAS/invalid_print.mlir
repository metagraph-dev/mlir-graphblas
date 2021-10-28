// RUN: graphblas-opt %s -split-input-file -verify-diagnostics

#BadSparseEncoding = #sparse_tensor.encoding<{
  dimLevelType = [ "dense" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @printer_func(%tensor: tensor<?xi64, #BadSparseEncoding>) {
        graphblas.print %tensor { strings = ["printed : "] } : tensor<?xi64, #BadSparseEncoding> // expected-error {{Vectors mmust be dense or sparse.}}
        return
    }
}

// -----

#BadSparseEncoding = #sparse_tensor.encoding<{
  dimLevelType = [ "singleton" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @printer_func(%tensor: tensor<?xi64, #BadSparseEncoding>) {
        graphblas.print %tensor { strings = ["printed : "] } : tensor<?xi64, #BadSparseEncoding> // expected-error {{Vectors mmust be dense or sparse.}}
        return
    }
}

// -----

#SparseEncoding = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @printer_func(%tensor: tensor<?x?xi64, #SparseEncoding>) {
        graphblas.print %tensor { strings = ["printed : "] } : tensor<?x?xi64, #SparseEncoding> // expected-error {{must have CSR or CSC compression, i.e. must have dimLevelType = [ "dense", "compressed" ] in the sparse encoding.}}
        return
    }
}

// -----

#SparseEncoding = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j,k) -> (i,j,k)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @printer_func(%tensor: tensor<?x?x?xi64, #SparseEncoding>) {
        graphblas.print %tensor { strings = ["printed : "] } : tensor<?x?x?xi64, #SparseEncoding> // expected-error {{Can only print sparse tensors with rank 1 or 2.}}
        return
    }
}
