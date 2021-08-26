// RUN: graphblas-opt %s -split-input-file -verify-diagnostics

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @matrix_reduce_to_vector_wrapper(%matrix: tensor<*xi32>) -> tensor<9xi32, #SparseVec64> {
        %vec = graphblas.matrix_reduce_to_vector %matrix { aggregator = "sum", axis = 0 } : tensor<*xi32> to tensor<9xi32, #SparseVec64> // expected-error {{Operand #0 must be a sparse tensor.}}
        return %vec : tensor<9xi32, #SparseVec64>
    }
}

// -----

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @matrix_reduce_to_vector_wrapper(%matrix: tensor<?x?xi32>) -> tensor<9xi32, #SparseVec64> {
        %vec = graphblas.matrix_reduce_to_vector %matrix { aggregator = "sum", axis = 0 } : tensor<?x?xi32> to tensor<9xi32, #SparseVec64> // expected-error {{Operand #0 must be a sparse tensor.}}
        return %vec : tensor<9xi32, #SparseVec64>
    }
}

// -----

#BADENCODING = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed", "dense" ],
  dimOrdering = affine_map<(i,j,k) -> (i,j,k)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @matrix_reduce_to_vector_wrapper(%matrix: tensor<7x9x1xi32, #BADENCODING>) -> tensor<9xi32, #SparseVec64> {
        %vec = graphblas.matrix_reduce_to_vector %matrix { aggregator = "sum", axis = 0 } : tensor<7x9x1xi32, #BADENCODING> to tensor<9xi32, #SparseVec64> // expected-error {{Operand #0 must have rank 2.}}
        return %vec : tensor<9xi32, #SparseVec64>
    }
}

// -----

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @matrix_reduce_to_vector_wrapper(%matrix: tensor<7x9xi32, #CSR64>) -> tensor<7x9xi32, #CSR64> {
        %vec = graphblas.matrix_reduce_to_vector %matrix { aggregator = "sum", axis = 0 } : tensor<7x9xi32, #CSR64> to tensor<7x9xi32, #CSR64> // expected-error {{Return value must have rank 1.}}
        return %vec : tensor<7x9xi32, #CSR64>
    }
}

// -----

// COM: TODO when https://github.com/metagraph-dev/mlir-graphblas/issues/66 is complete, try alll sorts of bad values for dimLevelType in the bad encoding 

#BADENCODING = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @matrix_reduce_to_vector_wrapper(%matrix: tensor<7x9xi32, #BADENCODING>) -> tensor<9xi32, #SparseVec64> {
        %vec = graphblas.matrix_reduce_to_vector %matrix { aggregator = "sum", axis = 0 } : tensor<7x9xi32, #BADENCODING> to tensor<9xi32, #SparseVec64> // expected-error {{Operand #0 must have CSR or CSC compression, i.e. must have dimLevelType = [ "dense", "compressed" ] in the sparse encoding.}}
        return %vec : tensor<9xi32, #SparseVec64>
    }
}

// -----

// COM: TODO when https://github.com/metagraph-dev/mlir-graphblas/issues/66 is complete, try alll sorts of bad values for dimLevelType in the bad encoding 

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#BADENCODING = #sparse_tensor.encoding<{
  dimLevelType = [ "singleton" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @matrix_reduce_to_vector_wrapper(%matrix: tensor<7x9xi32, #CSR64>) -> tensor<9xi32, #BADENCODING> {
        %vec = graphblas.matrix_reduce_to_vector %matrix { aggregator = "sum", axis = 0 } : tensor<7x9xi32, #CSR64> to tensor<9xi32, #BADENCODING> // expected-error {{Return value must be sparse, i.e. must have dimLevelType = [ "compressed" ] in the sparse encoding.}}
        return %vec : tensor<9xi32, #BADENCODING>
    }
}

// -----

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @matrix_reduce_to_vector_wrapper(%matrix: tensor<7x9xi32, #CSR64>) -> tensor<9xi32, #SparseVec64> {
        %vec = graphblas.matrix_reduce_to_vector %matrix { aggregator = "bad_reducer", axis = 0 } : tensor<7x9xi32, #CSR64> to tensor<9xi32, #SparseVec64> // expected-error {{"bad_reducer" is not a supported aggregator.}}
        return %vec : tensor<9xi32, #SparseVec64>
    }
}

// -----

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @matrix_reduce_to_vector_wrapper(%matrix: tensor<7x9xf64, #CSR64>) -> tensor<9xi32, #SparseVec64> {
        %vec = graphblas.matrix_reduce_to_vector %matrix { aggregator = "sum", axis = 0 } : tensor<7x9xf64, #CSR64> to tensor<9xi32, #SparseVec64> // expected-error {{Operand and output types are incompatible.}}
        return %vec : tensor<9xi32, #SparseVec64>
    }
}

// -----

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @matrix_reduce_to_vector_wrapper(%matrix: tensor<7x9xi32, #CSR64>) -> tensor<9xi32, #SparseVec64> {
        %vec = graphblas.matrix_reduce_to_vector %matrix { aggregator = "sum", axis = 3 } : tensor<7x9xi32, #CSR64> to tensor<9xi32, #SparseVec64> // expected-error {{The axis attribute is expected to be 0 or 1.}}
        return %vec : tensor<9xi32, #SparseVec64>
    }
}

// -----

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @matrix_reduce_to_vector_wrapper(%matrix: tensor<7x5xi32, #CSR64>) -> tensor<9xi32, #SparseVec64> {
        %vec = graphblas.matrix_reduce_to_vector %matrix { aggregator = "sum", axis = 0 } : tensor<7x5xi32, #CSR64> to tensor<9xi32, #SparseVec64> // expected-error {{Operand and output shapes are incompatible.}}
        return %vec : tensor<9xi32, #SparseVec64>
    }
}
