// RUN: graphblas-opt %s -split-input-file -verify-diagnostics

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @vector_accumulate_wrapper(%argA: tensor<8xi64, #SparseVec64>, %argB: tensor<8xi64, #SparseVec64>) -> () {
        graphblas.update %argA -> %argB { accumulate_operator = "bad_operator" } : tensor<8xi64, #SparseVec64> -> tensor<8xi64, #SparseVec64> // expected-error {{"bad_operator" is not a supported accumulate operator.}}
        return
    }
}

// -----

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @vector_accumulate_wrapper(%argA: tensor<8xi64>, %argB: tensor<8xi64, #SparseVec64>) -> () {
        graphblas.update %argA -> %argB { accumulate_operator = "plus" } : tensor<8xi64> -> tensor<8xi64, #SparseVec64> // expected-error {{Input argument must be a sparse vector or sparse matrix.}}
    }
}

// -----

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @vector_accumulate_wrapper(%argA: tensor<8xi64, #SparseVec64>, %argB: tensor<8xf64, #SparseVec64>) -> () {
        graphblas.update %argA -> %argB { accumulate_operator = "plus" } : tensor<8xi64, #SparseVec64> -> tensor<8xf64, #SparseVec64> // expected-error {{Arguments must have the same type.}}
    }
}

// -----

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @vector_accumulate_wrapper(%argA: tensor<3xi64, #SparseVec64>, %argB: tensor<8xi64, #SparseVec64>) -> () {
        graphblas.update %argA -> %argB { accumulate_operator = "plus" } : tensor<3xi64, #SparseVec64> -> tensor<8xi64, #SparseVec64> // expected-error {{Input and Output arguments must have compatible shapes.}}
    }
}
