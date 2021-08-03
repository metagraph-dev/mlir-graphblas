// RUN: graphblas-opt %s -split-input-file -verify-diagnostics

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @vector_accumulate_wrapper(%argA: tensor<8xi64, #SparseVec64>, %argB: tensor<8xi64, #SparseVec64>) -> () {
        graphblas.vector_accumulate %argA, %argB { accumulate_operator = "bad_operator" } : (tensor<8xi64, #SparseVec64>, tensor<8xi64, #SparseVec64>) // expected-error {{"bad_operator" is not a supported accumulate operator.}}
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
        graphblas.vector_accumulate %argA, %argB { accumulate_operator = "plus" } : (tensor<8xi64>, tensor<8xi64, #SparseVec64>) // expected-error {{Operand #0 must be a sparse tensor.}}
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
    func @vector_accumulate_wrapper(%argA: tensor<8xi64, #SparseVec64>, %argB: tensor<8xi64>) -> () {
        graphblas.vector_accumulate %argA, %argB { accumulate_operator = "plus" } : (tensor<8xi64, #SparseVec64>, tensor<8xi64>) // expected-error {{Input vectors must have the same type.}}
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
    func @vector_accumulate_wrapper(%argA: tensor<3xi64, #SparseVec64>, %argB: tensor<8xi64, #SparseVec64>) -> () {
        graphblas.vector_accumulate %argA, %argB { accumulate_operator = "plus" } : (tensor<3xi64, #SparseVec64>, tensor<8xi64, #SparseVec64>) // expected-error {{Input vectors must have compatible shapes.}}
        return
    }
}
