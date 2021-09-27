// RUN: graphblas-opt %s -split-input-file -verify-diagnostics

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @vector_accumulate_wrapper(%argA: tensor<8xi64, #CV64>, %argB: tensor<8xi64, #CV64>) -> () {
        graphblas.update %argA -> %argB { accumulate_operator = "bad_operator" } : tensor<8xi64, #CV64> -> tensor<8xi64, #CV64> // expected-error {{"bad_operator" is not a supported accumulate operator.}}
        return
    }
}

// -----

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @vector_accumulate_wrapper(%argA: tensor<8xi64>, %argB: tensor<8xi64, #CV64>) -> () {
        graphblas.update %argA -> %argB { accumulate_operator = "plus" } : tensor<8xi64> -> tensor<8xi64, #CV64> // expected-error {{input must be a sparse tensor.}}
    }
}

// -----

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @vector_accumulate_wrapper(%argA: tensor<8xi64, #CV64>, %argB: tensor<8xf64, #CV64>) -> () {
        graphblas.update %argA -> %argB { accumulate_operator = "plus" } : tensor<8xi64, #CV64> -> tensor<8xf64, #CV64> // expected-error {{input and output must have identical types.}}
    }
}

// -----

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @vector_accumulate_wrapper(%argA: tensor<3xi64, #CV64>, %argB: tensor<8xi64, #CV64>) -> () {
        graphblas.update %argA -> %argB { accumulate_operator = "plus" } : tensor<3xi64, #CV64> -> tensor<8xi64, #CV64> // expected-error {{Input and Output arguments must have identical shapes.}}
    }
}
