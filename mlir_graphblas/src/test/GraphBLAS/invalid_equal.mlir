// RUN: graphblas-opt %s -split-input-file -verify-diagnostics

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {

   func @vector_equals_wrapper(%argA: tensor<3xi64>, %argB: tensor<3xi64>) -> i1 {
       %answer = graphblas.equal %argA, %argB : tensor<3xi64>, tensor<3xi64> // expected-error {{"a" must be a sparse tensor.}}
       return %answer : i1
   }

}

// -----

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {

   func @vector_equals_wrapper(%argA: tensor<3xi64, #CV64>, %argB: tensor<3xf64, #CV64>) -> i1 {
       %answer = graphblas.equal %argA, %argB : tensor<3xi64, #CV64>, tensor<3xf64, #CV64> // expected-error {{"a" and "b" must have identical types.}}
       return %answer : i1
   }

}

// -----

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {

   func @vector_equals_wrapper(%argA: tensor<3xi64, #CV64>, %argB: tensor<99xi64, #CV64>) -> i1 {
       %answer = graphblas.equal %argA, %argB : tensor<3xi64, #CV64>, tensor<99xi64, #CV64> // expected-error {{"a" and "b" must have identical shapes.}}
       return %answer : i1
   }

}
