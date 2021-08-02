// RUN: graphblas-opt %s -split-input-file -verify-diagnostics

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {

   func @vector_equals_wrapper(%argA: tensor<3xi64>, %argB: tensor<3xi64>) -> i1 {
       %answer = graphblas.equal %argA, %argB : (tensor<3xi64>, tensor<3xi64>) // expected-error {{First argument must be a sparse vector or sparse matrix.}}
       return %answer : i1
   }

}

// -----

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {

   func @vector_equals_wrapper(%argA: tensor<3xi64, #SparseVec64>, %argB: tensor<3xf64, #SparseVec64>) -> i1 {
       %answer = graphblas.equal %argA, %argB : (tensor<3xi64, #SparseVec64>, tensor<3xf64, #SparseVec64>) // expected-error {{Arguments must have the same type.}}
       return %answer : i1
   }

}

// -----

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {

   func @vector_equals_wrapper(%argA: tensor<3xi64, #SparseVec64>, %argB: tensor<99xi64, #SparseVec64>) -> i1 {
       %answer = graphblas.equal %argA, %argB : (tensor<3xi64, #SparseVec64>, tensor<99xi64, #SparseVec64>) // expected-error {{Input vectors must have compatible shapes.}}
       return %answer : i1
   }

}
