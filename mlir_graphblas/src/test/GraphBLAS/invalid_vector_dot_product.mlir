// RUN: graphblas-opt %s -split-input-file -verify-diagnostics

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {

   func @vector_dot_product_wrapper(%argA: tensor<?xi64, #SparseVec64>, %argB: tensor<?xi64, #SparseVec64>) -> i64 {
       %answer = graphblas.matrix_multiply %argA, %argB { semiring = "BAD" } : (tensor<?xi64, #SparseVec64>, tensor<?xi64, #SparseVec64>) to i64 // expected-error {{"BAD" is not a supported semiring add.}}
       return %answer : i64
   }

}

// -----

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {

   func @vector_dot_product_wrapper(%argA: tensor<3xi64>, %argB: tensor<?xi64, #SparseVec64>) -> i64 {
       %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_times" } : (tensor<3xi64>, tensor<?xi64, #SparseVec64>) to i64 // expected-error {{First argument must be a sparse vector or sparse matrix.}}
       return %answer : i64
   }

}

// -----

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {

   func @vector_dot_product_wrapper(%argA: tensor<3xi64, #SparseVec64>, %argB: tensor<3xf64, #SparseVec64>) -> i64 {
       %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_times" } : (tensor<3xi64, #SparseVec64>, tensor<3xf64, #SparseVec64>) to i64 // expected-error {{Operand element types must be identical.}}
       return %answer : i64
   }

}

// -----

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {

   func @vector_dot_product_wrapper(%argA: tensor<3xi64, #SparseVec64>, %argB: tensor<3xi64, #SparseVec64>) -> i8 {
       %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_times" } : (tensor<3xi64, #SparseVec64>, tensor<3xi64, #SparseVec64>) to i8 // expected-error {{Result element type differs from the input element types.}}
       return %answer : i8
   }

}

// -----

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {

   func @vector_dot_product_wrapper(%argA: tensor<3xi64, #SparseVec64>, %argB: tensor<9xi64, #SparseVec64>) -> i64 {
       %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_times" } : (tensor<3xi64, #SparseVec64>, tensor<9xi64, #SparseVec64>) to i64 // expected-error {{Operand shapes are incompatible.}}
       return %answer : i64
   }

}
