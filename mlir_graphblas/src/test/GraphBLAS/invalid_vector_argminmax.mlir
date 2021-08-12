// RUN: graphblas-opt %s -split-input-file -verify-diagnostics

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {

   func @vector_argminmax_wrapper(%argA: tensor<3xi64>) -> index {
       %answer = graphblas.vector_argminmax %argA { minmax = "max" } : tensor<3xi64> // expected-error {{Operand #0 must be a sparse tensor.}}
       return %answer : index
   }

}

// -----

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {

   func @vector_argmin_wrapper(%argA: tensor<3xi64>) -> index {
       %answer = graphblas.vector_argmin %argA : tensor<3xi64> // expected-error {{Operand #0 must be a sparse tensor.}}
       return %answer : index
   }

}

// -----

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {

   func @vector_argmax_wrapper(%argA: tensor<3xi64>) -> index {
       %answer = graphblas.vector_argmax %argA : tensor<3xi64> // expected-error {{Operand #0 must be a sparse tensor.}}
       return %answer : index
   }

}
