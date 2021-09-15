// RUN: graphblas-opt %s -split-input-file -verify-diagnostics

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {

   func @vector_argminmax_wrapper(%argA: tensor<3xi64, #SparseVec64>) -> index {
       %answer = graphblas.vector_argminmax %argA { minmax = "bogus" } : tensor<3xi64, #SparseVec64> // expected-error {{The minmax attribute is expected to be "min" or "max"; got "bogus" instead.}}
       return %answer : index
   }

}

// -----

module {

   func @vector_argminmax_wrapper(%argA: tensor<3xi64>) -> index {
       %answer = graphblas.vector_argminmax %argA { minmax = "max" } : tensor<3xi64> // expected-error {{operand must be a sparse tensor.}}
       return %answer : index
   }

}

// -----

module {

   func @vector_argmin_wrapper(%argA: tensor<3xi64>) -> index {
       %answer = graphblas.vector_argmin %argA : tensor<3xi64> // expected-error {{operand must be a sparse tensor.}}
       return %answer : index
   }

}

// -----

module {

   func @vector_argmax_wrapper(%argA: tensor<3xi64>) -> index {
       %answer = graphblas.vector_argmax %argA : tensor<3xi64> // expected-error {{operand must be a sparse tensor.}}
       return %answer : index
   }

}
