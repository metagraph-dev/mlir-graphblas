// RUN: graphblas-opt %s -split-input-file -verify-diagnostics

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {

   func @vector_argmin_wrapper(%argA: tensor<3xi64, #CV64>) -> i64 {
       %answer = graphblas.reduce_to_scalar %argA { aggregator = "bogus" } : tensor<3xi64, #CV64> to i64 // expected-error {{"bogus" is not a supported aggregator.}}
       return %answer : i64
   }

}

// -----

module {

   func @vector_argmax_wrapper(%argA: tensor<3xi64>) -> i64 {
       %answer = graphblas.reduce_to_scalar %argA { aggregator = "argmax" } : tensor<3xi64> to i64 // expected-error {{operand must be a sparse tensor.}}
       return %answer : i64
   }

}

