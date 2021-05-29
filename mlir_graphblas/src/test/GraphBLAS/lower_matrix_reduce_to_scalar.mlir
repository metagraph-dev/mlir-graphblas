// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

// COM: graphblas.matrix_reduce_to_scalar currently lowers into a function call of an automatically generated private function.
// COM: These tests do not test the contents of these automatically generated private functions.
// COM: Testing for these automatically generated private functions should happen elsewhere where we verify their behavior as well.

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    // CHECK: func private @[[HELPER_FUNCTION:.*]](%[[ARG0:.*]]: [[CSR_TYPE:tensor<.*->.*>]]) -> [[RETURN_TYPE:.*]] {
    // CHECK: func @matrix_reduce_to_scalar(%[[ARG0:.*]]: [[CSR_TYPE]]) -> [[RETURN_TYPE]] {
	// CHECK-NEXT: %[[ANSWER:.*]] = call @[[HELPER_FUNCTION]](%[[ARG0]]) : ([[CSR_TYPE]]) -> [[RETURN_TYPE]]
        // COM: CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
    func @matrix_reduce_to_scalar(%sparse_tensor: tensor<?x?xf64, #CSR64>) -> f64 {
        %answer = graphblas.matrix_reduce_to_scalar %sparse_tensor { aggregator = "sum" } : tensor<?x?xf64, #CSR64> to f64
        return %answer : f64
    }
            
}

