// RUN: graphblas-opt %s | graphblas-opt | FileCheck %s

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#CSC64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "dense" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    
    // CHECK: func @transpose_wrapper_no_swap(%[[ARG0:.*]]: [[CSR_TYPE:.*]]) -> [[CSC_TYPE:.*]] {
    func @transpose_wrapper_no_swap(%sparse_tensor: tensor<2x3xf64, #CSR64>) -> tensor<2x3xf64, #CSC64> {
      	// CHECK-NEXT: %[[ANSWER:.*]] = graphblas.transpose %[[ARG0]] {swap_sizes = false} : [[CSR_TYPE]] to [[CSC_TYPE]]
        %answer = graphblas.transpose %sparse_tensor { swap_sizes = false } : tensor<2x3xf64, #CSR64> to tensor<2x3xf64, #CSC64>
        // CHECK-NEXT: return %[[ANSWER]] : [[CSC_TYPE]]
        return %answer : tensor<2x3xf64, #CSC64>
    }
    
    // CHECK: func @transpose_wrapper(%[[ARG0:.*]]: tensor<2x3xf64, [[CSR64:.*]]>) -> tensor<3x2xf64, [[CSC64:.*]]> {
    func @transpose_wrapper(%sparse_tensor: tensor<2x3xf64, #CSR64>) -> tensor<3x2xf64, #CSC64> {
      	// CHECK-NEXT: %[[ANSWER:.*]] = graphblas.transpose %[[ARG0]] {swap_sizes = true} : tensor<2x3xf64, [[CSR64]]> to tensor<3x2xf64, [[CSC64]]>
        %answer = graphblas.transpose %sparse_tensor { swap_sizes = true } : tensor<2x3xf64, #CSR64> to tensor<3x2xf64, #CSC64>
        // COM: CHECK-NEXT: return %[[ANSWER]] : tensor<3x2xf64, [[CSC64]]>
        return %answer : tensor<3x2xf64, #CSC64>
    }
        
}


module {
        
    // CHECK: func @matrix_select_triu(%[[ARG0:.*]]: [[CSR_TYPE:.*]]) -> [[CSR_TYPE]] {
    func @matrix_select_triu(%sparse_tensor: tensor<100x100xf64, #CSR64>) -> tensor<100x100xf64, #CSR64> {
      	// CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_select %[[ARG0]] {selector = "triu"} : [[CSR_TYPE]]
        %answer = graphblas.matrix_select %sparse_tensor { selector = "triu" } : tensor<100x100xf64, #CSR64>
        // CHECK-NEXT: return %[[ANSWER]] : [[CSR_TYPE]]
        return %answer : tensor<100x100xf64, #CSR64>
    }
    
    // CHECK: func @matrix_select_tril(%[[ARG0:.*]]: [[CSR_TYPE:.*]]) -> [[CSR_TYPE]] {
    func @matrix_select_tril(%sparse_tensor: tensor<100x100xf64, #CSR64>) -> tensor<100x100xf64, #CSR64> {
      	// CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_select %[[ARG0]] {selector = "tril"} : [[CSR_TYPE]]
        %answer = graphblas.matrix_select %sparse_tensor { selector = "tril" } : tensor<100x100xf64, #CSR64>
        // CHECK-NEXT: return %[[ANSWER]] : [[CSR_TYPE]]
        return %answer : tensor<100x100xf64, #CSR64>
    }
    
    // CHECK: func @matrix_select_gt0(%[[ARG0:.*]]: [[CSR_TYPE:.*]]) -> [[CSR_TYPE]] {
    func @matrix_select_gt0(%sparse_tensor: tensor<100x100xf64, #CSR64>) -> tensor<100x100xf64, #CSR64> {
      	// CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_select %[[ARG0]] {selector = "gt0"} : [[CSR_TYPE]]
        %answer = graphblas.matrix_select %sparse_tensor { selector = "gt0" } : tensor<100x100xf64, #CSR64>
        // CHECK-NEXT: return %[[ANSWER]] : [[CSR_TYPE]]
        return %answer : tensor<100x100xf64, #CSR64>
    }
    
}
