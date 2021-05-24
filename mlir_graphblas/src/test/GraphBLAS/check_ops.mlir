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
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = graphblas.foo %{{.*}} : i32
        %res = graphblas.foo %0 : i32
        return
    }
    
    // CHECK: func @transpose_wrapper(%[[ARG0:.*]]: [[CSR_TYPE:.*]]) -> [[CSC_TYPE:.*]] {
    func @transpose_wrapper(%sparse_tensor: tensor<2x3xf64, #CSR64>) -> tensor<2x3xf64, #CSC64> {
      	// CHECK-NEXT: %[[ANSWER:.*]] = graphblas.transpose %[[ARG0]] {swap_sizes = false} : [[CSR_TYPE]] to [[CSC_TYPE]]
        %answer = graphblas.transpose %sparse_tensor { swap_sizes = false } : tensor<2x3xf64, #CSR64> to tensor<2x3xf64, #CSC64>
        // CHECK-NEXT: return %[[ANSWER]] : [[CSC_TYPE]]
        return %answer : tensor<2x3xf64, #CSC64>
    }
}
