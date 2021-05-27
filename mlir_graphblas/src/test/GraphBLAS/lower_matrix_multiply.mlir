
// CHECK: {{.*}}
// COM: // RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s
// COM: TODO finish this
// COM: 
// COM: #CSR64 = #sparse_tensor.encoding<{
// COM:   dimLevelType = [ "dense", "compressed" ],
// COM:   dimOrdering = affine_map<(i,j) -> (i,j)>,
// COM:   pointerBitWidth = 64,
// COM:   indexBitWidth = 64
// COM: }>
// COM: 
// COM: #CSC64 = #sparse_tensor.encoding<{
// COM:   dimLevelType = [ "dense", "compressed" ],
// COM:   dimOrdering = affine_map<(i,j) -> (j,i)>,
// COM:   pointerBitWidth = 64,
// COM:   indexBitWidth = 64
// COM: }>
// COM: 
// COM: module {
// COM: 
// COM:     // CHECK: func @matrix_multiply_plus_times(%[[ARGA:.*]]: [[CSR_TYPE_A:tensor<.*->.*>]], %[[ARGB:.*]]: [[CSR_TYPE_B:tensor<.*->.*>]]) -> [[RETURN_TYPE:tensor<.*->.*>]] {
// COM:     func @matrix_multiply_plus_times(%argA: tensor<2x3xi64, #CSR64>, %argB: tensor<3x2xi64, #CSR64>) -> tensor<2x2xi64, #CSR64> {
// COM:         // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_multiply %[[ARGA]], %[[ARGB]] {semiring = "plus_times"} : ([[CSR_TYPE_A]], [[CSR_TYPE_B]]) to [[RETURN_TYPE]]
// COM:         %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_times" } : (tensor<2x3xi64, #CSR64>, tensor<3x2xi64, #CSR64>) to tensor<2x2xi64, #CSR64>
// COM:         // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
// COM:         return %answer : tensor<2x2xi64, #CSR64>
// COM:     }
// COM: 
// COM: }
// COM: 
