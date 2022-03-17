// RUN: graphblas-opt %s | graphblas-exec main | FileCheck %s
// RUN: graphblas-opt %s | graphblas-linalg-exec main | FileCheck %s

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#CSC64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

func @main() -> () {
    %dense_m1 = arith.constant dense<[
        [0, 1, 2, 0],
        [0, 0, 0, 3],
        [4, 0, 5, 0],
        [0, 6, 0, 7]
      ]> : tensor<4x4xi64>
    %m1_csr = sparse_tensor.convert %dense_m1 : tensor<4x4xi64> to tensor<?x?xi64, #CSR64>
    %m1_csc = sparse_tensor.convert %dense_m1 : tensor<4x4xi64> to tensor<?x?xi64, #CSC64>
    
    %dense_m2 = arith.constant dense<[
        [0, 0, 9, 0],
        [0, 0, 0, 8],
        [7, 0, 6, 0],
        [0, 0, 0, 5]
      ]> : tensor<4x4xi64>
    %m2_csr = sparse_tensor.convert %dense_m2 : tensor<4x4xi64> to tensor<?x?xi64, #CSR64>
    %m2_csc = sparse_tensor.convert %dense_m2 : tensor<4x4xi64> to tensor<?x?xi64, #CSC64>

    %csr_csr = graphblas.intersect %m1_csr, %m2_csr { intersect_operator = "times" } : (tensor<?x?xi64, #CSR64>, tensor<?x?xi64, #CSR64>) to tensor<?x?xi64, #CSR64>
    // CHECK: %csr_csr [
    // CHECK-NEXT:   [_, _, 18, _],
    // CHECK-NEXT:   [_, _, _, 24],
    // CHECK-NEXT:   [28, _, 30, _],
    // CHECK-NEXT:   [_, _, _, 35]
    // CHECK-NEXT: ]
    graphblas.print %csr_csr { strings = ["%csr_csr "] } : tensor<?x?xi64, #CSR64>

    %csc_csc = graphblas.intersect %m1_csc, %m2_csc { intersect_operator = "times" } : (tensor<?x?xi64, #CSC64>, tensor<?x?xi64, #CSC64>) to tensor<?x?xi64, #CSC64>
    // CHECK: %csc_csc [
    // CHECK-NEXT:   [_, _, 18, _],
    // CHECK-NEXT:   [_, _, _, 24],
    // CHECK-NEXT:   [28, _, 30, _],
    // CHECK-NEXT:   [_, _, _, 35]
    // CHECK-NEXT: ]
    graphblas.print %csc_csc { strings = ["%csc_csc "] } : tensor<?x?xi64, #CSC64>

    return
}
