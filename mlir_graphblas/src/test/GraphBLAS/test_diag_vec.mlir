// RUN: graphblas-opt %s | graphblas-exec main | FileCheck %s

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

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
  func @main() -> () {
    %dense_vec = arith.constant dense<[0.0, 7.0, 4.0, 0.0, 5.0, 0.0, 6.0, 8.0]> : tensor<8xf64>
    %vec = sparse_tensor.convert %dense_vec : tensor<8xf64> to tensor<?xf64, #CV64>

    %answer_csr = graphblas.diag %vec : tensor<?xf64, #CV64> to tensor<?x?xf64, #CSR64>
    // CHECK: %answer_csr [
    // CHECK-NEXT:   [_, _, _, _, _, _, _, _],
    // CHECK-NEXT:   [_, 7, _, _, _, _, _, _],
    // CHECK-NEXT:   [_, _, 4, _, _, _, _, _],
    // CHECK-NEXT:   [_, _, _, _, _, _, _, _],
    // CHECK-NEXT:   [_, _, _, _, 5, _, _, _],
    // CHECK-NEXT:   [_, _, _, _, _, _, _, _],
    // CHECK-NEXT:   [_, _, _, _, _, _, 6, _],
    // CHECK-NEXT:   [_, _, _, _, _, _, _, 8]
    // CHECK-NEXT: ]
    graphblas.print %answer_csr { strings = ["%answer_csr "] } : tensor<?x?xf64, #CSR64>
    
    %answer_csc = graphblas.diag %vec : tensor<?xf64, #CV64> to tensor<?x?xf64, #CSC64>
    // CHECK: %answer_csc [
    // CHECK-NEXT:   [_, _, _, _, _, _, _, _],
    // CHECK-NEXT:   [_, 7, _, _, _, _, _, _],
    // CHECK-NEXT:   [_, _, 4, _, _, _, _, _],
    // CHECK-NEXT:   [_, _, _, _, _, _, _, _],
    // CHECK-NEXT:   [_, _, _, _, 5, _, _, _],
    // CHECK-NEXT:   [_, _, _, _, _, _, _, _],
    // CHECK-NEXT:   [_, _, _, _, _, _, 6, _],
    // CHECK-NEXT:   [_, _, _, _, _, _, _, 8]
    // CHECK-NEXT: ]
    graphblas.print %answer_csc { strings = ["%answer_csc "] } : tensor<?x?xf64, #CSC64>

    return
  }
}
