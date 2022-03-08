// RUN: graphblas-opt %s | graphblas-exec entry | FileCheck %s

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
  func @entry() {
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index

    %indices = arith.constant dense<[
      [0, 1],
      [2, 3]
    ]> : tensor<2x2xindex>
    %vals = arith.constant dense<[11.1, 22.2]> : tensor<2xf64>

    %csr = graphblas.from_coo %indices, %vals [%c3, %c4] : tensor<2x2xindex>, tensor<2xf64> to tensor<?x?xf64, #CSR64>
    // CHECK: %csr [
    // CHECK-NEXT: [_, 11.1, _, _],
    // CHECK-NEXT: [_, _, _, _],
    // CHECK-NEXT: [_, _, _, 22.2]
    // CHECK-NEXT: ]
    graphblas.print %csr { strings=["%csr "] } : tensor<?x?xf64, #CSR64>

    return
  }
}
