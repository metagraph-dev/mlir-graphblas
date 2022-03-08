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
    %mat = arith.constant dense<[
        [0, 0, 9, 0],
        [0, 0, 0, 8],
        [7, 0, 6, 0],
        [0, 0, 0, 5]
      ]> : tensor<4x4xi64>
    %csr = sparse_tensor.convert %mat : tensor<4x4xi64> to tensor<?x?xi64, #CSR64>

    %indices_dense, %vals_dense = graphblas.to_coo %csr : tensor<?x?xi64, #CSR64> to tensor<?x?xindex>, tensor<?xi64>
    %indices = sparse_tensor.convert %indices_dense : tensor<?x?xindex> to tensor<?x?xindex, #CSR64>
    %vals = sparse_tensor.convert %vals_dense : tensor<?xi64> to tensor<?xi64, #CV64>
    // CHECK: %vals [9, 8, 7, 6, 5]
    graphblas.print %vals { strings=["%vals "] } : tensor<?xi64, #CV64>

    return
  }
}
