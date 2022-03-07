// RUN: graphblas-opt %s | graphblas-exec entry | FileCheck %s

#CV = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
  func @entry() {
    ///////////////
    // Test Vector
    ///////////////

    %v = arith.constant sparse<[
      [1], [2], [4], [7]
    ], [1, 2, 3, 4]> : tensor<9xi32>
    %v_cv = sparse_tensor.convert %v : tensor<9xi32> to tensor<?xi32, #CV>

    // CV apply in place
    //
    // CHECK: [_, 0.479426, 0.863209, _, 0.916166, -0.57844, _, _]
    //
    %130 = arith.constant sparse<[
      [1], [2], [4], [5]
    ], [0.5, 2.1, -4.3, -6.9]> : tensor<8xf32>
    %131 = sparse_tensor.convert %130 : tensor<8xf32> to tensor<?xf32, #CV>
    graphblas.apply %131 { apply_operator="sin", in_place=true } : (tensor<?xf32, #CV>) to tensor<?xf32, #CV>
    graphblas.print_tensor %131 { level=0 } : tensor<?xf32, #CV>

    return
  }
}
