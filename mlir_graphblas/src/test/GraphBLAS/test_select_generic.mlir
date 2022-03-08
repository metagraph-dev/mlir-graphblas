// RUN: graphblas-opt %s | graphblas-exec entry | FileCheck %s

#CSR64 = #sparse_tensor.encoding<{
    dimLevelType = [ "dense", "compressed" ],
    dimOrdering = affine_map<(d0, d1) -> (d0, d1)>,
    pointerBitWidth = 64,
    indexBitWidth = 64
}>

#CSC64 = #sparse_tensor.encoding<{
    dimLevelType = [ "dense", "compressed" ],
    dimOrdering = affine_map<(d0, d1) -> (d1, d0)>,
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
    ///////////////
    // Test Matrix
    ///////////////

    %m = arith.constant sparse<[
      [0, 1], [0, 2],
      [1, 0], [1, 3], [1, 4],
      [3, 2]
    ], [1., 2., 3., 4., 5., 6.]> : tensor<4x5xf64>
    %m_csr = sparse_tensor.convert %m : tensor<4x5xf64> to tensor<?x?xf64, #CSR64>
    %m_csc = sparse_tensor.convert %m : tensor<4x5xf64> to tensor<?x?xf64, #CSC64>

    // CSR select upper triangle
    //
    // CHECK:      shape=(4, 5)
    // CHECK-NEXT: pointers=(0, 2, 4, 4, 4)
    // CHECK-NEXT: indices=(1, 2, 3, 4)
    // CHECK-NEXT: values=(1, 2, 4, 5)
    //
    %2 = graphblas.select_generic %m_csr : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSR64> {
    ^bb0(%arg0: f64, %arg1: index, %arg2: index):
      %ans = arith.cmpi ugt, %arg2, %arg1 : index
      graphblas.yield select_out %ans : i1
    }
    graphblas.print_tensor %2 {level = 4 : i64} : tensor<?x?xf64, #CSR64>
    
    // CSC select lt thunk
    //
    // CHECK:      shape=(4, 5)
    // CHECK-NEXT: pointers=(0, 1, 2, 3, 3, 3)
    // CHECK-NEXT: indices=(1, 0, 0)
    // CHECK-NEXT: values=(3, 1, 2)
    //
    %c3_5_f64 = arith.constant 3.500000e+00 : f64
    %3 = graphblas.select_generic %m_csc : tensor<?x?xf64, #CSC64> to tensor<?x?xf64, #CSC64> {
    ^bb0(%arg0: f64):
      %ans = arith.cmpf olt, %arg0, %c3_5_f64 : f64
      graphblas.yield select_out %ans : i1
    }
    graphblas.print_tensor %3 {level = 4 : i64} : tensor<?x?xf64, #CSC64>

    ///////////////
    // Test Vector
    ///////////////

    %v = arith.constant sparse<[
      [1], [2], [4], [7]
    ], [1, 2, 3, 4]> : tensor<9xi32>
    %v_cv = sparse_tensor.convert %v : tensor<9xi32> to tensor<?xi32, #CV64>
    
    // CV select eq thunk with empty result
    //
    // CHECK:      shape=(9)
    // CHECK-NEXT: pointers=(0, 0)
    // CHECK-NEXT: indices=()
    // CHECK-NEXT: values=()
    //
    %c6_i32 = arith.constant 6 : i32
    %21 = graphblas.select_generic %v_cv : tensor<?xi32, #CV64> to tensor<?xi32, #CV64> {
    ^bb0(%arg0: i32):
      %ans = arith.cmpi eq, %arg0, %c6_i32 : i32
      graphblas.yield select_out %ans : i1
    }
    graphblas.print_tensor %21 { level=4 } : tensor<?xi32, #CV64>

    return
  }
}
