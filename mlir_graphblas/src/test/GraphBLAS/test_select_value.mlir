// RUN: graphblas-opt %s | graphblas-exec entry | FileCheck %s

#CSR = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#CSC = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#CV = #sparse_tensor.encoding<{
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
    %m_csr = sparse_tensor.convert %m : tensor<4x5xf64> to tensor<?x?xf64, #CSR>
    %m_csc = sparse_tensor.convert %m : tensor<4x5xf64> to tensor<?x?xf64, #CSC>

    // CSC select lt thunk
    //
    // CHECK:      shape=(4, 5)
    // CHECK-NEXT: pointers=(0, 1, 2, 3, 3, 3)
    // CHECK-NEXT: indices=(1, 0, 0)
    // CHECK-NEXT: values=(3, 1, 2)
    //
    %10 = arith.constant 3.5 : f64
    %11 = graphblas.select %m_csc, %10 { selector="lt" } : tensor<?x?xf64, #CSC>, f64 to tensor<?x?xf64, #CSC>
    graphblas.print_tensor %11 { level=4 } : tensor<?x?xf64, #CSC>


    ///////////////
    // Test Vector
    ///////////////

    %v = arith.constant sparse<[
      [1], [2], [4], [7]
    ], [1, 2, 3, 4]> : tensor<9xi32>
    %v_cv = sparse_tensor.convert %v : tensor<9xi32> to tensor<?xi32, #CV>

    // CV select eq thunk with empty result
    //
    // CHECK:      shape=(9)
    // CHECK-NEXT: pointers=(0, 0)
    // CHECK-NEXT: indices=()
    // CHECK-NEXT: values=()
    //
    %20 = arith.constant 6 : i32
    %21 = graphblas.select %v_cv, %20 { selector="eq" } : tensor<?xi32, #CV>, i32 to tensor<?xi32, #CV>
    graphblas.print_tensor %21 { level=4 } : tensor<?xi32, #CV>

    return
  }
}