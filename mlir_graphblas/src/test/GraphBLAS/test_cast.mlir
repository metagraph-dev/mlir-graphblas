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
    // Test Float
    ///////////////

    %m = arith.constant sparse<[
      [0, 1], [0, 2],
      [1, 0], [1, 3], [1, 4],
      [3, 2]
    ], [1., 2.25, 3., 4., 5., 6.]> : tensor<4x5xf64>
    %m_csr = sparse_tensor.convert %m : tensor<4x5xf64> to tensor<?x?xf64, #CSR>

    // CSR f64 cast to f32
    //
    // CHECK:      pointers=(0, 2, 5, 5, 6)
    // CHECK-NEXT: indices=(1, 2, 0, 3, 4, 2)
    // CHECK-NEXT: values=(1, 2.25, 3, 4, 5, 6)
    //
    %0 = graphblas.cast %m_csr : tensor<?x?xf64, #CSR> to tensor<?x?xf32, #CSR>
    graphblas.print_tensor %0 { level=3 } : tensor<?x?xf32, #CSR>

    // CSR f64 cast to i64
    //
    // CHECK:      pointers=(0, 2, 5, 5, 6)
    // CHECK-NEXT: indices=(1, 2, 0, 3, 4, 2)
    // CHECK-NEXT: values=(1, 2, 3, 4, 5, 6)
    //
    %10 = graphblas.cast %m_csr : tensor<?x?xf64, #CSR> to tensor<?x?xi64, #CSR>
    graphblas.print_tensor %10 { level=3 } : tensor<?x?xi64, #CSR>


    ///////////////
    // Test Integer
    ///////////////

    %v = arith.constant sparse<[
      [1], [2], [4], [7]
    ], [1, 2, 3, 4]> : tensor<9xi32>
    %v_cv = sparse_tensor.convert %v : tensor<9xi32> to tensor<?xi32, #CV>

    // CV i32 cast to i8
    //
    // CHECK:      pointers=(0, 4)
    // CHECK-NEXT: indices=(1, 2, 4, 7)
    // CHECK-NEXT: values=(1, 2, 3, 4)
    //
    %20 = graphblas.cast %v_cv : tensor<?xi32, #CV> to tensor<?xi8, #CV>
    graphblas.print_tensor %20 { level=3 } : tensor<?xi8, #CV>

    // CV i32 cast to i64
    //
    // CHECK:      pointers=(0, 4)
    // CHECK-NEXT: indices=(1, 2, 4, 7)
    // CHECK-NEXT: values=(1, 2, 3, 4)
    //
    %30 = graphblas.cast %v_cv : tensor<?xi32, #CV> to tensor<?xi64, #CV>
    graphblas.print_tensor %30 { level=3 } : tensor<?xi64, #CV>

    // CV i32 cast to f32
    //
    // CHECK:      pointers=(0, 4)
    // CHECK-NEXT: indices=(1, 2, 4, 7)
    // CHECK-NEXT: values=(1, 2, 3, 4)
    //
    %40 = graphblas.cast %v_cv : tensor<?xi32, #CV> to tensor<?xf32, #CV>
    graphblas.print_tensor %40 { level=3 } : tensor<?xf32, #CV>

    return
  }
}