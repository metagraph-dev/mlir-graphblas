// RUN: graphblas-opt %s | graphblas-exec entry | FileCheck %s

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

// COM: linalg-lower cannot yet handle mask_complement=true
// COM: linalg-lower cannot handle intersection with different dtype

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

    %mask = arith.constant sparse<[
      [0, 1], [0, 3], [0, 4],
      [1, 3],
      [2, 2],
      [3, 2]
    ], [100., 200., 300., 400., 500., 600.]> : tensor<4x5xf64>
    %mask_csr = sparse_tensor.convert %mask : tensor<4x5xf64> to tensor<?x?xf64, #CSR64>

    // CSR select mask complement
    //
    // CHECK:      shape=(4, 5)
    // CHECK:      pointers=(0, 1, 3, 3, 3)
    // CHECK-NEXT: indices=(2, 0, 4)
    // CHECK-NEXT: values=(2, 3, 5)
    %1 = graphblas.select_mask %m_csr, %mask_csr {mask_complement=true} : tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSR64>
    graphblas.print_tensor %1 { level=4 } : tensor<?x?xf64, #CSR64>

    // CSC select mask different element type
    //
    // CHECK:      rev=(1, 0)
    // CHECK:      shape=(4, 5)
    // CHECK:      pointers=(0, 0, 1, 2, 3, 3)
    // CHECK-NEXT: indices=(0, 3, 1)
    // CHECK-NEXT: values=(1, 6, 4)
    //
    %mask2 = arith.constant sparse<[
      [0, 1], [0, 3], [0, 4],
      [1, 3],
      [2, 2],
      [3, 2]
    ], [100, 200, 300, 400, 500, 600]> : tensor<4x5xi32>
    %mask_csc = sparse_tensor.convert %mask2 : tensor<4x5xi32> to tensor<?x?xi32, #CSC64>
    %10 = graphblas.select_mask %m_csc, %mask_csc : tensor<?x?xf64, #CSC64>, tensor<?x?xi32, #CSC64> to tensor<?x?xf64, #CSC64>
    graphblas.print_tensor %10 { level=5 } : tensor<?x?xf64, #CSC64>

    ///////////////
    // Test Vector
    ///////////////

    %v = arith.constant sparse<[
      [1], [2], [4], [7]
    ], [1., 2., 3., 4.]> : tensor<9xf64>
    %v_cv = sparse_tensor.convert %v : tensor<9xf64> to tensor<?xf64, #CV64>

    %mask3 = arith.constant sparse<[
      [2], [3], [4]
    ], [200., 300., 400.]> : tensor<9xf64>
    %mask_cv = sparse_tensor.convert %mask3 : tensor<9xf64> to tensor<?xf64, #CV64>

    // Vector select mask complement
    //
    // CHECK:      shape=(9)
    // CHECK:      pointers=(0, 2)
    // CHECK-NEXT: indices=(1, 7)
    // CHECK-NEXT: values=(1, 4)
    //
    %21 = graphblas.select_mask %v_cv, %mask_cv {mask_complement=true} : tensor<?xf64, #CV64>, tensor<?xf64, #CV64> to tensor<?xf64, #CV64>
    graphblas.print_tensor %21 { level=4 } : tensor<?xf64, #CV64>

    return
  }
}