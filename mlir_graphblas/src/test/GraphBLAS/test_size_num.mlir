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

    // CSR num rows, cols, vals
    //
    // CHECK: nrows=4 ncols=5 nvals=6
    //
    %0 = graphblas.num_rows %m_csr : tensor<?x?xf64, #CSR>
    %1 = graphblas.num_cols %m_csr : tensor<?x?xf64, #CSR>
    %2 = graphblas.num_vals %m_csr : tensor<?x?xf64, #CSR>
    graphblas.print %0, %1, %2 { strings=["nrows=", " ncols=", " nvals="] } : index, index, index

    // CSC num rows, cols, vals
    //
    // CHECK-NEXT: nrows=4 ncols=5 nvals=6
    //
    %10 = graphblas.num_rows %m_csc : tensor<?x?xf64, #CSC>
    %11 = graphblas.num_cols %m_csc : tensor<?x?xf64, #CSC>
    %12 = graphblas.num_vals %m_csc : tensor<?x?xf64, #CSC>
    graphblas.print %10, %11, %12 { strings=["nrows=", " ncols=", " nvals="] } : index, index, index


    ///////////////
    // Test Vector
    ///////////////

    %v = arith.constant sparse<[
      [1], [2], [4], [7]
    ], [1, 2, 3, 4]> : tensor<9xi32>
    %v_cv = sparse_tensor.convert %v : tensor<9xi32> to tensor<?xi32, #CV>

    // CV size, num vals
    //
    // CHECK: size=9 nvals=4
    //
    %20 = graphblas.size %v_cv : tensor<?xi32, #CV>
    %21 = graphblas.num_vals %v_cv : tensor<?xi32, #CV>
    graphblas.print %20 , %21{ strings=["size=", " nvals="] } : index, index

    return
  }
}