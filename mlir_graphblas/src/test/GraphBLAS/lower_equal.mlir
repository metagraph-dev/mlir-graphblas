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

module {
  func @entry() {
    %c2 = constant 2 : index
    %cf22 = constant 22.0 : f64

    ///////////////
    // Test Matrix
    ///////////////

    %m = constant dense<[
      [ 1.0,  0.0,  2.0,  0.0,  0.0,  0.0,  0.0,  3.0],
      [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
      [ 0.0,  0.0,  4.0,  0.0,  0.0,  0.0,  0.0,  0.0],
      [ 0.0,  0.0, 10.0,  0.0,  0.0,  0.0, 11.0, 12.0],
      [ 0.0, 13.0, 14.0,  0.0,  0.0,  0.0, 15.0, 16.0],
      [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
      [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 17.0,  0.0]
    ]> : tensor<7x8xf64>
    %m_fixed_csr = sparse_tensor.convert %m : tensor<7x8xf64> to tensor<7x8xf64, #CSR64>
    %m_csr = sparse_tensor.convert %m : tensor<7x8xf64> to tensor<?x?xf64, #CSR64>
    %m_csc = sparse_tensor.convert %m : tensor<7x8xf64> to tensor<?x?xf64, #CSC64>

    // Fixed-sized CSR Matrix should be equal to itself
    //
    // CHECK:  (0) is_equal=1
    //
    %0 = graphblas.equal %m_fixed_csr, %m_fixed_csr : tensor<7x8xf64, #CSR64>, tensor<7x8xf64, #CSR64>
    graphblas.print %0 { strings=["(0) is_equal="] } : i1

    // Dynamic-sized CSR Matrix should be equal to itself
    //
    // CHECK:  (1) is_equal=1
    //
    %1 = graphblas.equal %m_csr, %m_csr : tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64>
    graphblas.print %1 { strings=["(1) is_equal="] } : i1

    // Should be equal to its duplicate
    //
    // CHECK:  (2) is_equal=1
    //
    %m_dup = graphblas.dup %m_csr : tensor<?x?xf64, #CSR64>
    %2 = graphblas.equal %m_dup, %m_csr : tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64>
    graphblas.print %2 { strings=["(2) is_equal="] } : i1

    // CSR should not be equal to a modified version
    //
    // CHECK:  (3) is_equal=0
    //
    %m_dup_values = sparse_tensor.values %m_dup : tensor<?x?xf64, #CSR64> to memref<?xf64>
    memref.store %cf22, %m_dup_values[%c2] : memref<?xf64>
    %3 = graphblas.equal %m_dup, %m_csr : tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64>
    graphblas.print %3 { strings=["(3) is_equal="] } : i1

    // Dynamic-sized CSC Matrix should be equal to itself
    //
    // CHECK:  (10) is_equal=1
    //
    %10 = graphblas.equal %m_csc, %m_csc : tensor<?x?xf64, #CSC64>, tensor<?x?xf64, #CSC64>
    graphblas.print %10 { strings=["(10) is_equal="] } : i1

    ///////////////
    // Test Vector
    ///////////////

    %v = constant dense<
      [ 1.0,  2.0,  0.0, 0.0, -4.0, 0.0, 0.0, 0.0 ]
    > : tensor<8xf64>
    %v_fixed_cv = sparse_tensor.convert %v : tensor<8xf64> to tensor<8xf64, #CV64>
    %v_cv = sparse_tensor.convert %v : tensor<8xf64> to tensor<?xf64, #CV64>

    %v2 = constant dense<
      [ 1.0, 2.0, 0.0, 3.0]
    > : tensor<4xf64>
    %v2_cv = sparse_tensor.convert %v2 : tensor<4xf64> to tensor<?xf64, #CV64>

    // Fixed-sized Vector should be equal to itself
    //
    // CHECK:  (20) is_equal=1
    //
    %20 = graphblas.equal %v_fixed_cv, %v_fixed_cv : tensor<8xf64, #CV64>, tensor<8xf64, #CV64>
    graphblas.print %20 { strings=["(20) is_equal="] } : i1

    // Dynamic-sized Vector should be equal to itself
    //
    // CHECK:  (21) is_equal=1
    //
    %21 = graphblas.equal %v_cv, %v_cv : tensor<?xf64, #CV64>, tensor<?xf64, #CV64>
    graphblas.print %21 { strings=["(21) is_equal="] } : i1

    // Dynamic-sized Vector should not be equal to a differently sized dynamic vector
    //
    // CHECK:  (22) is_equal=0
    //
    %22 = graphblas.equal %v2_cv, %v_cv : tensor<?xf64, #CV64>, tensor<?xf64, #CV64>
    graphblas.print %22 { strings=["(22) is_equal="] } : i1

    return
  }
}
