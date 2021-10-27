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
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ci0 = arith.constant 0 : i64
    %cf0 = arith.constant 0.0 : f64

    ///////////////
    // Test Matrix
    ///////////////

    %m = arith.constant dense<[
      [ 1.0,  0.0,  2.0,  0.0,  0.0],
      [ 0.0,  0.0,  0.0,  0.0,  0.0],
      [ 0.0,  3.0,  4.0,  0.0,  0.0],
      [ 0.0,  0.0,  0.0,  0.0,  0.0]
    ]> : tensor<4x5xf64>
    %m_csr = sparse_tensor.convert %m : tensor<4x5xf64> to tensor<?x?xf64, #CSR64>
    %m_csc = sparse_tensor.convert %m : tensor<4x5xf64> to tensor<?x?xf64, #CSC64>

    %m2 = arith.constant dense<[
      [ 3.2,  0.0,  0.0,  0.0,  0.0],
      [ 0.0,  0.0,  0.0,  0.0,  0.0],
      [ 0.0,  0.0, -4.0,  0.0, 12.0],
      [ 0.0,  1.0,  0.0,  0.0,  0.0]
    ]> : tensor<4x5xf64>
    %m2_csr = sparse_tensor.convert %m2 : tensor<4x5xf64> to tensor<?x?xf64, #CSR64>
    %m2_csc = sparse_tensor.convert %m2 : tensor<4x5xf64> to tensor<?x?xf64, #CSC64>

    // CSR union plus
    //
    // CHECK: ( 0, 2, 2, 5, 6 )
    // CHECK: ( 0, 2, 1, 2, 4, 1 )
    // CHECK: ( 4.2, 2, 3, 0, 12, 1 )
    //
    %0 = graphblas.union %m_csr, %m2_csr { union_operator = "plus" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64>) to tensor<?x?xf64, #CSR64>
    %1 = sparse_tensor.pointers %0, %c1 : tensor<?x?xf64, #CSR64> to memref<?xi64>
    %2 = vector.transfer_read %1[%c0], %ci0: memref<?xi64>, vector<5xi64>
    vector.print %2 : vector<5xi64>
    %4 = sparse_tensor.indices %0, %c1 : tensor<?x?xf64, #CSR64> to memref<?xi64>
    %5 = vector.transfer_read %4[%c0], %ci0: memref<?xi64>, vector<6xi64>
    vector.print %5 : vector<6xi64>
    %7 = sparse_tensor.values %0 : tensor<?x?xf64, #CSR64> to memref<?xf64>
    %8 = vector.transfer_read %7[%c0], %cf0: memref<?xf64>, vector<6xf64>
    vector.print %8 : vector<6xf64>

    // CSC union min
    //
    // CHECK: ( 0, 1, 3, 5, 5, 6 )
    // CHECK: ( 0, 2, 3, 0, 2, 2 )
    // CHECK: ( 1, 3, 1, 2, -4, 12 )
    //
    %10 = graphblas.union %m_csc, %m2_csc { union_operator = "min" } : (tensor<?x?xf64, #CSC64>, tensor<?x?xf64, #CSC64>) to tensor<?x?xf64, #CSC64>
    %11 = sparse_tensor.pointers %10, %c1 : tensor<?x?xf64, #CSC64> to memref<?xi64>
    %12 = vector.transfer_read %11[%c0], %ci0: memref<?xi64>, vector<6xi64>
    vector.print %12 : vector<6xi64>
    %13 = sparse_tensor.indices %10, %c1 : tensor<?x?xf64, #CSC64> to memref<?xi64>
    %14 = vector.transfer_read %13[%c0], %ci0: memref<?xi64>, vector<6xi64>
    vector.print %14 : vector<6xi64>
    %15 = sparse_tensor.values %10 : tensor<?x?xf64, #CSC64> to memref<?xf64>
    %16 = vector.transfer_read %15[%c0], %cf0: memref<?xf64>, vector<6xf64>
    vector.print %16 : vector<6xf64>

    ///////////////
    // Test Vector
    ///////////////

    %v = arith.constant dense<
      [ 1.0,  2.0,  0.0, 0.0, -4.0, 0.0 ]
    > : tensor<6xf64>
    %v_cv = sparse_tensor.convert %v : tensor<6xf64> to tensor<?xf64, #CV64>

    %v2 = arith.constant dense<
      [ 0.0,  3.0,  0.0, 6.2, 4.0, 0.0 ]
    > : tensor<6xf64>
    %v2_cv = sparse_tensor.convert %v2 : tensor<6xf64> to tensor<?xf64, #CV64>

    // Union second
    //
    // CHECK: ( 0, 4 )
    // CHECK: ( 0, 1, 3, 4 )
    // CHECK: ( 1, 3, 6.2, 4 )
    //
    %20 = graphblas.union %v_cv, %v2_cv { union_operator = "second" } : (tensor<?xf64, #CV64>, tensor<?xf64, #CV64>) to tensor<?xf64, #CV64>
    %21 = sparse_tensor.pointers %20, %c0 : tensor<?xf64, #CV64> to memref<?xi64>
    %22 = vector.transfer_read %21[%c0], %ci0: memref<?xi64>, vector<2xi64>
    vector.print %22 : vector<2xi64>
    %23 = sparse_tensor.indices %20, %c0 : tensor<?xf64, #CV64> to memref<?xi64>
    %24 = vector.transfer_read %23[%c0], %ci0: memref<?xi64>, vector<4xi64>
    vector.print %24 : vector<4xi64>
    %25 = sparse_tensor.values %20 : tensor<?xf64, #CV64> to memref<?xf64>
    %26 = vector.transfer_read %25[%c0], %cf0: memref<?xf64>, vector<4xf64>
    vector.print %26 : vector<4xf64>

    return
  }
}