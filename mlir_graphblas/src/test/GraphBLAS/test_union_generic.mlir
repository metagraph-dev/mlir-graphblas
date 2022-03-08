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
    // CHECK:      pointers=(0, 2, 2, 5, 6)
    // CHECK-NEXT: indices=(0, 2, 1, 2, 4, 1)
    // CHECK-NEXT: values=(4.2, 2, 3, 0, 12, 1)
    //
    %0 = graphblas.union_generic %m_csr, %m2_csr : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64>) to tensor<?x?xf64, #CSR64> {
    ^bb0(%arg0: f64, %arg1: f64):
      %9 = arith.addf %arg0, %arg1 : f64
      graphblas.yield mult %9 : f64
    }
    graphblas.print_tensor %0 { level=3 } : tensor<?x?xf64, #CSR64>

    // CSC union min
    //
    // CHECK:      pointers=(0, 1, 3, 5, 5, 6)
    // CHECK-NEXT: indices=(0, 2, 3, 0, 2, 2)
    // CHECK-NEXT: values=(1, 3, 1, 2, -4, 12)
    //
    %10 = graphblas.union_generic %m_csc, %m2_csc : (tensor<?x?xf64, #CSC64>, tensor<?x?xf64, #CSC64>) to tensor<?x?xf64, #CSC64> {
    ^bb0(%arg0: f64, %arg1: f64):
      %9 = arith.cmpf olt, %arg0, %arg1 : f64
      %10 = arith.select %9, %arg0, %arg1 : f64
      graphblas.yield mult %10 : f64
    }
    graphblas.print_tensor %10 { level=3 } : tensor<?x?xf64, #CSC64>

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
    // CHECK:      pointers=(0, 4)
    // CHECK-NEXT: indices=(0, 1, 3, 4)
    // CHECK-NEXT: values=(1, 3, 6.2, 4)
    //
    %20 = graphblas.union_generic %v_cv, %v2_cv  : (tensor<?xf64, #CV64>, tensor<?xf64, #CV64>) to tensor<?xf64, #CV64> {
    ^bb0(%arg0: f64, %arg1: f64):
      graphblas.yield mult %arg1 : f64
    }
    graphblas.print_tensor %20 { level=3 } : tensor<?xf64, #CV64>

    return
  }
}
