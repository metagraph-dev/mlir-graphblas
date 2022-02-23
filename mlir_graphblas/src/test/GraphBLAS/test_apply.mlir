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

    // CSR apply generic
    //
    // CHECK:      pointers=(0, 2, 5, 5, 6)
    // CHECK-NEXT: indices=(1, 2, 0, 3, 4, 2)
    // CHECK-NEXT: values=(1, 2, 3, 4, 4.5, 4.5)
    //
    %thunk_f64 = arith.constant 4.5 : f64
    %0 = graphblas.apply_generic %m_csr : tensor<?x?xf64, #CSR> to tensor<?x?xf64, #CSR> {
      ^bb0(%val: f64):
        %pick = arith.cmpf olt, %val, %thunk_f64 : f64
        %result = arith.select %pick, %val, %thunk_f64 : f64
        graphblas.yield transform_out %result : f64
    }
    graphblas.print_tensor %0 { level=3 } : tensor<?x?xf64, #CSR>

    // CSC apply generic
    //
    // CHECK:      pointers=(0, 1, 2, 4, 5, 6)
    // CHECK-NEXT: indices=(1, 0, 0, 3, 1, 1)
    // CHECK-NEXT: values=(-3, -1, -2, -6, -4, -5)
    //
    %10 = graphblas.apply_generic %m_csc : tensor<?x?xf64, #CSC> to tensor<?x?xf64, #CSC> {
      ^bb0(%val: f64):
        %result = arith.negf %val : f64
        graphblas.yield transform_out %result : f64
    }
    graphblas.print_tensor %10 { level=3 } : tensor<?x?xf64, #CSC>

    // CSR apply column
    //
    // CHECK:      pointers=(0, 2, 5, 5, 6)
    // CHECK-NEXT: indices=(1, 2, 0, 3, 4, 2)
    // CHECK-NEXT: values=(1, 2, 0, 3, 4, 2)
    //
    %20 = graphblas.apply %m_csr { apply_operator="column" } : (tensor<?x?xf64, #CSR>) to tensor<?x?xi64, #CSR>
    graphblas.print_tensor %20 { level=3 } : tensor<?x?xi64, #CSR>

    // CSR apply div right
    //
    // CHECK:      pointers=(0, 2, 5, 5, 6)
    // CHECK-NEXT: indices=(1, 2, 0, 3, 4, 2)
    // CHECK-NEXT: values=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    //
    %divisor_f64 = arith.constant 10.0 : f64
    %30 = graphblas.apply %m_csr, %divisor_f64 { apply_operator="div" } : (tensor<?x?xf64, #CSR>, f64) to tensor<?x?xf64, #CSR>
    graphblas.print_tensor %30 { level=3 } : tensor<?x?xf64, #CSR>

    ///////////////
    // Test Vector
    ///////////////

    %v = arith.constant sparse<[
      [1], [2], [4], [7]
    ], [1, 2, 3, 4]> : tensor<9xi32>
    %v_cv = sparse_tensor.convert %v : tensor<9xi32> to tensor<?xi32, #CV>

    // CV apply generic
    //
    // CHECK: [_, -19, -18, _, -17, _, _, -16, _]
    //
    %thunk_i32 = arith.constant -20 : i32
    %100 = graphblas.apply_generic %v_cv : tensor<?xi32, #CV> to tensor<?xi32, #CV> {
      ^bb0(%val: i32):
        %result = arith.addi %val, %thunk_i32 : i32
        graphblas.yield transform_out %result : i32
    }
    graphblas.print_tensor %100 { level=0 } : tensor<?xi32, #CV>

    // CV apply minus left
    //
    // CHECK: [_, 89, 88, _, 87, _, _, 86, _]
    //
    %ninety_i32 = arith.constant 90 : i32
    %110 = graphblas.apply %ninety_i32, %v_cv { apply_operator="minus" } : (i32, tensor<?xi32, #CV>) to tensor<?xi32, #CV>
    graphblas.print_tensor %110 { level=0 } : tensor<?xi32, #CV>

    // CV apply gt right
    //
    // CHECK: [_, 0, 0, _, 1, _, _, 1, _]
    //
    %two_i32 = arith.constant 2 : i32
    %120 = graphblas.apply %v_cv, %two_i32 { apply_operator="gt" } : (tensor<?xi32, #CV>, i32) to tensor<?xi8, #CV>
    graphblas.print_tensor %120 { level=0 } : tensor<?xi8, #CV>

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