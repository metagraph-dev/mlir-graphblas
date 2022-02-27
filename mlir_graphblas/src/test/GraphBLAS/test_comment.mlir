// RUN: graphblas-opt %s | graphblas-exec main | FileCheck %s

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

func @main() -> () {
    graphblas.comment { comment = "comment number 0" }
    %m = arith.constant sparse<[
      [0, 1], [0, 2],
      [1, 0], [1, 3], [1, 4],
      [3, 2]
    ], [1., 2., 3., 4., 5., 6.]> : tensor<4x5xf64>
    graphblas.comment { comment = "comment number 1" }
    %m_csr = sparse_tensor.convert %m : tensor<4x5xf64> to tensor<?x?xf64, #CSR>
    graphblas.comment { comment = "comment number 2" }
    %m_csc = sparse_tensor.convert %m : tensor<4x5xf64> to tensor<?x?xf64, #CSC>
    graphblas.comment { comment = "comment number 3" }

    // CSR apply generic
    //
    // CHECK:      pointers=(0, 2, 5, 5, 6)
    // CHECK-NEXT: indices=(1, 2, 0, 3, 4, 2)
    // CHECK-NEXT: values=(1, 2, 3, 4, 4.5, 4.5)
    //
    graphblas.comment { comment = "comment number 4" }
    %thunk_f64 = arith.constant 4.5 : f64
    graphblas.comment { comment = "comment number 5" }
    %0 = graphblas.apply_generic %m_csr : tensor<?x?xf64, #CSR> to tensor<?x?xf64, #CSR> {
      ^bb0(%val: f64):
        graphblas.comment { comment = "comment number 6" }
        %pick = arith.cmpf olt, %val, %thunk_f64 : f64
        graphblas.comment { comment = "comment number 7" }
        %result = arith.select %pick, %val, %thunk_f64 : f64
        graphblas.comment { comment = "comment number 8" }
        graphblas.yield transform_out %result : f64
    }
    graphblas.comment { comment = "comment number 9" }
    graphblas.print_tensor %0 { level=3 } : tensor<?x?xf64, #CSR>
    graphblas.comment { comment = "comment number 10" }

    return
}
