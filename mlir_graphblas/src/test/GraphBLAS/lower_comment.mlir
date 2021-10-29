// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

// CHECK-LABEL:   func @create_and_destroy(
// CHECK-SAME:                             %[[VAL_0:.*]]: index,
// CHECK-SAME:                             %[[VAL_1:.*]]: index) -> index {
// CHECK:           %[[VAL_2:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]] : index
// CHECK:           %[[VAL_3:.*]] = sparse_tensor.init{{\[}}%[[VAL_0]], %[[VAL_1]]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           sparse_tensor.release %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           return %[[VAL_2]] : index
// CHECK:         }

func @create_and_destroy(%nrows: index, %ncols: index) -> index {
    graphblas.comment { comment = "comment number 1" }
    %sum = arith.addi %nrows, %ncols : index
    graphblas.comment { comment = "comment number 2" }
    %new_tensor = sparse_tensor.init [%nrows, %ncols] : tensor<?x?xf64, #CSR64>
    graphblas.comment { comment = "comment number 3" }
    sparse_tensor.release %new_tensor : tensor<?x?xf64, #CSR64>
    return %sum : index
}

// CHECK-LABEL:   func @do_nothing_func() {
// CHECK-NEXT:           return
func @do_nothing_func() -> () {
    graphblas.comment { comment = "comment number 1" }
    graphblas.comment { comment = "comment number 2" }
    graphblas.comment { comment = "comment number 3" }
    return
}
