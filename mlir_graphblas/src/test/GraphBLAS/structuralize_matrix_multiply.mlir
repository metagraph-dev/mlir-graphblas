// RUN: graphblas-opt %s | graphblas-opt --graphblas-structuralize | FileCheck %s

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

// CHECK-LABEL:   func @matrix_multiply_plus_times(
// CHECK-SAME:                                     %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_2:.*]] = constant 0.000000e+00 : f64
// CHECK:           %[[VAL_3:.*]] = graphblas.matrix_multiply_generic %[[VAL_0]], %[[VAL_1]] {mask_complement = false} : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) to tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:             graphblas.yield add_identity %[[VAL_2]] : f64
// CHECK:           },  {
// CHECK:           ^bb0(%[[VAL_5:.*]]: f64, %[[VAL_6:.*]]: f64):
// CHECK:             %[[VAL_7:.*]] = addf %[[VAL_5]], %[[VAL_6]] : f64
// CHECK:             graphblas.yield add %[[VAL_7]] : f64
// CHECK:           },  {
// CHECK:           ^bb0(%[[VAL_8:.*]]: f64, %[[VAL_9:.*]]: f64):
// CHECK:             %[[VAL_10:.*]] = mulf %[[VAL_8]], %[[VAL_9]] : f64
// CHECK:             graphblas.yield mult %[[VAL_10]] : f64
// CHECK:           }
// CHECK:           return %[[VAL_11:.*]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

func @matrix_multiply_plus_times(%a: tensor<?x?xf64, #CSR64>, %b: tensor<?x?xf64, #CSC64>) -> tensor<?x?xf64, #CSR64> {
    %answer = graphblas.matrix_multiply %a, %b { semiring = "plus_times" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>) to tensor<?x?xf64, #CSR64>
    return %answer : tensor<?x?xf64, #CSR64>
}
