// RUN: graphblas-opt %s | graphblas-opt --graphblas-structuralize | FileCheck %s

#SparseVec64 = #sparse_tensor.encoding<{ 
    dimLevelType = [ "compressed" ], 
    pointerBitWidth = 64, 
    indexBitWidth = 64 
}>

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

// CHECK-LABEL:   func @apply_min_matrix(
// CHECK-SAME:                    %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                    %[[VAL_1:.*]]: f64) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_2:.*]] = graphblas.apply_generic %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f64):
// CHECK:             %[[VAL_4:.*]] = cmpf olt, %[[VAL_3]], %[[VAL_1]] : f64
// CHECK:             %[[VAL_5:.*]] = select %[[VAL_4]], %[[VAL_3]], %[[VAL_1]] : f64
// CHECK:             graphblas.yield transform_out %[[VAL_5]] : f64
// CHECK:           }
// CHECK:           return %[[VAL_6:.*]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

func @apply_min_matrix(%sparse_tensor: tensor<?x?xf64, #CSR64>, %thunk: f64) -> tensor<?x?xf64, #CSR64> {
    %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<?x?xf64, #CSR64>, f64) to tensor<?x?xf64, #CSR64>
    return %answer : tensor<?x?xf64, #CSR64>
}

// CHECK-LABEL:   builtin.func @apply_min_vector(
// CHECK-SAME:                                   %[[VAL_0:.*]]: tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: f64) -> tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_2:.*]] = graphblas.apply_generic %[[VAL_0]] : tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f64):
// CHECK:             %[[VAL_4:.*]] = cmpf olt, %[[VAL_3]], %[[VAL_1]] : f64
// CHECK:             %[[VAL_5:.*]] = select %[[VAL_4]], %[[VAL_3]], %[[VAL_1]] : f64
// CHECK:             graphblas.yield transform_out %[[VAL_5]] : f64
// CHECK:           }
// CHECK:           return %[[VAL_6:.*]] : tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

func @apply_min_vector(%sparse_tensor: tensor<8xf64, #SparseVec64>, %thunk: f64) -> tensor<8xf64, #SparseVec64> {
    %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<8xf64, #SparseVec64>, f64) to tensor<8xf64, #SparseVec64>
    return %answer : tensor<8xf64, #SparseVec64>
}
