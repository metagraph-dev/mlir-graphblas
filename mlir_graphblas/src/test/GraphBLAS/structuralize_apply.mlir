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

module {

// CHECK:           builtin.func @apply_min_float_matrix(%[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, %[[VAL_1:.*]]: f64) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_2:.*]] = graphblas.apply_generic %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:             ^bb0(%[[VAL_3:.*]]: f64):
// CHECK:               %[[VAL_4:.*]] = cmpf olt, %[[VAL_3]], %[[VAL_1]] : f64
// CHECK:               %[[VAL_5:.*]] = select %[[VAL_4]], %[[VAL_3]], %[[VAL_1]] : f64
// CHECK:               graphblas.yield transform_out %[[VAL_5]] : f64
// CHECK:             }
// CHECK:             return %[[VAL_6:.*]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @apply_min_float_matrix(%sparse_tensor: tensor<?x?xf64, #CSR64>, %thunk: f64) -> tensor<?x?xf64, #CSR64> {
        %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<?x?xf64, #CSR64>, f64) to tensor<?x?xf64, #CSR64>
        return %answer : tensor<?x?xf64, #CSR64>
    }

// CHECK:           builtin.func @apply_min_float_vector(%[[VAL_7:.*]]: tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, %[[VAL_8:.*]]: f64) -> tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_9:.*]] = graphblas.apply_generic %[[VAL_7]] : tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:             ^bb0(%[[VAL_10:.*]]: f64):
// CHECK:               %[[VAL_11:.*]] = cmpf olt, %[[VAL_10]], %[[VAL_8]] : f64
// CHECK:               %[[VAL_12:.*]] = select %[[VAL_11]], %[[VAL_10]], %[[VAL_8]] : f64
// CHECK:               graphblas.yield transform_out %[[VAL_12]] : f64
// CHECK:             }
// CHECK:             return %[[VAL_13:.*]] : tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @apply_min_float_vector(%sparse_tensor: tensor<8xf64, #SparseVec64>, %thunk: f64) -> tensor<8xf64, #SparseVec64> {
        %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<8xf64, #SparseVec64>, f64) to tensor<8xf64, #SparseVec64>
        return %answer : tensor<8xf64, #SparseVec64>
    }

// CHECK:           builtin.func @apply_abs_float_matrix(%[[VAL_14:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_15:.*]] = graphblas.apply_generic %[[VAL_14]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:             ^bb0(%[[VAL_16:.*]]: f64):
// CHECK:               %[[VAL_17:.*]] = absf %[[VAL_16]] : f64
// CHECK:               graphblas.yield transform_out %[[VAL_17]] : f64
// CHECK:             }
// CHECK:             return %[[VAL_18:.*]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @apply_abs_float_matrix(%sparse_tensor: tensor<?x?xf64, #CSR64>) -> tensor<?x?xf64, #CSR64> {
        %answer = graphblas.apply %sparse_tensor { apply_operator = "abs" } : (tensor<?x?xf64, #CSR64>) to tensor<?x?xf64, #CSR64>
        return %answer : tensor<?x?xf64, #CSR64>
    }

// CHECK:           builtin.func @apply_abs_float_vector(%[[VAL_19:.*]]: tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_20:.*]] = graphblas.apply_generic %[[VAL_19]] : tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:             ^bb0(%[[VAL_21:.*]]: f64):
// CHECK:               %[[VAL_22:.*]] = absf %[[VAL_21]] : f64
// CHECK:               graphblas.yield transform_out %[[VAL_22]] : f64
// CHECK:             }
// CHECK:             return %[[VAL_23:.*]] : tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @apply_abs_float_vector(%sparse_tensor: tensor<8xf64, #SparseVec64>) -> tensor<8xf64, #SparseVec64> {
        %answer = graphblas.apply %sparse_tensor { apply_operator = "abs" } : (tensor<8xf64, #SparseVec64>) to tensor<8xf64, #SparseVec64>
        return %answer : tensor<8xf64, #SparseVec64>
    }

// CHECK-LABEL:   builtin.func @apply_minv_float_matrix(
// CHECK-SAME:                                          %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_1:.*]] = constant 1.000000e+00 : f64
// CHECK:           %[[VAL_2:.*]] = graphblas.apply_generic %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f64):
// CHECK:             %[[VAL_4:.*]] = divf %[[VAL_1]], %[[VAL_3]] : f64
// CHECK:             graphblas.yield transform_out %[[VAL_4]] : f64
// CHECK:           }
// CHECK:           return %[[VAL_5:.*]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

    func @apply_minv_float_matrix(%sparse_tensor: tensor<?x?xf64, #CSR64>) -> tensor<?x?xf64, #CSR64> {
        %answer = graphblas.apply %sparse_tensor { apply_operator = "minv" } : (tensor<?x?xf64, #CSR64>) to tensor<?x?xf64, #CSR64>
        return %answer : tensor<?x?xf64, #CSR64>
    }
    
// CHECK-LABEL:   builtin.func @apply_minv_float_vector(
// CHECK-SAME:                                          %[[VAL_0:.*]]: tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_1:.*]] = constant 1.000000e+00 : f64
// CHECK:           %[[VAL_2:.*]] = graphblas.apply_generic %[[VAL_0]] : tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f64):
// CHECK:             %[[VAL_4:.*]] = divf %[[VAL_1]], %[[VAL_3]] : f64
// CHECK:             graphblas.yield transform_out %[[VAL_4]] : f64
// CHECK:           }
// CHECK:           return %[[VAL_5:.*]] : tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

    func @apply_minv_float_vector(%sparse_tensor: tensor<8xf64, #SparseVec64>) -> tensor<8xf64, #SparseVec64> {
        %answer = graphblas.apply %sparse_tensor { apply_operator = "minv" } : (tensor<8xf64, #SparseVec64>) to tensor<8xf64, #SparseVec64>
        return %answer : tensor<8xf64, #SparseVec64>
    }

}

module {

// CHECK:           builtin.func @apply_min_int_matrix(%[[VAL_0:.*]]: tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, %[[VAL_1:.*]]: i8) -> tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_2:.*]] = graphblas.apply_generic %[[VAL_0]] : tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:             ^bb0(%[[VAL_3:.*]]: i8):
// CHECK:               %[[VAL_4:.*]] = cmpi slt, %[[VAL_3]], %[[VAL_1]] : i8
// CHECK:               %[[VAL_5:.*]] = select %[[VAL_4]], %[[VAL_3]], %[[VAL_1]] : i8
// CHECK:               graphblas.yield transform_out %[[VAL_5]] : i8
// CHECK:             }
// CHECK:             return %[[VAL_6:.*]] : tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @apply_min_int_matrix(%sparse_tensor: tensor<?x?xi8, #CSR64>, %thunk: i8) -> tensor<?x?xi8, #CSR64> {
        %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<?x?xi8, #CSR64>, i8) to tensor<?x?xi8, #CSR64>
        return %answer : tensor<?x?xi8, #CSR64>
    }

// CHECK:           builtin.func @apply_min_int_vector(%[[VAL_7:.*]]: tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, %[[VAL_8:.*]]: i8) -> tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_9:.*]] = graphblas.apply_generic %[[VAL_7]] : tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:             ^bb0(%[[VAL_10:.*]]: i8):
// CHECK:               %[[VAL_11:.*]] = cmpi slt, %[[VAL_10]], %[[VAL_8]] : i8
// CHECK:               %[[VAL_12:.*]] = select %[[VAL_11]], %[[VAL_10]], %[[VAL_8]] : i8
// CHECK:               graphblas.yield transform_out %[[VAL_12]] : i8
// CHECK:             }
// CHECK:             return %[[VAL_13:.*]] : tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @apply_min_int_vector(%sparse_tensor: tensor<8xi8, #SparseVec64>, %thunk: i8) -> tensor<8xi8, #SparseVec64> {
        %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<8xi8, #SparseVec64>, i8) to tensor<8xi8, #SparseVec64>
        return %answer : tensor<8xi8, #SparseVec64>
    }

// CHECK:           builtin.func @apply_abs_int_matrix(%[[VAL_0:.*]]: tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_1:.*]] = constant 7 : i8
// CHECK:             %[[VAL_2:.*]] = graphblas.apply_generic %[[VAL_0]] : tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:             ^bb0(%[[VAL_3:.*]]: i8):
// CHECK:               %[[VAL_4:.*]] = shift_right_signed %[[VAL_3]], %[[VAL_1]] : i8
// CHECK:               %[[VAL_5:.*]] = addi %[[VAL_4]], %[[VAL_3]] : i8
// CHECK:               %[[VAL_6:.*]] = xor %[[VAL_4]], %[[VAL_5]] : i8
// CHECK:               graphblas.yield transform_out %[[VAL_6]] : i8
// CHECK:             }
// CHECK:             return %[[VAL_7:.*]] : tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @apply_abs_int_matrix(%sparse_tensor: tensor<?x?xi8, #CSR64>) -> tensor<?x?xi8, #CSR64> {
        %answer = graphblas.apply %sparse_tensor { apply_operator = "abs" } : (tensor<?x?xi8, #CSR64>) to tensor<?x?xi8, #CSR64>
        return %answer : tensor<?x?xi8, #CSR64>
    }

// CHECK:           builtin.func @apply_abs_int_vector(%[[VAL_8:.*]]: tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_9:.*]] = constant 7 : i8
// CHECK:             %[[VAL_10:.*]] = graphblas.apply_generic %[[VAL_8]] : tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:             ^bb0(%[[VAL_11:.*]]: i8):
// CHECK:               %[[VAL_12:.*]] = shift_right_signed %[[VAL_11]], %[[VAL_9]] : i8
// CHECK:               %[[VAL_13:.*]] = addi %[[VAL_12]], %[[VAL_11]] : i8
// CHECK:               %[[VAL_14:.*]] = xor %[[VAL_12]], %[[VAL_13]] : i8
// CHECK:               graphblas.yield transform_out %[[VAL_14]] : i8
// CHECK:             }
// CHECK:             return %[[VAL_15:.*]] : tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @apply_abs_int_vector(%sparse_tensor: tensor<8xi8, #SparseVec64>) -> tensor<8xi8, #SparseVec64> {
        %answer = graphblas.apply %sparse_tensor { apply_operator = "abs" } : (tensor<8xi8, #SparseVec64>) to tensor<8xi8, #SparseVec64>
        return %answer : tensor<8xi8, #SparseVec64>
    }

// CHECK-LABEL:   builtin.func @apply_minv_int_matrix(
// CHECK-SAME:                                        %[[VAL_0:.*]]: tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_1:.*]] = constant 1 : i8
// CHECK:           %[[VAL_2:.*]] = graphblas.apply_generic %[[VAL_0]] : tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:           ^bb0(%[[VAL_3:.*]]: i8):
// CHECK:             %[[VAL_4:.*]] = divi_signed %[[VAL_1]], %[[VAL_3]] : i8
// CHECK:             graphblas.yield transform_out %[[VAL_4]] : i8
// CHECK:           }
// CHECK:           return %[[VAL_5:.*]] : tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

    func @apply_minv_int_matrix(%sparse_tensor: tensor<?x?xi8, #CSR64>) -> tensor<?x?xi8, #CSR64> {
        %answer = graphblas.apply %sparse_tensor { apply_operator = "minv" } : (tensor<?x?xi8, #CSR64>) to tensor<?x?xi8, #CSR64>
        return %answer : tensor<?x?xi8, #CSR64>
    }

// CHECK-LABEL:   builtin.func @apply_minv_int_vector(
// CHECK-SAME:                                        %[[VAL_0:.*]]: tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_1:.*]] = constant 1 : i8
// CHECK:           %[[VAL_2:.*]] = graphblas.apply_generic %[[VAL_0]] : tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:           ^bb0(%[[VAL_3:.*]]: i8):
// CHECK:             %[[VAL_4:.*]] = divi_signed %[[VAL_1]], %[[VAL_3]] : i8
// CHECK:             graphblas.yield transform_out %[[VAL_4]] : i8
// CHECK:           }
// CHECK:           return %[[VAL_5:.*]] : tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

    func @apply_minv_int_vector(%sparse_tensor: tensor<8xi8, #SparseVec64>) -> tensor<8xi8, #SparseVec64> {
        %answer = graphblas.apply %sparse_tensor { apply_operator = "minv" } : (tensor<8xi8, #SparseVec64>) to tensor<8xi8, #SparseVec64>
        return %answer : tensor<8xi8, #SparseVec64>
    }

}
