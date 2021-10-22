// RUN: graphblas-opt %s | graphblas-opt --graphblas-structuralize | FileCheck %s

#CV64 = #sparse_tensor.encoding<{
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

// CHECK:           func @apply_left_thunk_min_float_matrix(%[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, %[[VAL_1:.*]]: f64) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_2:.*]] = graphblas.apply_generic %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:             ^bb0(%[[VAL_3:.*]]: f64):
// CHECK:               %[[VAL_4:.*]] = arith.cmpf olt, %[[VAL_3]], %[[VAL_1]] : f64
// CHECK:               %[[VAL_5:.*]] = select %[[VAL_4]], %[[VAL_3]], %[[VAL_1]] : f64
// CHECK:               graphblas.yield transform_out %[[VAL_5]] : f64
// CHECK:             }
// CHECK:             return %[[VAL_6:.*]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @apply_left_thunk_min_float_matrix(%sparse_tensor: tensor<?x?xf64, #CSR64>, %thunk: f64) -> tensor<?x?xf64, #CSR64> {
        %answer = graphblas.apply %thunk, %sparse_tensor { apply_operator = "min" } : (f64, tensor<?x?xf64, #CSR64>) to tensor<?x?xf64, #CSR64>
        return %answer : tensor<?x?xf64, #CSR64>
    }

// CHECK:           func @apply_left_thunk_min_float_vector(%[[VAL_7:.*]]: tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, %[[VAL_8:.*]]: f64) -> tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_9:.*]] = graphblas.apply_generic %[[VAL_7]] : tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:             ^bb0(%[[VAL_10:.*]]: f64):
// CHECK:               %[[VAL_11:.*]] = arith.cmpf olt, %[[VAL_10]], %[[VAL_8]] : f64
// CHECK:               %[[VAL_12:.*]] = select %[[VAL_11]], %[[VAL_10]], %[[VAL_8]] : f64
// CHECK:               graphblas.yield transform_out %[[VAL_12]] : f64
// CHECK:             }
// CHECK:             return %[[VAL_13:.*]] : tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @apply_left_thunk_min_float_vector(%sparse_tensor: tensor<8xf64, #CV64>, %thunk: f64) -> tensor<8xf64, #CV64> {
        %answer = graphblas.apply %thunk, %sparse_tensor { apply_operator = "min" } : (f64, tensor<8xf64, #CV64>) to tensor<8xf64, #CV64>
        return %answer : tensor<8xf64, #CV64>
    }

// CHECK:           func @apply_right_thunk_min_float_matrix(%[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, %[[VAL_1:.*]]: f64) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_2:.*]] = graphblas.apply_generic %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:             ^bb0(%[[VAL_3:.*]]: f64):
// CHECK:               %[[VAL_4:.*]] = arith.cmpf olt, %[[VAL_3]], %[[VAL_1]] : f64
// CHECK:               %[[VAL_5:.*]] = select %[[VAL_4]], %[[VAL_3]], %[[VAL_1]] : f64
// CHECK:               graphblas.yield transform_out %[[VAL_5]] : f64
// CHECK:             }
// CHECK:             return %[[VAL_6:.*]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @apply_right_thunk_min_float_matrix(%sparse_tensor: tensor<?x?xf64, #CSR64>, %thunk: f64) -> tensor<?x?xf64, #CSR64> {
        %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<?x?xf64, #CSR64>, f64) to tensor<?x?xf64, #CSR64>
        return %answer : tensor<?x?xf64, #CSR64>
    }

// CHECK:           func @apply_right_thunk_min_float_vector(%[[VAL_7:.*]]: tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, %[[VAL_8:.*]]: f64) -> tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_9:.*]] = graphblas.apply_generic %[[VAL_7]] : tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:             ^bb0(%[[VAL_10:.*]]: f64):
// CHECK:               %[[VAL_11:.*]] = arith.cmpf olt, %[[VAL_10]], %[[VAL_8]] : f64
// CHECK:               %[[VAL_12:.*]] = select %[[VAL_11]], %[[VAL_10]], %[[VAL_8]] : f64
// CHECK:               graphblas.yield transform_out %[[VAL_12]] : f64
// CHECK:             }
// CHECK:             return %[[VAL_13:.*]] : tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @apply_right_thunk_min_float_vector(%sparse_tensor: tensor<8xf64, #CV64>, %thunk: f64) -> tensor<8xf64, #CV64> {
        %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<8xf64, #CV64>, f64) to tensor<8xf64, #CV64>
        return %answer : tensor<8xf64, #CV64>
    }

// CHECK-LABEL:   func @apply_left_thunk_div_float_matrix(
// CHECK-SAME:                                                    %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                                    %[[VAL_1:.*]]: f64) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_2:.*]] = graphblas.apply_generic %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f64):
// CHECK:             %[[VAL_4:.*]] = arith.divf %[[VAL_1]], %[[VAL_3]] : f64
// CHECK:             graphblas.yield transform_out %[[VAL_4]] : f64
// CHECK:           }
// CHECK:           return %[[VAL_5:.*]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

    func @apply_left_thunk_div_float_matrix(%sparse_tensor: tensor<?x?xf64, #CSR64>, %thunk: f64) -> tensor<?x?xf64, #CSR64> {
        %answer = graphblas.apply %thunk, %sparse_tensor { apply_operator = "div" } : (f64, tensor<?x?xf64, #CSR64>) to tensor<?x?xf64, #CSR64>
        return %answer : tensor<?x?xf64, #CSR64>
    }

// CHECK-LABEL:   func @apply_left_thunk_div_float_vector(
// CHECK-SAME:                                                    %[[VAL_0:.*]]: tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                                    %[[VAL_1:.*]]: f64) -> tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_2:.*]] = graphblas.apply_generic %[[VAL_0]] : tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f64):
// CHECK:             %[[VAL_4:.*]] = arith.divf %[[VAL_1]], %[[VAL_3]] : f64
// CHECK:             graphblas.yield transform_out %[[VAL_4]] : f64
// CHECK:           }
// CHECK:           return %[[VAL_5:.*]] : tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

    func @apply_left_thunk_div_float_vector(%sparse_tensor: tensor<8xf64, #CV64>, %thunk: f64) -> tensor<8xf64, #CV64> {
        %answer = graphblas.apply %thunk, %sparse_tensor { apply_operator = "div" } : (f64, tensor<8xf64, #CV64>) to tensor<8xf64, #CV64>
        return %answer : tensor<8xf64, #CV64>
    }

// CHECK-LABEL:   func @apply_right_thunk_div_float_matrix(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                                     %[[VAL_1:.*]]: f64) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_2:.*]] = graphblas.apply_generic %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f64):
// CHECK:             %[[VAL_4:.*]] = arith.divf %[[VAL_3]], %[[VAL_1]] : f64
// CHECK:             graphblas.yield transform_out %[[VAL_4]] : f64
// CHECK:           }
// CHECK:           return %[[VAL_5:.*]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

    func @apply_right_thunk_div_float_matrix(%sparse_tensor: tensor<?x?xf64, #CSR64>, %thunk: f64) -> tensor<?x?xf64, #CSR64> {
        %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "div" } : (tensor<?x?xf64, #CSR64>, f64) to tensor<?x?xf64, #CSR64>
        return %answer : tensor<?x?xf64, #CSR64>
    }

// CHECK-LABEL:   func @apply_right_thunk_div_float_vector(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                                     %[[VAL_1:.*]]: f64) -> tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_2:.*]] = graphblas.apply_generic %[[VAL_0]] : tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f64):
// CHECK:             %[[VAL_4:.*]] = arith.divf %[[VAL_3]], %[[VAL_1]] : f64
// CHECK:             graphblas.yield transform_out %[[VAL_4]] : f64
// CHECK:           }
// CHECK:           return %[[VAL_5:.*]] : tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

    func @apply_right_thunk_div_float_vector(%sparse_tensor: tensor<8xf64, #CV64>, %thunk: f64) -> tensor<8xf64, #CV64> {
        %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "div" } : (tensor<8xf64, #CV64>, f64) to tensor<8xf64, #CV64>
        return %answer : tensor<8xf64, #CV64>
    }

// CHECK:           func @apply_abs_float_matrix(%[[VAL_14:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_15:.*]] = graphblas.apply_generic %[[VAL_14]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:             ^bb0(%[[VAL_16:.*]]: f64):
// CHECK:               %[[VAL_17:.*]] = math.abs %[[VAL_16]] : f64
// CHECK:               graphblas.yield transform_out %[[VAL_17]] : f64
// CHECK:             }
// CHECK:             return %[[VAL_18:.*]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @apply_abs_float_matrix(%sparse_tensor: tensor<?x?xf64, #CSR64>) -> tensor<?x?xf64, #CSR64> {
        %answer = graphblas.apply %sparse_tensor { apply_operator = "abs" } : (tensor<?x?xf64, #CSR64>) to tensor<?x?xf64, #CSR64>
        return %answer : tensor<?x?xf64, #CSR64>
    }

// CHECK:           func @apply_abs_float_vector(%[[VAL_19:.*]]: tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_20:.*]] = graphblas.apply_generic %[[VAL_19]] : tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:             ^bb0(%[[VAL_21:.*]]: f64):
// CHECK:               %[[VAL_22:.*]] = math.abs %[[VAL_21]] : f64
// CHECK:               graphblas.yield transform_out %[[VAL_22]] : f64
// CHECK:             }
// CHECK:             return %[[VAL_23:.*]] : tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @apply_abs_float_vector(%sparse_tensor: tensor<8xf64, #CV64>) -> tensor<8xf64, #CV64> {
        %answer = graphblas.apply %sparse_tensor { apply_operator = "abs" } : (tensor<8xf64, #CV64>) to tensor<8xf64, #CV64>
        return %answer : tensor<8xf64, #CV64>
    }

// CHECK-LABEL:   func @apply_minv_float_matrix(
// CHECK-SAME:                                          %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1.000000e+00 : f64
// CHECK:           %[[VAL_2:.*]] = graphblas.apply_generic %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f64):
// CHECK:             %[[VAL_4:.*]] = arith.divf %[[VAL_1]], %[[VAL_3]] : f64
// CHECK:             graphblas.yield transform_out %[[VAL_4]] : f64
// CHECK:           }
// CHECK:           return %[[VAL_5:.*]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

    func @apply_minv_float_matrix(%sparse_tensor: tensor<?x?xf64, #CSR64>) -> tensor<?x?xf64, #CSR64> {
        %answer = graphblas.apply %sparse_tensor { apply_operator = "minv" } : (tensor<?x?xf64, #CSR64>) to tensor<?x?xf64, #CSR64>
        return %answer : tensor<?x?xf64, #CSR64>
    }
    
// CHECK-LABEL:   func @apply_minv_float_vector(
// CHECK-SAME:                                          %[[VAL_0:.*]]: tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1.000000e+00 : f64
// CHECK:           %[[VAL_2:.*]] = graphblas.apply_generic %[[VAL_0]] : tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f64):
// CHECK:             %[[VAL_4:.*]] = arith.divf %[[VAL_1]], %[[VAL_3]] : f64
// CHECK:             graphblas.yield transform_out %[[VAL_4]] : f64
// CHECK:           }
// CHECK:           return %[[VAL_5:.*]] : tensor<8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

    func @apply_minv_float_vector(%sparse_tensor: tensor<8xf64, #CV64>) -> tensor<8xf64, #CV64> {
        %answer = graphblas.apply %sparse_tensor { apply_operator = "minv" } : (tensor<8xf64, #CV64>) to tensor<8xf64, #CV64>
        return %answer : tensor<8xf64, #CV64>
    }

}

module {

// CHECK:           func @apply_left_thunk_min_int_matrix(%[[VAL_0:.*]]: tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, %[[VAL_1:.*]]: i8) -> tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_2:.*]] = graphblas.apply_generic %[[VAL_0]] : tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:             ^bb0(%[[VAL_3:.*]]: i8):
// CHECK:               %[[VAL_4:.*]] = arith.cmpi slt, %[[VAL_3]], %[[VAL_1]] : i8
// CHECK:               %[[VAL_5:.*]] = select %[[VAL_4]], %[[VAL_3]], %[[VAL_1]] : i8
// CHECK:               graphblas.yield transform_out %[[VAL_5]] : i8
// CHECK:             }
// CHECK:             return %[[VAL_6:.*]] : tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @apply_left_thunk_min_int_matrix(%sparse_tensor: tensor<?x?xi8, #CSR64>, %thunk: i8) -> tensor<?x?xi8, #CSR64> {
        %answer = graphblas.apply %thunk, %sparse_tensor { apply_operator = "min" } : (i8, tensor<?x?xi8, #CSR64>) to tensor<?x?xi8, #CSR64>
        return %answer : tensor<?x?xi8, #CSR64>
    }

// CHECK:           func @apply_left_thunk_min_int_vector(%[[VAL_7:.*]]: tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, %[[VAL_8:.*]]: i8) -> tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_9:.*]] = graphblas.apply_generic %[[VAL_7]] : tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:             ^bb0(%[[VAL_10:.*]]: i8):
// CHECK:               %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_8]] : i8
// CHECK:               %[[VAL_12:.*]] = select %[[VAL_11]], %[[VAL_10]], %[[VAL_8]] : i8
// CHECK:               graphblas.yield transform_out %[[VAL_12]] : i8
// CHECK:             }
// CHECK:             return %[[VAL_13:.*]] : tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @apply_left_thunk_min_int_vector(%sparse_tensor: tensor<8xi8, #CV64>, %thunk: i8) -> tensor<8xi8, #CV64> {
        %answer = graphblas.apply %thunk, %sparse_tensor { apply_operator = "min" } : (i8, tensor<8xi8, #CV64>) to tensor<8xi8, #CV64>
        return %answer : tensor<8xi8, #CV64>
    }

// CHECK:           func @apply_right_thunk_min_int_matrix(%[[VAL_0:.*]]: tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, %[[VAL_1:.*]]: i8) -> tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_2:.*]] = graphblas.apply_generic %[[VAL_0]] : tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:             ^bb0(%[[VAL_3:.*]]: i8):
// CHECK:               %[[VAL_4:.*]] = arith.cmpi slt, %[[VAL_3]], %[[VAL_1]] : i8
// CHECK:               %[[VAL_5:.*]] = select %[[VAL_4]], %[[VAL_3]], %[[VAL_1]] : i8
// CHECK:               graphblas.yield transform_out %[[VAL_5]] : i8
// CHECK:             }
// CHECK:             return %[[VAL_6:.*]] : tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @apply_right_thunk_min_int_matrix(%sparse_tensor: tensor<?x?xi8, #CSR64>, %thunk: i8) -> tensor<?x?xi8, #CSR64> {
        %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<?x?xi8, #CSR64>, i8) to tensor<?x?xi8, #CSR64>
        return %answer : tensor<?x?xi8, #CSR64>
    }

// CHECK:           func @apply_right_thunk_min_int_vector(%[[VAL_7:.*]]: tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, %[[VAL_8:.*]]: i8) -> tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_9:.*]] = graphblas.apply_generic %[[VAL_7]] : tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:             ^bb0(%[[VAL_10:.*]]: i8):
// CHECK:               %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_8]] : i8
// CHECK:               %[[VAL_12:.*]] = select %[[VAL_11]], %[[VAL_10]], %[[VAL_8]] : i8
// CHECK:               graphblas.yield transform_out %[[VAL_12]] : i8
// CHECK:             }
// CHECK:             return %[[VAL_13:.*]] : tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @apply_right_thunk_min_int_vector(%sparse_tensor: tensor<8xi8, #CV64>, %thunk: i8) -> tensor<8xi8, #CV64> {
        %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<8xi8, #CV64>, i8) to tensor<8xi8, #CV64>
        return %answer : tensor<8xi8, #CV64>
    }

// CHECK-LABEL:   func @apply_left_thunk_div_int_matrix(
// CHECK-SAME:                                                  %[[VAL_0:.*]]: tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                                  %[[VAL_1:.*]]: i8) -> tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_2:.*]] = graphblas.apply_generic %[[VAL_0]] : tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:           ^bb0(%[[VAL_3:.*]]: i8):
// CHECK:             %[[VAL_4:.*]] = arith.divsi %[[VAL_1]], %[[VAL_3]] : i8
// CHECK:             graphblas.yield transform_out %[[VAL_4]] : i8
// CHECK:           }
// CHECK:           return %[[VAL_5:.*]] : tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

    func @apply_left_thunk_div_int_matrix(%sparse_tensor: tensor<?x?xi8, #CSR64>, %thunk: i8) -> tensor<?x?xi8, #CSR64> {
        %answer = graphblas.apply %thunk, %sparse_tensor { apply_operator = "div" } : (i8, tensor<?x?xi8, #CSR64>) to tensor<?x?xi8, #CSR64>
        return %answer : tensor<?x?xi8, #CSR64>
    }

// CHECK-LABEL:   func @apply_left_thunk_div_int_vector(
// CHECK-SAME:                                                  %[[VAL_0:.*]]: tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                                  %[[VAL_1:.*]]: i8) -> tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_2:.*]] = graphblas.apply_generic %[[VAL_0]] : tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:           ^bb0(%[[VAL_3:.*]]: i8):
// CHECK:             %[[VAL_4:.*]] = arith.divsi %[[VAL_1]], %[[VAL_3]] : i8
// CHECK:             graphblas.yield transform_out %[[VAL_4]] : i8
// CHECK:           }
// CHECK:           return %[[VAL_5:.*]] : tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

    func @apply_left_thunk_div_int_vector(%sparse_tensor: tensor<8xi8, #CV64>, %thunk: i8) -> tensor<8xi8, #CV64> {
        %answer = graphblas.apply %thunk, %sparse_tensor { apply_operator = "div" } : (i8, tensor<8xi8, #CV64>) to tensor<8xi8, #CV64>
        return %answer : tensor<8xi8, #CV64>
    }

// CHECK-LABEL:   func @apply_right_thunk_div_int_matrix(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                                   %[[VAL_1:.*]]: i8) -> tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_2:.*]] = graphblas.apply_generic %[[VAL_0]] : tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:           ^bb0(%[[VAL_3:.*]]: i8):
// CHECK:             %[[VAL_4:.*]] = arith.divsi %[[VAL_3]], %[[VAL_1]] : i8
// CHECK:             graphblas.yield transform_out %[[VAL_4]] : i8
// CHECK:           }
// CHECK:           return %[[VAL_5:.*]] : tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

    func @apply_right_thunk_div_int_matrix(%sparse_tensor: tensor<?x?xi8, #CSR64>, %thunk: i8) -> tensor<?x?xi8, #CSR64> {
        %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "div" } : (tensor<?x?xi8, #CSR64>, i8) to tensor<?x?xi8, #CSR64>
        return %answer : tensor<?x?xi8, #CSR64>
    }
    
// CHECK-LABEL:   func @apply_right_thunk_div_int_vector(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                                   %[[VAL_1:.*]]: i8) -> tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_2:.*]] = graphblas.apply_generic %[[VAL_0]] : tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:           ^bb0(%[[VAL_3:.*]]: i8):
// CHECK:             %[[VAL_4:.*]] = arith.divsi %[[VAL_3]], %[[VAL_1]] : i8
// CHECK:             graphblas.yield transform_out %[[VAL_4]] : i8
// CHECK:           }
// CHECK:           return %[[VAL_5:.*]] : tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

    func @apply_right_thunk_div_int_vector(%sparse_tensor: tensor<8xi8, #CV64>, %thunk: i8) -> tensor<8xi8, #CV64> {
        %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "div" } : (tensor<8xi8, #CV64>, i8) to tensor<8xi8, #CV64>
        return %answer : tensor<8xi8, #CV64>
    }

// CHECK:           func @apply_abs_int_matrix(%[[VAL_0:.*]]: tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_1:.*]] = arith.constant 7 : i8
// CHECK:             %[[VAL_2:.*]] = graphblas.apply_generic %[[VAL_0]] : tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:             ^bb0(%[[VAL_3:.*]]: i8):
// CHECK:               %[[VAL_4:.*]] = arith.shrsi %[[VAL_3]], %[[VAL_1]] : i8
// CHECK:               %[[VAL_5:.*]] = arith.addi %[[VAL_4]], %[[VAL_3]] : i8
// CHECK:               %[[VAL_6:.*]] = arith.xori %[[VAL_4]], %[[VAL_5]] : i8
// CHECK:               graphblas.yield transform_out %[[VAL_6]] : i8
// CHECK:             }
// CHECK:             return %[[VAL_7:.*]] : tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @apply_abs_int_matrix(%sparse_tensor: tensor<?x?xi8, #CSR64>) -> tensor<?x?xi8, #CSR64> {
        %answer = graphblas.apply %sparse_tensor { apply_operator = "abs" } : (tensor<?x?xi8, #CSR64>) to tensor<?x?xi8, #CSR64>
        return %answer : tensor<?x?xi8, #CSR64>
    }

// CHECK:           func @apply_abs_int_vector(%[[VAL_8:.*]]: tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_9:.*]] = arith.constant 7 : i8
// CHECK:             %[[VAL_10:.*]] = graphblas.apply_generic %[[VAL_8]] : tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:             ^bb0(%[[VAL_11:.*]]: i8):
// CHECK:               %[[VAL_12:.*]] = arith.shrsi %[[VAL_11]], %[[VAL_9]] : i8
// CHECK:               %[[VAL_13:.*]] = arith.addi %[[VAL_12]], %[[VAL_11]] : i8
// CHECK:               %[[VAL_14:.*]] = arith.xori %[[VAL_12]], %[[VAL_13]] : i8
// CHECK:               graphblas.yield transform_out %[[VAL_14]] : i8
// CHECK:             }
// CHECK:             return %[[VAL_15:.*]] : tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @apply_abs_int_vector(%sparse_tensor: tensor<8xi8, #CV64>) -> tensor<8xi8, #CV64> {
        %answer = graphblas.apply %sparse_tensor { apply_operator = "abs" } : (tensor<8xi8, #CV64>) to tensor<8xi8, #CV64>
        return %answer : tensor<8xi8, #CV64>
    }

// CHECK-LABEL:   func @apply_minv_int_matrix(
// CHECK-SAME:                                        %[[VAL_0:.*]]: tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 7 : i8
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i8
// CHECK:           %[[VAL_3:.*]] = graphblas.apply_generic %[[VAL_0]] : tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:           ^bb0(%[[VAL_4:.*]]: i8):
// CHECK:             %[[VAL_5:.*]] = arith.shrsi %[[VAL_4]], %[[VAL_1]] : i8
// CHECK:             %[[VAL_6:.*]] = arith.addi %[[VAL_5]], %[[VAL_4]] : i8
// CHECK:             %[[VAL_7:.*]] = arith.xori %[[VAL_5]], %[[VAL_6]] : i8
// CHECK:             %[[VAL_8:.*]] = arith.cmpi eq, %[[VAL_7]], %[[VAL_2]] : i8
// CHECK:             %[[VAL_9:.*]] = arith.extsi %[[VAL_8]] : i1 to i8
// CHECK:             %[[VAL_10:.*]] = arith.andi %[[VAL_9]], %[[VAL_4]] : i8
// CHECK:             graphblas.yield transform_out %[[VAL_10]] : i8
// CHECK:           }
// CHECK:           return %[[VAL_11:.*]] : tensor<?x?xi8, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

    func @apply_minv_int_matrix(%sparse_tensor: tensor<?x?xi8, #CSR64>) -> tensor<?x?xi8, #CSR64> {
        %answer = graphblas.apply %sparse_tensor { apply_operator = "minv" } : (tensor<?x?xi8, #CSR64>) to tensor<?x?xi8, #CSR64>
        return %answer : tensor<?x?xi8, #CSR64>
    }

// CHECK-LABEL:   func @apply_minv_int_vector(
// CHECK-SAME:                                        %[[VAL_0:.*]]: tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 7 : i8
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i8
// CHECK:           %[[VAL_3:.*]] = graphblas.apply_generic %[[VAL_0]] : tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:           ^bb0(%[[VAL_4:.*]]: i8):
// CHECK:             %[[VAL_5:.*]] = arith.shrsi %[[VAL_4]], %[[VAL_1]] : i8
// CHECK:             %[[VAL_6:.*]] = arith.addi %[[VAL_5]], %[[VAL_4]] : i8
// CHECK:             %[[VAL_7:.*]] = arith.xori %[[VAL_5]], %[[VAL_6]] : i8
// CHECK:             %[[VAL_8:.*]] = arith.cmpi eq, %[[VAL_7]], %[[VAL_2]] : i8
// CHECK:             %[[VAL_9:.*]] = arith.extsi %[[VAL_8]] : i1 to i8
// CHECK:             %[[VAL_10:.*]] = arith.andi %[[VAL_9]], %[[VAL_4]] : i8
// CHECK:             graphblas.yield transform_out %[[VAL_10]] : i8
// CHECK:           }
// CHECK:           return %[[VAL_11:.*]] : tensor<8xi8, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

    func @apply_minv_int_vector(%sparse_tensor: tensor<8xi8, #CV64>) -> tensor<8xi8, #CV64> {
        %answer = graphblas.apply %sparse_tensor { apply_operator = "minv" } : (tensor<8xi8, #CV64>) to tensor<8xi8, #CV64>
        return %answer : tensor<8xi8, #CV64>
    }

}
