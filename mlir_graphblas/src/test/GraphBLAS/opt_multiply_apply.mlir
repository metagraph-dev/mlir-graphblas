// RUN: graphblas-opt %s | graphblas-opt --graphblas-structuralize --graphblas-optimize | FileCheck %s

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


// CHECK-LABEL:   func @fuse_adjacent(
// CHECK-SAME:                        %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                        %[[VAL_1:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                        %[[VAL_2:.*]]: f64) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_3:.*]] = constant 0.000000e+00 : f64
// CHECK:           %[[VAL_4:.*]] = graphblas.matrix_multiply_generic %[[VAL_0]], %[[VAL_1]] : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) to tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:             graphblas.yield add_identity %[[VAL_3]] : f64
// CHECK:           },  {
// CHECK:           ^bb0(%[[VAL_5:.*]]: f64, %[[VAL_6:.*]]: f64):
// CHECK:             %[[VAL_7:.*]] = addf %[[VAL_5]], %[[VAL_6]] : f64
// CHECK:             graphblas.yield add %[[VAL_7]] : f64
// CHECK:           },  {
// CHECK:           ^bb0(%[[VAL_8:.*]]: f64, %[[VAL_9:.*]]: f64):
// CHECK:             %[[VAL_10:.*]] = addf %[[VAL_8]], %[[VAL_9]] : f64
// CHECK:             graphblas.yield mult %[[VAL_10]] : f64
// CHECK:           },  {
// CHECK:           ^bb0(%[[VAL_11:.*]]: f64):
// CHECK:             %[[VAL_12:.*]] = cmpf olt, %[[VAL_11]], %[[VAL_2]] : f64
// CHECK:             %[[VAL_13:.*]] = select %[[VAL_12]], %[[VAL_11]], %[[VAL_2]] : f64
// CHECK:             graphblas.yield transform_out %[[VAL_13]] : f64
// CHECK:           }
// CHECK:           return %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }
func @fuse_adjacent(%A: tensor<?x?xf64, #CSR64>, %B: tensor<?x?xf64, #CSC64>, %thunk: f64) -> tensor<?x?xf64, #CSR64> {
    %C = graphblas.matrix_multiply %A, %B { semiring = "plus_plus" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>) to tensor<?x?xf64, #CSR64> 
    %apply_result = graphblas.matrix_apply %C, %thunk { apply_operator = "min" } : (tensor<?x?xf64, #CSR64>, f64) to tensor<?x?xf64, #CSR64>
    return %apply_result : tensor<?x?xf64, #CSR64>
}

// CHECK-LABEL:   func @fuse_adjacent_with_mask(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                  %[[VAL_2:.*]]: f64) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_3:.*]] = constant 0.000000e+00 : f64
// CHECK:           %[[VAL_4:.*]] = constant 1.000000e+00 : f64
// CHECK:           %[[VAL_5:.*]] = graphblas.matrix_multiply_generic %[[VAL_0]], %[[VAL_1]], %[[VAL_0]] : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) to tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:             graphblas.yield add_identity %[[VAL_3]] : f64
// CHECK:           },  {
// CHECK:           ^bb0(%[[VAL_6:.*]]: f64, %[[VAL_7:.*]]: f64):
// CHECK:             %[[VAL_8:.*]] = addf %[[VAL_6]], %[[VAL_7]] : f64
// CHECK:             graphblas.yield add %[[VAL_8]] : f64
// CHECK:           },  {
// CHECK:           ^bb0(%[[VAL_9:.*]]: f64, %[[VAL_10:.*]]: f64):
// CHECK:             graphblas.yield mult %[[VAL_4]] : f64
// CHECK:           },  {
// CHECK:           ^bb0(%[[VAL_11:.*]]: f64):
// CHECK:             %[[VAL_12:.*]] = cmpf olt, %[[VAL_11]], %[[VAL_2]] : f64
// CHECK:             %[[VAL_13:.*]] = select %[[VAL_12]], %[[VAL_11]], %[[VAL_2]] : f64
// CHECK:             graphblas.yield transform_out %[[VAL_13]] : f64
// CHECK:           }
// CHECK:           return %[[VAL_5]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }
func @fuse_adjacent_with_mask(%A: tensor<?x?xf64, #CSR64>, %B: tensor<?x?xf64, #CSC64>, %thunk: f64) -> tensor<?x?xf64, #CSR64> {
    %C = graphblas.matrix_multiply %A, %B, %A { semiring = "plus_pair" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>, tensor<?x?xf64, #CSR64>) to tensor<?x?xf64, #CSR64> 
    %apply_result = graphblas.matrix_apply %C, %thunk { apply_operator = "min" } : (tensor<?x?xf64, #CSR64>, f64) to tensor<?x?xf64, #CSR64>
    return %apply_result : tensor<?x?xf64, #CSR64>
}


// CHECK-LABEL:   func @nofuse_multi_use(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                           %[[VAL_1:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                           %[[VAL_2:.*]]: f64) -> (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) {
// CHECK:           %[[VAL_3:.*]] = constant 0.000000e+00 : f64
// CHECK:           %[[VAL_4:.*]] = graphblas.matrix_multiply_generic %[[VAL_0]], %[[VAL_1]] : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) to tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:             graphblas.yield add_identity %[[VAL_3]] : f64
// CHECK:           },  {
// CHECK:           ^bb0(%[[VAL_5:.*]]: f64, %[[VAL_6:.*]]: f64):
// CHECK:             %[[VAL_7:.*]] = addf %[[VAL_5]], %[[VAL_6]] : f64
// CHECK:             graphblas.yield add %[[VAL_7]] : f64
// CHECK:           },  {
// CHECK:           ^bb0(%[[VAL_8:.*]]: f64, %[[VAL_9:.*]]: f64):
// CHECK:             %[[VAL_10:.*]] = addf %[[VAL_8]], %[[VAL_9]] : f64
// CHECK:             graphblas.yield mult %[[VAL_10]] : f64
// CHECK:           }
// CHECK:           %[[VAL_11:.*]] = graphblas.matrix_apply_generic %[[VAL_12:.*]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:           ^bb0(%[[VAL_13:.*]]: f64):
// CHECK:             %[[VAL_14:.*]] = cmpf olt, %[[VAL_13]], %[[VAL_2]] : f64
// CHECK:             %[[VAL_15:.*]] = select %[[VAL_14]], %[[VAL_13]], %[[VAL_2]] : f64
// CHECK:             graphblas.yield transform_out %[[VAL_15]] : f64
// CHECK:           }
// CHECK:           return %[[VAL_11]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }
func @nofuse_multi_use(%A: tensor<?x?xf64, #CSR64>, %B: tensor<?x?xf64, #CSC64>, %thunk: f64) -> (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64>) {
    %C = graphblas.matrix_multiply %A, %B { semiring = "plus_plus" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>) to tensor<?x?xf64, #CSR64> 
    %apply_result = graphblas.matrix_apply %C, %thunk { apply_operator = "min" } : (tensor<?x?xf64, #CSR64>, f64) to tensor<?x?xf64, #CSR64>
    return %apply_result, %C : tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64>
}
