// RUN: graphblas-opt %s | graphblas-opt --graphblas-optimize | FileCheck %s

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
// CHECK-SAME:                        %[[VAL_1:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> f64 {
// CHECK:           %[[VAL_2:.*]] = constant 0.000000e+00 : f64
// CHECK:           %[[VAL_3:.*]] = graphblas.matrix_multiply_reduce_to_scalar_generic %[[VAL_0]], %[[VAL_1]] {mask_complement = false} : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) to f64  {
// CHECK:             graphblas.yield add_identity %[[VAL_2]] : f64
// CHECK:           },  {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f64, %[[VAL_5:.*]]: f64):
// CHECK:             %[[VAL_6:.*]] = addf %[[VAL_4]], %[[VAL_5]] : f64
// CHECK:             graphblas.yield add %[[VAL_6]] : f64
// CHECK:           },  {
// CHECK:           ^bb0(%[[VAL_7:.*]]: f64, %[[VAL_8:.*]]: f64):
// CHECK:             %[[VAL_9:.*]] = mulf %[[VAL_7]], %[[VAL_8]] : f64
// CHECK:             graphblas.yield mult %[[VAL_9]] : f64
// CHECK:           },  {
// CHECK:             graphblas.yield agg_identity %[[VAL_2]] : f64
// CHECK:           },  {
// CHECK:           ^bb0(%[[VAL_10:.*]]: f64, %[[VAL_11:.*]]: f64):
// CHECK:             %[[VAL_12:.*]] = addf %[[VAL_10]], %[[VAL_11]] : f64
// CHECK:             graphblas.yield agg %[[VAL_12]] : f64
// CHECK:           }
// CHECK:           return %[[VAL_3]] : f64
// CHECK:         }
func @fuse_adjacent(%A: tensor<?x?xf64, #CSR64>, %B: tensor<?x?xf64, #CSC64>) -> f64 {
    %cst = constant 0.000000e+00 : f64
    %C = graphblas.matrix_multiply_generic %A, %B {mask_complement = false} : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>) to tensor<?x?xf64, #CSR64> {
        ^bb0:
            %identity = constant 0.0 : f64
            graphblas.yield add_identity %identity : f64
    },{
        ^bb0(%add_a: f64, %add_b: f64):
            %add_result = std.addf %add_a, %add_b : f64
            graphblas.yield add %add_result : f64
    },{
        ^bb0(%mult_a: f64, %mult_b: f64):
            %mult_result = std.mulf %mult_a, %mult_b : f64
            graphblas.yield mult %mult_result : f64
    }
    %reduce_result = graphblas.reduce_to_scalar_generic %C : tensor<?x?xf64, #CSR64> to f64  {
      graphblas.yield agg_identity %cst : f64
    },  {
    ^bb0(%arg1: f64, %arg2: f64):
      %1 = addf %arg1, %arg2 : f64
      graphblas.yield agg %1 : f64
    }
    return %reduce_result : f64
}


// CHECK-LABEL:   func @fuse_adjacent_with_mask(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> f64 {
// CHECK:           %[[VAL_2:.*]] = constant 0.000000e+00 : f64
// CHECK:           %[[VAL_3:.*]] = graphblas.matrix_multiply_reduce_to_scalar_generic %[[VAL_0]], %[[VAL_1]], %[[VAL_0]] {mask_complement = false} : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) to f64  {
// CHECK:             graphblas.yield add_identity %[[VAL_2]] : f64
// CHECK:           },  {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f64, %[[VAL_5:.*]]: f64):
// CHECK:             %[[VAL_6:.*]] = addf %[[VAL_4]], %[[VAL_5]] : f64
// CHECK:             graphblas.yield add %[[VAL_6]] : f64
// CHECK:           },  {
// CHECK:           ^bb0(%[[VAL_7:.*]]: f64, %[[VAL_8:.*]]: f64):
// CHECK:             %[[VAL_9:.*]] = mulf %[[VAL_7]], %[[VAL_8]] : f64
// CHECK:             graphblas.yield mult %[[VAL_9]] : f64
// CHECK:           },  {
// CHECK:             graphblas.yield agg_identity %[[VAL_2]] : f64
// CHECK:           },  {
// CHECK:           ^bb0(%[[VAL_10:.*]]: f64, %[[VAL_11:.*]]: f64):
// CHECK:             %[[VAL_12:.*]] = addf %[[VAL_10]], %[[VAL_11]] : f64
// CHECK:             graphblas.yield agg %[[VAL_12]] : f64
// CHECK:           }
// CHECK:           return %[[VAL_3]] : f64
// CHECK:         }
func @fuse_adjacent_with_mask(%A: tensor<?x?xf64, #CSR64>, %B: tensor<?x?xf64, #CSC64>) -> f64 {
    %cst = constant 0.000000e+00 : f64
    %C = graphblas.matrix_multiply_generic %A, %B, %A {mask_complement = false} : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>, tensor<?x?xf64, #CSR64>) to tensor<?x?xf64, #CSR64> {
        ^bb0:
            %identity = constant 0.0 : f64
            graphblas.yield add_identity %identity : f64
    },{
        ^bb0(%add_a: f64, %add_b: f64):
            %add_result = std.addf %add_a, %add_b : f64
            graphblas.yield add %add_result : f64
    },{
        ^bb0(%mult_a: f64, %mult_b: f64):
            %mult_result = std.mulf %mult_a, %mult_b : f64
            graphblas.yield mult %mult_result : f64
    }
    %reduce_result = graphblas.reduce_to_scalar_generic %C : tensor<?x?xf64, #CSR64> to f64  {
      graphblas.yield agg_identity %cst : f64
    },  {
    ^bb0(%arg1: f64, %arg2: f64):
      %1 = addf %arg1, %arg2 : f64
      graphblas.yield agg %1 : f64
    }
    return %reduce_result : f64
}

// CHECK-LABEL:   func @nofuse_multi_use(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                           %[[VAL_1:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> (f64, tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) {
// CHECK:           %[[VAL_2:.*]] = constant 0.000000e+00 : f64
// CHECK:           %[[VAL_3:.*]] = graphblas.matrix_multiply_generic %[[VAL_0]], %[[VAL_1]], %[[VAL_0]] {mask_complement = false} : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) to tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>  {
// CHECK:             graphblas.yield add_identity %[[VAL_2]] : f64
// CHECK:           },  {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f64, %[[VAL_5:.*]]: f64):
// CHECK:             %[[VAL_6:.*]] = addf %[[VAL_4]], %[[VAL_5]] : f64
// CHECK:             graphblas.yield add %[[VAL_6]] : f64
// CHECK:           },  {
// CHECK:           ^bb0(%[[VAL_7:.*]]: f64, %[[VAL_8:.*]]: f64):
// CHECK:             %[[VAL_9:.*]] = mulf %[[VAL_7]], %[[VAL_8]] : f64
// CHECK:             graphblas.yield mult %[[VAL_9]] : f64
// CHECK:           }
// CHECK:           %[[VAL_10:.*]] = graphblas.reduce_to_scalar_generic %[[VAL_11:.*]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to f64  {
// CHECK:             graphblas.yield agg_identity %[[VAL_2]] : f64
// CHECK:           },  {
// CHECK:           ^bb0(%[[VAL_12:.*]]: f64, %[[VAL_13:.*]]: f64):
// CHECK:             %[[VAL_14:.*]] = addf %[[VAL_12]], %[[VAL_13]] : f64
// CHECK:             graphblas.yield agg %[[VAL_14]] : f64
// CHECK:           }
// CHECK:           return %[[VAL_10]], %[[VAL_3]] : f64, tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }
func @nofuse_multi_use(%A: tensor<?x?xf64, #CSR64>, %B: tensor<?x?xf64, #CSC64>) -> (f64, tensor<?x?xf64, #CSR64>) {
    %cst = constant 0.000000e+00 : f64
    %C = graphblas.matrix_multiply_generic %A, %B, %A {mask_complement = false} : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>, tensor<?x?xf64, #CSR64>) to tensor<?x?xf64, #CSR64> {
        ^bb0:
            %identity = constant 0.0 : f64
            graphblas.yield add_identity %identity : f64
    },{
        ^bb0(%add_a: f64, %add_b: f64):
            %add_result = std.addf %add_a, %add_b : f64
            graphblas.yield add %add_result : f64
    },{
        ^bb0(%mult_a: f64, %mult_b: f64):
            %mult_result = std.mulf %mult_a, %mult_b : f64
            graphblas.yield mult %mult_result : f64
    }
    %reduce_result = graphblas.reduce_to_scalar_generic %C : tensor<?x?xf64, #CSR64> to f64  {
      graphblas.yield agg_identity %cst : f64
    },  {
    ^bb0(%arg1: f64, %arg2: f64):
      %1 = addf %arg1, %arg2 : f64
      graphblas.yield agg %1 : f64
    }
    return %reduce_result, %C : f64, tensor<?x?xf64, #CSR64>
}
