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
// CHECK:           %[[VAL_2:.*]] = graphblas.matrix_multiply_reduce_to_scalar %[[VAL_0]], %[[VAL_1]] {aggregator = "sum", semiring = "plus_plus"} : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) to f64
// CHECK:           return %[[VAL_2]] : f64
// CHECK:         }
func @fuse_adjacent(%A: tensor<?x?xf64, #CSR64>, %B: tensor<?x?xf64, #CSC64>) -> f64 {
    %C = graphblas.matrix_multiply %A, %B { semiring = "plus_plus" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>) to tensor<?x?xf64, #CSR64> 
    %reduce_result = graphblas.matrix_reduce_to_scalar %C { aggregator = "sum" } : tensor<?x?xf64, #CSR64> to f64
    return %reduce_result : f64
}

// CHECK-LABEL:   func @fuse_adjacent_with_mask(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> f64 {
// CHECK:           %[[VAL_2:.*]] = graphblas.matrix_multiply_reduce_to_scalar %[[VAL_0]], %[[VAL_1]], %[[VAL_0]] {aggregator = "sum", semiring = "plus_pair"} : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) to f64
// CHECK:           return %[[VAL_2]] : f64
// CHECK:         }
func @fuse_adjacent_with_mask(%A: tensor<?x?xf64, #CSR64>, %B: tensor<?x?xf64, #CSC64>) -> f64 {
    %C = graphblas.matrix_multiply %A, %B, %A { semiring = "plus_pair" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>, tensor<?x?xf64, #CSR64>) to tensor<?x?xf64, #CSR64> 
    %reduce_result = graphblas.matrix_reduce_to_scalar %C { aggregator = "sum" } : tensor<?x?xf64, #CSR64> to f64
    return %reduce_result : f64
}


// CHECK-LABEL:   func @nofuse_multi_use(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                           %[[VAL_1:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> (f64, tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) {
// CHECK:           %[[VAL_2:.*]] = graphblas.matrix_multiply %[[VAL_0]], %[[VAL_1]] {semiring = "plus_plus"} : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) to tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_3:.*]] = graphblas.matrix_reduce_to_scalar %[[VAL_2]] {aggregator = "sum"} : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to f64
// CHECK:           return %[[VAL_3]], %[[VAL_2]] : f64, tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }
func @nofuse_multi_use(%A: tensor<?x?xf64, #CSR64>, %B: tensor<?x?xf64, #CSC64>) -> (f64, tensor<?x?xf64, #CSR64>) {
    %C = graphblas.matrix_multiply %A, %B { semiring = "plus_plus" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>) to tensor<?x?xf64, #CSR64> 
    %reduce_result = graphblas.matrix_reduce_to_scalar %C { aggregator = "sum" } : tensor<?x?xf64, #CSR64> to f64
    return %reduce_result, %C : f64, tensor<?x?xf64, #CSR64>
}
