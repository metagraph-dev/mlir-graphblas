// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

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

// CHECK-LABEL:   func private @cast_csx_to_csr(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         func private @dup_matrix(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         func private @cast_csr_to_csx(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         func private @dup_vector(tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>

// CHECK-LABEL:   func @apply_min_matrix(
// CHECK-SAME:                                   %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: f64) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_2:.*]] = constant 1 : index
// CHECK:           %[[VAL_3:.*]] = constant 0 : index
// CHECK:           %[[VAL_4:.*]] = call @cast_csr_to_csx(%[[VAL_0]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_5:.*]] = call @dup_matrix(%[[VAL_4]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_6:.*]] = call @cast_csx_to_csr(%[[VAL_5]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_8:.*]] = sparse_tensor.values %[[VAL_6]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_9:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_10:.*]] = tensor.dim %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_11:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_10]]] : memref<?xi64>
// CHECK:           %[[VAL_12:.*]] = index_cast %[[VAL_11]] : i64 to index
// CHECK:           scf.parallel (%[[VAL_13:.*]]) = (%[[VAL_3]]) to (%[[VAL_12]]) step (%[[VAL_2]]) {
// CHECK:             %[[VAL_14:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_13]]] : memref<?xf64>
// CHECK:             %[[VAL_15:.*]] = cmpf olt, %[[VAL_14]], %[[VAL_1]] : f64
// CHECK:             %[[VAL_16:.*]] = select %[[VAL_15]], %[[VAL_14]], %[[VAL_1]] : f64
// CHECK:             memref.store %[[VAL_16]], %[[VAL_8]]{{\[}}%[[VAL_13]]] : memref<?xf64>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           return %[[VAL_6]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

func @apply_min_matrix(%sparse_tensor: tensor<?x?xf64, #CSR64>, %thunk: f64) -> tensor<?x?xf64, #CSR64> {
    %answer = graphblas.apply_generic %sparse_tensor : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSR64> {
      ^bb0(%val: f64):
        %pick = cmpf olt, %val, %thunk : f64
        %result = select %pick, %val, %thunk : f64
        graphblas.yield transform_out %result : f64
    }
    return %answer : tensor<?x?xf64, #CSR64>
}

// CHECK-LABEL:   func @apply_min_vector(
// CHECK-SAME:                                   %[[VAL_0:.*]]: tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: f64) -> tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_2:.*]] = constant 0 : index
// CHECK:           %[[VAL_3:.*]] = constant 1 : index
// CHECK:           %[[VAL_4:.*]] = call @dup_vector(%[[VAL_0]]) : (tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_5:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_6:.*]] = sparse_tensor.values %[[VAL_4]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_7:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_2]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_8:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_3]]] : memref<?xi64>
// CHECK:           %[[VAL_9:.*]] = index_cast %[[VAL_8]] : i64 to index
// CHECK:           scf.parallel (%[[VAL_10:.*]]) = (%[[VAL_2]]) to (%[[VAL_9]]) step (%[[VAL_3]]) {
// CHECK:             %[[VAL_11:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_10]]] : memref<?xf64>
// CHECK:             %[[VAL_12:.*]] = cmpf olt, %[[VAL_11]], %[[VAL_1]] : f64
// CHECK:             %[[VAL_13:.*]] = select %[[VAL_12]], %[[VAL_11]], %[[VAL_1]] : f64
// CHECK:             memref.store %[[VAL_13]], %[[VAL_6]]{{\[}}%[[VAL_10]]] : memref<?xf64>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           return %[[VAL_4]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

func @apply_min_vector(%sparse_tensor: tensor<7xf64, #SparseVec64>, %thunk: f64) -> tensor<7xf64, #SparseVec64> {
    %answer = graphblas.apply_generic %sparse_tensor : tensor<7xf64, #SparseVec64> to tensor<7xf64, #SparseVec64> {
      ^bb0(%val: f64):
        %pick = cmpf olt, %val, %thunk : f64
        %result = select %pick, %val, %thunk : f64
        graphblas.yield transform_out %result : f64
    }
    return %answer : tensor<7xf64, #SparseVec64>
}
