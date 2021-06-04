// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

// CHECK-LABEL:   func @apply_min(
// CHECK-SAME:                    %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                    %[[VAL_1:.*]]: f64) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_2:.*]] = constant 0 : index
// CHECK:           %[[VAL_3:.*]] = constant 1 : index
// CHECK:           %[[VAL_4:.*]] = call @dup_tensor(%[[VAL_0]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_5:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_6:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_8:.*]] = memref.dim %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_9:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_8]]] : memref<?xi64>
// CHECK:           %[[VAL_10:.*]] = index_cast %[[VAL_9]] : i64 to index
// CHECK:           scf.parallel (%[[VAL_11:.*]]) = (%[[VAL_2]]) to (%[[VAL_10]]) step (%[[VAL_3]]) {
// CHECK:             %[[VAL_12:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_11]]] : memref<?xf64>
// CHECK:             %[[VAL_13:.*]] = cmpf olt, %[[VAL_12]], %[[VAL_1]] : f64
// CHECK:             %[[VAL_14:.*]] = select %[[VAL_13]], %[[VAL_12]], %[[VAL_1]] : f64
// CHECK:             memref.store %[[VAL_14]], %[[VAL_7]]{{\[}}%[[VAL_11]]] : memref<?xf64>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           return %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

func @apply_min(%sparse_tensor: tensor<?x?xf64, #CSR64>, %thunk: f64) -> tensor<?x?xf64, #CSR64> {
    %answer = graphblas.matrix_apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<?x?xf64, #CSR64>, f64) to tensor<?x?xf64, #CSR64>
    return %answer : tensor<?x?xf64, #CSR64>
}
