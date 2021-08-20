// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {

// CHECK-LABEL:   func @vector_argminmax_min(
// CHECK-SAME:                               %[[VAL_0:.*]]: tensor<3xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> index {
// CHECK:           %[[VAL_1:.*]] = constant 0 : index
// CHECK:           %[[VAL_2:.*]] = constant 1 : index
// CHECK:           %[[VAL_3:.*]] = constant 3 : index
// CHECK:           %[[VAL_4:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<3xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_5:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_1]]] : memref<?xi64>
// CHECK:           %[[VAL_6:.*]]:2 = scf.for %[[VAL_7:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_2]] iter_args(%[[VAL_8:.*]] = %[[VAL_5]], %[[VAL_9:.*]] = %[[VAL_1]]) -> (i64, index) {
// CHECK:             %[[VAL_10:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_7]]] : memref<?xi64>
// CHECK:             %[[VAL_11:.*]] = cmpi slt, %[[VAL_10]], %[[VAL_8]] : i64
// CHECK:             %[[VAL_12:.*]]:2 = scf.if %[[VAL_11]] -> (i64, index) {
// CHECK:               scf.yield %[[VAL_10]], %[[VAL_7]] : i64, index
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_8]], %[[VAL_9]] : i64, index
// CHECK:             }
// CHECK:             scf.yield %[[VAL_13:.*]]#0, %[[VAL_13]]#1 : i64, index
// CHECK:           }
// CHECK:           %[[VAL_14:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_1]] : tensor<3xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_15:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_16:.*]]#1] : memref<?xi64>
// CHECK:           %[[VAL_17:.*]] = index_cast %[[VAL_15]] : i64 to index
// CHECK:           return %[[VAL_17]] : index
// CHECK:         }

   func @vector_argminmax_min(%argA: tensor<3xi64, #SparseVec64>) -> index {
       %answer = graphblas.vector_argminmax %argA { minmax = "min" } : tensor<3xi64, #SparseVec64>
       return %answer : index
   }

// CHECK-LABEL:   func @vector_argminmax_max(
// CHECK-SAME:                               %[[VAL_0:.*]]: tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> index {
// CHECK:           %[[VAL_1:.*]] = constant 0 : index
// CHECK:           %[[VAL_2:.*]] = constant 1 : index
// CHECK:           %[[VAL_3:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_1]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_4:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_2]]] : memref<?xi64>
// CHECK:           %[[VAL_5:.*]] = index_cast %[[VAL_4]] : i64 to index
// CHECK:           %[[VAL_6:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_7:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_1]]] : memref<?xi64>
// CHECK:           %[[VAL_8:.*]]:2 = scf.for %[[VAL_9:.*]] = %[[VAL_2]] to %[[VAL_5]] step %[[VAL_2]] iter_args(%[[VAL_10:.*]] = %[[VAL_7]], %[[VAL_11:.*]] = %[[VAL_1]]) -> (i64, index) {
// CHECK:             %[[VAL_12:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:             %[[VAL_13:.*]] = cmpi sgt, %[[VAL_12]], %[[VAL_10]] : i64
// CHECK:             %[[VAL_14:.*]]:2 = scf.if %[[VAL_13]] -> (i64, index) {
// CHECK:               scf.yield %[[VAL_12]], %[[VAL_9]] : i64, index
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_10]], %[[VAL_11]] : i64, index
// CHECK:             }
// CHECK:             scf.yield %[[VAL_15:.*]]#0, %[[VAL_15]]#1 : i64, index
// CHECK:           }
// CHECK:           %[[VAL_16:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_1]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_17:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_18:.*]]#1] : memref<?xi64>
// CHECK:           %[[VAL_19:.*]] = index_cast %[[VAL_17]] : i64 to index
// CHECK:           return %[[VAL_19]] : index
// CHECK:         }

   func @vector_argminmax_max(%argA: tensor<?xi64, #SparseVec64>) -> index {
       %answer = graphblas.vector_argminmax %argA { minmax = "max" } : tensor<?xi64, #SparseVec64>
       return %answer : index
   }

}
