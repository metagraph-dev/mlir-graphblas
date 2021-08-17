// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

// CHECK-LABEL:   func @vector_equal(
// CHECK-SAME:                       %[[VAL_0:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                       %[[VAL_1:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> i1 {
// CHECK-DAG:       %[[VAL_2:.*]] = constant 1 : index
// CHECK-DAG:       %[[VAL_3:.*]] = constant false
// CHECK-DAG:       %[[VAL_4:.*]] = constant true
// CHECK-DAG:       %[[VAL_5:.*]] = constant 0 : index
// CHECK:           %[[VAL_6:.*]] = memref.dim %[[VAL_0]], %[[VAL_5]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_7:.*]] = memref.dim %[[VAL_1]], %[[VAL_5]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_8:.*]] = cmpi eq, %[[VAL_6]], %[[VAL_7]] : index
// CHECK:           %[[VAL_9:.*]] = scf.if %[[VAL_8]] -> (i1) {
// CHECK:             %[[VAL_10:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_5]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_12:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_2]]] : memref<?xi64>
// CHECK:             %[[VAL_16:.*]] = index_cast %[[VAL_12]] : i64 to index
// CHECK:             %[[VAL_11:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_5]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_13:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_2]]] : memref<?xi64>
// CHECK:             %[[VAL_116:.*]] = index_cast %[[VAL_13]] : i64 to index
// CHECK:             %[[VAL_14:.*]] = cmpi eq, %[[VAL_16]], %[[VAL_116]] : index
// CHECK:             %[[VAL_15:.*]] = scf.if %[[VAL_14]] -> (i1) {
// CHECK:               %[[VAL_17:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_5]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:               %[[VAL_18:.*]] = sparse_tensor.indices %[[VAL_1]], %[[VAL_5]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:               %[[VAL_19:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:               %[[VAL_20:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:               %[[VAL_21:.*]] = scf.parallel (%[[VAL_22:.*]]) = (%[[VAL_5]]) to (%[[VAL_16]]) step (%[[VAL_2]]) init (%[[VAL_4]]) -> i1 {
// CHECK:                 %[[VAL_23:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_22]]] : memref<?xi64>
// CHECK:                 %[[VAL_24:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_22]]] : memref<?xi64>
// CHECK:                 %[[VAL_25:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_22]]] : memref<?xf64>
// CHECK:                 %[[VAL_26:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_22]]] : memref<?xf64>
// CHECK:                 %[[VAL_27:.*]] = cmpi eq, %[[VAL_23]], %[[VAL_24]] : i64
// CHECK:                 %[[VAL_28:.*]] = cmpf oeq, %[[VAL_25]], %[[VAL_26]] : f64
// CHECK:                 %[[VAL_29:.*]] = and %[[VAL_27]], %[[VAL_28]] : i1
// CHECK:                 scf.reduce(%[[VAL_29]])  : i1 {
// CHECK:                 ^bb0(%[[VAL_30:.*]]: i1, %[[VAL_31:.*]]: i1):
// CHECK:                   %[[VAL_32:.*]] = and %[[VAL_30]], %[[VAL_31]] : i1
// CHECK:                   scf.reduce.return %[[VAL_32]] : i1
// CHECK:                 }
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               scf.yield %[[VAL_21:.*]] : i1
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_3]] : i1
// CHECK:             }
// CHECK:             scf.yield %[[VAL_15]] : i1
// CHECK:           } else {
// CHECK:             scf.yield %[[VAL_3]] : i1
// CHECK:           }
// CHECK:           return %[[VAL_9]] : i1
// CHECK:         }

func @vector_equal(%a: tensor<?xf64, #CV64>, %b: tensor<?xf64, #CV64>) -> i1 {
    %answer = graphblas.equal %a, %b : tensor<?xf64, #CV64>, tensor<?xf64, #CV64>
    return %answer : i1
}

// TODO: Check all type combinations
