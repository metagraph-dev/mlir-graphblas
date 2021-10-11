// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

// CHECK-LABEL:   func @vector_equal(
// CHECK-SAME:                       %[[VAL_0:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                       %[[VAL_1:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> i1 {
// CHECK:           %[[VAL_2:.*]] = constant 0 : index
// CHECK:           %[[VAL_3:.*]] = constant 1 : index
// CHECK:           %[[VAL_4:.*]] = constant false
// CHECK:           %[[VAL_5:.*]] = constant 1 : i32
// CHECK:           %[[VAL_6:.*]] = tensor.dim %[[VAL_0]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_7:.*]] = tensor.dim %[[VAL_1]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_8:.*]] = cmpi eq, %[[VAL_6]], %[[VAL_7]] : index
// CHECK:           %[[VAL_9:.*]] = scf.if %[[VAL_8]] -> (i1) {
// CHECK:             %[[VAL_10:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_11:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_3]]] : memref<?xi64>
// CHECK:             %[[VAL_12:.*]] = index_cast %[[VAL_11]] : i64 to index
// CHECK:             %[[VAL_13:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_14:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_3]]] : memref<?xi64>
// CHECK:             %[[VAL_15:.*]] = index_cast %[[VAL_14]] : i64 to index
// CHECK:             %[[VAL_16:.*]] = cmpi eq, %[[VAL_12]], %[[VAL_15]] : index
// CHECK:             %[[VAL_17:.*]] = scf.if %[[VAL_16]] -> (i1) {
// CHECK:               %[[VAL_18:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:               %[[VAL_19:.*]] = sparse_tensor.indices %[[VAL_1]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:               %[[VAL_20:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:               %[[VAL_21:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:               %[[VAL_22:.*]] = scf.parallel (%[[VAL_23:.*]]) = (%[[VAL_2]]) to (%[[VAL_12]]) step (%[[VAL_3]]) init (%[[VAL_5]]) -> i32 {
// CHECK:                 %[[VAL_24:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_23]]] : memref<?xi64>
// CHECK:                 %[[VAL_25:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_23]]] : memref<?xi64>
// CHECK:                 %[[VAL_26:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_23]]] : memref<?xf64>
// CHECK:                 %[[VAL_27:.*]] = memref.load %[[VAL_21]]{{\[}}%[[VAL_23]]] : memref<?xf64>
// CHECK:                 %[[VAL_28:.*]] = cmpi eq, %[[VAL_24]], %[[VAL_25]] : i64
// CHECK:                 %[[VAL_29:.*]] = cmpf oeq, %[[VAL_26]], %[[VAL_27]] : f64
// CHECK:                 %[[VAL_30:.*]] = and %[[VAL_28]], %[[VAL_29]] : i1
// CHECK:                 %[[VAL_31:.*]] = sexti %[[VAL_30]] : i1 to i32
// CHECK:                 scf.reduce(%[[VAL_31]])  : i32 {
// CHECK:                 ^bb0(%[[VAL_32:.*]]: i32, %[[VAL_33:.*]]: i32):
// CHECK:                   %[[VAL_34:.*]] = and %[[VAL_32]], %[[VAL_33]] : i32
// CHECK:                   scf.reduce.return %[[VAL_34]] : i32
// CHECK:                 }
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               %[[VAL_35:.*]] = trunci %[[VAL_36:.*]] : i32 to i1
// CHECK:               scf.yield %[[VAL_35]] : i1
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_4]] : i1
// CHECK:             }
// CHECK:             scf.yield %[[VAL_37:.*]] : i1
// CHECK:           } else {
// CHECK:             scf.yield %[[VAL_4]] : i1
// CHECK:           }
// CHECK:           return %[[VAL_9]] : i1
// CHECK:         }

func @vector_equal(%a: tensor<?xf64, #CV64>, %b: tensor<?xf64, #CV64>) -> i1 {
    %answer = graphblas.equal %a, %b : tensor<?xf64, #CV64>, tensor<?xf64, #CV64>
    return %answer : i1
}

// TODO: Check all type combinations
