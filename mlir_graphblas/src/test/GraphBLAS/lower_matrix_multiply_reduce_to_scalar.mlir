// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

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

// CHECK-LABEL:   func @matrix_multiply_plus_times_sum(
// CHECK-SAME:        %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:        %[[VAL_1:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK-SAME:    ) -> f64 {
// CHECK:           %[[VAL_2:.*]] = constant 0 : index
// CHECK:           %[[VAL_3:.*]] = constant 1 : index
// CHECK:           %[[VAL_4:.*]] = constant 0.000000e+00 : f64
// CHECK:           %[[VAL_5:.*]] = constant true
// CHECK:           %[[VAL_6:.*]] = constant false
// CHECK:           %[[VAL_7:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_8:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_9:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_10:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_11:.*]] = sparse_tensor.indices %[[VAL_1]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_12:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_13:.*]] = memref.dim %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_14:.*]] = memref.dim %[[VAL_1]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_15:.*]] = memref.dim %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_16:.*]] = scf.parallel (%[[VAL_17:.*]]) = (%[[VAL_2]]) to (%[[VAL_13]]) step (%[[VAL_3]]) init (%[[VAL_4]]) -> f64 {
// CHECK:             %[[VAL_18:.*]] = addi %[[VAL_17]], %[[VAL_3]] : index
// CHECK:             %[[VAL_19:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_17]]] : memref<?xi64>
// CHECK:             %[[VAL_20:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_18]]] : memref<?xi64>
// CHECK:             %[[VAL_21:.*]] = cmpi eq, %[[VAL_19]], %[[VAL_20]] : i64
// CHECK:             %[[VAL_22:.*]] = scf.if %[[VAL_21]] -> (f64) {
// CHECK:               scf.yield %[[VAL_4]] : f64
// CHECK:             } else {
// CHECK:               %[[VAL_23:.*]] = index_cast %[[VAL_19]] : i64 to index
// CHECK:               %[[VAL_24:.*]] = index_cast %[[VAL_20]] : i64 to index
// CHECK:               %[[VAL_25:.*]] = memref.alloc(%[[VAL_15]]) : memref<?xf64>
// CHECK:               %[[VAL_26:.*]] = memref.alloc(%[[VAL_15]]) : memref<?xi1>
// CHECK:               linalg.fill(%[[VAL_26]], %[[VAL_6]]) : memref<?xi1>, i1
// CHECK:               scf.parallel (%[[VAL_27:.*]]) = (%[[VAL_23]]) to (%[[VAL_24]]) step (%[[VAL_3]]) {
// CHECK:                 %[[VAL_28:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_27]]] : memref<?xi64>
// CHECK:                 %[[VAL_29:.*]] = index_cast %[[VAL_28]] : i64 to index
// CHECK:                 memref.store %[[VAL_5]], %[[VAL_26]]{{\[}}%[[VAL_29]]] : memref<?xi1>
// CHECK:                 %[[VAL_30:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_27]]] : memref<?xf64>
// CHECK:                 memref.store %[[VAL_30]], %[[VAL_25]]{{\[}}%[[VAL_29]]] : memref<?xf64>
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               %[[VAL_31:.*]] = scf.parallel (%[[VAL_32:.*]]) = (%[[VAL_2]]) to (%[[VAL_14]]) step (%[[VAL_3]]) init (%[[VAL_4]]) -> f64 {
// CHECK:                 %[[VAL_33:.*]] = addi %[[VAL_32]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_34:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_32]]] : memref<?xi64>
// CHECK:                 %[[VAL_35:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_33]]] : memref<?xi64>
// CHECK:                 %[[VAL_36:.*]] = index_cast %[[VAL_34]] : i64 to index
// CHECK:                 %[[VAL_37:.*]] = index_cast %[[VAL_35]] : i64 to index
// CHECK:                 %[[VAL_38:.*]] = scf.for %[[VAL_39:.*]] = %[[VAL_36]] to %[[VAL_37]] step %[[VAL_3]] iter_args(%[[VAL_40:.*]] = %[[VAL_4]]) -> (f64) {
// CHECK:                   %[[VAL_41:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_39]]] : memref<?xi64>
// CHECK:                   %[[VAL_42:.*]] = index_cast %[[VAL_41]] : i64 to index
// CHECK:                   %[[VAL_43:.*]] = memref.load %[[VAL_26]]{{\[}}%[[VAL_42]]] : memref<?xi1>
// CHECK:                   %[[VAL_44:.*]] = scf.if %[[VAL_43]] -> (f64) {
// CHECK:                     %[[VAL_45:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_42]]] : memref<?xf64>
// CHECK:                     %[[VAL_46:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_39]]] : memref<?xf64>
// CHECK:                     %[[VAL_47:.*]] = mulf %[[VAL_45]], %[[VAL_46]] : f64
// CHECK:                     %[[VAL_48:.*]] = addf %[[VAL_40]], %[[VAL_47]] : f64
// CHECK:                     scf.yield %[[VAL_48]] : f64
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_40]] : f64
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_49:.*]] : f64
// CHECK:                 }
// CHECK:                 scf.reduce(%[[VAL_50:.*]])  : f64 {
// CHECK:                 ^bb0(%[[VAL_51:.*]]: f64, %[[VAL_52:.*]]: f64):
// CHECK:                   %[[VAL_53:.*]] = addf %[[VAL_51]], %[[VAL_52]] : f64
// CHECK:                   scf.reduce.return %[[VAL_53]] : f64
// CHECK:                 }
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               scf.yield %[[VAL_54:.*]] : f64
// CHECK:             }
// CHECK:             scf.reduce(%[[VAL_55:.*]])  : f64 {
// CHECK:             ^bb0(%[[VAL_56:.*]]: f64, %[[VAL_57:.*]]: f64):
// CHECK:               %[[VAL_58:.*]] = addf %[[VAL_56]], %[[VAL_57]] : f64
// CHECK:               scf.reduce.return %[[VAL_58]] : f64
// CHECK:             }
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           return %[[VAL_59:.*]] : f64
// CHECK:         }


func @matrix_multiply_plus_times_sum(%a: tensor<?x?xf64, #CSR64>, %b: tensor<?x?xf64, #CSC64>) -> f64 {
    %answer = graphblas.matrix_multiply_reduce_to_scalar %a, %b { semiring = "plus_times", aggregator = "sum" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>) to f64
    return %answer : f64
}

// TODO: Check all type combinations
