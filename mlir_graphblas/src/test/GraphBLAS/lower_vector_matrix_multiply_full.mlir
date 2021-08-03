// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

#CSC64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

// CHECK-DAG:     func private @vector_resize_values(tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index)
// CHECK-DAG:     func private @vector_resize_index(tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index)
// CHECK-DAG:     func private @vector_resize_pointers(tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index)
// CHECK-DAG:     func private @vector_resize_dim(tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index)
// CHECK-DAG:     func private @vector_empty_like(tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>

// CHECK-LABEL:   func @vector_matrix_multiply_plus_times(
// CHECK-SAME:                                            %[[VAL_0:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                            %[[VAL_1:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK-DAG:       %[[VAL_2:.*]] = constant 2 : index
// CHECK-DAG:       %[[VAL_3:.*]] = constant 0 : i64
// CHECK-DAG:       %[[VAL_4:.*]] = constant 1 : i64
// CHECK-DAG:       %[[VAL_5:.*]] = constant 0 : index
// CHECK-DAG:       %[[VAL_6:.*]] = constant 1 : index
// CHECK-DAG:       %[[VAL_7:.*]] = constant true
// CHECK-DAG:       %[[VAL_8:.*]] = constant false
// CHECK-DAG:       %[[VAL_9:.*]] = constant 0.000000e+00 : f64
// CHECK:           %[[VAL_10:.*]] = memref.dim %[[VAL_1]], %[[VAL_6]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_11:.*]] = memref.dim %[[VAL_0]], %[[VAL_5]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_12:.*]] = call @vector_empty_like(%[[VAL_0]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           call @vector_resize_dim(%[[VAL_12]], %[[VAL_5]], %[[VAL_10]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @vector_resize_pointers(%[[VAL_12]], %[[VAL_5]], %[[VAL_2]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           %[[VAL_13:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_5]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_14:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_5]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_15:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_16:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_6]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_17:.*]] = sparse_tensor.indices %[[VAL_1]], %[[VAL_6]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_18:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_19:.*]] = sparse_tensor.pointers %[[VAL_12]], %[[VAL_5]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_20:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_6]]] : memref<?xi64>
// CHECK:           %[[VAL_21:.*]] = index_cast %[[VAL_20]] : i64 to index
// CHECK:           %[[VAL_22:.*]] = cmpi eq, %[[VAL_5]], %[[VAL_21]] : index
// CHECK:           %[[VAL_23:.*]] = scf.if %[[VAL_22]] -> (i64) {
// CHECK:             scf.yield %[[VAL_3]] : i64
// CHECK:           } else {
// CHECK:             %[[VAL_24:.*]] = memref.alloc(%[[VAL_11]]) : memref<?xi1>
// CHECK:             linalg.fill(%[[VAL_24]], %[[VAL_8]]) : memref<?xi1>, i1
// CHECK:             scf.parallel (%[[VAL_25:.*]]) = (%[[VAL_5]]) to (%[[VAL_21]]) step (%[[VAL_6]]) {
// CHECK:               %[[VAL_26:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_25]]] : memref<?xi64>
// CHECK:               %[[VAL_27:.*]] = index_cast %[[VAL_26]] : i64 to index
// CHECK:               memref.store %[[VAL_7]], %[[VAL_24]]{{\[}}%[[VAL_27]]] : memref<?xi1>
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             %[[VAL_28:.*]] = scf.parallel (%[[VAL_29:.*]]) = (%[[VAL_5]]) to (%[[VAL_10]]) step (%[[VAL_6]]) init (%[[VAL_3]]) -> i64 {
// CHECK:               %[[VAL_30:.*]] = addi %[[VAL_29]], %[[VAL_6]] : index
// CHECK:               %[[VAL_31:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_29]]] : memref<?xi64>
// CHECK:               %[[VAL_32:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_30]]] : memref<?xi64>
// CHECK:               %[[VAL_33:.*]] = cmpi eq, %[[VAL_31]], %[[VAL_32]] : i64
// CHECK:               %[[VAL_34:.*]] = scf.if %[[VAL_33]] -> (i64) {
// CHECK:                 scf.yield %[[VAL_3]] : i64
// CHECK:               } else {
// CHECK:                 %[[VAL_35:.*]] = scf.while (%[[VAL_36:.*]] = %[[VAL_31]]) : (i64) -> i64 {
// CHECK:                   %[[VAL_37:.*]] = cmpi uge, %[[VAL_36]], %[[VAL_32]] : i64
// CHECK:                   %[[VAL_38:.*]]:2 = scf.if %[[VAL_37]] -> (i1, i64) {
// CHECK:                     scf.yield %[[VAL_8]], %[[VAL_3]] : i1, i64
// CHECK:                   } else {
// CHECK:                     %[[VAL_39:.*]] = index_cast %[[VAL_36]] : i64 to index
// CHECK:                     %[[VAL_40:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_39]]] : memref<?xi64>
// CHECK:                     %[[VAL_41:.*]] = index_cast %[[VAL_40]] : i64 to index
// CHECK:                     %[[VAL_42:.*]] = memref.load %[[VAL_24]]{{\[}}%[[VAL_41]]] : memref<?xi1>
// CHECK:                     %[[VAL_43:.*]] = select %[[VAL_42]], %[[VAL_8]], %[[VAL_7]] : i1
// CHECK:                     %[[VAL_44:.*]] = select %[[VAL_42]], %[[VAL_4]], %[[VAL_36]] : i64
// CHECK:                     scf.yield %[[VAL_43]], %[[VAL_44]] : i1, i64
// CHECK:                   }
// CHECK:                   scf.condition(%[[VAL_45:.*]]#0) %[[VAL_45]]#1 : i64
// CHECK:                 } do {
// CHECK:                 ^bb0(%[[VAL_46:.*]]: i64):
// CHECK:                   %[[VAL_47:.*]] = addi %[[VAL_46]], %[[VAL_4]] : i64
// CHECK:                   scf.yield %[[VAL_47]] : i64
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_48:.*]] : i64
// CHECK:               }
// CHECK:               scf.reduce(%[[VAL_49:.*]])  : i64 {
// CHECK:               ^bb0(%[[VAL_50:.*]]: i64, %[[VAL_51:.*]]: i64):
// CHECK:                 %[[VAL_52:.*]] = addi %[[VAL_50]], %[[VAL_51]] : i64
// CHECK:                 scf.reduce.return %[[VAL_52]] : i64
// CHECK:               }
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_24]] : memref<?xi1>
// CHECK:             scf.yield %[[VAL_28]] : i64
// CHECK:           }
// CHECK:           %[[VAL_54:.*]] = index_cast %[[VAL_23]] : i64 to index
// CHECK:           memref.store %[[VAL_23]], %[[VAL_19]]{{\[}}%[[VAL_6]]] : memref<?xi64>
// CHECK:           call @vector_resize_index(%[[VAL_12]], %[[VAL_5]], %[[VAL_54]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @vector_resize_values(%[[VAL_12]], %[[VAL_54]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> ()
// CHECK:           %[[VAL_56:.*]] = sparse_tensor.indices %[[VAL_12]], %[[VAL_5]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_57:.*]] = sparse_tensor.values %[[VAL_12]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_58:.*]] = cmpi ne, %[[VAL_5]], %[[VAL_54]] : index
// CHECK:           scf.if %[[VAL_58]] {
// CHECK:             %[[VAL_59:.*]] = memref.alloc(%[[VAL_11]]) : memref<?xf64>
// CHECK:             %[[VAL_60:.*]] = memref.alloc(%[[VAL_11]]) : memref<?xi1>
// CHECK:             linalg.fill(%[[VAL_60]], %[[VAL_8]]) : memref<?xi1>, i1
// CHECK:             scf.parallel (%[[VAL_61:.*]]) = (%[[VAL_5]]) to (%[[VAL_21]]) step (%[[VAL_6]]) {
// CHECK:               %[[VAL_62:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_61]]] : memref<?xi64>
// CHECK:               %[[VAL_63:.*]] = index_cast %[[VAL_62]] : i64 to index
// CHECK:               memref.store %[[VAL_7]], %[[VAL_60]]{{\[}}%[[VAL_63]]] : memref<?xi1>
// CHECK:               %[[VAL_64:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_61]]] : memref<?xf64>
// CHECK:               memref.store %[[VAL_64]], %[[VAL_59]]{{\[}}%[[VAL_63]]] : memref<?xf64>
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             %[[VAL_65:.*]] = scf.for %[[VAL_66:.*]] = %[[VAL_5]] to %[[VAL_10]] step %[[VAL_6]] iter_args(%[[VAL_67:.*]] = %[[VAL_5]]) -> (index) {
// CHECK:               %[[VAL_68:.*]] = index_cast %[[VAL_66]] : index to i64
// CHECK:               %[[VAL_69:.*]] = addi %[[VAL_66]], %[[VAL_6]] : index
// CHECK:               %[[VAL_70:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_66]]] : memref<?xi64>
// CHECK:               %[[VAL_71:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_69]]] : memref<?xi64>
// CHECK:               %[[VAL_72:.*]] = index_cast %[[VAL_70]] : i64 to index
// CHECK:               %[[VAL_73:.*]] = index_cast %[[VAL_71]] : i64 to index
// CHECK:               %[[VAL_74:.*]]:2 = scf.for %[[VAL_75:.*]] = %[[VAL_72]] to %[[VAL_73]] step %[[VAL_6]] iter_args(%[[VAL_76:.*]] = %[[VAL_9]], %[[VAL_77:.*]] = %[[VAL_8]]) -> (f64, i1) {
// CHECK:                 %[[VAL_78:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_75]]] : memref<?xi64>
// CHECK:                 %[[VAL_79:.*]] = index_cast %[[VAL_78]] : i64 to index
// CHECK:                 %[[VAL_80:.*]] = memref.load %[[VAL_60]]{{\[}}%[[VAL_79]]] : memref<?xi1>
// CHECK:                 %[[VAL_81:.*]]:2 = scf.if %[[VAL_80]] -> (f64, i1) {
// CHECK:                   %[[VAL_82:.*]] = memref.load %[[VAL_59]]{{\[}}%[[VAL_79]]] : memref<?xf64>
// CHECK:                   %[[VAL_83:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_75]]] : memref<?xf64>
// CHECK:                   %[[VAL_84:.*]] = mulf %[[VAL_82]], %[[VAL_83]] : f64
// CHECK:                   %[[VAL_85:.*]] = addf %[[VAL_76]], %[[VAL_84]] : f64
// CHECK:                   scf.yield %[[VAL_85]], %[[VAL_7]] : f64, i1
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_76]], %[[VAL_77]] : f64, i1
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_86:.*]]#0, %[[VAL_86]]#1 : f64, i1
// CHECK:               }
// CHECK:               %[[VAL_87:.*]] = scf.if %[[VAL_88:.*]]#1 -> (index) {
// CHECK:                 memref.store %[[VAL_68]], %[[VAL_56]]{{\[}}%[[VAL_67]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_88]]#0, %[[VAL_57]]{{\[}}%[[VAL_67]]] : memref<?xf64>
// CHECK:                 %[[VAL_89:.*]] = addi %[[VAL_67]], %[[VAL_6]] : index
// CHECK:                 scf.yield %[[VAL_89]] : index
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_67]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_90:.*]] : index
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_59]] : memref<?xf64>
// CHECK:             memref.dealloc %[[VAL_60]] : memref<?xi1>
// CHECK:           }
// CHECK:           return %[[VAL_12]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }


func @vector_matrix_multiply_plus_times(%a: tensor<?xf64, #CV64>, %b: tensor<?x?xf64, #CSC64>) -> tensor<?xf64, #CV64> {
    %answer = graphblas.matrix_multiply %a, %b { semiring = "plus_times" } : (tensor<?xf64, #CV64>, tensor<?x?xf64, #CSC64>) to tensor<?xf64, #CV64>
    return %answer : tensor<?xf64, #CV64>
}

// TODO: Check all type combinations
