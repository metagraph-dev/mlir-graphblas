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

// CHECK-DAG:     func private @matrix_resize_values(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index)
// CHECK-DAG:     func private @matrix_resize_index(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index)
// CHECK-DAG:     func private @cast_csx_to_csr(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK-DAG:     func private @matrix_resize_pointers(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index)
// CHECK-DAG:     func private @matrix_resize_dim(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index)
// CHECK-DAG:     func private @matrix_empty_like(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK-DAG:     func private @cast_csr_to_csx(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>

// CHECK-LABEL:   func @matrix_multiply_plus_times(
// CHECK-SAME:                                     %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK-DAG:       %[[VAL_2:.*]] = constant 0 : index
// CHECK-DAG:       %[[VAL_3:.*]] = constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = constant 0 : i64
// CHECK-DAG:       %[[VAL_5:.*]] = constant 1 : i64
// CHECK-DAG:       %[[VAL_6:.*]] = constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_7:.*]] = constant true
// CHECK-DAG:       %[[VAL_8:.*]] = constant false
// CHECK:           %[[VAL_15:.*]] = tensor.dim %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_16:.*]] = tensor.dim %[[VAL_1]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_17:.*]] = tensor.dim %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_18:.*]] = addi %[[VAL_15]], %[[VAL_3]] : index
// CHECK:           %[[VAL_19:.*]] = call @cast_csr_to_csx(%[[VAL_0]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_20:.*]] = call @matrix_empty_like(%[[VAL_19]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           call @matrix_resize_dim(%[[VAL_20]], %[[VAL_2]], %[[VAL_15]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @matrix_resize_dim(%[[VAL_20]], %[[VAL_3]], %[[VAL_16]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @matrix_resize_pointers(%[[VAL_20]], %[[VAL_3]], %[[VAL_18]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           %[[VAL_21:.*]] = call @cast_csx_to_csr(%[[VAL_20]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_9:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_10:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_11:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_12:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_13:.*]] = sparse_tensor.indices %[[VAL_1]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_14:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_22:.*]] = sparse_tensor.pointers %[[VAL_21]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           scf.parallel (%[[VAL_23:.*]]) = (%[[VAL_2]]) to (%[[VAL_15]]) step (%[[VAL_3]]) {
// CHECK:             %[[VAL_24:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_23]]] : memref<?xi64>
// CHECK:             %[[VAL_25:.*]] = addi %[[VAL_23]], %[[VAL_3]] : index
// CHECK:             %[[VAL_26:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_25]]] : memref<?xi64>
// CHECK:             %[[VAL_27:.*]] = cmpi eq, %[[VAL_24]], %[[VAL_26]] : i64
// CHECK:             %[[VAL_28:.*]] = scf.if %[[VAL_27]] -> (i64) {
// CHECK:               scf.yield %[[VAL_4]] : i64
// CHECK:             } else {
// CHECK:               %[[VAL_29:.*]] = index_cast %[[VAL_24]] : i64 to index
// CHECK:               %[[VAL_30:.*]] = index_cast %[[VAL_26]] : i64 to index
// CHECK:               %[[VAL_31:.*]] = memref.alloc(%[[VAL_17]]) : memref<?xi1>
// CHECK:               linalg.fill(%[[VAL_8]], %[[VAL_31]]) : i1, memref<?xi1>
// CHECK:               scf.parallel (%[[VAL_32:.*]]) = (%[[VAL_29]]) to (%[[VAL_30]]) step (%[[VAL_3]]) {
// CHECK:                 %[[VAL_33:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_32]]] : memref<?xi64>
// CHECK:                 %[[VAL_34:.*]] = index_cast %[[VAL_33]] : i64 to index
// CHECK:                 memref.store %[[VAL_7]], %[[VAL_31]]{{\[}}%[[VAL_34]]] : memref<?xi1>
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               %[[VAL_35:.*]] = scf.parallel (%[[VAL_36:.*]]) = (%[[VAL_2]]) to (%[[VAL_16]]) step (%[[VAL_3]]) init (%[[VAL_4]]) -> i64 {
// CHECK:                 %[[VAL_37:.*]] = addi %[[VAL_36]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_38:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_36]]] : memref<?xi64>
// CHECK:                 %[[VAL_39:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_37]]] : memref<?xi64>
// CHECK:                 %[[VAL_40:.*]] = cmpi eq, %[[VAL_38]], %[[VAL_39]] : i64
// CHECK:                 %[[VAL_41:.*]] = scf.if %[[VAL_40]] -> (i64) {
// CHECK:                   scf.yield %[[VAL_4]] : i64
// CHECK:                 } else {
// CHECK:                   %[[VAL_42:.*]] = scf.while (%[[VAL_43:.*]] = %[[VAL_38]]) : (i64) -> i64 {
// CHECK:                     %[[VAL_44:.*]] = cmpi uge, %[[VAL_43]], %[[VAL_39]] : i64
// CHECK:                     %[[VAL_45:.*]]:2 = scf.if %[[VAL_44]] -> (i1, i64) {
// CHECK:                       scf.yield %[[VAL_8]], %[[VAL_4]] : i1, i64
// CHECK:                     } else {
// CHECK:                       %[[VAL_46:.*]] = index_cast %[[VAL_43]] : i64 to index
// CHECK:                       %[[VAL_47:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_46]]] : memref<?xi64>
// CHECK:                       %[[VAL_48:.*]] = index_cast %[[VAL_47]] : i64 to index
// CHECK:                       %[[VAL_49:.*]] = memref.load %[[VAL_31]]{{\[}}%[[VAL_48]]] : memref<?xi1>
// CHECK:                       %[[VAL_50:.*]] = select %[[VAL_49]], %[[VAL_8]], %[[VAL_7]] : i1
// CHECK:                       %[[VAL_51:.*]] = select %[[VAL_49]], %[[VAL_5]], %[[VAL_43]] : i64
// CHECK:                       scf.yield %[[VAL_50]], %[[VAL_51]] : i1, i64
// CHECK:                     }
// CHECK:                     scf.condition(%[[VAL_52:.*]]#0) %[[VAL_52]]#1 : i64
// CHECK:                   } do {
// CHECK:                   ^bb0(%[[VAL_53:.*]]: i64):
// CHECK:                     %[[VAL_54:.*]] = addi %[[VAL_53]], %[[VAL_5]] : i64
// CHECK:                     scf.yield %[[VAL_54]] : i64
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_55:.*]] : i64
// CHECK:                 }
// CHECK:                 scf.reduce(%[[VAL_56:.*]])  : i64 {
// CHECK:                 ^bb0(%[[VAL_57:.*]]: i64, %[[VAL_58:.*]]: i64):
// CHECK:                   %[[VAL_59:.*]] = addi %[[VAL_57]], %[[VAL_58]] : i64
// CHECK:                   scf.reduce.return %[[VAL_59]] : i64
// CHECK:                 }
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               scf.yield %[[VAL_60:.*]] : i64
// CHECK:             }
// CHECK:             memref.store %[[VAL_61:.*]], %[[VAL_22]]{{\[}}%[[VAL_23]]] : memref<?xi64>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           scf.for %[[VAL_62:.*]] = %[[VAL_2]] to %[[VAL_15]] step %[[VAL_3]] {
// CHECK:             %[[VAL_63:.*]] = memref.load %[[VAL_22]]{{\[}}%[[VAL_62]]] : memref<?xi64>
// CHECK:             %[[VAL_64:.*]] = memref.load %[[VAL_22]]{{\[}}%[[VAL_15]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_64]], %[[VAL_22]]{{\[}}%[[VAL_62]]] : memref<?xi64>
// CHECK:             %[[VAL_65:.*]] = addi %[[VAL_64]], %[[VAL_63]] : i64
// CHECK:             memref.store %[[VAL_65]], %[[VAL_22]]{{\[}}%[[VAL_15]]] : memref<?xi64>
// CHECK:           }
// CHECK:           %[[VAL_200:.*]] = sparse_tensor.pointers %[[VAL_21]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_201:.*]] = tensor.dim %[[VAL_21]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_66:.*]] = memref.load %[[VAL_200]]{{\[}}%[[VAL_201]]] : memref<?xi64>
// CHECK:           %[[VAL_67:.*]] = index_cast %[[VAL_66]] : i64 to index
// CHECK:           %[[VAL_68:.*]] = call @cast_csr_to_csx(%[[VAL_21]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           call @matrix_resize_index(%[[VAL_68]], %[[VAL_3]], %[[VAL_67]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           %[[VAL_69:.*]] = call @cast_csr_to_csx(%[[VAL_21]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           call @matrix_resize_values(%[[VAL_69]], %[[VAL_67]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> ()
// CHECK:           %[[VAL_70:.*]] = sparse_tensor.indices %[[VAL_21]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_71:.*]] = sparse_tensor.values %[[VAL_21]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           scf.parallel (%[[VAL_72:.*]]) = (%[[VAL_2]]) to (%[[VAL_15]]) step (%[[VAL_3]]) {
// CHECK:             %[[VAL_73:.*]] = addi %[[VAL_72]], %[[VAL_3]] : index
// CHECK:             %[[VAL_74:.*]] = memref.load %[[VAL_22]]{{\[}}%[[VAL_72]]] : memref<?xi64>
// CHECK:             %[[VAL_75:.*]] = memref.load %[[VAL_22]]{{\[}}%[[VAL_73]]] : memref<?xi64>
// CHECK:             %[[VAL_76:.*]] = cmpi ne, %[[VAL_74]], %[[VAL_75]] : i64
// CHECK:             scf.if %[[VAL_76]] {
// CHECK:               %[[VAL_77:.*]] = memref.load %[[VAL_22]]{{\[}}%[[VAL_72]]] : memref<?xi64>
// CHECK:               %[[VAL_78:.*]] = index_cast %[[VAL_77]] : i64 to index
// CHECK:               %[[VAL_79:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_72]]] : memref<?xi64>
// CHECK:               %[[VAL_80:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_73]]] : memref<?xi64>
// CHECK:               %[[VAL_81:.*]] = index_cast %[[VAL_79]] : i64 to index
// CHECK:               %[[VAL_82:.*]] = index_cast %[[VAL_80]] : i64 to index
// CHECK:               %[[VAL_83:.*]] = memref.alloc(%[[VAL_17]]) : memref<?xf64>
// CHECK:               %[[VAL_84:.*]] = memref.alloc(%[[VAL_17]]) : memref<?xi1>
// CHECK:               linalg.fill(%[[VAL_8]], %[[VAL_84]]) : i1, memref<?xi1>
// CHECK:               scf.parallel (%[[VAL_85:.*]]) = (%[[VAL_81]]) to (%[[VAL_82]]) step (%[[VAL_3]]) {
// CHECK:                 %[[VAL_86:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_85]]] : memref<?xi64>
// CHECK:                 %[[VAL_87:.*]] = index_cast %[[VAL_86]] : i64 to index
// CHECK:                 memref.store %[[VAL_7]], %[[VAL_84]]{{\[}}%[[VAL_87]]] : memref<?xi1>
// CHECK:                 %[[VAL_88:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_85]]] : memref<?xf64>
// CHECK:                 memref.store %[[VAL_88]], %[[VAL_83]]{{\[}}%[[VAL_87]]] : memref<?xf64>
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               %[[VAL_89:.*]] = scf.for %[[VAL_90:.*]] = %[[VAL_2]] to %[[VAL_16]] step %[[VAL_3]] iter_args(%[[VAL_91:.*]] = %[[VAL_2]]) -> (index) {
// CHECK:                 %[[VAL_92:.*]] = index_cast %[[VAL_90]] : index to i64
// CHECK:                 %[[VAL_93:.*]] = addi %[[VAL_90]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_94:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_90]]] : memref<?xi64>
// CHECK:                 %[[VAL_95:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_93]]] : memref<?xi64>
// CHECK:                 %[[VAL_96:.*]] = index_cast %[[VAL_94]] : i64 to index
// CHECK:                 %[[VAL_97:.*]] = index_cast %[[VAL_95]] : i64 to index
// CHECK:                 %[[VAL_98:.*]]:2 = scf.for %[[VAL_99:.*]] = %[[VAL_96]] to %[[VAL_97]] step %[[VAL_3]] iter_args(%[[VAL_100:.*]] = %[[VAL_6]], %[[VAL_101:.*]] = %[[VAL_8]]) -> (f64, i1) {
// CHECK:                   %[[VAL_102:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_99]]] : memref<?xi64>
// CHECK:                   %[[VAL_103:.*]] = index_cast %[[VAL_102]] : i64 to index
// CHECK:                   %[[VAL_104:.*]] = memref.load %[[VAL_84]]{{\[}}%[[VAL_103]]] : memref<?xi1>
// CHECK:                   %[[VAL_105:.*]]:2 = scf.if %[[VAL_104]] -> (f64, i1) {
// CHECK:                     %[[VAL_106:.*]] = memref.load %[[VAL_83]]{{\[}}%[[VAL_103]]] : memref<?xf64>
// CHECK:                     %[[VAL_107:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_99]]] : memref<?xf64>
// CHECK:                     %[[VAL_108:.*]] = mulf %[[VAL_106]], %[[VAL_107]] : f64
// CHECK:                     %[[VAL_109:.*]] = addf %[[VAL_100]], %[[VAL_108]] : f64
// CHECK:                     scf.yield %[[VAL_109]], %[[VAL_7]] : f64, i1
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_100]], %[[VAL_101]] : f64, i1
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_110:.*]]#0, %[[VAL_110]]#1 : f64, i1
// CHECK:                 }
// CHECK:                 %[[VAL_111:.*]] = scf.if %[[VAL_112:.*]]#1 -> (index) {
// CHECK:                   %[[VAL_113:.*]] = addi %[[VAL_78]], %[[VAL_91]] : index
// CHECK:                   memref.store %[[VAL_92]], %[[VAL_70]]{{\[}}%[[VAL_113]]] : memref<?xi64>
// CHECK:                   memref.store %[[VAL_112]]#0, %[[VAL_71]]{{\[}}%[[VAL_113]]] : memref<?xf64>
// CHECK:                   %[[VAL_114:.*]] = addi %[[VAL_91]], %[[VAL_3]] : index
// CHECK:                   scf.yield %[[VAL_114]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_91]] : index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_115:.*]] : index
// CHECK:               }
// CHECK:             }
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           return %[[VAL_21]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

func @matrix_multiply_plus_times(%a: tensor<?x?xf64, #CSR64>, %b: tensor<?x?xf64, #CSC64>) -> tensor<?x?xf64, #CSR64> {
    %answer = graphblas.matrix_multiply %a, %b { semiring = "plus_times" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>) to tensor<?x?xf64, #CSR64>
    return %answer : tensor<?x?xf64, #CSR64>
}

// TODO: Check all type combinations
