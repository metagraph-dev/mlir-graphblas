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

// CHECK-LABEL:   func private @resize_values(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index)
// CHECK:         func private @resize_index(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index)
// CHECK:         func private @cast_csx_to_csr(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         func private @resize_pointers(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index)
// CHECK:         func private @resize_dim(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index)
// CHECK:         func private @empty_like(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         func private @cast_csr_to_csx(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>

// CHECK-LABEL:   func @matrix_multiply_mask_plus_pair(
// CHECK-SAME:                                         %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                         %[[VAL_1:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                         %[[VAL_2:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_3:.*]] = constant 0 : index
// CHECK:           %[[VAL_4:.*]] = constant 1 : index
// CHECK:           %[[VAL_5:.*]] = constant 0 : i64
// CHECK:           %[[VAL_6:.*]] = constant 1 : i64
// CHECK:           %[[VAL_7:.*]] = constant 0.000000e+00 : f64
// CHECK:           %[[VAL_8:.*]] = constant 1.000000e+00 : f64
// CHECK:           %[[VAL_9:.*]] = constant true
// CHECK:           %[[VAL_10:.*]] = constant false
// CHECK:           %[[VAL_11:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_12:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_13:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_14:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_15:.*]] = sparse_tensor.indices %[[VAL_1]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_16:.*]] = memref.dim %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_17:.*]] = memref.dim %[[VAL_1]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_18:.*]] = memref.dim %[[VAL_0]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_19:.*]] = addi %[[VAL_16]], %[[VAL_4]] : index
// CHECK:           %[[VAL_20:.*]] = sparse_tensor.pointers %[[VAL_2]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_21:.*]] = sparse_tensor.indices %[[VAL_2]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_22:.*]] = call @cast_csr_to_csx(%[[VAL_0]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_23:.*]] = call @empty_like(%[[VAL_22]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           call @resize_dim(%[[VAL_23]], %[[VAL_3]], %[[VAL_16]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @resize_dim(%[[VAL_23]], %[[VAL_4]], %[[VAL_17]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @resize_pointers(%[[VAL_23]], %[[VAL_4]], %[[VAL_19]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           %[[VAL_24:.*]] = call @cast_csx_to_csr(%[[VAL_23]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_25:.*]] = sparse_tensor.pointers %[[VAL_24]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           scf.parallel (%[[VAL_26:.*]]) = (%[[VAL_3]]) to (%[[VAL_16]]) step (%[[VAL_4]]) {
// CHECK:             %[[VAL_27:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_26]]] : memref<?xi64>
// CHECK:             %[[VAL_28:.*]] = addi %[[VAL_26]], %[[VAL_4]] : index
// CHECK:             %[[VAL_29:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_28]]] : memref<?xi64>
// CHECK:             %[[VAL_30:.*]] = cmpi eq, %[[VAL_27]], %[[VAL_29]] : i64
// CHECK:             %[[VAL_31:.*]] = scf.if %[[VAL_30]] -> (i64) {
// CHECK:               scf.yield %[[VAL_5]] : i64
// CHECK:             } else {
// CHECK:               %[[VAL_32:.*]] = index_cast %[[VAL_27]] : i64 to index
// CHECK:               %[[VAL_33:.*]] = index_cast %[[VAL_29]] : i64 to index
// CHECK:               %[[VAL_34:.*]] = memref.alloc(%[[VAL_18]]) : memref<?xi1>
// CHECK:               linalg.fill(%[[VAL_34]], %[[VAL_10]]) : memref<?xi1>, i1
// CHECK:               scf.parallel (%[[VAL_35:.*]]) = (%[[VAL_32]]) to (%[[VAL_33]]) step (%[[VAL_4]]) {
// CHECK:                 %[[VAL_36:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_35]]] : memref<?xi64>
// CHECK:                 %[[VAL_37:.*]] = index_cast %[[VAL_36]] : i64 to index
// CHECK:                 memref.store %[[VAL_9]], %[[VAL_34]]{{\[}}%[[VAL_37]]] : memref<?xi1>
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               %[[VAL_38:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_26]]] : memref<?xi64>
// CHECK:               %[[VAL_39:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_28]]] : memref<?xi64>
// CHECK:               %[[VAL_40:.*]] = index_cast %[[VAL_38]] : i64 to index
// CHECK:               %[[VAL_41:.*]] = index_cast %[[VAL_39]] : i64 to index
// CHECK:               %[[VAL_42:.*]] = scf.parallel (%[[VAL_43:.*]]) = (%[[VAL_40]]) to (%[[VAL_41]]) step (%[[VAL_4]]) init (%[[VAL_5]]) -> i64 {
// CHECK:                 %[[VAL_44:.*]] = memref.load %[[VAL_21]]{{\[}}%[[VAL_43]]] : memref<?xi64>
// CHECK:                 %[[VAL_45:.*]] = index_cast %[[VAL_44]] : i64 to index
// CHECK:                 %[[VAL_46:.*]] = addi %[[VAL_45]], %[[VAL_4]] : index
// CHECK:                 %[[VAL_47:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_45]]] : memref<?xi64>
// CHECK:                 %[[VAL_48:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_46]]] : memref<?xi64>
// CHECK:                 %[[VAL_49:.*]] = cmpi eq, %[[VAL_47]], %[[VAL_48]] : i64
// CHECK:                 %[[VAL_50:.*]] = scf.if %[[VAL_49]] -> (i64) {
// CHECK:                   scf.yield %[[VAL_5]] : i64
// CHECK:                 } else {
// CHECK:                   %[[VAL_51:.*]] = scf.while (%[[VAL_52:.*]] = %[[VAL_47]]) : (i64) -> i64 {
// CHECK:                     %[[VAL_53:.*]] = cmpi uge, %[[VAL_52]], %[[VAL_48]] : i64
// CHECK:                     %[[VAL_54:.*]]:2 = scf.if %[[VAL_53]] -> (i1, i64) {
// CHECK:                       scf.yield %[[VAL_10]], %[[VAL_5]] : i1, i64
// CHECK:                     } else {
// CHECK:                       %[[VAL_55:.*]] = index_cast %[[VAL_52]] : i64 to index
// CHECK:                       %[[VAL_56:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_55]]] : memref<?xi64>
// CHECK:                       %[[VAL_57:.*]] = index_cast %[[VAL_56]] : i64 to index
// CHECK:                       %[[VAL_58:.*]] = memref.load %[[VAL_34]]{{\[}}%[[VAL_57]]] : memref<?xi1>
// CHECK:                       %[[VAL_59:.*]] = select %[[VAL_58]], %[[VAL_10]], %[[VAL_9]] : i1
// CHECK:                       %[[VAL_60:.*]] = select %[[VAL_58]], %[[VAL_6]], %[[VAL_52]] : i64
// CHECK:                       scf.yield %[[VAL_59]], %[[VAL_60]] : i1, i64
// CHECK:                     }
// CHECK:                     scf.condition(%[[VAL_61:.*]]#0) %[[VAL_61]]#1 : i64
// CHECK:                   } do {
// CHECK:                   ^bb0(%[[VAL_62:.*]]: i64):
// CHECK:                     %[[VAL_63:.*]] = addi %[[VAL_62]], %[[VAL_6]] : i64
// CHECK:                     scf.yield %[[VAL_63]] : i64
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_64:.*]] : i64
// CHECK:                 }
// CHECK:                 scf.reduce(%[[VAL_65:.*]])  : i64 {
// CHECK:                 ^bb0(%[[VAL_66:.*]]: i64, %[[VAL_67:.*]]: i64):
// CHECK:                   %[[VAL_68:.*]] = addi %[[VAL_66]], %[[VAL_67]] : i64
// CHECK:                   scf.reduce.return %[[VAL_68]] : i64
// CHECK:                 }
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               scf.yield %[[VAL_69:.*]] : i64
// CHECK:             }
// CHECK:             memref.store %[[VAL_70:.*]], %[[VAL_25]]{{\[}}%[[VAL_26]]] : memref<?xi64>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           scf.for %[[VAL_71:.*]] = %[[VAL_3]] to %[[VAL_16]] step %[[VAL_4]] {
// CHECK:             %[[VAL_72:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_71]]] : memref<?xi64>
// CHECK:             %[[VAL_73:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_16]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_73]], %[[VAL_25]]{{\[}}%[[VAL_71]]] : memref<?xi64>
// CHECK:             %[[VAL_74:.*]] = addi %[[VAL_73]], %[[VAL_72]] : i64
// CHECK:             memref.store %[[VAL_74]], %[[VAL_25]]{{\[}}%[[VAL_16]]] : memref<?xi64>
// CHECK:           }
// CHECK:           %[[VAL_75:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_16]]] : memref<?xi64>
// CHECK:           %[[VAL_76:.*]] = index_cast %[[VAL_75]] : i64 to index
// CHECK:           %[[VAL_77:.*]] = call @cast_csr_to_csx(%[[VAL_24]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           call @resize_index(%[[VAL_77]], %[[VAL_4]], %[[VAL_76]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           %[[VAL_78:.*]] = call @cast_csr_to_csx(%[[VAL_24]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           call @resize_values(%[[VAL_78]], %[[VAL_76]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> ()
// CHECK:           %[[VAL_79:.*]] = sparse_tensor.indices %[[VAL_24]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_80:.*]] = sparse_tensor.values %[[VAL_24]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           scf.parallel (%[[VAL_81:.*]]) = (%[[VAL_3]]) to (%[[VAL_16]]) step (%[[VAL_4]]) {
// CHECK:             %[[VAL_82:.*]] = addi %[[VAL_81]], %[[VAL_4]] : index
// CHECK:             %[[VAL_83:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_81]]] : memref<?xi64>
// CHECK:             %[[VAL_84:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_82]]] : memref<?xi64>
// CHECK:             %[[VAL_85:.*]] = cmpi ne, %[[VAL_83]], %[[VAL_84]] : i64
// CHECK:             scf.if %[[VAL_85]] {
// CHECK:               %[[VAL_86:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_81]]] : memref<?xi64>
// CHECK:               %[[VAL_87:.*]] = index_cast %[[VAL_86]] : i64 to index
// CHECK:               %[[VAL_88:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_81]]] : memref<?xi64>
// CHECK:               %[[VAL_89:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_82]]] : memref<?xi64>
// CHECK:               %[[VAL_90:.*]] = index_cast %[[VAL_88]] : i64 to index
// CHECK:               %[[VAL_91:.*]] = index_cast %[[VAL_89]] : i64 to index
// CHECK:               %[[VAL_92:.*]] = memref.alloc(%[[VAL_18]]) : memref<?xf64>
// CHECK:               %[[VAL_93:.*]] = memref.alloc(%[[VAL_18]]) : memref<?xi1>
// CHECK:               linalg.fill(%[[VAL_93]], %[[VAL_10]]) : memref<?xi1>, i1
// CHECK:               scf.parallel (%[[VAL_94:.*]]) = (%[[VAL_90]]) to (%[[VAL_91]]) step (%[[VAL_4]]) {
// CHECK:                 %[[VAL_95:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_94]]] : memref<?xi64>
// CHECK:                 %[[VAL_96:.*]] = index_cast %[[VAL_95]] : i64 to index
// CHECK:                 memref.store %[[VAL_9]], %[[VAL_93]]{{\[}}%[[VAL_96]]] : memref<?xi1>
// CHECK:                 %[[VAL_97:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_94]]] : memref<?xf64>
// CHECK:                 memref.store %[[VAL_97]], %[[VAL_92]]{{\[}}%[[VAL_96]]] : memref<?xf64>
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               %[[VAL_98:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_81]]] : memref<?xi64>
// CHECK:               %[[VAL_99:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_82]]] : memref<?xi64>
// CHECK:               %[[VAL_100:.*]] = index_cast %[[VAL_98]] : i64 to index
// CHECK:               %[[VAL_101:.*]] = index_cast %[[VAL_99]] : i64 to index
// CHECK:               %[[VAL_102:.*]] = scf.for %[[VAL_103:.*]] = %[[VAL_100]] to %[[VAL_101]] step %[[VAL_4]] iter_args(%[[VAL_104:.*]] = %[[VAL_3]]) -> (index) {
// CHECK:                 %[[VAL_105:.*]] = memref.load %[[VAL_21]]{{\[}}%[[VAL_103]]] : memref<?xi64>
// CHECK:                 %[[VAL_106:.*]] = index_cast %[[VAL_105]] : i64 to index
// CHECK:                 %[[VAL_107:.*]] = addi %[[VAL_106]], %[[VAL_4]] : index
// CHECK:                 %[[VAL_108:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_106]]] : memref<?xi64>
// CHECK:                 %[[VAL_109:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_107]]] : memref<?xi64>
// CHECK:                 %[[VAL_110:.*]] = index_cast %[[VAL_108]] : i64 to index
// CHECK:                 %[[VAL_111:.*]] = index_cast %[[VAL_109]] : i64 to index
// CHECK:                 %[[VAL_112:.*]]:2 = scf.for %[[VAL_113:.*]] = %[[VAL_110]] to %[[VAL_111]] step %[[VAL_4]] iter_args(%[[VAL_114:.*]] = %[[VAL_7]], %[[VAL_115:.*]] = %[[VAL_10]]) -> (f64, i1) {
// CHECK:                   %[[VAL_116:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_113]]] : memref<?xi64>
// CHECK:                   %[[VAL_117:.*]] = index_cast %[[VAL_116]] : i64 to index
// CHECK:                   %[[VAL_118:.*]] = memref.load %[[VAL_93]]{{\[}}%[[VAL_117]]] : memref<?xi1>
// CHECK:                   %[[VAL_119:.*]]:2 = scf.if %[[VAL_118]] -> (f64, i1) {
// CHECK:                     %[[VAL_120:.*]] = addf %[[VAL_114]], %[[VAL_8]] : f64
// CHECK:                     scf.yield %[[VAL_120]], %[[VAL_9]] : f64, i1
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_114]], %[[VAL_115]] : f64, i1
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_121:.*]]#0, %[[VAL_121]]#1 : f64, i1
// CHECK:                 }
// CHECK:                 %[[VAL_122:.*]] = scf.if %[[VAL_123:.*]]#1 -> (index) {
// CHECK:                   %[[VAL_124:.*]] = addi %[[VAL_87]], %[[VAL_104]] : index
// CHECK:                   memref.store %[[VAL_105]], %[[VAL_79]]{{\[}}%[[VAL_124]]] : memref<?xi64>
// CHECK:                   memref.store %[[VAL_123]]#0, %[[VAL_80]]{{\[}}%[[VAL_124]]] : memref<?xf64>
// CHECK:                   %[[VAL_125:.*]] = addi %[[VAL_104]], %[[VAL_4]] : index
// CHECK:                   scf.yield %[[VAL_125]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_104]] : index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_126:.*]] : index
// CHECK:               }
// CHECK:             }
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           return %[[VAL_24]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

func @matrix_multiply_mask_plus_pair(%a: tensor<?x?xf64, #CSR64>, %b: tensor<?x?xf64, #CSC64>, %m: tensor<?x?xf64, #CSR64>) -> tensor<?x?xf64, #CSR64> {
    %answer = graphblas.matrix_multiply %a, %b, %m { semiring = "plus_pair" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>, tensor<?x?xf64, #CSR64>) to tensor<?x?xf64, #CSR64>
    return %answer : tensor<?x?xf64, #CSR64>
}

// TODO: Check all type combinations
