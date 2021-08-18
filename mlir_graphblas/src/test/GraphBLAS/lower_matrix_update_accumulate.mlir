// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

// CHECK-DAG:     func private @delSparseMatrix(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>)
// CHECK-DAG:     func private @matrix_resize_values(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index)
// CHECK-DAG:     func private @matrix_resize_index(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index)
// CHECK-DAG:     func private @dup_matrix(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK-DAG:     func private @cast_csr_to_csx(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>

// CHECK-LABEL:   func @matrix_update_accumulate(
// CHECK-SAME:                                   %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> index {
// CHECK-DAG:       %[[VAL_2:.*]] = constant 0 : index
// CHECK-DAG:       %[[VAL_3:.*]] = constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = constant 0 : i64
// CHECK-DAG:       %[[VAL_5:.*]] = constant false
// CHECK-DAG:       %[[VAL_6:.*]] = constant true
// CHECK-DAG:       %[[VAL_7:.*]] = constant 0.000000e+00 : f64
// CHECK:           %[[VAL_9:.*]] = call @cast_csr_to_csx(%[[VAL_1]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_10:.*]] = call @dup_matrix(%[[VAL_9]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_8:.*]] = memref.dim %[[VAL_1]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_11:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_12:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_13:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_14:.*]] = sparse_tensor.pointers %[[VAL_10]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_15:.*]] = sparse_tensor.indices %[[VAL_10]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_16:.*]] = sparse_tensor.values %[[VAL_10]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_17:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           scf.parallel (%[[VAL_18:.*]]) = (%[[VAL_2]]) to (%[[VAL_8]]) step (%[[VAL_3]]) {
// CHECK:             %[[VAL_19:.*]] = addi %[[VAL_18]], %[[VAL_3]] : index
// CHECK:             %[[VAL_20:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_18]]] : memref<?xi64>
// CHECK:             %[[VAL_21:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_19]]] : memref<?xi64>
// CHECK:             %[[VAL_22:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_18]]] : memref<?xi64>
// CHECK:             %[[VAL_23:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_19]]] : memref<?xi64>
// CHECK:             %[[VAL_24:.*]] = cmpi eq, %[[VAL_20]], %[[VAL_21]] : i64
// CHECK:             %[[VAL_25:.*]] = cmpi eq, %[[VAL_22]], %[[VAL_23]] : i64
// CHECK:             %[[VAL_26:.*]] = and %[[VAL_24]], %[[VAL_25]] : i1
// CHECK:             %[[VAL_27:.*]] = scf.if %[[VAL_26]] -> (i64) {
// CHECK:               scf.yield %[[VAL_4]] : i64
// CHECK:             } else {
// CHECK:               %[[VAL_28:.*]] = index_cast %[[VAL_20]] : i64 to index
// CHECK:               %[[VAL_29:.*]] = index_cast %[[VAL_21]] : i64 to index
// CHECK:               %[[VAL_30:.*]] = index_cast %[[VAL_22]] : i64 to index
// CHECK:               %[[VAL_31:.*]] = index_cast %[[VAL_23]] : i64 to index
// CHECK:               %[[VAL_32:.*]]:7 = scf.while (%[[VAL_33:.*]] = %[[VAL_28]], %[[VAL_34:.*]] = %[[VAL_30]], %[[VAL_35:.*]] = %[[VAL_2]], %[[VAL_36:.*]] = %[[VAL_2]], %[[VAL_37:.*]] = %[[VAL_6]], %[[VAL_38:.*]] = %[[VAL_6]], %[[VAL_39:.*]] = %[[VAL_2]]) : (index, index, index, index, i1, i1, index) -> (index, index, index, index, i1, i1, index) {
// CHECK:                 %[[VAL_40:.*]] = cmpi ult, %[[VAL_33]], %[[VAL_29]] : index
// CHECK:                 %[[VAL_41:.*]] = cmpi ult, %[[VAL_34]], %[[VAL_31]] : index
// CHECK:                 %[[VAL_42:.*]] = and %[[VAL_40]], %[[VAL_41]] : i1
// CHECK:                 scf.condition(%[[VAL_42]]) %[[VAL_33]], %[[VAL_34]], %[[VAL_35]], %[[VAL_36]], %[[VAL_37]], %[[VAL_38]], %[[VAL_39]] : index, index, index, index, i1, i1, index
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_43:.*]]: index, %[[VAL_44:.*]]: index, %[[VAL_45:.*]]: index, %[[VAL_46:.*]]: index, %[[VAL_47:.*]]: i1, %[[VAL_48:.*]]: i1, %[[VAL_49:.*]]: index):
// CHECK:                 %[[VAL_50:.*]] = scf.if %[[VAL_47]] -> (index) {
// CHECK:                   %[[VAL_51:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_43]]] : memref<?xi64>
// CHECK:                   %[[VAL_52:.*]] = index_cast %[[VAL_51]] : i64 to index
// CHECK:                   scf.yield %[[VAL_52]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_45]] : index
// CHECK:                 }
// CHECK:                 %[[VAL_53:.*]] = scf.if %[[VAL_48]] -> (index) {
// CHECK:                   %[[VAL_54:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_44]]] : memref<?xi64>
// CHECK:                   %[[VAL_55:.*]] = index_cast %[[VAL_54]] : i64 to index
// CHECK:                   scf.yield %[[VAL_55]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_46]] : index
// CHECK:                 }
// CHECK:                 %[[VAL_56:.*]] = cmpi ult, %[[VAL_57:.*]], %[[VAL_58:.*]] : index
// CHECK:                 %[[VAL_59:.*]] = cmpi ugt, %[[VAL_57]], %[[VAL_58]] : index
// CHECK:                 %[[VAL_60:.*]] = addi %[[VAL_43]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_61:.*]] = addi %[[VAL_44]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_62:.*]] = addi %[[VAL_49]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_63:.*]]:5 = scf.if %[[VAL_56]] -> (index, index, i1, i1, index) {
// CHECK:                   scf.yield %[[VAL_60]], %[[VAL_44]], %[[VAL_6]], %[[VAL_5]], %[[VAL_62]] : index, index, i1, i1, index
// CHECK:                 } else {
// CHECK:                   %[[VAL_64:.*]]:5 = scf.if %[[VAL_59]] -> (index, index, i1, i1, index) {
// CHECK:                     scf.yield %[[VAL_43]], %[[VAL_61]], %[[VAL_5]], %[[VAL_6]], %[[VAL_62]] : index, index, i1, i1, index
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_60]], %[[VAL_61]], %[[VAL_6]], %[[VAL_6]], %[[VAL_62]] : index, index, i1, i1, index
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_65:.*]]#0, %[[VAL_65]]#1, %[[VAL_65]]#2, %[[VAL_65]]#3, %[[VAL_65]]#4 : index, index, i1, i1, index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_66:.*]]#0, %[[VAL_66]]#1, %[[VAL_57]], %[[VAL_58]], %[[VAL_66]]#2, %[[VAL_66]]#3, %[[VAL_66]]#4 : index, index, index, index, i1, i1, index
// CHECK:               }
// CHECK:               %[[VAL_67:.*]] = cmpi ult, %[[VAL_68:.*]]#0, %[[VAL_29]] : index
// CHECK:               %[[VAL_69:.*]] = scf.if %[[VAL_67]] -> (index) {
// CHECK:                 %[[VAL_70:.*]] = subi %[[VAL_29]], %[[VAL_68]]#0 : index
// CHECK:                 %[[VAL_71:.*]] = addi %[[VAL_68]]#6, %[[VAL_70]] : index
// CHECK:                 scf.yield %[[VAL_71]] : index
// CHECK:               } else {
// CHECK:                 %[[VAL_72:.*]] = cmpi ult, %[[VAL_68]]#1, %[[VAL_31]] : index
// CHECK:                 %[[VAL_73:.*]] = scf.if %[[VAL_72]] -> (index) {
// CHECK:                   %[[VAL_74:.*]] = subi %[[VAL_31]], %[[VAL_68]]#1 : index
// CHECK:                   %[[VAL_75:.*]] = addi %[[VAL_68]]#6, %[[VAL_74]] : index
// CHECK:                   scf.yield %[[VAL_75]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_68]]#6 : index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_76:.*]] : index
// CHECK:               }
// CHECK:               %[[VAL_77:.*]] = index_cast %[[VAL_78:.*]] : index to i64
// CHECK:               scf.yield %[[VAL_77]] : i64
// CHECK:             }
// CHECK:             memref.store %[[VAL_27]], %[[VAL_17]]{{\[}}%[[VAL_18]]] : memref<?xi64>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           memref.store %[[VAL_4]], %[[VAL_17]]{{\[}}%[[VAL_8]]] : memref<?xi64>
// CHECK:           scf.for %[[VAL_80:.*]] = %[[VAL_2]] to %[[VAL_8]] step %[[VAL_3]] {
// CHECK:             %[[VAL_81:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_80]]] : memref<?xi64>
// CHECK:             %[[VAL_82:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_8]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_82]], %[[VAL_17]]{{\[}}%[[VAL_80]]] : memref<?xi64>
// CHECK:             %[[VAL_83:.*]] = addi %[[VAL_82]], %[[VAL_81]] : i64
// CHECK:             memref.store %[[VAL_83]], %[[VAL_17]]{{\[}}%[[VAL_8]]] : memref<?xi64>
// CHECK:           }
// CHECK:           %[[VAL_84:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_85:.*]] = memref.dim %[[VAL_1]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_86:.*]] = memref.load %[[VAL_84]]{{\[}}%[[VAL_85]]] : memref<?xi64>
// CHECK:           %[[VAL_87:.*]] = index_cast %[[VAL_86]] : i64 to index
// CHECK:           %[[VAL_88:.*]] = call @cast_csr_to_csx(%[[VAL_1]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           call @matrix_resize_index(%[[VAL_88]], %[[VAL_3]], %[[VAL_87]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           %[[VAL_89:.*]] = call @cast_csr_to_csx(%[[VAL_1]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           call @matrix_resize_values(%[[VAL_89]], %[[VAL_87]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> ()
// CHECK:           %[[VAL_90:.*]] = sparse_tensor.indices %[[VAL_1]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_91:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           scf.parallel (%[[VAL_92:.*]]) = (%[[VAL_2]]) to (%[[VAL_8]]) step (%[[VAL_3]]) {
// CHECK:             %[[VAL_93:.*]] = addi %[[VAL_92]], %[[VAL_3]] : index
// CHECK:             %[[VAL_94:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_92]]] : memref<?xi64>
// CHECK:             %[[VAL_95:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_93]]] : memref<?xi64>
// CHECK:             %[[VAL_96:.*]] = cmpi ne, %[[VAL_94]], %[[VAL_95]] : i64
// CHECK:             scf.if %[[VAL_96]] {
// CHECK:               %[[VAL_97:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_92]]] : memref<?xi64>
// CHECK:               %[[VAL_98:.*]] = index_cast %[[VAL_97]] : i64 to index
// CHECK:               %[[VAL_99:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_92]]] : memref<?xi64>
// CHECK:               %[[VAL_100:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_93]]] : memref<?xi64>
// CHECK:               %[[VAL_101:.*]] = index_cast %[[VAL_99]] : i64 to index
// CHECK:               %[[VAL_102:.*]] = index_cast %[[VAL_100]] : i64 to index
// CHECK:               %[[VAL_103:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_92]]] : memref<?xi64>
// CHECK:               %[[VAL_104:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_93]]] : memref<?xi64>
// CHECK:               %[[VAL_105:.*]] = index_cast %[[VAL_103]] : i64 to index
// CHECK:               %[[VAL_106:.*]] = index_cast %[[VAL_104]] : i64 to index
// CHECK:               %[[VAL_107:.*]]:9 = scf.while (%[[VAL_108:.*]] = %[[VAL_101]], %[[VAL_109:.*]] = %[[VAL_105]], %[[VAL_110:.*]] = %[[VAL_98]], %[[VAL_111:.*]] = %[[VAL_4]], %[[VAL_112:.*]] = %[[VAL_4]], %[[VAL_113:.*]] = %[[VAL_7]], %[[VAL_114:.*]] = %[[VAL_7]], %[[VAL_115:.*]] = %[[VAL_6]], %[[VAL_116:.*]] = %[[VAL_6]]) : (index, index, index, i64, i64, f64, f64, i1, i1) -> (index, index, index, i64, i64, f64, f64, i1, i1) {
// CHECK:                 %[[VAL_117:.*]] = cmpi ult, %[[VAL_108]], %[[VAL_102]] : index
// CHECK:                 %[[VAL_118:.*]] = cmpi ult, %[[VAL_109]], %[[VAL_106]] : index
// CHECK:                 %[[VAL_119:.*]] = and %[[VAL_117]], %[[VAL_118]] : i1
// CHECK:                 scf.condition(%[[VAL_119]]) %[[VAL_108]], %[[VAL_109]], %[[VAL_110]], %[[VAL_111]], %[[VAL_112]], %[[VAL_113]], %[[VAL_114]], %[[VAL_115]], %[[VAL_116]] : index, index, index, i64, i64, f64, f64, i1, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_120:.*]]: index, %[[VAL_121:.*]]: index, %[[VAL_122:.*]]: index, %[[VAL_123:.*]]: i64, %[[VAL_124:.*]]: i64, %[[VAL_125:.*]]: f64, %[[VAL_126:.*]]: f64, %[[VAL_127:.*]]: i1, %[[VAL_128:.*]]: i1):
// CHECK:                 %[[VAL_129:.*]]:2 = scf.if %[[VAL_127]] -> (i64, f64) {
// CHECK:                   %[[VAL_130:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_120]]] : memref<?xi64>
// CHECK:                   %[[VAL_131:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_120]]] : memref<?xf64>
// CHECK:                   scf.yield %[[VAL_130]], %[[VAL_131]] : i64, f64
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_123]], %[[VAL_125]] : i64, f64
// CHECK:                 }
// CHECK:                 %[[VAL_132:.*]]:2 = scf.if %[[VAL_128]] -> (i64, f64) {
// CHECK:                   %[[VAL_133:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_121]]] : memref<?xi64>
// CHECK:                   %[[VAL_134:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_121]]] : memref<?xf64>
// CHECK:                   scf.yield %[[VAL_133]], %[[VAL_134]] : i64, f64
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_124]], %[[VAL_126]] : i64, f64
// CHECK:                 }
// CHECK:                 %[[VAL_135:.*]] = cmpi ult, %[[VAL_136:.*]]#0, %[[VAL_137:.*]]#0 : i64
// CHECK:                 %[[VAL_138:.*]] = cmpi ugt, %[[VAL_136]]#0, %[[VAL_137]]#0 : i64
// CHECK:                 %[[VAL_139:.*]] = addi %[[VAL_120]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_140:.*]] = addi %[[VAL_121]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_141:.*]] = addi %[[VAL_122]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_142:.*]]:5 = scf.if %[[VAL_135]] -> (index, index, index, i1, i1) {
// CHECK:                   memref.store %[[VAL_136]]#0, %[[VAL_90]]{{\[}}%[[VAL_122]]] : memref<?xi64>
// CHECK:                   memref.store %[[VAL_136]]#1, %[[VAL_91]]{{\[}}%[[VAL_122]]] : memref<?xf64>
// CHECK:                   scf.yield %[[VAL_139]], %[[VAL_121]], %[[VAL_141]], %[[VAL_6]], %[[VAL_5]] : index, index, index, i1, i1
// CHECK:                 } else {
// CHECK:                   %[[VAL_143:.*]]:5 = scf.if %[[VAL_138]] -> (index, index, index, i1, i1) {
// CHECK:                     memref.store %[[VAL_137]]#0, %[[VAL_90]]{{\[}}%[[VAL_122]]] : memref<?xi64>
// CHECK:                     memref.store %[[VAL_137]]#1, %[[VAL_91]]{{\[}}%[[VAL_122]]] : memref<?xf64>
// CHECK:                     scf.yield %[[VAL_120]], %[[VAL_140]], %[[VAL_141]], %[[VAL_5]], %[[VAL_6]] : index, index, index, i1, i1
// CHECK:                   } else {
// CHECK:                     memref.store %[[VAL_136]]#0, %[[VAL_90]]{{\[}}%[[VAL_122]]] : memref<?xi64>
// CHECK:                     %[[VAL_144:.*]] = cmpf olt, %[[VAL_136]]#1, %[[VAL_137]]#1 : f64
// CHECK:                     %[[VAL_145:.*]] = select %[[VAL_144]], %[[VAL_136]]#1, %[[VAL_137]]#1 : f64
// CHECK:                     memref.store %[[VAL_145]], %[[VAL_91]]{{\[}}%[[VAL_122]]] : memref<?xf64>
// CHECK:                     scf.yield %[[VAL_139]], %[[VAL_140]], %[[VAL_141]], %[[VAL_6]], %[[VAL_6]] : index, index, index, i1, i1
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_146:.*]]#0, %[[VAL_146]]#1, %[[VAL_146]]#2, %[[VAL_146]]#3, %[[VAL_146]]#4 : index, index, index, i1, i1
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_147:.*]]#0, %[[VAL_147]]#1, %[[VAL_147]]#2, %[[VAL_136]]#0, %[[VAL_137]]#0, %[[VAL_136]]#1, %[[VAL_137]]#1, %[[VAL_147]]#3, %[[VAL_147]]#4 : index, index, index, i64, i64, f64, f64, i1, i1
// CHECK:               }
// CHECK:               %[[VAL_148:.*]] = cmpi ult, %[[VAL_149:.*]]#0, %[[VAL_102]] : index
// CHECK:               %[[VAL_150:.*]] = scf.if %[[VAL_148]] -> (index) {
// CHECK:                 %[[VAL_151:.*]] = scf.for %[[VAL_152:.*]] = %[[VAL_149]]#0 to %[[VAL_102]] step %[[VAL_3]] iter_args(%[[VAL_153:.*]] = %[[VAL_149]]#2) -> (index) {
// CHECK:                   %[[VAL_154:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_152]]] : memref<?xi64>
// CHECK:                   %[[VAL_155:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_152]]] : memref<?xf64>
// CHECK:                   memref.store %[[VAL_154]], %[[VAL_90]]{{\[}}%[[VAL_153]]] : memref<?xi64>
// CHECK:                   memref.store %[[VAL_155]], %[[VAL_91]]{{\[}}%[[VAL_153]]] : memref<?xf64>
// CHECK:                   %[[VAL_156:.*]] = addi %[[VAL_153]], %[[VAL_3]] : index
// CHECK:                   scf.yield %[[VAL_156]] : index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_157:.*]] : index
// CHECK:               } else {
// CHECK:                 %[[VAL_158:.*]] = cmpi ult, %[[VAL_149]]#1, %[[VAL_106]] : index
// CHECK:                 %[[VAL_159:.*]] = scf.if %[[VAL_158]] -> (index) {
// CHECK:                   %[[VAL_160:.*]] = scf.for %[[VAL_161:.*]] = %[[VAL_149]]#1 to %[[VAL_106]] step %[[VAL_3]] iter_args(%[[VAL_162:.*]] = %[[VAL_149]]#2) -> (index) {
// CHECK:                     %[[VAL_163:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_161]]] : memref<?xi64>
// CHECK:                     %[[VAL_164:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_161]]] : memref<?xf64>
// CHECK:                     memref.store %[[VAL_163]], %[[VAL_90]]{{\[}}%[[VAL_162]]] : memref<?xi64>
// CHECK:                     memref.store %[[VAL_164]], %[[VAL_91]]{{\[}}%[[VAL_162]]] : memref<?xf64>
// CHECK:                     %[[VAL_165:.*]] = addi %[[VAL_162]], %[[VAL_3]] : index
// CHECK:                     scf.yield %[[VAL_165]] : index
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_166:.*]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_149]]#2 : index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_167:.*]] : index
// CHECK:               }
// CHECK:             }
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           call @delSparseMatrix(%[[VAL_10]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> ()
// CHECK:           return %[[VAL_2]] : index
// CHECK:         }

func @matrix_update_accumulate(%input: tensor<?x?xf64, #CSR64>, %output: tensor<?x?xf64, #CSR64>) -> index {
    %meaningless = graphblas.update %input -> %output { accumulate_operator = "min" } : tensor<?x?xf64, #CSR64> -> tensor<?x?xf64, #CSR64>
    return %meaningless : index
}

// TODO: Check all type combinations
