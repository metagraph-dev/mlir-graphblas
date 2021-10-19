// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

// CHECK-LABEL:   func @matrix_update_accumulate(
// CHECK-SAME:                                    %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                    %[[VAL_1:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> index {
// CHECK-DAG:       %[[VAL_2:.*]] = constant 0 : index
// CHECK-DAG:       %[[VAL_3:.*]] = constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = constant 0 : i64
// CHECK-DAG:       %[[VAL_5:.*]] = constant false
// CHECK-DAG:       %[[VAL_6:.*]] = constant true
// CHECK-DAG:       %[[VAL_7:.*]] = constant 0.000000e+00 : f64
// CHECK:           %[[VAL_8:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_1]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_9:.*]] = call @dup_tensor(%[[VAL_8]]) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_10:.*]] = call @ptr8_to_matrix_csr_f64_p64i64(%[[VAL_9]]) : (!llvm.ptr<i8>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_11:.*]] = tensor.dim %[[VAL_1]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_12:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_13:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_14:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_15:.*]] = sparse_tensor.pointers %[[VAL_10]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_16:.*]] = sparse_tensor.indices %[[VAL_10]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_17:.*]] = sparse_tensor.values %[[VAL_10]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_18:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           scf.parallel (%[[VAL_19:.*]]) = (%[[VAL_2]]) to (%[[VAL_11]]) step (%[[VAL_3]]) {
// CHECK:             %[[VAL_20:.*]] = addi %[[VAL_19]], %[[VAL_3]] : index
// CHECK:             %[[VAL_21:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_19]]] : memref<?xi64>
// CHECK:             %[[VAL_22:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_20]]] : memref<?xi64>
// CHECK:             %[[VAL_23:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_19]]] : memref<?xi64>
// CHECK:             %[[VAL_24:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_20]]] : memref<?xi64>
// CHECK:             %[[VAL_25:.*]] = cmpi eq, %[[VAL_21]], %[[VAL_22]] : i64
// CHECK:             %[[VAL_26:.*]] = cmpi eq, %[[VAL_23]], %[[VAL_24]] : i64
// CHECK:             %[[VAL_27:.*]] = and %[[VAL_25]], %[[VAL_26]] : i1
// CHECK:             %[[VAL_28:.*]] = scf.if %[[VAL_27]] -> (i64) {
// CHECK:               scf.yield %[[VAL_4]] : i64
// CHECK:             } else {
// CHECK:               %[[VAL_29:.*]] = index_cast %[[VAL_21]] : i64 to index
// CHECK:               %[[VAL_30:.*]] = index_cast %[[VAL_22]] : i64 to index
// CHECK:               %[[VAL_31:.*]] = index_cast %[[VAL_23]] : i64 to index
// CHECK:               %[[VAL_32:.*]] = index_cast %[[VAL_24]] : i64 to index
// CHECK:               %[[VAL_33:.*]]:7 = scf.while (%[[VAL_34:.*]] = %[[VAL_29]], %[[VAL_35:.*]] = %[[VAL_31]], %[[VAL_36:.*]] = %[[VAL_2]], %[[VAL_37:.*]] = %[[VAL_2]], %[[VAL_38:.*]] = %[[VAL_6]], %[[VAL_39:.*]] = %[[VAL_6]], %[[VAL_40:.*]] = %[[VAL_2]]) : (index, index, index, index, i1, i1, index) -> (index, index, index, index, i1, i1, index) {
// CHECK:                 %[[VAL_41:.*]] = cmpi ult, %[[VAL_34]], %[[VAL_30]] : index
// CHECK:                 %[[VAL_42:.*]] = cmpi ult, %[[VAL_35]], %[[VAL_32]] : index
// CHECK:                 %[[VAL_43:.*]] = and %[[VAL_41]], %[[VAL_42]] : i1
// CHECK:                 scf.condition(%[[VAL_43]]) %[[VAL_34]], %[[VAL_35]], %[[VAL_36]], %[[VAL_37]], %[[VAL_38]], %[[VAL_39]], %[[VAL_40]] : index, index, index, index, i1, i1, index
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_44:.*]]: index, %[[VAL_45:.*]]: index, %[[VAL_46:.*]]: index, %[[VAL_47:.*]]: index, %[[VAL_48:.*]]: i1, %[[VAL_49:.*]]: i1, %[[VAL_50:.*]]: index):
// CHECK:                 %[[VAL_51:.*]] = scf.if %[[VAL_48]] -> (index) {
// CHECK:                   %[[VAL_52:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_44]]] : memref<?xi64>
// CHECK:                   %[[VAL_53:.*]] = index_cast %[[VAL_52]] : i64 to index
// CHECK:                   scf.yield %[[VAL_53]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_46]] : index
// CHECK:                 }
// CHECK:                 %[[VAL_54:.*]] = scf.if %[[VAL_49]] -> (index) {
// CHECK:                   %[[VAL_55:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_45]]] : memref<?xi64>
// CHECK:                   %[[VAL_56:.*]] = index_cast %[[VAL_55]] : i64 to index
// CHECK:                   scf.yield %[[VAL_56]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_47]] : index
// CHECK:                 }
// CHECK:                 %[[VAL_57:.*]] = cmpi ult, %[[VAL_58:.*]], %[[VAL_59:.*]] : index
// CHECK:                 %[[VAL_60:.*]] = cmpi ugt, %[[VAL_58]], %[[VAL_59]] : index
// CHECK:                 %[[VAL_61:.*]] = addi %[[VAL_44]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_62:.*]] = addi %[[VAL_45]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_63:.*]] = addi %[[VAL_50]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_64:.*]]:5 = scf.if %[[VAL_57]] -> (index, index, i1, i1, index) {
// CHECK:                   scf.yield %[[VAL_61]], %[[VAL_45]], %[[VAL_6]], %[[VAL_5]], %[[VAL_63]] : index, index, i1, i1, index
// CHECK:                 } else {
// CHECK:                   %[[VAL_65:.*]]:5 = scf.if %[[VAL_60]] -> (index, index, i1, i1, index) {
// CHECK:                     scf.yield %[[VAL_44]], %[[VAL_62]], %[[VAL_5]], %[[VAL_6]], %[[VAL_63]] : index, index, i1, i1, index
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_61]], %[[VAL_62]], %[[VAL_6]], %[[VAL_6]], %[[VAL_63]] : index, index, i1, i1, index
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_66:.*]]#0, %[[VAL_66]]#1, %[[VAL_66]]#2, %[[VAL_66]]#3, %[[VAL_66]]#4 : index, index, i1, i1, index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_67:.*]]#0, %[[VAL_67]]#1, %[[VAL_58]], %[[VAL_59]], %[[VAL_67]]#2, %[[VAL_67]]#3, %[[VAL_67]]#4 : index, index, index, index, i1, i1, index
// CHECK:               }
// CHECK:               %[[VAL_68:.*]] = cmpi ult, %[[VAL_69:.*]]#0, %[[VAL_30]] : index
// CHECK:               %[[VAL_70:.*]] = scf.if %[[VAL_68]] -> (index) {
// CHECK:                 %[[VAL_71:.*]] = subi %[[VAL_30]], %[[VAL_69]]#0 : index
// CHECK:                 %[[VAL_72:.*]] = addi %[[VAL_69]]#6, %[[VAL_71]] : index
// CHECK:                 scf.yield %[[VAL_72]] : index
// CHECK:               } else {
// CHECK:                 %[[VAL_73:.*]] = cmpi ult, %[[VAL_69]]#1, %[[VAL_32]] : index
// CHECK:                 %[[VAL_74:.*]] = scf.if %[[VAL_73]] -> (index) {
// CHECK:                   %[[VAL_75:.*]] = subi %[[VAL_32]], %[[VAL_69]]#1 : index
// CHECK:                   %[[VAL_76:.*]] = addi %[[VAL_69]]#6, %[[VAL_75]] : index
// CHECK:                   scf.yield %[[VAL_76]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_69]]#6 : index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_77:.*]] : index
// CHECK:               }
// CHECK:               %[[VAL_78:.*]] = index_cast %[[VAL_79:.*]] : index to i64
// CHECK:               scf.yield %[[VAL_78]] : i64
// CHECK:             }
// CHECK:             memref.store %[[VAL_80:.*]], %[[VAL_18]]{{\[}}%[[VAL_19]]] : memref<?xi64>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           memref.store %[[VAL_4]], %[[VAL_18]]{{\[}}%[[VAL_11]]] : memref<?xi64>
// CHECK:           scf.for %[[VAL_81:.*]] = %[[VAL_2]] to %[[VAL_11]] step %[[VAL_3]] {
// CHECK:             %[[VAL_82:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_81]]] : memref<?xi64>
// CHECK:             %[[VAL_83:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_11]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_83]], %[[VAL_18]]{{\[}}%[[VAL_81]]] : memref<?xi64>
// CHECK:             %[[VAL_84:.*]] = addi %[[VAL_83]], %[[VAL_82]] : i64
// CHECK:             memref.store %[[VAL_84]], %[[VAL_18]]{{\[}}%[[VAL_11]]] : memref<?xi64>
// CHECK:           }
// CHECK:           %[[VAL_85:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_86:.*]] = tensor.dim %[[VAL_1]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_87:.*]] = memref.load %[[VAL_85]]{{\[}}%[[VAL_86]]] : memref<?xi64>
// CHECK:           %[[VAL_88:.*]] = index_cast %[[VAL_87]] : i64 to index
// CHECK:           %[[VAL_89:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_1]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_index(%[[VAL_89]], %[[VAL_3]], %[[VAL_88]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_90:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_1]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_values(%[[VAL_90]], %[[VAL_88]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:           %[[VAL_91:.*]] = sparse_tensor.indices %[[VAL_1]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_92:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           scf.parallel (%[[VAL_93:.*]]) = (%[[VAL_2]]) to (%[[VAL_11]]) step (%[[VAL_3]]) {
// CHECK:             %[[VAL_94:.*]] = addi %[[VAL_93]], %[[VAL_3]] : index
// CHECK:             %[[VAL_95:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_93]]] : memref<?xi64>
// CHECK:             %[[VAL_96:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_94]]] : memref<?xi64>
// CHECK:             %[[VAL_97:.*]] = cmpi ne, %[[VAL_95]], %[[VAL_96]] : i64
// CHECK:             scf.if %[[VAL_97]] {
// CHECK:               %[[VAL_98:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_93]]] : memref<?xi64>
// CHECK:               %[[VAL_99:.*]] = index_cast %[[VAL_98]] : i64 to index
// CHECK:               %[[VAL_100:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_93]]] : memref<?xi64>
// CHECK:               %[[VAL_101:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_94]]] : memref<?xi64>
// CHECK:               %[[VAL_102:.*]] = index_cast %[[VAL_100]] : i64 to index
// CHECK:               %[[VAL_103:.*]] = index_cast %[[VAL_101]] : i64 to index
// CHECK:               %[[VAL_104:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_93]]] : memref<?xi64>
// CHECK:               %[[VAL_105:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_94]]] : memref<?xi64>
// CHECK:               %[[VAL_106:.*]] = index_cast %[[VAL_104]] : i64 to index
// CHECK:               %[[VAL_107:.*]] = index_cast %[[VAL_105]] : i64 to index
// CHECK:               %[[VAL_108:.*]]:9 = scf.while (%[[VAL_109:.*]] = %[[VAL_102]], %[[VAL_110:.*]] = %[[VAL_106]], %[[VAL_111:.*]] = %[[VAL_99]], %[[VAL_112:.*]] = %[[VAL_4]], %[[VAL_113:.*]] = %[[VAL_4]], %[[VAL_114:.*]] = %[[VAL_7]], %[[VAL_115:.*]] = %[[VAL_7]], %[[VAL_116:.*]] = %[[VAL_6]], %[[VAL_117:.*]] = %[[VAL_6]]) : (index, index, index, i64, i64, f64, f64, i1, i1) -> (index, index, index, i64, i64, f64, f64, i1, i1) {
// CHECK:                 %[[VAL_118:.*]] = cmpi ult, %[[VAL_109]], %[[VAL_103]] : index
// CHECK:                 %[[VAL_119:.*]] = cmpi ult, %[[VAL_110]], %[[VAL_107]] : index
// CHECK:                 %[[VAL_120:.*]] = and %[[VAL_118]], %[[VAL_119]] : i1
// CHECK:                 scf.condition(%[[VAL_120]]) %[[VAL_109]], %[[VAL_110]], %[[VAL_111]], %[[VAL_112]], %[[VAL_113]], %[[VAL_114]], %[[VAL_115]], %[[VAL_116]], %[[VAL_117]] : index, index, index, i64, i64, f64, f64, i1, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_121:.*]]: index, %[[VAL_122:.*]]: index, %[[VAL_123:.*]]: index, %[[VAL_124:.*]]: i64, %[[VAL_125:.*]]: i64, %[[VAL_126:.*]]: f64, %[[VAL_127:.*]]: f64, %[[VAL_128:.*]]: i1, %[[VAL_129:.*]]: i1):
// CHECK:                 %[[VAL_130:.*]]:2 = scf.if %[[VAL_128]] -> (i64, f64) {
// CHECK:                   %[[VAL_131:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_121]]] : memref<?xi64>
// CHECK:                   %[[VAL_132:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_121]]] : memref<?xf64>
// CHECK:                   scf.yield %[[VAL_131]], %[[VAL_132]] : i64, f64
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_124]], %[[VAL_126]] : i64, f64
// CHECK:                 }
// CHECK:                 %[[VAL_133:.*]]:2 = scf.if %[[VAL_129]] -> (i64, f64) {
// CHECK:                   %[[VAL_134:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_122]]] : memref<?xi64>
// CHECK:                   %[[VAL_135:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_122]]] : memref<?xf64>
// CHECK:                   scf.yield %[[VAL_134]], %[[VAL_135]] : i64, f64
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_125]], %[[VAL_127]] : i64, f64
// CHECK:                 }
// CHECK:                 %[[VAL_136:.*]] = cmpi ult, %[[VAL_137:.*]]#0, %[[VAL_138:.*]]#0 : i64
// CHECK:                 %[[VAL_139:.*]] = cmpi ugt, %[[VAL_137]]#0, %[[VAL_138]]#0 : i64
// CHECK:                 %[[VAL_140:.*]] = addi %[[VAL_121]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_141:.*]] = addi %[[VAL_122]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_142:.*]] = addi %[[VAL_123]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_143:.*]]:5 = scf.if %[[VAL_136]] -> (index, index, index, i1, i1) {
// CHECK:                   memref.store %[[VAL_137]]#0, %[[VAL_91]]{{\[}}%[[VAL_123]]] : memref<?xi64>
// CHECK:                   memref.store %[[VAL_137]]#1, %[[VAL_92]]{{\[}}%[[VAL_123]]] : memref<?xf64>
// CHECK:                   scf.yield %[[VAL_140]], %[[VAL_122]], %[[VAL_142]], %[[VAL_6]], %[[VAL_5]] : index, index, index, i1, i1
// CHECK:                 } else {
// CHECK:                   %[[VAL_144:.*]]:5 = scf.if %[[VAL_139]] -> (index, index, index, i1, i1) {
// CHECK:                     memref.store %[[VAL_138]]#0, %[[VAL_91]]{{\[}}%[[VAL_123]]] : memref<?xi64>
// CHECK:                     memref.store %[[VAL_138]]#1, %[[VAL_92]]{{\[}}%[[VAL_123]]] : memref<?xf64>
// CHECK:                     scf.yield %[[VAL_121]], %[[VAL_141]], %[[VAL_142]], %[[VAL_5]], %[[VAL_6]] : index, index, index, i1, i1
// CHECK:                   } else {
// CHECK:                     memref.store %[[VAL_137]]#0, %[[VAL_91]]{{\[}}%[[VAL_123]]] : memref<?xi64>
// CHECK:                     %[[VAL_145:.*]] = cmpf olt, %[[VAL_137]]#1, %[[VAL_138]]#1 : f64
// CHECK:                     %[[VAL_146:.*]] = select %[[VAL_145]], %[[VAL_137]]#1, %[[VAL_138]]#1 : f64
// CHECK:                     memref.store %[[VAL_146]], %[[VAL_92]]{{\[}}%[[VAL_123]]] : memref<?xf64>
// CHECK:                     scf.yield %[[VAL_140]], %[[VAL_141]], %[[VAL_142]], %[[VAL_6]], %[[VAL_6]] : index, index, index, i1, i1
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_147:.*]]#0, %[[VAL_147]]#1, %[[VAL_147]]#2, %[[VAL_147]]#3, %[[VAL_147]]#4 : index, index, index, i1, i1
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_148:.*]]#0, %[[VAL_148]]#1, %[[VAL_148]]#2, %[[VAL_137]]#0, %[[VAL_138]]#0, %[[VAL_137]]#1, %[[VAL_138]]#1, %[[VAL_148]]#3, %[[VAL_148]]#4 : index, index, index, i64, i64, f64, f64, i1, i1
// CHECK:               }
// CHECK:               %[[VAL_149:.*]] = cmpi ult, %[[VAL_150:.*]]#0, %[[VAL_103]] : index
// CHECK:               %[[VAL_151:.*]] = scf.if %[[VAL_149]] -> (index) {
// CHECK:                 %[[VAL_152:.*]] = scf.for %[[VAL_153:.*]] = %[[VAL_150]]#0 to %[[VAL_103]] step %[[VAL_3]] iter_args(%[[VAL_154:.*]] = %[[VAL_150]]#2) -> (index) {
// CHECK:                   %[[VAL_155:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_153]]] : memref<?xi64>
// CHECK:                   %[[VAL_156:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_153]]] : memref<?xf64>
// CHECK:                   memref.store %[[VAL_155]], %[[VAL_91]]{{\[}}%[[VAL_154]]] : memref<?xi64>
// CHECK:                   memref.store %[[VAL_156]], %[[VAL_92]]{{\[}}%[[VAL_154]]] : memref<?xf64>
// CHECK:                   %[[VAL_157:.*]] = addi %[[VAL_154]], %[[VAL_3]] : index
// CHECK:                   scf.yield %[[VAL_157]] : index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_158:.*]] : index
// CHECK:               } else {
// CHECK:                 %[[VAL_159:.*]] = cmpi ult, %[[VAL_150]]#1, %[[VAL_107]] : index
// CHECK:                 %[[VAL_160:.*]] = scf.if %[[VAL_159]] -> (index) {
// CHECK:                   %[[VAL_161:.*]] = scf.for %[[VAL_162:.*]] = %[[VAL_150]]#1 to %[[VAL_107]] step %[[VAL_3]] iter_args(%[[VAL_163:.*]] = %[[VAL_150]]#2) -> (index) {
// CHECK:                     %[[VAL_164:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_162]]] : memref<?xi64>
// CHECK:                     %[[VAL_165:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_162]]] : memref<?xf64>
// CHECK:                     memref.store %[[VAL_164]], %[[VAL_91]]{{\[}}%[[VAL_163]]] : memref<?xi64>
// CHECK:                     memref.store %[[VAL_165]], %[[VAL_92]]{{\[}}%[[VAL_163]]] : memref<?xf64>
// CHECK:                     %[[VAL_166:.*]] = addi %[[VAL_163]], %[[VAL_3]] : index
// CHECK:                     scf.yield %[[VAL_166]] : index
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_167:.*]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_150]]#2 : index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_168:.*]] : index
// CHECK:               }
// CHECK:             }
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           sparse_tensor.release %[[VAL_10]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           return %[[VAL_2]] : index
// CHECK:         }

func @matrix_update_accumulate(%input: tensor<?x?xf64, #CSR64>, %output: tensor<?x?xf64, #CSR64>) -> index {
    %meaningless = graphblas.update %input -> %output { accumulate_operator = "min" } : tensor<?x?xf64, #CSR64> -> tensor<?x?xf64, #CSR64>
    return %meaningless : index
}

// TODO: Check all type combinations
