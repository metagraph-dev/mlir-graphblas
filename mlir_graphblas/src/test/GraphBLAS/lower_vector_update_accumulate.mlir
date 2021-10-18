// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

// CHECK-LABEL:   func @vector_update_accumulate(
// CHECK-SAME:                                   %[[VAL_0:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> index {
// CHECK-DAG:       %[[VAL_2:.*]] = constant 0 : index
// CHECK-DAG:       %[[VAL_3:.*]] = constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = constant 0 : i64
// CHECK-DAG:       %[[VAL_5:.*]] = constant false
// CHECK-DAG:       %[[VAL_6:.*]] = constant true
// CHECK-DAG:       %[[VAL_7:.*]] = constant 0.000000e+00 : f64
// CHECK:           %[[VAL_8:.*]] = call @vector_f64_p64i64_to_ptr8(%[[VAL_1]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_9:.*]] = call @dup_tensor(%[[VAL_8]]) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_10:.*]] = call @ptr8_to_vector_f64_p64i64(%[[VAL_9]]) : (!llvm.ptr<i8>) -> tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_11:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_12:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_3]]] : memref<?xi64>
// CHECK:           %[[VAL_13:.*]] = index_cast %[[VAL_12]] : i64 to index
// CHECK:           %[[VAL_14:.*]] = sparse_tensor.pointers %[[VAL_10]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_15:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_3]]] : memref<?xi64>
// CHECK:           %[[VAL_16:.*]] = index_cast %[[VAL_15]] : i64 to index
// CHECK:           %[[VAL_17:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_18:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_19:.*]] = sparse_tensor.indices %[[VAL_10]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_20:.*]] = sparse_tensor.values %[[VAL_10]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_21:.*]]:7 = scf.while (%[[VAL_22:.*]] = %[[VAL_2]], %[[VAL_23:.*]] = %[[VAL_2]], %[[VAL_24:.*]] = %[[VAL_2]], %[[VAL_25:.*]] = %[[VAL_2]], %[[VAL_26:.*]] = %[[VAL_6]], %[[VAL_27:.*]] = %[[VAL_6]], %[[VAL_28:.*]] = %[[VAL_2]]) : (index, index, index, index, i1, i1, index) -> (index, index, index, index, i1, i1, index) {
// CHECK:             %[[VAL_29:.*]] = cmpi ult, %[[VAL_22]], %[[VAL_13]] : index
// CHECK:             %[[VAL_30:.*]] = cmpi ult, %[[VAL_23]], %[[VAL_16]] : index
// CHECK:             %[[VAL_31:.*]] = and %[[VAL_29]], %[[VAL_30]] : i1
// CHECK:             scf.condition(%[[VAL_31]]) %[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %[[VAL_25]], %[[VAL_26]], %[[VAL_27]], %[[VAL_28]] : index, index, index, index, i1, i1, index
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_32:.*]]: index, %[[VAL_33:.*]]: index, %[[VAL_34:.*]]: index, %[[VAL_35:.*]]: index, %[[VAL_36:.*]]: i1, %[[VAL_37:.*]]: i1, %[[VAL_38:.*]]: index):
// CHECK:             %[[VAL_39:.*]] = scf.if %[[VAL_36]] -> (index) {
// CHECK:               %[[VAL_40:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_32]]] : memref<?xi64>
// CHECK:               %[[VAL_41:.*]] = index_cast %[[VAL_40]] : i64 to index
// CHECK:               scf.yield %[[VAL_41]] : index
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_34]] : index
// CHECK:             }
// CHECK:             %[[VAL_42:.*]] = scf.if %[[VAL_37]] -> (index) {
// CHECK:               %[[VAL_43:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_33]]] : memref<?xi64>
// CHECK:               %[[VAL_44:.*]] = index_cast %[[VAL_43]] : i64 to index
// CHECK:               scf.yield %[[VAL_44]] : index
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_35]] : index
// CHECK:             }
// CHECK:             %[[VAL_45:.*]] = cmpi ult, %[[VAL_46:.*]], %[[VAL_47:.*]] : index
// CHECK:             %[[VAL_48:.*]] = cmpi ugt, %[[VAL_46]], %[[VAL_47]] : index
// CHECK:             %[[VAL_49:.*]] = addi %[[VAL_32]], %[[VAL_3]] : index
// CHECK:             %[[VAL_50:.*]] = addi %[[VAL_33]], %[[VAL_3]] : index
// CHECK:             %[[VAL_51:.*]] = addi %[[VAL_38]], %[[VAL_3]] : index
// CHECK:             %[[VAL_52:.*]]:5 = scf.if %[[VAL_45]] -> (index, index, i1, i1, index) {
// CHECK:               scf.yield %[[VAL_49]], %[[VAL_33]], %[[VAL_6]], %[[VAL_5]], %[[VAL_51]] : index, index, i1, i1, index
// CHECK:             } else {
// CHECK:               %[[VAL_53:.*]]:5 = scf.if %[[VAL_48]] -> (index, index, i1, i1, index) {
// CHECK:                 scf.yield %[[VAL_32]], %[[VAL_50]], %[[VAL_5]], %[[VAL_6]], %[[VAL_51]] : index, index, i1, i1, index
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_49]], %[[VAL_50]], %[[VAL_6]], %[[VAL_6]], %[[VAL_51]] : index, index, i1, i1, index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_54:.*]]#0, %[[VAL_54]]#1, %[[VAL_54]]#2, %[[VAL_54]]#3, %[[VAL_54]]#4 : index, index, i1, i1, index
// CHECK:             }
// CHECK:             scf.yield %[[VAL_55:.*]]#0, %[[VAL_55]]#1, %[[VAL_46]], %[[VAL_47]], %[[VAL_55]]#2, %[[VAL_55]]#3, %[[VAL_55]]#4 : index, index, index, index, i1, i1, index
// CHECK:           }
// CHECK:           %[[VAL_56:.*]] = cmpi ult, %[[VAL_57:.*]]#0, %[[VAL_13]] : index
// CHECK:           %[[VAL_58:.*]] = scf.if %[[VAL_56]] -> (index) {
// CHECK:             %[[VAL_59:.*]] = subi %[[VAL_13]], %[[VAL_57]]#0 : index
// CHECK:             %[[VAL_60:.*]] = addi %[[VAL_57]]#6, %[[VAL_59]] : index
// CHECK:             scf.yield %[[VAL_60]] : index
// CHECK:           } else {
// CHECK:             %[[VAL_61:.*]] = cmpi ult, %[[VAL_57]]#1, %[[VAL_16]] : index
// CHECK:             %[[VAL_62:.*]] = scf.if %[[VAL_61]] -> (index) {
// CHECK:               %[[VAL_63:.*]] = subi %[[VAL_16]], %[[VAL_57]]#1 : index
// CHECK:               %[[VAL_64:.*]] = addi %[[VAL_57]]#6, %[[VAL_63]] : index
// CHECK:               scf.yield %[[VAL_64]] : index
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_57]]#6 : index
// CHECK:             }
// CHECK:             scf.yield %[[VAL_65:.*]] : index
// CHECK:           }
// CHECK:           %[[VAL_66:.*]] = index_cast %[[VAL_67:.*]] : index to i64
// CHECK:           %[[VAL_68:.*]] = call @vector_f64_p64i64_to_ptr8(%[[VAL_1]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_index(%[[VAL_68]], %[[VAL_2]], %[[VAL_67]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_69:.*]] = call @vector_f64_p64i64_to_ptr8(%[[VAL_1]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_values(%[[VAL_69]], %[[VAL_67]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:           %[[VAL_70:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           memref.store %[[VAL_66]], %[[VAL_70]]{{\[}}%[[VAL_3]]] : memref<?xi64>
// CHECK:           %[[VAL_71:.*]] = sparse_tensor.indices %[[VAL_1]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_72:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_73:.*]]:9 = scf.while (%[[VAL_74:.*]] = %[[VAL_2]], %[[VAL_75:.*]] = %[[VAL_2]], %[[VAL_76:.*]] = %[[VAL_2]], %[[VAL_77:.*]] = %[[VAL_4]], %[[VAL_78:.*]] = %[[VAL_4]], %[[VAL_79:.*]] = %[[VAL_7]], %[[VAL_80:.*]] = %[[VAL_7]], %[[VAL_81:.*]] = %[[VAL_6]], %[[VAL_82:.*]] = %[[VAL_6]]) : (index, index, index, i64, i64, f64, f64, i1, i1) -> (index, index, index, i64, i64, f64, f64, i1, i1) {
// CHECK:             %[[VAL_83:.*]] = cmpi ult, %[[VAL_74]], %[[VAL_13]] : index
// CHECK:             %[[VAL_84:.*]] = cmpi ult, %[[VAL_75]], %[[VAL_16]] : index
// CHECK:             %[[VAL_85:.*]] = and %[[VAL_83]], %[[VAL_84]] : i1
// CHECK:             scf.condition(%[[VAL_85]]) %[[VAL_74]], %[[VAL_75]], %[[VAL_76]], %[[VAL_77]], %[[VAL_78]], %[[VAL_79]], %[[VAL_80]], %[[VAL_81]], %[[VAL_82]] : index, index, index, i64, i64, f64, f64, i1, i1
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_86:.*]]: index, %[[VAL_87:.*]]: index, %[[VAL_88:.*]]: index, %[[VAL_89:.*]]: i64, %[[VAL_90:.*]]: i64, %[[VAL_91:.*]]: f64, %[[VAL_92:.*]]: f64, %[[VAL_93:.*]]: i1, %[[VAL_94:.*]]: i1):
// CHECK:             %[[VAL_95:.*]]:2 = scf.if %[[VAL_93]] -> (i64, f64) {
// CHECK:               %[[VAL_96:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_86]]] : memref<?xi64>
// CHECK:               %[[VAL_97:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_86]]] : memref<?xf64>
// CHECK:               scf.yield %[[VAL_96]], %[[VAL_97]] : i64, f64
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_89]], %[[VAL_91]] : i64, f64
// CHECK:             }
// CHECK:             %[[VAL_98:.*]]:2 = scf.if %[[VAL_94]] -> (i64, f64) {
// CHECK:               %[[VAL_99:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_87]]] : memref<?xi64>
// CHECK:               %[[VAL_100:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_87]]] : memref<?xf64>
// CHECK:               scf.yield %[[VAL_99]], %[[VAL_100]] : i64, f64
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_90]], %[[VAL_92]] : i64, f64
// CHECK:             }
// CHECK:             %[[VAL_101:.*]] = cmpi ult, %[[VAL_102:.*]]#0, %[[VAL_103:.*]]#0 : i64
// CHECK:             %[[VAL_104:.*]] = cmpi ugt, %[[VAL_102]]#0, %[[VAL_103]]#0 : i64
// CHECK:             %[[VAL_105:.*]] = addi %[[VAL_86]], %[[VAL_3]] : index
// CHECK:             %[[VAL_106:.*]] = addi %[[VAL_87]], %[[VAL_3]] : index
// CHECK:             %[[VAL_107:.*]] = addi %[[VAL_88]], %[[VAL_3]] : index
// CHECK:             %[[VAL_108:.*]]:5 = scf.if %[[VAL_101]] -> (index, index, index, i1, i1) {
// CHECK:               memref.store %[[VAL_102]]#0, %[[VAL_71]]{{\[}}%[[VAL_88]]] : memref<?xi64>
// CHECK:               memref.store %[[VAL_102]]#1, %[[VAL_72]]{{\[}}%[[VAL_88]]] : memref<?xf64>
// CHECK:               scf.yield %[[VAL_105]], %[[VAL_87]], %[[VAL_107]], %[[VAL_6]], %[[VAL_5]] : index, index, index, i1, i1
// CHECK:             } else {
// CHECK:               %[[VAL_109:.*]]:5 = scf.if %[[VAL_104]] -> (index, index, index, i1, i1) {
// CHECK:                 memref.store %[[VAL_103]]#0, %[[VAL_71]]{{\[}}%[[VAL_88]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_103]]#1, %[[VAL_72]]{{\[}}%[[VAL_88]]] : memref<?xf64>
// CHECK:                 scf.yield %[[VAL_86]], %[[VAL_106]], %[[VAL_107]], %[[VAL_5]], %[[VAL_6]] : index, index, index, i1, i1
// CHECK:               } else {
// CHECK:                 memref.store %[[VAL_102]]#0, %[[VAL_71]]{{\[}}%[[VAL_88]]] : memref<?xi64>
// CHECK:                 %[[VAL_110:.*]] = addf %[[VAL_102]]#1, %[[VAL_103]]#1 : f64
// CHECK:                 memref.store %[[VAL_110]], %[[VAL_72]]{{\[}}%[[VAL_88]]] : memref<?xf64>
// CHECK:                 scf.yield %[[VAL_105]], %[[VAL_106]], %[[VAL_107]], %[[VAL_6]], %[[VAL_6]] : index, index, index, i1, i1
// CHECK:               }
// CHECK:               scf.yield %[[VAL_111:.*]]#0, %[[VAL_111]]#1, %[[VAL_111]]#2, %[[VAL_111]]#3, %[[VAL_111]]#4 : index, index, index, i1, i1
// CHECK:             }
// CHECK:             scf.yield %[[VAL_112:.*]]#0, %[[VAL_112]]#1, %[[VAL_112]]#2, %[[VAL_102]]#0, %[[VAL_103]]#0, %[[VAL_102]]#1, %[[VAL_103]]#1, %[[VAL_112]]#3, %[[VAL_112]]#4 : index, index, index, i64, i64, f64, f64, i1, i1
// CHECK:           }
// CHECK:           %[[VAL_113:.*]] = cmpi ult, %[[VAL_114:.*]]#0, %[[VAL_13]] : index
// CHECK:           %[[VAL_115:.*]] = scf.if %[[VAL_113]] -> (index) {
// CHECK:             %[[VAL_116:.*]] = scf.for %[[VAL_117:.*]] = %[[VAL_114]]#0 to %[[VAL_13]] step %[[VAL_3]] iter_args(%[[VAL_118:.*]] = %[[VAL_114]]#2) -> (index) {
// CHECK:               %[[VAL_119:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_117]]] : memref<?xi64>
// CHECK:               %[[VAL_120:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_117]]] : memref<?xf64>
// CHECK:               memref.store %[[VAL_119]], %[[VAL_71]]{{\[}}%[[VAL_118]]] : memref<?xi64>
// CHECK:               memref.store %[[VAL_120]], %[[VAL_72]]{{\[}}%[[VAL_118]]] : memref<?xf64>
// CHECK:               %[[VAL_121:.*]] = addi %[[VAL_118]], %[[VAL_3]] : index
// CHECK:               scf.yield %[[VAL_121]] : index
// CHECK:             }
// CHECK:             scf.yield %[[VAL_122:.*]] : index
// CHECK:           } else {
// CHECK:             %[[VAL_123:.*]] = cmpi ult, %[[VAL_114]]#1, %[[VAL_16]] : index
// CHECK:             %[[VAL_124:.*]] = scf.if %[[VAL_123]] -> (index) {
// CHECK:               %[[VAL_125:.*]] = scf.for %[[VAL_126:.*]] = %[[VAL_114]]#1 to %[[VAL_16]] step %[[VAL_3]] iter_args(%[[VAL_127:.*]] = %[[VAL_114]]#2) -> (index) {
// CHECK:                 %[[VAL_128:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_126]]] : memref<?xi64>
// CHECK:                 %[[VAL_129:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_126]]] : memref<?xf64>
// CHECK:                 memref.store %[[VAL_128]], %[[VAL_71]]{{\[}}%[[VAL_127]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_129]], %[[VAL_72]]{{\[}}%[[VAL_127]]] : memref<?xf64>
// CHECK:                 %[[VAL_130:.*]] = addi %[[VAL_127]], %[[VAL_3]] : index
// CHECK:                 scf.yield %[[VAL_130]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_131:.*]] : index
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_114]]#2 : index
// CHECK:             }
// CHECK:             scf.yield %[[VAL_132:.*]] : index
// CHECK:           }
// CHECK:           sparse_tensor.release %[[VAL_10]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           return %[[VAL_2]] : index
// CHECK:         }

func @vector_update_accumulate(%input: tensor<?xf64, #CV64>, %output: tensor<?xf64, #CV64>) -> index {
    %final_position = graphblas.update %input -> %output { accumulate_operator = "plus" } : tensor<?xf64, #CV64> -> tensor<?xf64, #CV64>
    return %final_position : index
}

// TODO: Check all type combinations
