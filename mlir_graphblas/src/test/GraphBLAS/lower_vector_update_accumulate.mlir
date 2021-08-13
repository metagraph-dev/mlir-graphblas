// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

// CHECK-DAG:     func private @delSparseVector(tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>)
// CHECK-DAG:     func private @vector_resize_values(tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index)
// CHECK-DAG:     func private @vector_resize_index(tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index)
// CHECK-DAG:     func private @vector_resize_pointers(tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index)
// CHECK-DAG:     func private @vector_resize_dim(tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index)
// CHECK-DAG:     func private @dup_vector(tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>

// CHECK-LABEL:   func @vector_update_accumulate(
// CHECK-SAME:                                   %[[VAL_0:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> index {
// CHECK-DAG:       %[[VAL_2:.*]] = constant 2 : index
// CHECK-DAG:       %[[VAL_3:.*]] = constant 0 : index
// CHECK-DAG:       %[[VAL_4:.*]] = constant 1 : index
// CHECK-DAG:       %[[VAL_5:.*]] = constant 0 : i64
// CHECK-DAG:       %[[VAL_6:.*]] = constant false
// CHECK-DAG:       %[[VAL_7:.*]] = constant true
// CHECK-DAG:       %[[VAL_8:.*]] = constant 0.000000e+00 : f64
// CHECK:           %[[VAL_9:.*]] = tensor.dim %[[VAL_1]], %[[VAL_3]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_10:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_3]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_11:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_3]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_12:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_13:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_3]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_14:.*]] = sparse_tensor.indices %[[VAL_1]], %[[VAL_3]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_15:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_4]]] : memref<?xi64>
// CHECK:           %[[VAL_16:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_4]]] : memref<?xi64>
// CHECK:           %[[VAL_17:.*]] = index_cast %[[VAL_15]] : i64 to index
// CHECK:           %[[VAL_18:.*]] = index_cast %[[VAL_16]] : i64 to index
// CHECK:           %[[VAL_19:.*]]:7 = scf.while (%[[VAL_20:.*]] = %[[VAL_3]], %[[VAL_21:.*]] = %[[VAL_3]], %[[VAL_22:.*]] = %[[VAL_3]], %[[VAL_23:.*]] = %[[VAL_3]], %[[VAL_24:.*]] = %[[VAL_7]], %[[VAL_25:.*]] = %[[VAL_7]], %[[VAL_26:.*]] = %[[VAL_3]]) : (index, index, index, index, i1, i1, index) -> (index, index, index, index, i1, i1, index) {
// CHECK:             %[[VAL_27:.*]] = cmpi ult, %[[VAL_20]], %[[VAL_17]] : index
// CHECK:             %[[VAL_28:.*]] = cmpi ult, %[[VAL_21]], %[[VAL_18]] : index
// CHECK:             %[[VAL_29:.*]] = and %[[VAL_27]], %[[VAL_28]] : i1
// CHECK:             scf.condition(%[[VAL_29]]) %[[VAL_20]], %[[VAL_21]], %[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %[[VAL_25]], %[[VAL_26]] : index, index, index, index, i1, i1, index
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_30:.*]]: index, %[[VAL_31:.*]]: index, %[[VAL_32:.*]]: index, %[[VAL_33:.*]]: index, %[[VAL_34:.*]]: i1, %[[VAL_35:.*]]: i1, %[[VAL_36:.*]]: index):
// CHECK:             %[[VAL_37:.*]] = scf.if %[[VAL_34]] -> (index) {
// CHECK:               %[[VAL_38:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_30]]] : memref<?xi64>
// CHECK:               %[[VAL_39:.*]] = index_cast %[[VAL_38]] : i64 to index
// CHECK:               scf.yield %[[VAL_39]] : index
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_32]] : index
// CHECK:             }
// CHECK:             %[[VAL_40:.*]] = scf.if %[[VAL_35]] -> (index) {
// CHECK:               %[[VAL_41:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_31]]] : memref<?xi64>
// CHECK:               %[[VAL_42:.*]] = index_cast %[[VAL_41]] : i64 to index
// CHECK:               scf.yield %[[VAL_42]] : index
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_33]] : index
// CHECK:             }
// CHECK:             %[[VAL_43:.*]] = cmpi ult, %[[VAL_44:.*]], %[[VAL_45:.*]] : index
// CHECK:             %[[VAL_46:.*]] = cmpi ugt, %[[VAL_44]], %[[VAL_45]] : index
// CHECK:             %[[VAL_47:.*]] = addi %[[VAL_30]], %[[VAL_4]] : index
// CHECK:             %[[VAL_48:.*]] = addi %[[VAL_31]], %[[VAL_4]] : index
// CHECK:             %[[VAL_49:.*]] = addi %[[VAL_36]], %[[VAL_4]] : index
// CHECK:             %[[VAL_50:.*]]:5 = scf.if %[[VAL_43]] -> (index, index, i1, i1, index) {
// CHECK:               scf.yield %[[VAL_47]], %[[VAL_31]], %[[VAL_7]], %[[VAL_6]], %[[VAL_49]] : index, index, i1, i1, index
// CHECK:             } else {
// CHECK:               %[[VAL_51:.*]]:5 = scf.if %[[VAL_46]] -> (index, index, i1, i1, index) {
// CHECK:                 scf.yield %[[VAL_30]], %[[VAL_48]], %[[VAL_6]], %[[VAL_7]], %[[VAL_49]] : index, index, i1, i1, index
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_47]], %[[VAL_48]], %[[VAL_7]], %[[VAL_7]], %[[VAL_49]] : index, index, i1, i1, index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_52:.*]]#0, %[[VAL_52]]#1, %[[VAL_52]]#2, %[[VAL_52]]#3, %[[VAL_52]]#4 : index, index, i1, i1, index
// CHECK:             }
// CHECK:             scf.yield %[[VAL_53:.*]]#0, %[[VAL_53]]#1, %[[VAL_44]], %[[VAL_45]], %[[VAL_53]]#2, %[[VAL_53]]#3, %[[VAL_53]]#4 : index, index, index, index, i1, i1, index
// CHECK:           }
// CHECK:           %[[VAL_54:.*]] = cmpi ult, %[[VAL_55:.*]]#0, %[[VAL_17]] : index
// CHECK:           %[[VAL_56:.*]] = scf.if %[[VAL_54]] -> (index) {
// CHECK:             %[[VAL_57:.*]] = subi %[[VAL_17]], %[[VAL_55]]#0 : index
// CHECK:             %[[VAL_58:.*]] = addi %[[VAL_55]]#6, %[[VAL_57]] : index
// CHECK:             scf.yield %[[VAL_58]] : index
// CHECK:           } else {
// CHECK:             %[[VAL_59:.*]] = cmpi ult, %[[VAL_55]]#1, %[[VAL_18]] : index
// CHECK:             %[[VAL_60:.*]] = scf.if %[[VAL_59]] -> (index) {
// CHECK:               %[[VAL_61:.*]] = subi %[[VAL_18]], %[[VAL_55]]#1 : index
// CHECK:               %[[VAL_62:.*]] = addi %[[VAL_55]]#6, %[[VAL_61]] : index
// CHECK:               scf.yield %[[VAL_62]] : index
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_55]]#6 : index
// CHECK:             }
// CHECK:             scf.yield %[[VAL_63:.*]] : index
// CHECK:           }
// CHECK:           %[[VAL_64:.*]] = index_cast %[[VAL_65:.*]] : index to i64
// CHECK:           %[[VAL_66:.*]] = call @dup_vector(%[[VAL_1]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_67:.*]] = sparse_tensor.indices %[[VAL_66]], %[[VAL_3]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_68:.*]] = sparse_tensor.values %[[VAL_66]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           call @vector_resize_dim(%[[VAL_1]], %[[VAL_3]], %[[VAL_9]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @vector_resize_pointers(%[[VAL_1]], %[[VAL_3]], %[[VAL_2]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @vector_resize_index(%[[VAL_1]], %[[VAL_3]], %[[VAL_65]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @vector_resize_values(%[[VAL_1]], %[[VAL_65]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> ()
// CHECK:           %[[VAL_69:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_3]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           memref.store %[[VAL_64]], %[[VAL_69]]{{\[}}%[[VAL_4]]] : memref<?xi64>
// CHECK:           %[[VAL_70:.*]] = sparse_tensor.indices %[[VAL_1]], %[[VAL_3]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_71:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_72:.*]]:9 = scf.while (%[[VAL_73:.*]] = %[[VAL_3]], %[[VAL_74:.*]] = %[[VAL_3]], %[[VAL_75:.*]] = %[[VAL_3]], %[[VAL_76:.*]] = %[[VAL_5]], %[[VAL_77:.*]] = %[[VAL_5]], %[[VAL_78:.*]] = %[[VAL_8]], %[[VAL_79:.*]] = %[[VAL_8]], %[[VAL_80:.*]] = %[[VAL_7]], %[[VAL_81:.*]] = %[[VAL_7]]) : (index, index, index, i64, i64, f64, f64, i1, i1) -> (index, index, index, i64, i64, f64, f64, i1, i1) {
// CHECK:             %[[VAL_82:.*]] = cmpi ult, %[[VAL_73]], %[[VAL_17]] : index
// CHECK:             %[[VAL_83:.*]] = cmpi ult, %[[VAL_74]], %[[VAL_18]] : index
// CHECK:             %[[VAL_84:.*]] = and %[[VAL_82]], %[[VAL_83]] : i1
// CHECK:             scf.condition(%[[VAL_84]]) %[[VAL_73]], %[[VAL_74]], %[[VAL_75]], %[[VAL_76]], %[[VAL_77]], %[[VAL_78]], %[[VAL_79]], %[[VAL_80]], %[[VAL_81]] : index, index, index, i64, i64, f64, f64, i1, i1
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_85:.*]]: index, %[[VAL_86:.*]]: index, %[[VAL_87:.*]]: index, %[[VAL_88:.*]]: i64, %[[VAL_89:.*]]: i64, %[[VAL_90:.*]]: f64, %[[VAL_91:.*]]: f64, %[[VAL_92:.*]]: i1, %[[VAL_93:.*]]: i1):
// CHECK:             %[[VAL_94:.*]]:2 = scf.if %[[VAL_92]] -> (i64, f64) {
// CHECK:               %[[VAL_95:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_85]]] : memref<?xi64>
// CHECK:               %[[VAL_96:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_85]]] : memref<?xf64>
// CHECK:               scf.yield %[[VAL_95]], %[[VAL_96]] : i64, f64
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_88]], %[[VAL_90]] : i64, f64
// CHECK:             }
// CHECK:             %[[VAL_97:.*]]:2 = scf.if %[[VAL_93]] -> (i64, f64) {
// CHECK:               %[[VAL_98:.*]] = memref.load %[[VAL_67]]{{\[}}%[[VAL_86]]] : memref<?xi64>
// CHECK:               %[[VAL_99:.*]] = memref.load %[[VAL_68]]{{\[}}%[[VAL_86]]] : memref<?xf64>
// CHECK:               scf.yield %[[VAL_98]], %[[VAL_99]] : i64, f64
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_89]], %[[VAL_91]] : i64, f64
// CHECK:             }
// CHECK:             %[[VAL_100:.*]] = cmpi ult, %[[VAL_101:.*]]#0, %[[VAL_102:.*]]#0 : i64
// CHECK:             %[[VAL_103:.*]] = cmpi ugt, %[[VAL_101]]#0, %[[VAL_102]]#0 : i64
// CHECK:             %[[VAL_104:.*]] = addi %[[VAL_85]], %[[VAL_4]] : index
// CHECK:             %[[VAL_105:.*]] = addi %[[VAL_86]], %[[VAL_4]] : index
// CHECK:             %[[VAL_106:.*]] = addi %[[VAL_87]], %[[VAL_4]] : index
// CHECK:             %[[VAL_107:.*]]:5 = scf.if %[[VAL_100]] -> (index, index, index, i1, i1) {
// CHECK:               memref.store %[[VAL_101]]#0, %[[VAL_70]]{{\[}}%[[VAL_87]]] : memref<?xi64>
// CHECK:               memref.store %[[VAL_101]]#1, %[[VAL_71]]{{\[}}%[[VAL_87]]] : memref<?xf64>
// CHECK:               scf.yield %[[VAL_104]], %[[VAL_86]], %[[VAL_106]], %[[VAL_7]], %[[VAL_6]] : index, index, index, i1, i1
// CHECK:             } else {
// CHECK:               %[[VAL_108:.*]]:5 = scf.if %[[VAL_103]] -> (index, index, index, i1, i1) {
// CHECK:                 memref.store %[[VAL_102]]#0, %[[VAL_70]]{{\[}}%[[VAL_87]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_102]]#1, %[[VAL_71]]{{\[}}%[[VAL_87]]] : memref<?xf64>
// CHECK:                 scf.yield %[[VAL_85]], %[[VAL_105]], %[[VAL_106]], %[[VAL_6]], %[[VAL_7]] : index, index, index, i1, i1
// CHECK:               } else {
// CHECK:                 memref.store %[[VAL_101]]#0, %[[VAL_70]]{{\[}}%[[VAL_87]]] : memref<?xi64>
// CHECK:                 %[[VAL_109:.*]] = addf %[[VAL_101]]#1, %[[VAL_102]]#1 : f64
// CHECK:                 memref.store %[[VAL_109]], %[[VAL_71]]{{\[}}%[[VAL_87]]] : memref<?xf64>
// CHECK:                 scf.yield %[[VAL_104]], %[[VAL_105]], %[[VAL_106]], %[[VAL_7]], %[[VAL_7]] : index, index, index, i1, i1
// CHECK:               }
// CHECK:               scf.yield %[[VAL_110:.*]]#0, %[[VAL_110]]#1, %[[VAL_110]]#2, %[[VAL_110]]#3, %[[VAL_110]]#4 : index, index, index, i1, i1
// CHECK:             }
// CHECK:             scf.yield %[[VAL_111:.*]]#0, %[[VAL_111]]#1, %[[VAL_111]]#2, %[[VAL_101]]#0, %[[VAL_102]]#0, %[[VAL_101]]#1, %[[VAL_102]]#1, %[[VAL_111]]#3, %[[VAL_111]]#4 : index, index, index, i64, i64, f64, f64, i1, i1
// CHECK:           }
// CHECK:           %[[VAL_112:.*]] = cmpi ult, %[[VAL_113:.*]]#0, %[[VAL_17]] : index
// CHECK:           %[[VAL_114:.*]] = scf.if %[[VAL_112]] -> (index) {
// CHECK:             %[[VAL_115:.*]] = scf.for %[[VAL_116:.*]] = %[[VAL_113]]#0 to %[[VAL_17]] step %[[VAL_4]] iter_args(%[[VAL_117:.*]] = %[[VAL_113]]#2) -> (index) {
// CHECK:               %[[VAL_118:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_116]]] : memref<?xi64>
// CHECK:               %[[VAL_119:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_116]]] : memref<?xf64>
// CHECK:               memref.store %[[VAL_118]], %[[VAL_70]]{{\[}}%[[VAL_117]]] : memref<?xi64>
// CHECK:               memref.store %[[VAL_119]], %[[VAL_71]]{{\[}}%[[VAL_117]]] : memref<?xf64>
// CHECK:               %[[VAL_120:.*]] = addi %[[VAL_117]], %[[VAL_4]] : index
// CHECK:               scf.yield %[[VAL_120]] : index
// CHECK:             }
// CHECK:             scf.yield %[[VAL_121:.*]] : index
// CHECK:           } else {
// CHECK:             %[[VAL_122:.*]] = cmpi ult, %[[VAL_113]]#1, %[[VAL_18]] : index
// CHECK:             %[[VAL_123:.*]] = scf.if %[[VAL_122]] -> (index) {
// CHECK:               %[[VAL_124:.*]] = scf.for %[[VAL_125:.*]] = %[[VAL_113]]#1 to %[[VAL_18]] step %[[VAL_4]] iter_args(%[[VAL_126:.*]] = %[[VAL_113]]#2) -> (index) {
// CHECK:                 %[[VAL_127:.*]] = memref.load %[[VAL_67]]{{\[}}%[[VAL_125]]] : memref<?xi64>
// CHECK:                 %[[VAL_128:.*]] = memref.load %[[VAL_68]]{{\[}}%[[VAL_125]]] : memref<?xf64>
// CHECK:                 memref.store %[[VAL_127]], %[[VAL_70]]{{\[}}%[[VAL_126]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_128]], %[[VAL_71]]{{\[}}%[[VAL_126]]] : memref<?xf64>
// CHECK:                 %[[VAL_129:.*]] = addi %[[VAL_126]], %[[VAL_4]] : index
// CHECK:                 scf.yield %[[VAL_129]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_130:.*]] : index
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_113]]#2 : index
// CHECK:             }
// CHECK:             scf.yield %[[VAL_131:.*]] : index
// CHECK:           }
// CHECK:           call @delSparseVector(%[[VAL_66]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> ()
// CHECK:           return %[[VAL_132:.*]] : index
// CHECK:         }

func @vector_update_accumulate(%input: tensor<?xf64, #CV64>, %output: tensor<?xf64, #CV64>) -> index {
    %final_position = graphblas.update %input -> %output { accumulate_operator = "plus" } : tensor<?xf64, #CV64> -> tensor<?xf64, #CV64>
    return %final_position : index
}

// TODO: Check all type combinations
