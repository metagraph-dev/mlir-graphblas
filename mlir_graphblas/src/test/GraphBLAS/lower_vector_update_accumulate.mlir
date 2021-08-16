// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

// CHECK-DAG:     func private @delSparseVector(tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>)
// CHECK-DAG:     func private @vector_resize_values(tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index)
// CHECK-DAG:     func private @vector_resize_index(tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index)
// CHECK-DAG:     func private @dup_vector(tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>

// CHECK-LABEL:   func @vector_update_accumulate(
// CHECK-SAME:                                   %[[VAL_0:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> index {
// CHECK-DAG:       %[[VAL_2:.*]] = constant 0 : index
// CHECK-DAG:       %[[VAL_3:.*]] = constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = constant 0 : i64
// CHECK-DAG:       %[[VAL_5:.*]] = constant false
// CHECK-DAG:       %[[VAL_6:.*]] = constant true
// CHECK-DAG:       %[[VAL_7:.*]] = constant 0.000000e+00 : f64
// CHECK:           %[[VAL_8:.*]] = call @dup_vector(%[[VAL_1]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_9:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_10:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_3]]] : memref<?xi64>
// CHECK:           %[[VAL_11:.*]] = index_cast %[[VAL_10]] : i64 to index
// CHECK:           %[[VAL_12:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_13:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_3]]] : memref<?xi64>
// CHECK:           %[[VAL_14:.*]] = index_cast %[[VAL_13]] : i64 to index
// CHECK:           %[[VAL_15:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_16:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_17:.*]] = sparse_tensor.indices %[[VAL_8]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_18:.*]] = sparse_tensor.values %[[VAL_8]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_19:.*]]:7 = scf.while (%[[VAL_20:.*]] = %[[VAL_2]], %[[VAL_21:.*]] = %[[VAL_2]], %[[VAL_22:.*]] = %[[VAL_2]], %[[VAL_23:.*]] = %[[VAL_2]], %[[VAL_24:.*]] = %[[VAL_6]], %[[VAL_25:.*]] = %[[VAL_6]], %[[VAL_26:.*]] = %[[VAL_2]]) : (index, index, index, index, i1, i1, index) -> (index, index, index, index, i1, i1, index) {
// CHECK:             %[[VAL_27:.*]] = cmpi ult, %[[VAL_20]], %[[VAL_11]] : index
// CHECK:             %[[VAL_28:.*]] = cmpi ult, %[[VAL_21]], %[[VAL_14]] : index
// CHECK:             %[[VAL_29:.*]] = and %[[VAL_27]], %[[VAL_28]] : i1
// CHECK:             scf.condition(%[[VAL_29]]) %[[VAL_20]], %[[VAL_21]], %[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %[[VAL_25]], %[[VAL_26]] : index, index, index, index, i1, i1, index
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_30:.*]]: index, %[[VAL_31:.*]]: index, %[[VAL_32:.*]]: index, %[[VAL_33:.*]]: index, %[[VAL_34:.*]]: i1, %[[VAL_35:.*]]: i1, %[[VAL_36:.*]]: index):
// CHECK:             %[[VAL_37:.*]] = scf.if %[[VAL_34]] -> (index) {
// CHECK:               %[[VAL_38:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_30]]] : memref<?xi64>
// CHECK:               %[[VAL_39:.*]] = index_cast %[[VAL_38]] : i64 to index
// CHECK:               scf.yield %[[VAL_39]] : index
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_32]] : index
// CHECK:             }
// CHECK:             %[[VAL_40:.*]] = scf.if %[[VAL_35]] -> (index) {
// CHECK:               %[[VAL_41:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_31]]] : memref<?xi64>
// CHECK:               %[[VAL_42:.*]] = index_cast %[[VAL_41]] : i64 to index
// CHECK:               scf.yield %[[VAL_42]] : index
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_33]] : index
// CHECK:             }
// CHECK:             %[[VAL_43:.*]] = cmpi ult, %[[VAL_44:.*]], %[[VAL_45:.*]] : index
// CHECK:             %[[VAL_46:.*]] = cmpi ugt, %[[VAL_44]], %[[VAL_45]] : index
// CHECK:             %[[VAL_47:.*]] = addi %[[VAL_30]], %[[VAL_3]] : index
// CHECK:             %[[VAL_48:.*]] = addi %[[VAL_31]], %[[VAL_3]] : index
// CHECK:             %[[VAL_49:.*]] = addi %[[VAL_36]], %[[VAL_3]] : index
// CHECK:             %[[VAL_50:.*]]:5 = scf.if %[[VAL_43]] -> (index, index, i1, i1, index) {
// CHECK:               scf.yield %[[VAL_47]], %[[VAL_31]], %[[VAL_6]], %[[VAL_5]], %[[VAL_49]] : index, index, i1, i1, index
// CHECK:             } else {
// CHECK:               %[[VAL_51:.*]]:5 = scf.if %[[VAL_46]] -> (index, index, i1, i1, index) {
// CHECK:                 scf.yield %[[VAL_30]], %[[VAL_48]], %[[VAL_5]], %[[VAL_6]], %[[VAL_49]] : index, index, i1, i1, index
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_47]], %[[VAL_48]], %[[VAL_6]], %[[VAL_6]], %[[VAL_49]] : index, index, i1, i1, index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_52:.*]]#0, %[[VAL_52]]#1, %[[VAL_52]]#2, %[[VAL_52]]#3, %[[VAL_52]]#4 : index, index, i1, i1, index
// CHECK:             }
// CHECK:             scf.yield %[[VAL_53:.*]]#0, %[[VAL_53]]#1, %[[VAL_44]], %[[VAL_45]], %[[VAL_53]]#2, %[[VAL_53]]#3, %[[VAL_53]]#4 : index, index, index, index, i1, i1, index
// CHECK:           }
// CHECK:           %[[VAL_54:.*]] = cmpi ult, %[[VAL_55:.*]]#0, %[[VAL_11]] : index
// CHECK:           %[[VAL_56:.*]] = scf.if %[[VAL_54]] -> (index) {
// CHECK:             %[[VAL_57:.*]] = subi %[[VAL_11]], %[[VAL_55]]#0 : index
// CHECK:             %[[VAL_58:.*]] = addi %[[VAL_55]]#6, %[[VAL_57]] : index
// CHECK:             scf.yield %[[VAL_58]] : index
// CHECK:           } else {
// CHECK:             %[[VAL_59:.*]] = cmpi ult, %[[VAL_55]]#1, %[[VAL_14]] : index
// CHECK:             %[[VAL_60:.*]] = scf.if %[[VAL_59]] -> (index) {
// CHECK:               %[[VAL_61:.*]] = subi %[[VAL_14]], %[[VAL_55]]#1 : index
// CHECK:               %[[VAL_62:.*]] = addi %[[VAL_55]]#6, %[[VAL_61]] : index
// CHECK:               scf.yield %[[VAL_62]] : index
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_55]]#6 : index
// CHECK:             }
// CHECK:             scf.yield %[[VAL_63:.*]] : index
// CHECK:           }
// CHECK:           %[[VAL_64:.*]] = index_cast %[[VAL_65:.*]] : index to i64
// CHECK:           call @vector_resize_index(%[[VAL_1]], %[[VAL_2]], %[[VAL_65]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @vector_resize_values(%[[VAL_1]], %[[VAL_65]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> ()
// CHECK:           %[[VAL_66:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           memref.store %[[VAL_64]], %[[VAL_66]]{{\[}}%[[VAL_3]]] : memref<?xi64>
// CHECK:           %[[VAL_67:.*]] = sparse_tensor.indices %[[VAL_1]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_68:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_69:.*]]:9 = scf.while (%[[VAL_70:.*]] = %[[VAL_2]], %[[VAL_71:.*]] = %[[VAL_2]], %[[VAL_72:.*]] = %[[VAL_2]], %[[VAL_73:.*]] = %[[VAL_4]], %[[VAL_74:.*]] = %[[VAL_4]], %[[VAL_75:.*]] = %[[VAL_7]], %[[VAL_76:.*]] = %[[VAL_7]], %[[VAL_77:.*]] = %[[VAL_6]], %[[VAL_78:.*]] = %[[VAL_6]]) : (index, index, index, i64, i64, f64, f64, i1, i1) -> (index, index, index, i64, i64, f64, f64, i1, i1) {
// CHECK:             %[[VAL_79:.*]] = cmpi ult, %[[VAL_70]], %[[VAL_11]] : index
// CHECK:             %[[VAL_80:.*]] = cmpi ult, %[[VAL_71]], %[[VAL_14]] : index
// CHECK:             %[[VAL_81:.*]] = and %[[VAL_79]], %[[VAL_80]] : i1
// CHECK:             scf.condition(%[[VAL_81]]) %[[VAL_70]], %[[VAL_71]], %[[VAL_72]], %[[VAL_73]], %[[VAL_74]], %[[VAL_75]], %[[VAL_76]], %[[VAL_77]], %[[VAL_78]] : index, index, index, i64, i64, f64, f64, i1, i1
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_82:.*]]: index, %[[VAL_83:.*]]: index, %[[VAL_84:.*]]: index, %[[VAL_85:.*]]: i64, %[[VAL_86:.*]]: i64, %[[VAL_87:.*]]: f64, %[[VAL_88:.*]]: f64, %[[VAL_89:.*]]: i1, %[[VAL_90:.*]]: i1):
// CHECK:             %[[VAL_91:.*]]:2 = scf.if %[[VAL_89]] -> (i64, f64) {
// CHECK:               %[[VAL_92:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_82]]] : memref<?xi64>
// CHECK:               %[[VAL_93:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_82]]] : memref<?xf64>
// CHECK:               scf.yield %[[VAL_92]], %[[VAL_93]] : i64, f64
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_85]], %[[VAL_87]] : i64, f64
// CHECK:             }
// CHECK:             %[[VAL_94:.*]]:2 = scf.if %[[VAL_90]] -> (i64, f64) {
// CHECK:               %[[VAL_95:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_83]]] : memref<?xi64>
// CHECK:               %[[VAL_96:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_83]]] : memref<?xf64>
// CHECK:               scf.yield %[[VAL_95]], %[[VAL_96]] : i64, f64
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_86]], %[[VAL_88]] : i64, f64
// CHECK:             }
// CHECK:             %[[VAL_97:.*]] = cmpi ult, %[[VAL_98:.*]]#0, %[[VAL_99:.*]]#0 : i64
// CHECK:             %[[VAL_100:.*]] = cmpi ugt, %[[VAL_98]]#0, %[[VAL_99]]#0 : i64
// CHECK:             %[[VAL_101:.*]] = addi %[[VAL_82]], %[[VAL_3]] : index
// CHECK:             %[[VAL_102:.*]] = addi %[[VAL_83]], %[[VAL_3]] : index
// CHECK:             %[[VAL_103:.*]] = addi %[[VAL_84]], %[[VAL_3]] : index
// CHECK:             %[[VAL_104:.*]]:5 = scf.if %[[VAL_97]] -> (index, index, index, i1, i1) {
// CHECK:               memref.store %[[VAL_98]]#0, %[[VAL_67]]{{\[}}%[[VAL_84]]] : memref<?xi64>
// CHECK:               memref.store %[[VAL_98]]#1, %[[VAL_68]]{{\[}}%[[VAL_84]]] : memref<?xf64>
// CHECK:               scf.yield %[[VAL_101]], %[[VAL_83]], %[[VAL_103]], %[[VAL_6]], %[[VAL_5]] : index, index, index, i1, i1
// CHECK:             } else {
// CHECK:               %[[VAL_105:.*]]:5 = scf.if %[[VAL_100]] -> (index, index, index, i1, i1) {
// CHECK:                 memref.store %[[VAL_99]]#0, %[[VAL_67]]{{\[}}%[[VAL_84]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_99]]#1, %[[VAL_68]]{{\[}}%[[VAL_84]]] : memref<?xf64>
// CHECK:                 scf.yield %[[VAL_82]], %[[VAL_102]], %[[VAL_103]], %[[VAL_5]], %[[VAL_6]] : index, index, index, i1, i1
// CHECK:               } else {
// CHECK:                 memref.store %[[VAL_98]]#0, %[[VAL_67]]{{\[}}%[[VAL_84]]] : memref<?xi64>
// CHECK:                 %[[VAL_106:.*]] = addf %[[VAL_98]]#1, %[[VAL_99]]#1 : f64
// CHECK:                 memref.store %[[VAL_106]], %[[VAL_68]]{{\[}}%[[VAL_84]]] : memref<?xf64>
// CHECK:                 scf.yield %[[VAL_101]], %[[VAL_102]], %[[VAL_103]], %[[VAL_6]], %[[VAL_6]] : index, index, index, i1, i1
// CHECK:               }
// CHECK:               scf.yield %[[VAL_107:.*]]#0, %[[VAL_107]]#1, %[[VAL_107]]#2, %[[VAL_107]]#3, %[[VAL_107]]#4 : index, index, index, i1, i1
// CHECK:             }
// CHECK:             scf.yield %[[VAL_108:.*]]#0, %[[VAL_108]]#1, %[[VAL_108]]#2, %[[VAL_98]]#0, %[[VAL_99]]#0, %[[VAL_98]]#1, %[[VAL_99]]#1, %[[VAL_108]]#3, %[[VAL_108]]#4 : index, index, index, i64, i64, f64, f64, i1, i1
// CHECK:           }
// CHECK:           %[[VAL_109:.*]] = cmpi ult, %[[VAL_110:.*]]#0, %[[VAL_11]] : index
// CHECK:           %[[VAL_111:.*]] = scf.if %[[VAL_109]] -> (index) {
// CHECK:             %[[VAL_112:.*]] = scf.for %[[VAL_113:.*]] = %[[VAL_110]]#0 to %[[VAL_11]] step %[[VAL_3]] iter_args(%[[VAL_114:.*]] = %[[VAL_110]]#2) -> (index) {
// CHECK:               %[[VAL_115:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_113]]] : memref<?xi64>
// CHECK:               %[[VAL_116:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_113]]] : memref<?xf64>
// CHECK:               memref.store %[[VAL_115]], %[[VAL_67]]{{\[}}%[[VAL_114]]] : memref<?xi64>
// CHECK:               memref.store %[[VAL_116]], %[[VAL_68]]{{\[}}%[[VAL_114]]] : memref<?xf64>
// CHECK:               %[[VAL_117:.*]] = addi %[[VAL_114]], %[[VAL_3]] : index
// CHECK:               scf.yield %[[VAL_117]] : index
// CHECK:             }
// CHECK:             scf.yield %[[VAL_118:.*]] : index
// CHECK:           } else {
// CHECK:             %[[VAL_119:.*]] = cmpi ult, %[[VAL_110]]#1, %[[VAL_14]] : index
// CHECK:             %[[VAL_120:.*]] = scf.if %[[VAL_119]] -> (index) {
// CHECK:               %[[VAL_121:.*]] = scf.for %[[VAL_122:.*]] = %[[VAL_110]]#1 to %[[VAL_14]] step %[[VAL_3]] iter_args(%[[VAL_123:.*]] = %[[VAL_110]]#2) -> (index) {
// CHECK:                 %[[VAL_124:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_122]]] : memref<?xi64>
// CHECK:                 %[[VAL_125:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_122]]] : memref<?xf64>
// CHECK:                 memref.store %[[VAL_124]], %[[VAL_67]]{{\[}}%[[VAL_123]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_125]], %[[VAL_68]]{{\[}}%[[VAL_123]]] : memref<?xf64>
// CHECK:                 %[[VAL_126:.*]] = addi %[[VAL_123]], %[[VAL_3]] : index
// CHECK:                 scf.yield %[[VAL_126]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_127:.*]] : index
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_110]]#2 : index
// CHECK:             }
// CHECK:             scf.yield %[[VAL_128:.*]] : index
// CHECK:           }
// CHECK:           call @delSparseVector(%[[VAL_8]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> ()
// CHECK:           return %[[VAL_129:.*]] : index
// CHECK:         }

func @vector_update_accumulate(%input: tensor<?xf64, #CV64>, %output: tensor<?xf64, #CV64>) -> index {
    %final_position = graphblas.update %input -> %output { accumulate_operator = "plus" } : tensor<?xf64, #CV64> -> tensor<?xf64, #CV64>
    return %final_position : index
}

// TODO: Check all type combinations
