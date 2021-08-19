// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>


// CHECK-LABEL:   func @vector_union(
// CHECK-SAME:                       %[[VAL_0:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                       %[[VAL_1:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK-DAG:       %[[VAL_2:.*]] = constant 0 : index
// CHECK-DAG:       %[[VAL_3:.*]] = constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = constant 0 : i64
// CHECK-DAG:       %[[VAL_5:.*]] = constant false
// CHECK-DAG:       %[[VAL_6:.*]] = constant true
// CHECK-DAG:       %[[VAL_7:.*]] = constant 0.000000e+00 : f64
// CHECK:           %[[VAL_8:.*]] = call @vector_empty_like(%[[VAL_0]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_9:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_10:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_3]]] : memref<?xi64>
// CHECK:           %[[VAL_11:.*]] = index_cast %[[VAL_10]] : i64 to index
// CHECK:           %[[VAL_12:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_13:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_3]]] : memref<?xi64>
// CHECK:           %[[VAL_14:.*]] = index_cast %[[VAL_13]] : i64 to index
// CHECK:           %[[VAL_15:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_16:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_17:.*]] = sparse_tensor.indices %[[VAL_1]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_18:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
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
// CHECK:           call @vector_resize_index(%[[VAL_8]], %[[VAL_2]], %[[VAL_65]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @vector_resize_values(%[[VAL_8]], %[[VAL_65]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> ()
// CHECK:           %[[VAL_66:.*]] = sparse_tensor.pointers %[[VAL_8]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           memref.store %[[VAL_64]], %[[VAL_66]]{{\[}}%[[VAL_3]]] : memref<?xi64>
// CHECK:           %[[VAL_67:.*]] = sparse_tensor.indices %[[VAL_8]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_68:.*]] = sparse_tensor.values %[[VAL_8]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
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
// CHECK:           return %[[VAL_8]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

func @vector_union(%a: tensor<?xf64, #CV64>, %b: tensor<?xf64, #CV64>) -> tensor<?xf64, #CV64> {
    %result = graphblas.union %a, %b { union_operator = "plus" } : (tensor<?xf64, #CV64>, tensor<?xf64, #CV64>) to tensor<?xf64, #CV64>
    return %result : tensor<?xf64, #CV64>
}


// CHECK-LABEL:   func @matrix_union(
// CHECK-SAME:                       %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                       %[[VAL_1:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK-DAG:       %[[VAL_2:.*]] = constant 0 : index
// CHECK-DAG:       %[[VAL_3:.*]] = constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = constant 0 : i64
// CHECK-DAG:       %[[VAL_5:.*]] = constant false
// CHECK-DAG:       %[[VAL_6:.*]] = constant true
// CHECK-DAG:       %[[VAL_7:.*]] = constant 0.000000e+00 : f64
// CHECK:           %[[VAL_8:.*]] = call @cast_csr_to_csx(%[[VAL_0]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_9:.*]] = call @matrix_empty_like(%[[VAL_8]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_10:.*]] = memref.dim %[[VAL_9]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_11:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_12:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_13:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_14:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_15:.*]] = sparse_tensor.indices %[[VAL_1]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_16:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_17:.*]] = sparse_tensor.pointers %[[VAL_9]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           scf.parallel (%[[VAL_18:.*]]) = (%[[VAL_2]]) to (%[[VAL_10]]) step (%[[VAL_3]]) {
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
// CHECK:             memref.store %[[VAL_79:.*]], %[[VAL_17]]{{\[}}%[[VAL_18]]] : memref<?xi64>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           memref.store %[[VAL_4]], %[[VAL_17]]{{\[}}%[[VAL_10]]] : memref<?xi64>
// CHECK:           scf.for %[[VAL_80:.*]] = %[[VAL_2]] to %[[VAL_10]] step %[[VAL_3]] {
// CHECK:             %[[VAL_81:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_80]]] : memref<?xi64>
// CHECK:             %[[VAL_82:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_10]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_82]], %[[VAL_17]]{{\[}}%[[VAL_80]]] : memref<?xi64>
// CHECK:             %[[VAL_83:.*]] = addi %[[VAL_82]], %[[VAL_81]] : i64
// CHECK:             memref.store %[[VAL_83]], %[[VAL_17]]{{\[}}%[[VAL_10]]] : memref<?xi64>
// CHECK:           }
// CHECK:           %[[VAL_84:.*]] = sparse_tensor.pointers %[[VAL_9]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_85:.*]] = memref.dim %[[VAL_9]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_86:.*]] = memref.load %[[VAL_84]]{{\[}}%[[VAL_85]]] : memref<?xi64>
// CHECK:           %[[VAL_87:.*]] = index_cast %[[VAL_86]] : i64 to index
// CHECK:           call @matrix_resize_index(%[[VAL_9]], %[[VAL_3]], %[[VAL_87]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @matrix_resize_values(%[[VAL_9]], %[[VAL_87]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> ()
// CHECK:           %[[VAL_88:.*]] = sparse_tensor.indices %[[VAL_9]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_89:.*]] = sparse_tensor.values %[[VAL_9]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           scf.parallel (%[[VAL_90:.*]]) = (%[[VAL_2]]) to (%[[VAL_10]]) step (%[[VAL_3]]) {
// CHECK:             %[[VAL_91:.*]] = addi %[[VAL_90]], %[[VAL_3]] : index
// CHECK:             %[[VAL_92:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_90]]] : memref<?xi64>
// CHECK:             %[[VAL_93:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_91]]] : memref<?xi64>
// CHECK:             %[[VAL_94:.*]] = cmpi ne, %[[VAL_92]], %[[VAL_93]] : i64
// CHECK:             scf.if %[[VAL_94]] {
// CHECK:               %[[VAL_95:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_90]]] : memref<?xi64>
// CHECK:               %[[VAL_96:.*]] = index_cast %[[VAL_95]] : i64 to index
// CHECK:               %[[VAL_97:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_90]]] : memref<?xi64>
// CHECK:               %[[VAL_98:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_91]]] : memref<?xi64>
// CHECK:               %[[VAL_99:.*]] = index_cast %[[VAL_97]] : i64 to index
// CHECK:               %[[VAL_100:.*]] = index_cast %[[VAL_98]] : i64 to index
// CHECK:               %[[VAL_101:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_90]]] : memref<?xi64>
// CHECK:               %[[VAL_102:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_91]]] : memref<?xi64>
// CHECK:               %[[VAL_103:.*]] = index_cast %[[VAL_101]] : i64 to index
// CHECK:               %[[VAL_104:.*]] = index_cast %[[VAL_102]] : i64 to index
// CHECK:               %[[VAL_105:.*]]:9 = scf.while (%[[VAL_106:.*]] = %[[VAL_99]], %[[VAL_107:.*]] = %[[VAL_103]], %[[VAL_108:.*]] = %[[VAL_96]], %[[VAL_109:.*]] = %[[VAL_4]], %[[VAL_110:.*]] = %[[VAL_4]], %[[VAL_111:.*]] = %[[VAL_7]], %[[VAL_112:.*]] = %[[VAL_7]], %[[VAL_113:.*]] = %[[VAL_6]], %[[VAL_114:.*]] = %[[VAL_6]]) : (index, index, index, i64, i64, f64, f64, i1, i1) -> (index, index, index, i64, i64, f64, f64, i1, i1) {
// CHECK:                 %[[VAL_115:.*]] = cmpi ult, %[[VAL_106]], %[[VAL_100]] : index
// CHECK:                 %[[VAL_116:.*]] = cmpi ult, %[[VAL_107]], %[[VAL_104]] : index
// CHECK:                 %[[VAL_117:.*]] = and %[[VAL_115]], %[[VAL_116]] : i1
// CHECK:                 scf.condition(%[[VAL_117]]) %[[VAL_106]], %[[VAL_107]], %[[VAL_108]], %[[VAL_109]], %[[VAL_110]], %[[VAL_111]], %[[VAL_112]], %[[VAL_113]], %[[VAL_114]] : index, index, index, i64, i64, f64, f64, i1, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_118:.*]]: index, %[[VAL_119:.*]]: index, %[[VAL_120:.*]]: index, %[[VAL_121:.*]]: i64, %[[VAL_122:.*]]: i64, %[[VAL_123:.*]]: f64, %[[VAL_124:.*]]: f64, %[[VAL_125:.*]]: i1, %[[VAL_126:.*]]: i1):
// CHECK:                 %[[VAL_127:.*]]:2 = scf.if %[[VAL_125]] -> (i64, f64) {
// CHECK:                   %[[VAL_128:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_118]]] : memref<?xi64>
// CHECK:                   %[[VAL_129:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_118]]] : memref<?xf64>
// CHECK:                   scf.yield %[[VAL_128]], %[[VAL_129]] : i64, f64
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_121]], %[[VAL_123]] : i64, f64
// CHECK:                 }
// CHECK:                 %[[VAL_130:.*]]:2 = scf.if %[[VAL_126]] -> (i64, f64) {
// CHECK:                   %[[VAL_131:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_119]]] : memref<?xi64>
// CHECK:                   %[[VAL_132:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_119]]] : memref<?xf64>
// CHECK:                   scf.yield %[[VAL_131]], %[[VAL_132]] : i64, f64
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_122]], %[[VAL_124]] : i64, f64
// CHECK:                 }
// CHECK:                 %[[VAL_133:.*]] = cmpi ult, %[[VAL_134:.*]]#0, %[[VAL_135:.*]]#0 : i64
// CHECK:                 %[[VAL_136:.*]] = cmpi ugt, %[[VAL_134]]#0, %[[VAL_135]]#0 : i64
// CHECK:                 %[[VAL_137:.*]] = addi %[[VAL_118]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_138:.*]] = addi %[[VAL_119]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_139:.*]] = addi %[[VAL_120]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_140:.*]]:5 = scf.if %[[VAL_133]] -> (index, index, index, i1, i1) {
// CHECK:                   memref.store %[[VAL_134]]#0, %[[VAL_88]]{{\[}}%[[VAL_120]]] : memref<?xi64>
// CHECK:                   memref.store %[[VAL_134]]#1, %[[VAL_89]]{{\[}}%[[VAL_120]]] : memref<?xf64>
// CHECK:                   scf.yield %[[VAL_137]], %[[VAL_119]], %[[VAL_139]], %[[VAL_6]], %[[VAL_5]] : index, index, index, i1, i1
// CHECK:                 } else {
// CHECK:                   %[[VAL_141:.*]]:5 = scf.if %[[VAL_136]] -> (index, index, index, i1, i1) {
// CHECK:                     memref.store %[[VAL_135]]#0, %[[VAL_88]]{{\[}}%[[VAL_120]]] : memref<?xi64>
// CHECK:                     memref.store %[[VAL_135]]#1, %[[VAL_89]]{{\[}}%[[VAL_120]]] : memref<?xf64>
// CHECK:                     scf.yield %[[VAL_118]], %[[VAL_138]], %[[VAL_139]], %[[VAL_5]], %[[VAL_6]] : index, index, index, i1, i1
// CHECK:                   } else {
// CHECK:                     memref.store %[[VAL_134]]#0, %[[VAL_88]]{{\[}}%[[VAL_120]]] : memref<?xi64>
// CHECK:                     %[[VAL_142:.*]] = addf %[[VAL_134]]#1, %[[VAL_135]]#1 : f64
// CHECK:                     memref.store %[[VAL_142]], %[[VAL_89]]{{\[}}%[[VAL_120]]] : memref<?xf64>
// CHECK:                     scf.yield %[[VAL_137]], %[[VAL_138]], %[[VAL_139]], %[[VAL_6]], %[[VAL_6]] : index, index, index, i1, i1
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_143:.*]]#0, %[[VAL_143]]#1, %[[VAL_143]]#2, %[[VAL_143]]#3, %[[VAL_143]]#4 : index, index, index, i1, i1
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_144:.*]]#0, %[[VAL_144]]#1, %[[VAL_144]]#2, %[[VAL_134]]#0, %[[VAL_135]]#0, %[[VAL_134]]#1, %[[VAL_135]]#1, %[[VAL_144]]#3, %[[VAL_144]]#4 : index, index, index, i64, i64, f64, f64, i1, i1
// CHECK:               }
// CHECK:               %[[VAL_145:.*]] = cmpi ult, %[[VAL_146:.*]]#0, %[[VAL_100]] : index
// CHECK:               %[[VAL_147:.*]] = scf.if %[[VAL_145]] -> (index) {
// CHECK:                 %[[VAL_148:.*]] = scf.for %[[VAL_149:.*]] = %[[VAL_146]]#0 to %[[VAL_100]] step %[[VAL_3]] iter_args(%[[VAL_150:.*]] = %[[VAL_146]]#2) -> (index) {
// CHECK:                   %[[VAL_151:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_149]]] : memref<?xi64>
// CHECK:                   %[[VAL_152:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_149]]] : memref<?xf64>
// CHECK:                   memref.store %[[VAL_151]], %[[VAL_88]]{{\[}}%[[VAL_150]]] : memref<?xi64>
// CHECK:                   memref.store %[[VAL_152]], %[[VAL_89]]{{\[}}%[[VAL_150]]] : memref<?xf64>
// CHECK:                   %[[VAL_153:.*]] = addi %[[VAL_150]], %[[VAL_3]] : index
// CHECK:                   scf.yield %[[VAL_153]] : index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_154:.*]] : index
// CHECK:               } else {
// CHECK:                 %[[VAL_155:.*]] = cmpi ult, %[[VAL_146]]#1, %[[VAL_104]] : index
// CHECK:                 %[[VAL_156:.*]] = scf.if %[[VAL_155]] -> (index) {
// CHECK:                   %[[VAL_157:.*]] = scf.for %[[VAL_158:.*]] = %[[VAL_146]]#1 to %[[VAL_104]] step %[[VAL_3]] iter_args(%[[VAL_159:.*]] = %[[VAL_146]]#2) -> (index) {
// CHECK:                     %[[VAL_160:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_158]]] : memref<?xi64>
// CHECK:                     %[[VAL_161:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_158]]] : memref<?xf64>
// CHECK:                     memref.store %[[VAL_160]], %[[VAL_88]]{{\[}}%[[VAL_159]]] : memref<?xi64>
// CHECK:                     memref.store %[[VAL_161]], %[[VAL_89]]{{\[}}%[[VAL_159]]] : memref<?xf64>
// CHECK:                     %[[VAL_162:.*]] = addi %[[VAL_159]], %[[VAL_3]] : index
// CHECK:                     scf.yield %[[VAL_162]] : index
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_163:.*]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_146]]#2 : index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_164:.*]] : index
// CHECK:               }
// CHECK:             }
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           %[[VAL_165:.*]] = call @cast_csx_to_csr(%[[VAL_9]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           return %[[VAL_165]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

func @matrix_union(%a: tensor<?x?xf64, #CSR64>, %b: tensor<?x?xf64, #CSR64>) -> tensor<?x?xf64, #CSR64> {
    %result = graphblas.union %a, %b { union_operator = "plus" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64>) to tensor<?x?xf64, #CSR64>
    return %result : tensor<?x?xf64, #CSR64>
}

// TODO: Check all type combinations
