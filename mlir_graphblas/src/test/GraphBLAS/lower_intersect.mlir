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


// CHECK-LABEL:   func @vector_intersect(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                           %[[VAL_1:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
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
// CHECK:               scf.yield %[[VAL_47]], %[[VAL_31]], %[[VAL_6]], %[[VAL_5]], %[[VAL_36]] : index, index, i1, i1, index
// CHECK:             } else {
// CHECK:               %[[VAL_51:.*]]:5 = scf.if %[[VAL_46]] -> (index, index, i1, i1, index) {
// CHECK:                 scf.yield %[[VAL_30]], %[[VAL_48]], %[[VAL_5]], %[[VAL_6]], %[[VAL_36]] : index, index, i1, i1, index
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_47]], %[[VAL_48]], %[[VAL_6]], %[[VAL_6]], %[[VAL_49]] : index, index, i1, i1, index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_52:.*]]#0, %[[VAL_52]]#1, %[[VAL_52]]#2, %[[VAL_52]]#3, %[[VAL_52]]#4 : index, index, i1, i1, index
// CHECK:             }
// CHECK:             scf.yield %[[VAL_53:.*]]#0, %[[VAL_53]]#1, %[[VAL_44]], %[[VAL_45]], %[[VAL_53]]#2, %[[VAL_53]]#3, %[[VAL_53]]#4 : index, index, index, index, i1, i1, index
// CHECK:           }
// CHECK:           %[[VAL_54:.*]] = index_cast %[[VAL_55:.*]]#6 : index to i64
// CHECK:           call @vector_resize_index(%[[VAL_8]], %[[VAL_2]], %[[VAL_55]]#6) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @vector_resize_values(%[[VAL_8]], %[[VAL_55]]#6) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> ()
// CHECK:           %[[VAL_56:.*]] = sparse_tensor.pointers %[[VAL_8]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           memref.store %[[VAL_54]], %[[VAL_56]]{{\[}}%[[VAL_3]]] : memref<?xi64>
// CHECK:           %[[VAL_57:.*]] = sparse_tensor.indices %[[VAL_8]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_58:.*]] = sparse_tensor.values %[[VAL_8]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_59:.*]]:9 = scf.while (%[[VAL_60:.*]] = %[[VAL_2]], %[[VAL_61:.*]] = %[[VAL_2]], %[[VAL_62:.*]] = %[[VAL_2]], %[[VAL_63:.*]] = %[[VAL_4]], %[[VAL_64:.*]] = %[[VAL_4]], %[[VAL_65:.*]] = %[[VAL_7]], %[[VAL_66:.*]] = %[[VAL_7]], %[[VAL_67:.*]] = %[[VAL_6]], %[[VAL_68:.*]] = %[[VAL_6]]) : (index, index, index, i64, i64, f64, f64, i1, i1) -> (index, index, index, i64, i64, f64, f64, i1, i1) {
// CHECK:             %[[VAL_69:.*]] = cmpi ult, %[[VAL_60]], %[[VAL_11]] : index
// CHECK:             %[[VAL_70:.*]] = cmpi ult, %[[VAL_61]], %[[VAL_14]] : index
// CHECK:             %[[VAL_71:.*]] = and %[[VAL_69]], %[[VAL_70]] : i1
// CHECK:             scf.condition(%[[VAL_71]]) %[[VAL_60]], %[[VAL_61]], %[[VAL_62]], %[[VAL_63]], %[[VAL_64]], %[[VAL_65]], %[[VAL_66]], %[[VAL_67]], %[[VAL_68]] : index, index, index, i64, i64, f64, f64, i1, i1
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_72:.*]]: index, %[[VAL_73:.*]]: index, %[[VAL_74:.*]]: index, %[[VAL_75:.*]]: i64, %[[VAL_76:.*]]: i64, %[[VAL_77:.*]]: f64, %[[VAL_78:.*]]: f64, %[[VAL_79:.*]]: i1, %[[VAL_80:.*]]: i1):
// CHECK:             %[[VAL_81:.*]]:2 = scf.if %[[VAL_79]] -> (i64, f64) {
// CHECK:               %[[VAL_82:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_72]]] : memref<?xi64>
// CHECK:               %[[VAL_83:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_72]]] : memref<?xf64>
// CHECK:               scf.yield %[[VAL_82]], %[[VAL_83]] : i64, f64
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_75]], %[[VAL_77]] : i64, f64
// CHECK:             }
// CHECK:             %[[VAL_84:.*]]:2 = scf.if %[[VAL_80]] -> (i64, f64) {
// CHECK:               %[[VAL_85:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_73]]] : memref<?xi64>
// CHECK:               %[[VAL_86:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_73]]] : memref<?xf64>
// CHECK:               scf.yield %[[VAL_85]], %[[VAL_86]] : i64, f64
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_76]], %[[VAL_78]] : i64, f64
// CHECK:             }
// CHECK:             %[[VAL_87:.*]] = cmpi ult, %[[VAL_88:.*]]#0, %[[VAL_89:.*]]#0 : i64
// CHECK:             %[[VAL_90:.*]] = cmpi ugt, %[[VAL_88]]#0, %[[VAL_89]]#0 : i64
// CHECK:             %[[VAL_91:.*]] = addi %[[VAL_72]], %[[VAL_3]] : index
// CHECK:             %[[VAL_92:.*]] = addi %[[VAL_73]], %[[VAL_3]] : index
// CHECK:             %[[VAL_93:.*]] = addi %[[VAL_74]], %[[VAL_3]] : index
// CHECK:             %[[VAL_94:.*]]:5 = scf.if %[[VAL_87]] -> (index, index, index, i1, i1) {
// CHECK:               scf.yield %[[VAL_91]], %[[VAL_73]], %[[VAL_74]], %[[VAL_6]], %[[VAL_5]] : index, index, index, i1, i1
// CHECK:             } else {
// CHECK:               %[[VAL_95:.*]]:5 = scf.if %[[VAL_90]] -> (index, index, index, i1, i1) {
// CHECK:                 scf.yield %[[VAL_72]], %[[VAL_92]], %[[VAL_74]], %[[VAL_5]], %[[VAL_6]] : index, index, index, i1, i1
// CHECK:               } else {
// CHECK:                 memref.store %[[VAL_88]]#0, %[[VAL_57]]{{\[}}%[[VAL_74]]] : memref<?xi64>
// CHECK:                 %[[VAL_96:.*]] = mulf %[[VAL_88]]#1, %[[VAL_89]]#1 : f64
// CHECK:                 memref.store %[[VAL_96]], %[[VAL_58]]{{\[}}%[[VAL_74]]] : memref<?xf64>
// CHECK:                 scf.yield %[[VAL_91]], %[[VAL_92]], %[[VAL_93]], %[[VAL_6]], %[[VAL_6]] : index, index, index, i1, i1
// CHECK:               }
// CHECK:               scf.yield %[[VAL_97:.*]]#0, %[[VAL_97]]#1, %[[VAL_97]]#2, %[[VAL_97]]#3, %[[VAL_97]]#4 : index, index, index, i1, i1
// CHECK:             }
// CHECK:             scf.yield %[[VAL_98:.*]]#0, %[[VAL_98]]#1, %[[VAL_98]]#2, %[[VAL_88]]#0, %[[VAL_89]]#0, %[[VAL_88]]#1, %[[VAL_89]]#1, %[[VAL_98]]#3, %[[VAL_98]]#4 : index, index, index, i64, i64, f64, f64, i1, i1
// CHECK:           }
// CHECK:           return %[[VAL_8]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

func @vector_intersect(%a: tensor<?xf64, #CV64>, %b: tensor<?xf64, #CV64>) -> tensor<?xf64, #CV64> {
    %result = graphblas.intersect %a, %b { intersect_operator = "times" } : (tensor<?xf64, #CV64>, tensor<?xf64, #CV64>) to tensor<?xf64, #CV64>
    return %result : tensor<?xf64, #CV64>
}


// CHECK-LABEL:   func @matrix_intersect(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                           %[[VAL_1:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK-DAG:       %[[VAL_2:.*]] = constant 0 : index
// CHECK-DAG:       %[[VAL_3:.*]] = constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = constant 0 : i64
// CHECK-DAG:       %[[VAL_5:.*]] = constant false
// CHECK-DAG:       %[[VAL_6:.*]] = constant true
// CHECK-DAG:       %[[VAL_7:.*]] = constant 0.000000e+00 : f64
// CHECK:           %[[VAL_8:.*]] = call @cast_csr_to_csx(%[[VAL_0]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_9:.*]] = call @matrix_empty_like(%[[VAL_8]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_10:.*]] = tensor.dim %[[VAL_9]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
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
// CHECK:             %[[VAL_26:.*]] = or %[[VAL_24]], %[[VAL_25]] : i1
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
// CHECK:                   scf.yield %[[VAL_60]], %[[VAL_44]], %[[VAL_6]], %[[VAL_5]], %[[VAL_49]] : index, index, i1, i1, index
// CHECK:                 } else {
// CHECK:                   %[[VAL_64:.*]]:5 = scf.if %[[VAL_59]] -> (index, index, i1, i1, index) {
// CHECK:                     scf.yield %[[VAL_43]], %[[VAL_61]], %[[VAL_5]], %[[VAL_6]], %[[VAL_49]] : index, index, i1, i1, index
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_60]], %[[VAL_61]], %[[VAL_6]], %[[VAL_6]], %[[VAL_62]] : index, index, i1, i1, index
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_65:.*]]#0, %[[VAL_65]]#1, %[[VAL_65]]#2, %[[VAL_65]]#3, %[[VAL_65]]#4 : index, index, i1, i1, index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_66:.*]]#0, %[[VAL_66]]#1, %[[VAL_57]], %[[VAL_58]], %[[VAL_66]]#2, %[[VAL_66]]#3, %[[VAL_66]]#4 : index, index, index, index, i1, i1, index
// CHECK:               }
// CHECK:               %[[VAL_67:.*]] = index_cast %[[VAL_68:.*]]#6 : index to i64
// CHECK:               scf.yield %[[VAL_67]] : i64
// CHECK:             }
// CHECK:             memref.store %[[VAL_69:.*]], %[[VAL_17]]{{\[}}%[[VAL_18]]] : memref<?xi64>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           memref.store %[[VAL_4]], %[[VAL_17]]{{\[}}%[[VAL_10]]] : memref<?xi64>
// CHECK:           scf.for %[[VAL_70:.*]] = %[[VAL_2]] to %[[VAL_10]] step %[[VAL_3]] {
// CHECK:             %[[VAL_71:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_70]]] : memref<?xi64>
// CHECK:             %[[VAL_72:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_10]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_72]], %[[VAL_17]]{{\[}}%[[VAL_70]]] : memref<?xi64>
// CHECK:             %[[VAL_73:.*]] = addi %[[VAL_72]], %[[VAL_71]] : i64
// CHECK:             memref.store %[[VAL_73]], %[[VAL_17]]{{\[}}%[[VAL_10]]] : memref<?xi64>
// CHECK:           }
// CHECK:           %[[VAL_74:.*]] = sparse_tensor.pointers %[[VAL_9]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_75:.*]] = tensor.dim %[[VAL_9]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_76:.*]] = memref.load %[[VAL_74]]{{\[}}%[[VAL_75]]] : memref<?xi64>
// CHECK:           %[[VAL_77:.*]] = index_cast %[[VAL_76]] : i64 to index
// CHECK:           call @matrix_resize_index(%[[VAL_9]], %[[VAL_3]], %[[VAL_77]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @matrix_resize_values(%[[VAL_9]], %[[VAL_77]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> ()
// CHECK:           %[[VAL_78:.*]] = sparse_tensor.indices %[[VAL_9]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_79:.*]] = sparse_tensor.values %[[VAL_9]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           scf.parallel (%[[VAL_80:.*]]) = (%[[VAL_2]]) to (%[[VAL_10]]) step (%[[VAL_3]]) {
// CHECK:             %[[VAL_81:.*]] = addi %[[VAL_80]], %[[VAL_3]] : index
// CHECK:             %[[VAL_82:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_80]]] : memref<?xi64>
// CHECK:             %[[VAL_83:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_81]]] : memref<?xi64>
// CHECK:             %[[VAL_84:.*]] = cmpi ne, %[[VAL_82]], %[[VAL_83]] : i64
// CHECK:             scf.if %[[VAL_84]] {
// CHECK:               %[[VAL_85:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_80]]] : memref<?xi64>
// CHECK:               %[[VAL_86:.*]] = index_cast %[[VAL_85]] : i64 to index
// CHECK:               %[[VAL_87:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_80]]] : memref<?xi64>
// CHECK:               %[[VAL_88:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_81]]] : memref<?xi64>
// CHECK:               %[[VAL_89:.*]] = index_cast %[[VAL_87]] : i64 to index
// CHECK:               %[[VAL_90:.*]] = index_cast %[[VAL_88]] : i64 to index
// CHECK:               %[[VAL_91:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_80]]] : memref<?xi64>
// CHECK:               %[[VAL_92:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_81]]] : memref<?xi64>
// CHECK:               %[[VAL_93:.*]] = index_cast %[[VAL_91]] : i64 to index
// CHECK:               %[[VAL_94:.*]] = index_cast %[[VAL_92]] : i64 to index
// CHECK:               %[[VAL_95:.*]]:9 = scf.while (%[[VAL_96:.*]] = %[[VAL_89]], %[[VAL_97:.*]] = %[[VAL_93]], %[[VAL_98:.*]] = %[[VAL_86]], %[[VAL_99:.*]] = %[[VAL_4]], %[[VAL_100:.*]] = %[[VAL_4]], %[[VAL_101:.*]] = %[[VAL_7]], %[[VAL_102:.*]] = %[[VAL_7]], %[[VAL_103:.*]] = %[[VAL_6]], %[[VAL_104:.*]] = %[[VAL_6]]) : (index, index, index, i64, i64, f64, f64, i1, i1) -> (index, index, index, i64, i64, f64, f64, i1, i1) {
// CHECK:                 %[[VAL_105:.*]] = cmpi ult, %[[VAL_96]], %[[VAL_90]] : index
// CHECK:                 %[[VAL_106:.*]] = cmpi ult, %[[VAL_97]], %[[VAL_94]] : index
// CHECK:                 %[[VAL_107:.*]] = and %[[VAL_105]], %[[VAL_106]] : i1
// CHECK:                 scf.condition(%[[VAL_107]]) %[[VAL_96]], %[[VAL_97]], %[[VAL_98]], %[[VAL_99]], %[[VAL_100]], %[[VAL_101]], %[[VAL_102]], %[[VAL_103]], %[[VAL_104]] : index, index, index, i64, i64, f64, f64, i1, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_108:.*]]: index, %[[VAL_109:.*]]: index, %[[VAL_110:.*]]: index, %[[VAL_111:.*]]: i64, %[[VAL_112:.*]]: i64, %[[VAL_113:.*]]: f64, %[[VAL_114:.*]]: f64, %[[VAL_115:.*]]: i1, %[[VAL_116:.*]]: i1):
// CHECK:                 %[[VAL_117:.*]]:2 = scf.if %[[VAL_115]] -> (i64, f64) {
// CHECK:                   %[[VAL_118:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_108]]] : memref<?xi64>
// CHECK:                   %[[VAL_119:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_108]]] : memref<?xf64>
// CHECK:                   scf.yield %[[VAL_118]], %[[VAL_119]] : i64, f64
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_111]], %[[VAL_113]] : i64, f64
// CHECK:                 }
// CHECK:                 %[[VAL_120:.*]]:2 = scf.if %[[VAL_116]] -> (i64, f64) {
// CHECK:                   %[[VAL_121:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_109]]] : memref<?xi64>
// CHECK:                   %[[VAL_122:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_109]]] : memref<?xf64>
// CHECK:                   scf.yield %[[VAL_121]], %[[VAL_122]] : i64, f64
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_112]], %[[VAL_114]] : i64, f64
// CHECK:                 }
// CHECK:                 %[[VAL_123:.*]] = cmpi ult, %[[VAL_124:.*]]#0, %[[VAL_125:.*]]#0 : i64
// CHECK:                 %[[VAL_126:.*]] = cmpi ugt, %[[VAL_124]]#0, %[[VAL_125]]#0 : i64
// CHECK:                 %[[VAL_127:.*]] = addi %[[VAL_108]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_128:.*]] = addi %[[VAL_109]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_129:.*]] = addi %[[VAL_110]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_130:.*]]:5 = scf.if %[[VAL_123]] -> (index, index, index, i1, i1) {
// CHECK:                   scf.yield %[[VAL_127]], %[[VAL_109]], %[[VAL_110]], %[[VAL_6]], %[[VAL_5]] : index, index, index, i1, i1
// CHECK:                 } else {
// CHECK:                   %[[VAL_131:.*]]:5 = scf.if %[[VAL_126]] -> (index, index, index, i1, i1) {
// CHECK:                     scf.yield %[[VAL_108]], %[[VAL_128]], %[[VAL_110]], %[[VAL_5]], %[[VAL_6]] : index, index, index, i1, i1
// CHECK:                   } else {
// CHECK:                     memref.store %[[VAL_124]]#0, %[[VAL_78]]{{\[}}%[[VAL_110]]] : memref<?xi64>
// CHECK:                     %[[VAL_132:.*]] = mulf %[[VAL_124]]#1, %[[VAL_125]]#1 : f64
// CHECK:                     memref.store %[[VAL_132]], %[[VAL_79]]{{\[}}%[[VAL_110]]] : memref<?xf64>
// CHECK:                     scf.yield %[[VAL_127]], %[[VAL_128]], %[[VAL_129]], %[[VAL_6]], %[[VAL_6]] : index, index, index, i1, i1
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_133:.*]]#0, %[[VAL_133]]#1, %[[VAL_133]]#2, %[[VAL_133]]#3, %[[VAL_133]]#4 : index, index, index, i1, i1
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_134:.*]]#0, %[[VAL_134]]#1, %[[VAL_134]]#2, %[[VAL_124]]#0, %[[VAL_125]]#0, %[[VAL_124]]#1, %[[VAL_125]]#1, %[[VAL_134]]#3, %[[VAL_134]]#4 : index, index, index, i64, i64, f64, f64, i1, i1
// CHECK:               }
// CHECK:             }
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           %[[VAL_135:.*]] = call @cast_csx_to_csr(%[[VAL_9]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           return %[[VAL_135]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

func @matrix_intersect(%a: tensor<?x?xf64, #CSR64>, %b: tensor<?x?xf64, #CSR64>) -> tensor<?x?xf64, #CSR64> {
    %result = graphblas.intersect %a, %b { intersect_operator = "times" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64>) to tensor<?x?xf64, #CSR64>
    return %result : tensor<?x?xf64, #CSR64>
}
