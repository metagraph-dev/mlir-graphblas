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
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant false
// CHECK:           %[[VAL_6:.*]] = arith.constant true
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_8:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[VAL_9:.*]] = tensor.dim %[[VAL_0]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_10:.*]] = sparse_tensor.init{{\[}}%[[VAL_9]]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_11:.*]] = call @vector_f64_p64i64_to_ptr8(%[[VAL_10]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_pointers(%[[VAL_11]], %[[VAL_2]], %[[VAL_3]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_12:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_13:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_4]]] : memref<?xi64>
// CHECK:           %[[VAL_14:.*]] = arith.index_cast %[[VAL_13]] : i64 to index
// CHECK:           %[[VAL_15:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_16:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_4]]] : memref<?xi64>
// CHECK:           %[[VAL_17:.*]] = arith.index_cast %[[VAL_16]] : i64 to index
// CHECK:           %[[VAL_18:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_19:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_20:.*]] = sparse_tensor.indices %[[VAL_1]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_21:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_22:.*]]:7 = scf.while (%[[VAL_23:.*]] = %[[VAL_2]], %[[VAL_24:.*]] = %[[VAL_2]], %[[VAL_25:.*]] = %[[VAL_2]], %[[VAL_26:.*]] = %[[VAL_2]], %[[VAL_27:.*]] = %[[VAL_6]], %[[VAL_28:.*]] = %[[VAL_6]], %[[VAL_29:.*]] = %[[VAL_2]]) : (index, index, index, index, i1, i1, index) -> (index, index, index, index, i1, i1, index) {
// CHECK:             %[[VAL_30:.*]] = arith.cmpi ult, %[[VAL_23]], %[[VAL_14]] : index
// CHECK:             %[[VAL_31:.*]] = arith.cmpi ult, %[[VAL_24]], %[[VAL_17]] : index
// CHECK:             %[[VAL_32:.*]] = arith.andi %[[VAL_30]], %[[VAL_31]] : i1
// CHECK:             scf.condition(%[[VAL_32]]) %[[VAL_23]], %[[VAL_24]], %[[VAL_25]], %[[VAL_26]], %[[VAL_27]], %[[VAL_28]], %[[VAL_29]] : index, index, index, index, i1, i1, index
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_33:.*]]: index, %[[VAL_34:.*]]: index, %[[VAL_35:.*]]: index, %[[VAL_36:.*]]: index, %[[VAL_37:.*]]: i1, %[[VAL_38:.*]]: i1, %[[VAL_39:.*]]: index):
// CHECK:             %[[VAL_40:.*]] = scf.if %[[VAL_37]] -> (index) {
// CHECK:               %[[VAL_41:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_33]]] : memref<?xi64>
// CHECK:               %[[VAL_42:.*]] = arith.index_cast %[[VAL_41]] : i64 to index
// CHECK:               scf.yield %[[VAL_42]] : index
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_35]] : index
// CHECK:             }
// CHECK:             %[[VAL_43:.*]] = scf.if %[[VAL_38]] -> (index) {
// CHECK:               %[[VAL_44:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_34]]] : memref<?xi64>
// CHECK:               %[[VAL_45:.*]] = arith.index_cast %[[VAL_44]] : i64 to index
// CHECK:               scf.yield %[[VAL_45]] : index
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_36]] : index
// CHECK:             }
// CHECK:             %[[VAL_46:.*]] = arith.cmpi ult, %[[VAL_47:.*]], %[[VAL_48:.*]] : index
// CHECK:             %[[VAL_49:.*]] = arith.cmpi ugt, %[[VAL_47]], %[[VAL_48]] : index
// CHECK:             %[[VAL_50:.*]] = arith.addi %[[VAL_33]], %[[VAL_4]] : index
// CHECK:             %[[VAL_51:.*]] = arith.addi %[[VAL_34]], %[[VAL_4]] : index
// CHECK:             %[[VAL_52:.*]] = arith.addi %[[VAL_39]], %[[VAL_4]] : index
// CHECK:             %[[VAL_53:.*]]:5 = scf.if %[[VAL_46]] -> (index, index, i1, i1, index) {
// CHECK:               scf.yield %[[VAL_50]], %[[VAL_34]], %[[VAL_6]], %[[VAL_5]], %[[VAL_39]] : index, index, i1, i1, index
// CHECK:             } else {
// CHECK:               %[[VAL_54:.*]]:5 = scf.if %[[VAL_49]] -> (index, index, i1, i1, index) {
// CHECK:                 scf.yield %[[VAL_33]], %[[VAL_51]], %[[VAL_5]], %[[VAL_6]], %[[VAL_39]] : index, index, i1, i1, index
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_50]], %[[VAL_51]], %[[VAL_6]], %[[VAL_6]], %[[VAL_52]] : index, index, i1, i1, index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_55:.*]]#0, %[[VAL_55]]#1, %[[VAL_55]]#2, %[[VAL_55]]#3, %[[VAL_55]]#4 : index, index, i1, i1, index
// CHECK:             }
// CHECK:             scf.yield %[[VAL_56:.*]]#0, %[[VAL_56]]#1, %[[VAL_47]], %[[VAL_48]], %[[VAL_56]]#2, %[[VAL_56]]#3, %[[VAL_56]]#4 : index, index, index, index, i1, i1, index
// CHECK:           }
// CHECK:           %[[VAL_57:.*]] = arith.index_cast %[[VAL_58:.*]]#6 : index to i64
// CHECK:           %[[VAL_59:.*]] = call @vector_f64_p64i64_to_ptr8(%[[VAL_10]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_index(%[[VAL_59]], %[[VAL_2]], %[[VAL_58]]#6) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_60:.*]] = call @vector_f64_p64i64_to_ptr8(%[[VAL_10]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_values(%[[VAL_60]], %[[VAL_58]]#6) : (!llvm.ptr<i8>, index) -> ()
// CHECK:           %[[VAL_61:.*]] = sparse_tensor.pointers %[[VAL_10]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           memref.store %[[VAL_57]], %[[VAL_61]]{{\[}}%[[VAL_4]]] : memref<?xi64>
// CHECK:           %[[VAL_62:.*]] = sparse_tensor.indices %[[VAL_10]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_63:.*]] = sparse_tensor.values %[[VAL_10]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_64:.*]]:9 = scf.while (%[[VAL_65:.*]] = %[[VAL_2]], %[[VAL_66:.*]] = %[[VAL_2]], %[[VAL_67:.*]] = %[[VAL_2]], %[[VAL_68:.*]] = %[[VAL_7]], %[[VAL_69:.*]] = %[[VAL_7]], %[[VAL_70:.*]] = %[[VAL_8]], %[[VAL_71:.*]] = %[[VAL_8]], %[[VAL_72:.*]] = %[[VAL_6]], %[[VAL_73:.*]] = %[[VAL_6]]) : (index, index, index, i64, i64, f64, f64, i1, i1) -> (index, index, index, i64, i64, f64, f64, i1, i1) {
// CHECK:             %[[VAL_74:.*]] = arith.cmpi ult, %[[VAL_65]], %[[VAL_14]] : index
// CHECK:             %[[VAL_75:.*]] = arith.cmpi ult, %[[VAL_66]], %[[VAL_17]] : index
// CHECK:             %[[VAL_76:.*]] = arith.andi %[[VAL_74]], %[[VAL_75]] : i1
// CHECK:             scf.condition(%[[VAL_76]]) %[[VAL_65]], %[[VAL_66]], %[[VAL_67]], %[[VAL_68]], %[[VAL_69]], %[[VAL_70]], %[[VAL_71]], %[[VAL_72]], %[[VAL_73]] : index, index, index, i64, i64, f64, f64, i1, i1
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_77:.*]]: index, %[[VAL_78:.*]]: index, %[[VAL_79:.*]]: index, %[[VAL_80:.*]]: i64, %[[VAL_81:.*]]: i64, %[[VAL_82:.*]]: f64, %[[VAL_83:.*]]: f64, %[[VAL_84:.*]]: i1, %[[VAL_85:.*]]: i1):
// CHECK:             %[[VAL_86:.*]]:2 = scf.if %[[VAL_84]] -> (i64, f64) {
// CHECK:               %[[VAL_87:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_77]]] : memref<?xi64>
// CHECK:               %[[VAL_88:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_77]]] : memref<?xf64>
// CHECK:               scf.yield %[[VAL_87]], %[[VAL_88]] : i64, f64
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_80]], %[[VAL_82]] : i64, f64
// CHECK:             }
// CHECK:             %[[VAL_89:.*]]:2 = scf.if %[[VAL_85]] -> (i64, f64) {
// CHECK:               %[[VAL_90:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_78]]] : memref<?xi64>
// CHECK:               %[[VAL_91:.*]] = memref.load %[[VAL_21]]{{\[}}%[[VAL_78]]] : memref<?xf64>
// CHECK:               scf.yield %[[VAL_90]], %[[VAL_91]] : i64, f64
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_81]], %[[VAL_83]] : i64, f64
// CHECK:             }
// CHECK:             %[[VAL_92:.*]] = arith.cmpi ult, %[[VAL_93:.*]]#0, %[[VAL_94:.*]]#0 : i64
// CHECK:             %[[VAL_95:.*]] = arith.cmpi ugt, %[[VAL_93]]#0, %[[VAL_94]]#0 : i64
// CHECK:             %[[VAL_96:.*]] = arith.addi %[[VAL_77]], %[[VAL_4]] : index
// CHECK:             %[[VAL_97:.*]] = arith.addi %[[VAL_78]], %[[VAL_4]] : index
// CHECK:             %[[VAL_98:.*]] = arith.addi %[[VAL_79]], %[[VAL_4]] : index
// CHECK:             %[[VAL_99:.*]]:5 = scf.if %[[VAL_92]] -> (index, index, index, i1, i1) {
// CHECK:               scf.yield %[[VAL_96]], %[[VAL_78]], %[[VAL_79]], %[[VAL_6]], %[[VAL_5]] : index, index, index, i1, i1
// CHECK:             } else {
// CHECK:               %[[VAL_100:.*]]:5 = scf.if %[[VAL_95]] -> (index, index, index, i1, i1) {
// CHECK:                 scf.yield %[[VAL_77]], %[[VAL_97]], %[[VAL_79]], %[[VAL_5]], %[[VAL_6]] : index, index, index, i1, i1
// CHECK:               } else {
// CHECK:                 memref.store %[[VAL_93]]#0, %[[VAL_62]]{{\[}}%[[VAL_79]]] : memref<?xi64>
// CHECK:                 %[[VAL_101:.*]] = arith.mulf %[[VAL_93]]#1, %[[VAL_94]]#1 : f64
// CHECK:                 memref.store %[[VAL_101]], %[[VAL_63]]{{\[}}%[[VAL_79]]] : memref<?xf64>
// CHECK:                 scf.yield %[[VAL_96]], %[[VAL_97]], %[[VAL_98]], %[[VAL_6]], %[[VAL_6]] : index, index, index, i1, i1
// CHECK:               }
// CHECK:               scf.yield %[[VAL_102:.*]]#0, %[[VAL_102]]#1, %[[VAL_102]]#2, %[[VAL_102]]#3, %[[VAL_102]]#4 : index, index, index, i1, i1
// CHECK:             }
// CHECK:             scf.yield %[[VAL_103:.*]]#0, %[[VAL_103]]#1, %[[VAL_103]]#2, %[[VAL_93]]#0, %[[VAL_94]]#0, %[[VAL_93]]#1, %[[VAL_94]]#1, %[[VAL_103]]#3, %[[VAL_103]]#4 : index, index, index, i64, i64, f64, f64, i1, i1
// CHECK:           }
// CHECK:           return %[[VAL_10]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

func @vector_intersect(%a: tensor<?xf64, #CV64>, %b: tensor<?xf64, #CV64>) -> tensor<?xf64, #CV64> {
    %result = graphblas.intersect %a, %b { intersect_operator = "times" } : (tensor<?xf64, #CV64>, tensor<?xf64, #CV64>) to tensor<?xf64, #CV64>
    return %result : tensor<?xf64, #CV64>
}


// CHECK-LABEL:   func @matrix_intersect(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                           %[[VAL_1:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_5:.*]] = arith.constant false
// CHECK:           %[[VAL_6:.*]] = arith.constant true
// CHECK:           %[[VAL_7:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[VAL_8:.*]] = tensor.dim %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_9:.*]] = tensor.dim %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_10:.*]] = sparse_tensor.init{{\[}}%[[VAL_8]], %[[VAL_9]]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_11:.*]] = tensor.dim %[[VAL_10]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_12:.*]] = arith.addi %[[VAL_11]], %[[VAL_3]] : index
// CHECK:           %[[VAL_13:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_10]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_pointers(%[[VAL_13]], %[[VAL_3]], %[[VAL_12]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_14:.*]] = tensor.dim %[[VAL_10]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_15:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_16:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_17:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_18:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_19:.*]] = sparse_tensor.indices %[[VAL_1]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_20:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_21:.*]] = sparse_tensor.pointers %[[VAL_10]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           scf.parallel (%[[VAL_22:.*]]) = (%[[VAL_2]]) to (%[[VAL_14]]) step (%[[VAL_3]]) {
// CHECK:             %[[VAL_23:.*]] = arith.addi %[[VAL_22]], %[[VAL_3]] : index
// CHECK:             %[[VAL_24:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_22]]] : memref<?xi64>
// CHECK:             %[[VAL_25:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_23]]] : memref<?xi64>
// CHECK:             %[[VAL_26:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_22]]] : memref<?xi64>
// CHECK:             %[[VAL_27:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_23]]] : memref<?xi64>
// CHECK:             %[[VAL_28:.*]] = arith.cmpi eq, %[[VAL_24]], %[[VAL_25]] : i64
// CHECK:             %[[VAL_29:.*]] = arith.cmpi eq, %[[VAL_26]], %[[VAL_27]] : i64
// CHECK:             %[[VAL_30:.*]] = arith.ori %[[VAL_28]], %[[VAL_29]] : i1
// CHECK:             %[[VAL_31:.*]] = scf.if %[[VAL_30]] -> (i64) {
// CHECK:               scf.yield %[[VAL_4]] : i64
// CHECK:             } else {
// CHECK:               %[[VAL_32:.*]] = arith.index_cast %[[VAL_24]] : i64 to index
// CHECK:               %[[VAL_33:.*]] = arith.index_cast %[[VAL_25]] : i64 to index
// CHECK:               %[[VAL_34:.*]] = arith.index_cast %[[VAL_26]] : i64 to index
// CHECK:               %[[VAL_35:.*]] = arith.index_cast %[[VAL_27]] : i64 to index
// CHECK:               %[[VAL_36:.*]]:7 = scf.while (%[[VAL_37:.*]] = %[[VAL_32]], %[[VAL_38:.*]] = %[[VAL_34]], %[[VAL_39:.*]] = %[[VAL_2]], %[[VAL_40:.*]] = %[[VAL_2]], %[[VAL_41:.*]] = %[[VAL_6]], %[[VAL_42:.*]] = %[[VAL_6]], %[[VAL_43:.*]] = %[[VAL_2]]) : (index, index, index, index, i1, i1, index) -> (index, index, index, index, i1, i1, index) {
// CHECK:                 %[[VAL_44:.*]] = arith.cmpi ult, %[[VAL_37]], %[[VAL_33]] : index
// CHECK:                 %[[VAL_45:.*]] = arith.cmpi ult, %[[VAL_38]], %[[VAL_35]] : index
// CHECK:                 %[[VAL_46:.*]] = arith.andi %[[VAL_44]], %[[VAL_45]] : i1
// CHECK:                 scf.condition(%[[VAL_46]]) %[[VAL_37]], %[[VAL_38]], %[[VAL_39]], %[[VAL_40]], %[[VAL_41]], %[[VAL_42]], %[[VAL_43]] : index, index, index, index, i1, i1, index
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_47:.*]]: index, %[[VAL_48:.*]]: index, %[[VAL_49:.*]]: index, %[[VAL_50:.*]]: index, %[[VAL_51:.*]]: i1, %[[VAL_52:.*]]: i1, %[[VAL_53:.*]]: index):
// CHECK:                 %[[VAL_54:.*]] = scf.if %[[VAL_51]] -> (index) {
// CHECK:                   %[[VAL_55:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_47]]] : memref<?xi64>
// CHECK:                   %[[VAL_56:.*]] = arith.index_cast %[[VAL_55]] : i64 to index
// CHECK:                   scf.yield %[[VAL_56]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_49]] : index
// CHECK:                 }
// CHECK:                 %[[VAL_57:.*]] = scf.if %[[VAL_52]] -> (index) {
// CHECK:                   %[[VAL_58:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_48]]] : memref<?xi64>
// CHECK:                   %[[VAL_59:.*]] = arith.index_cast %[[VAL_58]] : i64 to index
// CHECK:                   scf.yield %[[VAL_59]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_50]] : index
// CHECK:                 }
// CHECK:                 %[[VAL_60:.*]] = arith.cmpi ult, %[[VAL_61:.*]], %[[VAL_62:.*]] : index
// CHECK:                 %[[VAL_63:.*]] = arith.cmpi ugt, %[[VAL_61]], %[[VAL_62]] : index
// CHECK:                 %[[VAL_64:.*]] = arith.addi %[[VAL_47]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_65:.*]] = arith.addi %[[VAL_48]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_66:.*]] = arith.addi %[[VAL_53]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_67:.*]]:5 = scf.if %[[VAL_60]] -> (index, index, i1, i1, index) {
// CHECK:                   scf.yield %[[VAL_64]], %[[VAL_48]], %[[VAL_6]], %[[VAL_5]], %[[VAL_53]] : index, index, i1, i1, index
// CHECK:                 } else {
// CHECK:                   %[[VAL_68:.*]]:5 = scf.if %[[VAL_63]] -> (index, index, i1, i1, index) {
// CHECK:                     scf.yield %[[VAL_47]], %[[VAL_65]], %[[VAL_5]], %[[VAL_6]], %[[VAL_53]] : index, index, i1, i1, index
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_64]], %[[VAL_65]], %[[VAL_6]], %[[VAL_6]], %[[VAL_66]] : index, index, i1, i1, index
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_69:.*]]#0, %[[VAL_69]]#1, %[[VAL_69]]#2, %[[VAL_69]]#3, %[[VAL_69]]#4 : index, index, i1, i1, index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_70:.*]]#0, %[[VAL_70]]#1, %[[VAL_61]], %[[VAL_62]], %[[VAL_70]]#2, %[[VAL_70]]#3, %[[VAL_70]]#4 : index, index, index, index, i1, i1, index
// CHECK:               }
// CHECK:               %[[VAL_71:.*]] = arith.index_cast %[[VAL_72:.*]]#6 : index to i64
// CHECK:               scf.yield %[[VAL_71]] : i64
// CHECK:             }
// CHECK:             memref.store %[[VAL_73:.*]], %[[VAL_21]]{{\[}}%[[VAL_22]]] : memref<?xi64>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           memref.store %[[VAL_4]], %[[VAL_21]]{{\[}}%[[VAL_14]]] : memref<?xi64>
// CHECK:           scf.for %[[VAL_74:.*]] = %[[VAL_2]] to %[[VAL_14]] step %[[VAL_3]] {
// CHECK:             %[[VAL_75:.*]] = memref.load %[[VAL_21]]{{\[}}%[[VAL_74]]] : memref<?xi64>
// CHECK:             %[[VAL_76:.*]] = memref.load %[[VAL_21]]{{\[}}%[[VAL_14]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_76]], %[[VAL_21]]{{\[}}%[[VAL_74]]] : memref<?xi64>
// CHECK:             %[[VAL_77:.*]] = arith.addi %[[VAL_76]], %[[VAL_75]] : i64
// CHECK:             memref.store %[[VAL_77]], %[[VAL_21]]{{\[}}%[[VAL_14]]] : memref<?xi64>
// CHECK:           }
// CHECK:           %[[VAL_78:.*]] = sparse_tensor.pointers %[[VAL_10]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_79:.*]] = tensor.dim %[[VAL_10]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_80:.*]] = memref.load %[[VAL_78]]{{\[}}%[[VAL_79]]] : memref<?xi64>
// CHECK:           %[[VAL_81:.*]] = arith.index_cast %[[VAL_80]] : i64 to index
// CHECK:           %[[VAL_82:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_10]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_index(%[[VAL_82]], %[[VAL_3]], %[[VAL_81]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_83:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_10]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_values(%[[VAL_83]], %[[VAL_81]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:           %[[VAL_84:.*]] = sparse_tensor.indices %[[VAL_10]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_85:.*]] = sparse_tensor.values %[[VAL_10]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           scf.parallel (%[[VAL_86:.*]]) = (%[[VAL_2]]) to (%[[VAL_14]]) step (%[[VAL_3]]) {
// CHECK:             %[[VAL_87:.*]] = arith.addi %[[VAL_86]], %[[VAL_3]] : index
// CHECK:             %[[VAL_88:.*]] = memref.load %[[VAL_21]]{{\[}}%[[VAL_86]]] : memref<?xi64>
// CHECK:             %[[VAL_89:.*]] = memref.load %[[VAL_21]]{{\[}}%[[VAL_87]]] : memref<?xi64>
// CHECK:             %[[VAL_90:.*]] = arith.cmpi ne, %[[VAL_88]], %[[VAL_89]] : i64
// CHECK:             scf.if %[[VAL_90]] {
// CHECK:               %[[VAL_91:.*]] = memref.load %[[VAL_21]]{{\[}}%[[VAL_86]]] : memref<?xi64>
// CHECK:               %[[VAL_92:.*]] = arith.index_cast %[[VAL_91]] : i64 to index
// CHECK:               %[[VAL_93:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_86]]] : memref<?xi64>
// CHECK:               %[[VAL_94:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_87]]] : memref<?xi64>
// CHECK:               %[[VAL_95:.*]] = arith.index_cast %[[VAL_93]] : i64 to index
// CHECK:               %[[VAL_96:.*]] = arith.index_cast %[[VAL_94]] : i64 to index
// CHECK:               %[[VAL_97:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_86]]] : memref<?xi64>
// CHECK:               %[[VAL_98:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_87]]] : memref<?xi64>
// CHECK:               %[[VAL_99:.*]] = arith.index_cast %[[VAL_97]] : i64 to index
// CHECK:               %[[VAL_100:.*]] = arith.index_cast %[[VAL_98]] : i64 to index
// CHECK:               %[[VAL_101:.*]]:9 = scf.while (%[[VAL_102:.*]] = %[[VAL_95]], %[[VAL_103:.*]] = %[[VAL_99]], %[[VAL_104:.*]] = %[[VAL_92]], %[[VAL_105:.*]] = %[[VAL_4]], %[[VAL_106:.*]] = %[[VAL_4]], %[[VAL_107:.*]] = %[[VAL_7]], %[[VAL_108:.*]] = %[[VAL_7]], %[[VAL_109:.*]] = %[[VAL_6]], %[[VAL_110:.*]] = %[[VAL_6]]) : (index, index, index, i64, i64, f64, f64, i1, i1) -> (index, index, index, i64, i64, f64, f64, i1, i1) {
// CHECK:                 %[[VAL_111:.*]] = arith.cmpi ult, %[[VAL_102]], %[[VAL_96]] : index
// CHECK:                 %[[VAL_112:.*]] = arith.cmpi ult, %[[VAL_103]], %[[VAL_100]] : index
// CHECK:                 %[[VAL_113:.*]] = arith.andi %[[VAL_111]], %[[VAL_112]] : i1
// CHECK:                 scf.condition(%[[VAL_113]]) %[[VAL_102]], %[[VAL_103]], %[[VAL_104]], %[[VAL_105]], %[[VAL_106]], %[[VAL_107]], %[[VAL_108]], %[[VAL_109]], %[[VAL_110]] : index, index, index, i64, i64, f64, f64, i1, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_114:.*]]: index, %[[VAL_115:.*]]: index, %[[VAL_116:.*]]: index, %[[VAL_117:.*]]: i64, %[[VAL_118:.*]]: i64, %[[VAL_119:.*]]: f64, %[[VAL_120:.*]]: f64, %[[VAL_121:.*]]: i1, %[[VAL_122:.*]]: i1):
// CHECK:                 %[[VAL_123:.*]]:2 = scf.if %[[VAL_121]] -> (i64, f64) {
// CHECK:                   %[[VAL_124:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_114]]] : memref<?xi64>
// CHECK:                   %[[VAL_125:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_114]]] : memref<?xf64>
// CHECK:                   scf.yield %[[VAL_124]], %[[VAL_125]] : i64, f64
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_117]], %[[VAL_119]] : i64, f64
// CHECK:                 }
// CHECK:                 %[[VAL_126:.*]]:2 = scf.if %[[VAL_122]] -> (i64, f64) {
// CHECK:                   %[[VAL_127:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_115]]] : memref<?xi64>
// CHECK:                   %[[VAL_128:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_115]]] : memref<?xf64>
// CHECK:                   scf.yield %[[VAL_127]], %[[VAL_128]] : i64, f64
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_118]], %[[VAL_120]] : i64, f64
// CHECK:                 }
// CHECK:                 %[[VAL_129:.*]] = arith.cmpi ult, %[[VAL_130:.*]]#0, %[[VAL_131:.*]]#0 : i64
// CHECK:                 %[[VAL_132:.*]] = arith.cmpi ugt, %[[VAL_130]]#0, %[[VAL_131]]#0 : i64
// CHECK:                 %[[VAL_133:.*]] = arith.addi %[[VAL_114]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_134:.*]] = arith.addi %[[VAL_115]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_135:.*]] = arith.addi %[[VAL_116]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_136:.*]]:5 = scf.if %[[VAL_129]] -> (index, index, index, i1, i1) {
// CHECK:                   scf.yield %[[VAL_133]], %[[VAL_115]], %[[VAL_116]], %[[VAL_6]], %[[VAL_5]] : index, index, index, i1, i1
// CHECK:                 } else {
// CHECK:                   %[[VAL_137:.*]]:5 = scf.if %[[VAL_132]] -> (index, index, index, i1, i1) {
// CHECK:                     scf.yield %[[VAL_114]], %[[VAL_134]], %[[VAL_116]], %[[VAL_5]], %[[VAL_6]] : index, index, index, i1, i1
// CHECK:                   } else {
// CHECK:                     memref.store %[[VAL_130]]#0, %[[VAL_84]]{{\[}}%[[VAL_116]]] : memref<?xi64>
// CHECK:                     %[[VAL_138:.*]] = arith.mulf %[[VAL_130]]#1, %[[VAL_131]]#1 : f64
// CHECK:                     memref.store %[[VAL_138]], %[[VAL_85]]{{\[}}%[[VAL_116]]] : memref<?xf64>
// CHECK:                     scf.yield %[[VAL_133]], %[[VAL_134]], %[[VAL_135]], %[[VAL_6]], %[[VAL_6]] : index, index, index, i1, i1
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_139:.*]]#0, %[[VAL_139]]#1, %[[VAL_139]]#2, %[[VAL_139]]#3, %[[VAL_139]]#4 : index, index, index, i1, i1
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_140:.*]]#0, %[[VAL_140]]#1, %[[VAL_140]]#2, %[[VAL_130]]#0, %[[VAL_131]]#0, %[[VAL_130]]#1, %[[VAL_131]]#1, %[[VAL_140]]#3, %[[VAL_140]]#4 : index, index, index, i64, i64, f64, f64, i1, i1
// CHECK:               }
// CHECK:             }
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           return %[[VAL_10]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

func @matrix_intersect(%a: tensor<?x?xf64, #CSR64>, %b: tensor<?x?xf64, #CSR64>) -> tensor<?x?xf64, #CSR64> {
    %result = graphblas.intersect %a, %b { intersect_operator = "times" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64>) to tensor<?x?xf64, #CSR64>
    return %result : tensor<?x?xf64, #CSR64>
}
