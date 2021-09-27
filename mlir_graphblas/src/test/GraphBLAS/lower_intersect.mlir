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


// CHECK-LABEL:   .func @vector_intersect(
// CHECK-SAME:                            %[[VAL_0:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                            %[[VAL_1:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK-DAG:       %[[VAL_2:.*]] = constant 0 : index
// CHECK-DAG:       %[[VAL_3:.*]] = constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = constant 0 : i64
// CHECK-DAG:       %[[VAL_5:.*]] = constant false
// CHECK-DAG:       %[[VAL_6:.*]] = constant true
// CHECK-DAG:       %[[VAL_7:.*]] = constant 0.000000e+00 : f64
// CHECK:           %[[VAL_8:.*]] = call @vector_f64_p64i64_to_ptr8(%[[VAL_0]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_9:.*]] = call @empty_like(%[[VAL_8]]) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_10:.*]] = call @ptr8_to_vector_f64_p64i64(%[[VAL_9]]) : (!llvm.ptr<i8>) -> tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_11:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_12:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_3]]] : memref<?xi64>
// CHECK:           %[[VAL_13:.*]] = index_cast %[[VAL_12]] : i64 to index
// CHECK:           %[[VAL_14:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_15:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_3]]] : memref<?xi64>
// CHECK:           %[[VAL_16:.*]] = index_cast %[[VAL_15]] : i64 to index
// CHECK:           %[[VAL_17:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_18:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_19:.*]] = sparse_tensor.indices %[[VAL_1]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_20:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
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
// CHECK:               scf.yield %[[VAL_49]], %[[VAL_33]], %[[VAL_6]], %[[VAL_5]], %[[VAL_38]] : index, index, i1, i1, index
// CHECK:             } else {
// CHECK:               %[[VAL_53:.*]]:5 = scf.if %[[VAL_48]] -> (index, index, i1, i1, index) {
// CHECK:                 scf.yield %[[VAL_32]], %[[VAL_50]], %[[VAL_5]], %[[VAL_6]], %[[VAL_38]] : index, index, i1, i1, index
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_49]], %[[VAL_50]], %[[VAL_6]], %[[VAL_6]], %[[VAL_51]] : index, index, i1, i1, index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_54:.*]]#0, %[[VAL_54]]#1, %[[VAL_54]]#2, %[[VAL_54]]#3, %[[VAL_54]]#4 : index, index, i1, i1, index
// CHECK:             }
// CHECK:             scf.yield %[[VAL_55:.*]]#0, %[[VAL_55]]#1, %[[VAL_46]], %[[VAL_47]], %[[VAL_55]]#2, %[[VAL_55]]#3, %[[VAL_55]]#4 : index, index, index, index, i1, i1, index
// CHECK:           }
// CHECK:           %[[VAL_56:.*]] = index_cast %[[VAL_57:.*]]#6 : index to i64
// CHECK:           %[[VAL_58:.*]] = call @vector_f64_p64i64_to_ptr8(%[[VAL_10]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_index(%[[VAL_58]], %[[VAL_2]], %[[VAL_57]]#6) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_59:.*]] = call @vector_f64_p64i64_to_ptr8(%[[VAL_10]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_values(%[[VAL_59]], %[[VAL_57]]#6) : (!llvm.ptr<i8>, index) -> ()
// CHECK:           %[[VAL_60:.*]] = sparse_tensor.pointers %[[VAL_10]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           memref.store %[[VAL_56]], %[[VAL_60]]{{\[}}%[[VAL_3]]] : memref<?xi64>
// CHECK:           %[[VAL_61:.*]] = sparse_tensor.indices %[[VAL_10]], %[[VAL_2]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_62:.*]] = sparse_tensor.values %[[VAL_10]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_63:.*]]:9 = scf.while (%[[VAL_64:.*]] = %[[VAL_2]], %[[VAL_65:.*]] = %[[VAL_2]], %[[VAL_66:.*]] = %[[VAL_2]], %[[VAL_67:.*]] = %[[VAL_4]], %[[VAL_68:.*]] = %[[VAL_4]], %[[VAL_69:.*]] = %[[VAL_7]], %[[VAL_70:.*]] = %[[VAL_7]], %[[VAL_71:.*]] = %[[VAL_6]], %[[VAL_72:.*]] = %[[VAL_6]]) : (index, index, index, i64, i64, f64, f64, i1, i1) -> (index, index, index, i64, i64, f64, f64, i1, i1) {
// CHECK:             %[[VAL_73:.*]] = cmpi ult, %[[VAL_64]], %[[VAL_13]] : index
// CHECK:             %[[VAL_74:.*]] = cmpi ult, %[[VAL_65]], %[[VAL_16]] : index
// CHECK:             %[[VAL_75:.*]] = and %[[VAL_73]], %[[VAL_74]] : i1
// CHECK:             scf.condition(%[[VAL_75]]) %[[VAL_64]], %[[VAL_65]], %[[VAL_66]], %[[VAL_67]], %[[VAL_68]], %[[VAL_69]], %[[VAL_70]], %[[VAL_71]], %[[VAL_72]] : index, index, index, i64, i64, f64, f64, i1, i1
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_76:.*]]: index, %[[VAL_77:.*]]: index, %[[VAL_78:.*]]: index, %[[VAL_79:.*]]: i64, %[[VAL_80:.*]]: i64, %[[VAL_81:.*]]: f64, %[[VAL_82:.*]]: f64, %[[VAL_83:.*]]: i1, %[[VAL_84:.*]]: i1):
// CHECK:             %[[VAL_85:.*]]:2 = scf.if %[[VAL_83]] -> (i64, f64) {
// CHECK:               %[[VAL_86:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_76]]] : memref<?xi64>
// CHECK:               %[[VAL_87:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_76]]] : memref<?xf64>
// CHECK:               scf.yield %[[VAL_86]], %[[VAL_87]] : i64, f64
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_79]], %[[VAL_81]] : i64, f64
// CHECK:             }
// CHECK:             %[[VAL_88:.*]]:2 = scf.if %[[VAL_84]] -> (i64, f64) {
// CHECK:               %[[VAL_89:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_77]]] : memref<?xi64>
// CHECK:               %[[VAL_90:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_77]]] : memref<?xf64>
// CHECK:               scf.yield %[[VAL_89]], %[[VAL_90]] : i64, f64
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_80]], %[[VAL_82]] : i64, f64
// CHECK:             }
// CHECK:             %[[VAL_91:.*]] = cmpi ult, %[[VAL_92:.*]]#0, %[[VAL_93:.*]]#0 : i64
// CHECK:             %[[VAL_94:.*]] = cmpi ugt, %[[VAL_92]]#0, %[[VAL_93]]#0 : i64
// CHECK:             %[[VAL_95:.*]] = addi %[[VAL_76]], %[[VAL_3]] : index
// CHECK:             %[[VAL_96:.*]] = addi %[[VAL_77]], %[[VAL_3]] : index
// CHECK:             %[[VAL_97:.*]] = addi %[[VAL_78]], %[[VAL_3]] : index
// CHECK:             %[[VAL_98:.*]]:5 = scf.if %[[VAL_91]] -> (index, index, index, i1, i1) {
// CHECK:               scf.yield %[[VAL_95]], %[[VAL_77]], %[[VAL_78]], %[[VAL_6]], %[[VAL_5]] : index, index, index, i1, i1
// CHECK:             } else {
// CHECK:               %[[VAL_99:.*]]:5 = scf.if %[[VAL_94]] -> (index, index, index, i1, i1) {
// CHECK:                 scf.yield %[[VAL_76]], %[[VAL_96]], %[[VAL_78]], %[[VAL_5]], %[[VAL_6]] : index, index, index, i1, i1
// CHECK:               } else {
// CHECK:                 memref.store %[[VAL_92]]#0, %[[VAL_61]]{{\[}}%[[VAL_78]]] : memref<?xi64>
// CHECK:                 %[[VAL_100:.*]] = mulf %[[VAL_92]]#1, %[[VAL_93]]#1 : f64
// CHECK:                 memref.store %[[VAL_100]], %[[VAL_62]]{{\[}}%[[VAL_78]]] : memref<?xf64>
// CHECK:                 scf.yield %[[VAL_95]], %[[VAL_96]], %[[VAL_97]], %[[VAL_6]], %[[VAL_6]] : index, index, index, i1, i1
// CHECK:               }
// CHECK:               scf.yield %[[VAL_101:.*]]#0, %[[VAL_101]]#1, %[[VAL_101]]#2, %[[VAL_101]]#3, %[[VAL_101]]#4 : index, index, index, i1, i1
// CHECK:             }
// CHECK:             scf.yield %[[VAL_102:.*]]#0, %[[VAL_102]]#1, %[[VAL_102]]#2, %[[VAL_92]]#0, %[[VAL_93]]#0, %[[VAL_92]]#1, %[[VAL_93]]#1, %[[VAL_102]]#3, %[[VAL_102]]#4 : index, index, index, i64, i64, f64, f64, i1, i1
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
// CHECK-DAG:       %[[VAL_2:.*]] = constant 0 : index
// CHECK-DAG:       %[[VAL_3:.*]] = constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = constant 0 : i64
// CHECK-DAG:       %[[VAL_5:.*]] = constant false
// CHECK-DAG:       %[[VAL_6:.*]] = constant true
// CHECK-DAG:       %[[VAL_7:.*]] = constant 0.000000e+00 : f64
// CHECK:           %[[VAL_8:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_0]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_9:.*]] = call @empty_like(%[[VAL_8]]) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_10:.*]] = call @ptr8_to_matrix_csr_f64_p64i64(%[[VAL_9]]) : (!llvm.ptr<i8>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_11:.*]] = tensor.dim %[[VAL_10]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_12:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_13:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_14:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_15:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_16:.*]] = sparse_tensor.indices %[[VAL_1]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_17:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_18:.*]] = sparse_tensor.pointers %[[VAL_10]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           scf.parallel (%[[VAL_19:.*]]) = (%[[VAL_2]]) to (%[[VAL_11]]) step (%[[VAL_3]]) {
// CHECK:             %[[VAL_20:.*]] = addi %[[VAL_19]], %[[VAL_3]] : index
// CHECK:             %[[VAL_21:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_19]]] : memref<?xi64>
// CHECK:             %[[VAL_22:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_20]]] : memref<?xi64>
// CHECK:             %[[VAL_23:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_19]]] : memref<?xi64>
// CHECK:             %[[VAL_24:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_20]]] : memref<?xi64>
// CHECK:             %[[VAL_25:.*]] = cmpi eq, %[[VAL_21]], %[[VAL_22]] : i64
// CHECK:             %[[VAL_26:.*]] = cmpi eq, %[[VAL_23]], %[[VAL_24]] : i64
// CHECK:             %[[VAL_27:.*]] = or %[[VAL_25]], %[[VAL_26]] : i1
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
// CHECK:                   scf.yield %[[VAL_61]], %[[VAL_45]], %[[VAL_6]], %[[VAL_5]], %[[VAL_50]] : index, index, i1, i1, index
// CHECK:                 } else {
// CHECK:                   %[[VAL_65:.*]]:5 = scf.if %[[VAL_60]] -> (index, index, i1, i1, index) {
// CHECK:                     scf.yield %[[VAL_44]], %[[VAL_62]], %[[VAL_5]], %[[VAL_6]], %[[VAL_50]] : index, index, i1, i1, index
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_61]], %[[VAL_62]], %[[VAL_6]], %[[VAL_6]], %[[VAL_63]] : index, index, i1, i1, index
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_66:.*]]#0, %[[VAL_66]]#1, %[[VAL_66]]#2, %[[VAL_66]]#3, %[[VAL_66]]#4 : index, index, i1, i1, index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_67:.*]]#0, %[[VAL_67]]#1, %[[VAL_58]], %[[VAL_59]], %[[VAL_67]]#2, %[[VAL_67]]#3, %[[VAL_67]]#4 : index, index, index, index, i1, i1, index
// CHECK:               }
// CHECK:               %[[VAL_68:.*]] = index_cast %[[VAL_69:.*]]#6 : index to i64
// CHECK:               scf.yield %[[VAL_68]] : i64
// CHECK:             }
// CHECK:             memref.store %[[VAL_70:.*]], %[[VAL_18]]{{\[}}%[[VAL_19]]] : memref<?xi64>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           memref.store %[[VAL_4]], %[[VAL_18]]{{\[}}%[[VAL_11]]] : memref<?xi64>
// CHECK:           scf.for %[[VAL_71:.*]] = %[[VAL_2]] to %[[VAL_11]] step %[[VAL_3]] {
// CHECK:             %[[VAL_72:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_71]]] : memref<?xi64>
// CHECK:             %[[VAL_73:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_11]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_73]], %[[VAL_18]]{{\[}}%[[VAL_71]]] : memref<?xi64>
// CHECK:             %[[VAL_74:.*]] = addi %[[VAL_73]], %[[VAL_72]] : i64
// CHECK:             memref.store %[[VAL_74]], %[[VAL_18]]{{\[}}%[[VAL_11]]] : memref<?xi64>
// CHECK:           }
// CHECK:           %[[VAL_75:.*]] = sparse_tensor.pointers %[[VAL_10]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_76:.*]] = tensor.dim %[[VAL_10]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_77:.*]] = memref.load %[[VAL_75]]{{\[}}%[[VAL_76]]] : memref<?xi64>
// CHECK:           %[[VAL_78:.*]] = index_cast %[[VAL_77]] : i64 to index
// CHECK:           %[[VAL_79:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_10]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_index(%[[VAL_79]], %[[VAL_3]], %[[VAL_78]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_80:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_10]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_values(%[[VAL_80]], %[[VAL_78]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:           %[[VAL_81:.*]] = sparse_tensor.indices %[[VAL_10]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_82:.*]] = sparse_tensor.values %[[VAL_10]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           scf.parallel (%[[VAL_83:.*]]) = (%[[VAL_2]]) to (%[[VAL_11]]) step (%[[VAL_3]]) {
// CHECK:             %[[VAL_84:.*]] = addi %[[VAL_83]], %[[VAL_3]] : index
// CHECK:             %[[VAL_85:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_83]]] : memref<?xi64>
// CHECK:             %[[VAL_86:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_84]]] : memref<?xi64>
// CHECK:             %[[VAL_87:.*]] = cmpi ne, %[[VAL_85]], %[[VAL_86]] : i64
// CHECK:             scf.if %[[VAL_87]] {
// CHECK:               %[[VAL_88:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_83]]] : memref<?xi64>
// CHECK:               %[[VAL_89:.*]] = index_cast %[[VAL_88]] : i64 to index
// CHECK:               %[[VAL_90:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_83]]] : memref<?xi64>
// CHECK:               %[[VAL_91:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_84]]] : memref<?xi64>
// CHECK:               %[[VAL_92:.*]] = index_cast %[[VAL_90]] : i64 to index
// CHECK:               %[[VAL_93:.*]] = index_cast %[[VAL_91]] : i64 to index
// CHECK:               %[[VAL_94:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_83]]] : memref<?xi64>
// CHECK:               %[[VAL_95:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_84]]] : memref<?xi64>
// CHECK:               %[[VAL_96:.*]] = index_cast %[[VAL_94]] : i64 to index
// CHECK:               %[[VAL_97:.*]] = index_cast %[[VAL_95]] : i64 to index
// CHECK:               %[[VAL_98:.*]]:9 = scf.while (%[[VAL_99:.*]] = %[[VAL_92]], %[[VAL_100:.*]] = %[[VAL_96]], %[[VAL_101:.*]] = %[[VAL_89]], %[[VAL_102:.*]] = %[[VAL_4]], %[[VAL_103:.*]] = %[[VAL_4]], %[[VAL_104:.*]] = %[[VAL_7]], %[[VAL_105:.*]] = %[[VAL_7]], %[[VAL_106:.*]] = %[[VAL_6]], %[[VAL_107:.*]] = %[[VAL_6]]) : (index, index, index, i64, i64, f64, f64, i1, i1) -> (index, index, index, i64, i64, f64, f64, i1, i1) {
// CHECK:                 %[[VAL_108:.*]] = cmpi ult, %[[VAL_99]], %[[VAL_93]] : index
// CHECK:                 %[[VAL_109:.*]] = cmpi ult, %[[VAL_100]], %[[VAL_97]] : index
// CHECK:                 %[[VAL_110:.*]] = and %[[VAL_108]], %[[VAL_109]] : i1
// CHECK:                 scf.condition(%[[VAL_110]]) %[[VAL_99]], %[[VAL_100]], %[[VAL_101]], %[[VAL_102]], %[[VAL_103]], %[[VAL_104]], %[[VAL_105]], %[[VAL_106]], %[[VAL_107]] : index, index, index, i64, i64, f64, f64, i1, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_111:.*]]: index, %[[VAL_112:.*]]: index, %[[VAL_113:.*]]: index, %[[VAL_114:.*]]: i64, %[[VAL_115:.*]]: i64, %[[VAL_116:.*]]: f64, %[[VAL_117:.*]]: f64, %[[VAL_118:.*]]: i1, %[[VAL_119:.*]]: i1):
// CHECK:                 %[[VAL_120:.*]]:2 = scf.if %[[VAL_118]] -> (i64, f64) {
// CHECK:                   %[[VAL_121:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_111]]] : memref<?xi64>
// CHECK:                   %[[VAL_122:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_111]]] : memref<?xf64>
// CHECK:                   scf.yield %[[VAL_121]], %[[VAL_122]] : i64, f64
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_114]], %[[VAL_116]] : i64, f64
// CHECK:                 }
// CHECK:                 %[[VAL_123:.*]]:2 = scf.if %[[VAL_119]] -> (i64, f64) {
// CHECK:                   %[[VAL_124:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_112]]] : memref<?xi64>
// CHECK:                   %[[VAL_125:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_112]]] : memref<?xf64>
// CHECK:                   scf.yield %[[VAL_124]], %[[VAL_125]] : i64, f64
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_115]], %[[VAL_117]] : i64, f64
// CHECK:                 }
// CHECK:                 %[[VAL_126:.*]] = cmpi ult, %[[VAL_127:.*]]#0, %[[VAL_128:.*]]#0 : i64
// CHECK:                 %[[VAL_129:.*]] = cmpi ugt, %[[VAL_127]]#0, %[[VAL_128]]#0 : i64
// CHECK:                 %[[VAL_130:.*]] = addi %[[VAL_111]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_131:.*]] = addi %[[VAL_112]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_132:.*]] = addi %[[VAL_113]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_133:.*]]:5 = scf.if %[[VAL_126]] -> (index, index, index, i1, i1) {
// CHECK:                   scf.yield %[[VAL_130]], %[[VAL_112]], %[[VAL_113]], %[[VAL_6]], %[[VAL_5]] : index, index, index, i1, i1
// CHECK:                 } else {
// CHECK:                   %[[VAL_134:.*]]:5 = scf.if %[[VAL_129]] -> (index, index, index, i1, i1) {
// CHECK:                     scf.yield %[[VAL_111]], %[[VAL_131]], %[[VAL_113]], %[[VAL_5]], %[[VAL_6]] : index, index, index, i1, i1
// CHECK:                   } else {
// CHECK:                     memref.store %[[VAL_127]]#0, %[[VAL_81]]{{\[}}%[[VAL_113]]] : memref<?xi64>
// CHECK:                     %[[VAL_135:.*]] = mulf %[[VAL_127]]#1, %[[VAL_128]]#1 : f64
// CHECK:                     memref.store %[[VAL_135]], %[[VAL_82]]{{\[}}%[[VAL_113]]] : memref<?xf64>
// CHECK:                     scf.yield %[[VAL_130]], %[[VAL_131]], %[[VAL_132]], %[[VAL_6]], %[[VAL_6]] : index, index, index, i1, i1
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_136:.*]]#0, %[[VAL_136]]#1, %[[VAL_136]]#2, %[[VAL_136]]#3, %[[VAL_136]]#4 : index, index, index, i1, i1
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_137:.*]]#0, %[[VAL_137]]#1, %[[VAL_137]]#2, %[[VAL_127]]#0, %[[VAL_128]]#0, %[[VAL_127]]#1, %[[VAL_128]]#1, %[[VAL_137]]#3, %[[VAL_137]]#4 : index, index, index, i64, i64, f64, f64, i1, i1
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
