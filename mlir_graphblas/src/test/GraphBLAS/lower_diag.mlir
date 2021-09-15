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

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {

// CHECK:           func @vec_to_mat_fixed_csr(%[[VAL_0:.*]]: tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_1:.*]] = constant 0 : i64
// CHECK:             %[[VAL_2:.*]] = constant 1 : i64
// CHECK:             %[[VAL_3:.*]] = constant 7 : index
// CHECK:             %[[VAL_4:.*]] = constant 0 : index
// CHECK:             %[[VAL_5:.*]] = constant 1 : index
// CHECK:             %[[VAL_6:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_4]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             %[[VAL_8:.*]] = call @new_matrix_csr_f64_p64i64(%[[VAL_3]], %[[VAL_3]]) : (index, index) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_9:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_4]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_10:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_5]]] : memref<?xi64>
// CHECK:             %[[VAL_11:.*]] = index_cast %[[VAL_10]] : i64 to index
// CHECK:             %[[VAL_12:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_8]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_index(%[[VAL_12]], %[[VAL_5]], %[[VAL_11]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_13:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_8]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_values(%[[VAL_13]], %[[VAL_11]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:             %[[VAL_14:.*]] = sparse_tensor.indices %[[VAL_8]], %[[VAL_5]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_15:.*]] = sparse_tensor.values %[[VAL_8]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             scf.for %[[VAL_16:.*]] = %[[VAL_4]] to %[[VAL_11]] step %[[VAL_5]] {
// CHECK:               %[[VAL_17:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_16]]] : memref<?xi64>
// CHECK:               memref.store %[[VAL_17]], %[[VAL_14]]{{\[}}%[[VAL_16]]] : memref<?xi64>
// CHECK:               %[[VAL_18:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_16]]] : memref<?xf64>
// CHECK:               memref.store %[[VAL_18]], %[[VAL_15]]{{\[}}%[[VAL_16]]] : memref<?xf64>
// CHECK:             }
// CHECK:             %[[VAL_19:.*]] = sparse_tensor.pointers %[[VAL_8]], %[[VAL_5]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_20:.*]]:3 = scf.for %[[VAL_21:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_5]] iter_args(%[[VAL_22:.*]] = %[[VAL_1]], %[[VAL_23:.*]] = %[[VAL_4]], %[[VAL_24:.*]] = %[[VAL_1]]) -> (i64, index, i64) {
// CHECK:               memref.store %[[VAL_22]], %[[VAL_19]]{{\[}}%[[VAL_21]]] : memref<?xi64>
// CHECK:               %[[VAL_25:.*]] = index_cast %[[VAL_21]] : index to i64
// CHECK:               %[[VAL_26:.*]] = cmpi eq, %[[VAL_24]], %[[VAL_25]] : i64
// CHECK:               %[[VAL_27:.*]]:3 = scf.if %[[VAL_26]] -> (i64, index, i64) {
// CHECK:                 %[[VAL_28:.*]] = addi %[[VAL_22]], %[[VAL_2]] : i64
// CHECK:                 %[[VAL_29:.*]] = addi %[[VAL_23]], %[[VAL_5]] : index
// CHECK:                 %[[VAL_30:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_29]]] : memref<?xi64>
// CHECK:                 scf.yield %[[VAL_28]], %[[VAL_29]], %[[VAL_30]] : i64, index, i64
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_22]], %[[VAL_23]], %[[VAL_24]] : i64, index, i64
// CHECK:               }
// CHECK:               scf.yield %[[VAL_31:.*]]#0, %[[VAL_31]]#1, %[[VAL_31]]#2 : i64, index, i64
// CHECK:             }
// CHECK:             memref.store %[[VAL_10]], %[[VAL_19]]{{\[}}%[[VAL_3]]] : memref<?xi64>
// CHECK:             return %[[VAL_8]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

   func @vec_to_mat_fixed_csr(%sparse_tensor: tensor<7xf64, #SparseVec64>) -> tensor<7x7xf64, #CSR64> {
       %answer = graphblas.diag %sparse_tensor : tensor<7xf64, #SparseVec64> to tensor<7x7xf64, #CSR64>
       return %answer : tensor<7x7xf64, #CSR64>
   }

// CHECK:           func @vec_to_mat_fixed_csc(%[[VAL_32:.*]]: tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_33:.*]] = constant 0 : i64
// CHECK:             %[[VAL_34:.*]] = constant 1 : i64
// CHECK:             %[[VAL_35:.*]] = constant 7 : index
// CHECK:             %[[VAL_36:.*]] = constant 0 : index
// CHECK:             %[[VAL_37:.*]] = constant 1 : index
// CHECK:             %[[VAL_38:.*]] = sparse_tensor.indices %[[VAL_32]], %[[VAL_36]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_39:.*]] = sparse_tensor.values %[[VAL_32]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             %[[VAL_40:.*]] = call @new_matrix_csc_f64_p64i64(%[[VAL_35]], %[[VAL_35]]) : (index, index) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_41:.*]] = sparse_tensor.pointers %[[VAL_32]], %[[VAL_36]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_42:.*]] = memref.load %[[VAL_41]]{{\[}}%[[VAL_37]]] : memref<?xi64>
// CHECK:             %[[VAL_43:.*]] = index_cast %[[VAL_42]] : i64 to index
// CHECK:             %[[VAL_44:.*]] = call @matrix_csc_f64_p64i64_to_ptr8(%[[VAL_40]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_index(%[[VAL_44]], %[[VAL_37]], %[[VAL_43]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_45:.*]] = call @matrix_csc_f64_p64i64_to_ptr8(%[[VAL_40]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_values(%[[VAL_45]], %[[VAL_43]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:             %[[VAL_46:.*]] = sparse_tensor.indices %[[VAL_40]], %[[VAL_37]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_47:.*]] = sparse_tensor.values %[[VAL_40]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             scf.for %[[VAL_48:.*]] = %[[VAL_36]] to %[[VAL_43]] step %[[VAL_37]] {
// CHECK:               %[[VAL_49:.*]] = memref.load %[[VAL_38]]{{\[}}%[[VAL_48]]] : memref<?xi64>
// CHECK:               memref.store %[[VAL_49]], %[[VAL_46]]{{\[}}%[[VAL_48]]] : memref<?xi64>
// CHECK:               %[[VAL_50:.*]] = memref.load %[[VAL_39]]{{\[}}%[[VAL_48]]] : memref<?xf64>
// CHECK:               memref.store %[[VAL_50]], %[[VAL_47]]{{\[}}%[[VAL_48]]] : memref<?xf64>
// CHECK:             }
// CHECK:             %[[VAL_51:.*]] = sparse_tensor.pointers %[[VAL_40]], %[[VAL_37]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_52:.*]]:3 = scf.for %[[VAL_53:.*]] = %[[VAL_36]] to %[[VAL_35]] step %[[VAL_37]] iter_args(%[[VAL_54:.*]] = %[[VAL_33]], %[[VAL_55:.*]] = %[[VAL_36]], %[[VAL_56:.*]] = %[[VAL_33]]) -> (i64, index, i64) {
// CHECK:               memref.store %[[VAL_54]], %[[VAL_51]]{{\[}}%[[VAL_53]]] : memref<?xi64>
// CHECK:               %[[VAL_57:.*]] = index_cast %[[VAL_53]] : index to i64
// CHECK:               %[[VAL_58:.*]] = cmpi eq, %[[VAL_56]], %[[VAL_57]] : i64
// CHECK:               %[[VAL_59:.*]]:3 = scf.if %[[VAL_58]] -> (i64, index, i64) {
// CHECK:                 %[[VAL_60:.*]] = addi %[[VAL_54]], %[[VAL_34]] : i64
// CHECK:                 %[[VAL_61:.*]] = addi %[[VAL_55]], %[[VAL_37]] : index
// CHECK:                 %[[VAL_62:.*]] = memref.load %[[VAL_38]]{{\[}}%[[VAL_61]]] : memref<?xi64>
// CHECK:                 scf.yield %[[VAL_60]], %[[VAL_61]], %[[VAL_62]] : i64, index, i64
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_54]], %[[VAL_55]], %[[VAL_56]] : i64, index, i64
// CHECK:               }
// CHECK:               scf.yield %[[VAL_63:.*]]#0, %[[VAL_63]]#1, %[[VAL_63]]#2 : i64, index, i64
// CHECK:             }
// CHECK:             memref.store %[[VAL_42]], %[[VAL_51]]{{\[}}%[[VAL_35]]] : memref<?xi64>
// CHECK:             return %[[VAL_40]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

   func @vec_to_mat_fixed_csc(%sparse_tensor: tensor<7xf64, #SparseVec64>) -> tensor<7x7xf64, #CSC64> {
       %answer = graphblas.diag %sparse_tensor : tensor<7xf64, #SparseVec64> to tensor<7x7xf64, #CSC64>
       return %answer : tensor<7x7xf64, #CSC64>
   }

// CHECK:           func @mat_to_vec_fixed_csr(%[[VAL_64:.*]]: tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_65:.*]] = constant true
// CHECK:             %[[VAL_66:.*]] = constant 1 : i64
// CHECK:             %[[VAL_67:.*]] = constant 0 : index
// CHECK:             %[[VAL_68:.*]] = constant 1 : index
// CHECK:             %[[VAL_69:.*]] = constant 2 : index
// CHECK:             %[[VAL_70:.*]] = constant 7 : i64
// CHECK:             %[[VAL_71:.*]] = constant 7 : index
// CHECK:             %[[VAL_72:.*]] = sparse_tensor.pointers %[[VAL_64]], %[[VAL_68]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_73:.*]] = sparse_tensor.indices %[[VAL_64]], %[[VAL_68]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_74:.*]] = sparse_tensor.values %[[VAL_64]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_75:.*]] = call @new_vector_i64_p64i64(%[[VAL_71]]) : (index) -> tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_76:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_75]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_dim(%[[VAL_76]], %[[VAL_67]], %[[VAL_71]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_77:.*]] = scf.for %[[VAL_78:.*]] = %[[VAL_67]] to %[[VAL_71]] step %[[VAL_68]] iter_args(%[[VAL_79:.*]] = %[[VAL_67]]) -> (index) {
// CHECK:               %[[VAL_80:.*]] = addi %[[VAL_78]], %[[VAL_68]] : index
// CHECK:               %[[VAL_81:.*]] = memref.load %[[VAL_72]]{{\[}}%[[VAL_78]]] : memref<?xi64>
// CHECK:               %[[VAL_82:.*]] = memref.load %[[VAL_72]]{{\[}}%[[VAL_80]]] : memref<?xi64>
// CHECK:               %[[VAL_83:.*]] = index_cast %[[VAL_81]] : i64 to index
// CHECK:               %[[VAL_84:.*]] = index_cast %[[VAL_82]] : i64 to index
// CHECK:               %[[VAL_85:.*]] = index_cast %[[VAL_78]] : index to i64
// CHECK:               %[[VAL_86:.*]]:2 = scf.while (%[[VAL_87:.*]] = %[[VAL_83]], %[[VAL_88:.*]] = %[[VAL_65]]) : (index, i1) -> (index, i1) {
// CHECK:                 %[[VAL_89:.*]] = cmpi ult, %[[VAL_87]], %[[VAL_84]] : index
// CHECK:                 %[[VAL_90:.*]] = and %[[VAL_88]], %[[VAL_89]] : i1
// CHECK:                 scf.condition(%[[VAL_90]]) %[[VAL_87]], %[[VAL_88]] : index, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_91:.*]]: index, %[[VAL_92:.*]]: i1):
// CHECK:                 %[[VAL_93:.*]] = memref.load %[[VAL_73]]{{\[}}%[[VAL_91]]] : memref<?xi64>
// CHECK:                 %[[VAL_94:.*]] = cmpi ne, %[[VAL_93]], %[[VAL_85]] : i64
// CHECK:                 %[[VAL_95:.*]] = addi %[[VAL_91]], %[[VAL_68]] : index
// CHECK:                 scf.yield %[[VAL_95]], %[[VAL_94]] : index, i1
// CHECK:               }
// CHECK:               %[[VAL_96:.*]] = scf.if %[[VAL_97:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_79]] : index
// CHECK:               } else {
// CHECK:                 %[[VAL_98:.*]] = addi %[[VAL_79]], %[[VAL_68]] : index
// CHECK:                 scf.yield %[[VAL_98]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_99:.*]] : index
// CHECK:             }
// CHECK:             %[[VAL_100:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_75]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_pointers(%[[VAL_100]], %[[VAL_67]], %[[VAL_69]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_101:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_75]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_index(%[[VAL_101]], %[[VAL_67]], %[[VAL_102:.*]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_103:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_75]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_values(%[[VAL_103]], %[[VAL_102]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:             %[[VAL_104:.*]] = sparse_tensor.pointers %[[VAL_75]], %[[VAL_67]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             memref.store %[[VAL_70]], %[[VAL_104]]{{\[}}%[[VAL_68]]] : memref<?xi64>
// CHECK:             %[[VAL_105:.*]] = sparse_tensor.indices %[[VAL_75]], %[[VAL_67]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_106:.*]] = sparse_tensor.values %[[VAL_75]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_107:.*]] = scf.for %[[VAL_108:.*]] = %[[VAL_67]] to %[[VAL_71]] step %[[VAL_68]] iter_args(%[[VAL_109:.*]] = %[[VAL_67]]) -> (index) {
// CHECK:               %[[VAL_110:.*]] = addi %[[VAL_108]], %[[VAL_68]] : index
// CHECK:               %[[VAL_111:.*]] = memref.load %[[VAL_72]]{{\[}}%[[VAL_108]]] : memref<?xi64>
// CHECK:               %[[VAL_112:.*]] = memref.load %[[VAL_72]]{{\[}}%[[VAL_110]]] : memref<?xi64>
// CHECK:               %[[VAL_113:.*]] = index_cast %[[VAL_111]] : i64 to index
// CHECK:               %[[VAL_114:.*]] = index_cast %[[VAL_112]] : i64 to index
// CHECK:               %[[VAL_115:.*]] = index_cast %[[VAL_108]] : index to i64
// CHECK:               %[[VAL_116:.*]]:3 = scf.while (%[[VAL_117:.*]] = %[[VAL_113]], %[[VAL_118:.*]] = %[[VAL_65]], %[[VAL_119:.*]] = %[[VAL_66]]) : (index, i1, i64) -> (index, i1, i64) {
// CHECK:                 %[[VAL_120:.*]] = cmpi ult, %[[VAL_117]], %[[VAL_114]] : index
// CHECK:                 %[[VAL_121:.*]] = and %[[VAL_118]], %[[VAL_120]] : i1
// CHECK:                 scf.condition(%[[VAL_121]]) %[[VAL_117]], %[[VAL_118]], %[[VAL_119]] : index, i1, i64
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_122:.*]]: index, %[[VAL_123:.*]]: i1, %[[VAL_124:.*]]: i64):
// CHECK:                 %[[VAL_125:.*]] = memref.load %[[VAL_73]]{{\[}}%[[VAL_122]]] : memref<?xi64>
// CHECK:                 %[[VAL_126:.*]] = cmpi ne, %[[VAL_125]], %[[VAL_115]] : i64
// CHECK:                 %[[VAL_127:.*]] = scf.if %[[VAL_126]] -> (i64) {
// CHECK:                   scf.yield %[[VAL_124]] : i64
// CHECK:                 } else {
// CHECK:                   %[[VAL_128:.*]] = memref.load %[[VAL_74]]{{\[}}%[[VAL_122]]] : memref<?xi64>
// CHECK:                   scf.yield %[[VAL_128]] : i64
// CHECK:                 }
// CHECK:                 %[[VAL_129:.*]] = addi %[[VAL_122]], %[[VAL_68]] : index
// CHECK:                 scf.yield %[[VAL_129]], %[[VAL_126]], %[[VAL_130:.*]] : index, i1, i64
// CHECK:               }
// CHECK:               %[[VAL_131:.*]] = scf.if %[[VAL_132:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_109]] : index
// CHECK:               } else {
// CHECK:                 memref.store %[[VAL_133:.*]]#2, %[[VAL_106]]{{\[}}%[[VAL_109]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_115]], %[[VAL_105]]{{\[}}%[[VAL_109]]] : memref<?xi64>
// CHECK:                 %[[VAL_134:.*]] = addi %[[VAL_109]], %[[VAL_68]] : index
// CHECK:                 scf.yield %[[VAL_134]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_135:.*]] : index
// CHECK:             }
// CHECK:             return %[[VAL_75]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @mat_to_vec_fixed_csr(%mat: tensor<7x7xi64, #CSC64>) -> tensor<7xi64, #SparseVec64> {
        %vec = graphblas.diag %mat : tensor<7x7xi64, #CSC64> to tensor<7xi64, #SparseVec64>
        return %vec : tensor<7xi64, #SparseVec64>
    }

// CHECK:           func @mat_to_vec_fixed_csc(%[[VAL_136:.*]]: tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_137:.*]] = constant true
// CHECK:             %[[VAL_138:.*]] = constant 1 : i64
// CHECK:             %[[VAL_139:.*]] = constant 0 : index
// CHECK:             %[[VAL_140:.*]] = constant 1 : index
// CHECK:             %[[VAL_141:.*]] = constant 2 : index
// CHECK:             %[[VAL_142:.*]] = constant 7 : i64
// CHECK:             %[[VAL_143:.*]] = constant 7 : index
// CHECK:             %[[VAL_144:.*]] = sparse_tensor.pointers %[[VAL_136]], %[[VAL_140]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_145:.*]] = sparse_tensor.indices %[[VAL_136]], %[[VAL_140]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_146:.*]] = sparse_tensor.values %[[VAL_136]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_147:.*]] = call @new_vector_i64_p64i64(%[[VAL_143]]) : (index) -> tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_148:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_147]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_dim(%[[VAL_148]], %[[VAL_139]], %[[VAL_143]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_149:.*]] = scf.for %[[VAL_150:.*]] = %[[VAL_139]] to %[[VAL_143]] step %[[VAL_140]] iter_args(%[[VAL_151:.*]] = %[[VAL_139]]) -> (index) {
// CHECK:               %[[VAL_152:.*]] = addi %[[VAL_150]], %[[VAL_140]] : index
// CHECK:               %[[VAL_153:.*]] = memref.load %[[VAL_144]]{{\[}}%[[VAL_150]]] : memref<?xi64>
// CHECK:               %[[VAL_154:.*]] = memref.load %[[VAL_144]]{{\[}}%[[VAL_152]]] : memref<?xi64>
// CHECK:               %[[VAL_155:.*]] = index_cast %[[VAL_153]] : i64 to index
// CHECK:               %[[VAL_156:.*]] = index_cast %[[VAL_154]] : i64 to index
// CHECK:               %[[VAL_157:.*]] = index_cast %[[VAL_150]] : index to i64
// CHECK:               %[[VAL_158:.*]]:2 = scf.while (%[[VAL_159:.*]] = %[[VAL_155]], %[[VAL_160:.*]] = %[[VAL_137]]) : (index, i1) -> (index, i1) {
// CHECK:                 %[[VAL_161:.*]] = cmpi ult, %[[VAL_159]], %[[VAL_156]] : index
// CHECK:                 %[[VAL_162:.*]] = and %[[VAL_160]], %[[VAL_161]] : i1
// CHECK:                 scf.condition(%[[VAL_162]]) %[[VAL_159]], %[[VAL_160]] : index, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_163:.*]]: index, %[[VAL_164:.*]]: i1):
// CHECK:                 %[[VAL_165:.*]] = memref.load %[[VAL_145]]{{\[}}%[[VAL_163]]] : memref<?xi64>
// CHECK:                 %[[VAL_166:.*]] = cmpi ne, %[[VAL_165]], %[[VAL_157]] : i64
// CHECK:                 %[[VAL_167:.*]] = addi %[[VAL_163]], %[[VAL_140]] : index
// CHECK:                 scf.yield %[[VAL_167]], %[[VAL_166]] : index, i1
// CHECK:               }
// CHECK:               %[[VAL_168:.*]] = scf.if %[[VAL_169:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_151]] : index
// CHECK:               } else {
// CHECK:                 %[[VAL_170:.*]] = addi %[[VAL_151]], %[[VAL_140]] : index
// CHECK:                 scf.yield %[[VAL_170]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_171:.*]] : index
// CHECK:             }
// CHECK:             %[[VAL_172:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_147]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_pointers(%[[VAL_172]], %[[VAL_139]], %[[VAL_141]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_173:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_147]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_index(%[[VAL_173]], %[[VAL_139]], %[[VAL_174:.*]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_175:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_147]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_values(%[[VAL_175]], %[[VAL_174]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:             %[[VAL_176:.*]] = sparse_tensor.pointers %[[VAL_147]], %[[VAL_139]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             memref.store %[[VAL_142]], %[[VAL_176]]{{\[}}%[[VAL_140]]] : memref<?xi64>
// CHECK:             %[[VAL_177:.*]] = sparse_tensor.indices %[[VAL_147]], %[[VAL_139]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_178:.*]] = sparse_tensor.values %[[VAL_147]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_179:.*]] = scf.for %[[VAL_180:.*]] = %[[VAL_139]] to %[[VAL_143]] step %[[VAL_140]] iter_args(%[[VAL_181:.*]] = %[[VAL_139]]) -> (index) {
// CHECK:               %[[VAL_182:.*]] = addi %[[VAL_180]], %[[VAL_140]] : index
// CHECK:               %[[VAL_183:.*]] = memref.load %[[VAL_144]]{{\[}}%[[VAL_180]]] : memref<?xi64>
// CHECK:               %[[VAL_184:.*]] = memref.load %[[VAL_144]]{{\[}}%[[VAL_182]]] : memref<?xi64>
// CHECK:               %[[VAL_185:.*]] = index_cast %[[VAL_183]] : i64 to index
// CHECK:               %[[VAL_186:.*]] = index_cast %[[VAL_184]] : i64 to index
// CHECK:               %[[VAL_187:.*]] = index_cast %[[VAL_180]] : index to i64
// CHECK:               %[[VAL_188:.*]]:3 = scf.while (%[[VAL_189:.*]] = %[[VAL_185]], %[[VAL_190:.*]] = %[[VAL_137]], %[[VAL_191:.*]] = %[[VAL_138]]) : (index, i1, i64) -> (index, i1, i64) {
// CHECK:                 %[[VAL_192:.*]] = cmpi ult, %[[VAL_189]], %[[VAL_186]] : index
// CHECK:                 %[[VAL_193:.*]] = and %[[VAL_190]], %[[VAL_192]] : i1
// CHECK:                 scf.condition(%[[VAL_193]]) %[[VAL_189]], %[[VAL_190]], %[[VAL_191]] : index, i1, i64
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_194:.*]]: index, %[[VAL_195:.*]]: i1, %[[VAL_196:.*]]: i64):
// CHECK:                 %[[VAL_197:.*]] = memref.load %[[VAL_145]]{{\[}}%[[VAL_194]]] : memref<?xi64>
// CHECK:                 %[[VAL_198:.*]] = cmpi ne, %[[VAL_197]], %[[VAL_187]] : i64
// CHECK:                 %[[VAL_199:.*]] = scf.if %[[VAL_198]] -> (i64) {
// CHECK:                   scf.yield %[[VAL_196]] : i64
// CHECK:                 } else {
// CHECK:                   %[[VAL_200:.*]] = memref.load %[[VAL_146]]{{\[}}%[[VAL_194]]] : memref<?xi64>
// CHECK:                   scf.yield %[[VAL_200]] : i64
// CHECK:                 }
// CHECK:                 %[[VAL_201:.*]] = addi %[[VAL_194]], %[[VAL_140]] : index
// CHECK:                 scf.yield %[[VAL_201]], %[[VAL_198]], %[[VAL_202:.*]] : index, i1, i64
// CHECK:               }
// CHECK:               %[[VAL_203:.*]] = scf.if %[[VAL_204:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_181]] : index
// CHECK:               } else {
// CHECK:                 memref.store %[[VAL_205:.*]]#2, %[[VAL_178]]{{\[}}%[[VAL_181]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_187]], %[[VAL_177]]{{\[}}%[[VAL_181]]] : memref<?xi64>
// CHECK:                 %[[VAL_206:.*]] = addi %[[VAL_181]], %[[VAL_140]] : index
// CHECK:                 scf.yield %[[VAL_206]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_207:.*]] : index
// CHECK:             }
// CHECK:             return %[[VAL_147]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @mat_to_vec_fixed_csc(%mat: tensor<7x7xi64, #CSC64>) -> tensor<7xi64, #SparseVec64> {
        %vec = graphblas.diag %mat : tensor<7x7xi64, #CSC64> to tensor<7xi64, #SparseVec64>
        return %vec : tensor<7xi64, #SparseVec64>
    }

}

module {

// CHECK:           func @vec_to_mat_arbitrary_csr(%[[VAL_0:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_1:.*]] = constant 0 : i64
// CHECK:             %[[VAL_2:.*]] = constant 1 : i64
// CHECK:             %[[VAL_3:.*]] = constant 0 : index
// CHECK:             %[[VAL_4:.*]] = constant 1 : index
// CHECK:             %[[VAL_5:.*]] = tensor.dim %[[VAL_0]], %[[VAL_3]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_6:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_3]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             %[[VAL_8:.*]] = call @new_matrix_csr_f64_p64i64(%[[VAL_5]], %[[VAL_5]]) : (index, index) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_9:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_3]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_10:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_4]]] : memref<?xi64>
// CHECK:             %[[VAL_11:.*]] = index_cast %[[VAL_10]] : i64 to index
// CHECK:             %[[VAL_12:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_8]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_index(%[[VAL_12]], %[[VAL_4]], %[[VAL_11]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_13:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_8]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_values(%[[VAL_13]], %[[VAL_11]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:             %[[VAL_14:.*]] = sparse_tensor.indices %[[VAL_8]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_15:.*]] = sparse_tensor.values %[[VAL_8]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             scf.for %[[VAL_16:.*]] = %[[VAL_3]] to %[[VAL_11]] step %[[VAL_4]] {
// CHECK:               %[[VAL_17:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_16]]] : memref<?xi64>
// CHECK:               memref.store %[[VAL_17]], %[[VAL_14]]{{\[}}%[[VAL_16]]] : memref<?xi64>
// CHECK:               %[[VAL_18:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_16]]] : memref<?xf64>
// CHECK:               memref.store %[[VAL_18]], %[[VAL_15]]{{\[}}%[[VAL_16]]] : memref<?xf64>
// CHECK:             }
// CHECK:             %[[VAL_19:.*]] = sparse_tensor.pointers %[[VAL_8]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_20:.*]]:3 = scf.for %[[VAL_21:.*]] = %[[VAL_3]] to %[[VAL_5]] step %[[VAL_4]] iter_args(%[[VAL_22:.*]] = %[[VAL_1]], %[[VAL_23:.*]] = %[[VAL_3]], %[[VAL_24:.*]] = %[[VAL_1]]) -> (i64, index, i64) {
// CHECK:               memref.store %[[VAL_22]], %[[VAL_19]]{{\[}}%[[VAL_21]]] : memref<?xi64>
// CHECK:               %[[VAL_25:.*]] = index_cast %[[VAL_21]] : index to i64
// CHECK:               %[[VAL_26:.*]] = cmpi eq, %[[VAL_24]], %[[VAL_25]] : i64
// CHECK:               %[[VAL_27:.*]]:3 = scf.if %[[VAL_26]] -> (i64, index, i64) {
// CHECK:                 %[[VAL_28:.*]] = addi %[[VAL_22]], %[[VAL_2]] : i64
// CHECK:                 %[[VAL_29:.*]] = addi %[[VAL_23]], %[[VAL_4]] : index
// CHECK:                 %[[VAL_30:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_29]]] : memref<?xi64>
// CHECK:                 scf.yield %[[VAL_28]], %[[VAL_29]], %[[VAL_30]] : i64, index, i64
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_22]], %[[VAL_23]], %[[VAL_24]] : i64, index, i64
// CHECK:               }
// CHECK:               scf.yield %[[VAL_31:.*]]#0, %[[VAL_31]]#1, %[[VAL_31]]#2 : i64, index, i64
// CHECK:             }
// CHECK:             memref.store %[[VAL_10]], %[[VAL_19]]{{\[}}%[[VAL_5]]] : memref<?xi64>
// CHECK:             return %[[VAL_8]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

   func @vec_to_mat_arbitrary_csr(%sparse_tensor: tensor<?xf64, #SparseVec64>) -> tensor<?x?xf64, #CSR64> {
       %answer = graphblas.diag %sparse_tensor : tensor<?xf64, #SparseVec64> to tensor<?x?xf64, #CSR64>
       return %answer : tensor<?x?xf64, #CSR64>
   }

// CHECK:           func @vec_to_mat_arbitrary_csc(%[[VAL_32:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_33:.*]] = constant 0 : i64
// CHECK:             %[[VAL_34:.*]] = constant 1 : i64
// CHECK:             %[[VAL_35:.*]] = constant 0 : index
// CHECK:             %[[VAL_36:.*]] = constant 1 : index
// CHECK:             %[[VAL_37:.*]] = tensor.dim %[[VAL_32]], %[[VAL_35]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_38:.*]] = sparse_tensor.indices %[[VAL_32]], %[[VAL_35]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_39:.*]] = sparse_tensor.values %[[VAL_32]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             %[[VAL_40:.*]] = call @new_matrix_csc_f64_p64i64(%[[VAL_37]], %[[VAL_37]]) : (index, index) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_41:.*]] = sparse_tensor.pointers %[[VAL_32]], %[[VAL_35]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_42:.*]] = memref.load %[[VAL_41]]{{\[}}%[[VAL_36]]] : memref<?xi64>
// CHECK:             %[[VAL_43:.*]] = index_cast %[[VAL_42]] : i64 to index
// CHECK:             %[[VAL_44:.*]] = call @matrix_csc_f64_p64i64_to_ptr8(%[[VAL_40]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_index(%[[VAL_44]], %[[VAL_36]], %[[VAL_43]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_45:.*]] = call @matrix_csc_f64_p64i64_to_ptr8(%[[VAL_40]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_values(%[[VAL_45]], %[[VAL_43]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:             %[[VAL_46:.*]] = sparse_tensor.indices %[[VAL_40]], %[[VAL_36]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_47:.*]] = sparse_tensor.values %[[VAL_40]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             scf.for %[[VAL_48:.*]] = %[[VAL_35]] to %[[VAL_43]] step %[[VAL_36]] {
// CHECK:               %[[VAL_49:.*]] = memref.load %[[VAL_38]]{{\[}}%[[VAL_48]]] : memref<?xi64>
// CHECK:               memref.store %[[VAL_49]], %[[VAL_46]]{{\[}}%[[VAL_48]]] : memref<?xi64>
// CHECK:               %[[VAL_50:.*]] = memref.load %[[VAL_39]]{{\[}}%[[VAL_48]]] : memref<?xf64>
// CHECK:               memref.store %[[VAL_50]], %[[VAL_47]]{{\[}}%[[VAL_48]]] : memref<?xf64>
// CHECK:             }
// CHECK:             %[[VAL_51:.*]] = sparse_tensor.pointers %[[VAL_40]], %[[VAL_36]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_52:.*]]:3 = scf.for %[[VAL_53:.*]] = %[[VAL_35]] to %[[VAL_37]] step %[[VAL_36]] iter_args(%[[VAL_54:.*]] = %[[VAL_33]], %[[VAL_55:.*]] = %[[VAL_35]], %[[VAL_56:.*]] = %[[VAL_33]]) -> (i64, index, i64) {
// CHECK:               memref.store %[[VAL_54]], %[[VAL_51]]{{\[}}%[[VAL_53]]] : memref<?xi64>
// CHECK:               %[[VAL_57:.*]] = index_cast %[[VAL_53]] : index to i64
// CHECK:               %[[VAL_58:.*]] = cmpi eq, %[[VAL_56]], %[[VAL_57]] : i64
// CHECK:               %[[VAL_59:.*]]:3 = scf.if %[[VAL_58]] -> (i64, index, i64) {
// CHECK:                 %[[VAL_60:.*]] = addi %[[VAL_54]], %[[VAL_34]] : i64
// CHECK:                 %[[VAL_61:.*]] = addi %[[VAL_55]], %[[VAL_36]] : index
// CHECK:                 %[[VAL_62:.*]] = memref.load %[[VAL_38]]{{\[}}%[[VAL_61]]] : memref<?xi64>
// CHECK:                 scf.yield %[[VAL_60]], %[[VAL_61]], %[[VAL_62]] : i64, index, i64
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_54]], %[[VAL_55]], %[[VAL_56]] : i64, index, i64
// CHECK:               }
// CHECK:               scf.yield %[[VAL_63:.*]]#0, %[[VAL_63]]#1, %[[VAL_63]]#2 : i64, index, i64
// CHECK:             }
// CHECK:             memref.store %[[VAL_42]], %[[VAL_51]]{{\[}}%[[VAL_37]]] : memref<?xi64>
// CHECK:             return %[[VAL_40]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

   func @vec_to_mat_arbitrary_csc(%sparse_tensor: tensor<?xf64, #SparseVec64>) -> tensor<?x?xf64, #CSC64> {
       %answer = graphblas.diag %sparse_tensor : tensor<?xf64, #SparseVec64> to tensor<?x?xf64, #CSC64>
       return %answer : tensor<?x?xf64, #CSC64>
   }

// CHECK:           func @mat_to_vec_arbitrary_csr(%[[VAL_64:.*]]: tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_65:.*]] = constant true
// CHECK:             %[[VAL_66:.*]] = constant 1 : i64
// CHECK:             %[[VAL_67:.*]] = constant 1 : index
// CHECK:             %[[VAL_68:.*]] = constant 2 : index
// CHECK:             %[[VAL_69:.*]] = constant 0 : index
// CHECK:             %[[VAL_70:.*]] = tensor.dim %[[VAL_64]], %[[VAL_69]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_71:.*]] = sparse_tensor.pointers %[[VAL_64]], %[[VAL_67]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_72:.*]] = sparse_tensor.indices %[[VAL_64]], %[[VAL_67]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_73:.*]] = sparse_tensor.values %[[VAL_64]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_74:.*]] = call @new_vector_i64_p64i64(%[[VAL_70]]) : (index) -> tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_75:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_74]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_dim(%[[VAL_75]], %[[VAL_69]], %[[VAL_70]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_76:.*]] = scf.for %[[VAL_77:.*]] = %[[VAL_69]] to %[[VAL_70]] step %[[VAL_67]] iter_args(%[[VAL_78:.*]] = %[[VAL_69]]) -> (index) {
// CHECK:               %[[VAL_79:.*]] = addi %[[VAL_77]], %[[VAL_67]] : index
// CHECK:               %[[VAL_80:.*]] = memref.load %[[VAL_71]]{{\[}}%[[VAL_77]]] : memref<?xi64>
// CHECK:               %[[VAL_81:.*]] = memref.load %[[VAL_71]]{{\[}}%[[VAL_79]]] : memref<?xi64>
// CHECK:               %[[VAL_82:.*]] = index_cast %[[VAL_80]] : i64 to index
// CHECK:               %[[VAL_83:.*]] = index_cast %[[VAL_81]] : i64 to index
// CHECK:               %[[VAL_84:.*]] = index_cast %[[VAL_77]] : index to i64
// CHECK:               %[[VAL_85:.*]]:2 = scf.while (%[[VAL_86:.*]] = %[[VAL_82]], %[[VAL_87:.*]] = %[[VAL_65]]) : (index, i1) -> (index, i1) {
// CHECK:                 %[[VAL_88:.*]] = cmpi ult, %[[VAL_86]], %[[VAL_83]] : index
// CHECK:                 %[[VAL_89:.*]] = and %[[VAL_87]], %[[VAL_88]] : i1
// CHECK:                 scf.condition(%[[VAL_89]]) %[[VAL_86]], %[[VAL_87]] : index, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_90:.*]]: index, %[[VAL_91:.*]]: i1):
// CHECK:                 %[[VAL_92:.*]] = memref.load %[[VAL_72]]{{\[}}%[[VAL_90]]] : memref<?xi64>
// CHECK:                 %[[VAL_93:.*]] = cmpi ne, %[[VAL_92]], %[[VAL_84]] : i64
// CHECK:                 %[[VAL_94:.*]] = addi %[[VAL_90]], %[[VAL_67]] : index
// CHECK:                 scf.yield %[[VAL_94]], %[[VAL_93]] : index, i1
// CHECK:               }
// CHECK:               %[[VAL_95:.*]] = scf.if %[[VAL_96:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_78]] : index
// CHECK:               } else {
// CHECK:                 %[[VAL_97:.*]] = addi %[[VAL_78]], %[[VAL_67]] : index
// CHECK:                 scf.yield %[[VAL_97]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_98:.*]] : index
// CHECK:             }
// CHECK:             %[[VAL_99:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_74]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_pointers(%[[VAL_99]], %[[VAL_69]], %[[VAL_68]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_100:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_74]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_index(%[[VAL_100]], %[[VAL_69]], %[[VAL_101:.*]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_102:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_74]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_values(%[[VAL_102]], %[[VAL_101]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:             %[[VAL_103:.*]] = sparse_tensor.pointers %[[VAL_74]], %[[VAL_69]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_104:.*]] = index_cast %[[VAL_70]] : index to i64
// CHECK:             memref.store %[[VAL_104]], %[[VAL_103]]{{\[}}%[[VAL_67]]] : memref<?xi64>
// CHECK:             %[[VAL_105:.*]] = sparse_tensor.indices %[[VAL_74]], %[[VAL_69]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_106:.*]] = sparse_tensor.values %[[VAL_74]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_107:.*]] = scf.for %[[VAL_108:.*]] = %[[VAL_69]] to %[[VAL_70]] step %[[VAL_67]] iter_args(%[[VAL_109:.*]] = %[[VAL_69]]) -> (index) {
// CHECK:               %[[VAL_110:.*]] = addi %[[VAL_108]], %[[VAL_67]] : index
// CHECK:               %[[VAL_111:.*]] = memref.load %[[VAL_71]]{{\[}}%[[VAL_108]]] : memref<?xi64>
// CHECK:               %[[VAL_112:.*]] = memref.load %[[VAL_71]]{{\[}}%[[VAL_110]]] : memref<?xi64>
// CHECK:               %[[VAL_113:.*]] = index_cast %[[VAL_111]] : i64 to index
// CHECK:               %[[VAL_114:.*]] = index_cast %[[VAL_112]] : i64 to index
// CHECK:               %[[VAL_115:.*]] = index_cast %[[VAL_108]] : index to i64
// CHECK:               %[[VAL_116:.*]]:3 = scf.while (%[[VAL_117:.*]] = %[[VAL_113]], %[[VAL_118:.*]] = %[[VAL_65]], %[[VAL_119:.*]] = %[[VAL_66]]) : (index, i1, i64) -> (index, i1, i64) {
// CHECK:                 %[[VAL_120:.*]] = cmpi ult, %[[VAL_117]], %[[VAL_114]] : index
// CHECK:                 %[[VAL_121:.*]] = and %[[VAL_118]], %[[VAL_120]] : i1
// CHECK:                 scf.condition(%[[VAL_121]]) %[[VAL_117]], %[[VAL_118]], %[[VAL_119]] : index, i1, i64
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_122:.*]]: index, %[[VAL_123:.*]]: i1, %[[VAL_124:.*]]: i64):
// CHECK:                 %[[VAL_125:.*]] = memref.load %[[VAL_72]]{{\[}}%[[VAL_122]]] : memref<?xi64>
// CHECK:                 %[[VAL_126:.*]] = cmpi ne, %[[VAL_125]], %[[VAL_115]] : i64
// CHECK:                 %[[VAL_127:.*]] = scf.if %[[VAL_126]] -> (i64) {
// CHECK:                   scf.yield %[[VAL_124]] : i64
// CHECK:                 } else {
// CHECK:                   %[[VAL_128:.*]] = memref.load %[[VAL_73]]{{\[}}%[[VAL_122]]] : memref<?xi64>
// CHECK:                   scf.yield %[[VAL_128]] : i64
// CHECK:                 }
// CHECK:                 %[[VAL_129:.*]] = addi %[[VAL_122]], %[[VAL_67]] : index
// CHECK:                 scf.yield %[[VAL_129]], %[[VAL_126]], %[[VAL_130:.*]] : index, i1, i64
// CHECK:               }
// CHECK:               %[[VAL_131:.*]] = scf.if %[[VAL_132:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_109]] : index
// CHECK:               } else {
// CHECK:                 memref.store %[[VAL_133:.*]]#2, %[[VAL_106]]{{\[}}%[[VAL_109]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_115]], %[[VAL_105]]{{\[}}%[[VAL_109]]] : memref<?xi64>
// CHECK:                 %[[VAL_134:.*]] = addi %[[VAL_109]], %[[VAL_67]] : index
// CHECK:                 scf.yield %[[VAL_134]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_135:.*]] : index
// CHECK:             }
// CHECK:             return %[[VAL_74]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @mat_to_vec_arbitrary_csr(%mat: tensor<?x?xi64, #CSC64>) -> tensor<?xi64, #SparseVec64> {
        %vec = graphblas.diag %mat : tensor<?x?xi64, #CSC64> to tensor<?xi64, #SparseVec64>
        return %vec : tensor<?xi64, #SparseVec64>
    }

// CHECK:           func @mat_to_vec_arbitrary_csc(%[[VAL_136:.*]]: tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_137:.*]] = constant true
// CHECK:             %[[VAL_138:.*]] = constant 1 : i64
// CHECK:             %[[VAL_139:.*]] = constant 1 : index
// CHECK:             %[[VAL_140:.*]] = constant 2 : index
// CHECK:             %[[VAL_141:.*]] = constant 0 : index
// CHECK:             %[[VAL_142:.*]] = tensor.dim %[[VAL_136]], %[[VAL_141]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_143:.*]] = sparse_tensor.pointers %[[VAL_136]], %[[VAL_139]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_144:.*]] = sparse_tensor.indices %[[VAL_136]], %[[VAL_139]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_145:.*]] = sparse_tensor.values %[[VAL_136]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_146:.*]] = call @new_vector_i64_p64i64(%[[VAL_142]]) : (index) -> tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_147:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_146]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_dim(%[[VAL_147]], %[[VAL_141]], %[[VAL_142]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_148:.*]] = scf.for %[[VAL_149:.*]] = %[[VAL_141]] to %[[VAL_142]] step %[[VAL_139]] iter_args(%[[VAL_150:.*]] = %[[VAL_141]]) -> (index) {
// CHECK:               %[[VAL_151:.*]] = addi %[[VAL_149]], %[[VAL_139]] : index
// CHECK:               %[[VAL_152:.*]] = memref.load %[[VAL_143]]{{\[}}%[[VAL_149]]] : memref<?xi64>
// CHECK:               %[[VAL_153:.*]] = memref.load %[[VAL_143]]{{\[}}%[[VAL_151]]] : memref<?xi64>
// CHECK:               %[[VAL_154:.*]] = index_cast %[[VAL_152]] : i64 to index
// CHECK:               %[[VAL_155:.*]] = index_cast %[[VAL_153]] : i64 to index
// CHECK:               %[[VAL_156:.*]] = index_cast %[[VAL_149]] : index to i64
// CHECK:               %[[VAL_157:.*]]:2 = scf.while (%[[VAL_158:.*]] = %[[VAL_154]], %[[VAL_159:.*]] = %[[VAL_137]]) : (index, i1) -> (index, i1) {
// CHECK:                 %[[VAL_160:.*]] = cmpi ult, %[[VAL_158]], %[[VAL_155]] : index
// CHECK:                 %[[VAL_161:.*]] = and %[[VAL_159]], %[[VAL_160]] : i1
// CHECK:                 scf.condition(%[[VAL_161]]) %[[VAL_158]], %[[VAL_159]] : index, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_162:.*]]: index, %[[VAL_163:.*]]: i1):
// CHECK:                 %[[VAL_164:.*]] = memref.load %[[VAL_144]]{{\[}}%[[VAL_162]]] : memref<?xi64>
// CHECK:                 %[[VAL_165:.*]] = cmpi ne, %[[VAL_164]], %[[VAL_156]] : i64
// CHECK:                 %[[VAL_166:.*]] = addi %[[VAL_162]], %[[VAL_139]] : index
// CHECK:                 scf.yield %[[VAL_166]], %[[VAL_165]] : index, i1
// CHECK:               }
// CHECK:               %[[VAL_167:.*]] = scf.if %[[VAL_168:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_150]] : index
// CHECK:               } else {
// CHECK:                 %[[VAL_169:.*]] = addi %[[VAL_150]], %[[VAL_139]] : index
// CHECK:                 scf.yield %[[VAL_169]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_170:.*]] : index
// CHECK:             }
// CHECK:             %[[VAL_171:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_146]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_pointers(%[[VAL_171]], %[[VAL_141]], %[[VAL_140]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_172:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_146]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_index(%[[VAL_172]], %[[VAL_141]], %[[VAL_173:.*]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_174:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_146]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_values(%[[VAL_174]], %[[VAL_173]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:             %[[VAL_175:.*]] = sparse_tensor.pointers %[[VAL_146]], %[[VAL_141]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_176:.*]] = index_cast %[[VAL_142]] : index to i64
// CHECK:             memref.store %[[VAL_176]], %[[VAL_175]]{{\[}}%[[VAL_139]]] : memref<?xi64>
// CHECK:             %[[VAL_177:.*]] = sparse_tensor.indices %[[VAL_146]], %[[VAL_141]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_178:.*]] = sparse_tensor.values %[[VAL_146]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_179:.*]] = scf.for %[[VAL_180:.*]] = %[[VAL_141]] to %[[VAL_142]] step %[[VAL_139]] iter_args(%[[VAL_181:.*]] = %[[VAL_141]]) -> (index) {
// CHECK:               %[[VAL_182:.*]] = addi %[[VAL_180]], %[[VAL_139]] : index
// CHECK:               %[[VAL_183:.*]] = memref.load %[[VAL_143]]{{\[}}%[[VAL_180]]] : memref<?xi64>
// CHECK:               %[[VAL_184:.*]] = memref.load %[[VAL_143]]{{\[}}%[[VAL_182]]] : memref<?xi64>
// CHECK:               %[[VAL_185:.*]] = index_cast %[[VAL_183]] : i64 to index
// CHECK:               %[[VAL_186:.*]] = index_cast %[[VAL_184]] : i64 to index
// CHECK:               %[[VAL_187:.*]] = index_cast %[[VAL_180]] : index to i64
// CHECK:               %[[VAL_188:.*]]:3 = scf.while (%[[VAL_189:.*]] = %[[VAL_185]], %[[VAL_190:.*]] = %[[VAL_137]], %[[VAL_191:.*]] = %[[VAL_138]]) : (index, i1, i64) -> (index, i1, i64) {
// CHECK:                 %[[VAL_192:.*]] = cmpi ult, %[[VAL_189]], %[[VAL_186]] : index
// CHECK:                 %[[VAL_193:.*]] = and %[[VAL_190]], %[[VAL_192]] : i1
// CHECK:                 scf.condition(%[[VAL_193]]) %[[VAL_189]], %[[VAL_190]], %[[VAL_191]] : index, i1, i64
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_194:.*]]: index, %[[VAL_195:.*]]: i1, %[[VAL_196:.*]]: i64):
// CHECK:                 %[[VAL_197:.*]] = memref.load %[[VAL_144]]{{\[}}%[[VAL_194]]] : memref<?xi64>
// CHECK:                 %[[VAL_198:.*]] = cmpi ne, %[[VAL_197]], %[[VAL_187]] : i64
// CHECK:                 %[[VAL_199:.*]] = scf.if %[[VAL_198]] -> (i64) {
// CHECK:                   scf.yield %[[VAL_196]] : i64
// CHECK:                 } else {
// CHECK:                   %[[VAL_200:.*]] = memref.load %[[VAL_145]]{{\[}}%[[VAL_194]]] : memref<?xi64>
// CHECK:                   scf.yield %[[VAL_200]] : i64
// CHECK:                 }
// CHECK:                 %[[VAL_201:.*]] = addi %[[VAL_194]], %[[VAL_139]] : index
// CHECK:                 scf.yield %[[VAL_201]], %[[VAL_198]], %[[VAL_202:.*]] : index, i1, i64
// CHECK:               }
// CHECK:               %[[VAL_203:.*]] = scf.if %[[VAL_204:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_181]] : index
// CHECK:               } else {
// CHECK:                 memref.store %[[VAL_205:.*]]#2, %[[VAL_178]]{{\[}}%[[VAL_181]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_187]], %[[VAL_177]]{{\[}}%[[VAL_181]]] : memref<?xi64>
// CHECK:                 %[[VAL_206:.*]] = addi %[[VAL_181]], %[[VAL_139]] : index
// CHECK:                 scf.yield %[[VAL_206]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_207:.*]] : index
// CHECK:             }
// CHECK:             return %[[VAL_146]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @mat_to_vec_arbitrary_csc(%mat: tensor<?x?xi64, #CSC64>) -> tensor<?xi64, #SparseVec64> {
        %vec = graphblas.diag %mat : tensor<?x?xi64, #CSC64> to tensor<?xi64, #SparseVec64>
        return %vec : tensor<?xi64, #SparseVec64>
    }

}
