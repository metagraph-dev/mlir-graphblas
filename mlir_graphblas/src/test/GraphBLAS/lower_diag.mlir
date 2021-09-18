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

// CHECK:           builtin.func @vec_to_mat_fixed_csr(%[[VAL_0:.*]]: tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_1:.*]] = constant 0 : i64
// CHECK:             %[[VAL_2:.*]] = constant 1 : i64
// CHECK:             %[[VAL_3:.*]] = constant 6 : index
// CHECK:             %[[VAL_4:.*]] = constant 7 : index
// CHECK:             %[[VAL_5:.*]] = constant 0 : index
// CHECK:             %[[VAL_6:.*]] = constant 1 : index
// CHECK:             %[[VAL_7:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_5]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_8:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             %[[VAL_9:.*]] = call @new_matrix_csr_f64_p64i64(%[[VAL_4]], %[[VAL_4]]) : (index, index) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_10:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_5]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_11:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_6]]] : memref<?xi64>
// CHECK:             %[[VAL_12:.*]] = index_cast %[[VAL_11]] : i64 to index
// CHECK:             %[[VAL_13:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_9]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_index(%[[VAL_13]], %[[VAL_6]], %[[VAL_12]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_14:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_9]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_values(%[[VAL_14]], %[[VAL_12]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:             %[[VAL_15:.*]] = sparse_tensor.indices %[[VAL_9]], %[[VAL_6]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_16:.*]] = sparse_tensor.values %[[VAL_9]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             scf.for %[[VAL_17:.*]] = %[[VAL_5]] to %[[VAL_12]] step %[[VAL_6]] {
// CHECK:               %[[VAL_18:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_17]]] : memref<?xi64>
// CHECK:               memref.store %[[VAL_18]], %[[VAL_15]]{{\[}}%[[VAL_17]]] : memref<?xi64>
// CHECK:               %[[VAL_19:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_17]]] : memref<?xf64>
// CHECK:               memref.store %[[VAL_19]], %[[VAL_16]]{{\[}}%[[VAL_17]]] : memref<?xf64>
// CHECK:             }
// CHECK:             %[[VAL_20:.*]] = sparse_tensor.pointers %[[VAL_9]], %[[VAL_6]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_21:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_5]]] : memref<?xi64>
// CHECK:             %[[VAL_22:.*]]:3 = scf.for %[[VAL_23:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_6]] iter_args(%[[VAL_24:.*]] = %[[VAL_1]], %[[VAL_25:.*]] = %[[VAL_5]], %[[VAL_26:.*]] = %[[VAL_21]]) -> (i64, index, i64) {
// CHECK:               memref.store %[[VAL_24]], %[[VAL_20]]{{\[}}%[[VAL_23]]] : memref<?xi64>
// CHECK:               %[[VAL_27:.*]] = index_cast %[[VAL_23]] : index to i64
// CHECK:               %[[VAL_28:.*]] = cmpi eq, %[[VAL_26]], %[[VAL_27]] : i64
// CHECK:               %[[VAL_29:.*]] = cmpi ne, %[[VAL_23]], %[[VAL_3]] : index
// CHECK:               %[[VAL_30:.*]] = and %[[VAL_29]], %[[VAL_28]] : i1
// CHECK:               %[[VAL_31:.*]]:3 = scf.if %[[VAL_30]] -> (i64, index, i64) {
// CHECK:                 %[[VAL_32:.*]] = addi %[[VAL_24]], %[[VAL_2]] : i64
// CHECK:                 %[[VAL_33:.*]] = addi %[[VAL_25]], %[[VAL_6]] : index
// CHECK:                 %[[VAL_34:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_33]]] : memref<?xi64>
// CHECK:                 scf.yield %[[VAL_32]], %[[VAL_33]], %[[VAL_34]] : i64, index, i64
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_24]], %[[VAL_25]], %[[VAL_26]] : i64, index, i64
// CHECK:               }
// CHECK:               scf.yield %[[VAL_35:.*]]#0, %[[VAL_35]]#1, %[[VAL_35]]#2 : i64, index, i64
// CHECK:             }
// CHECK:             memref.store %[[VAL_11]], %[[VAL_20]]{{\[}}%[[VAL_4]]] : memref<?xi64>
// CHECK:             return %[[VAL_9]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

   func @vec_to_mat_fixed_csr(%sparse_tensor: tensor<7xf64, #SparseVec64>) -> tensor<7x7xf64, #CSR64> {
       %answer = graphblas.diag %sparse_tensor : tensor<7xf64, #SparseVec64> to tensor<7x7xf64, #CSR64>
       return %answer : tensor<7x7xf64, #CSR64>
   }

// CHECK:           builtin.func @vec_to_mat_fixed_csc(%[[VAL_36:.*]]: tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_37:.*]] = constant 0 : i64
// CHECK:             %[[VAL_38:.*]] = constant 1 : i64
// CHECK:             %[[VAL_39:.*]] = constant 6 : index
// CHECK:             %[[VAL_40:.*]] = constant 7 : index
// CHECK:             %[[VAL_41:.*]] = constant 0 : index
// CHECK:             %[[VAL_42:.*]] = constant 1 : index
// CHECK:             %[[VAL_43:.*]] = sparse_tensor.indices %[[VAL_36]], %[[VAL_41]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_44:.*]] = sparse_tensor.values %[[VAL_36]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             %[[VAL_45:.*]] = call @new_matrix_csc_f64_p64i64(%[[VAL_40]], %[[VAL_40]]) : (index, index) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_46:.*]] = sparse_tensor.pointers %[[VAL_36]], %[[VAL_41]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_47:.*]] = memref.load %[[VAL_46]]{{\[}}%[[VAL_42]]] : memref<?xi64>
// CHECK:             %[[VAL_48:.*]] = index_cast %[[VAL_47]] : i64 to index
// CHECK:             %[[VAL_49:.*]] = call @matrix_csc_f64_p64i64_to_ptr8(%[[VAL_45]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_index(%[[VAL_49]], %[[VAL_42]], %[[VAL_48]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_50:.*]] = call @matrix_csc_f64_p64i64_to_ptr8(%[[VAL_45]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_values(%[[VAL_50]], %[[VAL_48]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:             %[[VAL_51:.*]] = sparse_tensor.indices %[[VAL_45]], %[[VAL_42]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_52:.*]] = sparse_tensor.values %[[VAL_45]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             scf.for %[[VAL_53:.*]] = %[[VAL_41]] to %[[VAL_48]] step %[[VAL_42]] {
// CHECK:               %[[VAL_54:.*]] = memref.load %[[VAL_43]]{{\[}}%[[VAL_53]]] : memref<?xi64>
// CHECK:               memref.store %[[VAL_54]], %[[VAL_51]]{{\[}}%[[VAL_53]]] : memref<?xi64>
// CHECK:               %[[VAL_55:.*]] = memref.load %[[VAL_44]]{{\[}}%[[VAL_53]]] : memref<?xf64>
// CHECK:               memref.store %[[VAL_55]], %[[VAL_52]]{{\[}}%[[VAL_53]]] : memref<?xf64>
// CHECK:             }
// CHECK:             %[[VAL_56:.*]] = sparse_tensor.pointers %[[VAL_45]], %[[VAL_42]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_57:.*]] = memref.load %[[VAL_43]]{{\[}}%[[VAL_41]]] : memref<?xi64>
// CHECK:             %[[VAL_58:.*]]:3 = scf.for %[[VAL_59:.*]] = %[[VAL_41]] to %[[VAL_40]] step %[[VAL_42]] iter_args(%[[VAL_60:.*]] = %[[VAL_37]], %[[VAL_61:.*]] = %[[VAL_41]], %[[VAL_62:.*]] = %[[VAL_57]]) -> (i64, index, i64) {
// CHECK:               memref.store %[[VAL_60]], %[[VAL_56]]{{\[}}%[[VAL_59]]] : memref<?xi64>
// CHECK:               %[[VAL_63:.*]] = index_cast %[[VAL_59]] : index to i64
// CHECK:               %[[VAL_64:.*]] = cmpi eq, %[[VAL_62]], %[[VAL_63]] : i64
// CHECK:               %[[VAL_65:.*]] = cmpi ne, %[[VAL_59]], %[[VAL_39]] : index
// CHECK:               %[[VAL_66:.*]] = and %[[VAL_65]], %[[VAL_64]] : i1
// CHECK:               %[[VAL_67:.*]]:3 = scf.if %[[VAL_66]] -> (i64, index, i64) {
// CHECK:                 %[[VAL_68:.*]] = addi %[[VAL_60]], %[[VAL_38]] : i64
// CHECK:                 %[[VAL_69:.*]] = addi %[[VAL_61]], %[[VAL_42]] : index
// CHECK:                 %[[VAL_70:.*]] = memref.load %[[VAL_43]]{{\[}}%[[VAL_69]]] : memref<?xi64>
// CHECK:                 scf.yield %[[VAL_68]], %[[VAL_69]], %[[VAL_70]] : i64, index, i64
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_60]], %[[VAL_61]], %[[VAL_62]] : i64, index, i64
// CHECK:               }
// CHECK:               scf.yield %[[VAL_71:.*]]#0, %[[VAL_71]]#1, %[[VAL_71]]#2 : i64, index, i64
// CHECK:             }
// CHECK:             memref.store %[[VAL_47]], %[[VAL_56]]{{\[}}%[[VAL_40]]] : memref<?xi64>
// CHECK:             return %[[VAL_45]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

   func @vec_to_mat_fixed_csc(%sparse_tensor: tensor<7xf64, #SparseVec64>) -> tensor<7x7xf64, #CSC64> {
       %answer = graphblas.diag %sparse_tensor : tensor<7xf64, #SparseVec64> to tensor<7x7xf64, #CSC64>
       return %answer : tensor<7x7xf64, #CSC64>
   }

// CHECK:           builtin.func @mat_to_vec_fixed_csr(%[[VAL_72:.*]]: tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_73:.*]] = constant true
// CHECK:             %[[VAL_74:.*]] = constant 1 : i64
// CHECK:             %[[VAL_75:.*]] = constant 0 : index
// CHECK:             %[[VAL_76:.*]] = constant 1 : index
// CHECK:             %[[VAL_77:.*]] = constant 2 : index
// CHECK:             %[[VAL_78:.*]] = constant 7 : index
// CHECK:             %[[VAL_79:.*]] = sparse_tensor.pointers %[[VAL_72]], %[[VAL_76]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_80:.*]] = sparse_tensor.indices %[[VAL_72]], %[[VAL_76]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_81:.*]] = sparse_tensor.values %[[VAL_72]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_82:.*]] = call @new_vector_i64_p64i64(%[[VAL_78]]) : (index) -> tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_83:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_82]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_dim(%[[VAL_83]], %[[VAL_75]], %[[VAL_78]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_84:.*]] = scf.for %[[VAL_85:.*]] = %[[VAL_75]] to %[[VAL_78]] step %[[VAL_76]] iter_args(%[[VAL_86:.*]] = %[[VAL_75]]) -> (index) {
// CHECK:               %[[VAL_87:.*]] = addi %[[VAL_85]], %[[VAL_76]] : index
// CHECK:               %[[VAL_88:.*]] = memref.load %[[VAL_79]]{{\[}}%[[VAL_85]]] : memref<?xi64>
// CHECK:               %[[VAL_89:.*]] = memref.load %[[VAL_79]]{{\[}}%[[VAL_87]]] : memref<?xi64>
// CHECK:               %[[VAL_90:.*]] = index_cast %[[VAL_88]] : i64 to index
// CHECK:               %[[VAL_91:.*]] = index_cast %[[VAL_89]] : i64 to index
// CHECK:               %[[VAL_92:.*]] = index_cast %[[VAL_85]] : index to i64
// CHECK:               %[[VAL_93:.*]]:2 = scf.while (%[[VAL_94:.*]] = %[[VAL_90]], %[[VAL_95:.*]] = %[[VAL_73]]) : (index, i1) -> (index, i1) {
// CHECK:                 %[[VAL_96:.*]] = cmpi ult, %[[VAL_94]], %[[VAL_91]] : index
// CHECK:                 %[[VAL_97:.*]] = and %[[VAL_95]], %[[VAL_96]] : i1
// CHECK:                 scf.condition(%[[VAL_97]]) %[[VAL_94]], %[[VAL_95]] : index, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_98:.*]]: index, %[[VAL_99:.*]]: i1):
// CHECK:                 %[[VAL_100:.*]] = memref.load %[[VAL_80]]{{\[}}%[[VAL_98]]] : memref<?xi64>
// CHECK:                 %[[VAL_101:.*]] = cmpi ne, %[[VAL_100]], %[[VAL_92]] : i64
// CHECK:                 %[[VAL_102:.*]] = addi %[[VAL_98]], %[[VAL_76]] : index
// CHECK:                 scf.yield %[[VAL_102]], %[[VAL_101]] : index, i1
// CHECK:               }
// CHECK:               %[[VAL_103:.*]] = scf.if %[[VAL_104:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_86]] : index
// CHECK:               } else {
// CHECK:                 %[[VAL_105:.*]] = addi %[[VAL_86]], %[[VAL_76]] : index
// CHECK:                 scf.yield %[[VAL_105]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_106:.*]] : index
// CHECK:             }
// CHECK:             %[[VAL_107:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_82]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_pointers(%[[VAL_107]], %[[VAL_75]], %[[VAL_77]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_108:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_82]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_index(%[[VAL_108]], %[[VAL_75]], %[[VAL_109:.*]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_110:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_82]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_values(%[[VAL_110]], %[[VAL_109]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:             %[[VAL_111:.*]] = sparse_tensor.pointers %[[VAL_82]], %[[VAL_75]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_112:.*]] = index_cast %[[VAL_109]] : index to i64
// CHECK:             memref.store %[[VAL_112]], %[[VAL_111]]{{\[}}%[[VAL_76]]] : memref<?xi64>
// CHECK:             %[[VAL_113:.*]] = sparse_tensor.indices %[[VAL_82]], %[[VAL_75]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_114:.*]] = sparse_tensor.values %[[VAL_82]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_115:.*]] = scf.for %[[VAL_116:.*]] = %[[VAL_75]] to %[[VAL_78]] step %[[VAL_76]] iter_args(%[[VAL_117:.*]] = %[[VAL_75]]) -> (index) {
// CHECK:               %[[VAL_118:.*]] = addi %[[VAL_116]], %[[VAL_76]] : index
// CHECK:               %[[VAL_119:.*]] = memref.load %[[VAL_79]]{{\[}}%[[VAL_116]]] : memref<?xi64>
// CHECK:               %[[VAL_120:.*]] = memref.load %[[VAL_79]]{{\[}}%[[VAL_118]]] : memref<?xi64>
// CHECK:               %[[VAL_121:.*]] = index_cast %[[VAL_119]] : i64 to index
// CHECK:               %[[VAL_122:.*]] = index_cast %[[VAL_120]] : i64 to index
// CHECK:               %[[VAL_123:.*]] = index_cast %[[VAL_116]] : index to i64
// CHECK:               %[[VAL_124:.*]]:3 = scf.while (%[[VAL_125:.*]] = %[[VAL_121]], %[[VAL_126:.*]] = %[[VAL_73]], %[[VAL_127:.*]] = %[[VAL_74]]) : (index, i1, i64) -> (index, i1, i64) {
// CHECK:                 %[[VAL_128:.*]] = cmpi ult, %[[VAL_125]], %[[VAL_122]] : index
// CHECK:                 %[[VAL_129:.*]] = and %[[VAL_126]], %[[VAL_128]] : i1
// CHECK:                 scf.condition(%[[VAL_129]]) %[[VAL_125]], %[[VAL_126]], %[[VAL_127]] : index, i1, i64
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_130:.*]]: index, %[[VAL_131:.*]]: i1, %[[VAL_132:.*]]: i64):
// CHECK:                 %[[VAL_133:.*]] = memref.load %[[VAL_80]]{{\[}}%[[VAL_130]]] : memref<?xi64>
// CHECK:                 %[[VAL_134:.*]] = cmpi ne, %[[VAL_133]], %[[VAL_123]] : i64
// CHECK:                 %[[VAL_135:.*]] = scf.if %[[VAL_134]] -> (i64) {
// CHECK:                   scf.yield %[[VAL_132]] : i64
// CHECK:                 } else {
// CHECK:                   %[[VAL_136:.*]] = memref.load %[[VAL_81]]{{\[}}%[[VAL_130]]] : memref<?xi64>
// CHECK:                   scf.yield %[[VAL_136]] : i64
// CHECK:                 }
// CHECK:                 %[[VAL_137:.*]] = addi %[[VAL_130]], %[[VAL_76]] : index
// CHECK:                 scf.yield %[[VAL_137]], %[[VAL_134]], %[[VAL_138:.*]] : index, i1, i64
// CHECK:               }
// CHECK:               %[[VAL_139:.*]] = scf.if %[[VAL_140:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_117]] : index
// CHECK:               } else {
// CHECK:                 memref.store %[[VAL_141:.*]]#2, %[[VAL_114]]{{\[}}%[[VAL_117]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_123]], %[[VAL_113]]{{\[}}%[[VAL_117]]] : memref<?xi64>
// CHECK:                 %[[VAL_142:.*]] = addi %[[VAL_117]], %[[VAL_76]] : index
// CHECK:                 scf.yield %[[VAL_142]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_143:.*]] : index
// CHECK:             }
// CHECK:             return %[[VAL_82]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @mat_to_vec_fixed_csr(%mat: tensor<7x7xi64, #CSC64>) -> tensor<7xi64, #SparseVec64> {
        %vec = graphblas.diag %mat : tensor<7x7xi64, #CSC64> to tensor<7xi64, #SparseVec64>
        return %vec : tensor<7xi64, #SparseVec64>
    }

// CHECK:           builtin.func @mat_to_vec_fixed_csc(%[[VAL_144:.*]]: tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_145:.*]] = constant true
// CHECK:             %[[VAL_146:.*]] = constant 1 : i64
// CHECK:             %[[VAL_147:.*]] = constant 0 : index
// CHECK:             %[[VAL_148:.*]] = constant 1 : index
// CHECK:             %[[VAL_149:.*]] = constant 2 : index
// CHECK:             %[[VAL_150:.*]] = constant 7 : index
// CHECK:             %[[VAL_151:.*]] = sparse_tensor.pointers %[[VAL_144]], %[[VAL_148]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_152:.*]] = sparse_tensor.indices %[[VAL_144]], %[[VAL_148]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_153:.*]] = sparse_tensor.values %[[VAL_144]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_154:.*]] = call @new_vector_i64_p64i64(%[[VAL_150]]) : (index) -> tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_155:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_154]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_dim(%[[VAL_155]], %[[VAL_147]], %[[VAL_150]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_156:.*]] = scf.for %[[VAL_157:.*]] = %[[VAL_147]] to %[[VAL_150]] step %[[VAL_148]] iter_args(%[[VAL_158:.*]] = %[[VAL_147]]) -> (index) {
// CHECK:               %[[VAL_159:.*]] = addi %[[VAL_157]], %[[VAL_148]] : index
// CHECK:               %[[VAL_160:.*]] = memref.load %[[VAL_151]]{{\[}}%[[VAL_157]]] : memref<?xi64>
// CHECK:               %[[VAL_161:.*]] = memref.load %[[VAL_151]]{{\[}}%[[VAL_159]]] : memref<?xi64>
// CHECK:               %[[VAL_162:.*]] = index_cast %[[VAL_160]] : i64 to index
// CHECK:               %[[VAL_163:.*]] = index_cast %[[VAL_161]] : i64 to index
// CHECK:               %[[VAL_164:.*]] = index_cast %[[VAL_157]] : index to i64
// CHECK:               %[[VAL_165:.*]]:2 = scf.while (%[[VAL_166:.*]] = %[[VAL_162]], %[[VAL_167:.*]] = %[[VAL_145]]) : (index, i1) -> (index, i1) {
// CHECK:                 %[[VAL_168:.*]] = cmpi ult, %[[VAL_166]], %[[VAL_163]] : index
// CHECK:                 %[[VAL_169:.*]] = and %[[VAL_167]], %[[VAL_168]] : i1
// CHECK:                 scf.condition(%[[VAL_169]]) %[[VAL_166]], %[[VAL_167]] : index, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_170:.*]]: index, %[[VAL_171:.*]]: i1):
// CHECK:                 %[[VAL_172:.*]] = memref.load %[[VAL_152]]{{\[}}%[[VAL_170]]] : memref<?xi64>
// CHECK:                 %[[VAL_173:.*]] = cmpi ne, %[[VAL_172]], %[[VAL_164]] : i64
// CHECK:                 %[[VAL_174:.*]] = addi %[[VAL_170]], %[[VAL_148]] : index
// CHECK:                 scf.yield %[[VAL_174]], %[[VAL_173]] : index, i1
// CHECK:               }
// CHECK:               %[[VAL_175:.*]] = scf.if %[[VAL_176:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_158]] : index
// CHECK:               } else {
// CHECK:                 %[[VAL_177:.*]] = addi %[[VAL_158]], %[[VAL_148]] : index
// CHECK:                 scf.yield %[[VAL_177]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_178:.*]] : index
// CHECK:             }
// CHECK:             %[[VAL_179:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_154]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_pointers(%[[VAL_179]], %[[VAL_147]], %[[VAL_149]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_180:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_154]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_index(%[[VAL_180]], %[[VAL_147]], %[[VAL_181:.*]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_182:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_154]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_values(%[[VAL_182]], %[[VAL_181]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:             %[[VAL_183:.*]] = sparse_tensor.pointers %[[VAL_154]], %[[VAL_147]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_184:.*]] = index_cast %[[VAL_181]] : index to i64
// CHECK:             memref.store %[[VAL_184]], %[[VAL_183]]{{\[}}%[[VAL_148]]] : memref<?xi64>
// CHECK:             %[[VAL_185:.*]] = sparse_tensor.indices %[[VAL_154]], %[[VAL_147]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_186:.*]] = sparse_tensor.values %[[VAL_154]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_187:.*]] = scf.for %[[VAL_188:.*]] = %[[VAL_147]] to %[[VAL_150]] step %[[VAL_148]] iter_args(%[[VAL_189:.*]] = %[[VAL_147]]) -> (index) {
// CHECK:               %[[VAL_190:.*]] = addi %[[VAL_188]], %[[VAL_148]] : index
// CHECK:               %[[VAL_191:.*]] = memref.load %[[VAL_151]]{{\[}}%[[VAL_188]]] : memref<?xi64>
// CHECK:               %[[VAL_192:.*]] = memref.load %[[VAL_151]]{{\[}}%[[VAL_190]]] : memref<?xi64>
// CHECK:               %[[VAL_193:.*]] = index_cast %[[VAL_191]] : i64 to index
// CHECK:               %[[VAL_194:.*]] = index_cast %[[VAL_192]] : i64 to index
// CHECK:               %[[VAL_195:.*]] = index_cast %[[VAL_188]] : index to i64
// CHECK:               %[[VAL_196:.*]]:3 = scf.while (%[[VAL_197:.*]] = %[[VAL_193]], %[[VAL_198:.*]] = %[[VAL_145]], %[[VAL_199:.*]] = %[[VAL_146]]) : (index, i1, i64) -> (index, i1, i64) {
// CHECK:                 %[[VAL_200:.*]] = cmpi ult, %[[VAL_197]], %[[VAL_194]] : index
// CHECK:                 %[[VAL_201:.*]] = and %[[VAL_198]], %[[VAL_200]] : i1
// CHECK:                 scf.condition(%[[VAL_201]]) %[[VAL_197]], %[[VAL_198]], %[[VAL_199]] : index, i1, i64
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_202:.*]]: index, %[[VAL_203:.*]]: i1, %[[VAL_204:.*]]: i64):
// CHECK:                 %[[VAL_205:.*]] = memref.load %[[VAL_152]]{{\[}}%[[VAL_202]]] : memref<?xi64>
// CHECK:                 %[[VAL_206:.*]] = cmpi ne, %[[VAL_205]], %[[VAL_195]] : i64
// CHECK:                 %[[VAL_207:.*]] = scf.if %[[VAL_206]] -> (i64) {
// CHECK:                   scf.yield %[[VAL_204]] : i64
// CHECK:                 } else {
// CHECK:                   %[[VAL_208:.*]] = memref.load %[[VAL_153]]{{\[}}%[[VAL_202]]] : memref<?xi64>
// CHECK:                   scf.yield %[[VAL_208]] : i64
// CHECK:                 }
// CHECK:                 %[[VAL_209:.*]] = addi %[[VAL_202]], %[[VAL_148]] : index
// CHECK:                 scf.yield %[[VAL_209]], %[[VAL_206]], %[[VAL_210:.*]] : index, i1, i64
// CHECK:               }
// CHECK:               %[[VAL_211:.*]] = scf.if %[[VAL_212:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_189]] : index
// CHECK:               } else {
// CHECK:                 memref.store %[[VAL_213:.*]]#2, %[[VAL_186]]{{\[}}%[[VAL_189]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_195]], %[[VAL_185]]{{\[}}%[[VAL_189]]] : memref<?xi64>
// CHECK:                 %[[VAL_214:.*]] = addi %[[VAL_189]], %[[VAL_148]] : index
// CHECK:                 scf.yield %[[VAL_214]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_215:.*]] : index
// CHECK:             }
// CHECK:             return %[[VAL_154]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @mat_to_vec_fixed_csc(%mat: tensor<7x7xi64, #CSC64>) -> tensor<7xi64, #SparseVec64> {
        %vec = graphblas.diag %mat : tensor<7x7xi64, #CSC64> to tensor<7xi64, #SparseVec64>
        return %vec : tensor<7xi64, #SparseVec64>
    }

}

module {

// CHECK:           builtin.func @vec_to_mat_arbitrary_csr(%[[VAL_0:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
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
// CHECK:             %[[VAL_20:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_3]]] : memref<?xi64>
// CHECK:             %[[VAL_21:.*]] = subi %[[VAL_5]], %[[VAL_4]] : index
// CHECK:             %[[VAL_22:.*]]:3 = scf.for %[[VAL_23:.*]] = %[[VAL_3]] to %[[VAL_5]] step %[[VAL_4]] iter_args(%[[VAL_24:.*]] = %[[VAL_1]], %[[VAL_25:.*]] = %[[VAL_3]], %[[VAL_26:.*]] = %[[VAL_20]]) -> (i64, index, i64) {
// CHECK:               memref.store %[[VAL_24]], %[[VAL_19]]{{\[}}%[[VAL_23]]] : memref<?xi64>
// CHECK:               %[[VAL_27:.*]] = index_cast %[[VAL_23]] : index to i64
// CHECK:               %[[VAL_28:.*]] = cmpi eq, %[[VAL_26]], %[[VAL_27]] : i64
// CHECK:               %[[VAL_29:.*]] = cmpi ne, %[[VAL_23]], %[[VAL_21]] : index
// CHECK:               %[[VAL_30:.*]] = and %[[VAL_29]], %[[VAL_28]] : i1
// CHECK:               %[[VAL_31:.*]]:3 = scf.if %[[VAL_30]] -> (i64, index, i64) {
// CHECK:                 %[[VAL_32:.*]] = addi %[[VAL_24]], %[[VAL_2]] : i64
// CHECK:                 %[[VAL_33:.*]] = addi %[[VAL_25]], %[[VAL_4]] : index
// CHECK:                 %[[VAL_34:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_33]]] : memref<?xi64>
// CHECK:                 scf.yield %[[VAL_32]], %[[VAL_33]], %[[VAL_34]] : i64, index, i64
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_24]], %[[VAL_25]], %[[VAL_26]] : i64, index, i64
// CHECK:               }
// CHECK:               scf.yield %[[VAL_35:.*]]#0, %[[VAL_35]]#1, %[[VAL_35]]#2 : i64, index, i64
// CHECK:             }
// CHECK:             memref.store %[[VAL_10]], %[[VAL_19]]{{\[}}%[[VAL_5]]] : memref<?xi64>
// CHECK:             return %[[VAL_8]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

   func @vec_to_mat_arbitrary_csr(%sparse_tensor: tensor<?xf64, #SparseVec64>) -> tensor<?x?xf64, #CSR64> {
       %answer = graphblas.diag %sparse_tensor : tensor<?xf64, #SparseVec64> to tensor<?x?xf64, #CSR64>
       return %answer : tensor<?x?xf64, #CSR64>
   }

// CHECK:           builtin.func @vec_to_mat_arbitrary_csc(%[[VAL_36:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_37:.*]] = constant 0 : i64
// CHECK:             %[[VAL_38:.*]] = constant 1 : i64
// CHECK:             %[[VAL_39:.*]] = constant 0 : index
// CHECK:             %[[VAL_40:.*]] = constant 1 : index
// CHECK:             %[[VAL_41:.*]] = tensor.dim %[[VAL_36]], %[[VAL_39]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_42:.*]] = sparse_tensor.indices %[[VAL_36]], %[[VAL_39]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_43:.*]] = sparse_tensor.values %[[VAL_36]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             %[[VAL_44:.*]] = call @new_matrix_csc_f64_p64i64(%[[VAL_41]], %[[VAL_41]]) : (index, index) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_45:.*]] = sparse_tensor.pointers %[[VAL_36]], %[[VAL_39]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_46:.*]] = memref.load %[[VAL_45]]{{\[}}%[[VAL_40]]] : memref<?xi64>
// CHECK:             %[[VAL_47:.*]] = index_cast %[[VAL_46]] : i64 to index
// CHECK:             %[[VAL_48:.*]] = call @matrix_csc_f64_p64i64_to_ptr8(%[[VAL_44]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_index(%[[VAL_48]], %[[VAL_40]], %[[VAL_47]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_49:.*]] = call @matrix_csc_f64_p64i64_to_ptr8(%[[VAL_44]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_values(%[[VAL_49]], %[[VAL_47]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:             %[[VAL_50:.*]] = sparse_tensor.indices %[[VAL_44]], %[[VAL_40]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_51:.*]] = sparse_tensor.values %[[VAL_44]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             scf.for %[[VAL_52:.*]] = %[[VAL_39]] to %[[VAL_47]] step %[[VAL_40]] {
// CHECK:               %[[VAL_53:.*]] = memref.load %[[VAL_42]]{{\[}}%[[VAL_52]]] : memref<?xi64>
// CHECK:               memref.store %[[VAL_53]], %[[VAL_50]]{{\[}}%[[VAL_52]]] : memref<?xi64>
// CHECK:               %[[VAL_54:.*]] = memref.load %[[VAL_43]]{{\[}}%[[VAL_52]]] : memref<?xf64>
// CHECK:               memref.store %[[VAL_54]], %[[VAL_51]]{{\[}}%[[VAL_52]]] : memref<?xf64>
// CHECK:             }
// CHECK:             %[[VAL_55:.*]] = sparse_tensor.pointers %[[VAL_44]], %[[VAL_40]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_56:.*]] = memref.load %[[VAL_42]]{{\[}}%[[VAL_39]]] : memref<?xi64>
// CHECK:             %[[VAL_57:.*]] = subi %[[VAL_41]], %[[VAL_40]] : index
// CHECK:             %[[VAL_58:.*]]:3 = scf.for %[[VAL_59:.*]] = %[[VAL_39]] to %[[VAL_41]] step %[[VAL_40]] iter_args(%[[VAL_60:.*]] = %[[VAL_37]], %[[VAL_61:.*]] = %[[VAL_39]], %[[VAL_62:.*]] = %[[VAL_56]]) -> (i64, index, i64) {
// CHECK:               memref.store %[[VAL_60]], %[[VAL_55]]{{\[}}%[[VAL_59]]] : memref<?xi64>
// CHECK:               %[[VAL_63:.*]] = index_cast %[[VAL_59]] : index to i64
// CHECK:               %[[VAL_64:.*]] = cmpi eq, %[[VAL_62]], %[[VAL_63]] : i64
// CHECK:               %[[VAL_65:.*]] = cmpi ne, %[[VAL_59]], %[[VAL_57]] : index
// CHECK:               %[[VAL_66:.*]] = and %[[VAL_65]], %[[VAL_64]] : i1
// CHECK:               %[[VAL_67:.*]]:3 = scf.if %[[VAL_66]] -> (i64, index, i64) {
// CHECK:                 %[[VAL_68:.*]] = addi %[[VAL_60]], %[[VAL_38]] : i64
// CHECK:                 %[[VAL_69:.*]] = addi %[[VAL_61]], %[[VAL_40]] : index
// CHECK:                 %[[VAL_70:.*]] = memref.load %[[VAL_42]]{{\[}}%[[VAL_69]]] : memref<?xi64>
// CHECK:                 scf.yield %[[VAL_68]], %[[VAL_69]], %[[VAL_70]] : i64, index, i64
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_60]], %[[VAL_61]], %[[VAL_62]] : i64, index, i64
// CHECK:               }
// CHECK:               scf.yield %[[VAL_71:.*]]#0, %[[VAL_71]]#1, %[[VAL_71]]#2 : i64, index, i64
// CHECK:             }
// CHECK:             memref.store %[[VAL_46]], %[[VAL_55]]{{\[}}%[[VAL_41]]] : memref<?xi64>
// CHECK:             return %[[VAL_44]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

   func @vec_to_mat_arbitrary_csc(%sparse_tensor: tensor<?xf64, #SparseVec64>) -> tensor<?x?xf64, #CSC64> {
       %answer = graphblas.diag %sparse_tensor : tensor<?xf64, #SparseVec64> to tensor<?x?xf64, #CSC64>
       return %answer : tensor<?x?xf64, #CSC64>
   }

// CHECK:           builtin.func @mat_to_vec_arbitrary_csr(%[[VAL_72:.*]]: tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_73:.*]] = constant true
// CHECK:             %[[VAL_74:.*]] = constant 1 : i64
// CHECK:             %[[VAL_75:.*]] = constant 1 : index
// CHECK:             %[[VAL_76:.*]] = constant 2 : index
// CHECK:             %[[VAL_77:.*]] = constant 0 : index
// CHECK:             %[[VAL_78:.*]] = tensor.dim %[[VAL_72]], %[[VAL_77]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_79:.*]] = sparse_tensor.pointers %[[VAL_72]], %[[VAL_75]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_80:.*]] = sparse_tensor.indices %[[VAL_72]], %[[VAL_75]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_81:.*]] = sparse_tensor.values %[[VAL_72]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_82:.*]] = call @new_vector_i64_p64i64(%[[VAL_78]]) : (index) -> tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_83:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_82]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_dim(%[[VAL_83]], %[[VAL_77]], %[[VAL_78]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_84:.*]] = scf.for %[[VAL_85:.*]] = %[[VAL_77]] to %[[VAL_78]] step %[[VAL_75]] iter_args(%[[VAL_86:.*]] = %[[VAL_77]]) -> (index) {
// CHECK:               %[[VAL_87:.*]] = addi %[[VAL_85]], %[[VAL_75]] : index
// CHECK:               %[[VAL_88:.*]] = memref.load %[[VAL_79]]{{\[}}%[[VAL_85]]] : memref<?xi64>
// CHECK:               %[[VAL_89:.*]] = memref.load %[[VAL_79]]{{\[}}%[[VAL_87]]] : memref<?xi64>
// CHECK:               %[[VAL_90:.*]] = index_cast %[[VAL_88]] : i64 to index
// CHECK:               %[[VAL_91:.*]] = index_cast %[[VAL_89]] : i64 to index
// CHECK:               %[[VAL_92:.*]] = index_cast %[[VAL_85]] : index to i64
// CHECK:               %[[VAL_93:.*]]:2 = scf.while (%[[VAL_94:.*]] = %[[VAL_90]], %[[VAL_95:.*]] = %[[VAL_73]]) : (index, i1) -> (index, i1) {
// CHECK:                 %[[VAL_96:.*]] = cmpi ult, %[[VAL_94]], %[[VAL_91]] : index
// CHECK:                 %[[VAL_97:.*]] = and %[[VAL_95]], %[[VAL_96]] : i1
// CHECK:                 scf.condition(%[[VAL_97]]) %[[VAL_94]], %[[VAL_95]] : index, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_98:.*]]: index, %[[VAL_99:.*]]: i1):
// CHECK:                 %[[VAL_100:.*]] = memref.load %[[VAL_80]]{{\[}}%[[VAL_98]]] : memref<?xi64>
// CHECK:                 %[[VAL_101:.*]] = cmpi ne, %[[VAL_100]], %[[VAL_92]] : i64
// CHECK:                 %[[VAL_102:.*]] = addi %[[VAL_98]], %[[VAL_75]] : index
// CHECK:                 scf.yield %[[VAL_102]], %[[VAL_101]] : index, i1
// CHECK:               }
// CHECK:               %[[VAL_103:.*]] = scf.if %[[VAL_104:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_86]] : index
// CHECK:               } else {
// CHECK:                 %[[VAL_105:.*]] = addi %[[VAL_86]], %[[VAL_75]] : index
// CHECK:                 scf.yield %[[VAL_105]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_106:.*]] : index
// CHECK:             }
// CHECK:             %[[VAL_107:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_82]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_pointers(%[[VAL_107]], %[[VAL_77]], %[[VAL_76]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_108:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_82]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_index(%[[VAL_108]], %[[VAL_77]], %[[VAL_109:.*]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_110:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_82]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_values(%[[VAL_110]], %[[VAL_109]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:             %[[VAL_111:.*]] = sparse_tensor.pointers %[[VAL_82]], %[[VAL_77]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_112:.*]] = index_cast %[[VAL_109]] : index to i64
// CHECK:             memref.store %[[VAL_112]], %[[VAL_111]]{{\[}}%[[VAL_75]]] : memref<?xi64>
// CHECK:             %[[VAL_113:.*]] = sparse_tensor.indices %[[VAL_82]], %[[VAL_77]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_114:.*]] = sparse_tensor.values %[[VAL_82]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_115:.*]] = scf.for %[[VAL_116:.*]] = %[[VAL_77]] to %[[VAL_78]] step %[[VAL_75]] iter_args(%[[VAL_117:.*]] = %[[VAL_77]]) -> (index) {
// CHECK:               %[[VAL_118:.*]] = addi %[[VAL_116]], %[[VAL_75]] : index
// CHECK:               %[[VAL_119:.*]] = memref.load %[[VAL_79]]{{\[}}%[[VAL_116]]] : memref<?xi64>
// CHECK:               %[[VAL_120:.*]] = memref.load %[[VAL_79]]{{\[}}%[[VAL_118]]] : memref<?xi64>
// CHECK:               %[[VAL_121:.*]] = index_cast %[[VAL_119]] : i64 to index
// CHECK:               %[[VAL_122:.*]] = index_cast %[[VAL_120]] : i64 to index
// CHECK:               %[[VAL_123:.*]] = index_cast %[[VAL_116]] : index to i64
// CHECK:               %[[VAL_124:.*]]:3 = scf.while (%[[VAL_125:.*]] = %[[VAL_121]], %[[VAL_126:.*]] = %[[VAL_73]], %[[VAL_127:.*]] = %[[VAL_74]]) : (index, i1, i64) -> (index, i1, i64) {
// CHECK:                 %[[VAL_128:.*]] = cmpi ult, %[[VAL_125]], %[[VAL_122]] : index
// CHECK:                 %[[VAL_129:.*]] = and %[[VAL_126]], %[[VAL_128]] : i1
// CHECK:                 scf.condition(%[[VAL_129]]) %[[VAL_125]], %[[VAL_126]], %[[VAL_127]] : index, i1, i64
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_130:.*]]: index, %[[VAL_131:.*]]: i1, %[[VAL_132:.*]]: i64):
// CHECK:                 %[[VAL_133:.*]] = memref.load %[[VAL_80]]{{\[}}%[[VAL_130]]] : memref<?xi64>
// CHECK:                 %[[VAL_134:.*]] = cmpi ne, %[[VAL_133]], %[[VAL_123]] : i64
// CHECK:                 %[[VAL_135:.*]] = scf.if %[[VAL_134]] -> (i64) {
// CHECK:                   scf.yield %[[VAL_132]] : i64
// CHECK:                 } else {
// CHECK:                   %[[VAL_136:.*]] = memref.load %[[VAL_81]]{{\[}}%[[VAL_130]]] : memref<?xi64>
// CHECK:                   scf.yield %[[VAL_136]] : i64
// CHECK:                 }
// CHECK:                 %[[VAL_137:.*]] = addi %[[VAL_130]], %[[VAL_75]] : index
// CHECK:                 scf.yield %[[VAL_137]], %[[VAL_134]], %[[VAL_138:.*]] : index, i1, i64
// CHECK:               }
// CHECK:               %[[VAL_139:.*]] = scf.if %[[VAL_140:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_117]] : index
// CHECK:               } else {
// CHECK:                 memref.store %[[VAL_141:.*]]#2, %[[VAL_114]]{{\[}}%[[VAL_117]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_123]], %[[VAL_113]]{{\[}}%[[VAL_117]]] : memref<?xi64>
// CHECK:                 %[[VAL_142:.*]] = addi %[[VAL_117]], %[[VAL_75]] : index
// CHECK:                 scf.yield %[[VAL_142]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_143:.*]] : index
// CHECK:             }
// CHECK:             return %[[VAL_82]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @mat_to_vec_arbitrary_csr(%mat: tensor<?x?xi64, #CSC64>) -> tensor<?xi64, #SparseVec64> {
        %vec = graphblas.diag %mat : tensor<?x?xi64, #CSC64> to tensor<?xi64, #SparseVec64>
        return %vec : tensor<?xi64, #SparseVec64>
    }

// CHECK:           builtin.func @mat_to_vec_arbitrary_csc(%[[VAL_144:.*]]: tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_145:.*]] = constant true
// CHECK:             %[[VAL_146:.*]] = constant 1 : i64
// CHECK:             %[[VAL_147:.*]] = constant 1 : index
// CHECK:             %[[VAL_148:.*]] = constant 2 : index
// CHECK:             %[[VAL_149:.*]] = constant 0 : index
// CHECK:             %[[VAL_150:.*]] = tensor.dim %[[VAL_144]], %[[VAL_149]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_151:.*]] = sparse_tensor.pointers %[[VAL_144]], %[[VAL_147]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_152:.*]] = sparse_tensor.indices %[[VAL_144]], %[[VAL_147]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_153:.*]] = sparse_tensor.values %[[VAL_144]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_154:.*]] = call @new_vector_i64_p64i64(%[[VAL_150]]) : (index) -> tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_155:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_154]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_dim(%[[VAL_155]], %[[VAL_149]], %[[VAL_150]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_156:.*]] = scf.for %[[VAL_157:.*]] = %[[VAL_149]] to %[[VAL_150]] step %[[VAL_147]] iter_args(%[[VAL_158:.*]] = %[[VAL_149]]) -> (index) {
// CHECK:               %[[VAL_159:.*]] = addi %[[VAL_157]], %[[VAL_147]] : index
// CHECK:               %[[VAL_160:.*]] = memref.load %[[VAL_151]]{{\[}}%[[VAL_157]]] : memref<?xi64>
// CHECK:               %[[VAL_161:.*]] = memref.load %[[VAL_151]]{{\[}}%[[VAL_159]]] : memref<?xi64>
// CHECK:               %[[VAL_162:.*]] = index_cast %[[VAL_160]] : i64 to index
// CHECK:               %[[VAL_163:.*]] = index_cast %[[VAL_161]] : i64 to index
// CHECK:               %[[VAL_164:.*]] = index_cast %[[VAL_157]] : index to i64
// CHECK:               %[[VAL_165:.*]]:2 = scf.while (%[[VAL_166:.*]] = %[[VAL_162]], %[[VAL_167:.*]] = %[[VAL_145]]) : (index, i1) -> (index, i1) {
// CHECK:                 %[[VAL_168:.*]] = cmpi ult, %[[VAL_166]], %[[VAL_163]] : index
// CHECK:                 %[[VAL_169:.*]] = and %[[VAL_167]], %[[VAL_168]] : i1
// CHECK:                 scf.condition(%[[VAL_169]]) %[[VAL_166]], %[[VAL_167]] : index, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_170:.*]]: index, %[[VAL_171:.*]]: i1):
// CHECK:                 %[[VAL_172:.*]] = memref.load %[[VAL_152]]{{\[}}%[[VAL_170]]] : memref<?xi64>
// CHECK:                 %[[VAL_173:.*]] = cmpi ne, %[[VAL_172]], %[[VAL_164]] : i64
// CHECK:                 %[[VAL_174:.*]] = addi %[[VAL_170]], %[[VAL_147]] : index
// CHECK:                 scf.yield %[[VAL_174]], %[[VAL_173]] : index, i1
// CHECK:               }
// CHECK:               %[[VAL_175:.*]] = scf.if %[[VAL_176:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_158]] : index
// CHECK:               } else {
// CHECK:                 %[[VAL_177:.*]] = addi %[[VAL_158]], %[[VAL_147]] : index
// CHECK:                 scf.yield %[[VAL_177]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_178:.*]] : index
// CHECK:             }
// CHECK:             %[[VAL_179:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_154]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_pointers(%[[VAL_179]], %[[VAL_149]], %[[VAL_148]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_180:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_154]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_index(%[[VAL_180]], %[[VAL_149]], %[[VAL_181:.*]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_182:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_154]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_values(%[[VAL_182]], %[[VAL_181]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:             %[[VAL_183:.*]] = sparse_tensor.pointers %[[VAL_154]], %[[VAL_149]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_184:.*]] = index_cast %[[VAL_181]] : index to i64
// CHECK:             memref.store %[[VAL_184]], %[[VAL_183]]{{\[}}%[[VAL_147]]] : memref<?xi64>
// CHECK:             %[[VAL_185:.*]] = sparse_tensor.indices %[[VAL_154]], %[[VAL_149]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_186:.*]] = sparse_tensor.values %[[VAL_154]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_187:.*]] = scf.for %[[VAL_188:.*]] = %[[VAL_149]] to %[[VAL_150]] step %[[VAL_147]] iter_args(%[[VAL_189:.*]] = %[[VAL_149]]) -> (index) {
// CHECK:               %[[VAL_190:.*]] = addi %[[VAL_188]], %[[VAL_147]] : index
// CHECK:               %[[VAL_191:.*]] = memref.load %[[VAL_151]]{{\[}}%[[VAL_188]]] : memref<?xi64>
// CHECK:               %[[VAL_192:.*]] = memref.load %[[VAL_151]]{{\[}}%[[VAL_190]]] : memref<?xi64>
// CHECK:               %[[VAL_193:.*]] = index_cast %[[VAL_191]] : i64 to index
// CHECK:               %[[VAL_194:.*]] = index_cast %[[VAL_192]] : i64 to index
// CHECK:               %[[VAL_195:.*]] = index_cast %[[VAL_188]] : index to i64
// CHECK:               %[[VAL_196:.*]]:3 = scf.while (%[[VAL_197:.*]] = %[[VAL_193]], %[[VAL_198:.*]] = %[[VAL_145]], %[[VAL_199:.*]] = %[[VAL_146]]) : (index, i1, i64) -> (index, i1, i64) {
// CHECK:                 %[[VAL_200:.*]] = cmpi ult, %[[VAL_197]], %[[VAL_194]] : index
// CHECK:                 %[[VAL_201:.*]] = and %[[VAL_198]], %[[VAL_200]] : i1
// CHECK:                 scf.condition(%[[VAL_201]]) %[[VAL_197]], %[[VAL_198]], %[[VAL_199]] : index, i1, i64
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_202:.*]]: index, %[[VAL_203:.*]]: i1, %[[VAL_204:.*]]: i64):
// CHECK:                 %[[VAL_205:.*]] = memref.load %[[VAL_152]]{{\[}}%[[VAL_202]]] : memref<?xi64>
// CHECK:                 %[[VAL_206:.*]] = cmpi ne, %[[VAL_205]], %[[VAL_195]] : i64
// CHECK:                 %[[VAL_207:.*]] = scf.if %[[VAL_206]] -> (i64) {
// CHECK:                   scf.yield %[[VAL_204]] : i64
// CHECK:                 } else {
// CHECK:                   %[[VAL_208:.*]] = memref.load %[[VAL_153]]{{\[}}%[[VAL_202]]] : memref<?xi64>
// CHECK:                   scf.yield %[[VAL_208]] : i64
// CHECK:                 }
// CHECK:                 %[[VAL_209:.*]] = addi %[[VAL_202]], %[[VAL_147]] : index
// CHECK:                 scf.yield %[[VAL_209]], %[[VAL_206]], %[[VAL_210:.*]] : index, i1, i64
// CHECK:               }
// CHECK:               %[[VAL_211:.*]] = scf.if %[[VAL_212:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_189]] : index
// CHECK:               } else {
// CHECK:                 memref.store %[[VAL_213:.*]]#2, %[[VAL_186]]{{\[}}%[[VAL_189]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_195]], %[[VAL_185]]{{\[}}%[[VAL_189]]] : memref<?xi64>
// CHECK:                 %[[VAL_214:.*]] = addi %[[VAL_189]], %[[VAL_147]] : index
// CHECK:                 scf.yield %[[VAL_214]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_215:.*]] : index
// CHECK:             }
// CHECK:             return %[[VAL_154]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @mat_to_vec_arbitrary_csc(%mat: tensor<?x?xi64, #CSC64>) -> tensor<?xi64, #SparseVec64> {
        %vec = graphblas.diag %mat : tensor<?x?xi64, #CSC64> to tensor<?xi64, #SparseVec64>
        return %vec : tensor<?xi64, #SparseVec64>
    }

}
