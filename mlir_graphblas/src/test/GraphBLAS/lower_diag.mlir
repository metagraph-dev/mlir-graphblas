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

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {

// CHECK:           func private @matrix_csr_f64_p64i64_to_ptr8(tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           func private @new_matrix_csr_f64_p64i64(index, index) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           func private @matrix_csc_f64_p64i64_to_ptr8(tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           func private @new_matrix_csc_f64_p64i64(index, index) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           func private @resize_values(!llvm.ptr<i8>, index)
// CHECK:           func private @resize_index(!llvm.ptr<i8>, index, index)
// CHECK:           func private @resize_pointers(!llvm.ptr<i8>, index, index)
// CHECK:           func private @resize_dim(!llvm.ptr<i8>, index, index)
// CHECK:           func private @vector_i64_p64i64_to_ptr8(tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           func private @new_vector_i64_p64i64(index) -> tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           func @vec_to_mat_fixed_csr(%[[VAL_0:.*]]: tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK:             %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK:             %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_5:.*]] = arith.constant 6 : index
// CHECK:             %[[VAL_6:.*]] = arith.constant 7 : index
// CHECK:             %[[VAL_7:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_3]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_8:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             %[[VAL_9:.*]] = call @new_matrix_csr_f64_p64i64(%[[VAL_6]], %[[VAL_6]]) : (index, index) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_10:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_3]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_11:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_4]]] : memref<?xi64>
// CHECK:             %[[VAL_12:.*]] = arith.index_cast %[[VAL_11]] : i64 to index
// CHECK:             %[[VAL_13:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_9]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_index(%[[VAL_13]], %[[VAL_4]], %[[VAL_12]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_14:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_9]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_values(%[[VAL_14]], %[[VAL_12]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:             %[[VAL_15:.*]] = sparse_tensor.indices %[[VAL_9]], %[[VAL_4]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_16:.*]] = sparse_tensor.values %[[VAL_9]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             scf.for %[[VAL_17:.*]] = %[[VAL_3]] to %[[VAL_12]] step %[[VAL_4]] {
// CHECK:               %[[VAL_18:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_17]]] : memref<?xi64>
// CHECK:               memref.store %[[VAL_18]], %[[VAL_15]]{{\[}}%[[VAL_17]]] : memref<?xi64>
// CHECK:               %[[VAL_19:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_17]]] : memref<?xf64>
// CHECK:               memref.store %[[VAL_19]], %[[VAL_16]]{{\[}}%[[VAL_17]]] : memref<?xf64>
// CHECK:             }
// CHECK:             %[[VAL_20:.*]] = sparse_tensor.pointers %[[VAL_9]], %[[VAL_4]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_21:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_3]]] : memref<?xi64>
// CHECK:             %[[VAL_22:.*]]:3 = scf.for %[[VAL_23:.*]] = %[[VAL_3]] to %[[VAL_6]] step %[[VAL_4]] iter_args(%[[VAL_24:.*]] = %[[VAL_1]], %[[VAL_25:.*]] = %[[VAL_3]], %[[VAL_26:.*]] = %[[VAL_21]]) -> (i64, index, i64) {
// CHECK:               memref.store %[[VAL_24]], %[[VAL_20]]{{\[}}%[[VAL_23]]] : memref<?xi64>
// CHECK:               %[[VAL_27:.*]] = arith.index_cast %[[VAL_23]] : index to i64
// CHECK:               %[[VAL_28:.*]] = arith.cmpi eq, %[[VAL_26]], %[[VAL_27]] : i64
// CHECK:               %[[VAL_29:.*]] = arith.cmpi ne, %[[VAL_23]], %[[VAL_5]] : index
// CHECK:               %[[VAL_30:.*]] = arith.andi %[[VAL_29]], %[[VAL_28]] : i1
// CHECK:               %[[VAL_31:.*]]:3 = scf.if %[[VAL_30]] -> (i64, index, i64) {
// CHECK:                 %[[VAL_32:.*]] = arith.addi %[[VAL_24]], %[[VAL_2]] : i64
// CHECK:                 %[[VAL_33:.*]] = arith.addi %[[VAL_25]], %[[VAL_4]] : index
// CHECK:                 %[[VAL_34:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_33]]] : memref<?xi64>
// CHECK:                 scf.yield %[[VAL_32]], %[[VAL_33]], %[[VAL_34]] : i64, index, i64
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_24]], %[[VAL_25]], %[[VAL_26]] : i64, index, i64
// CHECK:               }
// CHECK:               scf.yield %[[VAL_35:.*]]#0, %[[VAL_35]]#1, %[[VAL_35]]#2 : i64, index, i64
// CHECK:             }
// CHECK:             %[[VAL_36:.*]] = arith.index_cast %[[VAL_12]] : index to i64
// CHECK:             memref.store %[[VAL_36]], %[[VAL_20]]{{\[}}%[[VAL_6]]] : memref<?xi64>
// CHECK:             return %[[VAL_9]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

   func @vec_to_mat_fixed_csr(%sparse_tensor: tensor<7xf64, #CV64>) -> tensor<7x7xf64, #CSR64> {
       %answer = graphblas.diag %sparse_tensor : tensor<7xf64, #CV64> to tensor<7x7xf64, #CSR64>
       return %answer : tensor<7x7xf64, #CSR64>
   }

// CHECK:           func @vec_to_mat_fixed_csc(%[[VAL_37:.*]]: tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_38:.*]] = arith.constant 0 : i64
// CHECK:             %[[VAL_39:.*]] = arith.constant 1 : i64
// CHECK:             %[[VAL_40:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_41:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_42:.*]] = arith.constant 6 : index
// CHECK:             %[[VAL_43:.*]] = arith.constant 7 : index
// CHECK:             %[[VAL_44:.*]] = sparse_tensor.indices %[[VAL_37]], %[[VAL_40]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_45:.*]] = sparse_tensor.values %[[VAL_37]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             %[[VAL_46:.*]] = call @new_matrix_csc_f64_p64i64(%[[VAL_43]], %[[VAL_43]]) : (index, index) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_47:.*]] = sparse_tensor.pointers %[[VAL_37]], %[[VAL_40]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_48:.*]] = memref.load %[[VAL_47]]{{\[}}%[[VAL_41]]] : memref<?xi64>
// CHECK:             %[[VAL_49:.*]] = arith.index_cast %[[VAL_48]] : i64 to index
// CHECK:             %[[VAL_50:.*]] = call @matrix_csc_f64_p64i64_to_ptr8(%[[VAL_46]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_index(%[[VAL_50]], %[[VAL_41]], %[[VAL_49]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_51:.*]] = call @matrix_csc_f64_p64i64_to_ptr8(%[[VAL_46]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_values(%[[VAL_51]], %[[VAL_49]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:             %[[VAL_52:.*]] = sparse_tensor.indices %[[VAL_46]], %[[VAL_41]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_53:.*]] = sparse_tensor.values %[[VAL_46]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             scf.for %[[VAL_54:.*]] = %[[VAL_40]] to %[[VAL_49]] step %[[VAL_41]] {
// CHECK:               %[[VAL_55:.*]] = memref.load %[[VAL_44]]{{\[}}%[[VAL_54]]] : memref<?xi64>
// CHECK:               memref.store %[[VAL_55]], %[[VAL_52]]{{\[}}%[[VAL_54]]] : memref<?xi64>
// CHECK:               %[[VAL_56:.*]] = memref.load %[[VAL_45]]{{\[}}%[[VAL_54]]] : memref<?xf64>
// CHECK:               memref.store %[[VAL_56]], %[[VAL_53]]{{\[}}%[[VAL_54]]] : memref<?xf64>
// CHECK:             }
// CHECK:             %[[VAL_57:.*]] = sparse_tensor.pointers %[[VAL_46]], %[[VAL_41]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_58:.*]] = memref.load %[[VAL_44]]{{\[}}%[[VAL_40]]] : memref<?xi64>
// CHECK:             %[[VAL_59:.*]]:3 = scf.for %[[VAL_60:.*]] = %[[VAL_40]] to %[[VAL_43]] step %[[VAL_41]] iter_args(%[[VAL_61:.*]] = %[[VAL_38]], %[[VAL_62:.*]] = %[[VAL_40]], %[[VAL_63:.*]] = %[[VAL_58]]) -> (i64, index, i64) {
// CHECK:               memref.store %[[VAL_61]], %[[VAL_57]]{{\[}}%[[VAL_60]]] : memref<?xi64>
// CHECK:               %[[VAL_64:.*]] = arith.index_cast %[[VAL_60]] : index to i64
// CHECK:               %[[VAL_65:.*]] = arith.cmpi eq, %[[VAL_63]], %[[VAL_64]] : i64
// CHECK:               %[[VAL_66:.*]] = arith.cmpi ne, %[[VAL_60]], %[[VAL_42]] : index
// CHECK:               %[[VAL_67:.*]] = arith.andi %[[VAL_66]], %[[VAL_65]] : i1
// CHECK:               %[[VAL_68:.*]]:3 = scf.if %[[VAL_67]] -> (i64, index, i64) {
// CHECK:                 %[[VAL_69:.*]] = arith.addi %[[VAL_61]], %[[VAL_39]] : i64
// CHECK:                 %[[VAL_70:.*]] = arith.addi %[[VAL_62]], %[[VAL_41]] : index
// CHECK:                 %[[VAL_71:.*]] = memref.load %[[VAL_44]]{{\[}}%[[VAL_70]]] : memref<?xi64>
// CHECK:                 scf.yield %[[VAL_69]], %[[VAL_70]], %[[VAL_71]] : i64, index, i64
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_61]], %[[VAL_62]], %[[VAL_63]] : i64, index, i64
// CHECK:               }
// CHECK:               scf.yield %[[VAL_72:.*]]#0, %[[VAL_72]]#1, %[[VAL_72]]#2 : i64, index, i64
// CHECK:             }
// CHECK:             %[[VAL_73:.*]] = arith.index_cast %[[VAL_49]] : index to i64
// CHECK:             memref.store %[[VAL_73]], %[[VAL_57]]{{\[}}%[[VAL_43]]] : memref<?xi64>
// CHECK:             return %[[VAL_46]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

   func @vec_to_mat_fixed_csc(%sparse_tensor: tensor<7xf64, #CV64>) -> tensor<7x7xf64, #CSC64> {
       %answer = graphblas.diag %sparse_tensor : tensor<7xf64, #CV64> to tensor<7x7xf64, #CSC64>
       return %answer : tensor<7x7xf64, #CSC64>
   }
   
// CHECK:           func @mat_to_vec_fixed_csr(%[[VAL_76:.*]]: tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_77:.*]] = arith.constant true
// CHECK:             %[[VAL_78:.*]] = arith.constant 1 : i64
// CHECK:             %[[VAL_79:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_80:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_81:.*]] = arith.constant 2 : index
// CHECK:             %[[VAL_82:.*]] = arith.constant 7 : index
// CHECK:             %[[VAL_83:.*]] = sparse_tensor.pointers %[[VAL_76]], %[[VAL_80]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_84:.*]] = sparse_tensor.indices %[[VAL_76]], %[[VAL_80]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_85:.*]] = sparse_tensor.values %[[VAL_76]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_86:.*]] = call @new_vector_i64_p64i64(%[[VAL_82]]) : (index) -> tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_88:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_86]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_dim(%[[VAL_88]], %[[VAL_79]], %[[VAL_82]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_89:.*]] = scf.for %[[VAL_90:.*]] = %[[VAL_79]] to %[[VAL_82]] step %[[VAL_80]] iter_args(%[[VAL_91:.*]] = %[[VAL_79]]) -> (index) {
// CHECK:               %[[VAL_92:.*]] = arith.addi %[[VAL_90]], %[[VAL_80]] : index
// CHECK:               %[[VAL_93:.*]] = memref.load %[[VAL_83]]{{\[}}%[[VAL_90]]] : memref<?xi64>
// CHECK:               %[[VAL_94:.*]] = memref.load %[[VAL_83]]{{\[}}%[[VAL_92]]] : memref<?xi64>
// CHECK:               %[[VAL_95:.*]] = arith.index_cast %[[VAL_93]] : i64 to index
// CHECK:               %[[VAL_96:.*]] = arith.index_cast %[[VAL_94]] : i64 to index
// CHECK:               %[[VAL_97:.*]] = arith.index_cast %[[VAL_90]] : index to i64
// CHECK:               %[[VAL_98:.*]]:2 = scf.while (%[[VAL_99:.*]] = %[[VAL_95]], %[[VAL_100:.*]] = %[[VAL_77]]) : (index, i1) -> (index, i1) {
// CHECK:                 %[[VAL_101:.*]] = arith.cmpi ult, %[[VAL_99]], %[[VAL_96]] : index
// CHECK:                 %[[VAL_102:.*]] = arith.andi %[[VAL_100]], %[[VAL_101]] : i1
// CHECK:                 scf.condition(%[[VAL_102]]) %[[VAL_99]], %[[VAL_100]] : index, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_103:.*]]: index, %[[VAL_104:.*]]: i1):
// CHECK:                 %[[VAL_105:.*]] = memref.load %[[VAL_84]]{{\[}}%[[VAL_103]]] : memref<?xi64>
// CHECK:                 %[[VAL_106:.*]] = arith.cmpi ne, %[[VAL_105]], %[[VAL_97]] : i64
// CHECK:                 %[[VAL_107:.*]] = arith.addi %[[VAL_103]], %[[VAL_80]] : index
// CHECK:                 scf.yield %[[VAL_107]], %[[VAL_106]] : index, i1
// CHECK:               }
// CHECK:               %[[VAL_108:.*]] = scf.if %[[VAL_109:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_91]] : index
// CHECK:               } else {
// CHECK:                 %[[VAL_110:.*]] = arith.addi %[[VAL_91]], %[[VAL_80]] : index
// CHECK:                 scf.yield %[[VAL_110]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_111:.*]] : index
// CHECK:             }
// CHECK:             %[[VAL_112:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_86]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_pointers(%[[VAL_112]], %[[VAL_79]], %[[VAL_81]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_113:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_86]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_index(%[[VAL_113]], %[[VAL_79]], %[[VAL_114:.*]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_115:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_86]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_values(%[[VAL_115]], %[[VAL_114]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:             %[[VAL_116:.*]] = sparse_tensor.pointers %[[VAL_86]], %[[VAL_79]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_117:.*]] = arith.index_cast %[[VAL_114]] : index to i64
// CHECK:             memref.store %[[VAL_117]], %[[VAL_116]]{{\[}}%[[VAL_80]]] : memref<?xi64>
// CHECK:             %[[VAL_118:.*]] = sparse_tensor.indices %[[VAL_86]], %[[VAL_79]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_119:.*]] = sparse_tensor.values %[[VAL_86]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_120:.*]] = scf.for %[[VAL_121:.*]] = %[[VAL_79]] to %[[VAL_82]] step %[[VAL_80]] iter_args(%[[VAL_122:.*]] = %[[VAL_79]]) -> (index) {
// CHECK:               %[[VAL_123:.*]] = arith.addi %[[VAL_121]], %[[VAL_80]] : index
// CHECK:               %[[VAL_124:.*]] = memref.load %[[VAL_83]]{{\[}}%[[VAL_121]]] : memref<?xi64>
// CHECK:               %[[VAL_125:.*]] = memref.load %[[VAL_83]]{{\[}}%[[VAL_123]]] : memref<?xi64>
// CHECK:               %[[VAL_126:.*]] = arith.index_cast %[[VAL_124]] : i64 to index
// CHECK:               %[[VAL_127:.*]] = arith.index_cast %[[VAL_125]] : i64 to index
// CHECK:               %[[VAL_128:.*]] = arith.index_cast %[[VAL_121]] : index to i64
// CHECK:               %[[VAL_129:.*]]:3 = scf.while (%[[VAL_130:.*]] = %[[VAL_126]], %[[VAL_131:.*]] = %[[VAL_77]], %[[VAL_132:.*]] = %[[VAL_78]]) : (index, i1, i64) -> (index, i1, i64) {
// CHECK:                 %[[VAL_133:.*]] = arith.cmpi ult, %[[VAL_130]], %[[VAL_127]] : index
// CHECK:                 %[[VAL_134:.*]] = arith.andi %[[VAL_131]], %[[VAL_133]] : i1
// CHECK:                 scf.condition(%[[VAL_134]]) %[[VAL_130]], %[[VAL_131]], %[[VAL_132]] : index, i1, i64
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_135:.*]]: index, %[[VAL_136:.*]]: i1, %[[VAL_137:.*]]: i64):
// CHECK:                 %[[VAL_138:.*]] = memref.load %[[VAL_84]]{{\[}}%[[VAL_135]]] : memref<?xi64>
// CHECK:                 %[[VAL_139:.*]] = arith.cmpi ne, %[[VAL_138]], %[[VAL_128]] : i64
// CHECK:                 %[[VAL_140:.*]] = scf.if %[[VAL_139]] -> (i64) {
// CHECK:                   scf.yield %[[VAL_137]] : i64
// CHECK:                 } else {
// CHECK:                   %[[VAL_141:.*]] = memref.load %[[VAL_85]]{{\[}}%[[VAL_135]]] : memref<?xi64>
// CHECK:                   scf.yield %[[VAL_141]] : i64
// CHECK:                 }
// CHECK:                 %[[VAL_142:.*]] = arith.addi %[[VAL_135]], %[[VAL_80]] : index
// CHECK:                 scf.yield %[[VAL_142]], %[[VAL_139]], %[[VAL_143:.*]] : index, i1, i64
// CHECK:               }
// CHECK:               %[[VAL_144:.*]] = scf.if %[[VAL_145:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_122]] : index
// CHECK:               } else {
// CHECK:                 memref.store %[[VAL_146:.*]]#2, %[[VAL_119]]{{\[}}%[[VAL_122]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_128]], %[[VAL_118]]{{\[}}%[[VAL_122]]] : memref<?xi64>
// CHECK:                 %[[VAL_147:.*]] = arith.addi %[[VAL_122]], %[[VAL_80]] : index
// CHECK:                 scf.yield %[[VAL_147]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_148:.*]] : index
// CHECK:             }
// CHECK:             return %[[VAL_86]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }
    func @mat_to_vec_fixed_csr(%mat: tensor<7x7xi64, #CSC64>) -> tensor<7xi64, #CV64> {
        %vec = graphblas.diag %mat : tensor<7x7xi64, #CSC64> to tensor<7xi64, #CV64>
        return %vec : tensor<7xi64, #CV64>
    }

// CHECK:           func @mat_to_vec_fixed_csc(%[[VAL_149:.*]]: tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_150:.*]] = arith.constant true
// CHECK:             %[[VAL_151:.*]] = arith.constant 1 : i64
// CHECK:             %[[VAL_152:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_153:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_154:.*]] = arith.constant 2 : index
// CHECK:             %[[VAL_155:.*]] = arith.constant 7 : index
// CHECK:             %[[VAL_156:.*]] = sparse_tensor.pointers %[[VAL_149]], %[[VAL_153]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_157:.*]] = sparse_tensor.indices %[[VAL_149]], %[[VAL_153]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_158:.*]] = sparse_tensor.values %[[VAL_149]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_159:.*]] = call @new_vector_i64_p64i64(%[[VAL_155]]) : (index) -> tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_161:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_159]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_dim(%[[VAL_161]], %[[VAL_152]], %[[VAL_155]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_162:.*]] = scf.for %[[VAL_163:.*]] = %[[VAL_152]] to %[[VAL_155]] step %[[VAL_153]] iter_args(%[[VAL_164:.*]] = %[[VAL_152]]) -> (index) {
// CHECK:               %[[VAL_165:.*]] = arith.addi %[[VAL_163]], %[[VAL_153]] : index
// CHECK:               %[[VAL_166:.*]] = memref.load %[[VAL_156]]{{\[}}%[[VAL_163]]] : memref<?xi64>
// CHECK:               %[[VAL_167:.*]] = memref.load %[[VAL_156]]{{\[}}%[[VAL_165]]] : memref<?xi64>
// CHECK:               %[[VAL_168:.*]] = arith.index_cast %[[VAL_166]] : i64 to index
// CHECK:               %[[VAL_169:.*]] = arith.index_cast %[[VAL_167]] : i64 to index
// CHECK:               %[[VAL_170:.*]] = arith.index_cast %[[VAL_163]] : index to i64
// CHECK:               %[[VAL_171:.*]]:2 = scf.while (%[[VAL_172:.*]] = %[[VAL_168]], %[[VAL_173:.*]] = %[[VAL_150]]) : (index, i1) -> (index, i1) {
// CHECK:                 %[[VAL_174:.*]] = arith.cmpi ult, %[[VAL_172]], %[[VAL_169]] : index
// CHECK:                 %[[VAL_175:.*]] = arith.andi %[[VAL_173]], %[[VAL_174]] : i1
// CHECK:                 scf.condition(%[[VAL_175]]) %[[VAL_172]], %[[VAL_173]] : index, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_176:.*]]: index, %[[VAL_177:.*]]: i1):
// CHECK:                 %[[VAL_178:.*]] = memref.load %[[VAL_157]]{{\[}}%[[VAL_176]]] : memref<?xi64>
// CHECK:                 %[[VAL_179:.*]] = arith.cmpi ne, %[[VAL_178]], %[[VAL_170]] : i64
// CHECK:                 %[[VAL_180:.*]] = arith.addi %[[VAL_176]], %[[VAL_153]] : index
// CHECK:                 scf.yield %[[VAL_180]], %[[VAL_179]] : index, i1
// CHECK:               }
// CHECK:               %[[VAL_181:.*]] = scf.if %[[VAL_182:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_164]] : index
// CHECK:               } else {
// CHECK:                 %[[VAL_183:.*]] = arith.addi %[[VAL_164]], %[[VAL_153]] : index
// CHECK:                 scf.yield %[[VAL_183]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_184:.*]] : index
// CHECK:             }
// CHECK:             %[[VAL_185:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_159]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_pointers(%[[VAL_185]], %[[VAL_152]], %[[VAL_154]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_186:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_159]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_index(%[[VAL_186]], %[[VAL_152]], %[[VAL_187:.*]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_188:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_159]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_values(%[[VAL_188]], %[[VAL_187]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:             %[[VAL_189:.*]] = sparse_tensor.pointers %[[VAL_159]], %[[VAL_152]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_190:.*]] = arith.index_cast %[[VAL_187]] : index to i64
// CHECK:             memref.store %[[VAL_190]], %[[VAL_189]]{{\[}}%[[VAL_153]]] : memref<?xi64>
// CHECK:             %[[VAL_191:.*]] = sparse_tensor.indices %[[VAL_159]], %[[VAL_152]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_192:.*]] = sparse_tensor.values %[[VAL_159]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_193:.*]] = scf.for %[[VAL_194:.*]] = %[[VAL_152]] to %[[VAL_155]] step %[[VAL_153]] iter_args(%[[VAL_195:.*]] = %[[VAL_152]]) -> (index) {
// CHECK:               %[[VAL_196:.*]] = arith.addi %[[VAL_194]], %[[VAL_153]] : index
// CHECK:               %[[VAL_197:.*]] = memref.load %[[VAL_156]]{{\[}}%[[VAL_194]]] : memref<?xi64>
// CHECK:               %[[VAL_198:.*]] = memref.load %[[VAL_156]]{{\[}}%[[VAL_196]]] : memref<?xi64>
// CHECK:               %[[VAL_199:.*]] = arith.index_cast %[[VAL_197]] : i64 to index
// CHECK:               %[[VAL_200:.*]] = arith.index_cast %[[VAL_198]] : i64 to index
// CHECK:               %[[VAL_201:.*]] = arith.index_cast %[[VAL_194]] : index to i64
// CHECK:               %[[VAL_202:.*]]:3 = scf.while (%[[VAL_203:.*]] = %[[VAL_199]], %[[VAL_204:.*]] = %[[VAL_150]], %[[VAL_205:.*]] = %[[VAL_151]]) : (index, i1, i64) -> (index, i1, i64) {
// CHECK:                 %[[VAL_206:.*]] = arith.cmpi ult, %[[VAL_203]], %[[VAL_200]] : index
// CHECK:                 %[[VAL_207:.*]] = arith.andi %[[VAL_204]], %[[VAL_206]] : i1
// CHECK:                 scf.condition(%[[VAL_207]]) %[[VAL_203]], %[[VAL_204]], %[[VAL_205]] : index, i1, i64
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_208:.*]]: index, %[[VAL_209:.*]]: i1, %[[VAL_210:.*]]: i64):
// CHECK:                 %[[VAL_211:.*]] = memref.load %[[VAL_157]]{{\[}}%[[VAL_208]]] : memref<?xi64>
// CHECK:                 %[[VAL_212:.*]] = arith.cmpi ne, %[[VAL_211]], %[[VAL_201]] : i64
// CHECK:                 %[[VAL_213:.*]] = scf.if %[[VAL_212]] -> (i64) {
// CHECK:                   scf.yield %[[VAL_210]] : i64
// CHECK:                 } else {
// CHECK:                   %[[VAL_214:.*]] = memref.load %[[VAL_158]]{{\[}}%[[VAL_208]]] : memref<?xi64>
// CHECK:                   scf.yield %[[VAL_214]] : i64
// CHECK:                 }
// CHECK:                 %[[VAL_215:.*]] = arith.addi %[[VAL_208]], %[[VAL_153]] : index
// CHECK:                 scf.yield %[[VAL_215]], %[[VAL_212]], %[[VAL_216:.*]] : index, i1, i64
// CHECK:               }
// CHECK:               %[[VAL_217:.*]] = scf.if %[[VAL_218:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_195]] : index
// CHECK:               } else {
// CHECK:                 memref.store %[[VAL_219:.*]]#2, %[[VAL_192]]{{\[}}%[[VAL_195]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_201]], %[[VAL_191]]{{\[}}%[[VAL_195]]] : memref<?xi64>
// CHECK:                 %[[VAL_220:.*]] = arith.addi %[[VAL_195]], %[[VAL_153]] : index
// CHECK:                 scf.yield %[[VAL_220]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_221:.*]] : index
// CHECK:             }
// CHECK:             return %[[VAL_159]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }
    func @mat_to_vec_fixed_csc(%mat: tensor<7x7xi64, #CSC64>) -> tensor<7xi64, #CV64> {
        %vec = graphblas.diag %mat : tensor<7x7xi64, #CSC64> to tensor<7xi64, #CV64>
        return %vec : tensor<7xi64, #CV64>
    }

}

module {

// CHECK:           func private @matrix_csr_f64_p64i64_to_ptr8(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           func private @new_matrix_csr_f64_p64i64(index, index) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           func private @matrix_csc_f64_p64i64_to_ptr8(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           func private @new_matrix_csc_f64_p64i64(index, index) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           func private @resize_values(!llvm.ptr<i8>, index)
// CHECK:           func private @resize_index(!llvm.ptr<i8>, index, index)
// CHECK:           func private @resize_pointers(!llvm.ptr<i8>, index, index)
// CHECK:           func private @resize_dim(!llvm.ptr<i8>, index, index)
// CHECK:           func private @vector_i64_p64i64_to_ptr8(tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           func private @new_vector_i64_p64i64(index) -> tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           func @vec_to_mat_arbitrary_csr(%[[VAL_0:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK:             %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK:             %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_5:.*]] = tensor.dim %[[VAL_0]], %[[VAL_3]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_6:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_3]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             %[[VAL_8:.*]] = call @new_matrix_csr_f64_p64i64(%[[VAL_5]], %[[VAL_5]]) : (index, index) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_9:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_3]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_10:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_4]]] : memref<?xi64>
// CHECK:             %[[VAL_11:.*]] = arith.index_cast %[[VAL_10]] : i64 to index
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
// CHECK:             %[[VAL_21:.*]] = arith.subi %[[VAL_5]], %[[VAL_4]] : index
// CHECK:             %[[VAL_22:.*]]:3 = scf.for %[[VAL_23:.*]] = %[[VAL_3]] to %[[VAL_5]] step %[[VAL_4]] iter_args(%[[VAL_24:.*]] = %[[VAL_1]], %[[VAL_25:.*]] = %[[VAL_3]], %[[VAL_26:.*]] = %[[VAL_20]]) -> (i64, index, i64) {
// CHECK:               memref.store %[[VAL_24]], %[[VAL_19]]{{\[}}%[[VAL_23]]] : memref<?xi64>
// CHECK:               %[[VAL_27:.*]] = arith.index_cast %[[VAL_23]] : index to i64
// CHECK:               %[[VAL_28:.*]] = arith.cmpi eq, %[[VAL_26]], %[[VAL_27]] : i64
// CHECK:               %[[VAL_29:.*]] = arith.cmpi ne, %[[VAL_23]], %[[VAL_21]] : index
// CHECK:               %[[VAL_30:.*]] = arith.andi %[[VAL_29]], %[[VAL_28]] : i1
// CHECK:               %[[VAL_31:.*]]:3 = scf.if %[[VAL_30]] -> (i64, index, i64) {
// CHECK:                 %[[VAL_32:.*]] = arith.addi %[[VAL_24]], %[[VAL_2]] : i64
// CHECK:                 %[[VAL_33:.*]] = arith.addi %[[VAL_25]], %[[VAL_4]] : index
// CHECK:                 %[[VAL_34:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_33]]] : memref<?xi64>
// CHECK:                 scf.yield %[[VAL_32]], %[[VAL_33]], %[[VAL_34]] : i64, index, i64
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_24]], %[[VAL_25]], %[[VAL_26]] : i64, index, i64
// CHECK:               }
// CHECK:               scf.yield %[[VAL_35:.*]]#0, %[[VAL_35]]#1, %[[VAL_35]]#2 : i64, index, i64
// CHECK:             }
// CHECK:             %[[VAL_36:.*]] = arith.index_cast %[[VAL_11]] : index to i64
// CHECK:             memref.store %[[VAL_36]], %[[VAL_19]]{{\[}}%[[VAL_5]]] : memref<?xi64>
// CHECK:             return %[[VAL_8]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }
   func @vec_to_mat_arbitrary_csr(%sparse_tensor: tensor<?xf64, #CV64>) -> tensor<?x?xf64, #CSR64> {
       %answer = graphblas.diag %sparse_tensor : tensor<?xf64, #CV64> to tensor<?x?xf64, #CSR64>
       return %answer : tensor<?x?xf64, #CSR64>
   }

// CHECK:           func @vec_to_mat_arbitrary_csc(%[[VAL_37:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_38:.*]] = arith.constant 0 : i64
// CHECK:             %[[VAL_39:.*]] = arith.constant 1 : i64
// CHECK:             %[[VAL_40:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_41:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_42:.*]] = tensor.dim %[[VAL_37]], %[[VAL_40]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_43:.*]] = sparse_tensor.indices %[[VAL_37]], %[[VAL_40]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_44:.*]] = sparse_tensor.values %[[VAL_37]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             %[[VAL_45:.*]] = call @new_matrix_csc_f64_p64i64(%[[VAL_42]], %[[VAL_42]]) : (index, index) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_46:.*]] = sparse_tensor.pointers %[[VAL_37]], %[[VAL_40]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_47:.*]] = memref.load %[[VAL_46]]{{\[}}%[[VAL_41]]] : memref<?xi64>
// CHECK:             %[[VAL_48:.*]] = arith.index_cast %[[VAL_47]] : i64 to index
// CHECK:             %[[VAL_49:.*]] = call @matrix_csc_f64_p64i64_to_ptr8(%[[VAL_45]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_index(%[[VAL_49]], %[[VAL_41]], %[[VAL_48]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_50:.*]] = call @matrix_csc_f64_p64i64_to_ptr8(%[[VAL_45]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_values(%[[VAL_50]], %[[VAL_48]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:             %[[VAL_51:.*]] = sparse_tensor.indices %[[VAL_45]], %[[VAL_41]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_52:.*]] = sparse_tensor.values %[[VAL_45]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             scf.for %[[VAL_53:.*]] = %[[VAL_40]] to %[[VAL_48]] step %[[VAL_41]] {
// CHECK:               %[[VAL_54:.*]] = memref.load %[[VAL_43]]{{\[}}%[[VAL_53]]] : memref<?xi64>
// CHECK:               memref.store %[[VAL_54]], %[[VAL_51]]{{\[}}%[[VAL_53]]] : memref<?xi64>
// CHECK:               %[[VAL_55:.*]] = memref.load %[[VAL_44]]{{\[}}%[[VAL_53]]] : memref<?xf64>
// CHECK:               memref.store %[[VAL_55]], %[[VAL_52]]{{\[}}%[[VAL_53]]] : memref<?xf64>
// CHECK:             }
// CHECK:             %[[VAL_56:.*]] = sparse_tensor.pointers %[[VAL_45]], %[[VAL_41]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_57:.*]] = memref.load %[[VAL_43]]{{\[}}%[[VAL_40]]] : memref<?xi64>
// CHECK:             %[[VAL_58:.*]] = arith.subi %[[VAL_42]], %[[VAL_41]] : index
// CHECK:             %[[VAL_59:.*]]:3 = scf.for %[[VAL_60:.*]] = %[[VAL_40]] to %[[VAL_42]] step %[[VAL_41]] iter_args(%[[VAL_61:.*]] = %[[VAL_38]], %[[VAL_62:.*]] = %[[VAL_40]], %[[VAL_63:.*]] = %[[VAL_57]]) -> (i64, index, i64) {
// CHECK:               memref.store %[[VAL_61]], %[[VAL_56]]{{\[}}%[[VAL_60]]] : memref<?xi64>
// CHECK:               %[[VAL_64:.*]] = arith.index_cast %[[VAL_60]] : index to i64
// CHECK:               %[[VAL_65:.*]] = arith.cmpi eq, %[[VAL_63]], %[[VAL_64]] : i64
// CHECK:               %[[VAL_66:.*]] = arith.cmpi ne, %[[VAL_60]], %[[VAL_58]] : index
// CHECK:               %[[VAL_67:.*]] = arith.andi %[[VAL_66]], %[[VAL_65]] : i1
// CHECK:               %[[VAL_68:.*]]:3 = scf.if %[[VAL_67]] -> (i64, index, i64) {
// CHECK:                 %[[VAL_69:.*]] = arith.addi %[[VAL_61]], %[[VAL_39]] : i64
// CHECK:                 %[[VAL_70:.*]] = arith.addi %[[VAL_62]], %[[VAL_41]] : index
// CHECK:                 %[[VAL_71:.*]] = memref.load %[[VAL_43]]{{\[}}%[[VAL_70]]] : memref<?xi64>
// CHECK:                 scf.yield %[[VAL_69]], %[[VAL_70]], %[[VAL_71]] : i64, index, i64
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_61]], %[[VAL_62]], %[[VAL_63]] : i64, index, i64
// CHECK:               }
// CHECK:               scf.yield %[[VAL_72:.*]]#0, %[[VAL_72]]#1, %[[VAL_72]]#2 : i64, index, i64
// CHECK:             }
// CHECK:             %[[VAL_73:.*]] = arith.index_cast %[[VAL_48]] : index to i64
// CHECK:             memref.store %[[VAL_73]], %[[VAL_56]]{{\[}}%[[VAL_42]]] : memref<?xi64>
// CHECK:             return %[[VAL_45]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }
   func @vec_to_mat_arbitrary_csc(%sparse_tensor: tensor<?xf64, #CV64>) -> tensor<?x?xf64, #CSC64> {
       %answer = graphblas.diag %sparse_tensor : tensor<?xf64, #CV64> to tensor<?x?xf64, #CSC64>
       return %answer : tensor<?x?xf64, #CSC64>
   }

// CHECK:           func @mat_to_vec_arbitrary_csr(%[[VAL_76:.*]]: tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_77:.*]] = arith.constant true
// CHECK:             %[[VAL_78:.*]] = arith.constant 1 : i64
// CHECK:             %[[VAL_79:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_80:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_81:.*]] = arith.constant 2 : index
// CHECK:             %[[VAL_82:.*]] = tensor.dim %[[VAL_76]], %[[VAL_79]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_83:.*]] = sparse_tensor.pointers %[[VAL_76]], %[[VAL_80]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_84:.*]] = sparse_tensor.indices %[[VAL_76]], %[[VAL_80]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_85:.*]] = sparse_tensor.values %[[VAL_76]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_86:.*]] = call @new_vector_i64_p64i64(%[[VAL_82]]) : (index) -> tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_88:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_86]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_dim(%[[VAL_88]], %[[VAL_79]], %[[VAL_82]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_89:.*]] = scf.for %[[VAL_90:.*]] = %[[VAL_79]] to %[[VAL_82]] step %[[VAL_80]] iter_args(%[[VAL_91:.*]] = %[[VAL_79]]) -> (index) {
// CHECK:               %[[VAL_92:.*]] = arith.addi %[[VAL_90]], %[[VAL_80]] : index
// CHECK:               %[[VAL_93:.*]] = memref.load %[[VAL_83]]{{\[}}%[[VAL_90]]] : memref<?xi64>
// CHECK:               %[[VAL_94:.*]] = memref.load %[[VAL_83]]{{\[}}%[[VAL_92]]] : memref<?xi64>
// CHECK:               %[[VAL_95:.*]] = arith.index_cast %[[VAL_93]] : i64 to index
// CHECK:               %[[VAL_96:.*]] = arith.index_cast %[[VAL_94]] : i64 to index
// CHECK:               %[[VAL_97:.*]] = arith.index_cast %[[VAL_90]] : index to i64
// CHECK:               %[[VAL_98:.*]]:2 = scf.while (%[[VAL_99:.*]] = %[[VAL_95]], %[[VAL_100:.*]] = %[[VAL_77]]) : (index, i1) -> (index, i1) {
// CHECK:                 %[[VAL_101:.*]] = arith.cmpi ult, %[[VAL_99]], %[[VAL_96]] : index
// CHECK:                 %[[VAL_102:.*]] = arith.andi %[[VAL_100]], %[[VAL_101]] : i1
// CHECK:                 scf.condition(%[[VAL_102]]) %[[VAL_99]], %[[VAL_100]] : index, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_103:.*]]: index, %[[VAL_104:.*]]: i1):
// CHECK:                 %[[VAL_105:.*]] = memref.load %[[VAL_84]]{{\[}}%[[VAL_103]]] : memref<?xi64>
// CHECK:                 %[[VAL_106:.*]] = arith.cmpi ne, %[[VAL_105]], %[[VAL_97]] : i64
// CHECK:                 %[[VAL_107:.*]] = arith.addi %[[VAL_103]], %[[VAL_80]] : index
// CHECK:                 scf.yield %[[VAL_107]], %[[VAL_106]] : index, i1
// CHECK:               }
// CHECK:               %[[VAL_108:.*]] = scf.if %[[VAL_109:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_91]] : index
// CHECK:               } else {
// CHECK:                 %[[VAL_110:.*]] = arith.addi %[[VAL_91]], %[[VAL_80]] : index
// CHECK:                 scf.yield %[[VAL_110]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_111:.*]] : index
// CHECK:             }
// CHECK:             %[[VAL_112:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_86]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_pointers(%[[VAL_112]], %[[VAL_79]], %[[VAL_81]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_113:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_86]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_index(%[[VAL_113]], %[[VAL_79]], %[[VAL_114:.*]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_115:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_86]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_values(%[[VAL_115]], %[[VAL_114]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:             %[[VAL_116:.*]] = sparse_tensor.pointers %[[VAL_86]], %[[VAL_79]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_117:.*]] = arith.index_cast %[[VAL_114]] : index to i64
// CHECK:             memref.store %[[VAL_117]], %[[VAL_116]]{{\[}}%[[VAL_80]]] : memref<?xi64>
// CHECK:             %[[VAL_118:.*]] = sparse_tensor.indices %[[VAL_86]], %[[VAL_79]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_119:.*]] = sparse_tensor.values %[[VAL_86]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_120:.*]] = scf.for %[[VAL_121:.*]] = %[[VAL_79]] to %[[VAL_82]] step %[[VAL_80]] iter_args(%[[VAL_122:.*]] = %[[VAL_79]]) -> (index) {
// CHECK:               %[[VAL_123:.*]] = arith.addi %[[VAL_121]], %[[VAL_80]] : index
// CHECK:               %[[VAL_124:.*]] = memref.load %[[VAL_83]]{{\[}}%[[VAL_121]]] : memref<?xi64>
// CHECK:               %[[VAL_125:.*]] = memref.load %[[VAL_83]]{{\[}}%[[VAL_123]]] : memref<?xi64>
// CHECK:               %[[VAL_126:.*]] = arith.index_cast %[[VAL_124]] : i64 to index
// CHECK:               %[[VAL_127:.*]] = arith.index_cast %[[VAL_125]] : i64 to index
// CHECK:               %[[VAL_128:.*]] = arith.index_cast %[[VAL_121]] : index to i64
// CHECK:               %[[VAL_129:.*]]:3 = scf.while (%[[VAL_130:.*]] = %[[VAL_126]], %[[VAL_131:.*]] = %[[VAL_77]], %[[VAL_132:.*]] = %[[VAL_78]]) : (index, i1, i64) -> (index, i1, i64) {
// CHECK:                 %[[VAL_133:.*]] = arith.cmpi ult, %[[VAL_130]], %[[VAL_127]] : index
// CHECK:                 %[[VAL_134:.*]] = arith.andi %[[VAL_131]], %[[VAL_133]] : i1
// CHECK:                 scf.condition(%[[VAL_134]]) %[[VAL_130]], %[[VAL_131]], %[[VAL_132]] : index, i1, i64
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_135:.*]]: index, %[[VAL_136:.*]]: i1, %[[VAL_137:.*]]: i64):
// CHECK:                 %[[VAL_138:.*]] = memref.load %[[VAL_84]]{{\[}}%[[VAL_135]]] : memref<?xi64>
// CHECK:                 %[[VAL_139:.*]] = arith.cmpi ne, %[[VAL_138]], %[[VAL_128]] : i64
// CHECK:                 %[[VAL_140:.*]] = scf.if %[[VAL_139]] -> (i64) {
// CHECK:                   scf.yield %[[VAL_137]] : i64
// CHECK:                 } else {
// CHECK:                   %[[VAL_141:.*]] = memref.load %[[VAL_85]]{{\[}}%[[VAL_135]]] : memref<?xi64>
// CHECK:                   scf.yield %[[VAL_141]] : i64
// CHECK:                 }
// CHECK:                 %[[VAL_142:.*]] = arith.addi %[[VAL_135]], %[[VAL_80]] : index
// CHECK:                 scf.yield %[[VAL_142]], %[[VAL_139]], %[[VAL_143:.*]] : index, i1, i64
// CHECK:               }
// CHECK:               %[[VAL_144:.*]] = scf.if %[[VAL_145:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_122]] : index
// CHECK:               } else {
// CHECK:                 memref.store %[[VAL_146:.*]]#2, %[[VAL_119]]{{\[}}%[[VAL_122]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_128]], %[[VAL_118]]{{\[}}%[[VAL_122]]] : memref<?xi64>
// CHECK:                 %[[VAL_147:.*]] = arith.addi %[[VAL_122]], %[[VAL_80]] : index
// CHECK:                 scf.yield %[[VAL_147]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_148:.*]] : index
// CHECK:             }
// CHECK:             return %[[VAL_86]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }
    func @mat_to_vec_arbitrary_csr(%mat: tensor<?x?xi64, #CSC64>) -> tensor<?xi64, #CV64> {
        %vec = graphblas.diag %mat : tensor<?x?xi64, #CSC64> to tensor<?xi64, #CV64>
        return %vec : tensor<?xi64, #CV64>
    }

// CHECK:           func @mat_to_vec_arbitrary_csc(%[[VAL_149:.*]]: tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_150:.*]] = arith.constant true
// CHECK:             %[[VAL_151:.*]] = arith.constant 1 : i64
// CHECK:             %[[VAL_152:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_153:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_154:.*]] = arith.constant 2 : index
// CHECK:             %[[VAL_155:.*]] = tensor.dim %[[VAL_149]], %[[VAL_152]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_156:.*]] = sparse_tensor.pointers %[[VAL_149]], %[[VAL_153]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_157:.*]] = sparse_tensor.indices %[[VAL_149]], %[[VAL_153]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_158:.*]] = sparse_tensor.values %[[VAL_149]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_159:.*]] = call @new_vector_i64_p64i64(%[[VAL_155]]) : (index) -> tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_161:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_159]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_dim(%[[VAL_161]], %[[VAL_152]], %[[VAL_155]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_162:.*]] = scf.for %[[VAL_163:.*]] = %[[VAL_152]] to %[[VAL_155]] step %[[VAL_153]] iter_args(%[[VAL_164:.*]] = %[[VAL_152]]) -> (index) {
// CHECK:               %[[VAL_165:.*]] = arith.addi %[[VAL_163]], %[[VAL_153]] : index
// CHECK:               %[[VAL_166:.*]] = memref.load %[[VAL_156]]{{\[}}%[[VAL_163]]] : memref<?xi64>
// CHECK:               %[[VAL_167:.*]] = memref.load %[[VAL_156]]{{\[}}%[[VAL_165]]] : memref<?xi64>
// CHECK:               %[[VAL_168:.*]] = arith.index_cast %[[VAL_166]] : i64 to index
// CHECK:               %[[VAL_169:.*]] = arith.index_cast %[[VAL_167]] : i64 to index
// CHECK:               %[[VAL_170:.*]] = arith.index_cast %[[VAL_163]] : index to i64
// CHECK:               %[[VAL_171:.*]]:2 = scf.while (%[[VAL_172:.*]] = %[[VAL_168]], %[[VAL_173:.*]] = %[[VAL_150]]) : (index, i1) -> (index, i1) {
// CHECK:                 %[[VAL_174:.*]] = arith.cmpi ult, %[[VAL_172]], %[[VAL_169]] : index
// CHECK:                 %[[VAL_175:.*]] = arith.andi %[[VAL_173]], %[[VAL_174]] : i1
// CHECK:                 scf.condition(%[[VAL_175]]) %[[VAL_172]], %[[VAL_173]] : index, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_176:.*]]: index, %[[VAL_177:.*]]: i1):
// CHECK:                 %[[VAL_178:.*]] = memref.load %[[VAL_157]]{{\[}}%[[VAL_176]]] : memref<?xi64>
// CHECK:                 %[[VAL_179:.*]] = arith.cmpi ne, %[[VAL_178]], %[[VAL_170]] : i64
// CHECK:                 %[[VAL_180:.*]] = arith.addi %[[VAL_176]], %[[VAL_153]] : index
// CHECK:                 scf.yield %[[VAL_180]], %[[VAL_179]] : index, i1
// CHECK:               }
// CHECK:               %[[VAL_181:.*]] = scf.if %[[VAL_182:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_164]] : index
// CHECK:               } else {
// CHECK:                 %[[VAL_183:.*]] = arith.addi %[[VAL_164]], %[[VAL_153]] : index
// CHECK:                 scf.yield %[[VAL_183]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_184:.*]] : index
// CHECK:             }
// CHECK:             %[[VAL_185:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_159]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_pointers(%[[VAL_185]], %[[VAL_152]], %[[VAL_154]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_186:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_159]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_index(%[[VAL_186]], %[[VAL_152]], %[[VAL_187:.*]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_188:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_159]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_values(%[[VAL_188]], %[[VAL_187]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:             %[[VAL_189:.*]] = sparse_tensor.pointers %[[VAL_159]], %[[VAL_152]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_190:.*]] = arith.index_cast %[[VAL_187]] : index to i64
// CHECK:             memref.store %[[VAL_190]], %[[VAL_189]]{{\[}}%[[VAL_153]]] : memref<?xi64>
// CHECK:             %[[VAL_191:.*]] = sparse_tensor.indices %[[VAL_159]], %[[VAL_152]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_192:.*]] = sparse_tensor.values %[[VAL_159]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_193:.*]] = scf.for %[[VAL_194:.*]] = %[[VAL_152]] to %[[VAL_155]] step %[[VAL_153]] iter_args(%[[VAL_195:.*]] = %[[VAL_152]]) -> (index) {
// CHECK:               %[[VAL_196:.*]] = arith.addi %[[VAL_194]], %[[VAL_153]] : index
// CHECK:               %[[VAL_197:.*]] = memref.load %[[VAL_156]]{{\[}}%[[VAL_194]]] : memref<?xi64>
// CHECK:               %[[VAL_198:.*]] = memref.load %[[VAL_156]]{{\[}}%[[VAL_196]]] : memref<?xi64>
// CHECK:               %[[VAL_199:.*]] = arith.index_cast %[[VAL_197]] : i64 to index
// CHECK:               %[[VAL_200:.*]] = arith.index_cast %[[VAL_198]] : i64 to index
// CHECK:               %[[VAL_201:.*]] = arith.index_cast %[[VAL_194]] : index to i64
// CHECK:               %[[VAL_202:.*]]:3 = scf.while (%[[VAL_203:.*]] = %[[VAL_199]], %[[VAL_204:.*]] = %[[VAL_150]], %[[VAL_205:.*]] = %[[VAL_151]]) : (index, i1, i64) -> (index, i1, i64) {
// CHECK:                 %[[VAL_206:.*]] = arith.cmpi ult, %[[VAL_203]], %[[VAL_200]] : index
// CHECK:                 %[[VAL_207:.*]] = arith.andi %[[VAL_204]], %[[VAL_206]] : i1
// CHECK:                 scf.condition(%[[VAL_207]]) %[[VAL_203]], %[[VAL_204]], %[[VAL_205]] : index, i1, i64
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_208:.*]]: index, %[[VAL_209:.*]]: i1, %[[VAL_210:.*]]: i64):
// CHECK:                 %[[VAL_211:.*]] = memref.load %[[VAL_157]]{{\[}}%[[VAL_208]]] : memref<?xi64>
// CHECK:                 %[[VAL_212:.*]] = arith.cmpi ne, %[[VAL_211]], %[[VAL_201]] : i64
// CHECK:                 %[[VAL_213:.*]] = scf.if %[[VAL_212]] -> (i64) {
// CHECK:                   scf.yield %[[VAL_210]] : i64
// CHECK:                 } else {
// CHECK:                   %[[VAL_214:.*]] = memref.load %[[VAL_158]]{{\[}}%[[VAL_208]]] : memref<?xi64>
// CHECK:                   scf.yield %[[VAL_214]] : i64
// CHECK:                 }
// CHECK:                 %[[VAL_215:.*]] = arith.addi %[[VAL_208]], %[[VAL_153]] : index
// CHECK:                 scf.yield %[[VAL_215]], %[[VAL_212]], %[[VAL_216:.*]] : index, i1, i64
// CHECK:               }
// CHECK:               %[[VAL_217:.*]] = scf.if %[[VAL_218:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_195]] : index
// CHECK:               } else {
// CHECK:                 memref.store %[[VAL_219:.*]]#2, %[[VAL_192]]{{\[}}%[[VAL_195]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_201]], %[[VAL_191]]{{\[}}%[[VAL_195]]] : memref<?xi64>
// CHECK:                 %[[VAL_220:.*]] = arith.addi %[[VAL_195]], %[[VAL_153]] : index
// CHECK:                 scf.yield %[[VAL_220]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_221:.*]] : index
// CHECK:             }
// CHECK:             return %[[VAL_159]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }
    func @mat_to_vec_arbitrary_csc(%mat: tensor<?x?xi64, #CSC64>) -> tensor<?xi64, #CV64> {
        %vec = graphblas.diag %mat : tensor<?x?xi64, #CSC64> to tensor<?xi64, #CV64>
        return %vec : tensor<?xi64, #CV64>
    }

}
