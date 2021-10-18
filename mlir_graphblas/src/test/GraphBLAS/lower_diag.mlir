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
// CHECK:           func private @assign_rev(!llvm.ptr<i8>, index, index)
// CHECK:           func private @vector_i64_p64i64_to_ptr8(tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           func private @new_vector_i64_p64i64(index) -> tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           func @vec_to_mat_fixed_csr(%[[VAL_0:.*]]: tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_1:.*]] = constant 0 : i64
// CHECK:             %[[VAL_2:.*]] = constant 1 : i64
// CHECK:             %[[VAL_3:.*]] = constant 0 : index
// CHECK:             %[[VAL_4:.*]] = constant 1 : index
// CHECK:             %[[VAL_5:.*]] = constant 6 : index
// CHECK:             %[[VAL_6:.*]] = constant 7 : index
// CHECK:             %[[VAL_7:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_3]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_8:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             %[[VAL_9:.*]] = call @new_matrix_csr_f64_p64i64(%[[VAL_6]], %[[VAL_6]]) : (index, index) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_10:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_9]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @assign_rev(%[[VAL_10]], %[[VAL_3]], %[[VAL_3]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_11:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_9]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @assign_rev(%[[VAL_11]], %[[VAL_4]], %[[VAL_4]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_12:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_3]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_13:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_4]]] : memref<?xi64>
// CHECK:             %[[VAL_14:.*]] = index_cast %[[VAL_13]] : i64 to index
// CHECK:             %[[VAL_15:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_9]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_index(%[[VAL_15]], %[[VAL_4]], %[[VAL_14]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_16:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_9]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_values(%[[VAL_16]], %[[VAL_14]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:             %[[VAL_17:.*]] = sparse_tensor.indices %[[VAL_9]], %[[VAL_4]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_18:.*]] = sparse_tensor.values %[[VAL_9]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             scf.for %[[VAL_19:.*]] = %[[VAL_3]] to %[[VAL_14]] step %[[VAL_4]] {
// CHECK:               %[[VAL_20:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_19]]] : memref<?xi64>
// CHECK:               memref.store %[[VAL_20]], %[[VAL_17]]{{\[}}%[[VAL_19]]] : memref<?xi64>
// CHECK:               %[[VAL_21:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_19]]] : memref<?xf64>
// CHECK:               memref.store %[[VAL_21]], %[[VAL_18]]{{\[}}%[[VAL_19]]] : memref<?xf64>
// CHECK:             }
// CHECK:             %[[VAL_22:.*]] = sparse_tensor.pointers %[[VAL_9]], %[[VAL_4]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_23:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_3]]] : memref<?xi64>
// CHECK:             %[[VAL_24:.*]]:3 = scf.for %[[VAL_25:.*]] = %[[VAL_3]] to %[[VAL_6]] step %[[VAL_4]] iter_args(%[[VAL_26:.*]] = %[[VAL_1]], %[[VAL_27:.*]] = %[[VAL_3]], %[[VAL_28:.*]] = %[[VAL_23]]) -> (i64, index, i64) {
// CHECK:               memref.store %[[VAL_26]], %[[VAL_22]]{{\[}}%[[VAL_25]]] : memref<?xi64>
// CHECK:               %[[VAL_29:.*]] = index_cast %[[VAL_25]] : index to i64
// CHECK:               %[[VAL_30:.*]] = cmpi eq, %[[VAL_28]], %[[VAL_29]] : i64
// CHECK:               %[[VAL_31:.*]] = cmpi ne, %[[VAL_25]], %[[VAL_5]] : index
// CHECK:               %[[VAL_32:.*]] = and %[[VAL_31]], %[[VAL_30]] : i1
// CHECK:               %[[VAL_33:.*]]:3 = scf.if %[[VAL_32]] -> (i64, index, i64) {
// CHECK:                 %[[VAL_34:.*]] = addi %[[VAL_26]], %[[VAL_2]] : i64
// CHECK:                 %[[VAL_35:.*]] = addi %[[VAL_27]], %[[VAL_4]] : index
// CHECK:                 %[[VAL_36:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_35]]] : memref<?xi64>
// CHECK:                 scf.yield %[[VAL_34]], %[[VAL_35]], %[[VAL_36]] : i64, index, i64
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_26]], %[[VAL_27]], %[[VAL_28]] : i64, index, i64
// CHECK:               }
// CHECK:               scf.yield %[[VAL_37:.*]]#0, %[[VAL_37]]#1, %[[VAL_37]]#2 : i64, index, i64
// CHECK:             }
// CHECK:             memref.store %[[VAL_13]], %[[VAL_22]]{{\[}}%[[VAL_6]]] : memref<?xi64>
// CHECK:             return %[[VAL_9]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }
   func @vec_to_mat_fixed_csr(%sparse_tensor: tensor<7xf64, #CV64>) -> tensor<7x7xf64, #CSR64> {
       %answer = graphblas.diag %sparse_tensor : tensor<7xf64, #CV64> to tensor<7x7xf64, #CSR64>
       return %answer : tensor<7x7xf64, #CSR64>
   }

// CHECK:           func @vec_to_mat_fixed_csc(%[[VAL_38:.*]]: tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_39:.*]] = constant 0 : i64
// CHECK:             %[[VAL_40:.*]] = constant 1 : i64
// CHECK:             %[[VAL_41:.*]] = constant 0 : index
// CHECK:             %[[VAL_42:.*]] = constant 1 : index
// CHECK:             %[[VAL_43:.*]] = constant 6 : index
// CHECK:             %[[VAL_44:.*]] = constant 7 : index
// CHECK:             %[[VAL_45:.*]] = sparse_tensor.indices %[[VAL_38]], %[[VAL_41]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_46:.*]] = sparse_tensor.values %[[VAL_38]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             %[[VAL_47:.*]] = call @new_matrix_csc_f64_p64i64(%[[VAL_44]], %[[VAL_44]]) : (index, index) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_48:.*]] = call @matrix_csc_f64_p64i64_to_ptr8(%[[VAL_47]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @assign_rev(%[[VAL_48]], %[[VAL_41]], %[[VAL_42]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_49:.*]] = call @matrix_csc_f64_p64i64_to_ptr8(%[[VAL_47]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @assign_rev(%[[VAL_49]], %[[VAL_42]], %[[VAL_41]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_50:.*]] = sparse_tensor.pointers %[[VAL_38]], %[[VAL_41]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_51:.*]] = memref.load %[[VAL_50]]{{\[}}%[[VAL_42]]] : memref<?xi64>
// CHECK:             %[[VAL_52:.*]] = index_cast %[[VAL_51]] : i64 to index
// CHECK:             %[[VAL_53:.*]] = call @matrix_csc_f64_p64i64_to_ptr8(%[[VAL_47]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_index(%[[VAL_53]], %[[VAL_42]], %[[VAL_52]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_54:.*]] = call @matrix_csc_f64_p64i64_to_ptr8(%[[VAL_47]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_values(%[[VAL_54]], %[[VAL_52]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:             %[[VAL_55:.*]] = sparse_tensor.indices %[[VAL_47]], %[[VAL_42]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_56:.*]] = sparse_tensor.values %[[VAL_47]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             scf.for %[[VAL_57:.*]] = %[[VAL_41]] to %[[VAL_52]] step %[[VAL_42]] {
// CHECK:               %[[VAL_58:.*]] = memref.load %[[VAL_45]]{{\[}}%[[VAL_57]]] : memref<?xi64>
// CHECK:               memref.store %[[VAL_58]], %[[VAL_55]]{{\[}}%[[VAL_57]]] : memref<?xi64>
// CHECK:               %[[VAL_59:.*]] = memref.load %[[VAL_46]]{{\[}}%[[VAL_57]]] : memref<?xf64>
// CHECK:               memref.store %[[VAL_59]], %[[VAL_56]]{{\[}}%[[VAL_57]]] : memref<?xf64>
// CHECK:             }
// CHECK:             %[[VAL_60:.*]] = sparse_tensor.pointers %[[VAL_47]], %[[VAL_42]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_61:.*]] = memref.load %[[VAL_45]]{{\[}}%[[VAL_41]]] : memref<?xi64>
// CHECK:             %[[VAL_62:.*]]:3 = scf.for %[[VAL_63:.*]] = %[[VAL_41]] to %[[VAL_44]] step %[[VAL_42]] iter_args(%[[VAL_64:.*]] = %[[VAL_39]], %[[VAL_65:.*]] = %[[VAL_41]], %[[VAL_66:.*]] = %[[VAL_61]]) -> (i64, index, i64) {
// CHECK:               memref.store %[[VAL_64]], %[[VAL_60]]{{\[}}%[[VAL_63]]] : memref<?xi64>
// CHECK:               %[[VAL_67:.*]] = index_cast %[[VAL_63]] : index to i64
// CHECK:               %[[VAL_68:.*]] = cmpi eq, %[[VAL_66]], %[[VAL_67]] : i64
// CHECK:               %[[VAL_69:.*]] = cmpi ne, %[[VAL_63]], %[[VAL_43]] : index
// CHECK:               %[[VAL_70:.*]] = and %[[VAL_69]], %[[VAL_68]] : i1
// CHECK:               %[[VAL_71:.*]]:3 = scf.if %[[VAL_70]] -> (i64, index, i64) {
// CHECK:                 %[[VAL_72:.*]] = addi %[[VAL_64]], %[[VAL_40]] : i64
// CHECK:                 %[[VAL_73:.*]] = addi %[[VAL_65]], %[[VAL_42]] : index
// CHECK:                 %[[VAL_74:.*]] = memref.load %[[VAL_45]]{{\[}}%[[VAL_73]]] : memref<?xi64>
// CHECK:                 scf.yield %[[VAL_72]], %[[VAL_73]], %[[VAL_74]] : i64, index, i64
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_64]], %[[VAL_65]], %[[VAL_66]] : i64, index, i64
// CHECK:               }
// CHECK:               scf.yield %[[VAL_75:.*]]#0, %[[VAL_75]]#1, %[[VAL_75]]#2 : i64, index, i64
// CHECK:             }
// CHECK:             memref.store %[[VAL_51]], %[[VAL_60]]{{\[}}%[[VAL_44]]] : memref<?xi64>
// CHECK:             return %[[VAL_47]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }
   func @vec_to_mat_fixed_csc(%sparse_tensor: tensor<7xf64, #CV64>) -> tensor<7x7xf64, #CSC64> {
       %answer = graphblas.diag %sparse_tensor : tensor<7xf64, #CV64> to tensor<7x7xf64, #CSC64>
       return %answer : tensor<7x7xf64, #CSC64>
   }
   
// CHECK:           func @mat_to_vec_fixed_csr(%[[VAL_76:.*]]: tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_77:.*]] = constant true
// CHECK:             %[[VAL_78:.*]] = constant 1 : i64
// CHECK:             %[[VAL_79:.*]] = constant 0 : index
// CHECK:             %[[VAL_80:.*]] = constant 1 : index
// CHECK:             %[[VAL_81:.*]] = constant 2 : index
// CHECK:             %[[VAL_82:.*]] = constant 7 : index
// CHECK:             %[[VAL_83:.*]] = sparse_tensor.pointers %[[VAL_76]], %[[VAL_80]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_84:.*]] = sparse_tensor.indices %[[VAL_76]], %[[VAL_80]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_85:.*]] = sparse_tensor.values %[[VAL_76]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_86:.*]] = call @new_vector_i64_p64i64(%[[VAL_82]]) : (index) -> tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_87:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_86]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @assign_rev(%[[VAL_87]], %[[VAL_79]], %[[VAL_79]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_88:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_86]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_dim(%[[VAL_88]], %[[VAL_79]], %[[VAL_82]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_89:.*]] = scf.for %[[VAL_90:.*]] = %[[VAL_79]] to %[[VAL_82]] step %[[VAL_80]] iter_args(%[[VAL_91:.*]] = %[[VAL_79]]) -> (index) {
// CHECK:               %[[VAL_92:.*]] = addi %[[VAL_90]], %[[VAL_80]] : index
// CHECK:               %[[VAL_93:.*]] = memref.load %[[VAL_83]]{{\[}}%[[VAL_90]]] : memref<?xi64>
// CHECK:               %[[VAL_94:.*]] = memref.load %[[VAL_83]]{{\[}}%[[VAL_92]]] : memref<?xi64>
// CHECK:               %[[VAL_95:.*]] = index_cast %[[VAL_93]] : i64 to index
// CHECK:               %[[VAL_96:.*]] = index_cast %[[VAL_94]] : i64 to index
// CHECK:               %[[VAL_97:.*]] = index_cast %[[VAL_90]] : index to i64
// CHECK:               %[[VAL_98:.*]]:2 = scf.while (%[[VAL_99:.*]] = %[[VAL_95]], %[[VAL_100:.*]] = %[[VAL_77]]) : (index, i1) -> (index, i1) {
// CHECK:                 %[[VAL_101:.*]] = cmpi ult, %[[VAL_99]], %[[VAL_96]] : index
// CHECK:                 %[[VAL_102:.*]] = and %[[VAL_100]], %[[VAL_101]] : i1
// CHECK:                 scf.condition(%[[VAL_102]]) %[[VAL_99]], %[[VAL_100]] : index, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_103:.*]]: index, %[[VAL_104:.*]]: i1):
// CHECK:                 %[[VAL_105:.*]] = memref.load %[[VAL_84]]{{\[}}%[[VAL_103]]] : memref<?xi64>
// CHECK:                 %[[VAL_106:.*]] = cmpi ne, %[[VAL_105]], %[[VAL_97]] : i64
// CHECK:                 %[[VAL_107:.*]] = addi %[[VAL_103]], %[[VAL_80]] : index
// CHECK:                 scf.yield %[[VAL_107]], %[[VAL_106]] : index, i1
// CHECK:               }
// CHECK:               %[[VAL_108:.*]] = scf.if %[[VAL_109:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_91]] : index
// CHECK:               } else {
// CHECK:                 %[[VAL_110:.*]] = addi %[[VAL_91]], %[[VAL_80]] : index
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
// CHECK:             %[[VAL_117:.*]] = index_cast %[[VAL_114]] : index to i64
// CHECK:             memref.store %[[VAL_117]], %[[VAL_116]]{{\[}}%[[VAL_80]]] : memref<?xi64>
// CHECK:             %[[VAL_118:.*]] = sparse_tensor.indices %[[VAL_86]], %[[VAL_79]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_119:.*]] = sparse_tensor.values %[[VAL_86]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_120:.*]] = scf.for %[[VAL_121:.*]] = %[[VAL_79]] to %[[VAL_82]] step %[[VAL_80]] iter_args(%[[VAL_122:.*]] = %[[VAL_79]]) -> (index) {
// CHECK:               %[[VAL_123:.*]] = addi %[[VAL_121]], %[[VAL_80]] : index
// CHECK:               %[[VAL_124:.*]] = memref.load %[[VAL_83]]{{\[}}%[[VAL_121]]] : memref<?xi64>
// CHECK:               %[[VAL_125:.*]] = memref.load %[[VAL_83]]{{\[}}%[[VAL_123]]] : memref<?xi64>
// CHECK:               %[[VAL_126:.*]] = index_cast %[[VAL_124]] : i64 to index
// CHECK:               %[[VAL_127:.*]] = index_cast %[[VAL_125]] : i64 to index
// CHECK:               %[[VAL_128:.*]] = index_cast %[[VAL_121]] : index to i64
// CHECK:               %[[VAL_129:.*]]:3 = scf.while (%[[VAL_130:.*]] = %[[VAL_126]], %[[VAL_131:.*]] = %[[VAL_77]], %[[VAL_132:.*]] = %[[VAL_78]]) : (index, i1, i64) -> (index, i1, i64) {
// CHECK:                 %[[VAL_133:.*]] = cmpi ult, %[[VAL_130]], %[[VAL_127]] : index
// CHECK:                 %[[VAL_134:.*]] = and %[[VAL_131]], %[[VAL_133]] : i1
// CHECK:                 scf.condition(%[[VAL_134]]) %[[VAL_130]], %[[VAL_131]], %[[VAL_132]] : index, i1, i64
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_135:.*]]: index, %[[VAL_136:.*]]: i1, %[[VAL_137:.*]]: i64):
// CHECK:                 %[[VAL_138:.*]] = memref.load %[[VAL_84]]{{\[}}%[[VAL_135]]] : memref<?xi64>
// CHECK:                 %[[VAL_139:.*]] = cmpi ne, %[[VAL_138]], %[[VAL_128]] : i64
// CHECK:                 %[[VAL_140:.*]] = scf.if %[[VAL_139]] -> (i64) {
// CHECK:                   scf.yield %[[VAL_137]] : i64
// CHECK:                 } else {
// CHECK:                   %[[VAL_141:.*]] = memref.load %[[VAL_85]]{{\[}}%[[VAL_135]]] : memref<?xi64>
// CHECK:                   scf.yield %[[VAL_141]] : i64
// CHECK:                 }
// CHECK:                 %[[VAL_142:.*]] = addi %[[VAL_135]], %[[VAL_80]] : index
// CHECK:                 scf.yield %[[VAL_142]], %[[VAL_139]], %[[VAL_143:.*]] : index, i1, i64
// CHECK:               }
// CHECK:               %[[VAL_144:.*]] = scf.if %[[VAL_145:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_122]] : index
// CHECK:               } else {
// CHECK:                 memref.store %[[VAL_146:.*]]#2, %[[VAL_119]]{{\[}}%[[VAL_122]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_128]], %[[VAL_118]]{{\[}}%[[VAL_122]]] : memref<?xi64>
// CHECK:                 %[[VAL_147:.*]] = addi %[[VAL_122]], %[[VAL_80]] : index
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
// CHECK:             %[[VAL_150:.*]] = constant true
// CHECK:             %[[VAL_151:.*]] = constant 1 : i64
// CHECK:             %[[VAL_152:.*]] = constant 0 : index
// CHECK:             %[[VAL_153:.*]] = constant 1 : index
// CHECK:             %[[VAL_154:.*]] = constant 2 : index
// CHECK:             %[[VAL_155:.*]] = constant 7 : index
// CHECK:             %[[VAL_156:.*]] = sparse_tensor.pointers %[[VAL_149]], %[[VAL_153]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_157:.*]] = sparse_tensor.indices %[[VAL_149]], %[[VAL_153]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_158:.*]] = sparse_tensor.values %[[VAL_149]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_159:.*]] = call @new_vector_i64_p64i64(%[[VAL_155]]) : (index) -> tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_160:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_159]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @assign_rev(%[[VAL_160]], %[[VAL_152]], %[[VAL_152]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_161:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_159]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_dim(%[[VAL_161]], %[[VAL_152]], %[[VAL_155]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_162:.*]] = scf.for %[[VAL_163:.*]] = %[[VAL_152]] to %[[VAL_155]] step %[[VAL_153]] iter_args(%[[VAL_164:.*]] = %[[VAL_152]]) -> (index) {
// CHECK:               %[[VAL_165:.*]] = addi %[[VAL_163]], %[[VAL_153]] : index
// CHECK:               %[[VAL_166:.*]] = memref.load %[[VAL_156]]{{\[}}%[[VAL_163]]] : memref<?xi64>
// CHECK:               %[[VAL_167:.*]] = memref.load %[[VAL_156]]{{\[}}%[[VAL_165]]] : memref<?xi64>
// CHECK:               %[[VAL_168:.*]] = index_cast %[[VAL_166]] : i64 to index
// CHECK:               %[[VAL_169:.*]] = index_cast %[[VAL_167]] : i64 to index
// CHECK:               %[[VAL_170:.*]] = index_cast %[[VAL_163]] : index to i64
// CHECK:               %[[VAL_171:.*]]:2 = scf.while (%[[VAL_172:.*]] = %[[VAL_168]], %[[VAL_173:.*]] = %[[VAL_150]]) : (index, i1) -> (index, i1) {
// CHECK:                 %[[VAL_174:.*]] = cmpi ult, %[[VAL_172]], %[[VAL_169]] : index
// CHECK:                 %[[VAL_175:.*]] = and %[[VAL_173]], %[[VAL_174]] : i1
// CHECK:                 scf.condition(%[[VAL_175]]) %[[VAL_172]], %[[VAL_173]] : index, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_176:.*]]: index, %[[VAL_177:.*]]: i1):
// CHECK:                 %[[VAL_178:.*]] = memref.load %[[VAL_157]]{{\[}}%[[VAL_176]]] : memref<?xi64>
// CHECK:                 %[[VAL_179:.*]] = cmpi ne, %[[VAL_178]], %[[VAL_170]] : i64
// CHECK:                 %[[VAL_180:.*]] = addi %[[VAL_176]], %[[VAL_153]] : index
// CHECK:                 scf.yield %[[VAL_180]], %[[VAL_179]] : index, i1
// CHECK:               }
// CHECK:               %[[VAL_181:.*]] = scf.if %[[VAL_182:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_164]] : index
// CHECK:               } else {
// CHECK:                 %[[VAL_183:.*]] = addi %[[VAL_164]], %[[VAL_153]] : index
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
// CHECK:             %[[VAL_190:.*]] = index_cast %[[VAL_187]] : index to i64
// CHECK:             memref.store %[[VAL_190]], %[[VAL_189]]{{\[}}%[[VAL_153]]] : memref<?xi64>
// CHECK:             %[[VAL_191:.*]] = sparse_tensor.indices %[[VAL_159]], %[[VAL_152]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_192:.*]] = sparse_tensor.values %[[VAL_159]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_193:.*]] = scf.for %[[VAL_194:.*]] = %[[VAL_152]] to %[[VAL_155]] step %[[VAL_153]] iter_args(%[[VAL_195:.*]] = %[[VAL_152]]) -> (index) {
// CHECK:               %[[VAL_196:.*]] = addi %[[VAL_194]], %[[VAL_153]] : index
// CHECK:               %[[VAL_197:.*]] = memref.load %[[VAL_156]]{{\[}}%[[VAL_194]]] : memref<?xi64>
// CHECK:               %[[VAL_198:.*]] = memref.load %[[VAL_156]]{{\[}}%[[VAL_196]]] : memref<?xi64>
// CHECK:               %[[VAL_199:.*]] = index_cast %[[VAL_197]] : i64 to index
// CHECK:               %[[VAL_200:.*]] = index_cast %[[VAL_198]] : i64 to index
// CHECK:               %[[VAL_201:.*]] = index_cast %[[VAL_194]] : index to i64
// CHECK:               %[[VAL_202:.*]]:3 = scf.while (%[[VAL_203:.*]] = %[[VAL_199]], %[[VAL_204:.*]] = %[[VAL_150]], %[[VAL_205:.*]] = %[[VAL_151]]) : (index, i1, i64) -> (index, i1, i64) {
// CHECK:                 %[[VAL_206:.*]] = cmpi ult, %[[VAL_203]], %[[VAL_200]] : index
// CHECK:                 %[[VAL_207:.*]] = and %[[VAL_204]], %[[VAL_206]] : i1
// CHECK:                 scf.condition(%[[VAL_207]]) %[[VAL_203]], %[[VAL_204]], %[[VAL_205]] : index, i1, i64
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_208:.*]]: index, %[[VAL_209:.*]]: i1, %[[VAL_210:.*]]: i64):
// CHECK:                 %[[VAL_211:.*]] = memref.load %[[VAL_157]]{{\[}}%[[VAL_208]]] : memref<?xi64>
// CHECK:                 %[[VAL_212:.*]] = cmpi ne, %[[VAL_211]], %[[VAL_201]] : i64
// CHECK:                 %[[VAL_213:.*]] = scf.if %[[VAL_212]] -> (i64) {
// CHECK:                   scf.yield %[[VAL_210]] : i64
// CHECK:                 } else {
// CHECK:                   %[[VAL_214:.*]] = memref.load %[[VAL_158]]{{\[}}%[[VAL_208]]] : memref<?xi64>
// CHECK:                   scf.yield %[[VAL_214]] : i64
// CHECK:                 }
// CHECK:                 %[[VAL_215:.*]] = addi %[[VAL_208]], %[[VAL_153]] : index
// CHECK:                 scf.yield %[[VAL_215]], %[[VAL_212]], %[[VAL_216:.*]] : index, i1, i64
// CHECK:               }
// CHECK:               %[[VAL_217:.*]] = scf.if %[[VAL_218:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_195]] : index
// CHECK:               } else {
// CHECK:                 memref.store %[[VAL_219:.*]]#2, %[[VAL_192]]{{\[}}%[[VAL_195]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_201]], %[[VAL_191]]{{\[}}%[[VAL_195]]] : memref<?xi64>
// CHECK:                 %[[VAL_220:.*]] = addi %[[VAL_195]], %[[VAL_153]] : index
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
// CHECK:           func private @assign_rev(!llvm.ptr<i8>, index, index)
// CHECK:           func private @vector_i64_p64i64_to_ptr8(tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           func private @new_vector_i64_p64i64(index) -> tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           func @vec_to_mat_arbitrary_csr(%[[VAL_0:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_1:.*]] = constant 0 : i64
// CHECK:             %[[VAL_2:.*]] = constant 1 : i64
// CHECK:             %[[VAL_3:.*]] = constant 0 : index
// CHECK:             %[[VAL_4:.*]] = constant 1 : index
// CHECK:             %[[VAL_5:.*]] = tensor.dim %[[VAL_0]], %[[VAL_3]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_6:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_3]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             %[[VAL_8:.*]] = call @new_matrix_csr_f64_p64i64(%[[VAL_5]], %[[VAL_5]]) : (index, index) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_9:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_8]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @assign_rev(%[[VAL_9]], %[[VAL_3]], %[[VAL_3]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_10:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_8]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @assign_rev(%[[VAL_10]], %[[VAL_4]], %[[VAL_4]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_11:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_3]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_12:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_4]]] : memref<?xi64>
// CHECK:             %[[VAL_13:.*]] = index_cast %[[VAL_12]] : i64 to index
// CHECK:             %[[VAL_14:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_8]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_index(%[[VAL_14]], %[[VAL_4]], %[[VAL_13]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_15:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_8]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_values(%[[VAL_15]], %[[VAL_13]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:             %[[VAL_16:.*]] = sparse_tensor.indices %[[VAL_8]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_17:.*]] = sparse_tensor.values %[[VAL_8]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             scf.for %[[VAL_18:.*]] = %[[VAL_3]] to %[[VAL_13]] step %[[VAL_4]] {
// CHECK:               %[[VAL_19:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_18]]] : memref<?xi64>
// CHECK:               memref.store %[[VAL_19]], %[[VAL_16]]{{\[}}%[[VAL_18]]] : memref<?xi64>
// CHECK:               %[[VAL_20:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_18]]] : memref<?xf64>
// CHECK:               memref.store %[[VAL_20]], %[[VAL_17]]{{\[}}%[[VAL_18]]] : memref<?xf64>
// CHECK:             }
// CHECK:             %[[VAL_21:.*]] = sparse_tensor.pointers %[[VAL_8]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_22:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_3]]] : memref<?xi64>
// CHECK:             %[[VAL_23:.*]] = subi %[[VAL_5]], %[[VAL_4]] : index
// CHECK:             %[[VAL_24:.*]]:3 = scf.for %[[VAL_25:.*]] = %[[VAL_3]] to %[[VAL_5]] step %[[VAL_4]] iter_args(%[[VAL_26:.*]] = %[[VAL_1]], %[[VAL_27:.*]] = %[[VAL_3]], %[[VAL_28:.*]] = %[[VAL_22]]) -> (i64, index, i64) {
// CHECK:               memref.store %[[VAL_26]], %[[VAL_21]]{{\[}}%[[VAL_25]]] : memref<?xi64>
// CHECK:               %[[VAL_29:.*]] = index_cast %[[VAL_25]] : index to i64
// CHECK:               %[[VAL_30:.*]] = cmpi eq, %[[VAL_28]], %[[VAL_29]] : i64
// CHECK:               %[[VAL_31:.*]] = cmpi ne, %[[VAL_25]], %[[VAL_23]] : index
// CHECK:               %[[VAL_32:.*]] = and %[[VAL_31]], %[[VAL_30]] : i1
// CHECK:               %[[VAL_33:.*]]:3 = scf.if %[[VAL_32]] -> (i64, index, i64) {
// CHECK:                 %[[VAL_34:.*]] = addi %[[VAL_26]], %[[VAL_2]] : i64
// CHECK:                 %[[VAL_35:.*]] = addi %[[VAL_27]], %[[VAL_4]] : index
// CHECK:                 %[[VAL_36:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_35]]] : memref<?xi64>
// CHECK:                 scf.yield %[[VAL_34]], %[[VAL_35]], %[[VAL_36]] : i64, index, i64
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_26]], %[[VAL_27]], %[[VAL_28]] : i64, index, i64
// CHECK:               }
// CHECK:               scf.yield %[[VAL_37:.*]]#0, %[[VAL_37]]#1, %[[VAL_37]]#2 : i64, index, i64
// CHECK:             }
// CHECK:             memref.store %[[VAL_12]], %[[VAL_21]]{{\[}}%[[VAL_5]]] : memref<?xi64>
// CHECK:             return %[[VAL_8]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }
   func @vec_to_mat_arbitrary_csr(%sparse_tensor: tensor<?xf64, #CV64>) -> tensor<?x?xf64, #CSR64> {
       %answer = graphblas.diag %sparse_tensor : tensor<?xf64, #CV64> to tensor<?x?xf64, #CSR64>
       return %answer : tensor<?x?xf64, #CSR64>
   }

// CHECK:           func @vec_to_mat_arbitrary_csc(%[[VAL_38:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_39:.*]] = constant 0 : i64
// CHECK:             %[[VAL_40:.*]] = constant 1 : i64
// CHECK:             %[[VAL_41:.*]] = constant 0 : index
// CHECK:             %[[VAL_42:.*]] = constant 1 : index
// CHECK:             %[[VAL_43:.*]] = tensor.dim %[[VAL_38]], %[[VAL_41]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_44:.*]] = sparse_tensor.indices %[[VAL_38]], %[[VAL_41]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_45:.*]] = sparse_tensor.values %[[VAL_38]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             %[[VAL_46:.*]] = call @new_matrix_csc_f64_p64i64(%[[VAL_43]], %[[VAL_43]]) : (index, index) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_47:.*]] = call @matrix_csc_f64_p64i64_to_ptr8(%[[VAL_46]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @assign_rev(%[[VAL_47]], %[[VAL_41]], %[[VAL_42]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_48:.*]] = call @matrix_csc_f64_p64i64_to_ptr8(%[[VAL_46]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @assign_rev(%[[VAL_48]], %[[VAL_42]], %[[VAL_41]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_49:.*]] = sparse_tensor.pointers %[[VAL_38]], %[[VAL_41]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_50:.*]] = memref.load %[[VAL_49]]{{\[}}%[[VAL_42]]] : memref<?xi64>
// CHECK:             %[[VAL_51:.*]] = index_cast %[[VAL_50]] : i64 to index
// CHECK:             %[[VAL_52:.*]] = call @matrix_csc_f64_p64i64_to_ptr8(%[[VAL_46]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_index(%[[VAL_52]], %[[VAL_42]], %[[VAL_51]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_53:.*]] = call @matrix_csc_f64_p64i64_to_ptr8(%[[VAL_46]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_values(%[[VAL_53]], %[[VAL_51]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:             %[[VAL_54:.*]] = sparse_tensor.indices %[[VAL_46]], %[[VAL_42]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_55:.*]] = sparse_tensor.values %[[VAL_46]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             scf.for %[[VAL_56:.*]] = %[[VAL_41]] to %[[VAL_51]] step %[[VAL_42]] {
// CHECK:               %[[VAL_57:.*]] = memref.load %[[VAL_44]]{{\[}}%[[VAL_56]]] : memref<?xi64>
// CHECK:               memref.store %[[VAL_57]], %[[VAL_54]]{{\[}}%[[VAL_56]]] : memref<?xi64>
// CHECK:               %[[VAL_58:.*]] = memref.load %[[VAL_45]]{{\[}}%[[VAL_56]]] : memref<?xf64>
// CHECK:               memref.store %[[VAL_58]], %[[VAL_55]]{{\[}}%[[VAL_56]]] : memref<?xf64>
// CHECK:             }
// CHECK:             %[[VAL_59:.*]] = sparse_tensor.pointers %[[VAL_46]], %[[VAL_42]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_60:.*]] = memref.load %[[VAL_44]]{{\[}}%[[VAL_41]]] : memref<?xi64>
// CHECK:             %[[VAL_61:.*]] = subi %[[VAL_43]], %[[VAL_42]] : index
// CHECK:             %[[VAL_62:.*]]:3 = scf.for %[[VAL_63:.*]] = %[[VAL_41]] to %[[VAL_43]] step %[[VAL_42]] iter_args(%[[VAL_64:.*]] = %[[VAL_39]], %[[VAL_65:.*]] = %[[VAL_41]], %[[VAL_66:.*]] = %[[VAL_60]]) -> (i64, index, i64) {
// CHECK:               memref.store %[[VAL_64]], %[[VAL_59]]{{\[}}%[[VAL_63]]] : memref<?xi64>
// CHECK:               %[[VAL_67:.*]] = index_cast %[[VAL_63]] : index to i64
// CHECK:               %[[VAL_68:.*]] = cmpi eq, %[[VAL_66]], %[[VAL_67]] : i64
// CHECK:               %[[VAL_69:.*]] = cmpi ne, %[[VAL_63]], %[[VAL_61]] : index
// CHECK:               %[[VAL_70:.*]] = and %[[VAL_69]], %[[VAL_68]] : i1
// CHECK:               %[[VAL_71:.*]]:3 = scf.if %[[VAL_70]] -> (i64, index, i64) {
// CHECK:                 %[[VAL_72:.*]] = addi %[[VAL_64]], %[[VAL_40]] : i64
// CHECK:                 %[[VAL_73:.*]] = addi %[[VAL_65]], %[[VAL_42]] : index
// CHECK:                 %[[VAL_74:.*]] = memref.load %[[VAL_44]]{{\[}}%[[VAL_73]]] : memref<?xi64>
// CHECK:                 scf.yield %[[VAL_72]], %[[VAL_73]], %[[VAL_74]] : i64, index, i64
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_64]], %[[VAL_65]], %[[VAL_66]] : i64, index, i64
// CHECK:               }
// CHECK:               scf.yield %[[VAL_75:.*]]#0, %[[VAL_75]]#1, %[[VAL_75]]#2 : i64, index, i64
// CHECK:             }
// CHECK:             memref.store %[[VAL_50]], %[[VAL_59]]{{\[}}%[[VAL_43]]] : memref<?xi64>
// CHECK:             return %[[VAL_46]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }
   func @vec_to_mat_arbitrary_csc(%sparse_tensor: tensor<?xf64, #CV64>) -> tensor<?x?xf64, #CSC64> {
       %answer = graphblas.diag %sparse_tensor : tensor<?xf64, #CV64> to tensor<?x?xf64, #CSC64>
       return %answer : tensor<?x?xf64, #CSC64>
   }

// CHECK:           func @mat_to_vec_arbitrary_csr(%[[VAL_76:.*]]: tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_77:.*]] = constant true
// CHECK:             %[[VAL_78:.*]] = constant 1 : i64
// CHECK:             %[[VAL_79:.*]] = constant 0 : index
// CHECK:             %[[VAL_80:.*]] = constant 1 : index
// CHECK:             %[[VAL_81:.*]] = constant 2 : index
// CHECK:             %[[VAL_82:.*]] = tensor.dim %[[VAL_76]], %[[VAL_79]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_83:.*]] = sparse_tensor.pointers %[[VAL_76]], %[[VAL_80]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_84:.*]] = sparse_tensor.indices %[[VAL_76]], %[[VAL_80]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_85:.*]] = sparse_tensor.values %[[VAL_76]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_86:.*]] = call @new_vector_i64_p64i64(%[[VAL_82]]) : (index) -> tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_87:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_86]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @assign_rev(%[[VAL_87]], %[[VAL_79]], %[[VAL_79]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_88:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_86]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_dim(%[[VAL_88]], %[[VAL_79]], %[[VAL_82]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_89:.*]] = scf.for %[[VAL_90:.*]] = %[[VAL_79]] to %[[VAL_82]] step %[[VAL_80]] iter_args(%[[VAL_91:.*]] = %[[VAL_79]]) -> (index) {
// CHECK:               %[[VAL_92:.*]] = addi %[[VAL_90]], %[[VAL_80]] : index
// CHECK:               %[[VAL_93:.*]] = memref.load %[[VAL_83]]{{\[}}%[[VAL_90]]] : memref<?xi64>
// CHECK:               %[[VAL_94:.*]] = memref.load %[[VAL_83]]{{\[}}%[[VAL_92]]] : memref<?xi64>
// CHECK:               %[[VAL_95:.*]] = index_cast %[[VAL_93]] : i64 to index
// CHECK:               %[[VAL_96:.*]] = index_cast %[[VAL_94]] : i64 to index
// CHECK:               %[[VAL_97:.*]] = index_cast %[[VAL_90]] : index to i64
// CHECK:               %[[VAL_98:.*]]:2 = scf.while (%[[VAL_99:.*]] = %[[VAL_95]], %[[VAL_100:.*]] = %[[VAL_77]]) : (index, i1) -> (index, i1) {
// CHECK:                 %[[VAL_101:.*]] = cmpi ult, %[[VAL_99]], %[[VAL_96]] : index
// CHECK:                 %[[VAL_102:.*]] = and %[[VAL_100]], %[[VAL_101]] : i1
// CHECK:                 scf.condition(%[[VAL_102]]) %[[VAL_99]], %[[VAL_100]] : index, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_103:.*]]: index, %[[VAL_104:.*]]: i1):
// CHECK:                 %[[VAL_105:.*]] = memref.load %[[VAL_84]]{{\[}}%[[VAL_103]]] : memref<?xi64>
// CHECK:                 %[[VAL_106:.*]] = cmpi ne, %[[VAL_105]], %[[VAL_97]] : i64
// CHECK:                 %[[VAL_107:.*]] = addi %[[VAL_103]], %[[VAL_80]] : index
// CHECK:                 scf.yield %[[VAL_107]], %[[VAL_106]] : index, i1
// CHECK:               }
// CHECK:               %[[VAL_108:.*]] = scf.if %[[VAL_109:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_91]] : index
// CHECK:               } else {
// CHECK:                 %[[VAL_110:.*]] = addi %[[VAL_91]], %[[VAL_80]] : index
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
// CHECK:             %[[VAL_117:.*]] = index_cast %[[VAL_114]] : index to i64
// CHECK:             memref.store %[[VAL_117]], %[[VAL_116]]{{\[}}%[[VAL_80]]] : memref<?xi64>
// CHECK:             %[[VAL_118:.*]] = sparse_tensor.indices %[[VAL_86]], %[[VAL_79]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_119:.*]] = sparse_tensor.values %[[VAL_86]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_120:.*]] = scf.for %[[VAL_121:.*]] = %[[VAL_79]] to %[[VAL_82]] step %[[VAL_80]] iter_args(%[[VAL_122:.*]] = %[[VAL_79]]) -> (index) {
// CHECK:               %[[VAL_123:.*]] = addi %[[VAL_121]], %[[VAL_80]] : index
// CHECK:               %[[VAL_124:.*]] = memref.load %[[VAL_83]]{{\[}}%[[VAL_121]]] : memref<?xi64>
// CHECK:               %[[VAL_125:.*]] = memref.load %[[VAL_83]]{{\[}}%[[VAL_123]]] : memref<?xi64>
// CHECK:               %[[VAL_126:.*]] = index_cast %[[VAL_124]] : i64 to index
// CHECK:               %[[VAL_127:.*]] = index_cast %[[VAL_125]] : i64 to index
// CHECK:               %[[VAL_128:.*]] = index_cast %[[VAL_121]] : index to i64
// CHECK:               %[[VAL_129:.*]]:3 = scf.while (%[[VAL_130:.*]] = %[[VAL_126]], %[[VAL_131:.*]] = %[[VAL_77]], %[[VAL_132:.*]] = %[[VAL_78]]) : (index, i1, i64) -> (index, i1, i64) {
// CHECK:                 %[[VAL_133:.*]] = cmpi ult, %[[VAL_130]], %[[VAL_127]] : index
// CHECK:                 %[[VAL_134:.*]] = and %[[VAL_131]], %[[VAL_133]] : i1
// CHECK:                 scf.condition(%[[VAL_134]]) %[[VAL_130]], %[[VAL_131]], %[[VAL_132]] : index, i1, i64
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_135:.*]]: index, %[[VAL_136:.*]]: i1, %[[VAL_137:.*]]: i64):
// CHECK:                 %[[VAL_138:.*]] = memref.load %[[VAL_84]]{{\[}}%[[VAL_135]]] : memref<?xi64>
// CHECK:                 %[[VAL_139:.*]] = cmpi ne, %[[VAL_138]], %[[VAL_128]] : i64
// CHECK:                 %[[VAL_140:.*]] = scf.if %[[VAL_139]] -> (i64) {
// CHECK:                   scf.yield %[[VAL_137]] : i64
// CHECK:                 } else {
// CHECK:                   %[[VAL_141:.*]] = memref.load %[[VAL_85]]{{\[}}%[[VAL_135]]] : memref<?xi64>
// CHECK:                   scf.yield %[[VAL_141]] : i64
// CHECK:                 }
// CHECK:                 %[[VAL_142:.*]] = addi %[[VAL_135]], %[[VAL_80]] : index
// CHECK:                 scf.yield %[[VAL_142]], %[[VAL_139]], %[[VAL_143:.*]] : index, i1, i64
// CHECK:               }
// CHECK:               %[[VAL_144:.*]] = scf.if %[[VAL_145:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_122]] : index
// CHECK:               } else {
// CHECK:                 memref.store %[[VAL_146:.*]]#2, %[[VAL_119]]{{\[}}%[[VAL_122]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_128]], %[[VAL_118]]{{\[}}%[[VAL_122]]] : memref<?xi64>
// CHECK:                 %[[VAL_147:.*]] = addi %[[VAL_122]], %[[VAL_80]] : index
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
// CHECK:             %[[VAL_150:.*]] = constant true
// CHECK:             %[[VAL_151:.*]] = constant 1 : i64
// CHECK:             %[[VAL_152:.*]] = constant 0 : index
// CHECK:             %[[VAL_153:.*]] = constant 1 : index
// CHECK:             %[[VAL_154:.*]] = constant 2 : index
// CHECK:             %[[VAL_155:.*]] = tensor.dim %[[VAL_149]], %[[VAL_152]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_156:.*]] = sparse_tensor.pointers %[[VAL_149]], %[[VAL_153]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_157:.*]] = sparse_tensor.indices %[[VAL_149]], %[[VAL_153]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_158:.*]] = sparse_tensor.values %[[VAL_149]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_159:.*]] = call @new_vector_i64_p64i64(%[[VAL_155]]) : (index) -> tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_160:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_159]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @assign_rev(%[[VAL_160]], %[[VAL_152]], %[[VAL_152]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_161:.*]] = call @vector_i64_p64i64_to_ptr8(%[[VAL_159]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:             call @resize_dim(%[[VAL_161]], %[[VAL_152]], %[[VAL_155]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:             %[[VAL_162:.*]] = scf.for %[[VAL_163:.*]] = %[[VAL_152]] to %[[VAL_155]] step %[[VAL_153]] iter_args(%[[VAL_164:.*]] = %[[VAL_152]]) -> (index) {
// CHECK:               %[[VAL_165:.*]] = addi %[[VAL_163]], %[[VAL_153]] : index
// CHECK:               %[[VAL_166:.*]] = memref.load %[[VAL_156]]{{\[}}%[[VAL_163]]] : memref<?xi64>
// CHECK:               %[[VAL_167:.*]] = memref.load %[[VAL_156]]{{\[}}%[[VAL_165]]] : memref<?xi64>
// CHECK:               %[[VAL_168:.*]] = index_cast %[[VAL_166]] : i64 to index
// CHECK:               %[[VAL_169:.*]] = index_cast %[[VAL_167]] : i64 to index
// CHECK:               %[[VAL_170:.*]] = index_cast %[[VAL_163]] : index to i64
// CHECK:               %[[VAL_171:.*]]:2 = scf.while (%[[VAL_172:.*]] = %[[VAL_168]], %[[VAL_173:.*]] = %[[VAL_150]]) : (index, i1) -> (index, i1) {
// CHECK:                 %[[VAL_174:.*]] = cmpi ult, %[[VAL_172]], %[[VAL_169]] : index
// CHECK:                 %[[VAL_175:.*]] = and %[[VAL_173]], %[[VAL_174]] : i1
// CHECK:                 scf.condition(%[[VAL_175]]) %[[VAL_172]], %[[VAL_173]] : index, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_176:.*]]: index, %[[VAL_177:.*]]: i1):
// CHECK:                 %[[VAL_178:.*]] = memref.load %[[VAL_157]]{{\[}}%[[VAL_176]]] : memref<?xi64>
// CHECK:                 %[[VAL_179:.*]] = cmpi ne, %[[VAL_178]], %[[VAL_170]] : i64
// CHECK:                 %[[VAL_180:.*]] = addi %[[VAL_176]], %[[VAL_153]] : index
// CHECK:                 scf.yield %[[VAL_180]], %[[VAL_179]] : index, i1
// CHECK:               }
// CHECK:               %[[VAL_181:.*]] = scf.if %[[VAL_182:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_164]] : index
// CHECK:               } else {
// CHECK:                 %[[VAL_183:.*]] = addi %[[VAL_164]], %[[VAL_153]] : index
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
// CHECK:             %[[VAL_190:.*]] = index_cast %[[VAL_187]] : index to i64
// CHECK:             memref.store %[[VAL_190]], %[[VAL_189]]{{\[}}%[[VAL_153]]] : memref<?xi64>
// CHECK:             %[[VAL_191:.*]] = sparse_tensor.indices %[[VAL_159]], %[[VAL_152]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_192:.*]] = sparse_tensor.values %[[VAL_159]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_193:.*]] = scf.for %[[VAL_194:.*]] = %[[VAL_152]] to %[[VAL_155]] step %[[VAL_153]] iter_args(%[[VAL_195:.*]] = %[[VAL_152]]) -> (index) {
// CHECK:               %[[VAL_196:.*]] = addi %[[VAL_194]], %[[VAL_153]] : index
// CHECK:               %[[VAL_197:.*]] = memref.load %[[VAL_156]]{{\[}}%[[VAL_194]]] : memref<?xi64>
// CHECK:               %[[VAL_198:.*]] = memref.load %[[VAL_156]]{{\[}}%[[VAL_196]]] : memref<?xi64>
// CHECK:               %[[VAL_199:.*]] = index_cast %[[VAL_197]] : i64 to index
// CHECK:               %[[VAL_200:.*]] = index_cast %[[VAL_198]] : i64 to index
// CHECK:               %[[VAL_201:.*]] = index_cast %[[VAL_194]] : index to i64
// CHECK:               %[[VAL_202:.*]]:3 = scf.while (%[[VAL_203:.*]] = %[[VAL_199]], %[[VAL_204:.*]] = %[[VAL_150]], %[[VAL_205:.*]] = %[[VAL_151]]) : (index, i1, i64) -> (index, i1, i64) {
// CHECK:                 %[[VAL_206:.*]] = cmpi ult, %[[VAL_203]], %[[VAL_200]] : index
// CHECK:                 %[[VAL_207:.*]] = and %[[VAL_204]], %[[VAL_206]] : i1
// CHECK:                 scf.condition(%[[VAL_207]]) %[[VAL_203]], %[[VAL_204]], %[[VAL_205]] : index, i1, i64
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_208:.*]]: index, %[[VAL_209:.*]]: i1, %[[VAL_210:.*]]: i64):
// CHECK:                 %[[VAL_211:.*]] = memref.load %[[VAL_157]]{{\[}}%[[VAL_208]]] : memref<?xi64>
// CHECK:                 %[[VAL_212:.*]] = cmpi ne, %[[VAL_211]], %[[VAL_201]] : i64
// CHECK:                 %[[VAL_213:.*]] = scf.if %[[VAL_212]] -> (i64) {
// CHECK:                   scf.yield %[[VAL_210]] : i64
// CHECK:                 } else {
// CHECK:                   %[[VAL_214:.*]] = memref.load %[[VAL_158]]{{\[}}%[[VAL_208]]] : memref<?xi64>
// CHECK:                   scf.yield %[[VAL_214]] : i64
// CHECK:                 }
// CHECK:                 %[[VAL_215:.*]] = addi %[[VAL_208]], %[[VAL_153]] : index
// CHECK:                 scf.yield %[[VAL_215]], %[[VAL_212]], %[[VAL_216:.*]] : index, i1, i64
// CHECK:               }
// CHECK:               %[[VAL_217:.*]] = scf.if %[[VAL_218:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_195]] : index
// CHECK:               } else {
// CHECK:                 memref.store %[[VAL_219:.*]]#2, %[[VAL_192]]{{\[}}%[[VAL_195]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_201]], %[[VAL_191]]{{\[}}%[[VAL_195]]] : memref<?xi64>
// CHECK:                 %[[VAL_220:.*]] = addi %[[VAL_195]], %[[VAL_153]] : index
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
