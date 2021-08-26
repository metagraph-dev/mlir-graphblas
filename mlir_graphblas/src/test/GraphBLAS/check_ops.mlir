// RUN: graphblas-opt %s | graphblas-opt | FileCheck %s

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

    // CHECK: func @convert_layout_wrapper(%[[ARG0:.*]]: [[CSR_TYPE:tensor<.*->.*>]]) -> [[CSC_TYPE:tensor<.*->.*>]] {
    func @convert_layout_wrapper(%sparse_tensor: tensor<2x3xf64, #CSR64>) -> tensor<2x3xf64, #CSC64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.convert_layout %[[ARG0]] : [[CSR_TYPE]] to [[CSC_TYPE]]
        %answer = graphblas.convert_layout %sparse_tensor : tensor<2x3xf64, #CSR64> to tensor<2x3xf64, #CSC64>
        // CHECK-NEXT: return %[[ANSWER]] : [[CSC_TYPE]]
        return %answer : tensor<2x3xf64, #CSC64>
    }

}

module {

    // CHECK: func @transpose_wrapper(%[[ARG0:.*]]: [[CSR_TYPE:tensor<.*->.*>]]) -> [[CSC_TYPE:tensor<.*->.*>]] {
    func @transpose_wrapper(%sparse_tensor: tensor<2x3xf64, #CSR64>) -> tensor<3x2xf64, #CSC64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.transpose %[[ARG0]] : [[CSR_TYPE]] to [[CSC_TYPE]]
        %answer = graphblas.transpose %sparse_tensor : tensor<2x3xf64, #CSR64> to tensor<3x2xf64, #CSC64>
        // CHECK-NEXT: return %[[ANSWER]] : [[CSC_TYPE]]
        return %answer : tensor<3x2xf64, #CSC64>
    }

}

module {

    // CHECK: func @matrix_select_triu(%[[ARG0:.*]]: [[CSR_TYPE:tensor<.*->.*>]]) -> [[CSR_TYPE]] {
    func @matrix_select_triu(%sparse_tensor: tensor<100x100xf64, #CSR64>) -> tensor<100x100xf64, #CSR64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_select %[[ARG0]] {selectors = ["triu"]} : [[CSR_TYPE]] to [[CSR_TYPE]]
        %answer = graphblas.matrix_select %sparse_tensor { selectors = ["triu"] } : tensor<100x100xf64, #CSR64> to tensor<100x100xf64, #CSR64>
        // CHECK-NEXT: return %[[ANSWER]] : [[CSR_TYPE]]
        return %answer : tensor<100x100xf64, #CSR64>
    }

    // CHECK: func @matrix_select_tril(%[[ARG0:.*]]: [[CSR_TYPE:tensor<.*->.*>]]) -> [[CSR_TYPE]] {
    func @matrix_select_tril(%sparse_tensor: tensor<100x100xf64, #CSR64>) -> tensor<100x100xf64, #CSR64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_select %[[ARG0]] {selectors = ["tril"]} : [[CSR_TYPE]] to [[CSR_TYPE]]
        %answer = graphblas.matrix_select %sparse_tensor { selectors = ["tril"] } : tensor<100x100xf64, #CSR64> to tensor<100x100xf64, #CSR64>
        // CHECK-NEXT: return %[[ANSWER]] : [[CSR_TYPE]]
        return %answer : tensor<100x100xf64, #CSR64>
    }

    // CHECK: func @matrix_select_gt_thunk(%[[ARG0:.*]]: [[CSR_TYPE:tensor<.*->.*>]]) -> [[CSR_TYPE]] {
    func @matrix_select_gt_thunk(%sparse_tensor: tensor<100x100xf64, #CSR64>) -> tensor<100x100xf64, #CSR64> {
        // CHECK-NEXT: %[[THUNK:.*]] = constant 0.000000e+00 : [[THUNK_TYPE:.*]]
        %thunk = constant 0.0 : f64
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_select %[[ARG0]], %[[THUNK]] {selectors = ["gt"]} : [[CSR_TYPE]], [[THUNK_TYPE]] to [[CSR_TYPE]]
        %answer = graphblas.matrix_select %sparse_tensor, %thunk { selectors = ["gt"] } : tensor<100x100xf64, #CSR64>, f64 to tensor<100x100xf64, #CSR64>
        // CHECK-NEXT: return %[[ANSWER]] : [[CSR_TYPE]]
        return %answer : tensor<100x100xf64, #CSR64>
    }
 
}

module {

    // CHECK: func @matrix_reduce_to_scalar(%[[ARG0:.*]]: [[CSR_TYPE:tensor<.*->.*>]]) -> [[RETURN_TYPE:.*]] {
    func @matrix_reduce_to_scalar(%sparse_tensor: tensor<2x3xi64, #CSR64>) -> i64 {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_reduce_to_scalar %[[ARG0]] {aggregator = "plus"} : [[CSR_TYPE]] to [[RETURN_TYPE]]
        %answer = graphblas.matrix_reduce_to_scalar %sparse_tensor { aggregator = "plus" } : tensor<2x3xi64, #CSR64> to i64
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : i64
    }

}

module {

    // CHECK: func @matrix_apply(%[[ARG0:.*]]: [[CSR_TYPE:tensor<.*->.*>]]) -> [[RETURN_TYPE:.*]] {
    func @matrix_apply(%sparse_tensor: tensor<2x3xi64, #CSR64>) -> tensor<2x3xi64, #CSR64> {
        // CHECK-NEXT: %[[THUNK:.*]] = constant 100 : [[THUNK_TYPE:.*]]
        %thunk = constant 100 : i64
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_apply %[[ARG0]], %[[THUNK]] {apply_operator = "min"} : ([[CSR_TYPE]], [[THUNK_TYPE]]) to [[CSR_TYPE]]
        %answer = graphblas.matrix_apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<2x3xi64, #CSR64>, i64) to tensor<2x3xi64, #CSR64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<2x3xi64, #CSR64>
    }

}

module {

    // CHECK: func @matrix_multiply_plus_times(%[[ARGA:.*]]: [[CSR_TYPE_A:tensor<.*->.*>]], %[[ARGB:.*]]: [[CSC_TYPE_B:tensor<.*->.*>]]) -> [[RETURN_TYPE:tensor<.*->.*>]] {
    func @matrix_multiply_plus_times(%argA: tensor<2x3xi64, #CSR64>, %argB: tensor<3x2xi64, #CSC64>) -> tensor<2x2xi64, #CSR64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_multiply %[[ARGA]], %[[ARGB]] {semiring = "plus_times"} : ([[CSR_TYPE_A]], [[CSC_TYPE_B]]) to [[RETURN_TYPE]]
        %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_times" } : (tensor<2x3xi64, #CSR64>, tensor<3x2xi64, #CSC64>) to tensor<2x2xi64, #CSR64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<2x2xi64, #CSR64>
    }

    // CHECK: func @matrix_multiply_with_mask_plus_times(%[[ARGA:.*]]: [[CSR_TYPE_A:tensor<.*->.*>]], %[[ARGB:.*]]: [[CSC_TYPE_B:tensor<.*->.*>]], %[[MASK:.*]]: [[MASK_TYPE:tensor<.*->.*>]]) -> [[RETURN_TYPE:tensor<.*->.*>]] {
    func @matrix_multiply_with_mask_plus_times(%argA: tensor<2x2xf64, #CSR64>, %argB: tensor<2x2xf64, #CSC64>, %mask: tensor<2x2xf64, #CSR64>) -> tensor<2x2xf64, #CSR64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_multiply %[[ARGA]], %[[ARGB]], %[[MASK]] {semiring = "plus_times"} : ([[CSR_TYPE_A]], [[CSC_TYPE_B]], [[MASK_TYPE]]) to [[RETURN_TYPE]]
        %answer = graphblas.matrix_multiply %argA, %argB, %mask { semiring = "plus_times" } : (tensor<2x2xf64, #CSR64>, tensor<2x2xf64, #CSC64>, tensor<2x2xf64, #CSR64>) to tensor<2x2xf64, #CSR64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<2x2xf64, #CSR64>
    }

    // CHECK: func @matrix_multiply_plus_pair(%[[ARGA:.*]]: [[CSR_TYPE_A:tensor<.*->.*>]], %[[ARGB:.*]]: [[CSC_TYPE_B:tensor<.*->.*>]]) -> [[RETURN_TYPE:tensor<.*->.*>]] {
    func @matrix_multiply_plus_pair(%argA: tensor<2x3xi64, #CSR64>, %argB: tensor<3x2xi64, #CSC64>) -> tensor<2x2xi64, #CSR64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_multiply %[[ARGA]], %[[ARGB]] {semiring = "plus_pair"} : ([[CSR_TYPE_A]], [[CSC_TYPE_B]]) to [[RETURN_TYPE]]
        %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_pair" } : (tensor<2x3xi64, #CSR64>, tensor<3x2xi64, #CSC64>) to tensor<2x2xi64, #CSR64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<2x2xi64, #CSR64>
    }

    // CHECK: func @matrix_multiply_with_mask_plus_pair(%[[ARGA:.*]]: [[CSR_TYPE_A:tensor<.*->.*>]], %[[ARGB:.*]]: [[CSC_TYPE_B:tensor<.*->.*>]], %[[MASK:.*]]: [[MASK_TYPE:tensor<.*->.*>]]) -> [[RETURN_TYPE:tensor<.*->.*>]] {
    func @matrix_multiply_with_mask_plus_pair(%argA: tensor<2x2xf64, #CSR64>, %argB: tensor<2x2xf64, #CSC64>, %mask: tensor<2x2xf64, #CSR64>) -> tensor<2x2xf64, #CSR64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_multiply %[[ARGA]], %[[ARGB]], %[[MASK]] {semiring = "plus_pair"} : ([[CSR_TYPE_A]], [[CSC_TYPE_B]], [[MASK_TYPE]]) to [[RETURN_TYPE]]
        %answer = graphblas.matrix_multiply %argA, %argB, %mask { semiring = "plus_pair" } : (tensor<2x2xf64, #CSR64>, tensor<2x2xf64, #CSC64>, tensor<2x2xf64, #CSR64>) to tensor<2x2xf64, #CSR64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<2x2xf64, #CSR64>
    }

    // CHECK: func @matrix_multiply_plus_plus(%[[ARGA:.*]]: [[CSR_TYPE_A:tensor<.*->.*>]], %[[ARGB:.*]]: [[CSC_TYPE_B:tensor<.*->.*>]]) -> [[RETURN_TYPE:tensor<.*->.*>]] {
    func @matrix_multiply_plus_plus(%argA: tensor<2x3xi64, #CSR64>, %argB: tensor<3x2xi64, #CSC64>) -> tensor<2x2xi64, #CSR64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_multiply %[[ARGA]], %[[ARGB]] {semiring = "plus_plus"} : ([[CSR_TYPE_A]], [[CSC_TYPE_B]]) to [[RETURN_TYPE]]
        %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_plus" } : (tensor<2x3xi64, #CSR64>, tensor<3x2xi64, #CSC64>) to tensor<2x2xi64, #CSR64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<2x2xi64, #CSR64>
    }

    // CHECK: func @matrix_multiply_with_mask_plus_plus(%[[ARGA:.*]]: [[CSR_TYPE_A:tensor<.*->.*>]], %[[ARGB:.*]]: [[CSC_TYPE_B:tensor<.*->.*>]], %[[MASK:.*]]: [[MASK_TYPE:tensor<.*->.*>]]) -> [[RETURN_TYPE:tensor<.*->.*>]] {
    func @matrix_multiply_with_mask_plus_plus(%argA: tensor<2x2xf64, #CSR64>, %argB: tensor<2x2xf64, #CSC64>, %mask: tensor<2x2xf64, #CSR64>) -> tensor<2x2xf64, #CSR64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_multiply %[[ARGA]], %[[ARGB]], %[[MASK]] {semiring = "plus_plus"} : ([[CSR_TYPE_A]], [[CSC_TYPE_B]], [[MASK_TYPE]]) to [[RETURN_TYPE]]
        %answer = graphblas.matrix_multiply %argA, %argB, %mask { semiring = "plus_plus" } : (tensor<2x2xf64, #CSR64>, tensor<2x2xf64, #CSC64>, tensor<2x2xf64, #CSR64>) to tensor<2x2xf64, #CSR64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<2x2xf64, #CSR64>
    }

}

module {

    // CHECK: func @matrix_vector_multiply_plus_times(%[[MATRIX:.*]]: [[MATRIX_TYPE:tensor<.*->.*>]], %[[VECTOR:.*]]: [[VECTOR_TYPE:tensor<.*>]]) -> [[RETURN_TYPE:tensor<.*>]] {
    func @matrix_vector_multiply_plus_times(%matrix: tensor<2x3xi64, #CSR64>, %vector: tensor<3xi64, #SparseVec64>) -> tensor<2xi64, #SparseVec64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_multiply %[[MATRIX]], %[[VECTOR]] {semiring = "plus_times"} : ([[MATRIX_TYPE]], [[VECTOR_TYPE]]) to [[RETURN_TYPE]]
        %answer = graphblas.matrix_multiply %matrix, %vector { semiring = "plus_times" } : (tensor<2x3xi64, #CSR64>, tensor<3xi64, #SparseVec64>) to tensor<2xi64, #SparseVec64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<2xi64, #SparseVec64>
    }

    // CHECK: func @matrix_vector_multiply_with_mask_plus_times(%[[MATRIX:.*]]: [[MATRIX_TYPE:tensor<.*->.*>]], %[[VECTOR:.*]]: [[VECTOR_TYPE:tensor<.*>]], %[[MASK:.*]]: [[MASK_TYPE:tensor<.*>]]) -> [[RETURN_TYPE:tensor<.*>]] {
    func @matrix_vector_multiply_with_mask_plus_times(%matrix: tensor<2x2xf64, #CSR64>, %vector: tensor<2xf64, #SparseVec64>, %mask: tensor<2xf64, #SparseVec64>) -> tensor<2xf64, #SparseVec64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_multiply %[[MATRIX]], %[[VECTOR]], %[[MASK]] {semiring = "plus_times"} : ([[MATRIX_TYPE]], [[VECTOR_TYPE]], [[MASK_TYPE]]) to [[RETURN_TYPE]]
        %answer = graphblas.matrix_multiply %matrix, %vector, %mask { semiring = "plus_times" } : (tensor<2x2xf64, #CSR64>, tensor<2xf64, #SparseVec64>, tensor<2xf64, #SparseVec64>) to tensor<2xf64, #SparseVec64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<2xf64, #SparseVec64>
    }

    // CHECK: func @vector_matrix_multiply_plus_times(%[[VECTOR:.*]]: [[VECTOR_TYPE:tensor<.*>]], %[[MATRIX:.*]]: [[MATRIX_TYPE:tensor<.*->.*>]]) -> [[RETURN_TYPE:tensor<.*>]] {
    func @vector_matrix_multiply_plus_times(%vector: tensor<3xi64, #SparseVec64>, %matrix: tensor<3x2xi64, #CSC64>) -> tensor<2xi64, #SparseVec64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_multiply %[[VECTOR]], %[[MATRIX]] {semiring = "plus_times"} : ([[VECTOR_TYPE]], [[MATRIX_TYPE]]) to [[RETURN_TYPE]]
        %answer = graphblas.matrix_multiply %vector, %matrix { semiring = "plus_times" } : (tensor<3xi64, #SparseVec64>, tensor<3x2xi64, #CSC64>) to tensor<2xi64, #SparseVec64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<2xi64, #SparseVec64>
    }

    // CHECK: func @vector_matrix_multiply_with_mask_plus_times(%[[VECTOR:.*]]: [[VECTOR_TYPE:tensor<.*>]], %[[MATRIX:.*]]: [[MATRIX_TYPE:tensor<.*->.*>]], %[[MASK:.*]]: [[MASK_TYPE:tensor<.*>]]) -> [[RETURN_TYPE:tensor<.*>]] {
    func @vector_matrix_multiply_with_mask_plus_times(%vector: tensor<2xf64, #SparseVec64>, %matrix: tensor<2x2xf64, #CSC64>, %mask: tensor<2xf64, #SparseVec64>) -> tensor<2xf64, #SparseVec64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_multiply %[[VECTOR]], %[[MATRIX]], %[[MASK]] {semiring = "plus_times"} : ([[VECTOR_TYPE]], [[MATRIX_TYPE]], [[MASK_TYPE]]) to [[RETURN_TYPE]]
        %answer = graphblas.matrix_multiply %vector, %matrix, %mask { semiring = "plus_times" } : (tensor<2xf64, #SparseVec64>, tensor<2x2xf64, #CSC64>, tensor<2xf64, #SparseVec64>) to tensor<2xf64, #SparseVec64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<2xf64, #SparseVec64>
    }

    // CHECK: func @vector_dot_product_plus_times(%[[ARGA:.*]]: [[A_TYPE:tensor<.*>]], %[[ARGB:.*]]: [[B_TYPE:tensor<.*>]]) -> [[RETURN_TYPE:.*]] {
    func @vector_dot_product_plus_times(%argA: tensor<3xi64, #SparseVec64>, %argB: tensor<3xi64, #SparseVec64>) -> i64 {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_multiply %[[ARGA]], %[[ARGB]] {semiring = "plus_times"} : ([[A_TYPE]], [[B_TYPE]]) to [[RETURN_TYPE]]
        %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_times" } : (tensor<3xi64, #SparseVec64>, tensor<3xi64, #SparseVec64>) to i64
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : i64
    }

}

module {

    // CHECK: func @vector_update(%[[ARGA:.*]]: [[A_TYPE:tensor<.*>]], %[[ARGB:.*]]: [[B_TYPE:tensor<.*>]]) {
    func @vector_update(%other_vec: tensor<?xi64, #SparseVec64>, %vec: tensor<?xi64, #SparseVec64>) {
        // CHECK-NEXT: graphblas.update %[[ARGA]] -> %[[ARGB]] : [[A_TYPE]] -> [[B_TYPE]]
        graphblas.update %other_vec -> %vec : tensor<?xi64, #SparseVec64> -> tensor<?xi64, #SparseVec64>
        return
    }

    // CHECK: func @matrix_update_all_options(%[[ARGA:.*]]: [[A_TYPE:tensor<.*->.*>]], %[[ARGB:.*]]: [[B_TYPE:tensor<.*->.*>]], %[[ARGM:.*]]: [[M_TYPE:tensor<.*->.*>]]) {
    func @matrix_update_all_options(%input: tensor<?x?xf64, #CSR64>, %output: tensor<?x?xf64, #CSR64>, %mask: tensor<?x?xf64, #CSR64>) {
        // CHECK-NEXT: graphblas.update %[[ARGA]] -> %[[ARGB]](%[[ARGM]]) {accumulate_operator = "plus", mask_complement = true, replace = true} : [[A_TYPE]] -> [[B_TYPE]]([[M_TYPE]])
        graphblas.update %input -> %output(%mask) { accumulate_operator = "plus", replace = true, mask_complement = true }: tensor<?x?xf64, #CSR64> -> tensor<?x?xf64, #CSR64>(tensor<?x?xf64, #CSR64>)
        return
    }

}

module {

    // CHECK: func @vector_equaity_checking(%[[ARGA:.*]]: [[A_TYPE:tensor<.*>]], %[[ARGB:.*]]: [[B_TYPE:tensor<.*>]]) -> [[RETURN_TYPE:.*]] {
    func @vector_equaity_checking(%argA: tensor<3xi64, #SparseVec64>, %argB: tensor<3xi64, #SparseVec64>) -> i1 {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.equal %[[ARGA]], %[[ARGB]] : [[A_TYPE]], [[B_TYPE]]
        %answer = graphblas.equal %argA, %argB : tensor<3xi64, #SparseVec64>, tensor<3xi64, #SparseVec64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : i1
    }

}

module {

    // CHECK: func @comment_wrapper() {
    func @comment_wrapper() -> () {
        // CHECK-NEXT: graphblas.comment {comment = "here is a comment!"}
        graphblas.comment { comment = "here is a comment!" } 
        // CHECK-NEXT: return
        return
    }

}

module {

    // CHECK: func @vector_argminmax_min(%[[ARGA:.*]]: [[A_TYPE:tensor<.*>]]) -> [[RETURN_TYPE:.*]] {
    func @vector_argminmax_min(%vec: tensor<3xi64, #SparseVec64>) -> index {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.vector_argminmax %[[ARGA]] {minmax = "min"} : [[A_TYPE]]
        %answer = graphblas.vector_argminmax %vec { minmax = "min" } : tensor<3xi64, #SparseVec64>
	// CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : index
    }

    // CHECK: func @vector_argminmax_max(%[[ARGA:.*]]: [[A_TYPE:tensor<.*>]]) -> [[RETURN_TYPE:.*]] {
    func @vector_argminmax_max(%vec: tensor<3xi64, #SparseVec64>) -> index {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.vector_argminmax %[[ARGA]] {minmax = "max"} : [[A_TYPE]]
        %answer = graphblas.vector_argminmax %vec { minmax = "max" } : tensor<3xi64, #SparseVec64>
	// COM: CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : index
    }

}

module {

    // CHECK: func @vector_argmin_wrapper(%[[ARGA:.*]]: [[A_TYPE:tensor<.*>]]) -> [[RETURN_TYPE:.*]] {
    func @vector_argmin_wrapper(%vec: tensor<3xi64, #SparseVec64>) -> index {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.vector_argmin %[[ARGA]] : [[A_TYPE]]
        %answer = graphblas.vector_argmin %vec : tensor<3xi64, #SparseVec64>
	// CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : index
    }

}

module {

    // CHECK: func @vector_argmax_wrapper(%[[ARGA:.*]]: [[A_TYPE:tensor<.*>]]) -> [[RETURN_TYPE:.*]] {
    func @vector_argmax_wrapper(%vec: tensor<3xi64, #SparseVec64>) -> index {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.vector_argmax %[[ARGA]] : [[A_TYPE]]
        %answer = graphblas.vector_argmax %vec : tensor<3xi64, #SparseVec64>
	// CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : index
    }

}

module {

    // CHECK: func @matrix_reduce_to_vector_wrapper(%[[MATRIX:.*]]: [[MATRIX_TYPE:tensor<.*->.*>]]) -> ([[RETURN_TYPE_0:tensor<.*>]], [[RETURN_TYPE_1:tensor<.*>]]) {
    func @matrix_reduce_to_vector_wrapper(%matrix: tensor<7x9xi32, #CSR64>) -> (tensor<9xi32, #SparseVec64>, tensor<7xi32, #SparseVec64>) {
        // CHECK-NEXT: %[[ANSWER_0:.*]] = graphblas.matrix_reduce_to_vector %[[MATRIX]] {aggregator = "plus", axis = 0 : i64} : [[MATRIX_TYPE]] to [[RETURN_TYPE_0]]
        %vec1 = graphblas.matrix_reduce_to_vector %matrix { aggregator = "plus", axis = 0 } : tensor<7x9xi32, #CSR64> to tensor<9xi32, #SparseVec64>
        // CHECK-NEXT: %[[ANSWER_1:.*]] = graphblas.matrix_reduce_to_vector %[[MATRIX]] {aggregator = "plus", axis = 1 : i64} : [[MATRIX_TYPE]] to [[RETURN_TYPE_1]]
        %vec2 = graphblas.matrix_reduce_to_vector %matrix { aggregator = "plus", axis = 1 } : tensor<7x9xi32, #CSR64> to tensor<7xi32, #SparseVec64>
	// CHECK-NEXT: return %[[ANSWER_0]], %[[ANSWER_1]] : [[RETURN_TYPE_0]], [[RETURN_TYPE_1]]
        return %vec1, %vec2 : tensor<9xi32, #SparseVec64>, tensor<7xi32, #SparseVec64>
    }

}
