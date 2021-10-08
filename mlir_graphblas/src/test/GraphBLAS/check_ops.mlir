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

#CV64 = #sparse_tensor.encoding<{
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
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.select %[[ARG0]] {selectors = ["triu"]} : [[CSR_TYPE]] to [[CSR_TYPE]]
        %answer = graphblas.select %sparse_tensor { selectors = ["triu"] } : tensor<100x100xf64, #CSR64> to tensor<100x100xf64, #CSR64>
        // CHECK-NEXT: return %[[ANSWER]] : [[CSR_TYPE]]
        return %answer : tensor<100x100xf64, #CSR64>
    }

    // CHECK: func @matrix_select_tril(%[[ARG0:.*]]: [[CSR_TYPE:tensor<.*->.*>]]) -> [[CSR_TYPE]] {
    func @matrix_select_tril(%sparse_tensor: tensor<100x100xf64, #CSR64>) -> tensor<100x100xf64, #CSR64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.select %[[ARG0]] {selectors = ["tril"]} : [[CSR_TYPE]] to [[CSR_TYPE]]
        %answer = graphblas.select %sparse_tensor { selectors = ["tril"] } : tensor<100x100xf64, #CSR64> to tensor<100x100xf64, #CSR64>
        // CHECK-NEXT: return %[[ANSWER]] : [[CSR_TYPE]]
        return %answer : tensor<100x100xf64, #CSR64>
    }

    // CHECK: func @matrix_select_gt_thunk(%[[ARG0:.*]]: [[CSR_TYPE:tensor<.*->.*>]]) -> [[CSR_TYPE]] {
    func @matrix_select_gt_thunk(%sparse_tensor: tensor<100x100xf64, #CSR64>) -> tensor<100x100xf64, #CSR64> {
        // CHECK-NEXT: %[[THUNK:.*]] = constant 0.000000e+00 : [[THUNK_TYPE:.*]]
        %thunk = constant 0.0 : f64
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.select %[[ARG0]], %[[THUNK]] {selectors = ["gt"]} : [[CSR_TYPE]], [[THUNK_TYPE]] to [[CSR_TYPE]]
        %answer = graphblas.select %sparse_tensor, %thunk { selectors = ["gt"] } : tensor<100x100xf64, #CSR64>, f64 to tensor<100x100xf64, #CSR64>
        // CHECK-NEXT: return %[[ANSWER]] : [[CSR_TYPE]]
        return %answer : tensor<100x100xf64, #CSR64>
    }
 
}

module {

    // CHECK: func @matrix_reduce_to_scalar_plus(%[[ARG0:.*]]: [[CSR_TYPE:tensor<.*->.*>]]) -> [[RETURN_TYPE:.*]] {
    func @matrix_reduce_to_scalar_plus(%sparse_tensor: tensor<2x3xi64, #CSR64>) -> i64 {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.reduce_to_scalar %[[ARG0]] {aggregator = "plus"} : [[CSR_TYPE]] to [[RETURN_TYPE]]
        %answer = graphblas.reduce_to_scalar %sparse_tensor { aggregator = "plus" } : tensor<2x3xi64, #CSR64> to i64
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : i64
    }

    // CHECK: func @vector_reduce_to_scalar_count(%[[ARG0:.*]]: [[CV_TYPE:tensor<.*>]]) -> [[RETURN_TYPE:.*]] {
    func @vector_reduce_to_scalar_count(%sparse_tensor: tensor<2xi64, #CV64>) -> i64 {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.reduce_to_scalar %[[ARG0]] {aggregator = "count"} : [[CV_TYPE]] to [[RETURN_TYPE]]
        %answer = graphblas.reduce_to_scalar %sparse_tensor { aggregator = "count" } : tensor<2xi64, #CV64> to i64
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : i64
    }

    // CHECK: func @vector_reduce_to_scalar_argmin(%[[ARG0:.*]]: [[CV_TYPE:tensor<.*>]]) -> [[RETURN_TYPE:.*]] {
    func @vector_reduce_to_scalar_argmin(%sparse_tensor: tensor<2xf16, #CV64>) -> i64 {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.reduce_to_scalar %[[ARG0]] {aggregator = "argmin"} : [[CV_TYPE]] to [[RETURN_TYPE]]
        %answer = graphblas.reduce_to_scalar %sparse_tensor { aggregator = "argmin" } : tensor<2xf16, #CV64> to i64
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : i64
    }

    // CHECK: func @vector_reduce_to_scalar_argmax(%[[ARG0:.*]]: [[CV_TYPE:tensor<.*>]]) -> [[RETURN_TYPE:.*]] {
    func @vector_reduce_to_scalar_argmax(%sparse_tensor: tensor<2xf16, #CV64>) -> i64 {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.reduce_to_scalar %[[ARG0]] {aggregator = "argmax"} : [[CV_TYPE]] to [[RETURN_TYPE]]
        %answer = graphblas.reduce_to_scalar %sparse_tensor { aggregator = "argmax" } : tensor<2xf16, #CV64> to i64
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : i64
    }

}

module {

    // CHECK: func @apply_matrix_min(%[[ARG0:.*]]: [[CSR_TYPE:tensor<.*->.*>]]) -> [[RETURN_TYPE:.*]] {
    func @apply_matrix_min(%sparse_tensor: tensor<2x3xi64, #CSR64>) -> tensor<2x3xi64, #CSR64> {
        // CHECK-NEXT: %[[THUNK:.*]] = constant 100 : [[THUNK_TYPE:.*]]
        %thunk = constant 100 : i64
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.apply %[[ARG0]], %[[THUNK]] {apply_operator = "min"} : ([[CSR_TYPE]], [[THUNK_TYPE]]) to [[CSR_TYPE]]
        %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<2x3xi64, #CSR64>, i64) to tensor<2x3xi64, #CSR64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<2x3xi64, #CSR64>
    }
   
    // CHECK: func @apply_vector_min(%[[ARG0:.*]]: [[VECTOR_TYPE:tensor<.*>]]) -> [[RETURN_TYPE:.*]] {
    func @apply_vector_min(%sparse_tensor: tensor<3xi64, #CV64>) -> tensor<3xi64, #CV64> {
        // CHECK-NEXT: %[[THUNK:.*]] = constant 100 : [[THUNK_TYPE:.*]]
        %thunk = constant 100 : i64
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.apply %[[ARG0]], %[[THUNK]] {apply_operator = "min"} : ([[VECTOR_TYPE]], [[THUNK_TYPE]]) to [[VECTOR_TYPE]]
        %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<3xi64, #CV64>, i64) to tensor<3xi64, #CV64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<3xi64, #CV64>
    }

    // CHECK: func @apply_matrix_abs(%[[ARG0:.*]]: [[CSR_TYPE:tensor<.*->.*>]]) -> [[RETURN_TYPE:.*]] {
    func @apply_matrix_abs(%sparse_tensor: tensor<2x3xi64, #CSR64>) -> tensor<2x3xi64, #CSR64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.apply %[[ARG0]] {apply_operator = "abs"} : ([[CSR_TYPE]]) to [[CSR_TYPE]]
        %answer = graphblas.apply %sparse_tensor { apply_operator = "abs" } : (tensor<2x3xi64, #CSR64>) to tensor<2x3xi64, #CSR64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<2x3xi64, #CSR64>
    }
   
    // CHECK: func @apply_vector_abs(%[[ARG0:.*]]: [[VECTOR_TYPE:tensor<.*>]]) -> [[RETURN_TYPE:.*]] {
    func @apply_vector_abs(%sparse_tensor: tensor<3xi64, #CV64>) -> tensor<3xi64, #CV64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.apply %[[ARG0]] {apply_operator = "abs"} : ([[VECTOR_TYPE]]) to [[VECTOR_TYPE]]
        %answer = graphblas.apply %sparse_tensor { apply_operator = "abs" } : (tensor<3xi64, #CV64>) to tensor<3xi64, #CV64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<3xi64, #CV64>
    }

    // CHECK: func @apply_matrix_ainv(%[[ARG0:.*]]: [[CSR_TYPE:tensor<.*->.*>]]) -> [[RETURN_TYPE:.*]] {
    func @apply_matrix_ainv(%sparse_tensor: tensor<2x3xi64, #CSR64>) -> tensor<2x3xi64, #CSR64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.apply %[[ARG0]] {apply_operator = "ainv"} : ([[CSR_TYPE]]) to [[CSR_TYPE]]
        %answer = graphblas.apply %sparse_tensor { apply_operator = "ainv" } : (tensor<2x3xi64, #CSR64>) to tensor<2x3xi64, #CSR64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<2x3xi64, #CSR64>
    }
   
    // CHECK: func @apply_vector_ainv(%[[ARG0:.*]]: [[VECTOR_TYPE:tensor<.*>]]) -> [[RETURN_TYPE:.*]] {
    func @apply_vector_ainv(%sparse_tensor: tensor<3xi64, #CV64>) -> tensor<3xi64, #CV64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.apply %[[ARG0]] {apply_operator = "ainv"} : ([[VECTOR_TYPE]]) to [[VECTOR_TYPE]]
        %answer = graphblas.apply %sparse_tensor { apply_operator = "ainv" } : (tensor<3xi64, #CV64>) to tensor<3xi64, #CV64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<3xi64, #CV64>
    }

    // CHECK: func @apply_matrix_identity(%[[ARG0:.*]]: [[CSR_TYPE:tensor<.*->.*>]]) -> [[RETURN_TYPE:.*]] {
    func @apply_matrix_identity(%sparse_tensor: tensor<2x3xi64, #CSR64>) -> tensor<2x3xi64, #CSR64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.apply %[[ARG0]] {apply_operator = "identity"} : ([[CSR_TYPE]]) to [[CSR_TYPE]]
        %answer = graphblas.apply %sparse_tensor { apply_operator = "identity" } : (tensor<2x3xi64, #CSR64>) to tensor<2x3xi64, #CSR64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<2x3xi64, #CSR64>
    }
   
    // CHECK: func @apply_vector_identity(%[[ARG0:.*]]: [[VECTOR_TYPE:tensor<.*>]]) -> [[RETURN_TYPE:.*]] {
    func @apply_vector_identity(%sparse_tensor: tensor<3xi64, #CV64>) -> tensor<3xi64, #CV64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.apply %[[ARG0]] {apply_operator = "identity"} : ([[VECTOR_TYPE]]) to [[VECTOR_TYPE]]
        %answer = graphblas.apply %sparse_tensor { apply_operator = "identity" } : (tensor<3xi64, #CV64>) to tensor<3xi64, #CV64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<3xi64, #CV64>
    }

    // CHECK: func @apply_matrix_minv(%[[ARG0:.*]]: [[CSR_TYPE:tensor<.*->.*>]]) -> [[RETURN_TYPE:.*]] {
    func @apply_matrix_minv(%sparse_tensor: tensor<2x3xi64, #CSR64>) -> tensor<2x3xi64, #CSR64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.apply %[[ARG0]] {apply_operator = "minv"} : ([[CSR_TYPE]]) to [[CSR_TYPE]]
        %answer = graphblas.apply %sparse_tensor { apply_operator = "minv" } : (tensor<2x3xi64, #CSR64>) to tensor<2x3xi64, #CSR64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<2x3xi64, #CSR64>
    }
   
    // CHECK: func @apply_vector_minv(%[[ARG0:.*]]: [[VECTOR_TYPE:tensor<.*>]]) -> [[RETURN_TYPE:.*]] {
    func @apply_vector_minv(%sparse_tensor: tensor<3xi64, #CV64>) -> tensor<3xi64, #CV64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.apply %[[ARG0]] {apply_operator = "minv"} : ([[VECTOR_TYPE]]) to [[VECTOR_TYPE]]
        %answer = graphblas.apply %sparse_tensor { apply_operator = "minv" } : (tensor<3xi64, #CV64>) to tensor<3xi64, #CV64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<3xi64, #CV64>
    }

    // CHECK: func @apply_matrix_left_div(%[[ARG0:.*]]: [[CSR_TYPE:tensor<.*->.*>]]) -> [[RETURN_TYPE:.*]] {
    func @apply_matrix_left_div(%sparse_tensor: tensor<2x3xi64, #CSR64>) -> tensor<2x3xi64, #CSR64> {
        // CHECK-NEXT: %[[THUNK:.*]] = constant 100 : [[THUNK_TYPE:.*]]
        %thunk = constant 100 : i64
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.apply %[[THUNK]], %[[ARG0]] {apply_operator = "div"} : ([[THUNK_TYPE]], [[CSR_TYPE]]) to [[CSR_TYPE]]
        %answer = graphblas.apply %thunk, %sparse_tensor { apply_operator = "div" } : (i64, tensor<2x3xi64, #CSR64>) to tensor<2x3xi64, #CSR64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<2x3xi64, #CSR64>
    }
   
    // CHECK: func @apply_vector_left_div(%[[ARG0:.*]]: [[VECTOR_TYPE:tensor<.*>]]) -> [[RETURN_TYPE:.*]] {
    func @apply_vector_left_div(%sparse_tensor: tensor<3xi64, #CV64>) -> tensor<3xi64, #CV64> {
        // CHECK-NEXT: %[[THUNK:.*]] = constant 100 : [[THUNK_TYPE:.*]]
        %thunk = constant 100 : i64
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.apply %[[THUNK]], %[[ARG0]] {apply_operator = "div"} : ([[THUNK_TYPE]], [[VECTOR_TYPE]]) to [[VECTOR_TYPE]]
        %answer = graphblas.apply %thunk, %sparse_tensor { apply_operator = "div" } : (i64, tensor<3xi64, #CV64>) to tensor<3xi64, #CV64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<3xi64, #CV64>
    }

    // CHECK: func @apply_matrix_right_div(%[[ARG0:.*]]: [[CSR_TYPE:tensor<.*->.*>]]) -> [[RETURN_TYPE:.*]] {
    func @apply_matrix_right_div(%sparse_tensor: tensor<2x3xi64, #CSR64>) -> tensor<2x3xi64, #CSR64> {
        // CHECK-NEXT: %[[THUNK:.*]] = constant 100 : [[THUNK_TYPE:.*]]
        %thunk = constant 100 : i64
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.apply %[[ARG0]], %[[THUNK]] {apply_operator = "div"} : ([[CSR_TYPE]], [[THUNK_TYPE]]) to [[CSR_TYPE]]
        %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "div" } : (tensor<2x3xi64, #CSR64>, i64) to tensor<2x3xi64, #CSR64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<2x3xi64, #CSR64>
    }
   
    // CHECK: func @apply_vector_right_div(%[[ARG0:.*]]: [[VECTOR_TYPE:tensor<.*>]]) -> [[RETURN_TYPE:.*]] {
    func @apply_vector_right_div(%sparse_tensor: tensor<3xi64, #CV64>) -> tensor<3xi64, #CV64> {
        // CHECK-NEXT: %[[THUNK:.*]] = constant 100 : [[THUNK_TYPE:.*]]
        %thunk = constant 100 : i64
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.apply %[[ARG0]], %[[THUNK]] {apply_operator = "div"} : ([[VECTOR_TYPE]], [[THUNK_TYPE]]) to [[VECTOR_TYPE]]
        %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "div" } : (tensor<3xi64, #CV64>, i64) to tensor<3xi64, #CV64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<3xi64, #CV64>
    }

}

module {

    // COM: TODO as part of https://github.com/metagraph-dev/mlir-graphblas/issues/66 , handle all posssible semirings here.

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
    func @matrix_vector_multiply_plus_times(%matrix: tensor<2x3xi64, #CSR64>, %vector: tensor<3xi64, #CV64>) -> tensor<2xi64, #CV64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_multiply %[[MATRIX]], %[[VECTOR]] {semiring = "plus_times"} : ([[MATRIX_TYPE]], [[VECTOR_TYPE]]) to [[RETURN_TYPE]]
        %answer = graphblas.matrix_multiply %matrix, %vector { semiring = "plus_times" } : (tensor<2x3xi64, #CSR64>, tensor<3xi64, #CV64>) to tensor<2xi64, #CV64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<2xi64, #CV64>
    }

    // CHECK: func @matrix_vector_multiply_with_mask_plus_times(%[[MATRIX:.*]]: [[MATRIX_TYPE:tensor<.*->.*>]], %[[VECTOR:.*]]: [[VECTOR_TYPE:tensor<.*>]], %[[MASK:.*]]: [[MASK_TYPE:tensor<.*>]]) -> [[RETURN_TYPE:tensor<.*>]] {
    func @matrix_vector_multiply_with_mask_plus_times(%matrix: tensor<2x2xf64, #CSR64>, %vector: tensor<2xf64, #CV64>, %mask: tensor<2xf64, #CV64>) -> tensor<2xf64, #CV64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_multiply %[[MATRIX]], %[[VECTOR]], %[[MASK]] {semiring = "plus_times"} : ([[MATRIX_TYPE]], [[VECTOR_TYPE]], [[MASK_TYPE]]) to [[RETURN_TYPE]]
        %answer = graphblas.matrix_multiply %matrix, %vector, %mask { semiring = "plus_times" } : (tensor<2x2xf64, #CSR64>, tensor<2xf64, #CV64>, tensor<2xf64, #CV64>) to tensor<2xf64, #CV64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<2xf64, #CV64>
    }

    // CHECK: func @vector_matrix_multiply_plus_times(%[[VECTOR:.*]]: [[VECTOR_TYPE:tensor<.*>]], %[[MATRIX:.*]]: [[MATRIX_TYPE:tensor<.*->.*>]]) -> [[RETURN_TYPE:tensor<.*>]] {
    func @vector_matrix_multiply_plus_times(%vector: tensor<3xi64, #CV64>, %matrix: tensor<3x2xi64, #CSC64>) -> tensor<2xi64, #CV64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_multiply %[[VECTOR]], %[[MATRIX]] {semiring = "plus_times"} : ([[VECTOR_TYPE]], [[MATRIX_TYPE]]) to [[RETURN_TYPE]]
        %answer = graphblas.matrix_multiply %vector, %matrix { semiring = "plus_times" } : (tensor<3xi64, #CV64>, tensor<3x2xi64, #CSC64>) to tensor<2xi64, #CV64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<2xi64, #CV64>
    }

    // CHECK: func @vector_matrix_multiply_with_mask_plus_times(%[[VECTOR:.*]]: [[VECTOR_TYPE:tensor<.*>]], %[[MATRIX:.*]]: [[MATRIX_TYPE:tensor<.*->.*>]], %[[MASK:.*]]: [[MASK_TYPE:tensor<.*>]]) -> [[RETURN_TYPE:tensor<.*>]] {
    func @vector_matrix_multiply_with_mask_plus_times(%vector: tensor<2xf64, #CV64>, %matrix: tensor<2x2xf64, #CSC64>, %mask: tensor<2xf64, #CV64>) -> tensor<2xf64, #CV64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_multiply %[[VECTOR]], %[[MATRIX]], %[[MASK]] {semiring = "plus_times"} : ([[VECTOR_TYPE]], [[MATRIX_TYPE]], [[MASK_TYPE]]) to [[RETURN_TYPE]]
        %answer = graphblas.matrix_multiply %vector, %matrix, %mask { semiring = "plus_times" } : (tensor<2xf64, #CV64>, tensor<2x2xf64, #CSC64>, tensor<2xf64, #CV64>) to tensor<2xf64, #CV64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<2xf64, #CV64>
    }

    // CHECK: func @vector_dot_product_plus_times(%[[ARGA:.*]]: [[A_TYPE:tensor<.*>]], %[[ARGB:.*]]: [[B_TYPE:tensor<.*>]]) -> [[RETURN_TYPE:.*]] {
    func @vector_dot_product_plus_times(%argA: tensor<3xi64, #CV64>, %argB: tensor<3xi64, #CV64>) -> i64 {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_multiply %[[ARGA]], %[[ARGB]] {semiring = "plus_times"} : ([[A_TYPE]], [[B_TYPE]]) to [[RETURN_TYPE]]
        %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_times" } : (tensor<3xi64, #CV64>, tensor<3xi64, #CV64>) to i64
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : i64
    }

}

module {

    // CHECK: func @vector_update(%[[ARGA:.*]]: [[A_TYPE:tensor<.*>]], %[[ARGB:.*]]: [[B_TYPE:tensor<.*>]]) {
    func @vector_update(%other_vec: tensor<?xi64, #CV64>, %vec: tensor<?xi64, #CV64>) {
        // CHECK-NEXT: graphblas.update %[[ARGA]] -> %[[ARGB]] : [[A_TYPE]] -> [[B_TYPE]]
        graphblas.update %other_vec -> %vec : tensor<?xi64, #CV64> -> tensor<?xi64, #CV64>
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
    func @vector_equaity_checking(%argA: tensor<3xi64, #CV64>, %argB: tensor<3xi64, #CV64>) -> i1 {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.equal %[[ARGA]], %[[ARGB]] : [[A_TYPE]], [[B_TYPE]]
        %answer = graphblas.equal %argA, %argB : tensor<3xi64, #CV64>, tensor<3xi64, #CV64>
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

    // CHECK: func @print_wrapper() {
    func @print_wrapper() -> () {
    	// CHECK-NEXT: %[[VAL_0:.*]] = constant 0.000000e+00 : f32
        %0 = constant 0.0 : f32
	// CHECK-NEXT: graphblas.print %[[VAL_0]], %[[VAL_0]] {strings = ["start ", " middle ", " end"]} : f32, f32
	graphblas.print %0, %0 { strings = ["start ", " middle ", " end"] } : f32, f32
        // CHECK-NEXT: return
        return
    }

}

module {

    // CHECK: func @vector_argminmax_min(%[[ARGA:.*]]: [[A_TYPE:tensor<.*>]]) -> [[RETURN_TYPE:.*]] {
    func @vector_argminmax_min(%vec: tensor<3xi64, #CV64>) -> index {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.vector_argminmax %[[ARGA]] {minmax = "min"} : [[A_TYPE]]
        %answer = graphblas.vector_argminmax %vec { minmax = "min" } : tensor<3xi64, #CV64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : index
    }

    // CHECK: func @vector_argminmax_max(%[[ARGA:.*]]: [[A_TYPE:tensor<.*>]]) -> [[RETURN_TYPE:.*]] {
    func @vector_argminmax_max(%vec: tensor<3xi64, #CV64>) -> index {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.vector_argminmax %[[ARGA]] {minmax = "max"} : [[A_TYPE]]
        %answer = graphblas.vector_argminmax %vec { minmax = "max" } : tensor<3xi64, #CV64>
        // COM: CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : index
    }

}

module {

    // CHECK: func @vector_argmin_wrapper(%[[ARGA:.*]]: [[A_TYPE:tensor<.*>]]) -> [[RETURN_TYPE:.*]] {
    func @vector_argmin_wrapper(%vec: tensor<3xi64, #CV64>) -> index {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.vector_argmin %[[ARGA]] : [[A_TYPE]]
        %answer = graphblas.vector_argmin %vec : tensor<3xi64, #CV64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : index
    }

}

module {

    // CHECK: func @vector_argmax_wrapper(%[[ARGA:.*]]: [[A_TYPE:tensor<.*>]]) -> [[RETURN_TYPE:.*]] {
    func @vector_argmax_wrapper(%vec: tensor<3xi64, #CV64>) -> index {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.vector_argmax %[[ARGA]] : [[A_TYPE]]
        %answer = graphblas.vector_argmax %vec : tensor<3xi64, #CV64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : index
    }

}

module {

    // CHECK: func @diag_vec_to_mat_csr(%[[VECTOR:.*]]: [[VECTOR_TYPE:tensor<.*>]]) -> [[MATRIX_TYPE:tensor<.*->.*>]] {
    func @diag_vec_to_mat_csr(%vec: tensor<7xi64, #CV64>) -> tensor<7x7xi64, #CSR64> {
        // CHECK-NEXT: %[[MATRIX:.*]] = graphblas.diag %[[VECTOR]] : [[VECTOR_TYPE]] to [[MATRIX_TYPE]]
        %mat = graphblas.diag %vec : tensor<7xi64, #CV64> to tensor<7x7xi64, #CSR64>
        // CHECK-NEXT: return %[[MATRIX]] : [[MATRIX_TYPE]]
        return %mat : tensor<7x7xi64, #CSR64>
    }

    // CHECK: func @diag_mat_to_vec_csr(%[[MATRIX:.*]]: [[MATRIX_TYPE:tensor<.*->.*>]]) -> [[VECTOR_TYPE:tensor<.*>]] {
    func @diag_mat_to_vec_csr(%mat: tensor<7x7xi64, #CSR64>) -> tensor<7xi64, #CV64> {
        // CHECK-NEXT: %[[VECTOR:.*]] = graphblas.diag %[[MATRIX]] : [[MATRIX_TYPE]] to [[VECTOR_TYPE]]
        %vec = graphblas.diag %mat : tensor<7x7xi64, #CSR64> to tensor<7xi64, #CV64>
        // CHECK-NEXT: return %[[VECTOR]] : [[VECTOR_TYPE]]
        return %vec : tensor<7xi64, #CV64>
    }

    // CHECK: func @diag_vec_to_mat_csc(%[[VECTOR:.*]]: [[VECTOR_TYPE:tensor<.*>]]) -> [[MATRIX_TYPE:tensor<.*->.*>]] {
    func @diag_vec_to_mat_csc(%vec: tensor<7xi64, #CV64>) -> tensor<7x7xi64, #CSC64> {
        // CHECK-NEXT: %[[MATRIX:.*]] = graphblas.diag %[[VECTOR]] : [[VECTOR_TYPE]] to [[MATRIX_TYPE]]
        %mat = graphblas.diag %vec : tensor<7xi64, #CV64> to tensor<7x7xi64, #CSC64>
        // CHECK-NEXT: return %[[MATRIX]] : [[MATRIX_TYPE]]
        return %mat : tensor<7x7xi64, #CSC64>
    }

    // CHECK: func @diag_mat_to_vec_csc(%[[MATRIX:.*]]: [[MATRIX_TYPE:tensor<.*->.*>]]) -> [[VECTOR_TYPE:tensor<.*>]] {
    func @diag_mat_to_vec_csc(%mat: tensor<7x7xi64, #CSC64>) -> tensor<7xi64, #CV64> {
        // CHECK-NEXT: %[[VECTOR:.*]] = graphblas.diag %[[MATRIX]] : [[MATRIX_TYPE]] to [[VECTOR_TYPE]]
        %vec = graphblas.diag %mat : tensor<7x7xi64, #CSC64> to tensor<7xi64, #CV64>
        // CHECK-NEXT: return %[[VECTOR]] : [[VECTOR_TYPE]]
        return %vec : tensor<7xi64, #CV64>
    }

}

module {

    // CHECK: func @reduce_to_vector_plus(%[[MATRIX:.*]]: [[MATRIX_TYPE:tensor<.*->.*>]]) -> ([[RETURN_TYPE_0:tensor<.*>]], [[RETURN_TYPE_1:tensor<.*>]]) {
    func @reduce_to_vector_plus(%matrix: tensor<7x9xi32, #CSR64>) -> (tensor<9xi32, #CV64>, tensor<7xi32, #CV64>) {
        // CHECK-NEXT: %[[ANSWER_0:.*]] = graphblas.reduce_to_vector %[[MATRIX]] {aggregator = "plus", axis = 0 : i64} : [[MATRIX_TYPE]] to [[RETURN_TYPE_0]]
        %vec1 = graphblas.reduce_to_vector %matrix { aggregator = "plus", axis = 0 } : tensor<7x9xi32, #CSR64> to tensor<9xi32, #CV64>
        // CHECK-NEXT: %[[ANSWER_1:.*]] = graphblas.reduce_to_vector %[[MATRIX]] {aggregator = "plus", axis = 1 : i64} : [[MATRIX_TYPE]] to [[RETURN_TYPE_1]]
        %vec2 = graphblas.reduce_to_vector %matrix { aggregator = "plus", axis = 1 } : tensor<7x9xi32, #CSR64> to tensor<7xi32, #CV64>
        // CHECK-NEXT: return %[[ANSWER_0]], %[[ANSWER_1]] : [[RETURN_TYPE_0]], [[RETURN_TYPE_1]]
        return %vec1, %vec2 : tensor<9xi32, #CV64>, tensor<7xi32, #CV64>
    }

    // CHECK: func @reduce_to_vector_count(%[[MATRIX:.*]]: [[MATRIX_TYPE:tensor<.*->.*>]]) -> ([[RETURN_TYPE_0:tensor<.*>]], [[RETURN_TYPE_1:tensor<.*>]]) {
    func @reduce_to_vector_count(%matrix: tensor<7x9xi32, #CSR64>) -> (tensor<9xi32, #CV64>, tensor<7xi32, #CV64>) {
        // CHECK-NEXT: %[[ANSWER_0:.*]] = graphblas.reduce_to_vector %[[MATRIX]] {aggregator = "count", axis = 0 : i64} : [[MATRIX_TYPE]] to [[RETURN_TYPE_0]]
        %vec1 = graphblas.reduce_to_vector %matrix { aggregator = "count", axis = 0 } : tensor<7x9xi32, #CSR64> to tensor<9xi32, #CV64>
        // CHECK-NEXT: %[[ANSWER_1:.*]] = graphblas.reduce_to_vector %[[MATRIX]] {aggregator = "count", axis = 1 : i64} : [[MATRIX_TYPE]] to [[RETURN_TYPE_1]]
        %vec2 = graphblas.reduce_to_vector %matrix { aggregator = "count", axis = 1 } : tensor<7x9xi32, #CSR64> to tensor<7xi32, #CV64>
        // CHECK-NEXT: return %[[ANSWER_0]], %[[ANSWER_1]] : [[RETURN_TYPE_0]], [[RETURN_TYPE_1]]
        return %vec1, %vec2 : tensor<9xi32, #CV64>, tensor<7xi32, #CV64>
    }

    // CHECK: func @reduce_to_vector_argmin(%[[MATRIX:.*]]: [[MATRIX_TYPE:tensor<.*->.*>]]) -> ([[RETURN_TYPE_0:tensor<.*>]], [[RETURN_TYPE_1:tensor<.*>]]) {
    func @reduce_to_vector_argmin(%matrix: tensor<7x9xf32, #CSR64>) -> (tensor<9xi64, #CV64>, tensor<7xi64, #CV64>) {
        // CHECK-NEXT: %[[ANSWER_0:.*]] = graphblas.reduce_to_vector %[[MATRIX]] {aggregator = "argmin", axis = 0 : i64} : [[MATRIX_TYPE]] to [[RETURN_TYPE_0]]
        %vec1 = graphblas.reduce_to_vector %matrix { aggregator = "argmin", axis = 0 } : tensor<7x9xf32, #CSR64> to tensor<9xi64, #CV64>
        // CHECK-NEXT: %[[ANSWER_1:.*]] = graphblas.reduce_to_vector %[[MATRIX]] {aggregator = "argmin", axis = 1 : i64} : [[MATRIX_TYPE]] to [[RETURN_TYPE_1]]
        %vec2 = graphblas.reduce_to_vector %matrix { aggregator = "argmin", axis = 1 } : tensor<7x9xf32, #CSR64> to tensor<7xi64, #CV64>
        // CHECK-NEXT: return %[[ANSWER_0]], %[[ANSWER_1]] : [[RETURN_TYPE_0]], [[RETURN_TYPE_1]]
        return %vec1, %vec2 : tensor<9xi64, #CV64>, tensor<7xi64, #CV64>
    }

    // CHECK: func @reduce_to_vector_argmax(%[[MATRIX:.*]]: [[MATRIX_TYPE:tensor<.*->.*>]]) -> ([[RETURN_TYPE_0:tensor<.*>]], [[RETURN_TYPE_1:tensor<.*>]]) {
    func @reduce_to_vector_argmax(%matrix: tensor<7x9xf32, #CSR64>) -> (tensor<9xi64, #CV64>, tensor<7xi64, #CV64>) {
        // CHECK-NEXT: %[[ANSWER_0:.*]] = graphblas.reduce_to_vector %[[MATRIX]] {aggregator = "argmax", axis = 0 : i64} : [[MATRIX_TYPE]] to [[RETURN_TYPE_0]]
        %vec1 = graphblas.reduce_to_vector %matrix { aggregator = "argmax", axis = 0 } : tensor<7x9xf32, #CSR64> to tensor<9xi64, #CV64>
        // CHECK-NEXT: %[[ANSWER_1:.*]] = graphblas.reduce_to_vector %[[MATRIX]] {aggregator = "argmax", axis = 1 : i64} : [[MATRIX_TYPE]] to [[RETURN_TYPE_1]]
        %vec2 = graphblas.reduce_to_vector %matrix { aggregator = "argmax", axis = 1 } : tensor<7x9xf32, #CSR64> to tensor<7xi64, #CV64>
        // CHECK-NEXT: return %[[ANSWER_0]], %[[ANSWER_1]] : [[RETURN_TYPE_0]], [[RETURN_TYPE_1]]
        return %vec1, %vec2 : tensor<9xi64, #CV64>, tensor<7xi64, #CV64>
    }

}
