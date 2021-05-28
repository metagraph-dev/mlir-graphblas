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

module {
    
    // CHECK: func @transpose_wrapper_no_swap(%[[ARG0:.*]]: [[CSR_TYPE:tensor<.*->.*>]]) -> [[CSC_TYPE:tensor<.*->.*>]] {
    func @transpose_wrapper_no_swap(%sparse_tensor: tensor<2x3xf64, #CSR64>) -> tensor<2x3xf64, #CSC64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.transpose %[[ARG0]] {swap_sizes = false} : [[CSR_TYPE]] to [[CSC_TYPE]]
        %answer = graphblas.transpose %sparse_tensor { swap_sizes = false } : tensor<2x3xf64, #CSR64> to tensor<2x3xf64, #CSC64>
        // CHECK-NEXT: return %[[ANSWER]] : [[CSC_TYPE]]
        return %answer : tensor<2x3xf64, #CSC64>
    }
    
    // CHECK: func @transpose_wrapper(%[[ARG0:.*]]: tensor<2x3xf64, [[CSR64:#sparse_tensor.encoding<{.*->.*}>]]>) -> tensor<3x2xf64, [[CSC64:#sparse_tensor.encoding<{.*->.*}>]]> {
    func @transpose_wrapper(%sparse_tensor: tensor<2x3xf64, #CSR64>) -> tensor<3x2xf64, #CSC64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.transpose %[[ARG0]] {swap_sizes = true} : tensor<2x3xf64, [[CSR64]]> to tensor<3x2xf64, [[CSC64]]>
        %answer = graphblas.transpose %sparse_tensor { swap_sizes = true } : tensor<2x3xf64, #CSR64> to tensor<3x2xf64, #CSC64>
        // CHECK-NEXT: return %[[ANSWER]] : tensor<3x2xf64, [[CSC64]]>
        return %answer : tensor<3x2xf64, #CSC64>
    }
        
}

module {
        
    // CHECK: func @matrix_select_triu(%[[ARG0:.*]]: [[CSR_TYPE:tensor<.*->.*>]]) -> [[CSR_TYPE]] {
    func @matrix_select_triu(%sparse_tensor: tensor<100x100xf64, #CSR64>) -> tensor<100x100xf64, #CSR64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_select %[[ARG0]] {selector = "triu"} : [[CSR_TYPE]]
        %answer = graphblas.matrix_select %sparse_tensor { selector = "triu" } : tensor<100x100xf64, #CSR64>
        // CHECK-NEXT: return %[[ANSWER]] : [[CSR_TYPE]]
        return %answer : tensor<100x100xf64, #CSR64>
    }
    
    // CHECK: func @matrix_select_tril(%[[ARG0:.*]]: [[CSR_TYPE:tensor<.*->.*>]]) -> [[CSR_TYPE]] {
    func @matrix_select_tril(%sparse_tensor: tensor<100x100xf64, #CSR64>) -> tensor<100x100xf64, #CSR64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_select %[[ARG0]] {selector = "tril"} : [[CSR_TYPE]]
        %answer = graphblas.matrix_select %sparse_tensor { selector = "tril" } : tensor<100x100xf64, #CSR64>
        // CHECK-NEXT: return %[[ANSWER]] : [[CSR_TYPE]]
        return %answer : tensor<100x100xf64, #CSR64>
    }
    
    // CHECK: func @matrix_select_gt0(%[[ARG0:.*]]: [[CSR_TYPE:tensor<.*->.*>]]) -> [[CSR_TYPE]] {
    func @matrix_select_gt0(%sparse_tensor: tensor<100x100xf64, #CSR64>) -> tensor<100x100xf64, #CSR64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_select %[[ARG0]] {selector = "gt0"} : [[CSR_TYPE]]
        %answer = graphblas.matrix_select %sparse_tensor { selector = "gt0" } : tensor<100x100xf64, #CSR64>
        // CHECK-NEXT: return %[[ANSWER]] : [[CSR_TYPE]]
        return %answer : tensor<100x100xf64, #CSR64>
    }
    
}

module {
    
    // CHECK: func @matrix_reduce_to_scalar(%[[ARG0:.*]]: [[CSR_TYPE:tensor<.*->.*>]]) -> [[RETURN_TYPE:.*]] {
    func @matrix_reduce_to_scalar(%sparse_tensor: tensor<2x3xi64, #CSR64>) -> i64 {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_reduce_to_scalar %[[ARG0]] {aggregator = "sum"} : [[CSR_TYPE]] to [[RETURN_TYPE]]
        %answer = graphblas.matrix_reduce_to_scalar %sparse_tensor { aggregator = "sum" } : tensor<2x3xi64, #CSR64> to i64
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
    
    // CHECK: func @matrix_multiply_plus_times(%[[ARGA:.*]]: [[CSR_TYPE_A:tensor<.*->.*>]], %[[ARGB:.*]]: [[CSR_TYPE_B:tensor<.*->.*>]]) -> [[RETURN_TYPE:tensor<.*->.*>]] {
    func @matrix_multiply_plus_times(%argA: tensor<2x3xi64, #CSR64>, %argB: tensor<3x2xi64, #CSR64>) -> tensor<2x2xi64, #CSR64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_multiply %[[ARGA]], %[[ARGB]] {semiring = "plus_times"} : ([[CSR_TYPE_A]], [[CSR_TYPE_B]]) to [[RETURN_TYPE]]
        %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_times" } : (tensor<2x3xi64, #CSR64>, tensor<3x2xi64, #CSR64>) to tensor<2x2xi64, #CSR64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<2x2xi64, #CSR64>
    }
    
    // CHECK: func @matrix_multiply_plus_pair(%[[ARGA:.*]]: [[CSR_TYPE_A:tensor<.*->.*>]], %[[ARGB:.*]]: [[CSR_TYPE_B:tensor<.*->.*>]]) -> [[RETURN_TYPE:tensor<.*->.*>]] {
    func @matrix_multiply_plus_pair(%argA: tensor<2x3xi64, #CSR64>, %argB: tensor<3x2xi64, #CSR64>) -> tensor<2x2xi64, #CSR64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_multiply %[[ARGA]], %[[ARGB]] {semiring = "plus_pair"} : ([[CSR_TYPE_A]], [[CSR_TYPE_B]]) to [[RETURN_TYPE]]
        %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_pair" } : (tensor<2x3xi64, #CSR64>, tensor<3x2xi64, #CSR64>) to tensor<2x2xi64, #CSR64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<2x2xi64, #CSR64>
    }
    
    // CHECK: func @matrix_multiply_plus_plus(%[[ARGA:.*]]: [[CSR_TYPE_A:tensor<.*->.*>]], %[[ARGB:.*]]: [[CSR_TYPE_B:tensor<.*->.*>]]) -> [[RETURN_TYPE:tensor<.*->.*>]] {
    func @matrix_multiply_plus_plus(%argA: tensor<2x3xi64, #CSR64>, %argB: tensor<3x2xi64, #CSR64>) -> tensor<2x2xi64, #CSR64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_multiply %[[ARGA]], %[[ARGB]] {semiring = "plus_plus"} : ([[CSR_TYPE_A]], [[CSR_TYPE_B]]) to [[RETURN_TYPE]]
        %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_plus" } : (tensor<2x3xi64, #CSR64>, tensor<3x2xi64, #CSR64>) to tensor<2x2xi64, #CSR64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<2x2xi64, #CSR64>
    }
            
}

module {
    
    // CHECK: func @matrix_multiply_mask_plus_times(%[[ARGA:.*]]: [[TENSOR_TYPE:tensor<.*->.*>]], %[[ARGB:.*]]: [[TENSOR_TYPE]], %[[MASK:.*]]: [[TENSOR_TYPE]]) -> [[RETURN_TYPE:tensor<.*->.*>]] {
    func @matrix_multiply_mask_plus_times(%argA: tensor<2x2xf64, #CSC64>, %argB: tensor<2x2xf64, #CSC64>, %mask: tensor<2x2xf64, #CSC64>) -> tensor<2x2xf64, #CSC64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_multiply_mask %[[ARGA]], %[[ARGB]], %[[MASK]] {semiring = "plus_times"} : ([[TENSOR_TYPE]], [[TENSOR_TYPE]], [[TENSOR_TYPE]]) to [[TENSOR_TYPE]]
        %answer = graphblas.matrix_multiply_mask %argA, %argB, %mask { semiring = "plus_times" } : (tensor<2x2xf64, #CSC64>, tensor<2x2xf64, #CSC64>, tensor<2x2xf64, #CSC64>) to tensor<2x2xf64, #CSC64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<2x2xf64, #CSC64>
    }
    
    // CHECK: func @matrix_multiply_mask_plus_pair(%[[ARGA:.*]]: [[TENSOR_TYPE:tensor<.*->.*>]], %[[ARGB:.*]]: [[TENSOR_TYPE]], %[[MASK:.*]]: [[TENSOR_TYPE]]) -> [[RETURN_TYPE:tensor<.*->.*>]] {
    func @matrix_multiply_mask_plus_pair(%argA: tensor<2x2xf64, #CSC64>, %argB: tensor<2x2xf64, #CSC64>, %mask: tensor<2x2xf64, #CSC64>) -> tensor<2x2xf64, #CSC64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_multiply_mask %[[ARGA]], %[[ARGB]], %[[MASK]] {semiring = "plus_pair"} : ([[TENSOR_TYPE]], [[TENSOR_TYPE]], [[TENSOR_TYPE]]) to [[TENSOR_TYPE]]
        %answer = graphblas.matrix_multiply_mask %argA, %argB, %mask { semiring = "plus_pair" } : (tensor<2x2xf64, #CSC64>, tensor<2x2xf64, #CSC64>, tensor<2x2xf64, #CSC64>) to tensor<2x2xf64, #CSC64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<2x2xf64, #CSC64>
    }
    
    // CHECK: func @matrix_multiply_mask_plus_plus(%[[ARGA:.*]]: [[TENSOR_TYPE:tensor<.*->.*>]], %[[ARGB:.*]]: [[TENSOR_TYPE]], %[[MASK:.*]]: [[TENSOR_TYPE]]) -> [[RETURN_TYPE:tensor<.*->.*>]] {
    func @matrix_multiply_mask_plus_plus(%argA: tensor<2x2xf64, #CSC64>, %argB: tensor<2x2xf64, #CSC64>, %mask: tensor<2x2xf64, #CSC64>) -> tensor<2x2xf64, #CSC64> {
        // CHECK-NEXT: %[[ANSWER:.*]] = graphblas.matrix_multiply_mask %[[ARGA]], %[[ARGB]], %[[MASK]] {semiring = "plus_plus"} : ([[TENSOR_TYPE]], [[TENSOR_TYPE]], [[TENSOR_TYPE]]) to [[TENSOR_TYPE]]
        %answer = graphblas.matrix_multiply_mask %argA, %argB, %mask { semiring = "plus_plus" } : (tensor<2x2xf64, #CSC64>, tensor<2x2xf64, #CSC64>, tensor<2x2xf64, #CSC64>) to tensor<2x2xf64, #CSC64>
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
        return %answer : tensor<2x2xf64, #CSC64>
    }
    
}
