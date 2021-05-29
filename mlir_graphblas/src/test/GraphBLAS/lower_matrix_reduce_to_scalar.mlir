// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

// COM: graphblas.matrix_reduce_to_scalar currently lowers into a function call of an automatically generated private function.
// COM: These tests do not test the contents of these automatically generated private functions.
// COM: Testing for these automatically generated private functions should happen elsewhere where we verify their behavior as well.

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>
 
module {
    // CHECK: func private @[[HELPER_FUNCTION:.*]](%[[ARG0:.*]]: [[CSR_TYPE:tensor<.*->.*>]]) -> [[RETURN_TYPE:.*]] {
    // CHECK: func @[[WRAPPER_FUNC_NAME:.*]](%[[ARG0:.*]]: [[CSR_TYPE]]) -> [[RETURN_TYPE]] {
        // CHECK-NEXT: %[[ANSWER:.*]] = call @[[HELPER_FUNCTION]](%[[ARG0]]) : ([[CSR_TYPE]]) -> [[RETURN_TYPE]]
        // CHECK-NEXT: return %[[ANSWER]] : [[RETURN_TYPE]]
    func @matrix_reduce_to_scalar_f64(%sparse_tensor: tensor<?x?xf64, #CSR64>) -> f64 {
        %answer = graphblas.matrix_reduce_to_scalar %sparse_tensor { aggregator = "sum" } : tensor<?x?xf64, #CSR64> to f64
        return %answer : f64
    }
}
 
module {
    // CHECK: func private @[[HELPER_FUNCTION_I1:.*]](%[[ARG0:.*]]: [[TENSOR_TYPE_I1:tensor<.*->.*>]]) -> [[I1_TYPE:.*]] {
    // CHECK: func private @[[HELPER_FUNCTION_I4:.*]](%[[ARG0:.*]]: [[TENSOR_TYPE_I4:tensor<.*->.*>]]) -> [[I4_TYPE:.*]] {
    // CHECK: func private @[[HELPER_FUNCTION_I8:.*]](%[[ARG0:.*]]: [[TENSOR_TYPE_I8:tensor<.*->.*>]]) -> [[I8_TYPE:.*]] {
    // CHECK: func private @[[HELPER_FUNCTION_I16:.*]](%[[ARG0:.*]]: [[TENSOR_TYPE_I16:tensor<.*->.*>]]) -> [[I16_TYPE:.*]] {
    // CHECK: func private @[[HELPER_FUNCTION_I32:.*]](%[[ARG0:.*]]: [[TENSOR_TYPE_I32:tensor<.*->.*>]]) -> [[I32_TYPE:.*]] {
    // CHECK: func private @[[HELPER_FUNCTION_I64:.*]](%[[ARG0:.*]]: [[TENSOR_TYPE_I64:tensor<.*->.*>]]) -> [[I64_TYPE:.*]] {
    // CHECK: func private @[[HELPER_FUNCTION_F16:.*]](%[[ARG0:.*]]: [[TENSOR_TYPE_F16:tensor<.*->.*>]]) -> [[F16_TYPE:.*]] {
    // CHECK: func private @[[HELPER_FUNCTION_BF16:.*]](%[[ARG0:.*]]: [[TENSOR_TYPE_BF16:tensor<.*->.*>]]) -> [[BF16_TYPE:.*]] {
    // CHECK: func private @[[HELPER_FUNCTION_F32:.*]](%[[ARG0:.*]]: [[TENSOR_TYPE_F32:tensor<.*->.*>]]) -> [[F32_TYPE:.*]] {
    // CHECK: func private @[[HELPER_FUNCTION_F64:.*]](%[[ARG0:.*]]: [[TENSOR_TYPE_F64:tensor<.*->.*>]]) -> [[F64_TYPE:.*]] {
    // CHECK: func private @[[HELPER_FUNCTION_F80:.*]](%[[ARG0:.*]]: [[TENSOR_TYPE_F80:tensor<.*->.*>]]) -> [[F80_TYPE:.*]] {
    // CHECK: func private @[[HELPER_FUNCTION_F128:.*]](%[[ARG0:.*]]: [[TENSOR_TYPE_F128:tensor<.*->.*>]]) -> [[F128_TYPE:.*]] {
    
    // CHECK: func @[[MAIN_FUNC_NAME:.*]](%[[ARG_I1:.*]]: [[TENSOR_TYPE_I1]], %[[ARG_I4:.*]]: [[TENSOR_TYPE_I4]], %[[ARG_I8:.*]]: [[TENSOR_TYPE_I8]], %[[ARG_I16:.*]]: [[TENSOR_TYPE_I16]], %[[ARG_I32:.*]]: [[TENSOR_TYPE_I32]], %[[ARG_I64:.*]]: [[TENSOR_TYPE_I64]], %[[ARG_F16:.*]]: [[TENSOR_TYPE_F16]], %[[ARG_BF16:.*]]: [[TENSOR_TYPE_BF16]], %[[ARG_F32:.*]]: [[TENSOR_TYPE_F32]], %[[ARG_F64:.*]]: [[TENSOR_TYPE_F64]], %[[ARG_F80:.*]]: [[TENSOR_TYPE_F80]], %[[ARG_F128:.*]]: [[TENSOR_TYPE_F128]]) -> ([[I1_TYPE]], [[I4_TYPE]], [[I8_TYPE]], [[I16_TYPE]], [[I32_TYPE]], [[I64_TYPE]], [[F16_TYPE]], [[BF16_TYPE]], [[F32_TYPE]], [[F64_TYPE]], [[F80_TYPE]], [[F128_TYPE]]) {
        // CHECK-NEXT: %[[ANSWER_I1:.*]] = call @[[HELPER_FUNCTION_I1]](%[[ARG_I1]]) : ([[TENSOR_TYPE_I1]]) -> [[I1_TYPE]]
        // CHECK-NEXT: %[[ANSWER_I4:.*]] = call @[[HELPER_FUNCTION_I4]](%[[ARG_I4]]) : ([[TENSOR_TYPE_I4]]) -> [[I4_TYPE]]
        // CHECK-NEXT: %[[ANSWER_I8:.*]] = call @[[HELPER_FUNCTION_I8]](%[[ARG_I8]]) : ([[TENSOR_TYPE_I8]]) -> [[I8_TYPE]]
        // CHECK-NEXT: %[[ANSWER_I16:.*]] = call @[[HELPER_FUNCTION_I16]](%[[ARG_I16]]) : ([[TENSOR_TYPE_I16]]) -> [[I16_TYPE]]
        // CHECK-NEXT: %[[ANSWER_I32:.*]] = call @[[HELPER_FUNCTION_I32]](%[[ARG_I32]]) : ([[TENSOR_TYPE_I32]]) -> [[I32_TYPE]]
        // CHECK-NEXT: %[[ANSWER_I64:.*]] = call @[[HELPER_FUNCTION_I64]](%[[ARG_I64]]) : ([[TENSOR_TYPE_I64]]) -> [[I64_TYPE]]
        // CHECK-NEXT: %[[ANSWER_F16:.*]] = call @[[HELPER_FUNCTION_F16]](%[[ARG_F16]]) : ([[TENSOR_TYPE_F16]]) -> [[F16_TYPE]]
        // CHECK-NEXT: %[[ANSWER_BF16:.*]] = call @[[HELPER_FUNCTION_BF16]](%[[ARG_BF16]]) : ([[TENSOR_TYPE_BF16]]) -> [[BF16_TYPE]]
        // CHECK-NEXT: %[[ANSWER_F32:.*]] = call @[[HELPER_FUNCTION_F32]](%[[ARG_F32]]) : ([[TENSOR_TYPE_F32]]) -> [[F32_TYPE]]
        // CHECK-NEXT: %[[ANSWER_F64:.*]] = call @[[HELPER_FUNCTION_F64]](%[[ARG_F64]]) : ([[TENSOR_TYPE_F64]]) -> [[F64_TYPE]]
        // CHECK-NEXT: %[[ANSWER_F80:.*]] = call @[[HELPER_FUNCTION_F80]](%[[ARG_F80]]) : ([[TENSOR_TYPE_F80]]) -> [[F80_TYPE]]
        // CHECK-NEXT: %[[ANSWER_F128:.*]] = call @[[HELPER_FUNCTION_F128]](%[[ARG_F128]]) : ([[TENSOR_TYPE_F128]]) -> [[F128_TYPE]]
        // CHECK-NEXT: return %[[ANSWER_I1]], %[[ANSWER_I4]], %[[ANSWER_I8]], %[[ANSWER_I16]], %[[ANSWER_I32]], %[[ANSWER_I64]], %[[ANSWER_F16]], %[[ANSWER_BF16]], %[[ANSWER_F32]], %[[ANSWER_F64]], %[[ANSWER_F80]], %[[ANSWER_F128]] : [[I1_TYPE]], [[I4_TYPE]], [[I8_TYPE]], [[I16_TYPE]], [[I32_TYPE]], [[I64_TYPE]], [[F16_TYPE]], [[BF16_TYPE]], [[F32_TYPE]], [[F64_TYPE]], [[F80_TYPE]], [[F128_TYPE]]
    // CHECK-NEXT: }
    func @matrix_reduce_to_scalar_all_types(
         %sparse_tensor_i1: tensor<?x?xi1, #CSR64>,
         %sparse_tensor_i4: tensor<?x?xi4, #CSR64>,
         %sparse_tensor_i8: tensor<?x?xi8, #CSR64>,
         %sparse_tensor_i16: tensor<?x?xi16, #CSR64>,
         %sparse_tensor_i32: tensor<?x?xi32, #CSR64>,
         %sparse_tensor_i64: tensor<?x?xi64, #CSR64>,
         %sparse_tensor_f16: tensor<?x?xf16, #CSR64>,
         %sparse_tensor_bf16: tensor<?x?xbf16, #CSR64>,
         %sparse_tensor_f32: tensor<?x?xf32, #CSR64>,
         %sparse_tensor_f64: tensor<?x?xf64, #CSR64>,
         %sparse_tensor_f80: tensor<?x?xf80, #CSR64>,
         %sparse_tensor_f128: tensor<?x?xf128, #CSR64>
    ) -> (i1, i4, i8, i16, i32, i64, f16, bf16, f32, f64, f80, f128) {
         %answer_i1 = graphblas.matrix_reduce_to_scalar %sparse_tensor_i1 { aggregator = "sum" } : tensor<?x?xi1, #CSR64> to i1
         %answer_i4 = graphblas.matrix_reduce_to_scalar %sparse_tensor_i4 { aggregator = "sum" } : tensor<?x?xi4, #CSR64> to i4
         %answer_i8 = graphblas.matrix_reduce_to_scalar %sparse_tensor_i8 { aggregator = "sum" } : tensor<?x?xi8, #CSR64> to i8
         %answer_i16 = graphblas.matrix_reduce_to_scalar %sparse_tensor_i16 { aggregator = "sum" } : tensor<?x?xi16, #CSR64> to i16
         %answer_i32 = graphblas.matrix_reduce_to_scalar %sparse_tensor_i32 { aggregator = "sum" } : tensor<?x?xi32, #CSR64> to i32
         %answer_i64 = graphblas.matrix_reduce_to_scalar %sparse_tensor_i64 { aggregator = "sum" } : tensor<?x?xi64, #CSR64> to i64
         %answer_f16 = graphblas.matrix_reduce_to_scalar %sparse_tensor_f16 { aggregator = "sum" } : tensor<?x?xf16, #CSR64> to f16
         %answer_bf16 = graphblas.matrix_reduce_to_scalar %sparse_tensor_bf16 { aggregator = "sum" } : tensor<?x?xbf16, #CSR64> to bf16
         %answer_f32 = graphblas.matrix_reduce_to_scalar %sparse_tensor_f32 { aggregator = "sum" } : tensor<?x?xf32, #CSR64> to f32
         %answer_f64 = graphblas.matrix_reduce_to_scalar %sparse_tensor_f64 { aggregator = "sum" } : tensor<?x?xf64, #CSR64> to f64
         %answer_f80 = graphblas.matrix_reduce_to_scalar %sparse_tensor_f80 { aggregator = "sum" } : tensor<?x?xf80, #CSR64> to f80
         %answer_f128 = graphblas.matrix_reduce_to_scalar %sparse_tensor_f128 { aggregator = "sum" } : tensor<?x?xf128, #CSR64> to f128
         return %answer_i1, %answer_i4, %answer_i8, %answer_i16, %answer_i32, %answer_i64, %answer_f16, %answer_bf16, %answer_f32, %answer_f64, %answer_f80, %answer_f128 : i1, i4, i8, i16, i32, i64, f16, bf16, f32, f64, f80, f128
    }
}
