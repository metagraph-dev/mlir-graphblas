from typing import List
from .functions import (
    Transpose,
    MatrixSelect,
    MatrixReduceToScalar,
    MatrixApply,
    MatrixMultiply,
)
from mlir_graphblas.mlir_builder import MLIRVar, MLIRFunctionBuilder
from .sparse_utils import MLIRSparseTensor
import time


csr_to_csc = Transpose(swap_sizes=False)
matrix_select_triu = MatrixSelect("TRIU")
matrix_select_tril = MatrixSelect("TRIL")
matrix_select_gt0 = MatrixSelect("gt0")
matrix_reduce = MatrixReduceToScalar()
matrix_apply_min = MatrixApply("min")
mxm_plus_pair = MatrixMultiply("plus_pair", mask=True)
mxm_plus_times = MatrixMultiply("plus_times")
mxm_plus_plus = MatrixMultiply("plus_plus")


def triangle_count(A: MLIRSparseTensor) -> int:
    # Create U and L matrices
    U = matrix_select_triu.compile()(A)
    L = matrix_select_tril.compile()(A)
    # Count Triangles
    U_csc = csr_to_csc.compile()(U)
    C = mxm_plus_pair.compile()(L, U_csc, L)
    num_triangles = matrix_reduce.compile()(C)
    assert (
        int(num_triangles) == num_triangles
    ), f"{num_triangles} is unexpectedly not a whole number"
    return int(num_triangles)


_triangle_count_compiled = None


def triangle_count_combined(A: MLIRSparseTensor) -> int:
    global _triangle_count_compiled
    if _triangle_count_compiled is None:
        inp = MLIRVar("A", "tensor<?x?xf64, #CSR64>")

        irb = MLIRFunctionBuilder(
            "triangle_count", input_vars=[inp], return_types=["f64"]
        )
        U = irb.graphblas.matrix_select(inp, "triu")
        L = irb.graphblas.matrix_select(inp, "tril")
        U_csc = irb.graphblas.convert_layout(U, "tensor<?x?xf64, #CSC64>")
        C = irb.graphblas.matrix_multiply(L, U_csc, L, "plus_pair", "tensor<?x?xf64, #CSR64>")

        reduce_result = irb.graphblas.matrix_reduce_to_scalar(C, "sum", "f64")
        irb.return_vars(reduce_result)

        _triangle_count_compiled = irb.compile()

    num_triangles = _triangle_count_compiled(A)
    return int(num_triangles)


def dense_neural_network(
    W: List[MLIRSparseTensor],
    Bias: List[MLIRSparseTensor],
    Y0: MLIRSparseTensor,
    ymax=32.0,
) -> MLIRSparseTensor:
    nlayers = len(W)

    Y = Y0.dup()
    start = now = time.time()
    for layer in range(nlayers):
        W_csc = csr_to_csc.compile()(W[layer])
        Y = mxm_plus_times.compile()(Y, W_csc)

        # Normally, I would need to transpose this, but I know these are purely diagonal matrices
        Y = mxm_plus_plus.compile()(Y, Bias[layer])

        Y = matrix_select_gt0.compile()(Y)
        Y = matrix_apply_min.compile()(Y, ymax)

        curr = time.time()
        diff, now = curr - now, curr
        print(f"Layer {layer+1} of {nlayers} took {diff:.2f} sec")

    print(f"\nTotal time = {(now - start):.2f} sec")
    return Y


_dense_neural_network_compiled = None


def dense_neural_network_combined(
    W: List[MLIRSparseTensor],
    Bias: List[MLIRSparseTensor],
    Y0: MLIRSparseTensor,
    ymax=32.0,
) -> MLIRSparseTensor:
    global _dense_neural_network_compiled
    if _dense_neural_network_compiled is None:
        # Input Vars
        weight_list = MLIRVar("weight_list", "!llvm.ptr<!llvm.ptr<i8>>")
        bias_list = MLIRVar("bias_list", "!llvm.ptr<!llvm.ptr<i8>>")
        num_layers = MLIRVar("num_layers", "index")
        Y_init = MLIRVar("Y0", "tensor<?x?xf64, #CSR64>")
        clamp_threshold = MLIRVar("ymax", "f64")

        # Build Function
        irb = MLIRFunctionBuilder(
            "dense_neural_network",
            input_vars=[weight_list, bias_list, num_layers, Y_init, clamp_threshold],
            return_types=["tensor<?x?xf64, #CSR64>"],
        )
        c0 = irb.constant(0, "i64")
        c1 = irb.constant(1, "i64")

        Y_init_ptr8 = irb.util.tensor_to_ptr8(Y_init)

        Y_ptr8 = MLIRVar("Y_ptr8", "!llvm.ptr<i8>")
        layer_idx = MLIRVar("layer_index", "i64")

        with irb.for_loop(
            0, num_layers, iter_vars=[(Y_ptr8, Y_init_ptr8), (layer_idx, c0)]
        ) as for_vars:
            # Get weight matrix
            weight_matrix_ptr_ptr = irb.llvm.getelementptr(weight_list, layer_idx)
            weight_matrix_ptr = irb.llvm.load(weight_matrix_ptr_ptr, "!llvm.ptr<i8>")
            weight_matrix = irb.util.ptr8_to_tensor(weight_matrix_ptr, "tensor<?x?xf64, #CSR64>")

            # Get bias matrix
            bias_matrix_ptr_ptr = irb.llvm.getelementptr(bias_list, layer_idx)
            bias_matrix_ptr = irb.llvm.load(bias_matrix_ptr_ptr, "!llvm.ptr<i8>")
            bias_matrix = irb.util.ptr8_to_tensor(bias_matrix_ptr, "tensor<?x?xf64, #CSC64>")
            bias_matrix = MLIRVar("bias_matrix", "tensor<?x?xf64, #CSC64>")

            # Cast Y from pointer to tensor
            Y = irb.util.ptr8_to_tensor(Y_ptr8, "tensor<?x?xf64, #CSR64>")

            # Perform inference
            W_csc = irb.graphblas.convert_layout(weight_matrix, "tensor<?x?xf64, #CSC64>")
            matmul_result = irb.graphblas.matrix_multiply(Y, W_csc, None, "plus_times", "tensor<?x?xf64, #CSR64>")
            add_bias_result = irb.graphblas.matrix_multiply(matmul_result, bias_matrix, None, "plus_plus", "tensor<?x?xf64, #CSR64>")
            relu_result = irb.graphblas.matrix_select(add_bias_result, "gt0")
            clamp_result = irb.graphblas.matrix_apply(relu_result, "min", clamp_threshold, "tensor<?x?xf64, #CSR64>")

            # Cast clamp_result to a pointer
            result_ptr8 = irb.util.tensor_to_ptr8(clamp_result)

            # increment iterator vars
            incremented_layer_index_i64 = irb.addi(layer_idx, c1)
            for_vars.yield_vars(result_ptr8, incremented_layer_index_i64)

        # One final cast from ptr8 to tensor
        Y_final = irb.util.ptr8_to_tensor(for_vars.returned_variable[0], "tensor<?x?xf64, #CSR64>")

        irb.return_vars(Y_final)

        # Test Compiled Function
        _dense_neural_network_compiled = irb.compile()

    if len(W) != len(Bias):
        raise TypeError(f"num_layers mismatch: {len(W)} != {len(Bias)}")

    start = time.time()
    Y = _dense_neural_network_compiled(W, Bias, len(W), Y0, ymax)
    print(f"\nTotal time = {(time.time() - start):.2f} sec")
    return Y
