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

        ir_builder = MLIRFunctionBuilder(
            "triangle_count", input_vars=[inp], return_types=["f64"]
        )
        U = ir_builder.call(matrix_select_triu, inp)
        L = ir_builder.call(matrix_select_tril, inp)
        U_csc = ir_builder.call(csr_to_csc, U)
        C = ir_builder.call(mxm_plus_pair, L, U_csc, L)

        reduce_result = ir_builder.call(matrix_reduce, C)
        ir_builder.return_vars(reduce_result)

        _triangle_count_compiled = ir_builder.compile()

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
        Y_init = MLIRVar("Y0", "!llvm.ptr<i8>")
        clamp_threshold = MLIRVar("ymax", "f64")

        # Build Function
        ir_builder = MLIRFunctionBuilder(
            "dense_neural_network",
            input_vars=[weight_list, bias_list, num_layers, Y_init, clamp_threshold],
            return_types=["!llvm.ptr<i8>"],
        )
        ir_builder.add_statement("// pymlir-skip: begin")
        c0_var = ir_builder.constant(0, "i64")
        c1_var = ir_builder.constant(1, "i64")

        Y_var = MLIRVar("Y", "!llvm.ptr<i8>")
        layer_idx = MLIRVar("layer_index", "i64")

        with ir_builder.for_loop(
            0, num_layers, iter_vars=[(Y_var, Y_init), (layer_idx, c0_var)]
        ) as for_vars:
            # Get weight matrix
            ir_builder.add_statement(
                f"%weight_matrix_ptr_ptr = llvm.getelementptr {weight_list.access_string()}[{layer_idx.access_string()}] : (!llvm.ptr<!llvm.ptr<i8>>, i64) -> !llvm.ptr<!llvm.ptr<i8>>"
            )
            weight_matrix_ptr_var = MLIRVar("weight_matrix_ptr", "!llvm.ptr<i8>")
            ir_builder.add_statement(
                f"{weight_matrix_ptr_var.assign_string()} = llvm.load %weight_matrix_ptr_ptr : !llvm.ptr<!llvm.ptr<i8>>"
            )

            # Get bias matrix
            ir_builder.add_statement(
                f"%bias_matrix_ptr_ptr = llvm.getelementptr {bias_list.access_string()}[{layer_idx.access_string()}] : (!llvm.ptr<!llvm.ptr<i8>>, i64) -> !llvm.ptr<!llvm.ptr<i8>>"
            )
            bias_matrix_ptr_var = MLIRVar("bias_matrix_ptr", "!llvm.ptr<i8>")
            ir_builder.add_statement(
                f"{bias_matrix_ptr_var.assign_string()} = llvm.load %bias_matrix_ptr_ptr : !llvm.ptr<!llvm.ptr<i8>>"
            )

            # Perform inference
            W_csc_var = ir_builder.call(csr_to_csc, weight_matrix_ptr_var)
            matmul_result_var = ir_builder.call(mxm_plus_times, Y_var, W_csc_var)
            add_bias_result_var = ir_builder.call(
                mxm_plus_plus, matmul_result_var, bias_matrix_ptr_var
            )
            relu_result_var = ir_builder.call(matrix_select_gt0, add_bias_result_var)
            clamp_result_var = ir_builder.call(
                matrix_apply_min, relu_result_var, clamp_threshold
            )

            # increment iterator vars
            incremented_layer_index_i64_var = MLIRVar("incremented_layer_index", "i64")
            ir_builder.add_statement(
                f"{incremented_layer_index_i64_var.assign_string()} = addi {layer_idx.access_string()}, {c1_var.access_string()} : i64"
            )
            for_vars.yield_vars(clamp_result_var, incremented_layer_index_i64_var)

        ir_builder.add_statement("// pymlir-skip: end")
        ir_builder.return_vars(for_vars.returned_variable[0])

        # Test Compiled Function
        _dense_neural_network_compiled = ir_builder.compile()

    if len(W) != len(Bias):
        raise TypeError(f"num_layers mismatch: {len(W)} != {len(Bias)}")

    start = time.time()
    Y = _dense_neural_network_compiled(W, Bias, len(W), Y0, ymax)
    print(f"\nTotal time = {(time.time() - start):.2f} sec")
    return Y
