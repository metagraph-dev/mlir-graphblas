from typing import List
from .functions import (
    ConvertLayout,
    MatrixSelect,
    MatrixReduceToScalar,
    Apply,
    MatrixMultiply,
)
from mlir_graphblas.mlir_builder import MLIRVar, MLIRFunctionBuilder
from mlir_graphblas.types import AliasMap, SparseEncodingType, Type
from .sparse_utils import MLIRSparseTensor
from .engine import MlirJitEngine

graphblas_opt_passes = (
    "--graphblas-structuralize",
    "--graphblas-optimize",
    "--graphblas-lower",
    "--sparsification",
    "--sparse-tensor-conversion",
    "--linalg-bufferize",
    "--convert-scf-to-std",
    "--func-bufferize",
    "--tensor-bufferize",
    "--tensor-constant-bufferize",
    "--finalizing-bufferize",
    "--convert-linalg-to-loops",
    "--convert-scf-to-std",
    "--convert-memref-to-llvm",
    "--convert-std-to-llvm",
)

csr_to_csc = ConvertLayout()
matrix_select_triu = MatrixSelect("TRIU")
matrix_select_tril = MatrixSelect("TRIL")
matrix_select_gt = MatrixSelect("gt")
matrix_reduce = MatrixReduceToScalar()
apply_min = Apply("min")
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
        csr64 = SparseEncodingType(["dense", "compressed"], [0, 1], 64, 64)
        csc64 = SparseEncodingType(["dense", "compressed"], [1, 0], 64, 64)
        aliases = AliasMap()
        aliases["CSR64"] = csr64
        aliases["CSC64"] = csc64

        irb = MLIRFunctionBuilder(
            "triangle_count",
            input_types=["tensor<?x?xf64, #CSR64>"],
            return_types=["f64"],
            aliases=aliases,
        )
        (inp,) = irb.inputs
        U = irb.graphblas.matrix_select(inp, [], ["triu"])
        L = irb.graphblas.matrix_select(inp, [], ["tril"])
        U_csc = irb.graphblas.convert_layout(U, "tensor<?x?xf64, #CSC64>")
        C = irb.graphblas.matrix_multiply(L, U_csc, "plus_pair", mask=L)

        reduce_result = irb.graphblas.reduce_to_scalar(C, "plus")
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
    for layer in range(nlayers):
        W_csc = csr_to_csc.compile()(W[layer])
        Y = mxm_plus_times.compile()(Y, W_csc)

        # Normally, I would need to transpose this, but I know these are purely diagonal matrices
        Y = mxm_plus_plus.compile()(Y, Bias[layer])

        Y = matrix_select_gt.compile()(Y, 0.0)
        Y = apply_min.compile()(Y, ymax)

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
        csr64 = SparseEncodingType(["dense", "compressed"], [0, 1], 64, 64)
        csc64 = SparseEncodingType(["dense", "compressed"], [1, 0], 64, 64)
        csx64 = SparseEncodingType(["dense", "compressed"], None, 64, 64)
        aliases = AliasMap()
        aliases["CSR64"] = csr64
        aliases["CSC64"] = csc64
        aliases["CSX64"] = csx64

        ##############################################
        # Define inner function for a single iteration
        ##############################################
        irb_inner = MLIRFunctionBuilder(
            "dnn_step",
            input_types=[
                "tensor<?x?xf64, #CSR64>",
                "tensor<?x?xf64, #CSC64>",
                "tensor<?x?xf64, #CSR64>",
                "f64",
            ],
            return_types=["tensor<?x?xf64, #CSR64>"],
            aliases=aliases,
        )
        weights, biases, Y, threshold = irb_inner.inputs
        zero_f64 = irb_inner.constant(0.0, "f64")
        # Perform inference
        W_csc = irb_inner.graphblas.convert_layout(weights, "tensor<?x?xf64, #CSC64>")
        matmul_result = irb_inner.graphblas.matrix_multiply(Y, W_csc, "plus_times")
        add_bias_result = irb_inner.graphblas.matrix_multiply(
            matmul_result, biases, "plus_plus"
        )
        clamp_result = irb_inner.graphblas.apply(add_bias_result, "min", left=threshold)
        relu_result = irb_inner.graphblas.matrix_select(
            clamp_result, [zero_f64], ["gt"]
        )

        irb_inner.util.del_sparse_tensor(Y)

        irb_inner.return_vars(relu_result)

        ##############################################
        # Build outer loop (overly complicated due to pointer casting necessity
        ##############################################
        irb = MLIRFunctionBuilder(
            "dense_neural_network",
            input_types=[
                "!llvm.ptr<!llvm.ptr<i8>>",  # weight list
                "!llvm.ptr<!llvm.ptr<i8>>",  # bias list
                "index",  # number of layers
                "tensor<?x?xf64, #CSR64>",  # Y init
                "f64",  # clamp theshold
            ],
            return_types=["tensor<?x?xf64, #CSR64>"],
            aliases=aliases,
        )
        weight_list, bias_list, num_layers, Y_init, clamp_threshold = irb.inputs
        c0 = irb.constant(0, "i64")
        c1 = irb.constant(1, "i64")

        # Make a copy of Y_init for consistency of memory cleanup in the loop
        Y_init_dup = irb.graphblas.dup(Y_init)

        Y_init_ptr8 = irb.util.tensor_to_ptr8(Y_init_dup)

        Y_ptr8 = irb.new_var("!llvm.ptr<i8>")
        layer_idx = irb.new_var("i64")

        with irb.for_loop(
            0, num_layers, iter_vars=[(Y_ptr8, Y_init_ptr8), (layer_idx, c0)]
        ) as for_vars:
            # Get weight matrix
            weight_matrix_ptr_ptr = irb.llvm.getelementptr(weight_list, layer_idx)
            weight_matrix_ptr = irb.llvm.load(weight_matrix_ptr_ptr, "!llvm.ptr<i8>")
            weight_matrix = irb.util.ptr8_to_tensor(
                weight_matrix_ptr, "tensor<?x?xf64, #CSR64>"
            )

            # Get bias matrix
            bias_matrix_ptr_ptr = irb.llvm.getelementptr(bias_list, layer_idx)
            bias_matrix_ptr = irb.llvm.load(bias_matrix_ptr_ptr, "!llvm.ptr<i8>")
            bias_matrix_csc = irb.util.ptr8_to_tensor(
                bias_matrix_ptr, "tensor<?x?xf64, #CSC64>"
            )

            # Cast Y from pointer to tensor
            Y_loop = irb.util.ptr8_to_tensor(Y_ptr8, "tensor<?x?xf64, #CSR64>")

            loop_result = irb.call(
                irb_inner, weight_matrix, bias_matrix_csc, Y_loop, clamp_threshold
            )

            # Cast loop_result to a pointer
            result_ptr8 = irb.util.tensor_to_ptr8(loop_result)

            # increment iterator vars
            incremented_layer_index_i64 = irb.addi(layer_idx, c1)
            for_vars.yield_vars(result_ptr8, incremented_layer_index_i64)

        # One final cast from ptr8 to tensor
        Y_final = irb.util.ptr8_to_tensor(
            for_vars.returned_variable[0], "tensor<?x?xf64, #CSR64>"
        )

        irb.return_vars(Y_final)

        # Test Compiled Function
        _dense_neural_network_compiled = irb.compile()

    if len(W) != len(Bias):
        raise TypeError(f"num_layers mismatch: {len(W)} != {len(Bias)}")

    Y = _dense_neural_network_compiled(W, Bias, len(W), Y0, ymax)
    return Y


_sssp = None


def sssp(graph: MLIRSparseTensor, vector: MLIRSparseTensor) -> MLIRSparseTensor:
    global _sssp
    if _sssp is None:
        csr64 = SparseEncodingType(["dense", "compressed"], [0, 1], 64, 64)
        csc64 = SparseEncodingType(["dense", "compressed"], [1, 0], 64, 64)
        cv64 = SparseEncodingType(["compressed"], None, 64, 64)
        aliases = AliasMap()
        aliases["CSR64"] = csr64
        aliases["CSC64"] = csc64
        aliases["CV64"] = cv64

        irb = MLIRFunctionBuilder(
            "sssp",
            input_types=["tensor<?x?xf64, #CSR64>", "tensor<?xf64, #CV64>"],
            return_types=["tensor<?xf64, #CV64>"],
            aliases=aliases,
        )
        (m, v) = irb.inputs
        ctrue = irb.constant(1, "i1")
        cfalse = irb.constant(0, "i1")
        m2 = irb.graphblas.convert_layout(m, "tensor<?x?xf64, #CSC64>")
        w = irb.graphblas.dup(v)

        irb.add_statement(f"scf.while (%continue = {ctrue}): ({ctrue.type}) -> () {{")
        irb.add_statement(f"  scf.condition(%continue)")
        irb.add_statement("} do {")
        irb.add_statement("^bb0:")
        w_old = irb.graphblas.dup(w)
        tmp = irb.graphblas.matrix_multiply(w, m2, "min_plus")
        irb.graphblas.update(tmp, w, "min")
        no_change = irb.graphblas.equal(w, w_old)
        cont = irb.select(no_change, cfalse, ctrue)
        # irb.add_statement(f"call @delSparseVector({w_old}): ({w_old.type}) -> ()")
        irb.add_statement(f"scf.yield {cont}: {cont.type}")
        irb.add_statement("}")

        irb.return_vars(w)

        _sssp = irb.compile()

    w = _sssp(graph, vector)
    return w


_mssp = None


def mssp(graph: MLIRSparseTensor, matrix: MLIRSparseTensor) -> MLIRSparseTensor:
    global _mssp
    if _mssp is None:
        csr64 = SparseEncodingType(["dense", "compressed"], [0, 1], 64, 64)
        csc64 = SparseEncodingType(["dense", "compressed"], [1, 0], 64, 64)
        csx64 = SparseEncodingType(["dense", "compressed"], None, 64, 64)
        cv64 = SparseEncodingType(["compressed"], None, 64, 64)
        aliases = AliasMap()
        aliases["CSR64"] = csr64
        aliases["CSC64"] = csc64
        aliases["CSX64"] = csx64
        aliases["CV64"] = cv64

        irb = MLIRFunctionBuilder(
            "mssp",
            input_types=["tensor<?x?xf64, #CSR64>", "tensor<?x?xf64, #CSR64>"],
            return_types=["tensor<?x?xf64, #CSR64>"],
            aliases=aliases,
        )
        (m, v) = irb.inputs
        ctrue = irb.constant(1, "i1")
        cfalse = irb.constant(0, "i1")
        m2 = irb.graphblas.convert_layout(m, "tensor<?x?xf64, #CSC64>")
        w = irb.graphblas.dup(v)

        irb.add_statement(f"scf.while (%continue = {ctrue}): ({ctrue.type}) -> () {{")
        irb.add_statement(f"  scf.condition(%continue)")
        irb.add_statement("} do {")
        irb.add_statement("^bb0:")
        w_old = irb.graphblas.dup(w)
        tmp = irb.graphblas.matrix_multiply(w, m2, "min_plus")
        irb.graphblas.update(tmp, w, "min")
        no_change = irb.graphblas.equal(w, w_old)
        cont = irb.select(no_change, cfalse, ctrue)
        # wx_old = irb.util.cast_csr_to_csx(w_old)
        # irb.add_statement(f"call @delSparseMatrix({wx_old}): ({wx_old.type}) -> ()")
        irb.add_statement(f"scf.yield {cont}: {cont.type}")
        irb.add_statement("}")

        irb.return_vars(w)

        _mssp = irb.compile()

    w = _mssp(graph, matrix)
    return w


_vertex_nomination = None


def vertex_nomination(
    graph: MLIRSparseTensor, nodes_of_interest: MLIRSparseTensor
) -> int:
    global _vertex_nomination
    if _vertex_nomination is None:
        csr64 = SparseEncodingType(["dense", "compressed"], [0, 1], 64, 64)
        csc64 = SparseEncodingType(["dense", "compressed"], [1, 0], 64, 64)
        csx64 = SparseEncodingType(["dense", "compressed"], None, 64, 64)
        cv64 = SparseEncodingType(["compressed"], None, 64, 64)
        aliases = AliasMap()
        aliases["CSR64"] = csr64
        aliases["CSC64"] = csc64
        aliases["CSX64"] = csx64
        aliases["CV64"] = cv64

        irb = MLIRFunctionBuilder(
            "vertex_nomination",
            input_types=["tensor<?x?xf64, #CSR64>", "tensor<?xf64, #CV64>"],
            return_types=["index"],
            aliases=aliases,
        )
        (m, v) = irb.inputs
        mT = irb.graphblas.transpose(m, "tensor<?x?xf64, #CSR64>")
        v2 = irb.graphblas.matrix_multiply(
            mT, v, semiring="min_first", mask=v, mask_complement=True
        )
        result = irb.graphblas.vector_argmin(v2)

        irb.return_vars(result)

        _vertex_nomination = irb.compile()

    node_of_interest = _vertex_nomination(graph, nodes_of_interest)
    return node_of_interest


_pagerank = None


def pagerank(
    graph: MLIRSparseTensor, damping=0.85, tol=1e-6, *, maxiter=100
) -> MLIRSparseTensor:
    global _pagerank
    if _pagerank is None:
        csr64 = SparseEncodingType(["dense", "compressed"], [0, 1], 64, 64)
        csc64 = SparseEncodingType(["dense", "compressed"], [1, 0], 64, 64)
        csx64 = SparseEncodingType(["dense", "compressed"], None, 64, 64)
        cv64 = SparseEncodingType(["compressed"], None, 64, 64)
        aliases = AliasMap()
        aliases["CSR64"] = csr64
        aliases["CSC64"] = csc64
        aliases["CSX64"] = csx64
        aliases["CV64"] = cv64

        irb = MLIRFunctionBuilder(
            "pagerank",
            input_types=["tensor<?x?xf64, #CSR64>", "f64", "f64", "index"],
            return_types=["tensor<?xf64, #CV64>", "index"],
            aliases=aliases,
        )
        (A, var_damping, var_tol, var_maxiter) = irb.inputs

        nrows = irb.graphblas.num_rows(A)
        nrows_i64 = irb.index_cast(nrows, "i64")
        nrows_f64 = irb.sitofp(nrows_i64, "f64")

        c0 = irb.constant(0, "index")
        c1 = irb.constant(1, "index")
        cf1 = irb.constant(1.0, "f64")
        teleport = irb.subf(cf1, var_damping)
        teleport = irb.divf(teleport, nrows_f64)

        row_degree = irb.graphblas.reduce_to_vector(A, "count", axis=1)
        # prescale row_degree with damping factor, so it isn't done each iteration
        row_degree = irb.graphblas.apply(row_degree, "div", right=var_damping)

        # Use row_degree as a convenient vector to duplicate for starting score
        r = irb.graphblas.dup(row_degree)

        # r = 1/nrows
        nrows_inv = irb.divf(cf1, nrows_f64)
        starting = irb.graphblas.apply(r, "second", right=nrows_inv)
        starting_ptr8 = irb.util.tensor_to_ptr8(starting)

        # Pagerank iterations
        rdiff = irb.new_var("f64")
        prev_score_ptr8 = irb.new_var("!llvm.ptr<i8>")
        iter_count = irb.new_var("index")
        with irb.for_loop(
            0,
            var_maxiter,
            iter_vars=[
                (rdiff, cf1),
                (prev_score_ptr8, starting_ptr8),
                (iter_count, c0),
            ],
        ) as for_vars:
            converged = irb.new_var("i1")
            irb.add_statement(
                f'{converged.assign} = cmpf "olt", {rdiff}, {var_tol} : {rdiff.type}'
            )

            if_block = irb.new_tuple(
                f"{rdiff.type}", f"{prev_score_ptr8.type}", f"{iter_count.type}"
            )
            irb.add_statement(
                f"{if_block.assign} = scf.if {converged} -> ({rdiff.type}, {prev_score_ptr8.type}, {iter_count.type}) {{"
            )

            # Converged
            # ---------
            irb.add_statement(
                f"scf.yield {rdiff}, {prev_score_ptr8}, {iter_count} : {rdiff.type}, {prev_score_ptr8.type}, {iter_count.type}"
            )

            irb.add_statement("} else {")

            # Not converged
            # -------------

            # Cast prev_score from pointer to tensor
            prev_score = irb.util.ptr8_to_tensor(
                prev_score_ptr8, "tensor<?xf64, #CV64>"
            )

            # w = t ./ d
            w = irb.graphblas.intersect(
                prev_score, row_degree, "div", "tensor<?xf64, #CV64>"
            )

            # r = teleport
            # Perform this scalar assignment using an apply hack
            new_score = irb.graphblas.apply(prev_score, "second", right=teleport)

            # r += A'*w
            AT = irb.graphblas.transpose(A, "tensor<?x?xf64, #CSR64>")
            tmp = irb.graphblas.matrix_multiply(AT, w, "plus_second")
            irb.graphblas.update(tmp, new_score, accumulate="plus")

            # rdiff = sum(abs(prev_score - new_score))
            # TODO: this should technically be union, but we don't allow "minus" for union
            #       Replace with apply(neg), then union(plus)
            new_rdiff = irb.graphblas.intersect(
                prev_score, new_score, "minus", "tensor<?xf64, #CV64>"
            )
            new_rdiff = irb.graphblas.apply(new_rdiff, "abs")
            new_rdiff = irb.graphblas.reduce_to_scalar(new_rdiff, "plus")

            # Clean up previous score
            irb.util.del_sparse_tensor(prev_score)

            # Increment iteration count
            new_iter_count = irb.addi(iter_count, c1)

            # Yield
            new_score_ptr8 = irb.util.tensor_to_ptr8(new_score)
            irb.add_statement(
                f"scf.yield {new_rdiff}, {new_score_ptr8}, {new_iter_count} : {new_rdiff.type}, {new_score_ptr8.type}, {new_iter_count.type}"
            )
            irb.add_statement("}")

            # Yield values are: rdiff, score, iter_count
            for_vars.yield_vars(if_block[0], if_block[1], if_block[2])

        ret_val_ptr8 = for_vars.returned_variable[1]
        ret_val = irb.util.ptr8_to_tensor(ret_val_ptr8, "tensor<?xf64, #CV64>")

        # Return values are: score, iter_count
        irb.return_vars(ret_val, for_vars.returned_variable[2])

        _pagerank = irb.compile()

    pr = _pagerank(graph, damping, tol, maxiter)
    return pr
