import math
import numpy as np
from typing import List, Tuple
from mlir_graphblas.mlir_builder import MLIRFunctionBuilder
from mlir_graphblas.types import AliasMap, SparseEncodingType, AffineMap
from mlir_graphblas.random_utils import ChooseUniformContext, ChooseWeightedContext
from .sparse_utils import MLIRSparseTensor
from . import algo_utils


class Algorithm:
    def __init__(self):
        self.builder = self._build()
        self._cached = None

    def _build(self):
        raise NotImplementedError("must override _build")

    def __call__(self, *args, **kwargs):
        compile_with_passes = kwargs.pop("compile_with_passes", None)
        if compile_with_passes is None:
            if self._cached is None:
                self._cached = self.builder.compile()
            return self._cached(*args, **kwargs)
        else:
            # custom build for testing, is not cached
            func = self.builder.compile(passes=compile_with_passes)
            return func(*args, **kwargs)


def _build_common_aliases():
    csr64 = SparseEncodingType(["dense", "compressed"], [0, 1], 64, 64)
    csc64 = SparseEncodingType(["dense", "compressed"], [1, 0], 64, 64)
    cv64 = SparseEncodingType(["compressed"], None, 64, 64)
    aliases = AliasMap()
    aliases["CSR64"] = csr64
    aliases["CSC64"] = csc64
    aliases["CV64"] = cv64
    aliases["map1d"] = AffineMap("(d0)[s0, s1] -> (d0 * s1 + s0)")
    return aliases


class TriangleCount(Algorithm):
    def _build(self):
        irb = MLIRFunctionBuilder(
            "triangle_count",
            input_types=["tensor<?x?xf64, #CSR64>"],
            return_types=["f64"],
            aliases=_build_common_aliases(),
        )
        (inp,) = irb.inputs
        U = irb.graphblas.select(inp, "triu")
        L = irb.graphblas.select(inp, "tril")
        U_csc = irb.graphblas.convert_layout(U, "tensor<?x?xf64, #CSC64>")
        C = irb.graphblas.matrix_multiply(L, U_csc, "plus_pair", mask=L)

        reduce_result = irb.graphblas.reduce_to_scalar(C, "plus")
        irb.return_vars(reduce_result)

        return irb

    def __call__(self, A: MLIRSparseTensor, **kwargs) -> int:
        return int(super().__call__(A, **kwargs))


triangle_count = TriangleCount()


class DenseNeuralNetwork(Algorithm):
    def _build(self):
        aliases = _build_common_aliases()

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
        zero_f64 = irb_inner.arith.constant(0.0, "f64")
        # Perform inference
        W_csc = irb_inner.graphblas.convert_layout(weights, "tensor<?x?xf64, #CSC64>")
        matmul_result = irb_inner.graphblas.matrix_multiply(Y, W_csc, "plus_times")
        add_bias_result = irb_inner.graphblas.matrix_multiply(
            matmul_result, biases, "plus_plus"
        )
        clamp_result = irb_inner.graphblas.apply(add_bias_result, "min", left=threshold)
        relu_result = irb_inner.graphblas.select(clamp_result, "gt", zero_f64)

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
        c0 = irb.arith.constant(0, "i64")
        c1 = irb.arith.constant(1, "i64")

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
            incremented_layer_index_i64 = irb.arith.addi(layer_idx, c1)
            for_vars.yield_vars(result_ptr8, incremented_layer_index_i64)

        # One final cast from ptr8 to tensor
        Y_final = irb.util.ptr8_to_tensor(
            for_vars.returned_variable[0], "tensor<?x?xf64, #CSR64>"
        )

        irb.return_vars(Y_final)

        return irb

    def __call__(
        self,
        W: List[MLIRSparseTensor],
        Bias: List[MLIRSparseTensor],
        Y0: MLIRSparseTensor,
        ymax=32.0,
        **kwargs,
    ) -> MLIRSparseTensor:

        if len(W) != len(Bias):
            raise TypeError(f"num_layers mismatch: {len(W)} != {len(Bias)}")

        return super().__call__(W, Bias, len(W), Y0, ymax, **kwargs)


dense_neural_network = DenseNeuralNetwork()


class SSSP(Algorithm):
    def _build(self):
        irb = MLIRFunctionBuilder(
            "sssp",
            input_types=["tensor<?x?xf64, #CSR64>", "tensor<?xf64, #CV64>"],
            return_types=["tensor<?xf64, #CV64>"],
            aliases=_build_common_aliases(),
        )
        (m, v) = irb.inputs
        ctrue = irb.arith.constant(1, "i1")
        cfalse = irb.arith.constant(0, "i1")
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

        return irb

    def __call__(
        self, graph: MLIRSparseTensor, vector: MLIRSparseTensor, **kwargs
    ) -> MLIRSparseTensor:
        return super().__call__(graph, vector, **kwargs)


sssp = SSSP()


class MSSP(Algorithm):
    def _build(self):
        irb = MLIRFunctionBuilder(
            "mssp",
            input_types=["tensor<?x?xf64, #CSR64>", "tensor<?x?xf64, #CSR64>"],
            return_types=["tensor<?x?xf64, #CSR64>"],
            aliases=_build_common_aliases(),
        )
        (m, v) = irb.inputs
        ctrue = irb.arith.constant(1, "i1")
        cfalse = irb.arith.constant(0, "i1")
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
        return irb

    def __call__(
        self, graph: MLIRSparseTensor, matrix: MLIRSparseTensor, **kwargs
    ) -> MLIRSparseTensor:
        return super().__call__(graph, matrix, **kwargs)


mssp = MSSP()


class BipartiteProjectAndFilter(Algorithm):
    allowable_keep_nodes = ("row", "column")

    def __init__(self):
        self.builder = {kn: self._build(kn) for kn in self.allowable_keep_nodes}
        self._cached = {}

    def _build(self, keep_nodes):
        # Build Function
        ir_builder = MLIRFunctionBuilder(
            "left_project_and_filter",
            input_types=["tensor<?x?xf64, #CSR64>", "f64"],
            return_types=["tensor<?x?xf64, #CSR64>"],
            aliases=_build_common_aliases(),
        )
        (M, limit) = ir_builder.inputs
        if keep_nodes == "row":
            M_T = ir_builder.graphblas.transpose(M, "tensor<?x?xf64, #CSC64>")
            projection = ir_builder.graphblas.matrix_multiply(M, M_T, "plus_times")
        else:
            M_T = ir_builder.graphblas.transpose(M, "tensor<?x?xf64, #CSR64>")
            M_csc = ir_builder.graphblas.convert_layout(M, "tensor<?x?xf64, #CSC64>")
            projection = ir_builder.graphblas.matrix_multiply(M_T, M_csc, "plus_times")
        filtered = ir_builder.graphblas.select(projection, "ge", limit)
        ir_builder.return_vars(filtered)
        return ir_builder

    def __call__(
        self, graph: MLIRSparseTensor, keep_nodes: str = "row", cutoff=0.0, **kwargs
    ) -> MLIRSparseTensor:
        if keep_nodes not in self.allowable_keep_nodes:
            raise ValueError(
                f"Invalid keep_nodes argument: {keep_nodes}, must be one of {self.allowable_keep_nodes}"
            )

        compile_with_passes = kwargs.pop("compile_with_passes", None)
        if compile_with_passes is None:
            if keep_nodes not in self._cached:
                self._cached[keep_nodes] = self.builder[keep_nodes].compile()

            return self._cached[keep_nodes](graph, cutoff)
        else:
            func = self.builder[keep_nodes].compile(passes=compile_with_passes)
            return func(graph, cutoff, **kwargs)


bipartite_project_and_filter = BipartiteProjectAndFilter()


class VertexNomination(Algorithm):
    def _build(self):
        irb = MLIRFunctionBuilder(
            "vertex_nomination",
            input_types=["tensor<?x?xf64, #CSR64>", "tensor<?xf64, #CV64>"],
            return_types=["index"],
            aliases=_build_common_aliases(),
        )
        (m, v) = irb.inputs
        mT = irb.graphblas.transpose(m, "tensor<?x?xf64, #CSR64>")
        v2 = irb.graphblas.matrix_multiply(
            mT, v, semiring="min_first", mask=v, mask_complement=True
        )
        result_64 = irb.graphblas.reduce_to_scalar(v2, "argmin")
        result = irb.arith.index_cast(result_64, "index")

        irb.return_vars(result)

        return irb

    def __call__(
        self, graph: MLIRSparseTensor, nodes_of_interest: MLIRSparseTensor, **kwargs
    ) -> int:
        return super().__call__(graph, nodes_of_interest, **kwargs)


vertex_nomination = VertexNomination()


class ScanStatistics(Algorithm):
    def _build(self):
        ir_builder = MLIRFunctionBuilder(
            "scan_statistics",
            input_types=["tensor<?x?xf64, #CSR64>"],
            return_types=["index"],
            aliases=_build_common_aliases(),
        )
        (A,) = ir_builder.inputs
        L = ir_builder.graphblas.select(A, "tril")
        L_T = ir_builder.graphblas.transpose(L, "tensor<?x?xf64, #CSC64>")
        A_triangles = ir_builder.graphblas.matrix_multiply(A, L_T, "plus_pair", mask=A)
        tri = ir_builder.graphblas.reduce_to_vector(A_triangles, "plus", 1)
        answer_64 = ir_builder.graphblas.reduce_to_scalar(tri, "argmax")
        answer = ir_builder.arith.index_cast(answer_64, "index")
        ir_builder.return_vars(answer)
        return ir_builder

    def __call__(self, graph: MLIRSparseTensor, **kwargs) -> int:
        return super().__call__(graph, **kwargs)


scan_statistics = ScanStatistics()


class Pagerank(Algorithm):
    def _build(self):
        irb = MLIRFunctionBuilder(
            "pagerank",
            input_types=["tensor<?x?xf64, #CSR64>", "f64", "f64", "index"],
            return_types=["tensor<?xf64, #CV64>", "index"],
            aliases=_build_common_aliases(),
        )
        (A, var_damping, var_tol, var_maxiter) = irb.inputs

        nrows = irb.graphblas.num_rows(A)
        nrows_i64 = irb.arith.index_cast(nrows, "i64")
        nrows_f64 = irb.arith.sitofp(nrows_i64, "f64")

        c0 = irb.arith.constant(0, "index")
        c1 = irb.arith.constant(1, "index")
        cf1 = irb.arith.constant(1.0, "f64")
        teleport = irb.arith.subf(cf1, var_damping)
        teleport = irb.arith.divf(teleport, nrows_f64)

        row_degree = irb.graphblas.reduce_to_vector(A, "count", axis=1)
        row_degree = irb.graphblas.cast(row_degree, "tensor<?xf64, #CV64>")
        # prescale row_degree with damping factor, so it isn't done each iteration
        row_degree = irb.graphblas.apply(row_degree, "div", right=var_damping)

        # Use row_degree as a convenient vector to duplicate for starting score
        r = irb.graphblas.dup(row_degree)

        # r = 1/nrows
        nrows_inv = irb.arith.divf(cf1, nrows_f64)
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
            converged = irb.arith.cmpf(rdiff, var_tol, "olt")

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
            w = irb.graphblas.intersect(prev_score, row_degree, "div")

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
            new_rdiff = irb.graphblas.intersect(prev_score, new_score, "minus")
            new_rdiff = irb.graphblas.apply(new_rdiff, "abs")
            new_rdiff = irb.graphblas.reduce_to_scalar(new_rdiff, "plus")

            # Note: this is commented out due to https://github.com/metagraph-dev/mlir-graphblas/issues/163
            # Clean up previous score
            # irb.util.del_sparse_tensor(prev_score)

            # Increment iteration count
            new_iter_count = irb.arith.addi(iter_count, c1)

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

        return irb

    def __call__(
        self, graph: MLIRSparseTensor, damping=0.85, tol=1e-6, *, maxiter=100, **kwargs
    ) -> MLIRSparseTensor:
        return super().__call__(graph, damping, tol, maxiter, **kwargs)


pagerank = Pagerank()


class GraphSearch(Algorithm):
    allowable_methods = ("random", "random_weighted", "argmin", "argmax")

    def __init__(self):
        self.builder = {
            method: self._build(method) for method in self.allowable_methods
        }
        self._cached = {}

    def _build(self, method):
        input_types = [
            "tensor<?x?xf64, #CSR64>",
            "index",
            "tensor<?xi64>",
        ]
        if method[:6] == "random":
            input_types.append("!llvm.ptr<i8>")
        irb = MLIRFunctionBuilder(
            "graph_search",
            input_types=input_types,
            return_types=["tensor<?xf64, #CV64>"],
            aliases=_build_common_aliases(),
        )
        if method[:6] == "random":
            (A, nsteps, seeds, ctx) = irb.inputs
        else:
            (A, nsteps, seeds) = irb.inputs

        c0 = irb.arith.constant(0, "index")
        c1 = irb.arith.constant(1, "index")
        ci1 = irb.arith.constant(1, "i64")
        cf1 = irb.arith.constant(1.0, "f64")

        # Create count vector
        ncols = irb.graphblas.num_cols(A)
        count = irb.util.new_sparse_tensor("tensor<?xf64, #CV64>", ncols)

        # Create B matrix, sized (nseeds x ncols) with nnz=nseeds
        nseeds = irb.tensor.dim(seeds, c0)
        nseeds_64 = irb.arith.index_cast(nseeds, "i64")
        B = irb.util.new_sparse_tensor("tensor<?x?xf64, #CSR64>", nseeds, ncols)
        Bptr8 = irb.util.tensor_to_ptr8(B)
        irb.util.resize_sparse_index(Bptr8, c1, nseeds)
        irb.util.resize_sparse_values(Bptr8, nseeds)

        Bp = irb.sparse_tensor.pointers(B, c1)
        Bi = irb.sparse_tensor.indices(B, c1)
        Bx = irb.sparse_tensor.values(B)

        # Populate B matrix based on initial seed
        with irb.for_loop(0, nseeds) as for_vars:
            seed_num = for_vars.iter_var_index
            seed_num_64 = irb.arith.index_cast(seed_num, "i64")
            irb.memref.store(seed_num_64, Bp, seed_num)
            cur_node = irb.tensor.extract(seeds, seed_num)
            irb.memref.store(cur_node, Bi, seed_num)
            irb.memref.store(cf1, Bx, seed_num)
        irb.memref.store(nseeds_64, Bp, nseeds)

        # Convert layout of A
        A_csc = irb.graphblas.convert_layout(A, "tensor<?x?xf64, #CSC64>")

        # Perform graph search nsteps times
        with irb.for_loop(0, nsteps) as for_vars:
            # Compute neighbors of current nodes
            available_neighbors = irb.graphblas.matrix_multiply(
                B, A_csc, semiring="min_second"
            )
            # Select new neighbors
            if method[:6] == "random":
                rand_method = (
                    "choose_weighted"
                    if method == "random_weighted"
                    else "choose_uniform"
                )
                chosen_neighbors = irb.graphblas.matrix_select_random(
                    available_neighbors, ci1, ctx, rand_method
                )
                chosen_neighbors_idx = irb.sparse_tensor.indices(chosen_neighbors, c1)
            elif method[:3] == "arg":
                chosen_neighbors = irb.graphblas.reduce_to_vector(
                    available_neighbors, method, axis=1
                )
                chosen_neighbors_idx = irb.sparse_tensor.values(chosen_neighbors)
            # TODO: update this like RandomWalk, replacing B rather than updating
            # TODO: this will require converting the "arg" vector results into a matrix
            # Update B inplace with new neighbors
            with irb.for_loop(0, nseeds) as inner_for:
                inner_seed_num = inner_for.iter_var_index
                inner_cur_node = irb.memref.load(chosen_neighbors_idx, inner_seed_num)
                irb.memref.store(inner_cur_node, Bi, inner_seed_num)
            # Add new nodes to count
            small_count = irb.graphblas.reduce_to_vector(B, aggregator="plus", axis=0)
            irb.graphblas.update(small_count, count, accumulate="plus")

        irb.return_vars(count)

        return irb

    def __call__(
        self,
        graph: MLIRSparseTensor,
        num_steps,
        initial_seed_array,
        method="random",
        *,
        rand_seed=None,
        **kwargs,
    ) -> MLIRSparseTensor:
        if method not in self.allowable_methods:
            raise ValueError(
                f"Invalid method: {method}, must be one of {self.allowable_methods}"
            )

        if method not in self._cached:
            self._cached[method] = self.builder[method].compile()

        if not isinstance(initial_seed_array, np.ndarray):
            initial_seed_array = np.array(initial_seed_array, dtype=np.int64)

        extra = []
        if method == "random":
            extra.append(ChooseUniformContext(rand_seed))
        elif method == "random_weighted":
            extra.append(ChooseWeightedContext(rand_seed))

        return self._cached[method](
            graph, num_steps, initial_seed_array, *extra, **kwargs
        )


graph_search = GraphSearch()


class RandomWalk(Algorithm):
    def _build(self):
        irb = MLIRFunctionBuilder(
            "graph_search",
            input_types=[
                "tensor<?x?xf64, #CSR64>",
                "index",
                "tensor<?xi64>",
                "!llvm.ptr<i8>",
            ],
            return_types=["tensor<?x?xi64, #CSR64>"],
            aliases=_build_common_aliases(),
        )
        (A, nsteps, walkers, ctx) = irb.inputs

        c0 = irb.arith.constant(0, "index")
        c1 = irb.arith.constant(1, "index")
        ci1 = irb.arith.constant(1, "i64")
        cf1 = irb.arith.constant(1.0, "f64")

        # Create paths matrix, size (nwalkers x nsteps+1) with nnz=nwalkers
        # Size the indices and values for a fully dense matrix. Trim later if early terminations occur.
        nsteps_plus_1 = irb.arith.addi(nsteps, c1)
        nwalkers = irb.tensor.dim(walkers, c0)
        P = irb.util.new_sparse_tensor(
            "tensor<?x?xi64, #CSC64>", nwalkers, nsteps_plus_1
        )
        Pptr8 = irb.util.tensor_to_ptr8(P)
        max_possible_size = irb.arith.muli(nwalkers, nsteps_plus_1)
        irb.util.resize_sparse_index(Pptr8, c1, max_possible_size)
        irb.util.resize_sparse_values(Pptr8, max_possible_size)

        Pp = irb.sparse_tensor.pointers(P, c1)
        Pi = irb.sparse_tensor.indices(P, c1)
        Px = irb.sparse_tensor.values(P)

        # Create Frontier matrix, sized (nwalkers x ncols) with nnz=nwalkers
        ncols = irb.graphblas.num_cols(A)
        F = irb.util.new_sparse_tensor("tensor<?x?xf64, #CSR64>", nwalkers, ncols)
        Fptr8 = irb.util.tensor_to_ptr8(F)
        irb.util.resize_sparse_index(Fptr8, c1, nwalkers)
        irb.util.resize_sparse_values(Fptr8, nwalkers)

        Fp = irb.sparse_tensor.pointers(F, c1)
        Fi = irb.sparse_tensor.indices(F, c1)
        Fx = irb.sparse_tensor.values(F)

        # Populate F and P matrices based on initial seed
        with irb.for_loop(0, nwalkers) as for_vars:
            seed_num = for_vars.iter_var_index
            seed_num_64 = irb.arith.index_cast(seed_num, "i64")
            irb.memref.store(seed_num_64, Fp, seed_num)
            cur_node = irb.tensor.extract(walkers, seed_num)
            irb.memref.store(cur_node, Fi, seed_num)
            irb.memref.store(cf1, Fx, seed_num)
            irb.memref.store(seed_num_64, Pi, seed_num)
            irb.memref.store(cur_node, Px, seed_num)
        nwalkers_64 = irb.arith.index_cast(nwalkers, "i64")
        irb.memref.store(nwalkers_64, Fp, nwalkers)
        irb.memref.store(nwalkers_64, Pp, c1)

        # Convert layout of A
        A_csc = irb.graphblas.convert_layout(A, "tensor<?x?xf64, #CSC64>")

        # Perform graph search nsteps times
        frontier_ptr8 = irb.new_var("!llvm.ptr<i8>")
        with irb.for_loop(
            1, nsteps_plus_1, iter_vars=[(frontier_ptr8, Fptr8)]
        ) as for_vars:
            step_num = for_vars.iter_var_index
            step_num_plus_1 = irb.arith.addi(step_num, c1)
            frontier = irb.util.ptr8_to_tensor(frontier_ptr8, "tensor<?x?xf64, #CSR64>")

            # Compute neighbors of current nodes
            available_neighbors = irb.graphblas.matrix_multiply(
                frontier, A_csc, semiring="min_second"
            )
            # Select new neighbors
            chosen_neighbors = irb.graphblas.matrix_select_random(
                available_neighbors, ci1, ctx, "choose_uniform"
            )
            chosen_neighbors_ptr8 = irb.util.tensor_to_ptr8(chosen_neighbors)
            # Add new nodes to paths
            new_indices = irb.graphblas.reduce_to_vector(
                chosen_neighbors, aggregator="argmin", axis=1
            )
            NIp = irb.sparse_tensor.pointers(new_indices, c0)
            NIi = irb.sparse_tensor.indices(new_indices, c0)
            NIx = irb.sparse_tensor.values(new_indices)
            num_new_indices_64 = irb.memref.load(NIp, c1)
            num_new_indices = irb.arith.index_cast(num_new_indices_64, "index")
            cur_nnz_64 = irb.memref.load(Pp, step_num)
            cur_nnz = irb.arith.index_cast(cur_nnz_64, "index")
            new_nnz = irb.arith.addi(cur_nnz, num_new_indices)
            with irb.for_loop(0, num_new_indices) as ni_for:
                ni_pos = ni_for.iter_var_index
                ni_offset_pos = irb.arith.addi(cur_nnz, ni_pos)
                ni_index = irb.memref.load(NIi, ni_pos)
                ni_value = irb.memref.load(NIx, ni_pos)
                irb.memref.store(ni_index, Pi, ni_offset_pos)
                irb.memref.store(ni_value, Px, ni_offset_pos)
            new_nnz_64 = irb.arith.index_cast(new_nnz, "i64")
            irb.memref.store(new_nnz_64, Pp, step_num_plus_1)
            for_vars.yield_vars(chosen_neighbors_ptr8)

        # Trim indices and values
        final_nnz_64 = irb.memref.load(Pp, nsteps_plus_1)
        final_nnz = irb.arith.index_cast(final_nnz_64, "index")
        irb.util.resize_sparse_index(Pptr8, c1, final_nnz)
        irb.util.resize_sparse_values(Pptr8, final_nnz)

        # Convert from CSC to CSR
        output = irb.sparse_tensor.convert(P, "tensor<?x?xi64, #CSR64>")

        irb.return_vars(output)

        return irb

    def __call__(
        self,
        graph: MLIRSparseTensor,
        num_steps,
        initial_walkers,
        *,
        rand_seed=None,
        **kwargs,
    ) -> MLIRSparseTensor:
        ctx = ChooseUniformContext(rand_seed)
        if not isinstance(initial_walkers, np.ndarray):
            initial_walkers = np.array(initial_walkers, dtype=np.int64)
        return super().__call__(graph, num_steps, initial_walkers, ctx, **kwargs)


random_walk = RandomWalk()


class BFS(Algorithm):
    def _build(self):
        irb = MLIRFunctionBuilder(
            "bfs",
            input_types=["index", "tensor<?x?xf64, #CSR64>"],
            return_types=[
                "tensor<?xf64, #CV64>",
                "tensor<?xf64, #CV64>",
            ],
            aliases=_build_common_aliases(),
        )

        (source, A) = irb.inputs

        c0 = irb.arith.constant(0, "index")
        c1 = irb.arith.constant(1, "index")
        c1_i64 = irb.arith.constant(1, "i64")
        c0_f64 = irb.arith.constant(0, "f64")

        num_rows = irb.graphblas.num_rows(A)
        source_i64 = irb.arith.index_cast(source, "i64")
        source_f64 = irb.arith.sitofp(source_i64, "f64")

        # Initialize the frontier
        frontier = irb.util.new_sparse_tensor("tensor<?xf64, #CV64>", num_rows)
        frontier_ptr8 = irb.util.tensor_to_ptr8(frontier)
        irb.util.resize_sparse_index(frontier_ptr8, c0, c1)
        irb.util.resize_sparse_values(frontier_ptr8, c1)
        # i.e. frontier[source] = source
        frontier_pointers = irb.sparse_tensor.pointers(frontier, c0)
        frontier_indices = irb.sparse_tensor.indices(frontier, c0)
        frontier_values = irb.sparse_tensor.values(frontier)
        irb.memref.store(c1_i64, frontier_pointers, c1)
        irb.memref.store(source_i64, frontier_indices, c0)
        irb.memref.store(source_f64, frontier_values, c0)

        # Initialize the parents
        parents = irb.util.new_sparse_tensor("tensor<?xf64, #CV64>", num_rows)
        parents_ptr8 = irb.util.tensor_to_ptr8(parents)
        irb.util.resize_sparse_index(parents_ptr8, c0, c1)
        irb.util.resize_sparse_values(parents_ptr8, c1)
        # i.e. parents[source] = source
        parents_pointers = irb.sparse_tensor.pointers(parents, c0)
        parents_indices = irb.sparse_tensor.indices(parents, c0)
        parents_values = irb.sparse_tensor.values(parents)
        irb.memref.store(c1_i64, parents_pointers, c1)
        irb.memref.store(source_i64, parents_indices, c0)
        irb.memref.store(source_f64, parents_values, c0)

        # Initialize the levels
        levels = irb.util.new_sparse_tensor("tensor<?xf64, #CV64>", num_rows)
        levels_ptr8 = irb.util.tensor_to_ptr8(levels)
        irb.util.resize_sparse_index(levels_ptr8, c0, c1)
        irb.util.resize_sparse_values(levels_ptr8, c1)
        # i.e. levels[source] = -1
        levels_pointers = irb.sparse_tensor.pointers(levels, c0)
        levels_indices = irb.sparse_tensor.indices(levels, c0)
        levels_values = irb.sparse_tensor.values(levels)
        irb.memref.store(c1_i64, levels_pointers, c1)
        irb.memref.store(source_i64, levels_indices, c0)
        irb.memref.store(c0_f64, levels_values, c0)

        with irb.while_loop(c0, frontier_ptr8) as while_loop:
            with while_loop.before as before_region:
                level = before_region.arg_vars[0]
                current_frontier_ptr8 = before_region.arg_vars[1]
                current_frontier = irb.util.ptr8_to_tensor(
                    current_frontier_ptr8, "tensor<?xf64, #CV64>"
                )
                next_frontier = irb.graphblas.matrix_multiply(
                    A,
                    current_frontier,
                    "any_overlapi",
                    mask=parents,
                    mask_complement=True,
                )
                irb.graphblas.update(next_frontier, parents, "plus")
                next_frontier_ptr8 = irb.util.tensor_to_ptr8(next_frontier)
                next_frontier_size = irb.graphblas.num_vals(next_frontier)
                condition = irb.arith.cmpi(next_frontier_size, c0, "ne")
                before_region.condition(condition, level, next_frontier_ptr8)
            with while_loop.after as after_region:
                level = after_region.arg_vars[0]
                next_level = irb.arith.addi(level, c1)
                next_frontier_ptr8 = after_region.arg_vars[1]

                # update levels
                next_frontier = irb.util.ptr8_to_tensor(
                    next_frontier_ptr8, "tensor<?xf64, #CV64>"
                )
                next_level_i64 = irb.arith.index_cast(next_level, "i64")
                next_level_f64 = irb.arith.sitofp(next_level_i64, "f64")
                next_frontier_levels = irb.graphblas.apply(
                    next_frontier, "second", right=next_level_f64
                )
                irb.graphblas.update(next_frontier_levels, levels, "max")

                after_region.yield_vars(next_level, next_frontier_ptr8)

        irb.return_vars(parents, levels)

        return irb

    def __call__(
        self, source: int, A: MLIRSparseTensor, **kwargs
    ) -> Tuple[MLIRSparseTensor, MLIRSparseTensor]:
        return super().__call__(source, A, **kwargs)


bfs = BFS()


class TotallyInducedEdgeSampling(Algorithm):
    def _build(self):
        irb = MLIRFunctionBuilder(
            "totally_induced_edge_sampling",
            input_types=[
                "tensor<?x?xf64, #CSR64>",
                "f64",
                "!llvm.ptr<i8>",
            ],
            return_types=["tensor<?x?xf64, #CSR64>"],
            aliases=_build_common_aliases(),
        )
        (A, percentile, ctx) = irb.inputs

        # TODO: figure out a reasonable way to raise an error if nrows != ncols

        selected_edges = irb.graphblas.select(
            A, "probability", percentile, rng_context=ctx
        )
        row_counts = irb.graphblas.reduce_to_vector(selected_edges, "count", axis=1)
        col_counts = irb.graphblas.reduce_to_vector(selected_edges, "count", axis=0)
        selected_nodes = irb.graphblas.union(row_counts, col_counts, "plus")
        selected_nodes = irb.graphblas.cast(selected_nodes, "tensor<?xf64, #CV64>")
        # TODO: these next lines should be replaced with `extract` when available
        D_csr = irb.graphblas.diag(selected_nodes, "tensor<?x?xf64, #CSR64>")
        D_csc = irb.graphblas.diag(selected_nodes, "tensor<?x?xf64, #CSC64>")
        tmp_csr = irb.graphblas.matrix_multiply(A, D_csc, "any_first")
        tmp_csc = irb.graphblas.convert_layout(tmp_csr, "tensor<?x?xf64, #CSC64>")
        output = irb.graphblas.matrix_multiply(D_csr, tmp_csc, "any_second")

        irb.return_vars(output)

        return irb

    def __call__(
        self,
        graph: MLIRSparseTensor,
        sampling_percentage: float,
        *,
        rand_seed=None,
        **kwargs,
    ) -> MLIRSparseTensor:
        ctx = ChooseUniformContext(rand_seed)
        if not 0 <= sampling_percentage <= 1:
            raise ValueError("sampling_percentage must be between 0 and 1")
        return super().__call__(graph, sampling_percentage, ctx, **kwargs)


ties = totally_induced_edge_sampling = TotallyInducedEdgeSampling()


class GraphSAGE(Algorithm):
    def _build(self):
        irb = MLIRFunctionBuilder(
            "graph_sage",
            input_types=[
                "tensor<?x?xf64, #CSR64>",
                "tensor<?x?xf64, #CSR64>",
                "!llvm.ptr<!llvm.ptr<i8>>",
                "index",
                "!llvm.ptr<i64>",
                "!llvm.ptr<i8>",
            ],
            return_types=[
                "tensor<?x?xf64, #CSR64>",
            ],
            aliases=_build_common_aliases(),
        )

        (
            A,
            features_csr,
            weight_matrices,
            num_weight_matrices,  # equal to len(sample_count_per_layer)
            sample_count_per_layer,
            rng_context,
        ) = irb.inputs

        c0 = irb.arith.constant(0, "index")
        c1 = irb.arith.constant(1, "index")
        c0_f64 = irb.arith.constant(0, "f64")
        c_0_5_f64 = irb.arith.constant(0.5, "f64")
        c1_f64 = irb.arith.constant(1, "f64")
        c2_f64 = irb.arith.constant(2, "f64")
        c1_i64 = irb.arith.constant(1, "i64")

        num_nodes = irb.graphblas.num_rows(A)
        num_nodes_i64 = irb.arith.index_cast(num_nodes, "i64")

        concat_matrix = irb.util.new_sparse_tensor(
            "tensor<?x?xf64, #CSR64>", num_nodes, c1
        )
        concat_matrix_ptr8 = irb.util.tensor_to_ptr8(concat_matrix)

        # TODO move this to an inner loop for parallelization
        # TODO fill this up early on and reuse them if non-paralllel (make sure to delete them later)
        concat_matrix_vectors = irb.llvm.alloca(num_nodes_i64, "!llvm.ptr<i8>")

        # TODO move this to an inner loop for parallelization
        row_selector = irb.util.new_sparse_tensor("tensor<?xf64, #CV64>", num_nodes)
        row_selector_ptr8 = irb.util.tensor_to_ptr8(row_selector)
        irb.util.resize_sparse_index(row_selector_ptr8, c0, c1)
        irb.util.resize_sparse_values(row_selector_ptr8, c1)
        row_selector_pointers = irb.sparse_tensor.pointers(row_selector, c0)
        irb.memref.store(c1_i64, row_selector_pointers, c1)
        row_selector_indices = irb.sparse_tensor.indices(row_selector, c0)
        row_selector_values = irb.sparse_tensor.values(row_selector)

        features = irb.graphblas.convert_layout(features_csr, "tensor<?x?xf64, #CSC64>")
        features_ptr8 = irb.util.tensor_to_ptr8(features)

        h_ptr8 = irb.new_var("!llvm.ptr<i8>")
        with irb.for_loop(
            0, num_weight_matrices, iter_vars=[(h_ptr8, features_ptr8)]
        ) as k_loop_for_vars:
            layer_idx = k_loop_for_vars.iter_var_index
            layer_idx_i64 = irb.arith.index_cast(layer_idx, "i64")
            num_samples_ptr = irb.llvm.getelementptr(
                sample_count_per_layer, layer_idx_i64
            )
            num_samples = irb.llvm.load(num_samples_ptr, "i64")

            neighborhoods_csr = irb.graphblas.matrix_select_random(
                A, num_samples, rng_context, choose_n="choose_uniform"
            )
            neighborhoods = irb.graphblas.convert_layout(
                neighborhoods_csr, "tensor<?x?xf64, #CSC64>"
            )

            h = irb.util.ptr8_to_tensor(h_ptr8, "tensor<?x?xf64, #CSC64>")

            node_h_size = irb.graphblas.num_cols(h)
            neighbor_mean_size = node_h_size
            concat_size = irb.arith.addi(node_h_size, neighbor_mean_size)

            weight_matrix_ptr_ptr = irb.llvm.getelementptr(
                weight_matrices, layer_idx_i64
            )
            weight_matrix_ptr = irb.llvm.load(weight_matrix_ptr_ptr, "!llvm.ptr<i8>")
            weight_matrix_csr = irb.util.ptr8_to_tensor(
                weight_matrix_ptr, "tensor<?x?xf64, #CSR64>"
            )
            weight_matrix = irb.graphblas.convert_layout(
                weight_matrix_csr, "tensor<?x?xf64, #CSC64>"
            )

            weight_matrix_ncols = irb.graphblas.num_cols(weight_matrix)
            with irb.for_loop(0, num_nodes) as nodes_loop_for_vars:
                v = nodes_loop_for_vars.iter_var_index
                v_i64 = irb.arith.index_cast(v, "i64")
                irb.memref.store(v_i64, row_selector_indices, c0)
                irb.memref.store(c1_f64, row_selector_values, c0)

                neighborhood_vec = irb.graphblas.matrix_multiply(
                    row_selector, neighborhoods, "any_second"
                )
                neighborhood_mat = irb.graphblas.diag(
                    neighborhood_vec, "tensor<?x?xf64, #CSR64>"
                )

                neighborhood_hs = irb.graphblas.matrix_multiply(
                    neighborhood_mat, h, "any_second"
                )

                neighbor_sum = irb.graphblas.reduce_to_vector(
                    neighborhood_hs, "plus", 0
                )
                actual_num_samples = irb.graphblas.num_vals(neighborhood_vec)
                actual_num_samples_i64 = irb.arith.index_cast(actual_num_samples, "i64")
                actual_num_samples_f64 = irb.arith.sitofp(actual_num_samples_i64, "f64")
                neighbor_mean = irb.graphblas.apply(
                    neighbor_sum, "div", right=actual_num_samples_f64
                )

                node_h = irb.graphblas.matrix_multiply(row_selector, h, "any_second")

                concat = irb.util.new_sparse_tensor("tensor<?xf64, #CV64>", concat_size)

                node_h_num_vals = irb.graphblas.num_vals(node_h)
                neighbor_mean_num_vals = irb.graphblas.num_vals(neighbor_mean)
                concat_num_vals = irb.arith.addi(
                    node_h_num_vals, neighbor_mean_num_vals
                )

                concat_ptr8 = irb.util.tensor_to_ptr8(concat)
                irb.util.resize_sparse_index(concat_ptr8, c0, concat_num_vals)
                irb.util.resize_sparse_values(concat_ptr8, concat_num_vals)

                concat_pointers = irb.sparse_tensor.pointers(concat, c0)
                concat_num_vals_i64 = irb.arith.index_cast(concat_num_vals, "i64")
                irb.memref.store(concat_num_vals_i64, concat_pointers, c1)

                concat_indices = irb.sparse_tensor.indices(concat, c0)
                concat_values = irb.sparse_tensor.values(concat)

                # TODO resize one of these vectors and make that the final concat vector
                node_h_indices = irb.sparse_tensor.indices(node_h, c0)
                node_h_values = irb.sparse_tensor.values(node_h)
                with irb.for_loop(c0, node_h_num_vals) as node_h_to_concat_for_vars:
                    node_h_position = node_h_to_concat_for_vars.iter_var_index

                    node_h_indices_value = irb.memref.load(
                        node_h_indices, node_h_position
                    )
                    irb.memref.store(
                        node_h_indices_value, concat_indices, node_h_position
                    )

                    node_h_values_value = irb.memref.load(
                        node_h_values, node_h_position
                    )
                    irb.memref.store(
                        node_h_values_value, concat_values, node_h_position
                    )

                neighbor_mean_indices = irb.sparse_tensor.indices(neighbor_mean, c0)
                neighbor_mean_values = irb.sparse_tensor.values(neighbor_mean)
                node_h_size_i64 = irb.arith.index_cast(node_h_size, "i64")
                with irb.for_loop(
                    c0, neighbor_mean_num_vals
                ) as neighbor_mean_to_concat_for_vars:
                    neighbor_mean_position = (
                        neighbor_mean_to_concat_for_vars.iter_var_index
                    )
                    concat_position = irb.arith.addi(
                        neighbor_mean_position, node_h_num_vals
                    )

                    neighbor_mean_indices_value = irb.memref.load(
                        neighbor_mean_indices, neighbor_mean_position
                    )
                    concat_indices_value = irb.arith.addi(
                        node_h_size_i64, neighbor_mean_indices_value
                    )
                    irb.memref.store(
                        concat_indices_value, concat_indices, concat_position
                    )

                    neighbor_mean_values_value = irb.memref.load(
                        neighbor_mean_values, neighbor_mean_position
                    )
                    irb.memref.store(
                        neighbor_mean_values_value, concat_values, concat_position
                    )

                concat_pre_relu = irb.graphblas.matrix_multiply(
                    concat, weight_matrix, semiring="plus_times"
                )
                concat_post_relu = irb.graphblas.select(concat_pre_relu, "gt", c0_f64)

                concat_post_relu_squared = irb.graphblas.apply(
                    concat_post_relu, "pow", right=c2_f64
                )
                squared_sum = irb.graphblas.reduce_to_scalar(
                    concat_post_relu_squared, "plus"
                )
                l2_norm = irb.math.powf(squared_sum, c_0_5_f64)
                concat_normalized = irb.graphblas.apply(
                    concat_post_relu, "div", right=l2_norm
                )

                concat_normalized_ptr8 = irb.util.tensor_to_ptr8(concat_normalized)

                concat_destination_ptr = irb.llvm.getelementptr(
                    concat_matrix_vectors, v_i64
                )
                irb.llvm.store(concat_normalized_ptr8, concat_destination_ptr)

            irb.util.resize_sparse_dim(concat_matrix_ptr8, c1, weight_matrix_ncols)
            concat_matrix_pointers = irb.sparse_tensor.pointers(concat_matrix, c1)
            concat_matrix_current_value_count = irb.new_var("index")
            with irb.for_loop(
                0, num_nodes, iter_vars=[(concat_matrix_current_value_count, c0)]
            ) as concat_matrix_num_val_loop_for_vars:
                node_idx = concat_matrix_num_val_loop_for_vars.iter_var_index
                node_idx_i64 = irb.arith.index_cast(node_idx, "i64")
                concat_ptr = irb.llvm.getelementptr(concat_matrix_vectors, node_idx_i64)
                concat_ptr8 = irb.llvm.load(concat_ptr, "!llvm.ptr<i8>")
                concat_vector = irb.util.ptr8_to_tensor(
                    concat_ptr8, "tensor<?xf64, #CV64>"
                )
                concat_vector_num_values = irb.graphblas.num_vals(concat_vector)
                updated_concat_matrix_current_value_count = irb.arith.addi(
                    concat_matrix_current_value_count, concat_vector_num_values
                )
                node_idx_plus_one = irb.arith.addi(node_idx, c1)
                updated_concat_matrix_current_value_count_i64 = irb.arith.index_cast(
                    updated_concat_matrix_current_value_count, "i64"
                )
                irb.memref.store(
                    updated_concat_matrix_current_value_count_i64,
                    concat_matrix_pointers,
                    node_idx_plus_one,
                )
                concat_matrix_num_val_loop_for_vars.yield_vars(
                    updated_concat_matrix_current_value_count
                )
            concat_matrix_num_vals = (
                concat_matrix_num_val_loop_for_vars.returned_variable[0]
            )
            irb.util.resize_sparse_index(concat_matrix_ptr8, c1, concat_matrix_num_vals)
            irb.util.resize_sparse_values(concat_matrix_ptr8, concat_matrix_num_vals)

            concat_matrix_pointers = irb.sparse_tensor.pointers(concat_matrix, c1)
            concat_matrix_indices = irb.sparse_tensor.indices(concat_matrix, c1)
            concat_matrix_values = irb.sparse_tensor.values(concat_matrix)
            with irb.for_loop(0, num_nodes) as concat_matrix_pointers_loop_for_vars:
                pointers_position = concat_matrix_pointers_loop_for_vars.iter_var_index
                pointers_position_plus_one = irb.arith.addi(pointers_position, c1)
                ptr_i64 = irb.memref.load(concat_matrix_pointers, pointers_position)
                next_ptr_i64 = irb.memref.load(
                    concat_matrix_pointers, pointers_position_plus_one
                )
                ptr = irb.arith.index_cast(ptr_i64, "index")
                next_ptr = irb.arith.index_cast(next_ptr_i64, "index")
                node_idx = pointers_position
                node_idx_i64 = irb.arith.index_cast(node_idx, "i64")
                concat_ptr = irb.llvm.getelementptr(concat_matrix_vectors, node_idx_i64)
                concat_ptr8 = irb.llvm.load(concat_ptr, "!llvm.ptr<i8>")
                concat_vector = irb.util.ptr8_to_tensor(
                    concat_ptr8, "tensor<?xf64, #CV64>"
                )
                concat_vector_num_vals = irb.arith.subi(next_ptr, ptr)
                concat_vector_indices = irb.sparse_tensor.indices(concat_vector, c0)
                concat_vector_values = irb.sparse_tensor.values(concat_vector)
                with irb.for_loop(
                    c0, concat_vector_num_vals
                ) as concat_matrix_fill_loop_for_vars:
                    concat_vector_values_position = (
                        concat_matrix_fill_loop_for_vars.iter_var_index
                    )
                    concat_vector_indices_value = irb.memref.load(
                        concat_vector_indices, concat_vector_values_position
                    )
                    concat_vector_values_value = irb.memref.load(
                        concat_vector_values, concat_vector_values_position
                    )
                    concat_matrix_values_position = irb.arith.addi(
                        ptr, concat_vector_values_position
                    )
                    irb.memref.store(
                        concat_vector_indices_value,
                        concat_matrix_indices,
                        concat_matrix_values_position,
                    )
                    irb.memref.store(
                        concat_vector_values_value,
                        concat_matrix_values,
                        concat_matrix_values_position,
                    )
                irb.util.del_sparse_tensor(concat_vector)

            post_relu = concat_matrix

            next_h = irb.graphblas.convert_layout(post_relu, "tensor<?x?xf64, #CSC64>")
            next_h_ptr8 = irb.util.tensor_to_ptr8(next_h)

            k_loop_for_vars.yield_vars(next_h_ptr8)

        final_h_ptr8 = k_loop_for_vars.returned_variable[0]
        final_h_csc = irb.util.ptr8_to_tensor(final_h_ptr8, "tensor<?x?xf64, #CSC64>")
        final_h = irb.graphblas.convert_layout(final_h_csc, "tensor<?x?xf64, #CSR64>")

        irb.return_vars(final_h)

        return irb

    def __call__(
        self,
        A: MLIRSparseTensor,
        features: MLIRSparseTensor,
        weight_matrices: List[MLIRSparseTensor],
        num_weight_matrices: int,
        sample_count_per_layer: List[int],
        rng_context: ChooseUniformContext,
        **kwargs,
    ) -> MLIRSparseTensor:
        return super().__call__(
            A,
            features,
            weight_matrices,
            num_weight_matrices,
            sample_count_per_layer,
            rng_context,
            **kwargs,
        )


graph_sage = GraphSAGE()


class GeoLocation(Algorithm):
    def _build(self):
        irb = MLIRFunctionBuilder(
            "geolocation",
            input_types=[
                "tensor<?x?xf64, #CSR64>",
                "tensor<?xf64, #CV64>",
                "tensor<?xf64, #CV64>",
                "i64",
                "f64",
                "f64",
            ],
            return_types=["tensor<?xf64, #CV64>", "tensor<?xf64, #CV64>"],
            aliases=_build_common_aliases(),
        )
        (A, known_lat, known_lon, max_iter, eps, max_mad) = irb.inputs
        A_csc = irb.graphblas.convert_layout(A, "tensor<?x?xf64, #CSC64>")

        TO_RADIANS = irb.arith.constant(math.tau / 360.0, "f64")
        TO_DEGREES = irb.arith.constant(360.0 / math.tau, "f64")
        c0 = irb.arith.constant(0, "index")
        c1 = irb.arith.constant(1, "index")
        ci0 = irb.arith.constant(0, "i64")
        ci1 = irb.arith.constant(1, "i64")
        ci2 = irb.arith.constant(2, "i64")
        cf0 = irb.arith.constant(0.0, "f64")
        cf1 = irb.arith.constant(1.0, "f64")
        size = irb.graphblas.size(known_lat)
        # TODO: replace this with assign scalar
        all_ones = irb.graphblas.reduce_to_vector(A, "plus", axis=1)
        all_ones = irb.graphblas.apply(all_ones, "second", right=cf1)

        unknown = irb.util.new_sparse_tensor("tensor<?xf64, #CV64>", size)
        irb.graphblas.update(all_ones, unknown, mask=known_lat, mask_complement=True)

        U = irb.graphblas.matrix_multiply(
            irb.graphblas.diag(unknown, "tensor<?x?xf64, #CSR64>"), A_csc, "any_second"
        )

        lat = irb.graphblas.dup(known_lat)
        lon = irb.graphblas.dup(known_lon)

        # Outer Loop
        with irb.while_loop(c0) as outer_loop:
            with outer_loop.before as outer_before:
                outer_count = outer_before.arg_vars[0]

                Ulat = irb.graphblas.matrix_multiply(
                    U, irb.graphblas.diag(lat, "tensor<?x?xf64, #CSC64>"), "any_second"
                )
                Ulat = irb.graphblas.convert_layout(Ulat, "tensor<?x?xf64, #CSC64>")
                Ulon = irb.graphblas.matrix_multiply(
                    U, irb.graphblas.diag(lon, "tensor<?x?xf64, #CSC64>"), "any_second"
                )
                Ulon = irb.graphblas.convert_layout(Ulon, "tensor<?x?xf64, #CSC64>")
                degrees = irb.graphblas.reduce_to_vector(Ulat, "count", axis=1)

                one_neighbor = irb.graphblas.cast(
                    irb.graphblas.select(degrees, "eq", ci1), "tensor<?xf64, #CV64>"
                )
                two_neighbors = irb.graphblas.cast(
                    irb.graphblas.select(degrees, "eq", ci2), "tensor<?xf64, #CV64>"
                )
                many_neighbors = irb.graphblas.cast(
                    irb.graphblas.select(degrees, "gt", ci2), "tensor<?xf64, #CV64>"
                )

                # Compute where num neighbors == 1
                mad = irb.graphblas.apply(one_neighbor, "times", right=cf0)
                one_neighbor = irb.graphblas.diag(
                    one_neighbor, "tensor<?x?xf64, #CSR64>"
                )
                irb.graphblas.update(
                    irb.graphblas.reduce_to_vector(
                        irb.graphblas.matrix_multiply(one_neighbor, Ulat, "any_second"),
                        "plus",
                        axis=1,
                    ),
                    lat,
                    accumulate="second",
                )
                irb.graphblas.update(
                    irb.graphblas.reduce_to_vector(
                        irb.graphblas.matrix_multiply(one_neighbor, Ulon, "any_second"),
                        "plus",
                        axis=1,
                    ),
                    lon,
                    accumulate="second",
                )

                # Compute where num neighbors == 2
                two_neighbors = irb.graphblas.diag(
                    two_neighbors, "tensor<?x?xf64, #CSR64>"
                )
                two_lat = irb.graphblas.apply(
                    irb.graphblas.matrix_multiply(two_neighbors, Ulat, "any_second"),
                    "times",
                    right=TO_RADIANS,
                )
                two_lon = irb.graphblas.apply(
                    irb.graphblas.matrix_multiply(two_neighbors, Ulon, "any_second"),
                    "times",
                    right=TO_RADIANS,
                )
                lat1 = irb.graphblas.reduce_to_vector(two_lat, "first", axis=1)
                lat2 = irb.graphblas.reduce_to_vector(two_lat, "last", axis=1)
                lon1 = irb.graphblas.reduce_to_vector(two_lon, "first", axis=1)
                lon2 = irb.graphblas.reduce_to_vector(two_lon, "last", axis=1)

                cos_lat2 = irb.graphblas.apply(lat2, "cos")
                diff_lon = irb.graphblas.intersect(lon2, lon1, "minus")
                bx = irb.graphblas.intersect(
                    cos_lat2, irb.graphblas.apply(diff_lon, "cos"), "times"
                )
                by = irb.graphblas.intersect(
                    cos_lat2, irb.graphblas.apply(diff_lon, "sin"), "times"
                )
                cos_lat1_bx = irb.graphblas.intersect(
                    irb.graphblas.apply(lat1, "cos"), bx, "plus"
                )
                lat3 = irb.graphblas.intersect(
                    irb.graphblas.intersect(
                        irb.graphblas.apply(lat1, "sin"),
                        irb.graphblas.apply(lat2, "sin"),
                        "plus",
                    ),
                    irb.graphblas.intersect(cos_lat1_bx, by, "hypot"),
                    "atan2",
                )
                lon3 = irb.graphblas.intersect(
                    lon1, irb.graphblas.intersect(by, cos_lat1_bx, "atan2"), "plus"
                )
                irb.graphblas.update(
                    irb.graphblas.apply(lat3, "times", right=TO_DEGREES),
                    lat,
                    accumulate="second",
                )
                irb.graphblas.update(
                    irb.graphblas.apply(lon3, "times", right=TO_DEGREES),
                    lon,
                    accumulate="second",
                )

                irb.graphblas.update(
                    algo_utils.haversine_distance(
                        irb, lat1, lon1, lat3, lon3, to_radians=False
                    ),
                    mad,
                    accumulate="plus",
                )

                # Compute where num neighbors > 2
                many_neighbors = irb.graphblas.diag(
                    many_neighbors, "tensor<?x?xf64, #CSR64>"
                )
                many_lat = irb.graphblas.matrix_multiply(
                    many_neighbors, Ulat, "any_second"
                )
                many_lon = irb.graphblas.matrix_multiply(
                    many_neighbors, Ulon, "any_second"
                )

                many_lat_csc = irb.graphblas.convert_layout(
                    many_lat, "tensor<?x?xf64, #CSC64>"
                )
                many_lon_csc = irb.graphblas.convert_layout(
                    many_lon, "tensor<?x?xf64, #CSC64>"
                )

                # compute mean as sum/count
                sum_lat = irb.graphblas.reduce_to_vector(many_lat, "plus", axis=1)
                sum_lon = irb.graphblas.reduce_to_vector(many_lon, "plus", axis=1)
                count_lat = irb.graphblas.cast(
                    irb.graphblas.reduce_to_vector(many_lat, "count", axis=1),
                    "tensor<?xf64, #CV64>",
                )
                count_lon = irb.graphblas.cast(
                    irb.graphblas.reduce_to_vector(many_lon, "count", axis=1),
                    "tensor<?xf64, #CV64>",
                )
                cur_lat = irb.graphblas.intersect(sum_lat, count_lat, "div")
                cur_lon = irb.graphblas.intersect(sum_lon, count_lon, "div")

                cur_lat_ptr8 = irb.util.tensor_to_ptr8(cur_lat)
                cur_lon_ptr8 = irb.util.tensor_to_ptr8(cur_lon)

                with irb.while_loop(cur_lat_ptr8, cur_lon_ptr8, ci0) as inner_loop:
                    with inner_loop.before as before_region:
                        cur_lat = irb.util.ptr8_to_tensor(
                            before_region.arg_vars[0], "tensor<?xf64, #CV64>"
                        )
                        cur_lon = irb.util.ptr8_to_tensor(
                            before_region.arg_vars[1], "tensor<?xf64, #CV64>"
                        )
                        count = before_region.arg_vars[2]

                        D = algo_utils.haversine_distance(
                            irb, many_lat_csc, many_lon_csc, cur_lat, cur_lon
                        )
                        Dinv = irb.graphblas.select(D, "ne", cf0)
                        Dinv = irb.graphblas.apply(Dinv, "minv")
                        Dinv_csc = irb.graphblas.convert_layout(
                            Dinv, "tensor<?x?xf64, #CSC64>"
                        )
                        Dinvs = irb.graphblas.reduce_to_vector(Dinv, "plus", axis=1)
                        W = irb.graphblas.matrix_multiply(
                            irb.graphblas.diag(Dinvs, "tensor<?x?xf64, #CSR64>"),
                            Dinv_csc,
                            "any_rdiv",
                        )

                        Tlat = irb.graphblas.reduce_to_vector(
                            irb.graphblas.intersect(many_lat, W, "times"),
                            "plus",
                            axis=1,
                        )
                        Tlon = irb.graphblas.reduce_to_vector(
                            irb.graphblas.intersect(many_lon, W, "times"),
                            "plus",
                            axis=1,
                        )

                        Dcounts = irb.graphblas.reduce_to_vector(D, "count", axis=1)
                        Dinv_counts = irb.graphblas.reduce_to_vector(
                            Dinv, "count", axis=1
                        )

                        num_zeros = irb.graphblas.cast(
                            irb.graphblas.intersect(Dcounts, Dinv_counts, "minus"),
                            "tensor<?xf64, #CV64>",
                        )
                        Rlat = irb.graphblas.intersect(
                            irb.graphblas.intersect(Tlat, cur_lat, "minus"),
                            Dinvs,
                            "times",
                        )
                        Rlon = irb.graphblas.intersect(
                            irb.graphblas.intersect(Tlon, cur_lon, "minus"),
                            Dinvs,
                            "times",
                        )
                        r = irb.graphblas.intersect(Rlat, Rlon, "hypot")

                        rinv = irb.graphblas.intersect(num_zeros, r, "div")
                        rinv_zeros = irb.graphblas.apply(
                            irb.graphblas.select(rinv, "isinf"), "second", right=cf0
                        )
                        irb.graphblas.update(rinv_zeros, rinv, "second")

                        alpha = irb.graphblas.apply(
                            irb.graphblas.apply(rinv, "minus", left=cf1),
                            "max",
                            right=cf0,
                        )
                        beta = irb.graphblas.apply(rinv, "min", cf1)
                        next_lat = irb.graphblas.intersect(
                            irb.graphblas.intersect(alpha, Tlat, "times"),
                            irb.graphblas.intersect(beta, cur_lat, "times"),
                            "plus",
                        )
                        next_lon = irb.graphblas.intersect(
                            irb.graphblas.intersect(alpha, Tlon, "times"),
                            irb.graphblas.intersect(beta, cur_lon, "times"),
                            "plus",
                        )

                        next_lat_nvals = irb.graphblas.num_vals(next_lat)
                        cur_lat_nvals = irb.graphblas.num_vals(cur_lat)
                        diff_nvals = irb.arith.cmpi(next_lat_nvals, cur_lat_nvals, "ne")
                        irb.add_statement(f"scf.if {diff_nvals} {{")
                        irb.graphblas.update(cur_lat, next_lat, "first")
                        irb.graphblas.update(cur_lon, next_lon, "first")
                        irb.add_statement("}")

                        diff_lat = irb.graphblas.intersect(cur_lat, next_lat, "minus")
                        diff_lon = irb.graphblas.intersect(cur_lon, next_lon, "minus")
                        not_converged = irb.graphblas.intersect(
                            diff_lat, diff_lon, "hypot"
                        )
                        not_converged = irb.graphblas.apply(
                            not_converged, "gt", eps, return_type="tensor<?xi8, #CV64>"
                        )
                        not_converged = irb.graphblas.reduce_to_scalar(
                            not_converged, "lor"
                        )
                        not_converged = irb.arith.trunci(not_converged, "i1")

                        max_iter_not_reached = irb.arith.cmpi(count, max_iter, "slt")
                        condition = irb.arith.andi(not_converged, max_iter_not_reached)

                        next_lat_ptr8 = irb.util.tensor_to_ptr8(next_lat)
                        next_lon_ptr8 = irb.util.tensor_to_ptr8(next_lon)
                        next_count = irb.arith.addi(count, ci1)
                        D_ptr8 = irb.util.tensor_to_ptr8(D)
                        before_region.condition(
                            condition, next_lat_ptr8, next_lon_ptr8, next_count, D_ptr8
                        )
                    with inner_loop.after as after_region:
                        after_region.yield_vars(*after_region.arg_vars[:3])

                cur_lat = irb.util.ptr8_to_tensor(
                    inner_loop.returned_variable[0], "tensor<?xf64, #CV64>"
                )
                cur_lon = irb.util.ptr8_to_tensor(
                    inner_loop.returned_variable[1], "tensor<?xf64, #CV64>"
                )
                D = irb.util.ptr8_to_tensor(
                    inner_loop.returned_variable[3], "tensor<?x?xf64, #CSR64>"
                )

                irb.graphblas.update(cur_lat, lat, accumulate="second")
                irb.graphblas.update(cur_lon, lon, accumulate="second")

                # TODO: this should actually use median, not mean
                # compute mean as sum/count
                sum_D = irb.graphblas.reduce_to_vector(D, "plus", axis=1)
                count_D = irb.graphblas.cast(
                    irb.graphblas.reduce_to_vector(D, "count", axis=1),
                    "tensor<?xf64, #CV64>",
                )
                mean_D = irb.graphblas.intersect(sum_D, count_D, "div")
                irb.graphblas.update(mean_D, mad, accumulate="second")

                # Drop values with large absolute deviation
                mad_mask = irb.graphblas.select(mad, "gt", max_mad)
                irb.graphblas.update(
                    lat, lat, mask=mad_mask, mask_complement=True, replace=True
                )
                irb.graphblas.update(
                    lon, lon, mask=mad_mask, mask_complement=True, replace=True
                )

                lat_nvals = irb.graphblas.num_vals(lat)
                condition = irb.arith.cmpi(lat_nvals, size, "ne")

                next_outer_count = irb.arith.addi(outer_count, c1)
                outer_before.condition(condition, next_outer_count)
            with outer_loop.after as outer_after:
                outer_after.yield_vars(*outer_after.arg_vars)

        # End Outer Loop

        irb.return_vars(lat, lon)

        return irb

    def __call__(
        self,
        graph: MLIRSparseTensor,
        lat: MLIRSparseTensor,
        lon: MLIRSparseTensor,
        *,
        max_iter: int = 1000,
        eps: float = 0.001,
        max_mad: float = 1500.0,
        **kwargs,
    ) -> MLIRSparseTensor:
        return super().__call__(graph, lat, lon, max_iter, eps, max_mad, **kwargs)


geolocation = GeoLocation()


class ConnectedComponents(Algorithm):
    def _build(self):
        irb = MLIRFunctionBuilder(
            "connected_components",
            input_types=["tensor<?x?xf64, #CSR64>"],
            return_types=[
                "tensor<?xf64, #CV64>",
            ],
            aliases=_build_common_aliases(),
        )

        (A,) = irb.inputs

        c0 = irb.arith.constant(0, "index")
        c1 = irb.arith.constant(1, "index")
        c0_i1 = irb.arith.constant(0, "i1")

        n = irb.graphblas.num_rows(A)
        n_i64 = irb.arith.index_cast(n, "i64")

        f = irb.util.new_sparse_tensor("tensor<?xf64, #CV64>", n)
        f_ptr8 = irb.util.tensor_to_ptr8(f)
        irb.util.resize_sparse_index(f_ptr8, c0, n)
        irb.util.resize_sparse_values(f_ptr8, n)
        f_pointers = irb.sparse_tensor.pointers(f, c0)
        f_indices = irb.sparse_tensor.indices(f, c0)
        f_values = irb.sparse_tensor.values(f)
        with irb.for_loop(0, n) as f_arange_for_vars:
            f_values_position = f_arange_for_vars.iter_var_index
            f_values_position_i64 = irb.arith.index_cast(f_values_position, "i64")
            f_values_position_f64 = irb.arith.sitofp(f_values_position_i64, "f64")
            irb.memref.store(f_values_position_i64, f_pointers, f_values_position)
            irb.memref.store(f_values_position_i64, f_indices, f_values_position)
            irb.memref.store(f_values_position_f64, f_values, f_values_position)
        irb.memref.store(n_i64, f_pointers, c1)

        gp = irb.graphblas.dup(f)
        gp_values = irb.sparse_tensor.values(gp)

        with irb.while_loop() as while_loop:
            with while_loop.before as before_region:
                # TODO code below assume I, f, gp, and gp_dup are dense
                # when sparse output is supported, make them dense tensors
                # There's a lot of room to optimize here.

                I = irb.graphblas.dup(f)
                gp_dup = irb.graphblas.dup(gp)

                # mngp << op.min_second(A @ gp)
                mngp = irb.graphblas.matrix_multiply(A, gp, "min_second")
                mngp_indices = irb.sparse_tensor.indices(mngp, c0)
                mngp_values = irb.sparse_tensor.values(mngp)

                # f(binary.min)[I] << mngp
                # TODO eventually implement grapblas.assign
                I_values = irb.sparse_tensor.values(I)
                mngp_pointer = irb.new_var("index")
                with irb.for_loop(
                    0, n, iter_vars=[(mngp_pointer, c0)]
                ) as for_vars:  # TODO parallelize
                    I_values_position = for_vars.iter_var_index

                    mngp_index_i64 = irb.memref.load(mngp_indices, mngp_pointer)
                    mngp_index = irb.arith.index_cast(mngp_index_i64, "index")
                    mngp_present = irb.arith.cmpi(mngp_index, I_values_position, "eq")

                    updated_mngp_pointer = irb.new_var("index")
                    irb.add_statement(
                        f"""
{updated_mngp_pointer.assign} = scf.if {mngp_present} -> (index) {{
"""
                    )
                    I_value = irb.memref.load(I_values, I_values_position)
                    I_value_i64 = irb.arith.fptosi(I_value, "i64")
                    I_value_as_index = irb.arith.index_cast(I_value_i64, "index")

                    f_value = irb.memref.load(f_values, I_value_as_index)
                    mngp_value = irb.memref.load(mngp_values, mngp_pointer)

                    # TODO would an scf.if be faster? Or do they compile down to the same code?
                    f_value_is_min = irb.arith.cmpf(f_value, mngp_value, "olt")

                    value_to_store = irb.select(f_value_is_min, f_value, mngp_value)
                    irb.memref.store(value_to_store, f_values, I_value_as_index)

                    mngp_pointer_plus_one = irb.arith.addi(mngp_pointer, c1)
                    irb.add_statement(
                        f"""
  scf.yield {mngp_pointer_plus_one} : index
}} else {{
  scf.yield {mngp_pointer} : index
}}
"""
                    )
                    for_vars.yield_vars(updated_mngp_pointer)

                # f << op.min(f | mngp)
                irb.graphblas.update(mngp, f, "min")

                # f << op.min(f | gp)
                irb.graphblas.update(gp, f, "min")

                # gp << f[I] // I has same values as f here
                # TODO eventually implement grapblas.extract
                with irb.for_loop(0, n) as for_vars:  # TODO parallelize
                    f_values_position = for_vars.iter_var_index
                    f_value = irb.memref.load(f_values, f_values_position)
                    f_value_i64 = irb.arith.fptosi(f_value, "i64")
                    f_value_as_index = irb.arith.index_cast(f_value_i64, "index")

                    value_to_store = irb.memref.load(f_values, f_value_as_index)
                    irb.memref.store(value_to_store, gp_values, f_values_position)

                no_change = irb.graphblas.equal(gp, gp_dup)
                change = irb.arith.cmpi(no_change, c0_i1, "eq")
                before_region.condition(change)
            with while_loop.after as after_region:
                after_region.yield_vars(*after_region.arg_vars)

        irb.return_vars(f)

        return irb

    def __call__(
        self, A: MLIRSparseTensor, **kwargs
    ) -> Tuple[MLIRSparseTensor, MLIRSparseTensor]:
        return super().__call__(A, **kwargs)


connected_components = ConnectedComponents()


class ApplicationClassification(Algorithm):
    def _build(self):
        irb = MLIRFunctionBuilder(
            "application_classification",
            input_types=[
                "tensor<?x?xf64, #CSR64>",
                "tensor<?x?xf64, #CSR64>",
                "tensor<?x?xindex>",
                "tensor<?x?xf64, #CSR64>",
                "tensor<?x?xf64, #CSR64>",
                "tensor<?x?xindex>",
            ],
            return_types=["tensor<?x?xf64, #CSR64>"],
            aliases=_build_common_aliases(),
        )
        (
            data_vertex,
            data_edges_table,
            data_edges,
            pattern_vertex,
            pattern_edges_table,
            pattern_edges,
        ) = irb.inputs
        num_dv = irb.graphblas.num_rows(data_vertex)
        num_pv = irb.graphblas.num_rows(pattern_vertex)
        num_de = irb.graphblas.num_rows(data_edges_table)
        num_pe = irb.graphblas.num_rows(pattern_edges_table)

        pe_ones = algo_utils.tensor_fill(irb, num_pe, 1, "f64")
        pattern_graph = irb.graphblas.from_coo(pattern_edges, pe_ones, (num_pv, num_pv))

        # Vertex Similarity
        cv = algo_utils.euclidean_distance(irb, data_vertex, pattern_vertex)
        neg_cv = irb.graphblas.apply(cv, "ainv")
        mu = algo_utils.normprob(irb, neg_cv)
        cv = algo_utils.normprob(irb, cv)
        mu_max = irb.graphblas.reduce_to_vector(mu, "max", axis=0)
        pe_arange = algo_utils.tensor_arange(irb, num_pe)
        pe_arange_col = algo_utils.tensor_to_col(irb, pe_arange)
        v_bak_max_graph = irb.graphblas.matrix_multiply(
            irb.graphblas.diag(mu_max, "tensor<?x?xf64, #CSR64>"),
            pattern_graph,
            "min_first",
        )
        _, v_bak_values = irb.graphblas.to_coo(v_bak_max_graph)
        v_bak_max = irb.graphblas.from_coo(pe_arange_col, v_bak_values, (num_pe,))
        v_fwd_max_graph = irb.graphblas.matrix_multiply(
            pattern_graph,
            irb.graphblas.diag(mu_max, "tensor<?x?xf64, #CSC64>"),
            "min_second",
        )
        _, v_fwd_values = irb.graphblas.to_coo(v_fwd_max_graph)
        v_fwd_max = irb.graphblas.from_coo(pe_arange_col, v_fwd_values, (num_pe,))

        # Edge Similarity
        ce = algo_utils.euclidean_distance(irb, data_edges_table, pattern_edges_table)
        neg_ce = irb.graphblas.apply(ce, "ainv")
        xe = algo_utils.normprob(irb, neg_ce)
        ce = algo_utils.normprob(irb, ce)

        # Combine
        cnull = irb.graphblas.from_coo(
            pe_arange_col,
            algo_utils.tensor_fill(irb, num_pe, 0, "f64"),
            shape=(num_pe,),
        )
        de_arange = algo_utils.tensor_arange(irb, num_de)
        de_arange_col = algo_utils.tensor_to_col(irb, de_arange)
        de_ones = algo_utils.tensor_fill(irb, num_de, 1, "f64")
        # from_coo requires sorted order, so build the transpose, then flip
        data_fwd_graph = irb.graphblas.from_coo(
            algo_utils.tensor_insert_col(irb, de_arange_col, data_edges, 0),
            de_ones,
            shape=(num_de, num_dv),
        )
        data_fwd_graph = irb.graphblas.transpose(
            data_fwd_graph, "tensor<?x?xf64, #CSR64>"
        )
        fwd_max = irb.graphblas.matrix_multiply(data_fwd_graph, xe, "max_second")
        irb.graphblas.update(
            irb.graphblas.uniform_complement(fwd_max, 0), fwd_max, "second"
        )
        fwd_max = irb.graphblas.matrix_multiply(
            fwd_max, irb.graphblas.diag(v_bak_max, "tensor<?x?xf64, #CSC64>"), "min_max"
        )
        data_bak_graph = irb.graphblas.from_coo(
            algo_utils.tensor_insert_col(irb, de_arange_col, data_edges, 1),
            de_ones,
            shape=(num_dv, num_de),
        )
        bak_max = irb.graphblas.matrix_multiply(data_bak_graph, xe, "max_second")
        irb.graphblas.update(
            irb.graphblas.uniform_complement(bak_max, 0), bak_max, "second"
        )
        bak_max = irb.graphblas.matrix_multiply(
            bak_max, irb.graphblas.diag(v_fwd_max, "tensor<?x?xf64, #CSC64>"), "min_max"
        )

        # Prepare loop inputs
        mu_ptr8 = irb.new_var("!llvm.ptr<i8>")
        fwd_max_ptr8 = irb.new_var("!llvm.ptr<i8>")
        bak_max_ptr8 = irb.new_var("!llvm.ptr<i8>")
        mu_init_ptr8 = irb.util.tensor_to_ptr8(mu)
        fwd_max_init_ptr8 = irb.util.tensor_to_ptr8(fwd_max)
        bak_max_init_ptr8 = irb.util.tensor_to_ptr8(bak_max)

        # Loop
        with irb.for_loop(
            0,
            num_pv,
            iter_vars=[
                (mu_ptr8, mu_init_ptr8),
                (fwd_max_ptr8, fwd_max_init_ptr8),
                (bak_max_ptr8, bak_max_init_ptr8),
            ],
        ) as for_vars:
            # Convert pointers to tensors
            mu = irb.util.ptr8_to_tensor(mu_ptr8, "tensor<?x?xf64, #CSR64>")
            fwd_max = irb.util.ptr8_to_tensor(fwd_max_ptr8, "tensor<?x?xf64, #CSR64>")
            bak_max = irb.util.ptr8_to_tensor(bak_max_ptr8, "tensor<?x?xf64, #CSR64>")

            # from_coo requires sorted order, so build the transpose, then flip
            pattern_fwd_graph = irb.graphblas.from_coo(
                algo_utils.tensor_insert_col(irb, pe_arange_col, pattern_edges, 0),
                pe_ones,
                shape=(num_pe, num_pv),
            )
            pattern_fwd_graph = irb.graphblas.transpose(
                pattern_fwd_graph, "tensor<?x?xf64, #CSC64>"
            )
            v_fwd = irb.graphblas.matrix_multiply(mu, pattern_fwd_graph, "min_first")
            v_fwd = irb.graphblas.intersect(v_fwd, fwd_max, "minus")
            pattern_bak_graph = irb.graphblas.from_coo(
                algo_utils.tensor_insert_col(irb, pe_arange_col, pattern_edges, 1),
                pe_ones,
                shape=(num_pv, num_pe),
            )
            v_bak = irb.graphblas.matrix_multiply(mu, pattern_bak_graph, "min_first")
            v_bak = irb.graphblas.intersect(v_bak, bak_max, "minus")
            v_fwd_max = irb.graphblas.reduce_to_vector(v_fwd, "max", axis=0)
            v_bak_max = irb.graphblas.reduce_to_vector(v_bak, "max", axis=0)
            dbg_transpose = irb.graphblas.transpose(
                data_bak_graph, "tensor<?x?xf64, #CSR64>"
            )
            e_bak = irb.graphblas.matrix_multiply(dbg_transpose, v_fwd, "min_second")
            e_bak = irb.graphblas.intersect(e_bak, ce, "minus")
            e_fwd = irb.graphblas.matrix_multiply(dbg_transpose, v_bak, "min_second")
            e_fwd = irb.graphblas.intersect(e_fwd, ce, "minus")
            e_bak_norm = irb.graphblas.apply(e_bak, "exp")
            e_bak_norm = irb.graphblas.reduce_to_vector(e_bak_norm, "plus", axis=0)
            e_bak_norm = irb.graphblas.apply(e_bak_norm, "log")
            e_fwd_norm = irb.graphblas.apply(e_fwd, "exp")
            e_fwd_norm = irb.graphblas.reduce_to_vector(e_fwd_norm, "plus", axis=0)
            e_fwd_norm = irb.graphblas.apply(e_fwd_norm, "log")
            fwd_max = irb.graphblas.matrix_multiply(data_fwd_graph, e_fwd, "max_second")
            irb.graphblas.update(
                irb.graphblas.uniform_complement(fwd_max, -math.inf), fwd_max, "second"
            )
            bak_max = irb.graphblas.matrix_multiply(data_bak_graph, e_bak, "max_second")
            irb.graphblas.update(
                irb.graphblas.uniform_complement(bak_max, -math.inf), bak_max, "second"
            )
            fwd_max = irb.graphblas.matrix_multiply(
                fwd_max,
                irb.graphblas.diag(e_fwd_norm, "tensor<?x?xf64, #CSC64>"),
                "min_minus",
            )
            bak_max = irb.graphblas.matrix_multiply(
                bak_max,
                irb.graphblas.diag(e_bak_norm, "tensor<?x?xf64, #CSC64>"),
                "min_minus",
            )
            fwd_max = irb.graphblas.matrix_multiply(
                fwd_max,
                irb.graphblas.diag(
                    irb.graphblas.intersect(v_bak_max, cnull, "minus"),
                    "tensor<?x?xf64, #CSC64>",
                ),
                "min_max",
            )
            bak_max = irb.graphblas.matrix_multiply(
                bak_max,
                irb.graphblas.diag(
                    irb.graphblas.intersect(v_fwd_max, cnull, "minus"),
                    "tensor<?x?xf64, #CSC64>",
                ),
                "min_max",
            )
            mu = irb.graphblas.apply(cv, "ainv")
            fwd_tmp = irb.graphblas.matrix_multiply(
                fwd_max,
                irb.graphblas.transpose(pattern_fwd_graph, "tensor<?x?xf64, #CSC64>"),
                "plus_plus",
            )
            bak_tmp = irb.graphblas.matrix_multiply(
                bak_max,
                irb.graphblas.transpose(pattern_bak_graph, "tensor<?x?xf64, #CSC64>"),
                "plus_plus",
            )
            irb.graphblas.update(fwd_tmp, mu, "plus")
            irb.graphblas.update(bak_tmp, mu, "plus")
            mu = algo_utils.normprob(irb, mu)

            # Cast yield args to pointers
            mu_result_ptr8 = irb.util.tensor_to_ptr8(mu)
            fwd_max_result_ptr8 = irb.util.tensor_to_ptr8(fwd_max)
            bak_max_result_ptr8 = irb.util.tensor_to_ptr8(bak_max)

            for_vars.yield_vars(
                mu_result_ptr8, fwd_max_result_ptr8, bak_max_result_ptr8
            )

        # One final cast from ptr8 to tensor
        mu_final = irb.util.ptr8_to_tensor(
            for_vars.returned_variable[0], "tensor<?x?xf64, #CSR64>"
        )

        irb.return_vars(mu_final)

        return irb

    def __call__(
        self,
        data_vertex: np.ndarray,
        data_edge_table: np.ndarray,
        pattern_vertex: np.ndarray,
        pattern_edge_table: np.ndarray,
        data_edges: np.ndarray,
        pattern_edges: np.ndarray,
        **kwargs,
    ) -> MLIRSparseTensor:
        return super().__call__(
            data_vertex,
            data_edge_table,
            pattern_vertex,
            pattern_edge_table,
            data_edges,
            pattern_edges,
            **kwargs,
        )


application_classification = ApplicationClassification()


class GraphWave(Algorithm):
    def _build(self):
        irb = MLIRFunctionBuilder(
            "triangle_count",
            input_types=[
                "tensor<?x?xf64, #CSR64>",
                "!llvm.ptr<f64>",
                "index",
                "f64",
                "index",
                "tensor<?xf64, #CV64>",
            ],
            return_types=["tensor<?x?xf64>"],
            aliases=_build_common_aliases(),
        )
        (W, taus, num_taus, lmax, chebyshev_order, t) = irb.inputs

        # TODO Sweep this to find optimizations

        c0 = irb.arith.constant(0, "index")
        c1 = irb.arith.constant(1, "index")
        c2 = irb.arith.constant(2, "index")
        c1_i64 = irb.arith.constant(1, "i64")
        c0_f64 = irb.arith.constant(0, "f64")
        c1_f64 = irb.arith.constant(1, "f64")
        c2_f64 = irb.arith.constant(2, "f64")
        c_pi_f64 = irb.arith.constant(np.pi, "f64")
        c_e_f64 = irb.arith.constant(np.e, "f64")
        c_0_5_f64 = irb.arith.constant(0.5, "f64")
        c_negative_0_5_f64 = irb.arith.constant(-0.5, "f64")
        c_negative_1_f64 = irb.arith.constant(-1, "f64")

        num_nodes = irb.graphblas.num_rows(W)
        num_nodes_i64 = irb.arith.index_cast(num_nodes, "i64")
        num_nodes_f64 = irb.arith.sitofp(num_nodes_i64, "f64")
        num_nodes_plus_one = irb.arith.addi(num_nodes, c1)

        dw = irb.graphblas.reduce_to_vector(W, "plus", 1)
        d = irb.graphblas.apply(dw, "pow", right=c_negative_0_5_f64)
        D = irb.graphblas.diag(d, "tensor<?x?xf64, #CSR64>")
        minus_D = irb.graphblas.apply(D, "times", right=c_negative_1_f64)
        L = irb.graphblas.matrix_multiply(minus_D, W, "plus_times")
        L = irb.graphblas.matrix_multiply(L, D, "plus_times")

        eye = irb.util.new_sparse_tensor(
            "tensor<?x?xf64, #CSR64>", num_nodes, num_nodes
        )
        eye_ptr8 = irb.util.tensor_to_ptr8(eye)
        irb.util.resize_sparse_pointers(eye_ptr8, c1, num_nodes_plus_one)
        irb.util.resize_sparse_index(eye_ptr8, c1, num_nodes)
        irb.util.resize_sparse_values(eye_ptr8, num_nodes)
        eye_pointers = irb.sparse_tensor.pointers(eye, c1)
        eye_indices = irb.sparse_tensor.indices(eye, c1)
        eye_values = irb.sparse_tensor.values(eye)
        with irb.for_loop(0, num_nodes) as eye_arange_for_vars:
            eye_values_position = eye_arange_for_vars.iter_var_index
            eye_values_position_i64 = irb.arith.index_cast(eye_values_position, "i64")
            irb.memref.store(eye_values_position_i64, eye_pointers, eye_values_position)
            irb.memref.store(eye_values_position_i64, eye_indices, eye_values_position)
            irb.memref.store(c1_f64, eye_values, eye_values_position)
        irb.memref.store(num_nodes_i64, eye_pointers, num_nodes)

        irb.graphblas.update(eye, L, accumulate="plus")

        # different signal choices exist
        # TODO update this according to the notebook
        signal = eye  # TODO is this used?
        n_signals = num_nodes  # irb.graphblas.num_cols(signal)
        n_features_out = num_taus

        num_taus_i64 = num_nodes_i64  # irb.arith.index_cast(num_taus, "i64")
        c_tensors = irb.llvm.alloca(num_taus_i64, "!llvm.ptr<i8>")

        N = irb.arith.addi(chebyshev_order, c1)
        N_i64 = irb.arith.index_cast(N, "i64")
        range_N = irb.util.new_sparse_tensor("tensor<?xf64, #CV64>", N)
        range_N_ptr8 = irb.util.tensor_to_ptr8(range_N)
        irb.util.resize_sparse_index(range_N_ptr8, c0, N)
        irb.util.resize_sparse_values(range_N_ptr8, N)
        range_N_pointers = irb.sparse_tensor.pointers(range_N, c0)
        range_N_indices = irb.sparse_tensor.indices(range_N, c0)
        range_N_values = irb.sparse_tensor.values(range_N)
        # TODO make this parallel
        with irb.for_loop(0, N) as range_N_arange_for_vars:
            range_N_values_position = range_N_arange_for_vars.iter_var_index
            range_N_values_position_i64 = irb.arith.index_cast(
                range_N_values_position, "i64"
            )
            irb.memref.store(
                range_N_values_position_i64, range_N_indices, range_N_values_position
            )
            range_N_values_position_f64 = irb.arith.sitofp(
                range_N_values_position_i64, "f64"
            )
            irb.memref.store(
                range_N_values_position_f64, range_N_values, range_N_values_position
            )
        irb.memref.store(N_i64, range_N_pointers, c1)

        tmp_N = irb.graphblas.apply(range_N, "plus", right=c_0_5_f64)
        N_f64 = irb.arith.sitofp(N_i64, "f64")
        pi_over_N = irb.arith.divf(c_pi_f64, N_f64)
        tmp_N = irb.graphblas.apply(tmp_N, "times", right=pi_over_N)
        num = irb.graphblas.apply(tmp_N, "cos")

        a = irb.arith.mulf(lmax, c_0_5_f64)

        # TODO we need a graphblas.outer_product op so that we can have easy loop-fusion
        # TODO graphblas.outer_product is parallelizable
        tmp_N_values = irb.sparse_tensor.values(tmp_N)
        z_vectors = irb.llvm.alloca(num_taus_i64, "!llvm.ptr<i8>")
        with irb.for_loop(0, N) as z_for_vars:
            row_index = z_for_vars.iter_var_index
            row_index_i64 = irb.arith.index_cast(row_index, "i64")
            tmp_N_value = irb.memref.load(tmp_N_values, row_index)
            row = irb.graphblas.apply(range_N, "times", right=tmp_N_value)
            row_ptr8 = irb.util.tensor_to_ptr8(row)
            destination_ptr = irb.llvm.getelementptr(z_vectors, row_index_i64)
            irb.llvm.store(row_ptr8, destination_ptr)
        z = irb.util.new_sparse_tensor("tensor<?x?xf64, #CSR64>", N, N)
        z_ptr8 = irb.util.tensor_to_ptr8(z)
        N_squared = irb.arith.muli(N, N)
        N_squared_i64 = irb.arith.index_cast(N_squared, "i64")
        N_plus_one = irb.arith.addi(N, c1)
        irb.util.resize_sparse_pointers(z_ptr8, c1, N_plus_one)
        irb.util.resize_sparse_index(z_ptr8, c1, N_squared)
        irb.util.resize_sparse_values(z_ptr8, N_squared)
        z_pointers = irb.sparse_tensor.pointers(z, c1)
        z_indices = irb.sparse_tensor.indices(z, c1)
        z_values = irb.sparse_tensor.values(z)
        with irb.for_loop(0, N) as z_concat_loop_row_for_vars:
            row_index = z_concat_loop_row_for_vars.iter_var_index
            row_index_i64 = irb.arith.index_cast(row_index, "i64")
            n_times_row_index = irb.arith.muli(N, row_index)
            n_times_row_index_i64 = irb.arith.index_cast(n_times_row_index, "i64")
            irb.memref.store(n_times_row_index_i64, z_pointers, row_index)
            row_vector_ptr = irb.llvm.getelementptr(z_vectors, row_index_i64)
            row_vector_ptr8 = irb.llvm.load(row_vector_ptr, "!llvm.ptr<i8>")
            row_vector = irb.util.ptr8_to_tensor(
                row_vector_ptr8, "tensor<?xf64, #CV64>"
            )
            row_vector_values = irb.sparse_tensor.values(row_vector)
            with irb.for_loop(0, N) as z_concat_loop_col_for_vars:
                col_index = z_concat_loop_col_for_vars.iter_var_index
                col_index_i64 = irb.arith.index_cast(col_index, "i64")
                values_position = irb.arith.addi(n_times_row_index, col_index)
                irb.memref.store(col_index_i64, z_indices, values_position)
                value = irb.memref.load(row_vector_values, col_index)
                irb.memref.store(value, z_values, values_position)
        irb.memref.store(N_squared_i64, z_pointers, N)
        # TODO ideally done with loop fusion with the above loops
        z = irb.graphblas.apply(z, "cos")

        c_vectors = irb.llvm.alloca(num_taus_i64, "!llvm.ptr<i8>")
        two_over_N = irb.arith.divf(c2_f64, N_f64)
        negative_a_over_lmax = irb.arith.divf(a, lmax)
        negative_a_over_lmax = irb.arith.mulf(negative_a_over_lmax, c_negative_1_f64)
        # TODO make this parallel if num_taus is big enough
        # in practice; if users do not usually make num_taus
        # big enough, add a comment saying this is why we intentionally
        # made this loop sequential.
        with irb.for_loop(0, num_taus) as tau_for_vars:
            idx = tau_for_vars.iter_var_index
            idx_i64 = irb.arith.index_cast(idx, "i64")
            tau_ptr = irb.llvm.getelementptr(taus, idx_i64)
            tau = irb.llvm.load(tau_ptr, "f64")
            scalar_factor = irb.arith.mulf(tau, negative_a_over_lmax)
            y = irb.graphblas.apply(num, "plus", right=c1_f64)
            y = irb.graphblas.apply(y, "times", right=scalar_factor)
            # TODO make "exp" an apply and update operator.
            y = irb.graphblas.apply(y, "pow", left=c_e_f64)
            y = irb.graphblas.apply(y, "times", right=two_over_N)
            c_vector = irb.graphblas.matrix_multiply(y, z, "plus_times")
            c_vector_ptr8 = irb.util.tensor_to_ptr8(c_vector)
            destination_ptr = irb.llvm.getelementptr(c_vectors, idx_i64)
            irb.llvm.store(c_vector_ptr8, destination_ptr)

        n_features_out = num_taus
        n_features_out_i64 = num_taus_i64
        heat_print_matrices = irb.llvm.alloca(n_features_out_i64, "!llvm.ptr<i8>")

        twf_old = signal
        twf_cur = irb.graphblas.matrix_multiply(L, signal, "plus_times")
        twf_cur = irb.graphblas.apply(twf_cur, "div", right=a)
        negative_signal = irb.graphblas.apply(signal, "ainv")
        irb.graphblas.update(negative_signal, twf_cur, "plus")

        # TODO make this parallel if n_features_out is big enough
        # in practice; if users do not usually make n_features_out
        # big enough, add a comment saying this is why we intentionally
        # made this loop sequential.
        with irb.for_loop(0, n_features_out) as heat_print_for_vars:
            idx = heat_print_for_vars.iter_var_index
            idx_i64 = irb.arith.index_cast(idx, "i64")
            c_vector_ptr = irb.llvm.getelementptr(c_vectors, idx_i64)
            c_vector_ptr8 = irb.llvm.load(c_vector_ptr, "!llvm.ptr<i8>")
            c_vector = irb.util.ptr8_to_tensor(c_vector_ptr8, "tensor<?xf64, #CV64>")
            # assumes c_vector is dense
            c_vector_values = irb.sparse_tensor.values(c_vector)
            c_vector_0 = irb.memref.load(c_vector_values, c0)
            c_vector_1 = irb.memref.load(c_vector_values, c1)
            c_vector_0_over_2 = irb.arith.mulf(c_vector_0, c_0_5_f64)
            heat_print_matrix = irb.graphblas.apply(
                twf_old, "times", right=c_vector_0_over_2
            )
            right_summand = irb.graphblas.apply(twf_cur, "times", right=c_vector_1)
            irb.graphblas.update(right_summand, heat_print_matrix, accumulate="plus")
            heat_print_matrix_ptr8 = irb.util.tensor_to_ptr8(heat_print_matrix)
            destination_ptr = irb.llvm.getelementptr(heat_print_matrices, idx_i64)
            irb.llvm.store(heat_print_matrix_ptr8, destination_ptr)

        negative_a = irb.arith.mulf(a, c_negative_1_f64)
        factor = irb.graphblas.apply(eye, "second", right=negative_a)
        irb.graphblas.update(L, factor, "plus")
        two_over_a = irb.arith.divf(c2_f64, a)
        # TODO this apply should be done in place
        factor = irb.graphblas.apply(factor, "times", right=two_over_a)

        twf_cur_ptr8_init = irb.util.tensor_to_ptr8(twf_cur)
        twf_old_ptr8_init = irb.util.tensor_to_ptr8(twf_old)
        twf_cur_ptr8 = irb.new_var("!llvm.ptr<i8>")
        twf_old_ptr8 = irb.new_var("!llvm.ptr<i8>")
        with irb.for_loop(
            2,
            N,
            iter_vars=[
                (twf_cur_ptr8, twf_cur_ptr8_init),
                (twf_old_ptr8, twf_old_ptr8_init),
            ],
        ) as k_for_vars:
            k = k_for_vars.iter_var_index
            twf_cur = irb.util.ptr8_to_tensor(twf_cur_ptr8, "tensor<?x?xf64, #CSR64>")
            twf_old = irb.util.ptr8_to_tensor(twf_old_ptr8, "tensor<?x?xf64, #CSR64>")

            twf_new = irb.graphblas.matrix_multiply(factor, twf_cur, "plus_times")
            # TODO this should be handled by irb.graphblas.update
            negative_twf_old = irb.graphblas.apply(twf_old, "ainv")
            irb.graphblas.update(negative_twf_old, twf_new, "plus")
            with irb.for_loop(0, n_features_out) as i_for_vars:
                i = i_for_vars.iter_var_index
                i_i64 = irb.arith.index_cast(i, "i64")
                heat_print_matrix_ptr = irb.llvm.getelementptr(
                    heat_print_matrices, i_i64
                )
                heat_print_matrix_ptr8 = irb.llvm.load(
                    heat_print_matrix_ptr, "!llvm.ptr<i8>"
                )
                heat_print_matrix = irb.util.ptr8_to_tensor(
                    heat_print_matrix_ptr8, "tensor<?x?xf64, #CSR64>"
                )
                c_vector_ptr = irb.llvm.getelementptr(c_vectors, i_i64)
                c_vector_ptr8 = irb.llvm.load(c_vector_ptr, "!llvm.ptr<i8>")
                c_vector = irb.util.ptr8_to_tensor(
                    c_vector_ptr8, "tensor<?xf64, #CV64>"
                )
                # assumes c_vector is dense
                # TODO should we track the c_vector value arrays in a separate array since we use them often?
                c_vector_values = irb.sparse_tensor.values(c_vector)
                c_vector_k = irb.memref.load(c_vector_values, k)
                updater = irb.graphblas.apply(twf_new, "times", right=c_vector_k)
                irb.graphblas.update(updater, heat_print_matrix, accumulate="plus")
            twf_new_ptr8 = irb.util.tensor_to_ptr8(twf_new)
            k_for_vars.yield_vars(twf_new_ptr8, twf_cur_ptr8)

        t_size = irb.graphblas.size(t)
        t_size_i64 = irb.arith.index_cast(t_size, "i64")
        t_size_plus_one = irb.arith.addi(t_size, c1)
        t_indices = irb.sparse_tensor.indices(t, c0)
        t_values = irb.sparse_tensor.values(t)

        output_nrows = num_nodes
        output_ncols = irb.arith.muli(t_size, num_taus)
        output_ncols = irb.arith.muli(output_ncols, c2)
        output_memref = irb.memref.alloca("memref<?x?xf64>", output_nrows, output_ncols)
        output_tensor = irb.memref.tensor_load(output_memref, "tensor<?x?xf64>")

        # TODO is this parallelizable? Looks like it might be.
        # If this outer-loop has too few iterations in practice
        # (since n_features_out depends on usesr-given input),
        # consider making one of the inner-loops parallel.
        with irb.for_loop(0, n_features_out) as heat_print_for_vars:
            i = heat_print_for_vars.iter_var_index
            i_i64 = irb.arith.index_cast(i, "i64")
            heat_print_matrix_ptr = irb.llvm.getelementptr(heat_print_matrices, i_i64)
            heat_print_matrix_ptr8 = irb.llvm.load(
                heat_print_matrix_ptr, "!llvm.ptr<i8>"
            )
            heat_print_matrix = irb.util.ptr8_to_tensor(
                heat_print_matrix_ptr8, "tensor<?x?xf64, #CSR64>"
            )

            # TODO we really need graphblas.extract
            row_selector = irb.util.new_sparse_tensor("tensor<?xf64, #CV64>", num_nodes)
            row_selector_ptr8 = irb.util.tensor_to_ptr8(row_selector)
            irb.util.resize_sparse_index(row_selector_ptr8, c0, c1)
            irb.util.resize_sparse_values(row_selector_ptr8, c1)
            row_selector_pointers = irb.sparse_tensor.pointers(row_selector, c0)
            irb.memref.store(c1_i64, row_selector_pointers, c1)
            row_selector_indices = irb.sparse_tensor.indices(row_selector, c0)
            row_selector_values = irb.sparse_tensor.values(row_selector)
            with irb.for_loop(0, num_nodes) as node_for_vars:
                node_idx = node_for_vars.iter_var_index
                node_idx_i64 = irb.arith.index_cast(node_idx, "i64")
                irb.memref.store(node_idx_i64, row_selector_indices, c0)
                node_sig = irb.graphblas.matrix_multiply(
                    row_selector, heat_print_matrix, "any_second"
                )
                A = irb.graphblas.apply(node_sig, "cos")
                B = irb.graphblas.apply(node_sig, "sin")
                theta = irb.graphblas.intersect(B, A, "atan2")
                theta_values = irb.sparse_tensor.values(theta)

                # TODO we REALLY need a graphblas.outer_product op so that we can have easy loop-fusion
                t_position = irb.new_var("index")
                with irb.for_loop(
                    0, t_size, iter_vars=[(t_position, c0)]
                ) as row_for_vars:
                    t_index = row_for_vars.iter_var_index
                    t_index_i64 = irb.arith.index_cast(t_index, "i64")

                    t_value = irb.memref.load(t_values, t_position)
                    t_indices_value_i64 = irb.memref.load(t_indices, t_position)

                    # TODO maybe it's better to make the position variables for-loop vars
                    # since addition is faster than multiplication
                    output_base_tau_position = irb.arith.muli(c2, t_size)
                    output_base_tau_position = irb.arith.muli(
                        output_base_tau_position, i
                    )

                    cos_position = irb.arith.muli(c2, t_index)
                    cos_position = irb.arith.addi(
                        cos_position, output_base_tau_position
                    )
                    sin_position = irb.arith.addi(cos_position, c1)

                    row_present = irb.arith.cmpi(t_indices_value_i64, t_index_i64, "eq")
                    updated_t_position = irb.new_var("index")
                    irb.add_statement(
                        f"{updated_t_position.assign} = scf.if {row_present} -> (index) {{"
                    )
                    row = irb.graphblas.apply(theta, "times", right=t_value)

                    row_cos = irb.graphblas.apply(row, "cos")
                    row_cos_sum = irb.graphblas.reduce_to_scalar(row_cos, "plus")
                    row_cos_sum_normalized = irb.arith.divf(row_cos_sum, num_nodes_f64)
                    irb.memref.store(
                        row_cos_sum_normalized, output_memref, [node_idx, cos_position]
                    )

                    row_sin = irb.graphblas.apply(row, "sin")
                    row_sin_sum = irb.graphblas.reduce_to_scalar(row_sin, "plus")
                    row_sin_sum_normalized = irb.arith.divf(row_sin_sum, num_nodes_f64)
                    irb.memref.store(
                        row_sin_sum_normalized, output_memref, [node_idx, sin_position]
                    )

                    incremented_t_position = irb.arith.addi(t_position, c1)
                    irb.add_statement(f"scf.yield {incremented_t_position}: index")
                    irb.add_statement("} else {")
                    irb.memref.store(c1_f64, output_memref, [node_idx, cos_position])
                    irb.memref.store(c0_f64, output_memref, [node_idx, sin_position])
                    irb.add_statement(f"scf.yield {t_position} : index")
                    irb.add_statement("}")

                    row_for_vars.yield_vars(updated_t_position)

        irb.return_vars(output_tensor)

        return irb

    def __call__(
        self,
        W: MLIRSparseTensor,
        taus: List[float],
        lmax: float,  # max eigenvalue
        chebyshev_order: int,
        t: MLIRSparseTensor,
        **kwargs,
    ) -> int:
        num_taus = len(taus)
        assert chebyshev_order >= 2
        return super().__call__(W, taus, num_taus, lmax, chebyshev_order, t, **kwargs)


graph_wave = GraphWave()
