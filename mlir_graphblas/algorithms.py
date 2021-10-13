import numpy as np
from typing import List
from mlir_graphblas.mlir_builder import MLIRFunctionBuilder
from mlir_graphblas.types import AliasMap, SparseEncodingType, AffineMap
from mlir_graphblas.random_utils import ChooseUniformContext, ChooseWeightedContext
from .sparse_utils import MLIRSparseTensor


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
        U = irb.graphblas.select(inp, [], "triu")
        L = irb.graphblas.select(inp, [], "tril")
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
        zero_f64 = irb_inner.constant(0.0, "f64")
        # Perform inference
        W_csc = irb_inner.graphblas.convert_layout(weights, "tensor<?x?xf64, #CSC64>")
        matmul_result = irb_inner.graphblas.matrix_multiply(Y, W_csc, "plus_times")
        add_bias_result = irb_inner.graphblas.matrix_multiply(
            matmul_result, biases, "plus_plus"
        )
        clamp_result = irb_inner.graphblas.apply(add_bias_result, "min", left=threshold)
        relu_result = irb_inner.graphblas.select(clamp_result, [zero_f64], ["gt"])

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
        filtered = ir_builder.graphblas.select(projection, [limit], ["ge"])
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
        result = irb.index_cast(result_64, "index")

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
        L = ir_builder.graphblas.select(A, [], ["tril"])
        L_T = ir_builder.graphblas.transpose(L, "tensor<?x?xf64, #CSC64>")
        A_triangles = ir_builder.graphblas.matrix_multiply(A, L_T, "plus_pair", mask=A)
        tri = ir_builder.graphblas.reduce_to_vector(A_triangles, "plus", 1)
        answer_64 = ir_builder.graphblas.reduce_to_scalar(tri, "argmax")
        answer = ir_builder.index_cast(answer_64, "index")
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
        starting = irb.graphblas.apply(r, "fill", right=nrows_inv)
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
            converged = irb.cmpf(rdiff, var_tol, "olt")

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
            new_score = irb.graphblas.apply(prev_score, "fill", right=teleport)

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

            # Note: this is commented out due to https://github.com/metagraph-dev/mlir-graphblas/issues/163
            # Clean up previous score
            # irb.util.del_sparse_tensor(prev_score)

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

        c0 = irb.constant(0, "index")
        c1 = irb.constant(1, "index")
        ci1 = irb.constant(1, "i64")
        cf1 = irb.constant(1.0, "f64")

        # Create count vector
        ncols = irb.graphblas.num_cols(A)
        count = irb.util.new_sparse_tensor("tensor<?xf64, #CV64>", ncols)

        # Create B matrix, sized (nseeds x ncols) with nnz=nseeds
        nseeds = irb.tensor.dim(seeds, c0)
        nseeds_64 = irb.index_cast(nseeds, "i64")
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
            seed_num_64 = irb.index_cast(seed_num, "i64")
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
