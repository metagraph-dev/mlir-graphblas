"""
Various functions written in MLIR which implement pieces of GraphBLAS or other utilities
"""
import jinja2
from .engine import MlirJitEngine
from .sparse_utils import MLIRSparseTensor


class MLIRCompileError(Exception):
    pass


_default_engine = MlirJitEngine()

_standard_passes = (
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
    "--convert-std-to-llvm",
)


class BaseFunction:
    func_name = None
    _compiled = None  # (engine, passes, callable)

    def __init__(self):
        pass

    def get_mlir(self, *, make_private=True):
        """
        `make_private` is used to indicate whether the function should be private for fusion
        or public for standalone calling in the `compile` method.
        """
        raise NotImplementedError()

    MODULE_WRAPPER_TEXT = jinja2.Template(
        """\
#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module  {
    func private @empty(tensor<?x?xf64, #CSR64>, index) -> tensor<?x?xf64, #CSR64>
    func private @empty_like(tensor<?x?xf64, #CSR64>) -> tensor<?x?xf64, #CSR64>
    func private @dup_tensor(tensor<?x?xf64, #CSR64>) -> tensor<?x?xf64, #CSR64>
    func private @ptr8_to_tensor(!llvm.ptr<i8>) -> tensor<?x?xf64, #CSR64>
    func private @tensor_to_ptr8(tensor<?x?xf64, #CSR64>) -> !llvm.ptr<i8>

    func private @resize_pointers(tensor<?x?xf64, #CSR64>, index, index) -> ()
    func private @resize_index(tensor<?x?xf64, #CSR64>, index, index) -> ()
    func private @resize_values(tensor<?x?xf64, #CSR64>, index) -> ()
    func private @resize_dim(tensor<?x?xf64, #CSR64>, index, index) -> ()

    {{ body }}

}
        """
    )

    def get_mlir_module(self):
        """Get the MLIR text for this function wrapped in a MLIR module with
        declarations of external helper functions."""
        return self.MODULE_WRAPPER_TEXT.render(
            body=self.get_mlir(make_private=False),
        )

    def compile(self, engine=None, passes=None):
        if engine is None:
            engine = _default_engine
        if passes is None:
            passes = _standard_passes
        passes = tuple(passes)

        if self.func_name is None:
            raise MLIRCompileError(
                f"{self.__class__.__name__} does not define func_name"
            )

        if self._compiled is not None:
            prev_engine, prev_passes, compiled_func = self._compiled
            if prev_engine is engine and prev_passes == passes:
                return compiled_func

        # Force recompilation if name is already registered
        if self.func_name in engine.name_to_callable:
            del engine.name_to_callable[self.func_name]

        mlir = self.get_mlir_module()

        engine.add(mlir, passes)
        func = engine[self.func_name]
        self._compiled = (engine, tuple(passes), func)
        return func


class Transpose(BaseFunction):
    """
    Call signature:
      transpose(input: MLIRSparseTensor) -> MLIRSparseTensor
    """

    def __init__(self, swap_sizes=True):
        """
        swap_sizes will perform a normal transpose where the dimension sizes swap
        Set this to false if transposing to change from CSR to CSC format, and therefore
        don't want the dimension sizes to change.
        """
        super().__init__()
        func_name = "transpose_noswap" if not swap_sizes else "transpose"
        self.func_name = func_name
        self.swap_sizes = swap_sizes

    def get_mlir(self, *, make_private=True):
        return self.mlir_template.render(
            func_name=self.func_name,
            private_func=make_private,
            swap_sizes=self.swap_sizes,
        )

    mlir_template = jinja2.Template(
        """
      func {% if private_func %}private {% endif %}@{{ func_name }}(%input: tensor<?x?xf64, #CSR64>) -> tensor<?x?xf64, #CSR64> {
        // Attempting to implement identical code as in scipy
        // https://github.com/scipy/scipy/blob/3b36a574dc657d1ca116f6e230be694f3de31afc/scipy/sparse/sparsetools/csr.h
        // function csr_tocsc

        %c0 = constant 0 : index
        %c1 = constant 1 : index
        %c0_64 = constant 0 : i64
        %c1_64 = constant 1 : i64

        // pymlir-skip: begin

        %Ap = sparse_tensor.pointers %input, %c1 : tensor<?x?xf64, #CSR64> to memref<?xi64>
        %Aj = sparse_tensor.indices %input, %c1 : tensor<?x?xf64, #CSR64> to memref<?xi64>
        %Ax = sparse_tensor.values %input : tensor<?x?xf64, #CSR64> to memref<?xf64>

        %nrow = memref.dim %input, %c0 : tensor<?x?xf64, #CSR64>
        %ncol = memref.dim %input, %c1 : tensor<?x?xf64, #CSR64>
        %ncol_plus_one = addi %ncol, %c1 : index
        %nnz_64 = memref.load %Ap[%nrow] : memref<?xi64>
        %nnz = index_cast %nnz_64 : i64 to index

        %output = call @empty_like(%input) : (tensor<?x?xf64, #CSR64>) -> tensor<?x?xf64, #CSR64>
        {% if swap_sizes %}
        call @resize_dim(%output, %c0, %ncol) : (tensor<?x?xf64, #CSR64>, index, index) -> ()
        call @resize_dim(%output, %c1, %nrow) : (tensor<?x?xf64, #CSR64>, index, index) -> ()
        {% else %}
        call @resize_dim(%output, %c0, %nrow) : (tensor<?x?xf64, #CSR64>, index, index) -> ()
        call @resize_dim(%output, %c1, %ncol) : (tensor<?x?xf64, #CSR64>, index, index) -> ()
        {% endif %}
        call @resize_pointers(%output, %c1, %ncol_plus_one) : (tensor<?x?xf64, #CSR64>, index, index) -> ()
        call @resize_index(%output, %c1, %nnz) : (tensor<?x?xf64, #CSR64>, index, index) -> ()
        call @resize_values(%output, %nnz) : (tensor<?x?xf64, #CSR64>, index) -> ()
        
        %Bp = sparse_tensor.pointers %output, %c1 : tensor<?x?xf64, #CSR64> to memref<?xi64>
        %Bi = sparse_tensor.indices %output, %c1 : tensor<?x?xf64, #CSR64> to memref<?xi64>
        %Bx = sparse_tensor.values %output : tensor<?x?xf64, #CSR64> to memref<?xf64>

        // compute number of non-zero entries per column of A
        scf.for %arg2 = %c0 to %ncol step %c1 {
          memref.store %c0_64, %Bp[%arg2] : memref<?xi64>
        }
        scf.for %n = %c0 to %nnz step %c1 {
          %colA_64 = memref.load %Aj[%n] : memref<?xi64>
          %colA = index_cast %colA_64 : i64 to index
          %colB = memref.load %Bp[%colA] : memref<?xi64>
          %colB1 = addi %colB, %c1_64 : i64
          memref.store %colB1, %Bp[%colA] : memref<?xi64>
        }

        // cumsum the nnz per column to get Bp
        memref.store %c0_64, %Bp[%ncol] : memref<?xi64>
        scf.for %col = %c0 to %ncol step %c1 {
          %temp = memref.load %Bp[%col] : memref<?xi64>
          %cumsum = memref.load %Bp[%ncol] : memref<?xi64>
          memref.store %cumsum, %Bp[%col] : memref<?xi64>
          %cumsum2 = addi %cumsum, %temp : i64
          memref.store %cumsum2, %Bp[%ncol] : memref<?xi64>
        }

        scf.for %row = %c0 to %nrow step %c1 {
          %row_64 = index_cast %row : index to i64
          %j_start_64 = memref.load %Ap[%row] : memref<?xi64>
          %j_start = index_cast %j_start_64 : i64 to index
          %row_plus1 = addi %row, %c1 : index
          %j_end_64 = memref.load %Ap[%row_plus1] : memref<?xi64>
          %j_end = index_cast %j_end_64 : i64 to index
          scf.for %jj = %j_start to %j_end step %c1 {
            %col_64 = memref.load %Aj[%jj] : memref<?xi64>
            %col = index_cast %col_64 : i64 to index
            %dest_64 = memref.load %Bp[%col] : memref<?xi64>
            %dest = index_cast %dest_64 : i64 to index

            memref.store %row_64, %Bi[%dest] : memref<?xi64>
            %axjj = memref.load %Ax[%jj] : memref<?xf64>
            memref.store %axjj, %Bx[%dest] : memref<?xf64>

            // Bp[col]++
            %bp_inc = memref.load %Bp[%col] : memref<?xi64>
            %bp_inc1 = addi %bp_inc, %c1_64 : i64
            memref.store %bp_inc1, %Bp[%col]: memref<?xi64>
          }
        }

        %last_last = memref.load %Bp[%ncol] : memref<?xi64>
        memref.store %c0_64, %Bp[%ncol] : memref<?xi64>
        scf.for %col = %c0 to %ncol step %c1 {
          %temp = memref.load %Bp[%col] : memref<?xi64>
          %last = memref.load %Bp[%ncol] : memref<?xi64>
          memref.store %last, %Bp[%col] : memref<?xi64>
          memref.store %temp, %Bp[%ncol] : memref<?xi64>
        }
        memref.store %last_last, %Bp[%ncol] : memref<?xi64>

        // pymlir-skip: end

        return %output : tensor<?x?xf64, #CSR64>
      }
    """
    )


class MatrixSelect(BaseFunction):
    """
    Call signature:
      matrix_select(input: MLIRSparseTensor) -> MLIRSparseTensor
    """

    _valid_selectors = {
        # name: (needs_col, needs_val)
        "triu": (True, False),
        "tril": (True, False),
        "gt0": (False, True),
    }

    def __init__(self, selector="triu"):
        super().__init__()

        sel = selector.lower()
        if sel not in self._valid_selectors:
            raise TypeError(
                f"Invalid selector: {selector}, must be one of {list(self._valid_selectors.keys())}"
            )

        self.func_name = f"matrix_select_{sel}"
        self.selector = sel

    def get_mlir(self, *, make_private=True):
        needs_col, needs_val = self._valid_selectors[self.selector]
        return self.mlir_template.render(
            func_name=self.func_name,
            private_func=make_private,
            selector=self.selector,
            needs_col=needs_col,
            needs_val=needs_val,
        )

    mlir_template = jinja2.Template(
        """
      func {% if private_func %}private {% endif %}@{{ func_name }}(%input: tensor<?x?xf64, #CSR64>) -> tensor<?x?xf64, #CSR64> {
        %c0 = constant 0 : index
        %c1 = constant 1 : index
        %c0_64 = constant 0 : i64
        %c1_64 = constant 1 : i64
        %cf0 = constant 0.0 : f64

        // pymlir-skip: begin

        %nrow = memref.dim %input, %c0 : tensor<?x?xf64, #CSR64>
        %ncol = memref.dim %input, %c1 : tensor<?x?xf64, #CSR64>
        %Ap = sparse_tensor.pointers %input, %c1 : tensor<?x?xf64, #CSR64> to memref<?xi64>
        %Aj = sparse_tensor.indices %input, %c1 : tensor<?x?xf64, #CSR64> to memref<?xi64>
        %Ax = sparse_tensor.values %input : tensor<?x?xf64, #CSR64> to memref<?xf64>
        
        %output = call @dup_tensor(%input) : (tensor<?x?xf64, #CSR64>) -> tensor<?x?xf64, #CSR64>
        %Bp = sparse_tensor.pointers %output, %c1 : tensor<?x?xf64, #CSR64> to memref<?xi64>
        %Bj = sparse_tensor.indices %output, %c1 : tensor<?x?xf64, #CSR64> to memref<?xi64>
        %Bx = sparse_tensor.values %output : tensor<?x?xf64, #CSR64> to memref<?xf64>

        // Algorithm logic:
        // Walk thru the rows and columns of A
        // If col > row, add it to B
        //
        // Method for constructing B in CSR format:
        // 1. Bp[0] = 0
        // 2. At the start of each row, copy Bp[row] into Bp[row+1]
        // 3. When writing row X, col Y, val V
        //    a. Bj[Bp[row+1]] = Y
        //    b. Bx[Bp[row+1]] = V
        //    c. Bp[row+1] += 1

        // Bp[0] = 0
        memref.store %c0_64, %Bp[%c0] : memref<?xi64>
        scf.for %row = %c0 to %nrow step %c1 {
          // Copy Bp[row] into Bp[row+1]
          %row_plus1 = addi %row, %c1 : index
          %bp_curr_count = memref.load %Bp[%row] : memref<?xi64>
          memref.store %bp_curr_count, %Bp[%row_plus1] : memref<?xi64>

          // Read start/end positions from Ap
          %j_start_64 = memref.load %Ap[%row] : memref<?xi64>
          %j_end_64 = memref.load %Ap[%row_plus1] : memref<?xi64>
          %j_start = index_cast %j_start_64 : i64 to index
          %j_end = index_cast %j_end_64 : i64 to index
          scf.for %jj = %j_start to %j_end step %c1 {
            {% if needs_col -%}
              %col_64 = memref.load %Aj[%jj] : memref<?xi64>
              %col = index_cast %col_64 : i64 to index
            {%- endif %}
            {% if needs_val -%}
              %val = memref.load %Ax[%jj] : memref<?xf64>
            {%- endif %}

            {# When updating these, be sure to also update _valid_selectors in the class #}
            {% if selector == 'triu' -%}
              %keep = cmpi ugt, %col, %row : index
            {%- elif selector == 'tril' -%}
              %keep = cmpi ult, %col, %row : index
            {%- elif selector == 'gt0' -%}
              %keep = cmpf ogt, %val, %cf0 : f64
            {%- endif %}

            scf.if %keep {
              %bj_pos_64 = memref.load %Bp[%row_plus1] : memref<?xi64>
              %bj_pos = index_cast %bj_pos_64 : i64 to index

              {# These conditions are inverted because if not defined above, they are still needed here #}
              {% if not needs_col -%}
                %col_64 = memref.load %Aj[%jj] : memref<?xi64>
              {%- endif %}
              memref.store %col_64, %Bj[%bj_pos] : memref<?xi64>
              {% if not needs_val -%}
                %val = memref.load %Ax[%jj] : memref<?xf64>
              {%- endif %}
              memref.store %val, %Bx[%bj_pos] : memref<?xf64>

              // Increment Bp[row+1] += 1
              %bj_pos_plus1 = addi %bj_pos_64, %c1_64 : i64
              memref.store %bj_pos_plus1, %Bp[%row_plus1] : memref<?xi64>
            } else {
            }
          }
        }

        // Trim output
        %nnz_64 = memref.load %Bp[%nrow] : memref<?xi64>
        %nnz = index_cast %nnz_64 : i64 to index
        call @resize_index(%output, %c1, %nnz) : (tensor<?x?xf64, #CSR64>, index, index) -> ()
        call @resize_values(%output, %nnz) : (tensor<?x?xf64, #CSR64>, index) -> ()

        // pymlir-skip: end

        return %output : tensor<?x?xf64, #CSR64>
      }
    """
    )


class MatrixReduceToScalar(BaseFunction):
    """
    Call signature:
      matrix_reduce_to_scalar(input: MLIRSparseTensor) -> float64
    """

    _valid_aggregators = {"sum"}
    _agg_aliases = {
        "plus": "sum",
    }

    def __init__(self, aggregator="sum"):
        super().__init__()

        agg = aggregator.lower()
        agg = self._agg_aliases.get(agg, agg)
        if agg not in self._valid_aggregators:
            raise TypeError(
                f"Invalid aggregator: {aggregator}, must be one of {list(self._valid_aggregators)}"
            )

        self.func_name = f"matrix_reduce_to_scalar_{agg}"
        self.agg = agg

    def get_mlir(self, *, make_private=True):
        return self.mlir_template.render(
            func_name=self.func_name,
            private_func=make_private,
            agg=self.agg,
        )

    mlir_template = jinja2.Template(
        """
      func {% if private_func %}private {% endif %}@{{ func_name }}(%input: tensor<?x?xf64, #CSR64>) -> f64 {
        %cf0 = constant 0.0 : f64
        %c0 = constant 0 : index
        %c1 = constant 1 : index

        // pymlir-skip: begin
        
        %Ap = sparse_tensor.pointers %input, %c1 : tensor<?x?xf64, #CSR64> to memref<?xi64>
        %Ax = sparse_tensor.values %input : tensor<?x?xf64, #CSR64> to memref<?xf64>
        %nrows = memref.dim %input, %c0 : tensor<?x?xf64, #CSR64>
        %nnz_64 = memref.load %Ap[%nrows] : memref<?xi64>
        %nnz = index_cast %nnz_64 : i64 to index

        %total = scf.parallel (%pos) = (%c0) to (%nnz) step (%c1) init(%cf0) -> f64 {
          %y = memref.load %Ax[%pos] : memref<?xf64>
          scf.reduce(%y) : f64 {
            ^bb0(%lhs : f64, %rhs: f64):

            {% if agg == 'sum' -%}
              %z = addf %lhs, %rhs : f64
            {%- endif %}

              scf.reduce.return %z : f64
          }
        }
        // pymlir-skip: end

        return %total : f64
      }
    """
    )


class MatrixApply(BaseFunction):
    """
    Call signature:
      matrix_apply(input: MLIRSparseTensor, thunk: f64) -> MLIRSparseTensor
    """

    _valid_operators = {"min"}

    def __init__(self, operator="min"):
        super().__init__()

        op = operator.lower()
        if op not in self._valid_operators:
            raise TypeError(
                f"Invalid operator: {operator}, must be one of {list(self._valid_operators)}"
            )

        self.func_name = f"matrix_apply_{op}"
        self.op = op

    def get_mlir(self, *, make_private=True):
        return self.mlir_template.render(
            func_name=self.func_name,
            private_func=make_private,
            op=self.op,
        )

    mlir_template = jinja2.Template(
        """
      func {% if private_func %}private {% endif %}@{{ func_name }}(%input: tensor<?x?xf64, #CSR64>, %thunk: f64) -> tensor<?x?xf64, #CSR64> {
        %c0 = constant 0 : index
        %c1 = constant 1 : index

        // pymlir-skip: begin

        %output = call @dup_tensor(%input) : (tensor<?x?xf64, #CSR64>) -> tensor<?x?xf64, #CSR64>
        %Ap = sparse_tensor.pointers %input, %c1 : tensor<?x?xf64, #CSR64> to memref<?xi64>
        %Ax = sparse_tensor.values %input : tensor<?x?xf64, #CSR64> to memref<?xf64>
        %Bx = sparse_tensor.values %output : tensor<?x?xf64, #CSR64> to memref<?xf64>

        %nrow = memref.dim %input, %c0 : tensor<?x?xf64, #CSR64>
        %nnz_64 = memref.load %Ap[%nrow] : memref<?xi64>
        %nnz = index_cast %nnz_64 : i64 to index

        scf.parallel (%pos) = (%c0) to (%nnz) step (%c1) {
          %val = memref.load %Ax[%pos] : memref<?xf64>

          {% if op == "min" %}
          %cmp = cmpf olt, %val, %thunk : f64
          %new = select %cmp, %val, %thunk : f64
          {% endif %}

          memref.store %new, %Bx[%pos] : memref<?xf64>
        }

        // pymlir-skip: end

        return %output : tensor<?x?xf64, #CSR64>
      }
    """
    )


class MatrixMultiply(BaseFunction):
    """
    Call signature:
      If using a mask:
        matrix_multiply(
            a: MLIRSparseTensor,
            b: MLIRSparseTensor,
            mask: MLIRSparseTensor,
        ) -> MLIRSparseTensor

      If not using a mask:
        matrix_multiply(
            a: MLIRSparseTensor,
            b: MLIRSparseTensor,
        ) -> MLIRSparseTensor
    """

    _valid_semirings = {"plus_times", "plus_pair", "plus_plus"}

    def __init__(self, semiring="plus_times", mask=False):
        super().__init__()

        semi = semiring.lower()
        if semi not in self._valid_semirings:
            raise TypeError(
                f"Invalid semiring: {semiring}, must be one of {list(self._valid_semirings)}"
            )

        self.func_name = f"matrix_multiply_{semi}"
        self.semiring = semi
        self.structural_mask = mask

    def get_mlir(self, *, make_private=True):
        return self.mlir_template.render(
            func_name=self.func_name,
            private_func=make_private,
            semiring=self.semiring,
            structural_mask=self.structural_mask,
        )

    mlir_template = jinja2.Template(
        """
      func {% if private_func %}private {% endif %}@{{ func_name }}(
          %A: tensor<?x?xf64, #CSR64>, %B: tensor<?x?xf64, #CSR64>
          {%- if structural_mask -%}
          , %mask: tensor<?x?xf64, #CSR64>
          {%- endif -%}
      ) -> tensor<?x?xf64, #CSR64> {
        %c0 = constant 0 : index
        %c1 = constant 1 : index
        %c0_64 = constant 0 : i64
        %c1_64 = constant 1 : i64
        %cf0 = constant 0.0 : f64
        %cf1 = constant 1.0 : f64
        %ctrue = constant 1 : i1
        %cfalse = constant 0 : i1

        // pymlir-skip: begin

        %Ap = sparse_tensor.pointers %A, %c1 : tensor<?x?xf64, #CSR64> to memref<?xi64>
        %Aj = sparse_tensor.indices %A, %c1 : tensor<?x?xf64, #CSR64> to memref<?xi64>
        %Ax = sparse_tensor.values %A : tensor<?x?xf64, #CSR64> to memref<?xf64>
        %Bp = sparse_tensor.pointers %B, %c1 : tensor<?x?xf64, #CSR64> to memref<?xi64>
        %Bi = sparse_tensor.indices %B, %c1 : tensor<?x?xf64, #CSR64> to memref<?xi64>
        %Bx = sparse_tensor.values %B : tensor<?x?xf64, #CSR64> to memref<?xf64>

        %nrow = memref.dim %A, %c0 : tensor<?x?xf64, #CSR64>
        %ncol = memref.dim %B, %c1 : tensor<?x?xf64, #CSR64> // TODO: this should be CSC64
        %nk = memref.dim %A, %c1 : tensor<?x?xf64, #CSR64>
        %nrow_plus_one = addi %nrow, %c1 : index

        {% if structural_mask %}
        %Mp = sparse_tensor.pointers %mask, %c1 : tensor<?x?xf64, #CSR64> to memref<?xi64>
        %Mj = sparse_tensor.indices %mask, %c1 : tensor<?x?xf64, #CSR64> to memref<?xi64>
        {% endif %}

        %output = call @empty_like(%A) : (tensor<?x?xf64, #CSR64>) -> tensor<?x?xf64, #CSR64>
        call @resize_dim(%output, %c0, %nrow) : (tensor<?x?xf64, #CSR64>, index, index) -> ()
        call @resize_dim(%output, %c1, %ncol) : (tensor<?x?xf64, #CSR64>, index, index) -> ()
        call @resize_pointers(%output, %c1, %nrow_plus_one) : (tensor<?x?xf64, #CSR64>, index, index) -> ()

        %Cp = sparse_tensor.pointers %output, %c1 : tensor<?x?xf64, #CSR64> to memref<?xi64>

        // 1st pass
        //   Using nested parallel loops for each row and column,
        //   compute the number of nonzero entries per row.
        //   Store results in Cp
        scf.parallel (%row) = (%c0) to (%nrow) step (%c1) {
          %colStart_64 = memref.load %Ap[%row] : memref<?xi64>
          %row_plus1 = addi %row, %c1 : index
          %colEnd_64 = memref.load %Ap[%row_plus1] : memref<?xi64>
          %cmp_colSame = cmpi eq, %colStart_64, %colEnd_64 : i64
          %row_total = scf.if %cmp_colSame -> i64 {
            scf.yield %c0_64: i64
          } else {
            // Construct a dense array indicating valid row positions
            %colStart = index_cast %colStart_64 : i64 to index
            %colEnd = index_cast %colEnd_64 : i64 to index
            %kvec_i1 = memref.alloc(%nk) : memref<?xi1>
            linalg.fill(%kvec_i1, %cfalse) : memref<?xi1>, i1
            scf.parallel (%jj) = (%colStart) to (%colEnd) step (%c1) {
              %col_64 = memref.load %Aj[%jj] : memref<?xi64>
              %col = index_cast %col_64 : i64 to index
              memref.store %ctrue, %kvec_i1[%col] : memref<?xi1>
            }

            // Loop thru all columns; count number of resulting nonzeros in the row
            {% if structural_mask %}
            %mcol_start_64 = memref.load %Mp[%row] : memref<?xi64>
            %mcol_end_64 = memref.load %Mp[%row_plus1] : memref<?xi64>
            %mcol_start = index_cast %mcol_start_64: i64 to index
            %mcol_end = index_cast %mcol_end_64: i64 to index

            %total = scf.parallel (%mm) = (%mcol_start) to (%mcol_end) step (%c1) init (%c0_64) -> i64 {
              %col_64 = memref.load %Mj[%mm] : memref<?xi64>
              %col = index_cast %col_64 : i64 to index
              // NOTE: for valued masks, we would need to check the value and yield if false
            {% else %}
            %total = scf.parallel (%col) = (%c0) to (%ncol) step (%c1) init (%c0_64) -> i64 {
            {% endif %}

              %col_plus_one = addi %col, %c1 : index
              %rowStart_64 = memref.load %Bp[%col] : memref<?xi64>
              %rowEnd_64 = memref.load %Bp[%col_plus_one] : memref<?xi64>
              %cmp_rowSame = cmpi eq, %rowStart_64, %rowEnd_64 : i64

              // Find overlap in column indices with %kvec
              %overlap = scf.if %cmp_rowSame -> i64 {
                scf.yield %c0_64 : i64
              } else {
                // Walk thru the indices; on a match yield 1, else yield 0
                %res = scf.while (%ii_64 = %rowStart_64) : (i64) -> i64 {
                  // Check if ii >= rowEnd
                  %cmp_end_reached = cmpi uge, %ii_64, %rowEnd_64 : i64
                  %continue_search, %val_to_send = scf.if %cmp_end_reached -> (i1, i64) {
                    scf.yield %cfalse, %c0_64 : i1, i64
                  } else {
                    // Check if row has a match in kvec
                    %ii = index_cast %ii_64 : i64 to index
                    %kk_64 = memref.load %Bi[%ii] : memref<?xi64>
                    %kk = index_cast %kk_64 : i64 to index
                    %cmp_pair = memref.load %kvec_i1[%kk] : memref<?xi1>
                    %cmp_result0 = select %cmp_pair, %cfalse, %ctrue : i1
                    %cmp_result1 = select %cmp_pair, %c1_64, %ii_64 : i64
                    scf.yield %cmp_result0, %cmp_result1 : i1, i64
                  }
                  scf.condition(%continue_search) %val_to_send : i64

                } do {
                ^bb0(%ii_prev: i64):
                  %ii_next = addi %ii_prev, %c1_64 : i64
                  scf.yield %ii_next : i64
                }
                scf.yield %res : i64
              }

              scf.reduce(%overlap) : i64 {
                ^bb0(%lhs : i64, %rhs: i64):
                  %z = addi %lhs, %rhs : i64
                  scf.reduce.return %z : i64
              }
            }
            scf.yield %total : i64
          }
          memref.store %row_total, %Cp[%row] : memref<?xi64>
        }

        // 2nd pass
        //   Compute the cumsum of values in Cp to build the final Cp
        //   Then resize output indices and values
        scf.for %cs_i = %c0 to %nrow step %c1 {
          %cs_temp = memref.load %Cp[%cs_i] : memref<?xi64>
          %cumsum = memref.load %Cp[%nrow] : memref<?xi64>
          memref.store %cumsum, %Cp[%cs_i] : memref<?xi64>
          %cumsum2 = addi %cumsum, %cs_temp : i64
          memref.store %cumsum2, %Cp[%nrow] : memref<?xi64>
        }

        %nnz_64 = memref.load %Cp[%nrow] : memref<?xi64>
        %nnz = index_cast %nnz_64 : i64 to index
        call @resize_index(%output, %c1, %nnz) : (tensor<?x?xf64, #CSR64>, index, index) -> ()
        call @resize_values(%output, %nnz) : (tensor<?x?xf64, #CSR64>, index) -> ()
        %Cj = sparse_tensor.indices %output, %c1 : tensor<?x?xf64, #CSR64> to memref<?xi64>
        %Cx = sparse_tensor.values %output : tensor<?x?xf64, #CSR64> to memref<?xf64>

        // 3rd pass
        //   In parallel over the rows,
        //   compute the nonzero columns and associated values.
        //   Store in Cj and Cx

        scf.parallel (%row) = (%c0) to (%nrow) step (%c1) {
          %row_plus1 = addi %row, %c1 : index
          %cpStart_64 = memref.load %Cp[%row] : memref<?xi64>
          %cpEnd_64 = memref.load %Cp[%row_plus1] : memref<?xi64>
          %cmp_cpDifferent = cmpi ne, %cpStart_64, %cpEnd_64 : i64
          scf.if %cmp_cpDifferent {
            %base_index_64 = memref.load %Cp[%row] : memref<?xi64>
            %base_index = index_cast %base_index_64 : i64 to index

            // Construct a dense array of row values
            %colStart_64 = memref.load %Ap[%row] : memref<?xi64>
            %colEnd_64 = memref.load %Ap[%row_plus1] : memref<?xi64>
            %colStart = index_cast %colStart_64 : i64 to index
            %colEnd = index_cast %colEnd_64 : i64 to index
            %kvec = memref.alloc(%nk) : memref<?xf64>
            %kvec_i1 = memref.alloc(%nk) : memref<?xi1>
            linalg.fill(%kvec_i1, %cfalse) : memref<?xi1>, i1
            scf.parallel (%jj) = (%colStart) to (%colEnd) step (%c1) {
              %col_64 = memref.load %Aj[%jj] : memref<?xi64>
              %col = index_cast %col_64 : i64 to index
              memref.store %ctrue, %kvec_i1[%col] : memref<?xi1>
              %val = memref.load %Ax[%jj] : memref<?xf64>
              memref.store %val, %kvec[%col] : memref<?xf64>
            }

            {% if structural_mask %}
            %mcol_start_64 = memref.load %Mp[%row] : memref<?xi64>
            %mcol_end_64 = memref.load %Mp[%row_plus1] : memref<?xi64>
            %mcol_start = index_cast %mcol_start_64: i64 to index
            %mcol_end = index_cast %mcol_end_64: i64 to index

            scf.for %mm = %mcol_start to %mcol_end step %c1 iter_args(%offset = %c0) -> index {
              %col_64 = memref.load %Mj[%mm] : memref<?xi64>
              %col = index_cast %col_64 : i64 to index
              // NOTE: for valued masks, we would need to check the value and yield if false
            {% else %}
            scf.for %col = %c0 to %ncol step %c1 iter_args(%offset = %c0) -> index {
              %col_64 = index_cast %col : index to i64
            {% endif %}

              %col_plus1 = addi %col, %c1 : index
              %istart_64 = memref.load %Bp[%col] : memref<?xi64>
              %iend_64 = memref.load %Bp[%col_plus1] : memref<?xi64>
              %istart = index_cast %istart_64 : i64 to index
              %iend = index_cast %iend_64 : i64 to index

              %total, %not_empty = scf.for %ii = %istart to %iend step %c1 iter_args(%curr = %cf0, %alive = %cfalse) -> (f64, i1) {
                // Figure out if there is a match
                %kk_64 = memref.load %Bi[%ii] : memref<?xi64>
                %kk = index_cast %kk_64 : i64 to index
                %cmp_pair = memref.load %kvec_i1[%kk] : memref<?xi1>
                %new_curr, %new_alive = scf.if %cmp_pair -> (f64, i1) {

                {% if semiring == "plus_pair" -%}
                  %new = addf %curr, %cf1 : f64
                {% else %}
                  %a_val = memref.load %kvec[%kk]: memref <?xf64>
                  %b_val = memref.load %Bx[%ii]: memref <?xf64>
                  {% if semiring == "plus_times" -%}
                    %val = mulf %a_val, %b_val : f64
                    %new = addf %curr, %val : f64
                  {%- elif semiring == "plus_plus" -%}
                    %val = addf %a_val, %b_val : f64
                    %new = addf %curr, %val : f64
                  {% endif %}
                {%- endif %}

                  scf.yield %new, %ctrue : f64, i1
                } else {
                  scf.yield %curr, %alive : f64, i1
                }

                scf.yield %new_curr, %new_alive : f64, i1
              }

              %new_offset = scf.if %not_empty -> index {
                // Store total in Cx
                %cj_pos = addi %base_index, %offset : index
                memref.store %col_64, %Cj[%cj_pos] : memref<?xi64>
                memref.store %total, %Cx[%cj_pos] : memref<?xf64>
                // Increment offset
                %offset_plus_one = addi %offset, %c1 : index
                scf.yield %offset_plus_one : index
              } else {
                scf.yield %offset : index
              }
              scf.yield %new_offset : index
            }
          }
        }

        // pymlir-skip: end

        return %output : tensor<?x?xf64, #CSR64>
      }
    """
    )
