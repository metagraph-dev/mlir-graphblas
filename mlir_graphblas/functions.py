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
        %c10 = constant 10 : index
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
        %ncol = memref.dim %B, %c1 : tensor<?x?xf64, #CSR64>
        %nrow_plus_one = addi %nrow, %c1 : index
        
        {% if structural_mask %}
        %Mp = sparse_tensor.pointers %mask, %c1 : tensor<?x?xf64, #CSR64> to memref<?xi64>
        %Mj = sparse_tensor.indices %mask, %c1 : tensor<?x?xf64, #CSR64> to memref<?xi64>
        %nnz_init_64 = memref.load %Mp[%nrow] : memref<?xi64>
        %nnz_init = index_cast %nnz_init_64 : i64 to index
        {% else %}
        %nnz_64 = memref.load %Ap[%nrow] : memref<?xi64>
        %nnz = index_cast %nnz_64 : i64 to index
        %nnz_init = muli %nnz, %c10 : index
        {% endif %}

        %output = call @empty_like(%A) : (tensor<?x?xf64, #CSR64>) -> tensor<?x?xf64, #CSR64>
        call @resize_dim(%output, %c0, %nrow) : (tensor<?x?xf64, #CSR64>, index, index) -> ()
        call @resize_dim(%output, %c1, %ncol) : (tensor<?x?xf64, #CSR64>, index, index) -> ()
        call @resize_pointers(%output, %c1, %nrow_plus_one) : (tensor<?x?xf64, #CSR64>, index, index) -> ()
        call @resize_index(%output, %c1, %nnz_init) : (tensor<?x?xf64, #CSR64>, index, index) -> ()
        call @resize_values(%output, %nnz_init) : (tensor<?x?xf64, #CSR64>, index) -> ()

        %Cp = sparse_tensor.pointers %output, %c1 : tensor<?x?xf64, #CSR64> to memref<?xi64>
        %Cj = sparse_tensor.indices %output, %c1 : tensor<?x?xf64, #CSR64> to memref<?xi64>
        %Cx = sparse_tensor.values %output : tensor<?x?xf64, #CSR64> to memref<?xf64>
        %accumulator = memref.alloc() : memref<f64>
        %not_empty = memref.alloc() : memref<i1>
        %cur_size = memref.alloc() : memref<index>
        memref.store %nnz_init, %cur_size[]: memref<index>

        // Method for constructing C in CSR format:
        // 1. Cp[0] = 0
        // 2. At the start of each row, copy Cp[row] into Cp[row+1]
        // 3. When writing row X, col Y, val V
        //    a. Cj[Cp[row+1]] = Y
        //    b. Cx[Cp[row+1]] = V
        //    c. Cp[row+1] += 1

        // Cp[0] = 0
        memref.store %c0_64, %Cp[%c0] : memref<?xi64>
        scf.for %row = %c0 to %nrow step %c1 {
          // Copy Cp[row] into Cp[row+1]
          %row_plus1 = addi %row, %c1 : index
          %cp_curr_count = memref.load %Cp[%row] : memref<?xi64>
          memref.store %cp_curr_count, %Cp[%row_plus1] : memref<?xi64>

          {% if structural_mask %}
          %mcol_start_64 = memref.load %Mp[%row] : memref<?xi64>
          %mcol_end_64 = memref.load %Mp[%row_plus1] : memref<?xi64>
          %mcol_start = index_cast %mcol_start_64: i64 to index
          %mcol_end = index_cast %mcol_end_64: i64 to index

          scf.for %mm = %mcol_start to %mcol_end step %c1 {
            %col_64 = memref.load %Mj[%mm] : memref<?xi64>
            %col = index_cast %col_64 : i64 to index
            // NOTE: for valued masks, we would need to check the value and yield if false
          {% else %}
          scf.for %col = %c0 to %ncol step %c1 {
            %col_64 = index_cast %col : index to i64
          {% endif %}

            // Iterate over row in A as ka, col in B as kb
            // Find matches, ignore otherwise
            %col_plus1 = addi %col, %c1 : index
            %jstart_64 = memref.load %Ap[%row] : memref<?xi64>
            %jend_64 = memref.load %Ap[%row_plus1] : memref<?xi64>
            %istart_64 = memref.load %Bp[%col] : memref<?xi64>
            %iend_64 = memref.load %Bp[%col_plus1] : memref<?xi64>
            %jstart = index_cast %jstart_64 : i64 to index
            %jend = index_cast %jend_64 : i64 to index
            %istart = index_cast %istart_64 : i64 to index
            %iend = index_cast %iend_64 : i64 to index

            memref.store %cf0, %accumulator[] : memref<f64>
            memref.store %cfalse, %not_empty[] : memref<i1>

            scf.while (%jj = %jstart, %ii = %istart) : (index, index) -> (index, index) {
              %jj_not_done = cmpi ult, %jj, %jend : index
              %ii_not_done = cmpi ult, %ii, %iend : index
              %cond = and %jj_not_done, %ii_not_done : i1
              scf.condition(%cond) %jj, %ii : index, index
            } do {
            ^bb0(%jj: index, %ii: index):  // no predecessors
              %kj = memref.load %Aj[%jj] : memref<?xi64>
              %ki = memref.load %Bi[%ii] : memref<?xi64>
              %ks_match = cmpi eq, %kj, %ki : i64
              scf.if %ks_match {
                %existing = memref.load %accumulator[] : memref<f64>

                {% if semiring == "plus_pair" -%}
                  %new = addf %existing, %cf1 : f64
                {% else %}
                  %a_val = memref.load %Ax[%jj]: memref <?xf64>
                  %b_val = memref.load %Bx[%ii]: memref <?xf64>
                  {% if semiring == "plus_times" -%}
                    %val = mulf %a_val, %b_val : f64
                    %new = addf %existing, %val : f64
                  {%- elif semiring == "plus_plus" -%}
                    %val = addf %a_val, %b_val : f64
                    %new = addf %existing, %val : f64
                  {% endif %}
                {%- endif %}

                memref.store %new, %accumulator[] : memref<f64>
                memref.store %ctrue, %not_empty[] : memref<i1>
              } else {
              }
              // Increment lowest k index (or both if k indices match)
              %jj_plus1 = addi %jj, %c1 : index
              %ii_plus1 = addi %ii, %c1 : index
              %16 = cmpi ult, %ki, %kj : i64
              %k_lowest = select %16, %ki, %kj : i64
              %21 = cmpi eq, %kj, %k_lowest : i64
              %jj_choice = select %21, %jj_plus1, %jj : index
              %24 = cmpi eq, %ki, %k_lowest : i64
              %ii_choice = select %24, %ii_plus1, %ii : index
              scf.yield %jj_choice, %ii_choice : index, index
            }

            %is_not_empty = memref.load %not_empty[] : memref<i1>
            scf.if %is_not_empty {
              // Store accumulated value
              %cj_pos_64 = memref.load %Cp[%row_plus1] : memref<?xi64>
              %cj_pos = index_cast %cj_pos_64 : i64 to index
              memref.store %col_64, %Cj[%cj_pos] : memref<?xi64>
              %accumulated = memref.load %accumulator[] : memref<f64>
              memref.store %accumulated, %Cx[%cj_pos] : memref<?xf64>
              // Increment Cp[row+1] += 1
              %cj_pos_plus1 = addi %cj_pos_64, %c1_64 : i64
              memref.store %cj_pos_plus1, %Cp[%row_plus1] : memref<?xi64>
            } else {
            }
          }
        }

        // Trim output
        %nnz_final_64 = memref.load %Cp[%nrow] : memref<?xi64>
        %nnz_final = index_cast %nnz_final_64 : i64 to index
        call @resize_index(%output, %c1, %nnz_final) : (tensor<?x?xf64, #CSR64>, index, index) -> ()
        call @resize_values(%output, %nnz_final) : (tensor<?x?xf64, #CSR64>, index) -> ()

        // pymlir-skip: end

        return %output : tensor<?x?xf64, #CSR64>
      }
    """
    )
