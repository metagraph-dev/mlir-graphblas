"""
Various functions written in MLIR which implement pieces of GraphBLAS or other utilities
"""
import jinja2
import numpy as np
from .engine import MlirJitEngine
from .sparse_utils import MLIRSparseTensor


class MLIRCompileError(Exception):
    pass


_default_engine = MlirJitEngine()

_standard_passes = [
    "--test-sparsification=lower",
    "--linalg-bufferize",
    "--convert-scf-to-std",
    "--func-bufferize",
    "--tensor-bufferize",
    "--tensor-constant-bufferize",
    "--finalizing-bufferize",
    "--convert-linalg-to-loops",
    "--convert-scf-to-std",
    "--convert-std-to-llvm",
]


class BaseFunction:
    func_name = None

    def __init__(self):
        pass

    def get_mlir(self, *, make_private=True):
        """
        `make_private` is used to indicate whether the function should be private for fusion
        or public for standalone calling in the `compile` method.

        `module_wrapper` indicates whether to wrap the function text in a MLIR module
        with external function declarations for common helper functions.
        """
        raise NotImplementedError()

    MODULE_WRAPPER_TEXT = jinja2.Template(
        """\
module  {
    func private @empty(!llvm.ptr<i8>, index) -> !llvm.ptr<i8>
    func private @empty_like(!llvm.ptr<i8>) -> !llvm.ptr<i8>
    func private @dup_tensor(!llvm.ptr<i8>) -> !llvm.ptr<i8>

    func private @resize_pointers(!llvm.ptr<i8>, index, index) -> ()
    func private @resize_index(!llvm.ptr<i8>, index, index) -> ()
    func private @resize_values(!llvm.ptr<i8>, index) -> ()
    func private @resize_dim(!llvm.ptr<i8>, index, index) -> ()

    func private @sparsePointers64(!llvm.ptr<i8>, index) -> memref<?xindex>
    func private @sparseIndices64(!llvm.ptr<i8>, index) -> memref<?xindex>
    func private @sparseValuesF64(!llvm.ptr<i8>) -> memref<?xf64>
    func private @sparseDimSize(!llvm.ptr<i8>, index) -> index

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

        if self.func_name is None:
            raise MLIRCompileError(
                f"{self.__class__.__name__} does not define func_name"
            )

        # Force recompilation if name is already registered
        if self.func_name in engine.name_to_callable:
            del engine.name_to_callable[self.func_name]

        mlir = self.get_mlir_module()

        engine.add(mlir, passes)
        func = engine[self.func_name]
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

    def compile(self, engine=None, passes=None):
        func = super().compile(engine, passes)

        def transpose(input: MLIRSparseTensor) -> MLIRSparseTensor:
            ptr = func(input)
            tensor = MLIRSparseTensor.from_raw_pointer(
                ptr, input.pointer_dtype, input.index_dtype, input.value_dtype
            )
            return tensor

        return transpose

    mlir_template = jinja2.Template(
        """
      func {% if private_func %}private {% endif %}@{{ func_name }}(%input: !llvm.ptr<i8>) -> !llvm.ptr<i8> {
        // Attempting to implement identical code as in scipy
        // https://github.com/scipy/scipy/blob/3b36a574dc657d1ca116f6e230be694f3de31afc/scipy/sparse/sparsetools/csr.h
        // function csr_tocsc

        %c0 = constant 0 : index
        %c1 = constant 1 : index

        // pymlir-skip: begin

        %Ap = call @sparsePointers64(%input, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %Aj = call @sparseIndices64(%input, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %Ax = call @sparseValuesF64(%input) : (!llvm.ptr<i8>) -> memref<?xf64>

        %nrow = call @sparseDimSize(%input, %c0) : (!llvm.ptr<i8>, index) -> index
        %ncol = call @sparseDimSize(%input, %c1) : (!llvm.ptr<i8>, index) -> index
        %ncol_plus_one = addi %ncol, %c1 : index
        %nnz = memref.load %Ap[%nrow] : memref<?xindex>

        %output = call @empty_like(%input) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
        {% if swap_sizes %}
        call @resize_dim(%output, %c0, %ncol) : (!llvm.ptr<i8>, index, index) -> ()
        call @resize_dim(%output, %c1, %nrow) : (!llvm.ptr<i8>, index, index) -> ()
        {% else %}
        call @resize_dim(%output, %c0, %nrow) : (!llvm.ptr<i8>, index, index) -> ()
        call @resize_dim(%output, %c1, %ncol) : (!llvm.ptr<i8>, index, index) -> ()
        {% endif %}
        call @resize_pointers(%output, %c1, %ncol_plus_one) : (!llvm.ptr<i8>, index, index) -> ()
        call @resize_index(%output, %c1, %nnz) : (!llvm.ptr<i8>, index, index) -> ()
        call @resize_values(%output, %nnz) : (!llvm.ptr<i8>, index) -> ()
        
        %Bp = call @sparsePointers64(%output, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %Bi = call @sparseIndices64(%output, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %Bx = call @sparseValuesF64(%output) : (!llvm.ptr<i8>) -> memref<?xf64>

        // compute number of non-zero entries per column of A
        scf.for %arg2 = %c0 to %ncol step %c1 {
          memref.store %c0, %Bp[%arg2] : memref<?xindex>
        }
        scf.for %n = %c0 to %nnz step %c1 {
          %colA = memref.load %Aj[%n] : memref<?xindex>
          %colB = memref.load %Bp[%colA] : memref<?xindex>
          %colB1 = addi %colB, %c1 : index
          memref.store %colB1, %Bp[%colA] : memref<?xindex>
        }

        // cumsum the nnz per column to get Bp
        memref.store %c0, %Bp[%ncol] : memref<?xindex>
        scf.for %col = %c0 to %ncol step %c1 {
          %temp = memref.load %Bp[%col] : memref<?xindex>
          %cumsum = memref.load %Bp[%ncol] : memref<?xindex>
          memref.store %cumsum, %Bp[%col] : memref<?xindex>
          %cumsum2 = addi %cumsum, %temp : index
          memref.store %cumsum2, %Bp[%ncol] : memref<?xindex>
        }

        scf.for %row = %c0 to %nrow step %c1 {
          %j_start = memref.load %Ap[%row] : memref<?xindex>
          %row_plus1 = addi %row, %c1 : index
          %j_end = memref.load %Ap[%row_plus1] : memref<?xindex>
          scf.for %jj = %j_start to %j_end step %c1 {
            %col = memref.load %Aj[%jj] : memref<?xindex>
            %dest = memref.load %Bp[%col] : memref<?xindex>

            memref.store %row, %Bi[%dest] : memref<?xindex>
            %axjj = memref.load %Ax[%jj] : memref<?xf64>
            memref.store %axjj, %Bx[%dest] : memref<?xf64>

            // Bp[col]++
            %bp_inc = memref.load %Bp[%col] : memref<?xindex>
            %bp_inc1 = addi %bp_inc, %c1 : index
            memref.store %bp_inc1, %Bp[%col]: memref<?xindex>
          }
        }

        %last_last = memref.load %Bp[%ncol] : memref<?xindex>
        memref.store %c0, %Bp[%ncol] : memref<?xindex>
        scf.for %col = %c0 to %ncol step %c1 {
          %temp = memref.load %Bp[%col] : memref<?xindex>
          %last = memref.load %Bp[%ncol] : memref<?xindex>
          memref.store %last, %Bp[%col] : memref<?xindex>
          memref.store %temp, %Bp[%ncol] : memref<?xindex>
        }
        memref.store %last_last, %Bp[%ncol] : memref<?xindex>

        // pymlir-skip: end

        return %output : !llvm.ptr<i8>
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

    def compile(self, engine=None, passes=None):
        func = super().compile(engine, passes)

        def matrix_select(input: MLIRSparseTensor) -> MLIRSparseTensor:
            ptr = func(input)
            tensor = MLIRSparseTensor.from_raw_pointer(
                ptr, input.pointer_dtype, input.index_dtype, input.value_dtype
            )
            return tensor

        return matrix_select

    mlir_template = jinja2.Template(
        """
      func {% if private_func %}private {% endif %}@{{ func_name }}(%input: !llvm.ptr<i8>) -> !llvm.ptr<i8> {
        %c0 = constant 0 : index
        %c1 = constant 1 : index
        %cf0 = constant 0.0 : f64

        // pymlir-skip: begin

        %nrow = call @sparseDimSize(%input, %c0) : (!llvm.ptr<i8>, index) -> index
        %ncol = call @sparseDimSize(%input, %c1) : (!llvm.ptr<i8>, index) -> index
        %Ap = call @sparsePointers64(%input, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %Aj = call @sparseIndices64(%input, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %Ax = call @sparseValuesF64(%input) : (!llvm.ptr<i8>) -> memref<?xf64>
        
        %output = call @dup_tensor(%input) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
        %Bp = call @sparsePointers64(%output, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %Bj = call @sparseIndices64(%output, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %Bx = call @sparseValuesF64(%output) : (!llvm.ptr<i8>) -> memref<?xf64>

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
        memref.store %c0, %Bp[%c0] : memref<?xindex>
        scf.for %row = %c0 to %nrow step %c1 {
          // Copy Bp[row] into Bp[row+1]
          %row_plus1 = addi %row, %c1 : index
          %bp_curr_count = memref.load %Bp[%row] : memref<?xindex>
          memref.store %bp_curr_count, %Bp[%row_plus1] : memref<?xindex>

          // Read start/end positions from Ap
          %j_start = memref.load %Ap[%row] : memref<?xindex>
          %j_end = memref.load %Ap[%row_plus1] : memref<?xindex>
          scf.for %jj = %j_start to %j_end step %c1 {
            {% if needs_col -%}
              %col = memref.load %Aj[%jj] : memref<?xindex>
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
              %bj_pos = memref.load %Bp[%row_plus1] : memref<?xindex>

              {# These conditions are inverted because if not defined above, they are still needed here #}
              {% if not needs_col -%}
                %col = memref.load %Aj[%jj] : memref<?xindex>
              {%- endif %}
              memref.store %col, %Bj[%bj_pos] : memref<?xindex>
              {% if not needs_val -%}
                %val = memref.load %Ax[%jj] : memref<?xf64>
              {%- endif %}
              memref.store %val, %Bx[%bj_pos] : memref<?xf64>

              // Increment Bp[row+1] += 1
              %bj_pos_plus1 = addi %bj_pos, %c1 : index
              memref.store %bj_pos_plus1, %Bp[%row_plus1] : memref<?xindex>
            } else {
            }
          }
        }

        // Trim output
        %nnz = memref.load %Bp[%nrow] : memref<?xindex>
        call @resize_index(%output, %c1, %nnz) : (!llvm.ptr<i8>, index, index) -> ()
        call @resize_values(%output, %nnz) : (!llvm.ptr<i8>, index) -> ()

        // pymlir-skip: end

        return %output : !llvm.ptr<i8>
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
      func {% if private_func %}private {% endif %}@{{ func_name }}(%input: !llvm.ptr<i8>) -> f64 {
        %cf0 = constant 0.0 : f64
        %c0 = constant 0 : index
        %c1 = constant 1 : index

        // pymlir-skip: begin

        %0 = call @sparseDimSize(%input, %c0) : (!llvm.ptr<i8>, index) -> index
        %1 = call @sparsePointers64(%input, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %2 = call @sparseIndices64(%input, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %3 = call @sparseValuesF64(%input) : (!llvm.ptr<i8>) -> memref<?xf64>
        //%4 = memref.buffer_cast %cst : memref<f64>
        %5 = memref.alloc() : memref<f64>
        memref.store %cf0, %5[] : memref<f64>
        //linalg.copy(%4, %5) : memref<f64>, memref<f64>
        scf.for %arg1 = %c0 to %0 step %c1 {
          %8 = memref.load %1[%arg1] : memref<?xindex>
          %9 = addi %arg1, %c1 : index
          %10 = memref.load %1[%9] : memref<?xindex>
          %11 = memref.load %5[] : memref<f64>
          %12 = scf.for %arg2 = %8 to %10 step %c1 iter_args(%x = %11) -> (f64) {
            %y = memref.load %3[%arg2] : memref<?xf64>

            {% if agg == 'sum' -%}
              %z = addf %x, %y : f64
            {%- endif %}

            scf.yield %z : f64
          }
          memref.store %12, %5[] : memref<f64>
        }
        %7 = memref.load %5[] : memref<f64>

        // pymlir-skip: end

        return %7 : f64
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

    def compile(self, engine=None, passes=None):
        func = super().compile(engine, passes)

        def matrix_apply(input: MLIRSparseTensor, thunk) -> MLIRSparseTensor:
            ptr = func(input, thunk)
            tensor = MLIRSparseTensor.from_raw_pointer(
                ptr, input.pointer_dtype, input.index_dtype, input.value_dtype
            )
            return tensor

        return matrix_apply

    mlir_template = jinja2.Template(
        """
      func {% if private_func %}private {% endif %}@{{ func_name }}(%input: !llvm.ptr<i8>, %thunk: f64) -> !llvm.ptr<i8> {
        %c0 = constant 0 : index
        %c1 = constant 1 : index
        %cf0 = constant 0.0 : f64

        // pymlir-skip: begin

        %nrow = call @sparseDimSize(%input, %c0) : (!llvm.ptr<i8>, index) -> index
        %ncol = call @sparseDimSize(%input, %c1) : (!llvm.ptr<i8>, index) -> index
        %Ap = call @sparsePointers64(%input, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %Aj = call @sparseIndices64(%input, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %Ax = call @sparseValuesF64(%input) : (!llvm.ptr<i8>) -> memref<?xf64>
        
        %output = call @dup_tensor(%input) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
        %Bp = call @sparsePointers64(%output, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %Bj = call @sparseIndices64(%output, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %Bx = call @sparseValuesF64(%output) : (!llvm.ptr<i8>) -> memref<?xf64>

        memref.store %c0, %Bp[%c0] : memref<?xindex>
        scf.for %row = %c0 to %nrow step %c1 {
          // Read start/end positions from Ap
          %row_plus1 = addi %row, %c1 : index
          %j_start = memref.load %Ap[%row] : memref<?xindex>
          %j_end = memref.load %Ap[%row_plus1] : memref<?xindex>
          memref.store %j_end, %Bp[%row_plus1] : memref<?xindex>

          scf.for %jj = %j_start to %j_end step %c1 {
            %col = memref.load %Aj[%jj] : memref<?xindex>
            %val = memref.load %Ax[%jj] : memref<?xf64>

            memref.store %col, %Bj[%jj] : memref<?xindex>

            {% if op == "min" %}
            %cmp = cmpf olt, %val, %thunk : f64
            %new = select %cmp, %val, %thunk : f64
            {% endif %}

            memref.store %new, %Bx[%jj] : memref<?xf64>
          }
        }

        // pymlir-skip: end

        return %output : !llvm.ptr<i8>
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

    def compile(self, engine=None, passes=None):
        func = super().compile(engine, passes)

        if self.structural_mask:

            def matrix_multiply(
                a: MLIRSparseTensor, b: MLIRSparseTensor, mask: MLIRSparseTensor
            ) -> MLIRSparseTensor:
                ptr = func(a, b, mask)
                tensor = MLIRSparseTensor.from_raw_pointer(
                    ptr, a.pointer_dtype, a.index_dtype, a.value_dtype
                )
                return tensor

        else:

            def matrix_multiply(
                a: MLIRSparseTensor, b: MLIRSparseTensor
            ) -> MLIRSparseTensor:
                ptr = func(a, b)
                tensor = MLIRSparseTensor.from_raw_pointer(
                    ptr, a.pointer_dtype, a.index_dtype, a.value_dtype
                )
                return tensor

        return matrix_multiply

    mlir_template = jinja2.Template(
        """
      func {% if private_func %}private {% endif %}@{{ func_name }}(
          %A: !llvm.ptr<i8>, %B: !llvm.ptr<i8>
          {%- if structural_mask -%}
          , %mask: !llvm.ptr<i8>
          {%- endif -%}
      ) -> !llvm.ptr<i8> {
        %c0 = constant 0 : index
        %c1 = constant 1 : index
        %c10 = constant 10 : index
        %cf0 = constant 0.0 : f64
        %cf1 = constant 1.0 : f64
        %ctrue = constant 1 : i1
        %cfalse = constant 0 : i1

        // pymlir-skip: begin

        %Ap = call @sparsePointers64(%A, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %Aj = call @sparseIndices64(%A, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %Ax = call @sparseValuesF64(%A) : (!llvm.ptr<i8>) -> memref<?xf64>
        %Bp = call @sparsePointers64(%B, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %Bi = call @sparseIndices64(%B, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %Bx = call @sparseValuesF64(%B) : (!llvm.ptr<i8>) -> memref<?xf64>

        %nrow = call @sparseDimSize(%A, %c0) : (!llvm.ptr<i8>, index) -> index
        %ncol = call @sparseDimSize(%B, %c1) : (!llvm.ptr<i8>, index) -> index
        %nrow_plus_one = addi %nrow, %c1 : index
        
        {% if structural_mask %}
        %Mp = call @sparsePointers64(%mask, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %Mj = call @sparseIndices64(%mask, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %nnz_init = memref.load %Mp[%nrow] : memref<?xindex>
        {% else %}
        %nnz = memref.load %Ap[%nrow] : memref<?xindex>
        %nnz_init = muli %nnz, %c10 : index
        {% endif %}

        %output = call @empty_like(%A) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
        call @resize_dim(%output, %c0, %nrow) : (!llvm.ptr<i8>, index, index) -> ()
        call @resize_dim(%output, %c1, %ncol) : (!llvm.ptr<i8>, index, index) -> ()
        call @resize_pointers(%output, %c1, %nrow_plus_one) : (!llvm.ptr<i8>, index, index) -> ()
        call @resize_index(%output, %c1, %nnz_init) : (!llvm.ptr<i8>, index, index) -> ()
        call @resize_values(%output, %nnz_init) : (!llvm.ptr<i8>, index) -> ()

        %Cp = call @sparsePointers64(%output, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %Cj = call @sparseIndices64(%output, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %Cx = call @sparseValuesF64(%output) : (!llvm.ptr<i8>) -> memref<?xf64>
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
        memref.store %c0, %Cp[%c0] : memref<?xindex>
        scf.for %row = %c0 to %nrow step %c1 {
          // Copy Cp[row] into Cp[row+1]
          %row_plus1 = addi %row, %c1 : index
          %cp_curr_count = memref.load %Cp[%row] : memref<?xindex>
          memref.store %cp_curr_count, %Cp[%row_plus1] : memref<?xindex>

          {% if structural_mask %}
          %mcol_start = memref.load %Mp[%row] : memref<?xindex>
          %mcol_end = memref.load %Mp[%row_plus1] : memref<?xindex>

          scf.for %mm = %mcol_start to %mcol_end step %c1 {
            %col = memref.load %Mj[%mm] : memref<?xindex>
            // NOTE: for valued masks, we would need to check the value and yield if false
          {% else %}
          scf.for %col = %c0 to %ncol step %c1 {
          {% endif %}

            // Iterate over row in A as ka, col in B as kb
            // Find matches, ignore otherwise
            %col_plus1 = addi %col, %c1 : index
            %jstart = memref.load %Ap[%row] : memref<?xindex>
            %jend = memref.load %Ap[%row_plus1] : memref<?xindex>
            %istart = memref.load %Bp[%col] : memref<?xindex>
            %iend = memref.load %Bp[%col_plus1] : memref<?xindex>

            memref.store %cf0, %accumulator[] : memref<f64>
            memref.store %cfalse, %not_empty[] : memref<i1>

            scf.while (%jj = %jstart, %ii = %istart) : (index, index) -> (index, index) {
              %jj_not_done = cmpi ult, %jj, %jend : index
              %ii_not_done = cmpi ult, %ii, %iend : index
              %cond = and %jj_not_done, %ii_not_done : i1
              scf.condition(%cond) %jj, %ii : index, index
            } do {
            ^bb0(%jj: index, %ii: index):  // no predecessors
              %kj = memref.load %Aj[%jj] : memref<?xindex>
              %ki = memref.load %Bi[%ii] : memref<?xindex>
              %ks_match = cmpi eq, %kj, %ki : index
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
              %16 = cmpi ult, %ki, %kj : index
              %k_lowest = select %16, %ki, %kj : index
              %21 = cmpi eq, %kj, %k_lowest : index
              %jj_choice = select %21, %jj_plus1, %jj : index
              %24 = cmpi eq, %ki, %k_lowest : index
              %ii_choice = select %24, %ii_plus1, %ii : index
              scf.yield %jj_choice, %ii_choice : index, index
            }

            %is_not_empty = memref.load %not_empty[] : memref<i1>
            scf.if %is_not_empty {
              // Store accumulated value
              %cj_pos = memref.load %Cp[%row_plus1] : memref<?xindex>
              memref.store %col, %Cj[%cj_pos] : memref<?xindex>
              %accumulated = memref.load %accumulator[] : memref<f64>
              memref.store %accumulated, %Cx[%cj_pos] : memref<?xf64>
              // Increment Cp[row+1] += 1
              %cj_pos_plus1 = addi %cj_pos, %c1 : index
              memref.store %cj_pos_plus1, %Cp[%row_plus1] : memref<?xindex>
            } else {
            }
          }
        }

        // Trim output
        %nnz_final = memref.load %Cp[%nrow] : memref<?xindex>
        call @resize_index(%output, %c1, %nnz_final) : (!llvm.ptr<i8>, index, index) -> ()
        call @resize_values(%output, %nnz_final) : (!llvm.ptr<i8>, index) -> ()

        // pymlir-skip: end

        return %output : !llvm.ptr<i8>
      }
    """
    )
