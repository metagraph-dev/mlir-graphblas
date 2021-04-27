"""
Various functions written in MLIR which implement pieces of GraphBLAS or other utilities
"""
import jinja2
import numpy as np
from string import Template
from typing import Optional
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


def create_empty_matrix(nrows, ncols, nnz) -> MLIRSparseTensor:
    indices = np.empty((nnz, 2), dtype=np.uint64)
    indices[:, 0] = np.arange(nnz) // nrows
    indices[:, 1] = np.arange(nnz) % nrows
    values = np.zeros((nnz,), dtype=np.float64)
    sizes = np.array([nrows, ncols], dtype=np.uint64)
    sparsity = np.array([False, True], dtype=np.bool8)
    return MLIRSparseTensor(indices, values, sizes, sparsity)


def matrix_reduce(input: MLIRSparseTensor, agg="sum", *, engine=None) -> float:
    if engine is None:
        engine = _default_engine

    if agg.lower() in ("sum", "plus"):
        func_name = "matrix_reduce_sum"
        agg_func = "addf %x, %y : f64"
    else:
        raise TypeError(f"Unsupported agg: {agg}")
    mlir_text = matrix_reduce_mlir_template.substitute(
        FUNC_NAME=func_name, AGG_FUNC=agg_func
    )

    if func_name not in engine.name_to_callable:
        engine.add(mlir_text, _standard_passes)
    func = getattr(engine, func_name)
    return float(func(input))


def mxm(
    output: MLIRSparseTensor,
    a: MLIRSparseTensor,
    b: MLIRSparseTensor,
    mask: Optional[MLIRSparseTensor] = None,
    semiring="plus_times",
    *,
    engine=None,
):
    """
    output, a, and mask must be in CSR format
    b must be in CSC format
    """
    if engine is None:
        engine = _default_engine

    sr = semiring.lower()

    if mask is None:
        func_name = f"mxm_{sr}"
    else:
        func_name = f"structural_masked_mxm_{sr}"

    if sr == "plus_times":
        op = """
        %a_val = load %Ax[%jj]: memref <?xf64>
        %b_val = load %Bx[%ii]: memref <?xf64>
        %val = mulf %a_val, %b_val : f64
        %new = addf %existing, %val : f64
        """
    elif sr == "plus_pair":
        op = """
        %new = addf %existing, %cf1 : f64
        """
    elif sr == "plus_plus":
        op = """
        %a_val = load %Ax[%jj]: memref <?xf64>
        %b_val = load %Bx[%ii]: memref <?xf64>
        %val = addf %a_val, %b_val : f64
        %new = addf %existing, %val : f64
        """
    else:
        raise TypeError(f"Unsupported semiring: {semiring}")
    mlir_text = mxm_mlir_template.render(
        FUNC_NAME=func_name, OP=op, structural_mask=(mask is not None)
    )

    if func_name not in engine.name_to_callable:
        engine.add(mlir_text, _standard_passes)
    func = getattr(engine, func_name)
    if mask is None:
        return func(output, a, b)
    else:
        return func(output, a, b, mask)


def matrix_apply(
    output: MLIRSparseTensor,
    input: MLIRSparseTensor,
    operator="min",
    thunk=None,
    *,
    engine=None,
):
    if engine is None:
        engine = _default_engine

    if thunk is None:
        thunk = 0
    thunk_str = f"constant {thunk:f} : f64"

    op = operator.lower()
    func_name = f"matrix_apply_{op}_{thunk}"

    if op == "min":
        apply_str = """
        %cmp = cmpf olt, %val, %thunk : f64
        %new = select %cmp, %val, %thunk : f64
        """
    else:
        raise TypeError(f"Unsupported operator: {operator}")
    mlir_text = matrix_apply_mlir_template.substitute(
        FUNC_NAME=func_name, APPLY=apply_str, THUNK=thunk_str
    )

    if func_name not in engine.name_to_callable:
        engine.add(mlir_text, _standard_passes)
    func = getattr(engine, func_name)
    return func(output, input)


class BaseFunction:
    func_name = None

    def __init__(self):
        pass

    def get_mlir(self, *, make_private=True):
        """
        `make_private` is used to indicate whether the function should be private for fusion
        or public for standalone calling in the `compile` method
        """
        raise NotImplementedError()

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

        # Add module wrapper
        wrapped_text = self.module_wrapper_text.render(
            body=self.get_mlir(make_private=False)
        )

        engine.add(wrapped_text, passes)
        func = getattr(engine, self.func_name)
        return func

    module_wrapper_text = jinja2.Template(
        """
    module  {
      func private @sparseValuesF64(!llvm.ptr<i8>) -> memref<?xf64>
      func private @sparseIndices64(!llvm.ptr<i8>, index) -> memref<?xindex>
      func private @sparsePointers64(!llvm.ptr<i8>, index) -> memref<?xindex>
      func private @sparseDimSize(!llvm.ptr<i8>, index) -> index

      {{ body }}

    }
    """
    )


class Transpose(BaseFunction):
    """
    Call signature:
      transpose(output: MLIRSparseTensor, input: MLIRSparseTensor) -> None

      input is (nrows x ncols with nnz non-zero values)
      output must be (ncols x nrows with nnz non-zero values)

      The output will be written to during the function call
    """

    def __init__(self):
        super().__init__()
        self.func_name = "transpose"

    def get_mlir(self, *, make_private=True):
        return self.mlir_template.render(private_func=make_private)

    mlir_template = jinja2.Template(
        """
      func {% if private_func %}private{% endif %}@transpose(%output: !llvm.ptr<i8>, %input: !llvm.ptr<i8>) -> index {
        // Attempting to implement identical code as in scipy
        // https://github.com/scipy/scipy/blob/3b36a574dc657d1ca116f6e230be694f3de31afc/scipy/sparse/sparsetools/csr.h
        // function csr_tocsc

        %c0 = constant 0 : index
        %c1 = constant 1 : index

        // pymlir-skip: begin

        %n_row = call @sparseDimSize(%input, %c0) : (!llvm.ptr<i8>, index) -> index
        %n_col = call @sparseDimSize(%input, %c1) : (!llvm.ptr<i8>, index) -> index
        %Ap = call @sparsePointers64(%input, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %Aj = call @sparseIndices64(%input, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %Ax = call @sparseValuesF64(%input) : (!llvm.ptr<i8>) -> memref<?xf64>
        %Bp = call @sparsePointers64(%output, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %Bi = call @sparseIndices64(%output, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %Bx = call @sparseValuesF64(%output) : (!llvm.ptr<i8>) -> memref<?xf64>

        %nnz = load %Ap[%n_row] : memref<?xindex>

        // compute number of non-zero entries per column of A
        scf.for %arg2 = %c0 to %n_col step %c1 {
          store %c0, %Bp[%arg2] : memref<?xindex>
        }
        scf.for %n = %c0 to %nnz step %c1 {
          %colA = load %Aj[%n] : memref<?xindex>
          %colB = load %Bp[%colA] : memref<?xindex>
          %colB1 = addi %colB, %c1 : index
          store %colB1, %Bp[%colA] : memref<?xindex>
        }

        // cumsum the nnz per column to get Bp
        store %c0, %Bp[%n_col] : memref<?xindex>
        scf.for %col = %c0 to %n_col step %c1 {
          %temp = load %Bp[%col] : memref<?xindex>
          %cumsum = load %Bp[%n_col] : memref<?xindex>
          store %cumsum, %Bp[%col] : memref<?xindex>
          %cumsum2 = addi %cumsum, %temp : index
          store %cumsum2, %Bp[%n_col] : memref<?xindex>
        }

        scf.for %row = %c0 to %n_row step %c1 {
          %j_start = load %Ap[%row] : memref<?xindex>
          %row_plus1 = addi %row, %c1 : index
          %j_end = load %Ap[%row_plus1] : memref<?xindex>
          scf.for %jj = %j_start to %j_end step %c1 {
            %col = load %Aj[%jj] : memref<?xindex>
            %dest = load %Bp[%col] : memref<?xindex>

            store %row, %Bi[%dest] : memref<?xindex>
            %axjj = load %Ax[%jj] : memref<?xf64>
            store %axjj, %Bx[%dest] : memref<?xf64>

            // Bp[col]++
            %bp_inc = load %Bp[%col] : memref<?xindex>
            %bp_inc1 = addi %bp_inc, %c1 : index
            store %bp_inc1, %Bp[%col]: memref<?xindex>
          }
        }

        %last_last = load %Bp[%n_col] : memref<?xindex>
        store %c0, %Bp[%n_col] : memref<?xindex>
        scf.for %col = %c0 to %n_col step %c1 {
          %temp = load %Bp[%col] : memref<?xindex>
          %last = load %Bp[%n_col] : memref<?xindex>
          store %last, %Bp[%col] : memref<?xindex>
          store %temp, %Bp[%n_col] : memref<?xindex>
        }
        store %last_last, %Bp[%n_col] : memref<?xindex>

        // pymlir-skip: end

        return %c0 : index
      }
    """
    )


class MatrixSelect(BaseFunction):
    """
    Call signature:
      matrix_select(output: MLIRSparseTensor, input: MLIRSparseTensor) -> None

      input is (nrows x ncols with nnz non-zero values)
      output must be (nrows x ncols with nnz non-zero values)

      The output will be written to during the function call.
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
      func {% if private_func %}private{% endif %}@{{ func_name }}(%output: !llvm.ptr<i8>, %input: !llvm.ptr<i8>) -> index {
        %c0 = constant 0 : index
        %c1 = constant 1 : index
        %cf0 = constant 0.0 : f64

        // pymlir-skip: begin

        %n_row = call @sparseDimSize(%input, %c0) : (!llvm.ptr<i8>, index) -> index
        %n_col = call @sparseDimSize(%input, %c1) : (!llvm.ptr<i8>, index) -> index
        %Ap = call @sparsePointers64(%input, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %Aj = call @sparseIndices64(%input, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %Ax = call @sparseValuesF64(%input) : (!llvm.ptr<i8>) -> memref<?xf64>
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
        store %c0, %Bp[%c0] : memref<?xindex>
        scf.for %row = %c0 to %n_row step %c1 {
          // Copy Bp[row] into Bp[row+1]
          %row_plus1 = addi %row, %c1 : index
          %bp_curr_count = load %Bp[%row] : memref<?xindex>
          store %bp_curr_count, %Bp[%row_plus1] : memref<?xindex>

          // Read start/end positions from Ap
          %j_start = load %Ap[%row] : memref<?xindex>
          %j_end = load %Ap[%row_plus1] : memref<?xindex>
          scf.for %jj = %j_start to %j_end step %c1 {
            {% if needs_col -%}
              %col = load %Aj[%jj] : memref<?xindex>
            {%- endif %}
            {% if needs_val -%}
              %val = load %Ax[%jj] : memref<?xf64>
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
              %bj_pos = load %Bp[%row_plus1] : memref<?xindex>

              {# These conditions are inverted because if not defined above, they are still needed here #}
              {% if not needs_col -%}
                %col = load %Aj[%jj] : memref<?xindex>
              {%- endif %}
              store %col, %Bj[%bj_pos] : memref<?xindex>
              {% if not needs_val -%}
                %val = load %Ax[%jj] : memref<?xf64>
              {%- endif %}
              store %val, %Bx[%bj_pos] : memref<?xf64>

              // Increment Bp[row+1] += 1
              %bj_pos_plus1 = addi %bj_pos, %c1 : index
              store %bj_pos_plus1, %Bp[%row_plus1] : memref<?xindex>
            } else {
            }
          }
        }

        // pymlir-skip: end

        return %c0 : index
      }
    """
    )


matrix_reduce_mlir_template = Template(
    """
module  {
  func private @sparseValuesF64(!llvm.ptr<i8>) -> memref<?xf64>
  func private @sparseIndices64(!llvm.ptr<i8>, index) -> memref<?xindex>
  func private @sparsePointers64(!llvm.ptr<i8>, index) -> memref<?xindex>
  func private @sparseDimSize(!llvm.ptr<i8>, index) -> index

  func @${FUNC_NAME}(%input: !llvm.ptr<i8>) -> f64 {
    %cst = constant dense<0.000000e+00> : tensor<f64>
    %c0 = constant 0 : index
    %c1 = constant 1 : index

    // pymlir-skip: begin

    %0 = call @sparseDimSize(%input, %c0) : (!llvm.ptr<i8>, index) -> index
    %1 = call @sparsePointers64(%input, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
    %2 = call @sparseIndices64(%input, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
    %3 = call @sparseValuesF64(%input) : (!llvm.ptr<i8>) -> memref<?xf64>
    %4 = tensor_to_memref %cst : memref<f64>
    %5 = alloc() : memref<f64>
    linalg.copy(%4, %5) : memref<f64>, memref<f64>
    scf.for %arg1 = %c0 to %0 step %c1 {
      %8 = load %1[%arg1] : memref<?xindex>
      %9 = addi %arg1, %c1 : index
      %10 = load %1[%9] : memref<?xindex>
      %11 = load %5[] : memref<f64>
      %12 = scf.for %arg2 = %8 to %10 step %c1 iter_args(%x = %11) -> (f64) {
        %y = load %3[%arg2] : memref<?xf64>
        %z = ${AGG_FUNC}
        scf.yield %z : f64
      }
      store %12, %5[] : memref<f64>
    }
    %6 = tensor_load %5 : memref<f64>
    %7 = tensor.extract %6[] : tensor<f64>

    // pymlir-skip: end

    return %7 : f64
  }
}
"""
)

mxm_mlir_template = jinja2.Template(
    """
module  {
  func private @sparseValuesF64(!llvm.ptr<i8>) -> memref<?xf64>
  func private @sparseIndices64(!llvm.ptr<i8>, index) -> memref<?xindex>
  func private @sparsePointers64(!llvm.ptr<i8>, index) -> memref<?xindex>
  func private @sparseDimSize(!llvm.ptr<i8>, index) -> index

  func @{{ FUNC_NAME }}(
      %output: !llvm.ptr<i8>, %A: !llvm.ptr<i8>, %B: !llvm.ptr<i8>
      {%- if structural_mask -%}
      , %mask: !llvm.ptr<i8>
      {%- endif -%}
  ) -> index {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %cf0 = constant 0.0 : f64
    %cf1 = constant 1.0 : f64
    %ctrue = constant 1 : i1
    %cfalse = constant 0 : i1

    // pymlir-skip: begin

    %nrows = call @sparseDimSize(%A, %c0) : (!llvm.ptr<i8>, index) -> index
    %ncols = call @sparseDimSize(%B, %c1) : (!llvm.ptr<i8>, index) -> index
    %Ap = call @sparsePointers64(%A, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
    %Aj = call @sparseIndices64(%A, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
    %Ax = call @sparseValuesF64(%A) : (!llvm.ptr<i8>) -> memref<?xf64>
    %Bp = call @sparsePointers64(%B, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
    %Bi = call @sparseIndices64(%B, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
    %Bx = call @sparseValuesF64(%B) : (!llvm.ptr<i8>) -> memref<?xf64>
    %Cp = call @sparsePointers64(%output, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
    %Cj = call @sparseIndices64(%output, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
    %Cx = call @sparseValuesF64(%output) : (!llvm.ptr<i8>) -> memref<?xf64>
    {% if structural_mask -%}
    %Mp = call @sparsePointers64(%mask, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
    %Mj = call @sparseIndices64(%mask, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
    {%- endif %}
    %accumulator = alloc() : memref<f64>
    %not_empty = alloc() : memref<i1>

    // Method for constructing C in CSR format:
    // 1. Cp[0] = 0
    // 2. At the start of each row, copy Cp[row] into Cp[row+1]
    // 3. When writing row X, col Y, val V
    //    a. Cj[Cp[row+1]] = Y
    //    b. Cx[Cp[row+1]] = V
    //    c. Cp[row+1] += 1

    // Cp[0] = 0
    store %c0, %Cp[%c0] : memref<?xindex>
    scf.for %row = %c0 to %nrows step %c1 {
      // Copy Cp[row] into Cp[row+1]
      %row_plus1 = addi %row, %c1 : index
      %cp_curr_count = load %Cp[%row] : memref<?xindex>
      store %cp_curr_count, %Cp[%row_plus1] : memref<?xindex>

      {% if structural_mask %}
      %mcol_start = load %Mp[%row] : memref<?xindex>
      %mcol_end = load %Mp[%row_plus1] : memref<?xindex>

      scf.for %mm = %mcol_start to %mcol_end step %c1 {
        %col = load %Mj[%mm] : memref<?xindex>
        // NOTE: for valued masks, we would need to check the value and yield if false
      {% else %}
      scf.for %col = %c0 to %ncols step %c1 {
      {% endif %}

        // Iterate over row in A as ka, col in B as kb
        // Find matches, ignore otherwise
        %col_plus1 = addi %col, %c1 : index
        %jstart = load %Ap[%row] : memref<?xindex>
        %jend = load %Ap[%row_plus1] : memref<?xindex>
        %istart = load %Bp[%col] : memref<?xindex>
        %iend = load %Bp[%col_plus1] : memref<?xindex>

        store %cf0, %accumulator[] : memref<f64>
        store %cfalse, %not_empty[] : memref<i1>

        scf.while (%jj = %jstart, %ii = %istart) : (index, index) -> (index, index) {
          %jj_not_done = cmpi ult, %jj, %jend : index
          %ii_not_done = cmpi ult, %ii, %iend : index
          %cond = and %jj_not_done, %ii_not_done : i1
          scf.condition(%cond) %jj, %ii : index, index
        } do {
        ^bb0(%jj: index, %ii: index):  // no predecessors
          %kj = load %Aj[%jj] : memref<?xindex>
          %ki = load %Bi[%ii] : memref<?xindex>
          %ks_match = cmpi eq, %kj, %ki : index
          scf.if %ks_match {
            %existing = load %accumulator[] : memref<f64>
            {{ OP }}
            store %new, %accumulator[] : memref<f64>
            store %ctrue, %not_empty[] : memref<i1>
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

        %is_not_empty = load %not_empty[] : memref<i1>
        scf.if %is_not_empty {
          // Store accumulated value
          %cj_pos = load %Cp[%row_plus1] : memref<?xindex>
          store %col, %Cj[%cj_pos] : memref<?xindex>
          %accumulated = load %accumulator[] : memref<f64>
          store %accumulated, %Cx[%cj_pos] : memref<?xf64>
          // Increment Cp[row+1] += 1
          %cj_pos_plus1 = addi %cj_pos, %c1 : index
          store %cj_pos_plus1, %Cp[%row_plus1] : memref<?xindex>
        } else {
        }
      }
    }

    // pymlir-skip: end

    return %c0 : index
  }
}
"""
)


matrix_apply_mlir_template = Template(
    """
module  {
  func private @sparseValuesF64(!llvm.ptr<i8>) -> memref<?xf64>
  func private @sparseIndices64(!llvm.ptr<i8>, index) -> memref<?xindex>
  func private @sparsePointers64(!llvm.ptr<i8>, index) -> memref<?xindex>
  func private @sparseDimSize(!llvm.ptr<i8>, index) -> index

  func @${FUNC_NAME}(%output: !llvm.ptr<i8>, %input: !llvm.ptr<i8>) -> index {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %cf0 = constant 0.0 : f64
    %thunk = ${THUNK}

    // pymlir-skip: begin

    %n_row = call @sparseDimSize(%input, %c0) : (!llvm.ptr<i8>, index) -> index
    %n_col = call @sparseDimSize(%input, %c1) : (!llvm.ptr<i8>, index) -> index
    %Ap = call @sparsePointers64(%input, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
    %Aj = call @sparseIndices64(%input, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
    %Ax = call @sparseValuesF64(%input) : (!llvm.ptr<i8>) -> memref<?xf64>
    %Bp = call @sparsePointers64(%output, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
    %Bj = call @sparseIndices64(%output, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
    %Bx = call @sparseValuesF64(%output) : (!llvm.ptr<i8>) -> memref<?xf64>

    store %c0, %Bp[%c0] : memref<?xindex>
    scf.for %row = %c0 to %n_row step %c1 {
      // Read start/end positions from Ap
      %row_plus1 = addi %row, %c1 : index
      %j_start = load %Ap[%row] : memref<?xindex>
      %j_end = load %Ap[%row_plus1] : memref<?xindex>
      store %j_end, %Bp[%row_plus1] : memref<?xindex>

      scf.for %jj = %j_start to %j_end step %c1 {
        %col = load %Aj[%jj] : memref<?xindex>
        %val = load %Ax[%jj] : memref<?xf64>

        store %col, %Bj[%jj] : memref<?xindex>
        ${APPLY}
        store %new, %Bx[%jj] : memref<?xf64>
      }
    }

    // pymlir-skip: end

    return %c0 : index
  }
}
"""
)
