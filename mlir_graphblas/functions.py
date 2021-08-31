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

#CSC64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#CSX64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module  {
    func private @cast_csr_to_csx(tensor<?x?xf64, #CSR64>) -> tensor<?x?xf64, #CSX64>
    func private @cast_csc_to_csx(tensor<?x?xf64, #CSC64>) -> tensor<?x?xf64, #CSX64>
    func private @cast_csx_to_csr(tensor<?x?xf64, #CSX64>) -> tensor<?x?xf64, #CSR64>
    func private @cast_csx_to_csc(tensor<?x?xf64, #CSX64>) -> tensor<?x?xf64, #CSC64>
    
    func private @ptr8_to_matrix(!llvm.ptr<i8>) -> tensor<?x?xf64, #CSX64>
    func private @matrix_to_ptr8(tensor<?x?xf64, #CSX64>) -> !llvm.ptr<i8>
    
    func private @delSparseMatrix(tensor<?x?xf64, #CSX64>) -> ()
    func private @dup_matrix(tensor<?x?xf64, #CSX64>) -> tensor<?x?xf64, #CSX64>

    {{ body }}

}
        """,
        undefined=jinja2.StrictUndefined,
    )

    def get_mlir_module(self, make_private=False):
        """Get the MLIR text for this function wrapped in a MLIR module with
        declarations of external helper functions."""
        return self.MODULE_WRAPPER_TEXT.render(
            body=self.get_mlir(make_private=make_private),
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


class ConvertLayout(BaseFunction):
    """
    Call signature:
      convert_layout(input: MLIRSparseTensor) -> MLIRSparseTensor
    """

    _valid_layouts = {"csr", "csc"}

    def __init__(self, destination_layout="csc"):
        super().__init__()

        dest_layout = destination_layout.lower()
        if dest_layout not in self._valid_layouts:
            raise TypeError(
                f"Invalid layout: {destination_layout}, must be one of {list(self._valid_layouts)}"
            )

        self.func_name = f"convert_layout_to_{dest_layout}"
        self.destination_layout = dest_layout

    def get_mlir(self, *, make_private=True):
        return self.mlir_template.render(
            func_name=self.func_name,
            private_func=make_private,
            destination_layout=self.destination_layout,
        )

    mlir_template = jinja2.Template(
        """
      // Attempting to implement identical code as in scipy
      // https://github.com/scipy/scipy/blob/3b36a574dc657d1ca116f6e230be694f3de31afc/scipy/sparse/sparsetools/csr.h
      // function csr_tocsc

      {% if destination_layout == "csr" %}

      func {% if private_func %}private {% endif %}@{{ func_name }}(%input: tensor<?x?xf64, #CSC64>) -> tensor<?x?xf64, #CSR64> {

        %output = graphblas.convert_layout %input : tensor<?x?xf64, #CSC64> to tensor<?x?xf64, #CSR64>

        return %output : tensor<?x?xf64, #CSR64>

      {% else %}

      func {% if private_func %}private {% endif %}@{{ func_name }}(%input: tensor<?x?xf64, #CSR64>) -> tensor<?x?xf64, #CSC64> {

        %output = graphblas.convert_layout %input : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSC64>

        return %output : tensor<?x?xf64, #CSC64>

      {% endif %}
      }
    """,
        undefined=jinja2.StrictUndefined,
    )


class MatrixSelect(BaseFunction):
    """
    Call signature:
      If using a thunk-requiring selector:
        matrix_select(input: MLIRSparseTensor, thunk: float) -> MLIRSparseTensor
      Otherwise:
        matrix_select(input: MLIRSparseTensor) -> MLIRSparseTensor
    """

    _valid_selectors = {"triu", "tril", "gt"}
    _thunk_requiring_selectors = {"gt"}

    def __init__(self, selector="triu"):
        super().__init__()

        # TODO support multiple selectors

        sel = selector.lower()
        if sel not in self._valid_selectors:
            raise TypeError(
                f"Invalid selector: {selector}, must be one of {list(self._valid_selectors)}"
            )

        # TODO we need to account for other properties to avoid
        # collisions, e.g. the input tensor's element type
        self.func_name = f"matrix_select_{sel}"
        self.selector = sel

    def get_mlir(self, *, make_private=True):
        return self.mlir_template.render(
            func_name=self.func_name,
            private_func=make_private,
            selector=self.selector,
            needs_thunk=self.selector in self._thunk_requiring_selectors,
        )

    mlir_template = jinja2.Template(
        """
      func {% if private_func %}private {% endif %}@{{ func_name }}(
          %input: tensor<?x?xf64, #CSR64>
          {%- if needs_thunk -%}
          , %thunk: f64
          {%- endif -%}
      ) -> tensor<?x?xf64, #CSR64> {
        %output = graphblas.matrix_select %input{% if needs_thunk %}, %thunk{% endif %} { selectors = ["{{ selector }}"] } : tensor<?x?xf64, #CSR64>{% if needs_thunk %}, f64{% endif %} to tensor<?x?xf64, #CSR64>
        return %output : tensor<?x?xf64, #CSR64>
      }
    """,
        undefined=jinja2.StrictUndefined,
    )


class MatrixReduceToScalar(BaseFunction):
    """
    Call signature:
      matrix_reduce_to_scalar(input: MLIRSparseTensor) -> float64
    """

    _valid_aggregators = {"plus"}
    _agg_aliases = {
        "sum": "plus",
    }

    def __init__(self, aggregator="plus"):
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
        %total = graphblas.matrix_reduce_to_scalar %input { aggregator = "{{ agg }}" } : tensor<?x?xf64, #CSR64> to f64

        return %total : f64
      }
    """,
        undefined=jinja2.StrictUndefined,
    )


class Apply(BaseFunction):
    """
    Call signature:
      apply(input: MLIRSparseTensor, thunk: f64) -> MLIRSparseTensor
    """

    _valid_operators = {"min"}

    def __init__(self, operator="min"):
        super().__init__()

        op = operator.lower()
        if op not in self._valid_operators:
            raise TypeError(
                f"Invalid operator: {operator}, must be one of {list(self._valid_operators)}"
            )

        self.func_name = f"apply_{op}"
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
        %output = graphblas.apply %input, %thunk { apply_operator = "{{ op }}" } : (tensor<?x?xf64, #CSR64>, f64) to tensor<?x?xf64, #CSR64>

        return %output : tensor<?x?xf64, #CSR64>
      }
    """,
        undefined=jinja2.StrictUndefined,
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
          %A: tensor<?x?xf64, #CSR64>, %B: tensor<?x?xf64, #CSC64>
          {%- if structural_mask -%}
          , %mask: tensor<?x?xf64, #CSR64>
          {%- endif -%}
      ) -> tensor<?x?xf64, #CSR64> {
        {% if structural_mask %}
        %c1 = constant 1 : index
        %Mp = sparse_tensor.pointers %mask, %c1 : tensor<?x?xf64, #CSR64> to memref<?xi64>
        {% endif %}
        %output = graphblas.matrix_multiply %A, %B{% if structural_mask %}, %mask{% endif %} { semiring = "{{ semiring }}" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>{% if structural_mask %}, tensor<?x?xf64, #CSR64>{% endif %}) to tensor<?x?xf64, #CSR64>

        return %output : tensor<?x?xf64, #CSR64>
      }
    """,
        undefined=jinja2.StrictUndefined,
    )
