"""
Various ops written in MLIR which implement dialects or other utilities
"""
from typing import Tuple
from .mlir_builder import MLIRFunctionBuilder, MLIRVar


class BaseOp:
    dialect = None  # This is optional if op is in the std dialect; otherwise define it
    name = None

    @classmethod
    def call(
        cls, irbuilder: MLIRFunctionBuilder, *args, **kwargs
    ) -> Tuple[MLIRVar, str]:
        raise NotImplementedError()

    def __init_subclass__(cls):
        MLIRFunctionBuilder.register_op(cls)


class ConstantOp(BaseOp):
    name = "constant"

    @classmethod
    def call(cls, irbuilder, value, type):
        if type in {"f128", "f64", "f32", "f16", "f8"}:
            value = float(value)
        ret_val = irbuilder.new_var(type)
        return ret_val, f"{ret_val.assign_string()} = constant {value} : {type}"


class GraphBLAS_ConvertLayout(BaseOp):
    dialect = "graphblas"
    name = "convert_layout"

    @classmethod
    def call(cls, irbuilder, input, return_type):
        ret_val = irbuilder.new_var(return_type)
        return ret_val, (
            f"{ret_val.assign_string()} = graphblas.convert_layout {input.access_string()} : "
            f"{input.type} to {return_type}"
        )


class GraphBLAS_MatrixSelect(BaseOp):
    dialect = "graphblas"
    name = "matrix_select"

    @classmethod
    def call(cls, irbuilder, input, selector):
        ret_val = irbuilder.new_var(input.var_type)
        return ret_val, (
            f"{ret_val.assign_string()} = graphblas.matrix_select {input.access_string()} "
            f'{{ selector = "{selector}" }} : {input.var_type}'
        )


class GraphBLAS_MatrixReduceToScalar(BaseOp):
    dialect = "graphblas"
    name = "matrix_reduce_to_scalar"

    @classmethod
    def call(cls, irbuilder, input, aggregator, return_type):
        ret_val = irbuilder.new_var(return_type)
        return ret_val, (
            f"{ret_val.assign_string()} = graphblas.matrix_reduce_to_scalar {input.access_string()} "
            f'{{ aggregator = "{aggregator}" }} : {input.var_type} to {return_type}'
        )


class GraphBLAS_MatrixApply(BaseOp):
    dialect = "graphblas"
    name = "matrix_apply"

    @classmethod
    def call(cls, irbuilder, input, apply_op, thunk, thunk_type, return_type):
        thunk_val = irbuilder.new_var(thunk_type)
        ret_val = irbuilder.new_var(return_type)
        return ret_val, (
            f"{thunk_val.assign_string()} = constant {thunk}: {thunk_type}\n"
            f"{ret_val.assign_string()} = graphblas.matrix_apply {input.access_string()}, {thunk_val.access_string()} "
            f'{{ apply_operator = "{apply_op}" }} : ({input.var_type}, {thunk_type}) to {return_type}'
        )


class GraphBLAS_MatrixMultiply(BaseOp):
    dialect = "graphblas"
    name = "matrix_multiply"

    @classmethod
    def call(cls, irbuilder, a, b, mask, semiring, return_type):
        ret_val = irbuilder.new_var(return_type)
        if mask:
            mlir = (
                f"{ret_val.assign_string()} = graphblas.matrix_multiply {a.access_string()}, {b.access_string()}, "
                f"{mask.access_string()} "
                f'{{ semiring = "{semiring}" }} : ({a.var_type}, {b.var_type}, {mask.var_type}) to {return_type}'
            )
        else:
            mlir = (
                f"{ret_val.assign_string()} = graphblas.matrix_multiply {a.access_string()}, {b.access_string()} "
                f'{{ semiring = "{semiring}" }} : ({a.var_type}, {b.var_type}) to {return_type}'
            )
        return ret_val, mlir
