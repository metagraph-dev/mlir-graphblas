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

###########################################
# std ops
###########################################

class ConstantOp(BaseOp):
    name = "constant"

    @classmethod
    def call(cls, irbuilder, value, type):
        if type in {"f128", "f64", "f32", "f16", "f8"}:
            value = float(value)
        ret_val = irbuilder.new_var(type)
        return ret_val, (
            f"{ret_val.assign_string()} = constant {value} : {type}"
        )


class AddIOp(BaseOp):
    name = "addi"

    @classmethod
    def call(cls, irbuilder, lhs, rhs):
        if lhs.var_type != rhs.var_type:
            raise TypeError(f"Type mismatch: {lhs.var_type} != {rhs.var_type}")
        ret_val = irbuilder.new_var(lhs.var_type)
        return ret_val, (
            f"{ret_val.assign_string()} = addi {lhs.access_string()}, {rhs.access_string()} : {lhs.var_type}"
        )


###########################################
# llvm ops
###########################################

class LLVMGetElementPtrOp(BaseOp):
    dialect = "llvm"
    name = "getelementptr"

    @classmethod
    def call(cls, irbuilder, list, index):
        ret_val = irbuilder.new_var(list.var_type)
        return ret_val, (
            f"{ret_val.assign_string()} = llvm.getelementptr {list.access_string()}[{index.access_string()}] : "
            f"({list.var_type}, {index.var_type}) -> {list.var_type}"
        )


class LLVMLoadOp(BaseOp):
    dialect = "llvm"
    name = "load"

    @classmethod
    def call(cls, irbuilder, pointer, return_type):
        ret_val = irbuilder.new_var(return_type)
        return ret_val, (
            f"{ret_val.assign_string()} = llvm.load {pointer.access_string()} : {pointer.var_type}"
        )


###########################################
# graphblas ops
###########################################

class GraphBLAS_ConvertLayout(BaseOp):
    dialect = "graphblas"
    name = "convert_layout"

    @classmethod
    def call(cls, irbuilder, input, return_type):
        ret_val = irbuilder.new_var(return_type)
        return ret_val, (
            f"{ret_val.assign_string()} = graphblas.convert_layout {input.access_string()} : "
            f"{input.var_type} to {return_type}"
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
    def call(cls, irbuilder, input, apply_op, thunk, return_type):
        assert isinstance(thunk, MLIRVar), "thunk must be an MLIRVar"
        ret_val = irbuilder.new_var(return_type)
        return ret_val, (
            f"{ret_val.assign_string()} = graphblas.matrix_apply {input.access_string()}, {thunk.access_string()} "
            f'{{ apply_operator = "{apply_op}" }} : ({input.var_type}, {thunk.var_type}) to {return_type}'
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


###########################################
# util ops
###########################################

class PtrToTensorOp(BaseOp):
    dialect = "util"
    name = "ptr8_to_tensor"

    @classmethod
    def call(cls, irbuilder, input, return_type):
        ret_val = irbuilder.new_var(return_type)
        return ret_val, (
            f"{ret_val.assign_string()} = call @ptr8_to_tensor({input.access_string()}) : "
            f"(!llvm.ptr<i8>) -> {return_type}"
        )

class TensorToPtrOp(BaseOp):
    dialect = "util"
    name = "tensor_to_ptr8"

    @classmethod
    def call(cls, irbuilder, input):
        ret_val = irbuilder.new_var("!llvm.ptr<i8>")
        return ret_val, (
            f"{ret_val.assign_string()} = call @tensor_to_ptr8_to_tensor({input.access_string()}) : "
            f"({input.var_type}) -> !llvm.ptr<i8>"
        )