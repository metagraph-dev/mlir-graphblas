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
            f"{ret_val.assign} = constant {value} : {type}"
        )


class AddIOp(BaseOp):
    name = "addi"

    @classmethod
    def call(cls, irbuilder, lhs, rhs):
        if lhs.type != rhs.type:
            raise TypeError(f"Type mismatch: {lhs.type} != {rhs.type}")
        ret_val = irbuilder.new_var(lhs.type)
        return ret_val, (
            f"{ret_val.assign} = addi {lhs}, {rhs} : {lhs.type}"
        )


###########################################
# llvm ops
###########################################

class LLVMGetElementPtrOp(BaseOp):
    dialect = "llvm"
    name = "getelementptr"

    @classmethod
    def call(cls, irbuilder, list, index):
        ret_val = irbuilder.new_var(list.type)
        return ret_val, (
            f"{ret_val.assign} = llvm.getelementptr {list}[{index}] : "
            f"({list.type}, {index.type}) -> {list.type}"
        )


class LLVMLoadOp(BaseOp):
    dialect = "llvm"
    name = "load"

    @classmethod
    def call(cls, irbuilder, pointer, return_type):
        ret_val = irbuilder.new_var(return_type)
        return ret_val, (
            f"{ret_val.assign} = llvm.load {pointer} : {pointer.type}"
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
            f"{ret_val.assign} = graphblas.convert_layout {input} : "
            f"{input.type} to {return_type}"
        )


class GraphBLAS_MatrixSelect(BaseOp):
    dialect = "graphblas"
    name = "matrix_select"

    @classmethod
    def call(cls, irbuilder, input, selector):
        ret_val = irbuilder.new_var(input.type)
        return ret_val, (
            f"{ret_val.assign} = graphblas.matrix_select {input} "
            f'{{ selector = "{selector}" }} : {input.type}'
        )


class GraphBLAS_MatrixReduceToScalar(BaseOp):
    dialect = "graphblas"
    name = "matrix_reduce_to_scalar"

    @classmethod
    def call(cls, irbuilder, input, aggregator, return_type):
        ret_val = irbuilder.new_var(return_type)
        return ret_val, (
            f"{ret_val.assign} = graphblas.matrix_reduce_to_scalar {input} "
            f'{{ aggregator = "{aggregator}" }} : {input.type} to {return_type}'
        )


class GraphBLAS_MatrixApply(BaseOp):
    dialect = "graphblas"
    name = "matrix_apply"

    @classmethod
    def call(cls, irbuilder, input, apply_op, thunk, return_type):
        assert isinstance(thunk, MLIRVar), "thunk must be an MLIRVar"
        ret_val = irbuilder.new_var(return_type)
        return ret_val, (
            f"{ret_val.assign} = graphblas.matrix_apply {input}, {thunk} "
            f'{{ apply_operator = "{apply_op}" }} : ({input.type}, {thunk.type}) to {return_type}'
        )


class GraphBLAS_MatrixMultiply(BaseOp):
    dialect = "graphblas"
    name = "matrix_multiply"

    @classmethod
    def call(cls, irbuilder, a, b, mask, semiring, return_type):
        ret_val = irbuilder.new_var(return_type)
        if mask:
            mlir = (
                f"{ret_val.assign} = graphblas.matrix_multiply {a}, {b}, "
                f"{mask} "
                f'{{ semiring = "{semiring}" }} : ({a.type}, {b.type}, {mask.type}) to {return_type}'
            )
        else:
            mlir = (
                f"{ret_val.assign} = graphblas.matrix_multiply {a}, {b} "
                f'{{ semiring = "{semiring}" }} : ({a.type}, {b.type}) to {return_type}'
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
            f"{ret_val.assign} = call @ptr8_to_tensor({input}) : "
            f"(!llvm.ptr<i8>) -> {return_type}"
        )


class TensorToPtrOp(BaseOp):
    dialect = "util"
    name = "tensor_to_ptr8"

    @classmethod
    def call(cls, irbuilder, input):
        ret_val = irbuilder.new_var("!llvm.ptr<i8>")
        return ret_val, (
            f"{ret_val.assign} = call @tensor_to_ptr8({input}) : "
            f"({input.type}) -> !llvm.ptr<i8>"
        )


class CastCsrToCscOp(BaseOp):
    dialect = "util"
    name = "cast_csr_to_csc"

    @classmethod
    def call(cls, irbuilder, input):
        ret_val = irbuilder.new_var("tensor<?x?xf64, #CSC64>")
        return ret_val, (
            f"{ret_val.assign} = call @cast_csr_to_csc({input}) : "
            f"({input.type}) -> tensor<?x?xf64, #CSC64>"
        )


class CastCscToCsrOp(BaseOp):
    dialect = "util"
    name = "cast_csc_to_csr"

    @classmethod
    def call(cls, irbuilder, input):
        ret_val = irbuilder.new_var("tensor<?x?xf64, #CSR64>")
        return ret_val, (
            f"{ret_val.assign} = call @cast_csc_to_csr({input}) : "
            f"({input.type}) -> tensor<?x?xf64, #CSR64>"
        )
