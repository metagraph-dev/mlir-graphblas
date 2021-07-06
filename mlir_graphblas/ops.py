"""
Various ops written in MLIR which implement dialects or other utilities
"""
from typing import Tuple, Sequence, Optional, Union
from .mlir_builder import MLIRFunctionBuilder, MLIRVar
from .types import MemrefType, TensorType, SparseEncodingType


class BaseOp:
    dialect = None  # This is optional if op is in the std dialect; otherwise define it
    name = None

    @classmethod
    def call(
        cls, irbuilder: MLIRFunctionBuilder, *args, **kwargs
    ) -> Tuple[MLIRVar, str]:
        raise NotImplementedError()

    @classmethod
    def ensure_mlirvar(cls, obj, type_=None):
        """
        Raises a TypeError is obj is not an MLIRVar
        If type_ is specified, raises a TypeError if obj.type is not of type type_
        """
        if not isinstance(obj, MLIRVar):
            raise TypeError(f"{cls.name} expects an MLIRVar, but got {obj}.")
        if type_ is not None:
            if not isinstance(obj.type, type_):
                raise TypeError(
                    f"{cls.name} expects an MLIRVar with type {type_}, but got {obj!r}"
                )

    def __init_subclass__(cls):
        MLIRFunctionBuilder.register_op(cls)


###########################################
# std ops
###########################################


class ConstantOp(BaseOp):
    name = "constant"

    @classmethod
    def call(cls, irbuilder, value, type):
        if str(type) in {"bf16", "f16", "f32", "f64", "f80", "f128"}:
            value = float(value)
        ret_val = irbuilder.new_var(type)
        return ret_val, (f"{ret_val.assign} = constant {value} : {type}")


class IndexCastOp(BaseOp):
    name = "index_cast"

    @classmethod
    def call(cls, irbuilder, value: MLIRVar, result_type):
        cls.ensure_mlirvar(value)
        ret_val = irbuilder.new_var(result_type)
        return ret_val, (
            f"{ret_val.assign} = std.index_cast {value} : {value.type} to {ret_val.type}"
        )


class AddIOp(BaseOp):
    name = "addi"

    @classmethod
    def call(cls, irbuilder, lhs, rhs):
        cls.ensure_mlirvar(lhs)
        cls.ensure_mlirvar(rhs)
        if lhs.type != rhs.type:
            raise TypeError(f"Type mismatch: {lhs.type} != {rhs.type}")
        ret_val = irbuilder.new_var(lhs.type)
        return ret_val, (f"{ret_val.assign} = addi {lhs}, {rhs} : {lhs.type}")


class MulIOp(BaseOp):
    name = "muli"

    @classmethod
    def call(cls, irbuilder, lhs, rhs):
        cls.ensure_mlirvar(lhs)
        cls.ensure_mlirvar(rhs)
        if lhs.type != rhs.type:
            raise TypeError(f"Type mismatch: {lhs.type} != {rhs.type}")
        ret_val = irbuilder.new_var(lhs.type)
        return ret_val, (f"{ret_val.assign} = muli {lhs}, {rhs} : {lhs.type}")


class AddFOp(BaseOp):
    name = "addf"

    @classmethod
    def call(cls, irbuilder, lhs, rhs):
        cls.ensure_mlirvar(lhs)
        cls.ensure_mlirvar(rhs)
        if lhs.type != rhs.type:
            raise TypeError(f"Type mismatch: {lhs.type} != {rhs.type}")
        ret_val = irbuilder.new_var(lhs.type)
        return ret_val, (f"{ret_val.assign} = addf {lhs}, {rhs} : {lhs.type}")


class MulFOp(BaseOp):
    name = "mulf"

    @classmethod
    def call(cls, irbuilder, lhs, rhs):
        cls.ensure_mlirvar(lhs)
        cls.ensure_mlirvar(rhs)
        if lhs.type != rhs.type:
            raise TypeError(f"Type mismatch: {lhs.type} != {rhs.type}")
        ret_val = irbuilder.new_var(lhs.type)
        return ret_val, (f"{ret_val.assign} = mulf {lhs}, {rhs} : {lhs.type}")


###########################################
# memref ops
###########################################


class MemrefAllocOp(BaseOp):
    dialect = "memref"
    name = "alloc"

    @classmethod
    def call(cls, irbuilder, type: str):
        ret_val = irbuilder.new_var(type)
        return ret_val, (f"{ret_val.assign} = memref.alloc() : {ret_val.type}")


class MemrefStoreOp(BaseOp):
    dialect = "memref"
    name = "store"

    @classmethod
    def call(
        cls,
        irbuilder,
        value,
        destination: MLIRVar,
        indices: Sequence[Union[MLIRVar, int]],
    ):
        cls.ensure_mlirvar(destination)
        indices_string = ", ".join(map(str, indices))
        return None, (
            f"memref.store {value}, {destination}[{indices_string}] : {destination.type}"
        )


class MemrefLoadOp(BaseOp):
    dialect = "memref"
    name = "load"

    @classmethod
    def call(
        cls, irbuilder, input_memref: MLIRVar, indices: Sequence[Union[MLIRVar, int]]
    ):
        cls.ensure_mlirvar(input_memref, MemrefType)
        indices_string = ", ".join(map(str, indices))
        ret_val = irbuilder.new_var(input_memref.type.value_type)
        return ret_val, (
            f"{ret_val.assign} = memref.load {input_memref}[{indices_string}] : {input_memref.type}"
        )


###########################################
# llvm ops
###########################################


class LLVMGetElementPtrOp(BaseOp):
    dialect = "llvm"
    name = "getelementptr"

    @classmethod
    def call(cls, irbuilder, list_, index):
        cls.ensure_mlirvar(list_)
        ret_val = irbuilder.new_var(list_.type)
        return ret_val, (
            f"{ret_val.assign} = llvm.getelementptr {list_}[{index}] : "
            f"({list_.type}, {index.type}) -> {list_.type}"
        )


class LLVMLoadOp(BaseOp):
    dialect = "llvm"
    name = "load"

    @classmethod
    def call(cls, irbuilder, pointer, return_type):
        cls.ensure_mlirvar(pointer)
        ret_val = irbuilder.new_var(return_type)
        return ret_val, (f"{ret_val.assign} = llvm.load {pointer} : {pointer.type}")


###########################################
# graphblas ops
###########################################


class GraphBLAS_NumRows(BaseOp):
    dialect = "graphblas"
    name = "num_rows"

    @classmethod
    def call(cls, irbuilder, input):
        cls.ensure_mlirvar(input, TensorType)
        ret_val = irbuilder.new_var("index")
        return ret_val, (
            f"{ret_val.assign} = graphblas.num_rows {input} : {input.type}"
        )


class GraphBLAS_ConvertLayout(BaseOp):
    dialect = "graphblas"
    name = "convert_layout"

    @classmethod
    def call(cls, irbuilder, input, return_type):
        cls.ensure_mlirvar(input, TensorType)
        ret_val = irbuilder.new_var(return_type)
        return ret_val, (
            f"{ret_val.assign} = graphblas.convert_layout {input} : "
            f"{input.type} to {ret_val.type}"
        )


class GraphBLAS_MatrixSelect(BaseOp):
    dialect = "graphblas"
    name = "matrix_select"

    @classmethod
    def call(cls, irbuilder, input, selector):
        cls.ensure_mlirvar(input, TensorType)
        ret_val = irbuilder.new_var(input.type)
        return ret_val, (
            f"{ret_val.assign} = graphblas.matrix_select {input} "
            f'{{ selectors = ["{selector}"] }} : {input.type} to {input.type}'
        )


class GraphBLAS_MatrixReduceToScalar(BaseOp):
    dialect = "graphblas"
    name = "matrix_reduce_to_scalar"
    allowed_aggregators = {"sum"}

    @classmethod
    def call(cls, irbuilder, input, aggregator):
        cls.ensure_mlirvar(input, TensorType)
        if aggregator not in cls.allowed_aggregators:
            raise TypeError(
                f"Illegal aggregator: {aggregator}, must be one of {cls.allowed_aggregators}"
            )
        return_type = input.type.value_type
        # TODO: return_type might be influenced by future allowable aggregators
        ret_val = irbuilder.new_var(return_type)
        return ret_val, (
            f"{ret_val.assign} = graphblas.matrix_reduce_to_scalar {input} "
            f'{{ aggregator = "{aggregator}" }} : {input.type} to {ret_val.type}'
        )


class GraphBLAS_MatrixApply(BaseOp):
    dialect = "graphblas"
    name = "matrix_apply"
    allowed_ops = {"min"}

    @classmethod
    def call(cls, irbuilder, input, apply_op, thunk):
        cls.ensure_mlirvar(input, TensorType)
        cls.ensure_mlirvar(thunk)
        if apply_op not in cls.allowed_ops:
            raise TypeError(
                f"Illegal apply_op: {apply_op}, must be one of {cls.allowed_ops}"
            )
        return_type = input.type
        # TODO: return_type might be influenced by future allowable ops
        ret_val = irbuilder.new_var(return_type)
        return ret_val, (
            f"{ret_val.assign} = graphblas.matrix_apply {input}, {thunk} "
            f'{{ apply_operator = "{apply_op}" }} : ({input.type}, {thunk.type}) to {ret_val.type}'
        )


class GraphBLAS_MatrixMultiply(BaseOp):
    dialect = "graphblas"
    name = "matrix_multiply"
    allowed_semirings = {"plus_plus", "plus_times", "plus_pair"}

    @classmethod
    def call(cls, irbuilder, a, b, semiring, *, mask=None):
        cls.ensure_mlirvar(a, TensorType)
        cls.ensure_mlirvar(b, TensorType)
        if semiring not in cls.allowed_semirings:
            raise TypeError(
                f"Illegal semiring: {semiring}, must be one of {cls.allowed_semirings}"
            )
        return_type = a.type
        # TODO: make the return type more robust; may depend on a, b, and/or semiring
        ret_val = irbuilder.new_var(return_type)
        if mask:
            cls.ensure_mlirvar(mask)
            mlir = (
                f"{ret_val.assign} = graphblas.matrix_multiply {a}, {b}, "
                f"{mask} "
                f'{{ semiring = "{semiring}" }} : ({a.type}, {b.type}, {mask.type}) to {ret_val.type}'
            )
        else:
            mlir = (
                f"{ret_val.assign} = graphblas.matrix_multiply {a}, {b} "
                f'{{ semiring = "{semiring}" }} : ({a.type}, {b.type}) to {ret_val.type}'
            )
        return ret_val, mlir


###########################################
# util ops
###########################################

# TODO these are used by the ops below ; should we inspect the
# inputs to get the bitwidths instead of assuming 64?
CSR64 = SparseEncodingType(["dense", "compressed"], [0, 1], 64, 64)
CSC64 = SparseEncodingType(["dense", "compressed"], [1, 0], 64, 64)
CSX64 = SparseEncodingType(["dense", "compressed"], None, 64, 64)


class CastCsrToCsxOp(BaseOp):
    dialect = "util"
    name = "cast_csr_to_csx"

    @classmethod
    def call(cls, irbuilder, input):
        cls.ensure_mlirvar(input)
        ret_val = irbuilder.new_var(f"tensor<?x?xf64, {CSX64}>")
        return ret_val, (
            f"{ret_val.assign} = call @cast_csr_to_csx({input}) : "
            f"({input.type}) -> tensor<?x?xf64, {CSX64}>"
        )


class CastCscToCsxOp(BaseOp):
    dialect = "util"
    name = "cast_csc_to_csx"

    @classmethod
    def call(cls, irbuilder, input):
        cls.ensure_mlirvar(input)
        ret_val = irbuilder.new_var(f"tensor<?x?xf64, {CSX64}>")
        return ret_val, (
            f"{ret_val.assign} = call @cast_csc_to_csx({input}) : "
            f"({input.type}) -> tensor<?x?xf64, {CSX64}>"
        )


class CastCsxToCsrOp(BaseOp):
    dialect = "util"
    name = "cast_csx_to_csr"

    @classmethod
    def call(cls, irbuilder, input):
        cls.ensure_mlirvar(input)
        ret_val = irbuilder.new_var(f"tensor<?x?xf64, {CSR64}>")
        return ret_val, (
            f"{ret_val.assign} = call @cast_csx_to_csr({input}) : "
            f"({input.type}) -> tensor<?x?xf64, {CSR64}>"
        )


class CastCsxToCscOp(BaseOp):
    dialect = "util"
    name = "cast_csx_to_csc"

    @classmethod
    def call(cls, irbuilder, input):
        cls.ensure_mlirvar(input)
        ret_val = irbuilder.new_var(f"tensor<?x?xf64, {CSC64}>")
        return ret_val, (
            f"{ret_val.assign} = call @cast_csx_to_csc({input}) : "
            f"({input.type}) -> tensor<?x?xf64, {CSC64}>"
        )


class PtrToTensorOp(BaseOp):
    dialect = "util"
    name = "ptr8_to_tensor"

    @classmethod
    def call(cls, irbuilder, input, return_type):
        cls.ensure_mlirvar(input)
        tensor_type = TensorType.parse(return_type, irbuilder.aliases)
        encoding = tensor_type.encoding
        if encoding.levels != ["dense", "compressed"]:
            raise TypeError(
                f"Return type must denote a CSR or CSC tensor (got {return_type})."
            )
        ret_val = irbuilder.new_var(f"tensor<?x?xf64, {CSX64}>")
        ret_string = (
            f"{ret_val.assign} = call @ptr8_to_tensor({input}) : "
            f"(!llvm.ptr<i8>) -> tensor<?x?xf64, {CSX64}>"
        )
        if encoding.ordering is None:
            pass
        elif encoding.ordering == [0, 1]:
            ret_val, cast_string = CastCsxToCsrOp.call(irbuilder, ret_val)
            ret_string = ret_string + "\n" + cast_string
        elif encoding.ordering == [1, 0]:
            ret_val, cast_string = CastCsxToCscOp.call(irbuilder, ret_val)
            ret_string = ret_string + "\n" + cast_string
        else:
            raise TypeError(
                f"Return type must denote a CSR or CSC tensor (got {return_type})."
            )
        return ret_val, ret_string


class TensorToPtrOp(BaseOp):
    dialect = "util"
    name = "tensor_to_ptr8"

    @classmethod
    def call(cls, irbuilder, input):
        cls.ensure_mlirvar(input, TensorType)

        encoding = input.type.encoding
        if encoding.levels != ["dense", "compressed"]:
            raise TypeError(
                f"Input type must denote a CSR or CSC tensor (got {input.type})."
            )

        if encoding.ordering is None:
            cast_string = ""
        elif encoding.ordering == [0, 1]:
            input, cast_string = CastCsrToCsxOp.call(irbuilder, input)
            cast_string += "\n"
        elif encoding.ordering == [1, 0]:
            input, cast_string = CastCscToCsxOp.call(irbuilder, input)
            cast_string += "\n"
        else:
            raise TypeError(
                f"Return type must denote a CSR or CSC tensor (got {input.type})."
            )

        ret_val = irbuilder.new_var("!llvm.ptr<i8>")
        return ret_val, cast_string + (
            f"{ret_val.assign} = call @tensor_to_ptr8({input}) : "
            f"({input.type}) -> !llvm.ptr<i8>"
        )


class DelSparseTensor(BaseOp):
    dialect = "util"
    name = "del_sparse_tensor"

    @classmethod
    def call(cls, irbuilder, input):
        cls.ensure_mlirvar(input, TensorType)
        encoding = input.type.encoding
        if encoding.ordering is None:
            cast_string = ""
        elif encoding.ordering == [0, 1]:
            input, cast_string = CastCsrToCsxOp.call(irbuilder, input)
            cast_string += "\n"
        elif encoding.ordering == [1, 0]:
            input, cast_string = CastCscToCsxOp.call(irbuilder, input)
            cast_string += "\n"

        return None, cast_string + (
            f"call @delSparseTensor({input}) : ({input.type}) -> ()"
        )


class DupTensor(BaseOp):
    dialect = "util"
    name = "dup_tensor"

    @classmethod
    def call(cls, irbuilder, input):
        cls.ensure_mlirvar(input, TensorType)
        encoding = input.type.encoding
        if encoding.ordering is None:
            cast_string = ""
        elif encoding.ordering == [0, 1]:
            input, cast_string = CastCsrToCsxOp.call(irbuilder, input)
            cast_string += "\n"
        elif encoding.ordering == [1, 0]:
            input, cast_string = CastCscToCsxOp.call(irbuilder, input)
            cast_string += "\n"

        ret_val = irbuilder.new_var(input.type)
        return ret_val, cast_string + (
            f"{ret_val.assign} = call @dup_tensor({input}) : ({input.type}) -> {input.type}"
        )
