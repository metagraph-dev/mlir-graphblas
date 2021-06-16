"""
Various ops written in MLIR which implement dialects or other utilities
"""
from typing import Tuple, Sequence, Optional, Union
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
        if type in {"bf16", "f16", "f32", "f64", "f80", "f128"}:
            value = float(value)
        ret_val = irbuilder.new_var(type)
        return ret_val, (f"{ret_val.assign} = constant {value} : {type}")


class IndexCastOp(BaseOp):
    name = "index_cast"

    @classmethod
    def call(cls, irbuilder, value: MLIRVar, result_type):
        if not isinstance(value, MLIRVar):
            raise TypeError(
                f"{cls.name} expected an {MLIRVar.__qualname__}, but got {lhs}."
            )
        ret_val = irbuilder.new_var(result_type)
        return ret_val, (
            f"{ret_val.assign} = std.index_cast {value} : {value.type} to {result_type}"
        )


class AddIOp(BaseOp):
    name = "addi"

    @classmethod
    def call(cls, irbuilder, lhs, rhs):
        if not isinstance(lhs, MLIRVar):
            raise TypeError(
                f"{cls.name} expected an {MLIRVar.__qualname__}, but got {lhs}."
            )
        if not isinstance(rhs, MLIRVar):
            raise TypeError(
                f"{cls.name} expected an {MLIRVar.__qualname__}, but got {rhs}."
            )
        if lhs.type != rhs.type:
            raise TypeError(f"Type mismatch: {lhs.type} != {rhs.type}")
        ret_val = irbuilder.new_var(lhs.type)
        return ret_val, (f"{ret_val.assign} = addi {lhs}, {rhs} : {lhs.type}")


class MulIOp(BaseOp):
    name = "muli"

    @classmethod
    def call(cls, irbuilder, lhs, rhs):
        if not isinstance(lhs, MLIRVar):
            raise TypeError(
                f"{cls.name} expected an {MLIRVar.__qualname__}, but got {lhs}."
            )
        if not isinstance(rhs, MLIRVar):
            raise TypeError(
                f"{cls.name} expected an {MLIRVar.__qualname__}, but got {rhs}."
            )
        if lhs.type != rhs.type:
            raise TypeError(f"Type mismatch: {lhs.type} != {rhs.type}")
        ret_val = irbuilder.new_var(lhs.type)
        return ret_val, (f"{ret_val.assign} = muli {lhs}, {rhs} : {lhs.type}")


class AddFOp(BaseOp):
    name = "addf"

    @classmethod
    def call(cls, irbuilder, lhs, rhs):
        if not isinstance(lhs, MLIRVar):
            raise TypeError(
                f"{cls.name} expected an {MLIRVar.__qualname__}, but got {lhs}."
            )
        if not isinstance(rhs, MLIRVar):
            raise TypeError(
                f"{cls.name} expected an {MLIRVar.__qualname__}, but got {rhs}."
            )
        if lhs.type != rhs.type:
            raise TypeError(f"Type mismatch: {lhs.type} != {rhs.type}")
        ret_val = irbuilder.new_var(lhs.type)
        return ret_val, (f"{ret_val.assign} = addf {lhs}, {rhs} : {lhs.type}")


class MulFOp(BaseOp):
    name = "mulf"

    @classmethod
    def call(cls, irbuilder, lhs, rhs):
        if not isinstance(lhs, MLIRVar):
            raise TypeError(
                f"{cls.name} expected an {MLIRVar.__qualname__}, but got {lhs}."
            )
        if not isinstance(rhs, MLIRVar):
            raise TypeError(
                f"{cls.name} expected an {MLIRVar.__qualname__}, but got {rhs}."
            )
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
    def call(cls, irbuilder, shape: Optional[Sequence[Optional[int]]], type: str):
        """
        If shape is None, then the memref is unranked.
        If shape is a sequence, NoneType elements indicate an arbitrarily shaped dimension.
        """
        shape = (
            ["*"]
            if shape is None
            else ["?" if dim is None else str(dim) for dim in shape]
        )
        shape_and_dtype = "x".join(shape + [type])
        memref_type = f"memref<{shape_and_dtype}>"
        ret_val = irbuilder.new_var(memref_type)
        return ret_val, (f"{ret_val.assign} = memref.alloc() : {memref_type}")


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
        indices_string = ", ".join(map(str, indices))
        # TODO can we do this parsing in PyMLIR?
        _, ret_type = input_memref.type.split("memref<")
        ret_type = ret_type[:-1]  # before this, it endsx with ">"
        ret_type = ret_type.split(",")[
            0
        ].strip()  # Grab the dimensions+dtype string, e.g. ?x?xf32
        ret_type = ret_type.split("x")[-1]
        ret_val = irbuilder.new_var(ret_type)
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
        return ret_val, (f"{ret_val.assign} = llvm.load {pointer} : {pointer.type}")


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
