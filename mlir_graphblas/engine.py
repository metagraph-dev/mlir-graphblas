import os
import re
import subprocess
import itertools
import mlir
import ctypes
import glob
import operator
import llvmlite.binding as llvm
import numpy as np
from .sparse_utils import MLIRSparseTensor
from functools import reduce, partial
from .cli import MlirOptCli, MlirOptError, DebugResult
from typing import (
    Tuple,
    List,
    Iterable,
    Set,
    Dict,
    Callable,
    Union,
    Any,
    Optional,
)

# TODO we need O(1) access to the types of each dialect ; make this part of PyMLIR
_DIALECT_TYPES = {
    dialect.name: {
        dialect_type.__name__: dialect_type for dialect_type in dialect.types
    }
    for dialect in mlir.dialects.STANDARD_DIALECTS
}

_CURRENT_MODULE_DIR = os.path.dirname(__file__)
_SPARSE_UTILS_SO_FILE_PATTERN = os.path.join(_CURRENT_MODULE_DIR, "SparseUtils*.so")
_SPARSE_UTILS_SO_FILES = glob.glob(_SPARSE_UTILS_SO_FILE_PATTERN)
if len(_SPARSE_UTILS_SO_FILES) == 0:
    # TODO this hard-codes the setup.py option and the location of setup.py
    raise RuntimeError(
        f'{_SPARSE_UTILS_SO_FILE_PATTERN} not found. This can typically be solved by running "python setup.py build_ext" from {os.path.dirname(_CURRENT_MODULE_DIR)}.'
    )
elif len(_SPARSE_UTILS_SO_FILES) > 1:
    raise RuntimeError(
        f"Multiple files matching {_SPARSE_UTILS_SO_FILE_PATTERN} found."
    )
[_SPARSE_UTILS_SO] = _SPARSE_UTILS_SO_FILES
llvm.load_library_permanently(
    _SPARSE_UTILS_SO
)  # TODO will this cause name collisions with other uses of llvmlite by third-party libraries?
llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

MLIR_FLOAT_ENUM_TO_NP_TYPE = {
    mlir.astnodes.FloatTypeEnum.f16: np.float16,
    mlir.astnodes.FloatTypeEnum.f32: np.float32,
    mlir.astnodes.FloatTypeEnum.f64: np.float64,
    # TODO handle mlir.astnodes.FloatTypeEnum.bf16
}

INTEGER_WIDTH_TO_NP_TYPE = {  # TODO is there some nice accessor to get np.int* types from integers ?
    8: np.int8,
    16: np.int16,
    32: np.int32,
    64: np.int64,
    # TODO do we need np.longlong here?
}

NP_TYPE_TO_CTYPES_TYPE = {
    np.dtype(ctype).type: ctype
    for ctype in [  # from numpy.ctypeslib._get_scalar_type_map
        ctypes.c_byte,
        ctypes.c_short,
        ctypes.c_int,
        ctypes.c_long,
        ctypes.c_longlong,
        ctypes.c_ubyte,
        ctypes.c_ushort,
        ctypes.c_uint,
        ctypes.c_ulong,
        ctypes.c_ulonglong,
        ctypes.c_float,
        ctypes.c_double,
        ctypes.c_bool,
    ]
}


def convert_mlir_atomic_type(
    mlir_type: mlir.astnodes.Type, return_pointer_type: bool = False
) -> Tuple[type, type]:
    """
    First return value is a subclass of np.generic .
    Second return value is a subclass of _ctypes._CData .
    """

    # TODO Add a custom hash/eq method for mlir.astnodes.Node so that we can put
    # all the mlir.astnodes.*Type instances into a dict
    # it'll obviate the need for this section of code.
    np_type = None
    if isinstance(mlir_type, mlir.astnodes.FloatType):
        np_type = MLIR_FLOAT_ENUM_TO_NP_TYPE[mlir_type.type]
    elif isinstance(mlir_type, mlir.astnodes.IntegerType):
        np_type = INTEGER_WIDTH_TO_NP_TYPE[int(mlir_type.width)]
    elif isinstance(mlir_type, mlir.astnodes.IndexType):
        np_type = np.uint64
    if np_type is None:
        raise ValueError(f"Could not determine numpy type corresonding to {mlir_type}")

    ctypes_type = (
        np.ctypeslib.ndpointer(np_type)
        if return_pointer_type
        else NP_TYPE_TO_CTYPES_TYPE[np_type]
    )

    return np_type, ctypes_type


############################
# MLIR Return Type Helpers #
############################


def return_pretty_dialect_type_to_ctypes(
    pretty_type: mlir.astnodes.PrettyDialectType,
) -> Tuple[type, Callable]:
    # Pretty dialect types must be special-cased since they're arbitrarily structured.
    raise NotImplementedError(f"Converting {pretty_type} to ctypes not yet supported.")


def return_tensor_to_ctypes(
    tensor_type: mlir.astnodes.RankedTensorType,
) -> Tuple[type, Callable]:
    if tensor_type.encoding is not None:
        # We assume anything with an encoding must be a sparse tensor
        # After lowering, this will become !llvm.ptr<i8>
        CtypesType = ctypes.POINTER(ctypes.c_int8)

        if isinstance(tensor_type.encoding, mlir.astnodes.SparseTensorEncoding):
            pointer_type = f"uint{tensor_type.encoding.pointer_bit_width}"
            index_type = f"uint{tensor_type.encoding.index_bit_width}"
        else:
            pointer_type = "uint64"
            index_type = "uint64"
        value_type = {
            "i32": "int32",
            "i64": "int64",
            "f32": "float32",
            "f64": "float64",
        }[tensor_type.element_type.type.name]

        def decoder(arg, ptype=pointer_type, itype=index_type, vtype=value_type) -> int:
            ptr = ctypes.cast(arg, ctypes.c_void_p).value
            return MLIRSparseTensor.from_raw_pointer(ptr, ptype, itype, vtype)

    else:
        np_type, ctypes_type = convert_mlir_atomic_type(tensor_type.element_type)
        fields = [
            ("alloc", ctypes.POINTER(ctypes_type)),
            ("base", ctypes.POINTER(ctypes_type)),
            ("offset", ctypes.c_int64),
        ]
        rank = len(tensor_type.dimensions)
        for prefix in ("size", "stride"):
            for i in range(rank):
                field = (f"{prefix}{i}", ctypes.c_int64)
                fields.append(field)

        class CtypesType(ctypes.Structure):
            _fields_ = fields

        def decoder(result: CtypesType) -> np.ndarray:
            if not isinstance(result, CtypesType):
                raise TypeError(
                    f"Return value {result} expected to have type {CtypesType}."
                )
            dimensions = [
                getattr(result, f"size{dim_index}") for dim_index in range(rank)
            ]
            element_count = reduce(operator.mul, dimensions)
            decoded_result = np.frombuffer(
                (ctypes_type * element_count).from_address(
                    ctypes.addressof(result.base.contents)
                ),
                dtype=np_type,
            ).reshape(dimensions)
            return decoded_result

    return CtypesType, decoder


def return_llvm_pointer_to_ctypes(
    mlir_type: _DIALECT_TYPES["llvm"]["LLVMPtr"],
) -> Tuple[type, Callable]:
    raise NotImplementedError(f"Converting {mlir_type} to ctypes not yet supported.")


def return_llvm_type_to_ctypes(mlir_type: mlir.astnodes.Type) -> Tuple[type, Callable]:
    if isinstance(mlir_type, _DIALECT_TYPES["llvm"]["LLVMPtr"]):
        result = return_llvm_pointer_to_ctypes(mlir_type)
    elif isinstance(mlir_type, _DIALECT_TYPES["llvm"]["LLVMVec"]):
        raise NotImplementedError(
            f"Converting {mlir_type} to ctypes not yet supported."
        )
    else:
        raise NotImplementedError(
            f"Converting {mlir_type} to ctypes not yet supported."
        )
    return result


def return_scalar_to_ctypes(mlir_type: mlir.astnodes.Type) -> Tuple[type, Callable]:
    np_type, ctypes_type = convert_mlir_atomic_type(mlir_type)

    def decoder(result) -> np_type:
        if not np.can_cast(result, np_type):
            raise TypeError(
                f"Return value {result} expected to be castable to {np_type}."
            )
        return np_type(result)

    return ctypes_type, decoder


def return_type_to_ctypes(mlir_type: mlir.astnodes.Type) -> Tuple[type, Callable]:
    """Returns a single ctypes type for a single given MLIR type as well as a decoder."""
    # TODO handle all other child classes of mlir.astnodes.Type
    # TODO consider inlining this if it only has 2 cases

    if isinstance(mlir_type, mlir.astnodes.PrettyDialectType):
        result = return_pretty_dialect_type_to_ctypes(mlir_type)
    elif isinstance(mlir_type, mlir.astnodes.RankedTensorType):
        result = return_tensor_to_ctypes(mlir_type)
    elif any(
        isinstance(mlir_type, llvm_type)
        for llvm_type in _DIALECT_TYPES["llvm"].values()
    ):
        result = return_llvm_type_to_ctypes(mlir_type)
    else:
        result = return_scalar_to_ctypes(mlir_type)

    return result


###########################
# MLIR Input Type Helpers #
###########################

LLVM_DIALECT_TYPE_STRING_TO_CTYPES_POINTER_TYPE: Dict[str, "_ctypes._CData"] = {
    "i8": ctypes.POINTER(ctypes.c_int8),
    # TODO extend this
}


def input_pretty_dialect_type_to_ctypes(
    pretty_type: mlir.astnodes.PrettyDialectType,
) -> Tuple[list, Callable]:
    # Special handling for !llvm.ptr<ptr<i8>>
    if (
        pretty_type.dialect == "llvm"
        and pretty_type.type == "ptr"
        and pretty_type.body[0].type == "ptr"
        and pretty_type.body[0].body[0] == "i8"
    ):
        # Convert to an LLVMPtr<LLVMPtr<i8>>
        LLVMPtr = _DIALECT_TYPES["llvm"]["LLVMPtr"]
        inner_obj = object.__new__(LLVMPtr)
        inner_obj.type = mlir.astnodes.IntegerType(width=8)
        outer_obj = object.__new__(LLVMPtr)
        outer_obj.type = inner_obj
        return input_llvm_pointer_to_ctypes(outer_obj)

    # Pretty dialect types must be special-cased since they're arbitrarily structured.
    raise NotImplementedError(f"Converting {pretty_type} to ctypes not yet supported.")


def input_tensor_to_ctypes(
    tensor_type: mlir.astnodes.RankedTensorType,
) -> Tuple[list, Callable]:

    if tensor_type.encoding is not None:
        # We assume anything with an encoding must be a sparse tensor
        # After lowering, this will become !llvm.ptr<i8>
        ctypes_type = ctypes.POINTER(ctypes.c_int8)
        input_c_types = [ctypes_type]

        def encoder(arg: MLIRSparseTensor) -> List[ctypes_type]:
            # protocol for indicating an object can be interpreted as a MLIRSparseTensor
            if hasattr(arg, "__mlir_sparse__"):
                arg = arg.__mlir_sparse__

            if not isinstance(arg, MLIRSparseTensor):
                raise TypeError(
                    f"{repr(arg)} is expected to be an instance of {MLIRSparseTensor.__qualname__}"
                )
            return [ctypes.cast(arg.data, ctypes_type)]

    else:
        input_c_types = []
        np_type, pointer_type = convert_mlir_atomic_type(
            tensor_type.element_type, return_pointer_type=True
        )
        input_c_types.append(pointer_type)  # allocated pointer (for free())
        input_c_types.append(pointer_type)  # base pointer
        input_c_types.append(ctypes.c_int64)  # offset from base
        for _ in range(2 * len(tensor_type.dimensions)):  # dim sizes and strides
            input_c_types.append(ctypes.c_int64)

        dimensions = [dim.value for dim in tensor_type.dimensions]

        def encoder(arg: np.ndarray) -> list:
            if not isinstance(arg, np.ndarray):
                raise TypeError(
                    f"{repr(arg)} is expected to be an instance of {np.ndarray.__qualname__}"
                )
            if not arg.dtype == np_type:
                raise TypeError(f"{repr(arg)} is expected to have dtype {np_type}")
            if not len(dimensions) == len(arg.shape):
                raise ValueError(
                    f"{repr(arg)} is expected to have rank {len(dimensions)} but has rank {len(arg.shape)}."
                )

            encoded_args = [arg, arg, 0]

            for dim_index, dim_size in enumerate(arg.shape):
                expected_dim_size = dimensions[dim_index]
                if (
                    expected_dim_size is not None
                    and arg.shape[dim_index] != expected_dim_size
                ):
                    raise ValueError(
                        f"{repr(arg)} is expected to have size {expected_dim_size} in the {dim_index}th dimension but has size {arg.shape[dim_index]}."
                    )
                encoded_args.append(arg.shape[dim_index])

            for dimension_index in range(len(arg.shape)):
                stride = arg.strides[dimension_index] // arg.itemsize
                encoded_args.append(stride)

            return encoded_args

    return input_c_types, encoder


def input_llvm_pointer_to_ctypes(
    mlir_type: _DIALECT_TYPES["llvm"]["LLVMPtr"],
) -> Tuple[list, Callable]:
    if (
        isinstance(mlir_type.type, mlir.astnodes.IntegerType)
        and int(mlir_type.type.width) == 8
    ):
        # We blindly assume that an i8 pointer points to a sparse tensor
        # since MLIR's sparse tensor object isn't supported inside an LLVMPtr
        # Instead, we pass a ptr<ptr<i8>> and blindly assume it means a list of sparse tensors
        type_string = mlir_type.type.dump()
        ctypes_type = LLVM_DIALECT_TYPE_STRING_TO_CTYPES_POINTER_TYPE[type_string]
        ctypes_input_types = [ctypes_type]

        def encoder(arg: MLIRSparseTensor) -> list:
            # protocol for indicating an object can be interpreted as a MLIRSparseTensor
            if hasattr(arg, "__mlir_sparse__"):
                arg = arg.__mlir_sparse__
            if not isinstance(arg, MLIRSparseTensor):
                raise TypeError(
                    f"{repr(arg)} is expected to be an instance of {MLIRSparseTensor.__qualname__}"
                )
            return [ctypes.cast(arg.data, ctypes_type)]

    else:
        # Treat the pointer as an array (intended to represent a Python sequence).
        element_ctypes_input_types, element_encoder = input_type_to_ctypes(
            mlir_type.type
        )
        # element_ctypes_input_types has exactly one element type since
        # a pointer type can only point to one type
        (element_ctypes_input_type,) = element_ctypes_input_types
        ctypes_input_types = [ctypes.POINTER(element_ctypes_input_type)]

        def encoder(arg: Union[list, tuple]) -> List[ctypes.Array]:
            if not isinstance(arg, (list, tuple)):
                raise TypeError(
                    f"{repr(arg)} is expected to be an instance of {list} or {tuple}."
                )
            ArrayType = element_ctypes_input_type * len(arg)
            encoded_elements = sum(map(element_encoder, arg), [])
            array = ArrayType(*encoded_elements)
            return [array]

    return ctypes_input_types, encoder


def input_llvm_type_to_ctypes(mlir_type: mlir.astnodes.Type) -> Tuple[list, Callable]:
    if isinstance(mlir_type, _DIALECT_TYPES["llvm"]["LLVMPtr"]):
        result = input_llvm_pointer_to_ctypes(mlir_type)
    elif isinstance(mlir_type, _DIALECT_TYPES["llvm"]["LLVMVec"]):
        raise NotImplementedError(
            f"Converting {mlir_type} to ctypes not yet supported."
        )
    else:
        raise NotImplementedError(
            f"Converting {mlir_type} to ctypes not yet supported."
        )
    return result


def input_scalar_to_ctypes(mlir_type: mlir.astnodes.Type) -> Tuple[list, Callable]:
    np_type, ctypes_type = convert_mlir_atomic_type(mlir_type)
    ctypes_input_types = [ctypes_type]

    def encoder(arg) -> list:
        try:
            can_cast = np.can_cast(arg, np_type, "safe")
        except TypeError:
            can_cast = False
        if not can_cast:
            raise TypeError(f"{repr(arg)} cannot be cast to {np_type}")
        if not isinstance(arg, (np.number, int, float)):
            raise TypeError(
                f"{repr(arg)} is expected to be a scalar with dtype {np_type}"
            )
        return [arg]

    return ctypes_input_types, encoder


def input_type_to_ctypes(mlir_type: mlir.astnodes.Type) -> Tuple[list, Callable]:
    # TODO handle all other child classes of mlir.astnodes.Type
    # TODO consider inlining this if it only has 2 cases

    if isinstance(mlir_type, mlir.astnodes.PrettyDialectType):
        result = input_pretty_dialect_type_to_ctypes(mlir_type)
    elif isinstance(mlir_type, mlir.astnodes.RankedTensorType):
        result = input_tensor_to_ctypes(mlir_type)
    elif any(
        isinstance(mlir_type, llvm_type)
        for llvm_type in _DIALECT_TYPES["llvm"].values()
    ):
        result = input_llvm_type_to_ctypes(mlir_type)
    else:
        result = input_scalar_to_ctypes(mlir_type)
    return result


def mlir_function_input_encoders(
    mlir_function: mlir.astnodes.Function,
) -> Tuple[List[type], List[Callable]]:
    ctypes_input_types = []
    encoders: List[Callable] = []
    for arg in mlir_function.args:
        arg_ctypes_input_types, encoder = input_type_to_ctypes(arg.type)
        ctypes_input_types += arg_ctypes_input_types
        encoders.append(encoder)
    return ctypes_input_types, encoders


####################
# PyMLIR Utilities #
####################


def _resolve_type_aliases(
    node: Any,
    type_alias_table: Dict[str, mlir.astnodes.PrettyDialectType],
) -> Any:
    if isinstance(node, (mlir.astnodes.Node, mlir.astnodes.Module)):
        for field in node._fields_:
            field_value = getattr(node, field)
            field_type = type(field_value)
            if field_type in (list, tuple):
                resolved_field_value = field_type(
                    _resolve_type_aliases(sub_node, type_alias_table)
                    for sub_node in field_value
                )
            elif issubclass(field_type, mlir.astnodes.TypeAlias):
                alias_name = field_value.value
                alias_value = type_alias_table[alias_name]
                resolved_field_value = _resolve_type_aliases(
                    alias_value, type_alias_table
                )
            else:
                resolved_field_value = _resolve_type_aliases(
                    field_value, type_alias_table
                )
            setattr(node, field, resolved_field_value)
    return node


def resolve_type_aliases(module: mlir.astnodes.Module) -> None:
    """Modifies module in place."""
    # TODO this is currently only called by MlirJitEngine.add, which only uses the functions in the
    # module, but we resolve all AST nodes, not just the functions. Consider whether or not it's necessary
    # to resolve all AST nodes besides those of type mlir.astnodes.AttrAlias and mlir.astnodes.Function.
    type_alias_table = {
        alias.name.value: alias.value
        for alias in module.body
        if isinstance(alias, mlir.astnodes.TypeAliasDef)
    }
    if len(type_alias_table) > 0:
        _resolve_type_aliases(module, type_alias_table)
    return


def parse_mlir_functions(
    mlir_text: Union[str, bytes], cli: MlirOptCli
) -> mlir.astnodes.Module:
    if isinstance(mlir_text, str):
        mlir_text = mlir_text.encode()
    # Run text thru mlir-opt to apply aliases and flatten function signatures
    mlir_text = cli.apply_passes(mlir_text, [])
    # Remove everything except function signatures
    func_lines = [
        line.strip() for line in mlir_text.splitlines() if line.lstrip()[:5] == "func "
    ]
    # Add in trailing "}" to make defined functions valid
    func_lines = [line + "}" if line[-1] == "{" else line for line in func_lines]

    mlir_ast = mlir.parse_string("\n".join(func_lines))
    return mlir_ast


#################
# MlirJitEngine #
#################


class MlirJitEngine:
    def __init__(self, cli_executable=None, cli_options=None, llvmlite_engine=None):
        if llvmlite_engine is None:
            # Create a target machine representing the host
            target = llvm.Target.from_default_triple()
            target_machine = target.create_target_machine()
            # And an execution engine with an empty backing module
            backing_mod = llvm.parse_assembly("")
            llvmlite_engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
        self._engine = llvmlite_engine
        self._cli = MlirOptCli(cli_executable, cli_options)
        self.name_to_callable: Dict[str, Callable] = {}

    def _add_mlir_module(
        self, mlir_text: bytes, passes: List[str], debug=False
    ) -> Optional[DebugResult]:
        """Translates MLIR code -> LLVM dialect of MLIR -> actual LLVM IR."""
        if debug:
            try:
                llvm_dialect_text = self._cli.apply_passes(mlir_text, passes)
            except MlirOptError as e:
                return e.debug_result
        else:
            llvm_dialect_text = self._cli.apply_passes(mlir_text, passes)

        mlir_translate_run = subprocess.run(
            ["mlir-translate", "--mlir-to-llvmir"],
            input=llvm_dialect_text.encode(),
            capture_output=True,
        )
        if mlir_translate_run.returncode != 0:
            raise RuntimeError(
                f"mlir-translate failed on the following input: \n{llvm_dialect_text}"
            )

        llvm_ir_text = mlir_translate_run.stdout.decode()

        # Create a LLVM module object from the IR
        mod = llvm.parse_assembly(llvm_ir_text)
        mod.verify()
        # Now add the module and make sure it is ready for execution
        self._engine.add_module(mod)
        self._engine.finalize_object()
        self._engine.run_static_constructors()

        return

    def _generate_zero_or_single_valued_functions(
        self,
        mlir_functions: Iterable[mlir.astnodes.Function],
    ) -> Dict[str, Callable]:
        """Generates a Python callable from a function returning zero values or one value."""
        name_to_callable: Dict[str, Callable] = {}
        for mlir_function in mlir_functions:

            name: str = mlir_function.name.value
            function_pointer: int = self._engine.get_function_address(name)

            if function_pointer == 0:
                raise ValueError(
                    f"The address for the function {repr(name)} is the null pointer."
                )

            mlir_types = mlir_function.result_types
            if not isinstance(mlir_types, list):
                ctypes_return_type, decoder = return_type_to_ctypes(mlir_types)
            elif len(mlir_types) == 0:
                ctypes_return_type = ctypes.c_char  # arbitrary dummy type
                decoder = lambda *args: None
            else:
                raise ValueError(
                    f"MLIR functions with multiple return values should be handled elsewhere."
                )
            ctypes_input_types, encoders = mlir_function_input_encoders(mlir_function)
            c_callable = ctypes.CFUNCTYPE(ctypes_return_type, *ctypes_input_types)(
                function_pointer
            )

            def python_callable(mlir_function, encoders, c_callable, decoder, *args):
                if len(args) != len(mlir_function.args):
                    raise ValueError(
                        f"{name} expected {len(mlir_function.args)} args but got {len(args)}."
                    )
                encoded_args = (encoder(arg) for arg, encoder in zip(args, encoders))
                encoded_args = sum(encoded_args, [])
                encoded_result = c_callable(*encoded_args)
                result = decoder(encoded_result)

                return result

            bound_func = partial(
                python_callable, mlir_function, encoders, c_callable, decoder
            )
            name_to_callable[name] = bound_func

        return name_to_callable

    def _lower_types_to_strings(
        self, ast_types: Iterable[mlir.astnodes.Type], passes: List[str]
    ) -> Dict[str, str]:
        """
        Uses mlir-opt to lower types. This assumes that the passes will
        lower to the LLVM dialect.
        """
        # TODO this costs one mlir-opt subprocess ; can we avoid it?
        # TODO must do string manipulation bc PyMLIR doesn't work with nested LLVM dialect
        # types, e.g. !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)
        ast_type_strings = list({ast_type.dump() for ast_type in ast_types})
        if len(ast_type_strings) == 0:
            return {}
        padding = int(np.ceil(np.log10(len(ast_types))))
        dummy_declarations_string = "\n".join(
            f"func private @func_{i:#0{padding}}() -> {ast_type_string}"
            for i, ast_type_string in enumerate(ast_type_strings)
        ).encode()
        lowered_text = self._cli.apply_passes(dummy_declarations_string, passes)
        lowered_lines = list(filter(len, lowered_text.splitlines()))
        assert lowered_lines[0] == 'module attributes {llvm.data_layout = ""}  {'
        assert lowered_lines[-1] == "}"
        lowered_lines = lowered_lines[1:-1]
        assert all(
            line.endswith(' attributes {sym_visibility = "private"}')
            for line in lowered_lines
        )
        assert all(line.startswith("  llvm.func @func_") for line in lowered_lines)
        lowered_type_strings = [line[24 + padding : -40] for line in lowered_lines]
        return dict(zip(ast_type_strings, lowered_type_strings))

    def _generate_mlir_string_for_multivalued_functions(
        self, mlir_functions: Iterable[mlir.astnodes.Function], passes: List[str]
    ) -> Tuple[str, str]:

        result_type_name_to_lowered_result_type_name = self._lower_types_to_strings(
            sum((mlir_function.result_types for mlir_function in mlir_functions), []),
            passes,
        )

        # Generate conglomerate MLIR string for all wrappers
        mlir_wrapper_texts: List[str] = []
        wrapper_names = [
            mlir_function.name.value + "wrapper" for mlir_function in mlir_functions
        ]
        for mlir_function, wrapper_name in zip(mlir_functions, wrapper_names):
            lowered_result_type_names = [
                result_type_name_to_lowered_result_type_name[result_type.dump()]
                for result_type in mlir_function.result_types
            ]
            joined_result_types = ", ".join(lowered_result_type_names)
            joined_original_arg_signature = ", ".join(
                arg.dump() for arg in mlir_function.args
            )
            declaration = f"func private @{mlir_function.name.value}({joined_original_arg_signature}) -> ({joined_result_types})"

            new_var_names = (f"var{i}" for i in itertools.count())
            arg_names: Set[str] = {arg.name.value for arg in mlir_function.args}
            num_results = len(mlir_function.result_types)
            lowered_result_arg_types = [
                f"!llvm.ptr<{result_type_name}>"
                for result_type_name in lowered_result_type_names
            ]
            result_arg_names = (name for name in new_var_names if name not in arg_names)
            result_arg_names = list(itertools.islice(result_arg_names, num_results))
            wrapper_signature = ", ".join(
                f"%{name}: {result_arg_type}"
                for name, result_arg_type in zip(
                    result_arg_names, lowered_result_arg_types
                )
            )
            if len(mlir_function.args) > 0:
                wrapper_signature += ", " + joined_original_arg_signature
            joined_arg_types = ", ".join(arg.type.dump() for arg in mlir_function.args)
            joined_arg_names = ", ".join(arg.name.dump() for arg in mlir_function.args)
            aggregate_result_var_name = next(new_var_names)
            body_lines = itertools.chain(
                [
                    f"%{aggregate_result_var_name}:{num_results} "
                    f"= call @{mlir_function.name.value}({joined_arg_names}) "
                    f": ({joined_arg_types}) -> ({joined_result_types})"
                ],
                (
                    f"llvm.store %{aggregate_result_var_name}#{i}, %{result_arg_name} : {result_arg_type}"
                    for i, (result_arg_name, result_arg_type) in enumerate(
                        zip(result_arg_names, lowered_result_arg_types)
                    )
                ),
            )
            body = "\n  ".join(body_lines)

            mlir_wrapper_text = f"""
{declaration}

func @{wrapper_name}({wrapper_signature}) -> () {{
  {body}
  return
}}
"""
            mlir_wrapper_texts.append(mlir_wrapper_text)

        mlir_text = "\n".join(mlir_wrapper_texts)
        return mlir_text, wrapper_names

    def _generate_multivalued_functions(
        self, mlir_functions: Iterable[mlir.astnodes.Function], passes: List[str]
    ) -> Dict[str, Callable]:
        name_to_callable: Dict[str, Callable] = {}

        mlir_text, wrapper_names = self._generate_mlir_string_for_multivalued_functions(
            mlir_functions, passes
        )

        # this is guaranteed to not fail since the user-provided
        # code was already added (failures would occur then)
        self._add_mlir_module(mlir_text.encode(), passes)

        # Generate callables
        for mlir_function, wrapper_name in zip(mlir_functions, wrapper_names):
            ctypes_input_types, input_encoders = mlir_function_input_encoders(
                mlir_function
            )
            ctypes_result_arg_pointer_types = []
            ctypes_result_arg_types = []
            decoders = []
            for result_type in mlir_function.result_types:
                result_type_ctypes_type, decoder = return_type_to_ctypes(result_type)
                ctypes_result_arg_pointer_types.append(
                    ctypes.POINTER(result_type_ctypes_type)
                )
                ctypes_result_arg_types.append(result_type_ctypes_type)
                decoders.append(decoder)

            function_pointer: int = self._engine.get_function_address(wrapper_name)
            if function_pointer == 0:
                raise ValueError(
                    f"The address for the function {repr(wrapper_name)} is the null pointer."
                )
            c_callable = ctypes.CFUNCTYPE(
                None, *ctypes_result_arg_pointer_types, *ctypes_input_types
            )(function_pointer)

            def python_callable(
                mlir_function,
                ctypes_result_arg_types,
                input_encoders,
                c_callable,
                decoders,
                *args,
            ) -> tuple:
                if len(args) != len(mlir_function.args):
                    raise ValueError(
                        f"{mlir_function.name.value} expected {len(mlir_function.args)} args but got {len(args)}."
                    )
                result_arg_values = [
                    result_arg_type() for result_arg_type in ctypes_result_arg_types
                ]
                result_arg_pointers = [
                    ctypes.pointer(value) for value in result_arg_values
                ]
                encoded_args = (
                    encoder(arg) for arg, encoder in zip(args, input_encoders)
                )
                encoded_args = itertools.chain(*encoded_args)
                returned_result = c_callable(*result_arg_pointers, *encoded_args)
                assert returned_result is None
                return tuple(
                    decoder(result_arg_pointer.contents)
                    for decoder, result_arg_pointer in zip(
                        decoders, result_arg_pointers
                    )
                )

            bound_func = partial(
                python_callable,
                mlir_function,
                ctypes_result_arg_types,
                input_encoders,
                c_callable,
                decoders,
            )
            name_to_callable[mlir_function.name.value] = bound_func

        return name_to_callable

    def _walk_module(self, mlir_ast):
        """Recursively walks an MLIR Module, yielding all non-Module objects"""
        assert isinstance(
            mlir_ast, mlir.astnodes.Module
        ), f"Cannot walk a {type(mlir_ast)}; expected a Module"
        for item in mlir_ast.body:
            if isinstance(item, mlir.astnodes.Module):
                yield from self._walk_module(item)
            else:
                yield item

    def add(
        self, mlir_text: Union[str, bytes], passes: Tuple[str], debug=False
    ) -> Union[List[str], DebugResult]:
        """List of new function names added."""
        if isinstance(mlir_text, str):
            mlir_text = mlir_text.encode()

        optional_debug_result = self._add_mlir_module(mlir_text, passes, debug)
        if isinstance(optional_debug_result, DebugResult):
            return optional_debug_result

        function_names: List[str] = []
        mlir_ast = parse_mlir_functions(mlir_text, self._cli)
        mlir_functions: List[mlir.astnodes.Function] = [
            obj
            for obj in self._walk_module(mlir_ast)
            if isinstance(obj, mlir.astnodes.Function) and obj.visibility == "public"
        ]

        # Separate zero/single return valued funcs from multivalued funcs
        zero_or_single_valued_funcs = []
        multivalued_funcs = []
        for mlir_function in mlir_functions:
            name: str = mlir_function.name.value
            if name in self.name_to_callable:
                raise ValueError(f"The function {repr(name)} is already defined.")
            function_names.append(name)

            if (
                not isinstance(mlir_function.result_types, list)
                or len(mlir_function.result_types) == 0
            ):
                zero_or_single_valued_funcs.append(mlir_function)
            else:
                multivalued_funcs.append(mlir_function)

        # Compile & add functions
        name_to_zero_or_single_callable = (
            self._generate_zero_or_single_valued_functions(zero_or_single_valued_funcs)
        )
        # TODO we currently need two separate compilations ; we can avoid this if
        # we can use PyMLIR to simply add on the extra functions/wrappers we need
        # to handle multivalued functions (we would just parse for an AST, add onto
        # the AST, and then dump the AST). This is currently not possible since
        # PyMLIR can't parse all MLIR. It'd also be difficult without an IR
        # builder (which is currently a PyMLIR WIP).
        name_to_multicallable = self._generate_multivalued_functions(
            multivalued_funcs, passes
        )

        for name, python_callable in itertools.chain(
            name_to_zero_or_single_callable.items(), name_to_multicallable.items()
        ):
            # python_callable only tracks the function pointer, not the
            # function itself. If self._engine, gets garbage collected,
            # we get a seg fault. Thus, we must keep the engine alive.
            setattr(python_callable, "jit_engine", self)

            self.name_to_callable[name] = python_callable

        return function_names

    def __getitem__(self, func_name: str) -> Callable:
        return self.name_to_callable[func_name]

    def __getattr__(self, func_name: str) -> Callable:
        return self[func_name]
