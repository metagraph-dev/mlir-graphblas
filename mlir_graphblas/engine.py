import os
import re
import subprocess
import mlir
import ctypes
import glob
import operator
import llvmlite.binding as llvm
import numpy as np
from .sparse_utils import MLIRSparseTensor
from functools import reduce
from .cli import MlirOptCli, MlirOptError
from typing import Tuple, List, Dict, Callable, Union, Any

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
    raise NotImplementedError(f"Converting {mlir_type} to ctypes not yet supported.")


def return_tensor_to_ctypes(
    tensor_type: mlir.astnodes.RankedTensorType,
) -> Tuple[type, Callable]:
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
        dimensions = [getattr(result, f"size{dim_index}") for dim_index in range(rank)]
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
    if (
        isinstance(mlir_type.type, mlir.astnodes.IntegerType)
        and int(mlir_type.type.width) == 8
    ):
        type_string = mlir_type.type.dump()
        ctypes_type = LLVM_DIALECT_TYPE_STRING_TO_CTYPES_POINTER_TYPE[type_string]

        def decoder(arg) -> int:
            # TODO we blindly assume that an i8 pointer points to a sparse tensor
            # since MLIR's sparse tensor support is currently up-in-the-air and this
            # is how they currently handle sparse tensors

            # Return the pointer address as a Python int
            # To create the MLIRSparseTensor, use MLIRSparseTensor.from_raw_pointer(),
            # which we can't do here because we are missing dtype information.
            return ctypes.cast(arg, ctypes.c_void_p).value

    else:
        raise NotImplementedError(
            f"Converting {mlir_type} to ctypes not yet supported."
        )

    return ctypes_type, decoder


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

    def decoder(result):
        if not np.can_cast(result, np_type):
            raise TypeError(
                f"Return value {result} expected to be castable to {np_type}."
            )
        return result

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


def mlir_function_return_decoder(
    mlir_function: mlir.astnodes.Function,
) -> Tuple[type, Callable]:
    mlir_types = mlir_function.result_types

    if not isinstance(mlir_types, list):
        ctypes_type, decoder = return_type_to_ctypes(mlir_types)
    elif len(mlir_types) == 0:
        ctypes_type = ctypes.c_char  # arbitrary dummy type
        decoder = lambda *args: None
    else:
        ctypes_return_types, element_decoders = zip(
            *map(return_type_to_ctypes, mlir_types)
        )
        field_names = [f"result_{i}" for i in range(len(ctypes_return_types))]

        class ctypes_type(ctypes.Structure):
            _fields_ = [
                (field_name, return_type)
                for field_name, return_type in zip(field_names, ctypes_return_types)
            ]

        def decoder(encoded_result: ctypes_type) -> tuple:
            encoded_elements = (
                getattr(encoded_result, field_name) for field_name in field_names
            )
            return tuple(
                element_decoder(encoded_element)
                for element_decoder, encoded_element in zip(
                    element_decoders, encoded_elements
                )
            )

    return ctypes_type, decoder


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
    # Pretty dialect types must be special-cased since they're arbitrarily structured.
    raise NotImplementedError(f"Converting {pretty_type} to ctypes not yet supported.")


def input_tensor_to_ctypes(
    tensor_type: mlir.astnodes.RankedTensorType,
) -> Tuple[list, Callable]:

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
        # TODO we blindly assume that an i8 pointer points to a sparse tensor
        # since MLIR's sparse tensor support is currently up-in-the-air and this
        # is how they currently handle sparse tensors
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
        # TODO treat the pointer as an array (intended to represent a Python sequence).
        element_ctypes_input_types, element_encoder = input_type_to_ctypes(
            mlir_type.type
        )
        # element_ctypes_input_types has exactly one element type since
        # a pointer type can only point to one type
        (element_ctypes_input_type,) = element_ctypes_input_types
        ctypes_input_types = [ctypes.POINTER(element_ctypes_input_type)]

        def encoder(arg: Union[list, tuple]) -> list:
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


def _preprocess_mlir(mlir_text: str):
    START_MARKER = "// pymlir-skip: begin"
    END_MARKER = "// pymlir-skip: end"
    PATTERN = START_MARKER + r".*?" + END_MARKER
    regex = re.compile(PATTERN, flags=re.DOTALL)
    preprocessed_mlir_text = regex.sub("", mlir_text)
    return preprocessed_mlir_text


def parse_mlir_string(mlir_text: Union[str, bytes]) -> mlir.astnodes.Module:
    if isinstance(mlir_text, bytes):
        mlir_text = mlir_text.decode()
    mlir_text = _preprocess_mlir(mlir_text)
    mlir_ast = mlir.parse_string(mlir_text)
    resolve_type_aliases(mlir_ast)
    return mlir_ast


#################
# MlirJitEngine #
#################


class MlirJitEngine:
    def __init__(self, llvmlite_engine=None):
        if llvmlite_engine is None:
            # Create a target machine representing the host
            target = llvm.Target.from_default_triple()
            target_machine = target.create_target_machine()
            # And an execution engine with an empty backing module
            backing_mod = llvm.parse_assembly("")
            llvmlite_engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
        self._engine = llvmlite_engine
        self._cli = MlirOptCli()
        self.name_to_callable: Dict[str, Callable] = {}

    def _generate_python_callable(
        self, mlir_function: mlir.astnodes.Function
    ) -> Callable:
        name: str = mlir_function.name.value
        function_pointer: int = self._engine.get_function_address(name)

        if function_pointer == 0:
            raise ValueError(
                f"The address for the function {repr(name)} is the null pointer."
            )

        ctypes_return_type, decoder = mlir_function_return_decoder(mlir_function)
        ctypes_input_types, encoders = mlir_function_input_encoders(mlir_function)
        c_callable = ctypes.CFUNCTYPE(ctypes_return_type, *ctypes_input_types)(
            function_pointer
        )

        def python_callable(*args):
            if len(args) != len(mlir_function.args):
                raise ValueError(
                    f"{name} expected {len(mlir_function.args)} args but got {len(args)}."
                )
            encoded_args = (encoder(arg) for arg, encoder in zip(args, encoders))
            encoded_args = sum(encoded_args, [])
            encoded_result = c_callable(*encoded_args)
            result = decoder(encoded_result)

            return result

        # python_callable only tracks the function pointer, not the
        # function itself. If self._engine, gets garbage collected,
        # we get a seg fault. Thus, we must keep the engine alive.
        setattr(python_callable, "jit_engine", self)

        return python_callable

    def add(
        self, mlir_text: Union[str, bytes], passes: List[str], debug=False
    ) -> List[str]:
        """List of new function names added."""
        if isinstance(mlir_text, str):
            mlir_text = mlir_text.encode()
        if debug:
            try:
                llvmir_text = self._cli.apply_passes(mlir_text, passes)
            except MlirOptError as e:
                return e.debug_result
        else:
            llvmir_text = self._cli.apply_passes(mlir_text, passes)
        mlir_translate_run = subprocess.run(
            ["mlir-translate", "--mlir-to-llvmir"],
            input=llvmir_text.encode(),
            capture_output=True,
        )
        llvm_text = mlir_translate_run.stdout.decode()

        # Create a LLVM module object from the IR
        mod = llvm.parse_assembly(llvm_text)
        mod.verify()
        # Now add the module and make sure it is ready for execution
        self._engine.add_module(mod)
        self._engine.finalize_object()
        self._engine.run_static_constructors()

        # Generate Python callables
        mlir_ast = parse_mlir_string(mlir_text)

        mlir_functions = filter(
            lambda e: isinstance(e, mlir.astnodes.Function)
            and e.visibility == "public",
            mlir_ast.body,
        )

        function_names: List[str] = []
        for mlir_function in mlir_functions:
            name: str = mlir_function.name.value
            if name in self.name_to_callable:
                raise ValueError(f"The function {repr(name)} is already defined.")
            function_names.append(name)
            python_callable = self._generate_python_callable(mlir_function)
            self.name_to_callable[name] = python_callable

        return function_names

    def __getitem__(self, func_name: str) -> Callable:
        return self.name_to_callable[func_name]

    def __getattr__(self, func_name: str) -> Callable:
        return self[func_name]
