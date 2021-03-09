import subprocess
import mlir
import ctypes
import operator
import llvmlite.binding as llvm
import numpy as np
from functools import reduce
from .cli import MlirOptCli, MlirOptError
from typing import Tuple, List, Dict, Callable, Union

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
    First return value is a subclass of _ctypes._CData .
    Second return value is a subclass of np.generic .
    """

    # TODO Add a custom hash/eq method for mlir.astnodes.Node so that we can put
    # all the mlir.astnodes.*Type instances into a dict
    # it'll obviate the need for this section of code.
    np_type = None
    if isinstance(mlir_type, mlir.astnodes.FloatType):
        np_type = MLIR_FLOAT_ENUM_TO_NP_TYPE[mlir_type.type]
    elif isinstance(mlir_type, mlir.astnodes.IntegerType):
        np_type = INTEGER_WIDTH_TO_NP_TYPE[int(mlir_type.width)]
    if np_type is None:
        raise ValueError(f"Could not determine numpy type corresonding to {mlir_type}")

    ctypes_type = NP_TYPE_TO_CTYPES_TYPE[np_type]
    if return_pointer_type:
        ctypes_type = np.ctypeslib.ndpointer(np_type)

    return np_type, ctypes_type


############################
# MLIR Return Type Helpers #
############################


def return_scalar_to_ctypes(mlir_type: mlir.astnodes.Type) -> Tuple[type, Callable]:
    np_type, ctypes_type = convert_mlir_atomic_type(mlir_type)

    def decoder(result):
        if not np.can_cast(result, np_type):
            raise TypeError(
                f"Return value {result} expected to be castable to {np_type}."
            )
        return result

    return ctypes_type, decoder


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

    def decoder(result: CtypesType):
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


def return_type_to_ctypes(mlir_type: mlir.astnodes.Type) -> Tuple[type, Callable]:
    """Returns a ctypes type for the given MLIR type."""
    # TODO handle all other child classes of mlir.astnodes.Type
    # TODO consider inlining this if it only has 2 cases
    if isinstance(mlir_type, mlir.astnodes.RankedTensorType):
        result = return_tensor_to_ctypes(mlir_type)
    else:
        result = return_scalar_to_ctypes(mlir_type)
    return result


###########################
# MLIR Input Type Helpers #
###########################


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
            raise TypeError(f"{arg} is expected to be an instance of {np.ndarray}")
        if not arg.dtype == np_type:
            raise TypeError(f"{arg} is expected to have dtype {np_type}")
        if not len(dimensions) == len(arg.shape):
            raise ValueError(
                f"{arg} is expected to have rank {len(dimensions)} but has rank {len(arg.shape)}."
            )

        encoded_args = [arg, arg, 0]

        for dim_index, dim_size in enumerate(arg.shape):
            expected_dim_size = dimensions[dim_index]
            if (
                expected_dim_size is not None
                and arg.shape[dim_index] != expected_dim_size
            ):
                raise ValueError(
                    f"{arg} is expected to have size {expected_dim_size} in the {dim_index}th dimension but has size {arg.shape[dim_index]}."
                )
            encoded_args.append(arg.shape[dim_index])

        for dimension_index in range(len(arg.shape)):
            stride = arg.strides[dimension_index] // arg.itemsize
            encoded_args.append(stride)

        return encoded_args

    return input_c_types, encoder


def input_scalar_to_ctypes(mlir_type: mlir.astnodes.Type) -> Tuple[list, Callable]:
    np_type, ctypes_type = convert_mlir_atomic_type(mlir_type)
    ctypes_input_types = [ctypes_type]

    def encoder(arg) -> list:
        if not np.can_cast(arg, np_type):
            raise TypeError(f"{arg} cannot be cast to {np_type}")
        return [arg]

    return ctypes_input_types, encoder


def input_type_to_ctypes(mlir_type: mlir.astnodes.Type) -> Tuple[list, Callable]:
    # TODO handle all other child classes of mlir.astnodes.Type
    # TODO consider inlining this if it only has 2 cases
    if isinstance(mlir_type, mlir.astnodes.RankedTensorType):
        result = input_tensor_to_ctypes(mlir_type)
    else:
        result = input_scalar_to_ctypes(mlir_type)
    return result


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

        mlir_result_type = mlir_function.result_types
        ctypes_return_type, decoder = return_type_to_ctypes(mlir_result_type)

        ctypes_input_types = []
        encoders: List[Callable] = []
        for arg in mlir_function.args:
            arg_ctypes_input_types, encoder = input_type_to_ctypes(arg.type)
            ctypes_input_types += arg_ctypes_input_types
            encoders.append(encoder)

        c_callable = ctypes.CFUNCTYPE(ctypes_return_type, *ctypes_input_types)(
            function_pointer
        )

        def python_callable(*args):
            if len(args) != len(mlir_function.args):
                raise ValueError(
                    f"{name} expected {len(mlir_function.args)} args but got {len(args)}."
                )
            encoded_args = (encoder(arg) for arg, encoder in zip(args, encoders))
            encoded_args = reduce(operator.add, encoded_args)
            result = c_callable(*encoded_args)
            result = decoder(result)

            return result

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
        mlir_ast = mlir.parse_string(mlir_text.decode())
        mlir_functions = filter(
            lambda e: isinstance(e, mlir.astnodes.Function), mlir_ast.body
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
