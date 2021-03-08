import subprocess
import mlir
import ctypes
import operator
import llvmlite.binding as llvm
import numpy as np
from functools import reduce
from .cli import MlirOptCli, MlirOptError
from typing import List, Dict, Callable, Union

# TODO sweep for uses  of  "dtype" and make sure they're acutally for dtypes and not numpy classes.

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

# TODO if any of these global dictionaries are used in multiple places, see if we can abstract stuff out

MLIR_FLOAT_ENUM_TO_NP_TYPE = {
    mlir.astnodes.FloatTypeEnum.f16: np.float16,
    mlir.astnodes.FloatTypeEnum.f32: np.float32,
    mlir.astnodes.FloatTypeEnum.f64: np.float64,
    # TODO handle mlir.astnodes.FloatTypeEnum.bf16
}

INTEGER_WIDTH_TO_NP_TYPE = {  # TODO is there some nice accessor for this?
    8: np.int8,
    16: np.int16,
    32: np.int32,
    64: np.int64,
    # TODO do we need np.longlong here?
}


def mlir_atomic_type_to_np_type(mlir_type: mlir.astnodes.Type) -> type:
    # TODO Add a custom hash/eq method for mlir.astnodes.Node so that we can put
    # all the mlir.astnodes.*Type instances into a dict
    # it'll obviate the need for this function.
    result = None
    if isinstance(mlir_type, mlir.astnodes.FloatType):
        result = MLIR_FLOAT_ENUM_TO_NP_TYPE[mlir_type.type]
    elif isinstance(mlir_type, mlir.astnodes.IntegerType):
        result = INTEGER_WIDTH_TO_NP_TYPE[int(mlir_type.width)]
    # TODO handle other types, e.g. int
    if result is None:
        raise ValueError(f"Could not determine numpy type corresonding to {mlir_type}")
    return result


NP_D_TYPE_TO_CTYPES_TYPE = {
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

############################
# MLIR Return Type Helpers #
############################


def return_scalar_to_ctypes(mlir_type: mlir.astnodes.Type):
    np_type = mlir_atomic_type_to_np_type(mlir_type)
    ctypes_type = NP_D_TYPE_TO_CTYPES_TYPE[np_type]
    return ctypes_type


def return_tensor_to_ctypes(tensor_type: mlir.astnodes.RankedTensorType):
    # TODO support unranked tensors
    np_type = mlir_atomic_type_to_np_type(
        tensor_type.element_type
    )  # TODO can we use return_scalar_to_ctypes here?
    ctypes_type = NP_D_TYPE_TO_CTYPES_TYPE[np_type]
    fields = [
        ("alloc", ctypes.POINTER(element_ctypes_type)),
        ("base", ctypes.POINTER(element_ctypes_type)),
        ("offset", ctypes.c_int64),
    ]
    for prefix in ("size", "stride"):
        for i in range(len(tensor_type.dimensions)):
            field = (f"{prefix}{i}", ctypes.c_int64)
            fields.append(field)

    class CtypesType(ctypes.Structure):
        _fields_ = fields

    return CtypesType


def return_type_to_ctypes(mlir_type: mlir.astnodes.Type):
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


def input_tensor_to_ctypes(mlir_type: mlir.astnodes.RankedTensorType,) -> list:
    # TODO handle other tensor types
    input_c_types = []
    np_type = mlir_atomic_type_to_np_type(mlir_type.element_type)
    pointer_type = np.ctypeslib.ndpointer(np_type)
    input_c_types.append(pointer_type)  # allocated pointer (for free())
    input_c_types.append(pointer_type)  # base pointer
    input_c_types.append(ctypes.c_int64)  # offset from base
    for _ in range(2 * len(mlir_type.dimensions)):  # dim sizes and strides
        input_c_types.append(ctypes.c_int64)
    return input_c_types


def input_scalar_to_ctypes(mlir_type: mlir.astnodes.Type) -> list:
    np_type = mlir_atomic_type_to_np_type(mlir_type)
    ctypes_type = NP_D_TYPE_TO_CTYPES_TYPE[np_type]
    ctypes_input_types = [ctypes_type]
    return ctypes_input_types


def input_type_to_ctypes(mlir_type: mlir.astnodes.Type) -> list:
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
        ctypes_return_type = return_type_to_ctypes(mlir_result_type)

        ctypes_input_types = []
        for arg in mlir_function.args:
            ctypes_input_types += input_type_to_ctypes(arg.type)

        c_callable = ctypes.CFUNCTYPE(ctypes_return_type, *ctypes_input_types)(
            function_pointer
        )

        def python_callable(*args):
            if len(args) != len(mlir_function.args):
                raise ValueError(
                    f"{name} expected {len(mlir_function.args)} args but got {len(args)}."
                )
            updated_args = []
            for arg_index, (arg, mlir_arg) in enumerate(zip(args, mlir_function.args)):
                if isinstance(mlir_arg.type, mlir.astnodes.RankedTensorType):
                    if not isinstance(arg, np.ndarray):
                        raise TypeError(
                            f"Argument {arg_index} expected to be a numpy array."
                        )
                    # TODO check the dtype as well
                    # TODO check the dimensions
                    # TODO check if the expected mlir type is tensor<?xf32> that we get a numpy array
                    updated_args.append(arg)  # allocated pointer (for free())
                    updated_args.append(arg)  # base pointer
                    updated_args.append(0)  # offset from base
                    for dimension_size in arg.shape:
                        updated_args.append(arg.shape[0])
                    for dimension_index in range(len(arg.shape)):
                        updated_args.append(
                            arg.strides[dimension_index] // arg.itemsize
                        )
                else:
                    # TODO check this type
                    updated_args.append(arg)
            result = c_callable(*updated_args)
            if isinstance(mlir_result_type, mlir.astnodes.RankedTensorType):
                dtype = mlir_atomic_type_to_np_type(
                    mlir_result_type.element_type
                )  # TODO can we use input_type_to_ctypes here?
                ctypes_type = NP_D_TYPE_TO_CTYPES_TYPE[dtype]
                dimensions = [
                    dimension.value for dimension in mlir_result_type.dimensions
                ]
                if None in dimensions:  # TODO handle this case
                    raise NotImplementedError(
                        f"Currently can't handle non-fixed-size outputs."
                    )
                element_count = reduce(operator.mul, dimensions)
                result = np.frombuffer(
                    (ctypes_type * element_count).from_address(
                        ctypes.addressof(result.base.contents)
                    ),
                    dtype=dtype,
                ).reshape(dimensions)
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
        mlir_translate_process = subprocess.run(
            ["mlir-translate", "--mlir-to-llvmir"],
            input=llvmir_text.encode(),
            capture_output=True,
        )
        llvm_text = mlir_translate_process.stdout.decode()
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
