"""
An IR builder for MLIR.
"""

###########
# Imports #
###########

import jinja2
from collections import OrderedDict
from .sparse_utils import MLIRSparseTensor
from .functions import BaseFunction
from .engine import parse_mlir_string

from typing import Dict, List, Tuple

#############
# IR Builer #
#############


class MLIRVar:

    """Input variables to be used by MLIRFunctionBuilder."""

    def __init__(self, var_name: str, var_type: str) -> None:
        """var_type is the type as represented in MLIR code."""
        self.var_name = var_name
        self.var_type = var_type
        return

    @property
    def var_string(self) -> str:
        return f"%{self.var_name}"


class MLIRFunctionBuilder(BaseFunction):

    # TODO consider making this class have a method to return a BaseFunction and not be a subclass of BaseFunction
    # In other words, make it a "builder" that creates BaseFunction subclasses or instances instead of being a BaseFunction itself

    function_wrapper_text = jinja2.Template(
        """
      func {% if private_func %}private{% endif %}@{{func_name}}({{signature}}) -> {{return_type}} {
        {{statements}}
      }
"""
    )  # TODO make this hard-coded indentation not hard-coded

    def __init__(
        self, func_name: str, *input_variables: MLIRVar, return_type: str
    ) -> None:
        # TODO mlir functions can return zero or more results https://mlir.llvm.org/docs/LangRef/#operations
        # handle the cases where the number of return types is not 1
        self.func_name = func_name
        self.input_variables = input_variables
        self.return_type = return_type

        self.var_name_counter = 0
        self.function_body_statements: List[str] = []

        # function_name -> (function_mlir_definition, input_mlir_types, return_mlir_type)
        self.needed_function_table: Dict[
            str, Tuple[str, List[str], str]
        ] = OrderedDict()

        return

    def compile(self, engine=None, passes=None):
        func = super().compile(engine, passes)

        if self.return_type == "!llvm.ptr<i8>":

            def func_wrapper(*args, **kwargs):
                # Find an MLIRSparseTensor in the inputs, then use its dtypes
                for x in list(args) + list(kwargs.values()):
                    if isinstance(x, MLIRSparseTensor):
                        break
                else:
                    msg = (
                        "Unable to find an MLIRSparseTensor in the inputs.\n"
                        "Cannot return MLIRSparseTensor from raw pointer"
                    )
                    raise TypeError(msg)

                ptr = func(*args, **kwargs)
                tensor = MLIRSparseTensor.from_raw_pointer(
                    ptr, x.pointer_dtype, x.index_dtype, x.value_dtype
                )
                return tensor

            return func_wrapper

        return func

    def get_mlir(self, make_private=True) -> str:
        joined_statements = "\n".join(self.function_body_statements)
        signature = ", ".join(
            f"{var.var_string}: {var.var_type}" for var in self.input_variables
        )
        needed_function_definitions = "\n\n".join(
            func_def for func_def, _, _ in self.needed_function_table.values()
        )
        return needed_function_definitions + self.function_wrapper_text.render(
            private_func=make_private,
            func_name=self.func_name,
            signature=signature,
            return_type=self.return_type,
            statements=joined_statements,
        )

    def return_var(self, returned_value: MLIRVar) -> None:
        if self.return_type != returned_value.var_type:
            raise TypeError(
                f"Type of {returned_value} does not match function return type of {self.return_type}."
            )
        returned_value_string = (
            returned_value.var_string
            if isinstance(returned_value, MLIRVar)
            else str(returned_value)
        )
        statement = f"return {returned_value_string} : {self.return_type}"
        self.function_body_statements.append(statement)
        return

    def call(
        self,
        function: BaseFunction,
        *inputs: MLIRVar,
    ) -> MLIRVar:

        if function.func_name in self.needed_function_table:
            function_mlir_text, input_types, return_type = self.needed_function_table[
                function.func_name
            ]
            assert function.get_mlir(make_private=True) == function_mlir_text
        else:
            function_mlir_text = function.get_mlir(make_private=True)
            (function_ast,) = parse_mlir_string(function_mlir_text).body
            input_types = [arg.type.dump() for arg in function_ast.args]
            return_type = function_ast.result_types.dump()
            self.needed_function_table[function.func_name] = (
                function_mlir_text,
                input_types,
                return_type,
            )

        result_var = MLIRVar(f"var_{self.var_name_counter}", return_type)
        self.var_name_counter += 1
        statement = "".join(
            [
                f"{result_var.var_string} = call @{function.func_name}(",
                ", ".join(
                    f"{input_val.var_string}"
                    if isinstance(input_val, MLIRVar)
                    else str(input_val)
                    for input_val in inputs
                ),
                ") : (",
                ", ".join(input_types),
                ") -> ",
                return_type,
            ]
        )
        self.function_body_statements.append(statement)
        return result_var
