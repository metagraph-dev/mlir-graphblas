"""
An IR builder for MLIR.
"""

###########
# Imports #
###########

import jinja2
import mlir
from contextlib import contextmanager
from collections import OrderedDict
from .sparse_utils import MLIRSparseTensor
from .functions import BaseFunction
from .engine import parse_mlir_string

from typing import Dict, List, Tuple, Sequence, Generator, Optional, Union

#############
# IR Builer #
#############


def _canonicalize_mlir_type_string(type_string: str) -> str:
    """Canonicalize by round-tripping through PyMLIR."""
    return mlir.parse_string(f"!dummy_alias = type {type_string}").body[0].value.dump()


def mlir_type_strings_equal(type_1: str, type_2: str) -> bool:
    # The type strings can vacuously differ (e.g. in white space).
    # Thus, we must deterministically canonicalize the type strings.
    if type_1 == type_2:
        return True
    elif _canonicalize_mlir_type_string(type_1) == _canonicalize_mlir_type_string(
        type_2
    ):
        return True
    return False


class MLIRVar:

    """Input variables to be used by MLIRFunctionBuilder."""

    # TODO consider making a separate class for single type MLIR variables
    # and multi-type MLIR variables.

    def __init__(self, var_name: str, *var_types: str) -> None:
        """Elements of var_types are types as represented in MLIR code."""
        assert len(var_types) > 0
        self.var_name = var_name
        self.var_types = var_types
        return

    def __eq__(self, other: "MLIRVar") -> bool:
        return (
            isinstance(other, MLIRVar)
            and self.var_name == other.var_name
            and all(
                mlir_type_strings_equal(*type_pair)
                for type_pair in zip(self.var_types, other.var_types)
            )
        )

    @property
    def num_values(self) -> int:
        return len(self.var_types)

    @property
    def var_type(self) -> str:
        if not len(self.var_types) == 1:
            raise TypeError(f"{self} has multiple types.")
        (var_type,) = self.var_types
        return var_type

    def __repr__(self) -> str:
        attributes_string = ", ".join(
            f"{k}={repr(self.__dict__[k])}" for k in sorted(self.__dict__.keys())
        )
        return f"{self.__class__.__name__}({attributes_string})"

    def assign_string(self) -> str:
        # TODO specify number of expected returned values as arg and sanity check
        answer = f"%{self.var_name}"
        if self.num_values != 1:
            answer += f":{self.num_values}"
        return answer

    def access_string(self, index: Optional[int] = None) -> str:
        answer = f"%{self.var_name}"
        if index is None:
            if self.num_values != 1:
                raise ValueError(
                    "Attempting to use variable containing multiple values as a singleton variable."
                )
        elif not isinstance(index, int):
            raise TypeError(f"{self.__class__} only supports integer indexing.")
        elif index < 0 or self.num_values <= index:
            raise IndexError(
                f"Index must be an integer in the range [0, {self.num_values})."
            )
        else:
            answer += f"#{index}"
        return answer

    def __getitem__(self, index: int) -> Tuple["MLIRVar", int]:
        # Convenience function for use with methods like MLIRFunctionBuilder.return_vars
        if not isinstance(index, int):
            raise TypeError(
                f"{self.__class__.__name__} indices must be integers, not {type(index)}."
            )
        return (self, index)


class MLIRFunctionBuilder(BaseFunction):

    # TODO consider making this class have a method to return a BaseFunction
    # and not be a subclass of BaseFunction
    # In other words, make it a "builder" that creates BaseFunction subclasses
    # or instances instead of being a BaseFunction itself
    # TODO Consider defining all the BaseFunction instances in functions.py via
    # this class.

    # TODO move this indentation to some parent class
    default_indentation_size = 6
    indentation_delta_size = 2
    function_wrapper_text = jinja2.Template(
        "\n"
        + (" " * default_indentation_size)
        + "func {% if private_func %}private{% endif %}@{{func_name}}({{signature}}) -> {{return_type}} {"
        + "\n"
        + "{{statements}}"
        + "\n"
        + (" " * default_indentation_size)
        + "}"
    )

    def __init__(
        self, func_name: str, input_vars: Sequence[MLIRVar], return_types: Sequence[str]
    ) -> None:
        # TODO mlir functions can return zero or more results https://mlir.llvm.org/docs/LangRef/#operations
        # handle the cases where the number of return types is not 1
        self.func_name = func_name
        self.input_vars = input_vars
        self.return_types = return_types

        self.var_name_counter = 0
        self.function_body_statements: List[str] = []

        # function_name -> (function_mlir_definition, input_mlir_types, return_mlir_type)
        self.needed_function_table: Dict[
            str, Tuple[str, List[str], str]
        ] = OrderedDict()

        self.indentation_level = 1
        return

    #######################################
    # MLIR Generation/Compilation Methods #
    #######################################

    def compile(self, engine=None, passes=None):
        func = super().compile(engine, passes)

        indices_of_returned_sparse_tensors = {
            i
            for i, return_type in enumerate(self.return_types)
            if mlir_type_strings_equal(return_type, "!llvm.ptr<i8>")
        }

        if len(indices_of_returned_sparse_tensors) != 0:

            def func_wrapper(*args, **kwargs):
                # Find an MLIRSparseTensor in the inputs, then use its dtypes
                # TODO find a more principled approach
                try:
                    input_sparse_tensor = next(
                        x
                        for x in list(args) + list(kwargs.values())
                        if isinstance(x, MLIRSparseTensor)
                    )
                except StopIteration:
                    raise TypeError(
                        "Unable to find an MLIRSparseTensor in the inputs.\n"
                        "Cannot return MLIRSparseTensor from raw pointer"
                    )

                raw_results = func(*args, **kwargs)
                if len(self.return_types) == 1:
                    raw_results = (raw_results,)

                dwimmed_results = tuple(
                    MLIRSparseTensor.from_raw_pointer(
                        raw_result,
                        input_sparse_tensor.pointer_dtype,
                        input_sparse_tensor.index_dtype,
                        input_sparse_tensor.value_dtype,
                    )
                    if i in indices_of_returned_sparse_tensors
                    else raw_result
                    for i, raw_result in enumerate(raw_results)
                )

                if len(self.return_types) == 1:
                    (dwimmed_results,) = dwimmed_results

                return dwimmed_results

            return func_wrapper

        return func

    def get_mlir(self, make_private=True) -> str:
        needed_function_definitions = "\n\n".join(
            func_def for func_def, _, _ in self.needed_function_table.values()
        )

        joined_statements = "\n".join(self.function_body_statements)

        return_type = ", ".join(self.return_types)
        if len(self.return_types) != 1:
            return_type = f"({return_type})"

        signature = ", ".join(
            f"{var.access_string()}: {var.var_type}" for var in self.input_vars
        )

        return needed_function_definitions + self.function_wrapper_text.render(
            private_func=make_private,
            func_name=self.func_name,
            signature=signature,
            return_type=return_type,
            statements=joined_statements,
        )

    ################################
    # MLIR Building Method Helpers #
    ################################

    @contextmanager
    def more_indentation(self, num_levels: int = 1) -> Generator[None, None, None]:
        self.indentation_level += num_levels
        yield
        self.indentation_level -= num_levels
        return

    def add_statement(self, statement: str) -> None:
        for line in map(str.strip, statement.split("\n")):
            self.function_body_statements.append(
                " " * self.default_indentation_size
                + " " * self.indentation_delta_size * self.indentation_level
                + line
            )
        return

    def new_var(self, *var_types: str) -> MLIRVar:
        var_name = f"var_{self.var_name_counter}"
        self.var_name_counter += 1
        return MLIRVar(var_name, *var_types)

    #########################
    # MLIR Building Methods #
    #########################

    def constant(self, var_value: Union[str, int, float], var_type: str) -> MLIRVar:
        # TODO consider taking in an MLIRVar as an input to be assigned to.
        var = self.new_var(var_type)
        self.add_statement(f"{var.assign_string()} = constant {var_value} : {var_type}")
        return var

    def return_vars(
        self, *returned_values: Union[Tuple[MLIRVar, int], MLIRVar]
    ) -> None:
        # TODO instead of accepting Tuple[MLIRVar, int], consider having
        # MLIRVar.__getitem__ return a new MLIRVar instance with an "index"
        # attribute (default to None) set the desired index.
        var_index_pairs = []
        for value in returned_values:
            if isinstance(value, MLIRVar):
                var_index_pair = value, None
            elif (
                isinstance(value, tuple)
                and len(value) == 2
                and isinstance(value[0], MLIRVar)
                and isinstance(value[1], int)
            ):
                var_index_pair = value
            else:
                raise TypeError(f"{value} does not denote a valid return value.")
            var_index_pairs.append(var_index_pair)
        if not all(
            mlir_type_strings_equal(
                expected, var.var_type if index is None else var.var_types[index]
            )
            for expected, (var, index) in zip(self.return_types, var_index_pairs)
        ):
            returned_types = tuple(var.var_type for var, _ in var_index_pairs)
            raise TypeError(
                f"Types for {returned_types} do not match function "
                f"return types of {tuple(self.return_types)}."
            )
        returned_values_string = ", ".join(
            var.access_string(index) for var, index in var_index_pairs
        )
        returned_types_string = ", ".join(
            return_types for return_types in self.return_types
        )
        statement = f"return {returned_values_string} : {returned_types_string}"
        self.add_statement(statement)
        return

    class ForLoopVars:
        def __init__(
            self,
            iter_var_index: MLIRVar,
            lower_var_index: MLIRVar,
            upper_var_index: MLIRVar,
            step_var_index: MLIRVar,
            iter_vars: Sequence[MLIRVar],
            returned_variable: Optional[MLIRVar],
            builder: "MLIRFunctionBuilder",
        ) -> None:
            self.iter_var_index = iter_var_index
            self.lower_var_index = lower_var_index
            self.upper_var_index = upper_var_index
            self.step_var_index = step_var_index
            self.iter_vars = iter_vars
            self.returned_variable = returned_variable
            self.builder = builder
            return

        def yield_vars(self, *yielded_vars: MLIRVar) -> None:
            var_strings = []
            var_types = []
            if len(yielded_vars) != len(self.iter_vars):
                raise ValueError(
                    f"Expected {len(self.iter_vars)} yielded values, but got {len(yielded_vars)}."
                )
            for var, iter_var in zip(yielded_vars, self.iter_vars):
                if not mlir_type_strings_equal(var.var_type, iter_var.var_type):
                    raise TypeError(f"{var} and {iter_var} have different types.")
                var_strings.append(var.access_string())
                var_types.append(var.var_type)
            self.builder.add_statement(
                f"scf.yield {', '.join(var_strings)} : {', '.join(var_types)}"
            )
            return

    @contextmanager
    def for_loop(
        self,
        lower: Union[int, MLIRVar],
        upper: Union[int, MLIRVar],
        step: Union[int, MLIRVar] = 1,
        *,
        iter_vars: Optional[Sequence[Tuple[MLIRVar, MLIRVar]]] = None,
    ) -> Generator[ForLoopVars, None, None]:
        iter_var_index = self.new_var("index")
        lower_var_index = (
            lower if isinstance(lower, MLIRVar) else self.constant(lower, "index")
        )
        upper_var_index = (
            upper if isinstance(upper, MLIRVar) else self.constant(upper, "index")
        )
        step_var_index = (
            step if isinstance(step, MLIRVar) else self.constant(step, "index")
        )
        for_loop_open_statment = (
            f"scf.for {iter_var_index.assign_string()} = {lower_var_index.access_string()} "
            f"to {upper_var_index.access_string()} step {step_var_index.access_string()}"
        )
        _iter_vars = []
        if iter_vars is not None:
            iter_var_init_strings = []
            iter_var_types = []
            for iter_var, init_var in iter_vars:
                if not mlir_type_strings_equal(iter_var.var_type, init_var.var_type):
                    raise TypeError(f"{iter_var} and {init_var} have different types.")
                _iter_vars.append(iter_var)
                iter_var_init_strings.append(
                    f"{iter_var.assign_string()}={init_var.access_string()}"
                )
                iter_var_types.append(iter_var.var_type)
            for_loop_open_statment += (
                f" iter_args("
                + ", ".join(iter_var_init_strings)
                + ") -> ("
                + ", ".join(iter_var_types)
                + ")"
            )
        for_loop_open_statment += " {"
        if len(_iter_vars) > 0:
            returned_var = self.new_var(*(var.var_type for var in _iter_vars))
            for_loop_open_statment = (
                f"{returned_var.assign_string()} = {for_loop_open_statment}"
            )
        else:
            returned_var = None
        self.add_statement(for_loop_open_statment)
        with self.more_indentation():
            yield self.ForLoopVars(
                iter_var_index,
                lower_var_index,
                upper_var_index,
                step_var_index,
                _iter_vars,
                returned_var,
                self,
            )
        self.add_statement("}")
        return

    def call(
        self,
        function: BaseFunction,
        *inputs: MLIRVar,
    ) -> MLIRVar:
        # TODO update this method to handle multiple and zero return types.
        if function.func_name in self.needed_function_table:
            (
                function_mlir_text,
                input_types,
                return_type,
            ) = self.needed_function_table[  # TODO handle non-singleton returns here
                function.func_name
            ]
            assert function.get_mlir(make_private=True) == function_mlir_text
        else:
            function_mlir_text = function.get_mlir(make_private=True)
            (function_ast,) = parse_mlir_string(function_mlir_text).body
            input_types = [arg.type.dump() for arg in function_ast.args]
            return_type = (
                function_ast.result_types.dump()
            )  # TODO handle non-singleton returns here
            self.needed_function_table[function.func_name] = (
                function_mlir_text,
                input_types,
                return_type,  # TODO handle non-singleton returns here
            )

        result_var = self.new_var(return_type)  # TODO handle non-singleton returns here
        statement = "".join(
            [
                f"{result_var.assign_string()} = call @{function.func_name}(",  # TODO handle non-singleton returns here
                ", ".join(
                    f"{input_val.access_string()}"
                    if isinstance(input_val, MLIRVar)
                    else str(input_val)  # TODO add test for this else case
                    for input_val in inputs
                ),
                ") : (",
                ", ".join(input_types),
                ") -> ",
                return_type,  # TODO handle non-singleton returns here
            ]
        )

        self.add_statement(statement)
        return result_var
