"""
An IR builder for MLIR.
"""

###########
# Imports #
###########

import jinja2
import mlir
import functools
import itertools
from contextlib import contextmanager
from collections import OrderedDict
from .sparse_utils import MLIRSparseTensor
from .engine import parse_mlir_functions, MlirJitEngine
from .types import Type, AliasMap

from typing import Dict, List, Tuple, Sequence, Generator, Optional, Union

##############
# IR Builder #
##############


class MLIRCompileError(Exception):
    pass


DEFAULT_ENGINE = MlirJitEngine()

GRAPHBLAS_TO_SCF_PASSES = (
    "--graphblas-structuralize",
    "--graphblas-optimize",
    "--graphblas-lower",
    "--sparsification",
    "--sparse-tensor-conversion",
    "--linalg-bufferize",
    "--arith-bufferize",
    "--func-bufferize",
    "--tensor-bufferize",
    "--finalizing-bufferize",
    "--convert-linalg-to-loops",
)

SCF_TO_LLVM_PASSES = (
    "--convert-vector-to-llvm",
    "--convert-math-to-llvm",
    "--convert-math-to-libm",
    "--convert-scf-to-cf",
    "--convert-memref-to-llvm",
    "--convert-openmp-to-llvm",
    "--convert-arith-to-llvm",
    "--convert-std-to-llvm",
    "--reconcile-unrealized-casts",
)

GRAPHBLAS_PASSES = GRAPHBLAS_TO_SCF_PASSES + SCF_TO_LLVM_PASSES
GRAPHBLAS_OPENMP_PASSES = (
    GRAPHBLAS_TO_SCF_PASSES + ("--convert-scf-to-openmp",) + SCF_TO_LLVM_PASSES
)
# TODO: fixme -- this hack avoids broken openmp tests in test_algorithms.py
GRAPHBLAS_OPENMP_PASSES = GRAPHBLAS_PASSES


class MLIRVar:
    """
    Represents an MLIR SSA variable.
    Upon initialization, must be assigned to exactly once, and can then be accessed many times.

    foo = MLIRVar('foo', 'f64')
    add_statement(f"{foo.assign} = arith.constant 1.0 : {foo.type}")
    bar = MLIRVar('bar', 'f64')
    add_statement(f"{bar.assign} = arith.addf {foo}, {baz} : {bar.type}")
    """

    def __init__(self, name: str, type_: Type):
        if not isinstance(type_, Type):
            raise TypeError(
                f"type_ must be a Type instance, not {type(type_)}; use Type.find(str) to convert"
            )
        self.name = name
        self.type = type_
        self._initialized = False

    def __eq__(self, other):
        if not isinstance(other, MLIRVar):
            return NotImplemented
        return self.name == other.name and self.type == other.type

    def __repr__(self):
        return f"MLIRVar(name={self.name}, type={self.type})"

    def __str__(self):
        if not self._initialized:
            raise TypeError(f"Attempt to access {self.name} prior to assign")
        return f"%{self.name}"

    @property
    def assign(self):
        """Must be called exactly once before being accessed for reading"""
        if self._initialized:
            raise TypeError(f"Attempt to assign to {self.name} twice")
        self._initialized = True
        return f"%{self.name}"


class MLIRTuple:
    def __init__(self, name: str, types: Tuple[Type]):
        self.name = name
        self.types = types
        self._initialized = False

    def __eq__(self, other):
        if not isinstance(other, MLIRTuple):
            return NotImplemented
        return self.name == other.name and self.types == other.types

    def __len__(self):
        return len(self.types)

    def __repr__(self):
        return f"MLIRTuple(name={self.name}, types={self.types})"

    def __str__(self):
        raise TypeError(
            f"Cannot access MLIRTuple {self.name} directly. Use index notation to access an element."
        )

    def __getitem__(self, index):
        # Create an initialized SSA which points to the element at `index`
        if not isinstance(index, int):
            raise TypeError(f"Expects int, not {type(index)}")
        type_ = self.types[index]
        element = MLIRVar(f"{self.name}#{index}", type_)
        element._initialized = True
        return element

    @property
    def assign(self):
        """Must be called exactly once before being accessed for reading"""
        if self._initialized:
            raise TypeError(f"Attempt to assign to {self.name} twice")
        self._initialized = True
        return f"%{self.name}:{len(self)}"


class Dialect:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"{self.name} dialect"


class MLIRFunctionBuilder:
    _ops = {}

    default_indentation_size = 4
    indentation_delta_size = 2
    module_wrapper_text = jinja2.Template(
        "{{ aliases }}\n\nmodule {\n\n    {{ body }}\n}\n",
        undefined=jinja2.StrictUndefined,
    )
    function_wrapper_text = jinja2.Template(
        "\n"
        + (" " * default_indentation_size)
        + "func {% if private_func %}private {% endif %}@{{func_name}}({{signature}}) -> {{return_type}} {"
        + "\n"
        + "{{statements}}"
        + "\n"
        + (" " * default_indentation_size)
        + "}",
        undefined=jinja2.StrictUndefined,
    )

    def __init__(
        self,
        func_name: str,
        input_types: Sequence[Union[str, Type]],
        return_types: Sequence[Union[str, Type]],
        aliases: AliasMap = None,
        engine: MlirJitEngine = None,
    ) -> None:
        # TODO mlir functions can return zero or more results https://mlir.llvm.org/docs/LangRef/#operations
        # handle the cases where the number of return types is not 1
        if aliases is None:
            aliases = AliasMap()
        self.aliases = aliases

        if engine is None:
            engine = DEFAULT_ENGINE
        self.engine = engine

        # Build input vars and ensure all types are proper Types
        inputs = []
        for i, it in enumerate(input_types):
            it = Type.find(it, aliases)
            # Create initialized MLIRVar
            iv = MLIRVar(f"arg{i}", it)
            iv._initialized = True
            inputs.append(iv)
        return_types = [Type.find(rt, aliases) for rt in return_types]

        self.func_name = func_name
        self.inputs = inputs
        self.return_types = return_types

        self.var_name_counter = itertools.count()
        self.function_body_statements: List[str] = []
        self.temporary_statement_lists: List[List[str]] = []

        # function_name -> (function_mlir_definition, input_mlir_types, return_mlir_type)
        self.needed_function_table: Dict[
            str, Tuple[str, List[str], str]
        ] = OrderedDict()

        self.indentation_level = 1
        self._initialize_ops()

    @classmethod
    def register_op(cls, opclass):
        subops = cls._ops.setdefault(opclass.dialect, {})
        if opclass.name in subops:
            fullname = (
                opclass.name
                if opclass.dialect is None
                else f"{opclass.dialect}.{opclass.name}"
            )
            raise TypeError(f"{fullname} is already a registered op in {cls.__name__}")
        subops[opclass.name] = opclass

    def _initialize_ops(self):
        for dialect, ops in self._ops.items():
            if dialect is None:
                attach_point = self
            else:
                attach_point = getattr(self, dialect, None)
                if attach_point is None:
                    attach_point = Dialect(dialect)
                    setattr(self, dialect, attach_point)

            for opclass in ops.values():

                def op(opclass, *args, **kwargs) -> Optional[MLIRVar]:
                    ret_val, mlir = opclass.call(self, *args, **kwargs)
                    self.add_statement(mlir)
                    return ret_val

                func = functools.partial(op, opclass)
                setattr(attach_point, opclass.name, func)

    #######################################
    # MLIR Generation/Compilation Methods #
    #######################################

    def get_mlir_module(self, make_private=False):
        """Get the MLIR text for this function wrapped in a MLIR module with
        declarations of external helper functions."""
        aliases = "\n".join(
            f"#{name} = {typ.to_pretty_string()}" for name, typ in self.aliases.items()
        )
        body = self.get_mlir(make_private=make_private)
        return self.module_wrapper_text.render(aliases=aliases, body=body)

    def get_mlir(self, make_private=True, include_func_defs=True) -> str:
        if include_func_defs:
            needed_function_definitions = "\n    ".join(
                func_def for func_def, _, _ in self.needed_function_table.values()
            )
        else:
            needed_function_definitions = ""

        if len(self.temporary_statement_lists) > 0:
            raise RuntimeError(
                "Cannot get MLIR code while using temporary statement storage."
            )
        joined_statements = "\n".join(self.function_body_statements)

        return_type = ", ".join(str(rt) for rt in self.return_types)
        if len(self.return_types) != 1:
            return_type = f"({return_type})"

        signature = ", ".join(f"{var}: {var.type}" for var in self.inputs)

        return needed_function_definitions + self.function_wrapper_text.render(
            private_func=make_private,
            func_name=self.func_name,
            signature=signature,
            return_type=return_type,
            statements=joined_statements,
        )

    def print_mlir(self):
        from .tools import tersify_mlir

        mlir_text = self.get_mlir_module(make_private=False)
        print(tersify_mlir(mlir_text, self.aliases))

    def compile(self, engine=None, passes=None):
        if engine is None:
            engine = self.engine
        if passes is None:
            passes = GRAPHBLAS_PASSES
        passes = tuple(passes)

        # Force recompilation if name is already registered
        if self.func_name in engine.name_to_callable:
            del engine.name_to_callable[self.func_name]

        mlir = self.get_mlir_module()

        engine.add(mlir, passes)
        func = engine[self.func_name]
        func.builder = self
        return func

    ################################
    # MLIR Building Method Helpers #
    ################################

    @contextmanager
    def more_indentation(self, num_levels: int = 1) -> Generator[None, None, None]:
        self.indentation_level += num_levels
        yield
        self.indentation_level -= num_levels

    @property
    def current_statement_list(self) -> List[str]:
        return (
            self.temporary_statement_lists[-1]
            if len(self.temporary_statement_lists) > 0
            else self.function_body_statements
        )

    @contextmanager
    def temporary_statement_list(self) -> Generator[None, None, None]:
        temp_list = []
        self.temporary_statement_lists.append(temp_list)
        assert self.current_statement_list is temp_list
        yield temp_list
        self.temporary_statement_lists.pop()

    def add_statement(self, statement: str) -> None:
        """In an ideal world, no human would ever call this method."""
        if not statement:
            return

        for line in map(str.strip, statement.split("\n")):
            self.current_statement_list.append(
                " " * self.default_indentation_size
                + " " * self.indentation_delta_size * self.indentation_level
                + line
            )

    def new_var(self, var_type: str) -> MLIRVar:
        var_num = next(self.var_name_counter)
        var_name = f"var{var_num}"
        var_type = Type.find(var_type, self.aliases)
        return MLIRVar(var_name, var_type)

    def new_tuple(self, *var_types: str) -> MLIRTuple:
        var_name = f"var{next(self.var_name_counter)}"
        var_types = [Type.find(vt, self.aliases) for vt in var_types]
        return MLIRTuple(var_name, var_types)

    #########################
    # MLIR Building Methods #
    #########################

    def return_vars(self, *returned_values: MLIRVar) -> None:
        if len(returned_values) > 0:
            for expected, var in zip(self.return_types, returned_values):
                if not isinstance(var, MLIRVar):
                    raise TypeError(
                        f"{var!r} is not a valid return value, expected MLIRVar."
                    )
                if var.type != expected:
                    raise TypeError(f"Return type of {var!r} does not match {expected}")
            ret_vals = ", ".join(str(var) for var in returned_values)
            ret_types = ", ".join(str(rt) for rt in self.return_types)
            statement = f"return {ret_vals} : {ret_types}"
        else:
            statement = f"return"
        self.add_statement(statement)

    ################################
    # SCF While Loop Functionality #
    ################################

    class WhileLoopBeforeRegion:
        def __init__(
            self,
            input_initial_values: Sequence[MLIRVar],
            builder: "MLIRFunctionBuilder",
        ):
            self.input_initial_values = input_initial_values
            self.builder = builder
            self.arg_vars = [builder.new_var(var.type) for var in input_initial_values]
            self.forwarded_vars: Optional[Sequence[MLIRVar]] = None
            self.returned_var: Optional[MLIRTuple] = None
            self.preamble_template: Optional[jinja2.Template] = None

        def start_preamble(self):
            if self.preamble_template is not None:
                raise RuntimeError("scf.while preamble template already generated.")
            preamble_template_string = ""
            if len(self.arg_vars) != 0:
                self.preamble_template = jinja2.Template(
                    "scf.while ("
                    + ", ".join(
                        f"{arg_var.assign} = {init_val}"
                        for arg_var, init_val in zip(
                            self.arg_vars, self.input_initial_values
                        )
                    )
                    + ") : ("
                    + ", ".join(str(var.type) for var in self.arg_vars)
                    + ") -> ({{ after_region_arg_types }}) {",
                    undefined=jinja2.StrictUndefined,
                )
            else:
                self.preamble_template = jinja2.Template(
                    "scf.while : () -> ({{ after_region_arg_types }}) {",
                    undefined=jinja2.StrictUndefined,
                )

        def finish_preamble(self):
            if not self.condition_has_been_built():
                raise RuntimeError(
                    'Cannot write preamble of the "before" region of an scf.while '
                    "loop before an scf.condition statement is built."
                )
            if self.preamble_template is None:
                raise RuntimeError("scf.while preamble template not yet generated.")
            assert self.returned_var is None
            if len(self.forwarded_vars) != 0:
                self.returned_var = self.builder.new_tuple(
                    *(var.type for var in self.forwarded_vars)
                )
                after_region_arg_types = ", ".join(
                    str(t) for t in self.returned_var.types
                )
            else:
                after_region_arg_types = ""
            preamble = self.preamble_template.render(
                after_region_arg_types=after_region_arg_types,
            )
            if self.returned_var is not None:
                preamble = f"{self.returned_var.assign} = {preamble}"
            self.builder.add_statement(preamble)
            return

        def write_postamble(self):
            if not self.condition_has_been_built():
                raise RuntimeError(
                    'Cannot write postamble of the "before" region of an scf.while '
                    "loop before an scf.condition statement is built."
                )
            self.builder.add_statement("} do {")

        def condition_has_been_built(self) -> bool:
            return self.forwarded_vars is not None

        def condition(self, condition_var: MLIRVar, *after_region_args: MLIRVar):
            condition_string = f"scf.condition({condition_var}) "
            if len(after_region_args) != 0:
                condition_string += (
                    ", ".join(map(str, after_region_args))
                    + " : "
                    + ", ".join(str(arg.type) for arg in after_region_args)
                )
            self.builder.add_statement(condition_string)
            self.forwarded_vars = after_region_args
            assert self.condition_has_been_built()
            return

    class WhileLoopAfterRegion:
        def __init__(
            self, before_region: "WhileLoopBeforeRegion", builder: "MLIRFunctionBuilder"
        ):
            self.before_region = before_region
            self.builder = builder
            self.arg_vars = [
                builder.new_var(var.type) for var in before_region.forwarded_vars
            ]

        def write_preamble(self):
            preamble_string = "^bb0"
            if len(self.arg_vars) != 0:
                preamble_string += (
                    "("
                    + ", ".join(f"{var.assign}: {var.type}" for var in self.arg_vars)
                    + ")"
                )
            preamble_string += ":"
            self.builder.add_statement(preamble_string)

        def write_postamble(self):
            self.builder.add_statement("}")

        def yield_vars(self, *yielded_vars: MLIRVar):
            if len(yielded_vars) != len(self.before_region.arg_vars):
                raise ValueError(
                    f"Expected {len(self.before_region.arg_vars)} yielded values, "
                    f"but got {len(yielded_vars)}."
                )
            for yielded_var, before_region_arg_var in zip(
                yielded_vars, self.before_region.arg_vars
            ):
                if yielded_var.type != before_region_arg_var.type:
                    raise TypeError(
                        f"{yielded_vars!r} and {before_region_arg_var!r} have different types."
                    )
            yield_string = "scf.yield "
            if len(yielded_vars) != 0:
                yield_vals = ", ".join(str(var) for var in yielded_vars)
                yield_types = ", ".join(str(var.type) for var in yielded_vars)
                yield_string += f"{yield_vals} : {yield_types}"
            self.builder.add_statement(yield_string)

    class WhileLoopRegions:
        def __init__(
            self,
            before_region_inputs: Sequence[MLIRVar],
            builder: "MLIRFunctionBuilder",
        ):
            self.before_region_inputs = before_region_inputs
            self.before_region: Optional["WhileLoopBeforeRegion"] = None
            self.after_region: Optional["WhileLoopAfterRegion"] = None
            self.builder = builder

        @property
        def returned_variable(self) -> MLIRTuple:
            if self.before_region is None:
                raise RuntimeError(
                    'Cannot access result of scf.while loop until the "before" region has been built.'
                )
            return self.before_region.returned_var

        @property
        @contextmanager
        def before(self) -> Generator["WhileLoopBeforeRegion", None, None]:
            if self.before_region is not None:
                raise RuntimeError(
                    'Cannot create a new "before" region as one already exists.'
                )
            before_region = self.builder.WhileLoopBeforeRegion(
                self.before_region_inputs, self.builder
            )
            before_region.start_preamble()
            with self.builder.temporary_statement_list() as temp_statements:
                with self.builder.more_indentation():
                    yield before_region
            before_region.finish_preamble()
            with self.builder.more_indentation():
                for s in temp_statements:
                    self.builder.add_statement(s)
            before_region.write_postamble()
            self.before_region = before_region

        @property
        @contextmanager
        def after(self) -> Generator["WhileLoopAfterRegion", None, None]:
            if self.after_region is not None:
                raise RuntimeError(
                    'Cannot create a new "after" region as one already exists.'
                )
            if self.before_region is None:
                raise RuntimeError(
                    'Cannot build "after" region of a while loop before building the "before" region.'
                )
            after_region = self.builder.WhileLoopAfterRegion(
                self.before_region, self.builder
            )
            with self.builder.temporary_statement_list() as temp_statements:
                with self.builder.more_indentation():
                    after_region.write_preamble()
                    yield after_region
            with self.builder.more_indentation():
                with self.builder.more_indentation():
                    for s in temp_statements:
                        self.builder.add_statement(s)
            after_region.write_postamble()
            self.after_region = after_region

    @contextmanager
    def while_loop(
        self,
        *before_region_inputs: MLIRVar,
    ) -> Generator[WhileLoopRegions, None, None]:
        regions = self.WhileLoopRegions(before_region_inputs, self)
        yield regions
        if regions.before_region is None:
            raise RuntimeError(
                'Cannot build scf.while loop without completing "before" region.'
            )
        if regions.after_region is None:
            raise RuntimeError(
                'Cannot build scf.while loop without completing "after" region.'
            )
        return

    ##############################
    # SCF For Loop Functionality #
    ##############################

    class ForLoopVars:
        def __init__(
            self,
            iter_var_index: MLIRVar,
            lower_var_index: MLIRVar,
            upper_var_index: MLIRVar,
            step_var_index: MLIRVar,
            iter_vars: Sequence[MLIRVar],
            returned_variable: Optional[MLIRTuple],
            builder: "MLIRFunctionBuilder",
        ):
            self.iter_var_index = iter_var_index
            self.lower_var_index = lower_var_index
            self.upper_var_index = upper_var_index
            self.step_var_index = step_var_index
            self.iter_vars = iter_vars
            self.returned_variable = returned_variable
            self.builder = builder

        def yield_vars(self, *yielded_vars: MLIRVar):
            if len(yielded_vars) != len(self.iter_vars):
                raise ValueError(
                    f"Expected {len(self.iter_vars)} yielded values, but got {len(yielded_vars)}."
                )
            for var, iter_var in zip(yielded_vars, self.iter_vars):
                if iter_var.type != var.type:
                    raise TypeError(f"{var!r} and {iter_var!r} have different types.")
            yield_vals = ", ".join(str(var) for var in yielded_vars)
            yield_types = ", ".join(str(var.type) for var in yielded_vars)
            self.builder.add_statement(f"scf.yield {yield_vals} : {yield_types}")

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
            lower if isinstance(lower, MLIRVar) else self.arith.constant(lower, "index")
        )
        upper_var_index = (
            upper if isinstance(upper, MLIRVar) else self.arith.constant(upper, "index")
        )
        step_var_index = (
            step if isinstance(step, MLIRVar) else self.arith.constant(step, "index")
        )
        for_loop_open_statment = (
            f"scf.for {iter_var_index.assign} = {lower_var_index} "
            f"to {upper_var_index} step {step_var_index}"
        )
        _iter_vars = []
        if iter_vars is not None:
            iter_var_init_strings = []
            iter_var_types = []
            for iter_var, init_var in iter_vars:
                if init_var.type != iter_var.type:
                    raise TypeError(
                        f"{iter_var!r} and {init_var!r} have different types."
                    )
                _iter_vars.append(iter_var)
                iter_var_init_strings.append(f"{iter_var.assign}={init_var}")
                iter_var_types.append(iter_var.type)
            for_loop_open_statment += (
                f" iter_args("
                + ", ".join(iter_var_init_strings)
                + ") -> ("
                + ", ".join(str(ivt) for ivt in iter_var_types)
                + ")"
            )
        for_loop_open_statment += " {"
        if len(_iter_vars) > 0:
            returned_var = self.new_tuple(*(var.type for var in _iter_vars))
            for_loop_open_statment = f"{returned_var.assign} = {for_loop_open_statment}"
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

    def call(
        self,
        function: "MlirFunctionBuilder",
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
            if isinstance(function, MLIRFunctionBuilder):
                function_mlir_text = function.get_mlir(
                    make_private=True, include_func_defs=False
                )
            else:
                function_mlir_text = function.get_mlir(make_private=True)
            full_function_mlir_text = function.get_mlir_module(make_private=True)
            mlir_ast = parse_mlir_functions(full_function_mlir_text, self.engine._cli)
            mlir_functions: List[mlir.astnodes.Function] = [
                obj
                for obj in self.engine._walk_module(mlir_ast)
                if isinstance(obj, mlir.astnodes.Function)
                and obj.name.value == function.func_name
            ]
            function_ast = mlir_functions[0]
            input_types = [arg.type.dump() for arg in function_ast.args]
            return_type = (
                function_ast.result_types.dump()
            )  # TODO handle non-singleton returns here
            self.needed_function_table[function.func_name] = (
                function_mlir_text,
                input_types,
                return_type,  # TODO handle non-singleton returns here
            )
            # Add function header definitions to self.needed_function_table
            # Doing this avoid duplication conflicts
            if isinstance(function, MLIRFunctionBuilder):
                for key, vals in function.needed_function_table.items():
                    self.needed_function_table[key] = vals

        result_var = self.new_var(return_type)  # TODO handle non-singleton returns here
        statement = "".join(
            [
                f"{result_var.assign} = call @{function.func_name}(",  # TODO handle non-singleton returns here
                ", ".join(str(input_val) for input_val in inputs),
                ") : (",
                ", ".join(input_types),
                ") -> ",
                return_type,  # TODO handle non-singleton returns here
            ]
        )

        self.add_statement(statement)
        return result_var


# Force ops to register with the builder
from . import ops

del ops
