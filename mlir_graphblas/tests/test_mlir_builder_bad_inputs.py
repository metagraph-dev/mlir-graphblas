import pytest
from mlir_graphblas import MlirJitEngine
from mlir_graphblas.mlir_builder import MLIRVar, MLIRFunctionBuilder


@pytest.fixture(scope="module")
def engine():
    return MlirJitEngine()


def test_ir_builder_bad_input_multi_value_mlir_variable():
    ir_builder = MLIRFunctionBuilder("some_func", input_types=[], return_types=("i8",))

    iter_i8_var = ir_builder.new_var("i8")
    lower_i8_var = ir_builder.constant(1, "i8")
    iter_i64_var = ir_builder.new_var("i64")
    lower_i64_var = ir_builder.constant(1, "i64")
    with ir_builder.for_loop(
        0, 1, 1, iter_vars=[(iter_i8_var, lower_i8_var), (iter_i64_var, lower_i64_var)]
    ) as for_vars:
        constant_i8_var = ir_builder.constant(8, "i8")
        constant_i64_var = ir_builder.constant(64, "i64")

        # Raise when yielding too few values
        with pytest.raises(ValueError, match="Expected 2 yielded values, but got 1."):
            for_vars.yield_vars(constant_i8_var)

        # Raise when yielding too many values
        with pytest.raises(ValueError, match="Expected 2 yielded values, but got 3."):
            for_vars.yield_vars(constant_i8_var, constant_i64_var, lower_i64_var)

        # Raise when yielding incorrect types
        with pytest.raises(TypeError, match=" have different types."):
            for_vars.yield_vars(constant_i64_var, constant_i8_var)

        for_vars.yield_vars(constant_i8_var, constant_i64_var)

    # Raise when returning multiple valued variable
    with pytest.raises(TypeError, match=" is not a valid return value"):
        ir_builder.return_vars(for_vars.returned_variable)

    # Raise when using multiple valued variable as operand
    assigned_to_i8_var = ir_builder.new_var("i8")
    c1_i8_var = ir_builder.constant(1, "i8")
    with pytest.raises(
        TypeError,
        match="Cannot access MLIRTuple .+ directly. Use index notation to access an element.",
    ):
        ir_builder.add_statement(
            f"{assigned_to_i8_var.assign} = addi {c1_i8_var}, {for_vars.returned_variable} : i8"
        )

    # Raise when using multiple valued variable indexed via out-of-bound int index as operand
    with pytest.raises(IndexError):
        for_vars.returned_variable[999]

    # Raise when indexing into multiple valued variable via slice
    with pytest.raises(TypeError, match="Expects int, not"):
        ir_builder.return_vars(for_vars.returned_variable[:])

    # Raise when returning a non-MLIRVar
    with pytest.raises(
        TypeError, match="10 is not a valid return value, expected MLIRVar."
    ):
        ir_builder.return_vars(10)

    # Raise when returning value incompatible with return type.
    c1_i64_var = ir_builder.constant(1, "i64")
    with pytest.raises(
        TypeError, match="Return type of MLIRVar\(name=.+, type=i64\) does not match i8"
    ):
        ir_builder.return_vars(c1_i64_var)

    # Raise when iterator variables have incompatible types
    with pytest.raises(TypeError, match=" have different types."):
        with ir_builder.for_loop(
            0,
            1,
            1,
            iter_vars=[(iter_i8_var, lower_i64_var), (iter_i64_var, lower_i8_var)],
        ) as bad_for_vars:
            pass

    ir_builder.return_vars(for_vars.returned_variable[0])
