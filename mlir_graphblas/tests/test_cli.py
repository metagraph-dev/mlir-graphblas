import pytest
from mlir_graphblas import MlirOptCli, MlirOptError
from mlir_graphblas.cli import DebugResult


@pytest.fixture
def cli_input():
    return b"""\
#trait_1d_scalar = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // A
    affine_map<(i) -> (i)>   // X (out)
  ],
  iterator_types = ["parallel"],
  doc = "X(i) = A(i) OP Scalar"
}
func @scale_func(%input: tensor<?xf32>, %scale: f32) -> tensor<?xf32> {
  %0 = linalg.generic #trait_1d_scalar
     ins(%input: tensor<?xf32>)
     outs(%input: tensor<?xf32>) {
      ^bb(%a: f32, %s: f32):
        %0 = mulf %a, %scale  : f32
        linalg.yield %0 : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
"""


def test_apply_passes(cli_input):
    cli = MlirOptCli()
    passes = [
        "--linalg-bufferize",
        "--func-bufferize",
        "--finalizing-bufferize",
        "--convert-linalg-to-affine-loops",
        "--lower-affine",
        "--convert-scf-to-std",
    ]
    result = cli.apply_passes(cli_input, passes)
    assert (
        result
        == """\
module  {
  func @scale_func(%arg0: memref<?xf32>, %arg1: f32) -> memref<?xf32> {
    %c0 = constant 0 : index
    %0 = dim %arg0, %c0 : memref<?xf32>
    %1 = alloc(%0) : memref<?xf32>
    %2 = dim %arg0, %c0 : memref<?xf32>
    %c0_0 = constant 0 : index
    %c1 = constant 1 : index
    br ^bb1(%c0_0 : index)
  ^bb1(%3: index):  // 2 preds: ^bb0, ^bb2
    %4 = cmpi slt, %3, %2 : index
    cond_br %4, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %5 = load %arg0[%3] : memref<?xf32>
    %6 = mulf %5, %arg1 : f32
    store %6, %1[%3] : memref<?xf32>
    %7 = addi %3, %c1 : index
    br ^bb1(%7 : index)
  ^bb3:  // pred: ^bb1
    return %1 : memref<?xf32>
  }
}

"""
    )


def test_apply_passes_fails(cli_input):
    cli = MlirOptCli()
    passes = ["--linalg-bufferize"]
    cli_input = cli_input.replace(b"return %0 : tensor<?xf32>", b"return %0 : memref<?xf32>")
    with pytest.raises(MlirOptError) as excinfo:
        cli.apply_passes(cli_input, passes)
    err = excinfo.value
    assert hasattr(err, "debug_result")
    assert isinstance(err.debug_result, DebugResult)


def test_debug_passes(cli_input):
    cli = MlirOptCli()
    passes = [
        "--linalg-bufferize",
        "--func-bufferize",
        "--finalizing-bufferize",
        "--convert-linalg-to-affine-loops",
        "--lower-affine",
        "--convert-scf-to-std",
    ]
    result = cli.debug_passes(cli_input, passes)
    assert isinstance(result, DebugResult)
    assert result.passes == [p[2:] for p in passes]
    assert len(result.stages) == len(passes) + 1
    assert result.stages[0] == cli_input.decode()
