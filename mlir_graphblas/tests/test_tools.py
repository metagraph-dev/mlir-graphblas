import pytest
import subprocess
import mlir_graphblas

TEST_CASES = (
    pytest.param(
        """
builtin.module  {
  builtin.func @test_func(%arg0: tensor<2x3xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<2x3xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
    return %arg0 : tensor<2x3xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
  }
}
""",
        """
#CSC64 = #sparse_tensor.encoding<{
    dimLevelType = [ "dense", "compressed" ],
    dimOrdering = affine_map<(d0, d1) -> (d1, d0)>,
    pointerBitWidth = 64,
    indexBitWidth = 64
}>

builtin.module  {
  builtin.func @test_func(%arg0: tensor<2x3xf64, #CSC64>) -> tensor<2x3xf64, #CSC64> {
    return %arg0 : tensor<2x3xf64, #CSC64>
  }
}
""",
        id="csc",
    ),
    pytest.param(
        """
builtin.module  {
  builtin.func @test_func(%arg0: tensor<2x3xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<2x3xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
    return %arg0 : tensor<2x3xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
  }
}
""",
        """
#CSR64 = #sparse_tensor.encoding<{
    dimLevelType = [ "dense", "compressed" ],
    dimOrdering = affine_map<(d0, d1) -> (d0, d1)>,
    pointerBitWidth = 64,
    indexBitWidth = 64
}>

builtin.module  {
  builtin.func @test_func(%arg0: tensor<2x3xf64, #CSR64>) -> tensor<2x3xf64, #CSR64> {
    return %arg0 : tensor<2x3xf64, #CSR64>
  }
}
""",
        id="csr",
    ),
    pytest.param(
        """
builtin.module  {
  builtin.func @convert_layout_wrapper(%arg0: tensor<2x3xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<2x3xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
    %0 = graphblas.convert_layout %arg0 : tensor<2x3xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to tensor<2x3xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
    return %0 : tensor<2x3xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
  }
}
""",
        """
#CSR64 = #sparse_tensor.encoding<{
    dimLevelType = [ "dense", "compressed" ],
    dimOrdering = affine_map<(d0, d1) -> (d0, d1)>,
    pointerBitWidth = 64,
    indexBitWidth = 64
}>

#CSC64 = #sparse_tensor.encoding<{
    dimLevelType = [ "dense", "compressed" ],
    dimOrdering = affine_map<(d0, d1) -> (d1, d0)>,
    pointerBitWidth = 64,
    indexBitWidth = 64
}>

builtin.module  {
  builtin.func @convert_layout_wrapper(%arg0: tensor<2x3xf64, #CSR64>) -> tensor<2x3xf64, #CSC64> {
    %0 = graphblas.convert_layout %arg0 : tensor<2x3xf64, #CSR64> to tensor<2x3xf64, #CSC64>
    return %0 : tensor<2x3xf64, #CSC64>
  }
}
""",
        id="csr_and_csc",
    ),
    pytest.param(
        """
builtin.module  {
  builtin.func @vector_argmin(%arg0: tensor<3xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> i64 {
    %0 = graphblas.reduce_to_scalar %arg0 {aggregator = "argmin"} : tensor<3xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to i64
    return %0 : i64
  }
}
""",
        """
#CV64 = #sparse_tensor.encoding<{
    dimLevelType = [ "compressed" ],
    pointerBitWidth = 64,
    indexBitWidth = 64
}>

builtin.module  {
  builtin.func @vector_argmin(%arg0: tensor<3xi64, #CV64>) -> i64 {
    %0 = graphblas.reduce_to_scalar %arg0 {aggregator = "argmin"} : tensor<3xi64, #CV64> to i64
    return %0 : i64
  }
}
""",
        id="CV64",
    ),
)


@pytest.mark.parametrize("input_mlir, output_mlir", TEST_CASES)
def test_tersify_mlir(input_mlir, output_mlir):
    """Check that the tersify_mlir CLI tool works."""
    process = subprocess.run(
        ["tersify_mlir"],
        capture_output=True,
        input=input_mlir.encode(),
    )
    assert process.returncode == 0
    stdout = process.stdout.decode().strip()
    assert stdout == output_mlir.strip()
    assert stdout == mlir_graphblas.tools.tersify_mlir(input_mlir).strip()
    return


def test_tersify_mlir_with_invalid_mlir():
    input_mlir = "asdf"
    process = subprocess.run(
        ["tersify_mlir"],
        capture_output=True,
        input=input_mlir.encode(),
    )
    assert process.returncode != 0
    assert len(process.stdout) == 0
    assert input_mlir in process.stderr.decode()
    return
