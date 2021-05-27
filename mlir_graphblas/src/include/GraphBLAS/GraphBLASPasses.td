//===-- GraphBLASPasses.td - Conversion pass definition file --------*- tablegen -*-===//
//
// TODO add documentation
//
//===-------------------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_PASSES
#define MLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// GraphBLASLowerMatrixMultiply
//===----------------------------------------------------------------------===//

def GraphBLASLowerMatrixMultiply : Pass<"graphblas-lower-matrix-multiply", "ModuleOp"> {
  let summary = "TODO add documentation";
  let constructor = "mlir::createGraphBLASLowerMatrixMultiplyPass()";
  let dependentDialects = [
    "linalg::LinalgDialect",
    "AffineDialect",
    "memref::MemRefDialect"
  ];
}

#endif // MLIR_CONVERSION_PASSES
