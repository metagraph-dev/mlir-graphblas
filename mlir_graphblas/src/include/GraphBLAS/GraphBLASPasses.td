//===-- GraphBLASPasses.td - Conversion pass definition file --------*- tablegen -*-===//
//
// TODO add documentation
//
//===-------------------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_PASSES
#define MLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// GraphBLASLowering
//===----------------------------------------------------------------------===//

def GraphBLASLowering : Pass<"graphblas-lower", "ModuleOp"> {
  let summary = "TODO add documentation";
  let constructor = "mlir::createGraphBLASLoweringPass()";
  let dependentDialects = [
    "linalg::LinalgDialect",
    "AffineDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
}

#endif // MLIR_CONVERSION_PASSES
