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
    "LLVM::LLVMDialect",
    "linalg::LinalgDialect",
    "AffineDialect",
    "memref::MemRefDialect",
    "tensor::TensorDialect",
    "math::MathDialect",
    "arith::ArithmeticDialect",
    "scf::SCFDialect"
  ];
}

//===----------------------------------------------------------------------===//
// GraphBLASOptimize
//===----------------------------------------------------------------------===//

def GraphBLASOptimize : Pass<"graphblas-optimize", "ModuleOp"> {
  let summary = "TODO add documentation";
  let constructor = "mlir::createGraphBLASOptimizePass()";
  let dependentDialects = [
  ];
}

//===----------------------------------------------------------------------===//
// GraphBLASStructuralize
//===----------------------------------------------------------------------===//

def GraphBLASStructuralize : Pass<"graphblas-structuralize", "ModuleOp"> {
  let summary = "TODO add documentation";
  let constructor = "mlir::createGraphBLASStructuralizePass()";
  let dependentDialects = [
    "memref::MemRefDialect",
    "math::MathDialect",
    "arith::ArithmeticDialect",
    "scf::SCFDialect"
  ];
}

#endif // MLIR_CONVERSION_PASSES
