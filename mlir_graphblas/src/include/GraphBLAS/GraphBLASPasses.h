
//===- GraphBLASPasses.h - GraphBLAS dialect passes -----------------*- C++ -*-===//
//
// TODO add documentation
//
//===--------------------------------------------------------------------------===//

#ifndef GRAPHBLAS_GRAPHBLASPASSES_H
#define GRAPHBLAS_GRAPHBLASPASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
  
std::unique_ptr<OperationPass<ModuleOp>> createGraphBLASLoweringPass();
std::unique_ptr<OperationPass<ModuleOp>> createGraphBLASOptimizePass();
}

//===----------------------------------------------------------------------===//
// Ops declaration.
//===----------------------------------------------------------------------===//
#include "GraphBLAS/GraphBLASOpsEnums.h.inc"

#define GET_OP_CLASSES
#include "GraphBLAS/GraphBLASOps.h.inc"

//===----------------------------------------------------------------------===//
// Registration.
//===----------------------------------------------------------------------===//

// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "GraphBLAS/GraphBLASPasses.h.inc"

#endif // GRAPHBLAS_GRAPHBLASPASSES_H
