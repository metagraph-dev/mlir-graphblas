
//===- GraphBLASPasses.h - GraphBLAS dialect passes -----------------*- C++
//-*-===//
//
// TODO add documentation
//
//===--------------------------------------------------------------------------===//

#ifndef GRAPHBLAS_GRAPHBLASPASSES_H
#define GRAPHBLAS_GRAPHBLASPASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
std::unique_ptr<OperationPass<ModuleOp>> createGraphBLASLoweringPass();
std::unique_ptr<OperationPass<ModuleOp>> createGraphBLASLinalgLoweringPass();
std::unique_ptr<OperationPass<ModuleOp>> createGraphBLASOptimizePass();
std::unique_ptr<OperationPass<ModuleOp>> createGraphBLASStructuralizePass();
std::unique_ptr<OperationPass<ModuleOp>> createGraphBLASDWIMPass();
} // namespace mlir

//===----------------------------------------------------------------------===//
// Registration.
//===----------------------------------------------------------------------===//

// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "GraphBLAS/GraphBLASPasses.h.inc"

#endif // GRAPHBLAS_GRAPHBLASPASSES_H
