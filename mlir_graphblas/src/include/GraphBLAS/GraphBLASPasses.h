//===- GraphBLASPasses.h - GraphBLAS dialect passes -----------------*- C++ -*-===//
//
// TODO add documentation
//
//===--------------------------------------------------------------------------===//

#ifndef GRAPHBLAS_GRAPHBLASPASSES_H
#define GRAPHBLAS_GRAPHBLASPASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

namespace graphblas {

void populateGraphBLASLowerMatrixMultiplyPatterns(RewritePatternSet &patterns);

}
  
std::unique_ptr<OperationPass<FuncOp>> createGraphBLASLowerMatrixMultiplyPass();

}

//===----------------------------------------------------------------------===//
// Registration.
//===----------------------------------------------------------------------===//

// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "GraphBLAS/GraphBLASPasses.h.inc"

#endif // GRAPHBLAS_GRAPHBLASPASSES_H
