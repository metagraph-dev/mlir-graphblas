//===- GraphBLASPasses.cpp - GraphBLAS dialect passes ---------*- C++ -*-===//
//
// TODO add documentation
//
//===--------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "GraphBLAS/GraphBLASPasses.h"
#include "mlir/Pass/Pass.h"

using namespace ::mlir;

namespace {

//===----------------------------------------------------------------------===//
// Passes declaration.
//===----------------------------------------------------------------------===//

#define GEN_PASS_CLASSES
#include "GraphBLAS/GraphBLASPasses.h.inc"

//===----------------------------------------------------------------------===//
// Passes implementation.
//===----------------------------------------------------------------------===//

struct GraphBLASLowerMatrixMultiplyPass : public GraphBLASLowerMatrixMultiplyBase<GraphBLASLowerMatrixMultiplyPass> {
  void runOnOperation() override {
    // TODO implement something
    signalPassFailure();
  }
};

} // end anonymous namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createGraphBLASLowerMatrixMultiplyPass() {
  return std::make_unique<GraphBLASLowerMatrixMultiplyPass>();
}
