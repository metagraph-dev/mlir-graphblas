//===- GraphBLASPasses.cpp - GraphBLAS dialect passes ---------*- C++ -*-===//
//
// TODO add documentation
//
//===--------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Pass/Pass.h"

#include "GraphBLAS/GraphBLASPasses.h"

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

class LowerMatrixMultiplyRewrite : public OpRewritePattern<mlir::graphblas::MatrixMultiplyOp> {
public:
  using OpRewritePattern<mlir::graphblas::MatrixMultiplyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::graphblas::MatrixMultiplyOp op, PatternRewriter &rewriter) const {
    // TODO fill this in
    return failure();
    // return success();
  };
};

void populateGraphBLASLowerMatrixMultiplyPatterns(RewritePatternSet &patterns) {
  patterns.add<LowerMatrixMultiplyRewrite>(patterns.getContext());
}

struct GraphBLASLowerMatrixMultiplyPass : public GraphBLASLowerMatrixMultiplyBase<GraphBLASLowerMatrixMultiplyPass> {
  void runOnOperation() override {
    auto *ctx = &getContext();
    auto module = getOperation();
    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);
    // TODO update the legal and illegal ops
    // target.addLegalDialect<
    //   AffineDialect,
    //   memref::MemRefDialect,
    //   scf::SCFDialect,
    //   StandardOpsDialect
    //   >();
    // target.addLegalOp<ModuleOp, FuncOp, ReturnOp>();
    // target.addIllegalOp<NewOp, ToPointersOp, ToIndicesOp, ToValuesOp>();
    populateGraphBLASLowerMatrixMultiplyPatterns(patterns);
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // end anonymous namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createGraphBLASLowerMatrixMultiplyPass() {
  return std::make_unique<GraphBLASLowerMatrixMultiplyPass>();
}
