//===- GraphBLASPasses.cpp - GraphBLAS dialect passes ---------*- C++ -*-===//
//
// TODO add documentation
//
//===--------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h" // TODO do we need this?

#include "GraphBLAS/GraphBLASPasses.h"

#include <iostream> // TODO remove this

using namespace ::mlir;

namespace {

//===----------------------------------------------------------------------===//
// Passes declaration.
//===----------------------------------------------------------------------===//

// TODO should this go in GraphBLASPasses.h ?
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
    // ~/code/llvm-project/mlir/lib/Dialect/SparseTensor/Transforms/Sparsification.cpp
    std::cout << "1 --------------------------------" << std::endl;
    op.dump();
    std::cout << "2 --------------------------------" << std::endl;
    llvm::StringRef semiring = op.semiring();
    std::cout << "semiring: " << semiring.str() << std::endl;
    std::cout << "3 --------------------------------" << std::endl;
    mlir::Value a = op.a();
    a.dump();
    std::cout << "4 --------------------------------" << std::endl;
    mlir::Value b = op.b();
    b.dump();
    std::cout << "5 --------------------------------" << std::endl;
    mlir::Value output = op.output();
    output.dump();
    std::cout << "6 --------------------------------" << std::endl;
    // mlir::ReturnOp returnOp = rewriter.create<ReturnOp>(rewriter.getUnknownLoc());
    // returnOp.dump();
    std::cout << "7 --------------------------------" << std::endl;
    auto tensor = a;
    auto tensorType = tensor.getType();
    
    // auto func = getFunc(mod, loc, "empty_like", tensorType, tensorType);
    MLIRContext *context = rewriter.getContext();
    auto loc = rewriter.getUnknownLoc();
    rewriter.create<FuncOp>(loc, "dummy_func", FunctionType::get(context, tensorType, tensorType))
            .setPrivate();
    auto func = SymbolRefAttr::get(context, "dummy_func");
  
    auto callOp = rewriter.create<mlir::CallOp>(loc, func, tensorType, tensor);
    callOp.dump();
    rewriter.replaceOp(op, callOp->getResults());
    std::cout << "8 --------------------------------" << std::endl;
    rewriter.getBlock()->dump();
    std::cout << "9 --------------------------------" << std::endl;
    // op.dump();
    std::cout << "10 --------------------------------" << std::endl;
    std::cout << "\n\n\n\n\n\n\n\n\n\n\n" << std::endl;
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
    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);
    populateGraphBLASLowerMatrixMultiplyPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // end anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createGraphBLASLowerMatrixMultiplyPass() {
  return std::make_unique<GraphBLASLowerMatrixMultiplyPass>();
}
