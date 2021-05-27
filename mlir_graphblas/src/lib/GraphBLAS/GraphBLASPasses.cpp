//===- GraphBLASPasses.cpp - GraphBLAS dialect passes ---------*- C++ -*-===//
//
// TODO add documentation
//
//===--------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "GraphBLAS/GraphBLASPasses.h"

#include <iostream> // TODO remove this

using namespace ::mlir;

namespace {

//===----------------------------------------------------------------------===//
// Passes Implementation Helpers.
//===----------------------------------------------------------------------===//

mlir::RankedTensorType getCSRTensorType(mlir::MLIRContext *context, mlir::Type valueType) {
    SmallVector<mlir::sparse_tensor::SparseTensorEncodingAttr::DimLevelType, 2> dlt;
    dlt.push_back(mlir::sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Dense);
    dlt.push_back(mlir::sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Compressed);
    unsigned ptr = 64;
    unsigned ind = 64;
    AffineMap map = AffineMap::getMultiDimIdentityMap(2, context);

    RankedTensorType csrTensor = RankedTensorType::get(
        {-1, -1}, /* 2D, unknown size */
        valueType,
        mlir::sparse_tensor::SparseTensorEncodingAttr::get(context, dlt, map, ptr, ind));

    return csrTensor;
}

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
    
    MLIRContext *context = op->getContext();
    auto module = op->getParentOfType<ModuleOp>();
    
    // TODO get the types from the inputs during the "match" part of this func
    auto valueType = rewriter.getI64Type();
    RankedTensorType csrTensorType = getCSRTensorType(context, valueType);
    auto func_type = FunctionType::get(context, {csrTensorType, csrTensorType}, csrTensorType);

    llvm::StringRef semi_ring = op.semiring();
    std::string func_name = "matrix_multiply_" + semi_ring.str();
    auto func = module.lookupSymbol<FuncOp>(func_name);
    if (!func) {
      OpBuilder moduleBuilder(module.getBodyRegion());
      moduleBuilder.create<FuncOp>(op->getLoc(), func_name, func_type).setPrivate();
    }
    auto funcSymbol = SymbolRefAttr::get(context, func_name);
    
    mlir::Value a = op.a();
    mlir::Value b = op.b();
    auto loc = rewriter.getUnknownLoc();
    
    auto callOp = rewriter.create<mlir::CallOp>(loc,
						funcSymbol,
						csrTensorType,
						llvm::ArrayRef<mlir::Value>({a, b})
						);
    
    rewriter.replaceOp(op, callOp->getResults());
    
    return success();
  };
};

class LowerMatrixApplyRewrite : public OpRewritePattern<mlir::graphblas::MatrixApplyOp> {
public:
  using OpRewritePattern<mlir::graphblas::MatrixApplyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::graphblas::MatrixApplyOp op, PatternRewriter &rewriter) const {
    
    // MLIRContext *context = op->getContext();
    // auto module = op->getParentOfType<ModuleOp>();
    
    // // TODO get the types from the inputs during the "match" part of this func
    // auto valueType = rewriter.getI64Type();
    // RankedTensorType csrTensorType = getCSRTensorType(context, valueType);
    // auto func_type = FunctionType::get(context, {csrTensorType, csrTensorType}, csrTensorType);

    // llvm::StringRef semi_ring = op.semiring();
    // std::string func_name = "matrix_apply_" + semi_ring.str();
    // auto func = module.lookupSymbol<FuncOp>(func_name);
    // if (!func) {
    //   OpBuilder moduleBuilder(module.getBodyRegion());
    //   moduleBuilder.create<FuncOp>(op->getLoc(), func_name, func_type).setPrivate();
    // }
    // auto funcSymbol = SymbolRefAttr::get(context, func_name);
    
    // mlir::Value a = op.a();
    // mlir::Value b = op.b();
    // auto loc = rewriter.getUnknownLoc();
    
    // auto callOp = rewriter.create<mlir::CallOp>(loc,
    // 						funcSymbol,
    // 						csrTensorType,
    // 						llvm::ArrayRef<mlir::Value>({a, b})
    // 						);
    
    // rewriter.replaceOp(op, callOp->getResults());
    
    // return success();
    return failure();
  };
};

void populateGraphBLASLoweringPatterns(RewritePatternSet &patterns) {
  patterns.add<
    LowerMatrixApplyRewrite,
    LowerMatrixMultiplyRewrite
    >(patterns.getContext());
}

struct GraphBLASLoweringPass : public GraphBLASLoweringBase<GraphBLASLoweringPass> {
  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);
    populateGraphBLASLoweringPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // end anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createGraphBLASLoweringPass() {
  return std::make_unique<GraphBLASLoweringPass>();
}
