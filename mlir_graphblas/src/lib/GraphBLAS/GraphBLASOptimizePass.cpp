//===- GraphBLASPasses.cpp - GraphBLAS dialect passes ---------*- C++ -*-===//
//
// TODO add documentation
//
//===--------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include "GraphBLAS/GraphBLASPasses.h"
#include "GraphBLAS/GraphBLASUtils.h"

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

class FuseMatrixMultiplyReduceRewrite
    : public OpRewritePattern<graphblas::ReduceToScalarGenericOp> {
public:
  using OpRewritePattern<graphblas::ReduceToScalarGenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::ReduceToScalarGenericOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.input();
    graphblas::MatrixMultiplyGenericOp predecessor =
        input.getDefiningOp<graphblas::MatrixMultiplyGenericOp>();
    if (predecessor != nullptr && predecessor->hasOneUse()) {
      Location loc = op->getLoc();

      if (getRank(predecessor.a()) < 2 || getRank(predecessor.b()) < 2)
        return failure();

      // Build new MatrixMultiplyReduceToScalarGeneric op with the operands and
      // regions of the multiply, then add in the aggregator from the reduce
      ValueRange operands = predecessor.getOperands();
      NamedAttrList attributes = predecessor->getAttrs();
      RegionRange multiplyExtensions = predecessor.extensions();
      unsigned newRegions = multiplyExtensions.size();

      ExtensionBlocks multiplyBlocks;
      std::set<graphblas::YieldKind> required = {
          graphblas::YieldKind::ADD_IDENTITY, graphblas::YieldKind::ADD,
          graphblas::YieldKind::MULT};
      std::set<graphblas::YieldKind> optional = {
          graphblas::YieldKind::TRANSFORM_OUT};
      LogicalResult result = multiplyBlocks.extractBlocks(
          op, multiplyExtensions, required, optional);
      if (result.failed()) {
        return result;
      }

      if (multiplyBlocks.transformOut) {
        return failure(); // FIXME: cannot fuse with existing transform for now
      } else {
        newRegions += 2; // adding new agg and agg identity block
      }

      graphblas::MatrixMultiplyReduceToScalarGenericOp newMultOp =
          rewriter.create<graphblas::MatrixMultiplyReduceToScalarGenericOp>(
              loc, op->getResultTypes(), operands, attributes.getAttrs(),
              newRegions);

      for (unsigned i = 0; i < newRegions - 2; i++) {
        newMultOp.getRegion(i).takeBody(*multiplyExtensions[i]);
      }

      RegionRange reduceExtensions = op.extensions();
      Region &aggRegion0 = newMultOp.getRegion(newRegions - 2);
      aggRegion0.takeBody(*reduceExtensions[0]);
      Region &aggRegion1 = newMultOp.getRegion(newRegions - 1);
      aggRegion1.takeBody(*reduceExtensions[1]);

      rewriter.replaceOp(op, newMultOp.getResult());
      rewriter.eraseOp(predecessor);
      return success();
    }
    return failure();
  };
};

class FuseMatrixMultiplyApplyRewrite
    : public OpRewritePattern<graphblas::ApplyGenericOp> {
public:
  using OpRewritePattern<graphblas::ApplyGenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::ApplyGenericOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.input();
    graphblas::MatrixMultiplyGenericOp predecessor =
        input.getDefiningOp<graphblas::MatrixMultiplyGenericOp>();

    if (predecessor != nullptr && predecessor->hasOneUse()) {
      Location loc = op->getLoc();

      // Build new MatrixMultiplyApply op with the operands and regions of the
      // multiply, then add in the aggregator from the Apply
      ValueRange operands = predecessor.getOperands();
      NamedAttrList attributes = predecessor->getAttrs();
      RegionRange multiplyExtensions = predecessor.extensions();
      unsigned newRegions = multiplyExtensions.size();

      ExtensionBlocks multiplyBlocks;
      std::set<graphblas::YieldKind> required = {
          graphblas::YieldKind::ADD_IDENTITY, graphblas::YieldKind::ADD,
          graphblas::YieldKind::MULT};
      std::set<graphblas::YieldKind> optional = {
          graphblas::YieldKind::TRANSFORM_OUT};
      LogicalResult result = multiplyBlocks.extractBlocks(
          op, multiplyExtensions, required, optional);
      if (result.failed()) {
        return result;
      }

      if (multiplyBlocks.transformOut) {
        return failure(); // FIXME: cannot fuse with existing transform for now
      } else {
        newRegions += 1; // adding new transformOut block
      }

      RegionRange applyExtensions = op.extensions();

      graphblas::MatrixMultiplyGenericOp newMultOp =
          rewriter.create<graphblas::MatrixMultiplyGenericOp>(
              loc, op->getResultTypes(), operands, attributes.getAttrs(),
              newRegions);

      for (unsigned i = 0; i < newRegions - 1; i++) {
        newMultOp.getRegion(i).takeBody(*multiplyExtensions[i]);
      }

      Region &transformOutRegion = newMultOp.getRegion(newRegions - 1);
      transformOutRegion.takeBody(*applyExtensions[0]);

      rewriter.replaceOp(op, newMultOp.getResult());

      return success();
    }
    return failure();
  };
};

void populateGraphBLASOptimizePatterns(RewritePatternSet &patterns) {
  patterns.add<FuseMatrixMultiplyApplyRewrite, FuseMatrixMultiplyReduceRewrite>(
      patterns.getContext());
}

struct GraphBLASOptimizePass
    : public GraphBLASOptimizeBase<GraphBLASOptimizePass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);
    populateGraphBLASOptimizePatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // end anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createGraphBLASOptimizePass() {
  return std::make_unique<GraphBLASOptimizePass>();
}
