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

#include "GraphBLAS/GraphBLASCommonPasses.h"
#include "GraphBLAS/GraphBLASDialect.h"
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

class LinalgLowerApplyGenericRewrite
    : public OpRewritePattern<graphblas::ApplyGenericOp> {
public:
  using OpRewritePattern<graphblas::ApplyGenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::ApplyGenericOp op,
                                PatternRewriter &rewriter) const override {

    MLIRContext *context = op.getContext();
    // ModuleOp mod = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    // sparse tensor rewriting of linalg.generic does not handle overwriting
    // tensors
    if (op.in_place())
      return rewriter.notifyMatchFailure(op, "in_place=true");

    Value inputTensor = op.input();
    RankedTensorType inputTensorType =
        inputTensor.getType().cast<RankedTensorType>();
    RankedTensorType outputTensorType =
        op.getResult().getType().cast<RankedTensorType>();
    sparse_tensor::SparseTensorEncodingAttr inEncoding =
        sparse_tensor::getSparseTensorEncoding(inputTensorType);
    sparse_tensor::SparseTensorEncodingAttr outEncoding =
        sparse_tensor::getSparseTensorEncoding(outputTensorType);

    Value accumulatorTensor;
    SmallVector<AffineMap, 2> affineMaps;
    SmallVector<StringRef, 2> iteratorTypes;

    unsigned rank = inputTensorType.getRank();
    if (rank == 1) {
      Value size = rewriter.create<graphblas::SizeOp>(loc, inputTensor);
      accumulatorTensor = rewriter.create<sparse_tensor::InitOp>(
          loc, outputTensorType, ValueRange{size});
      AffineMap map = AffineMap::getMultiDimIdentityMap(1, context);
      affineMaps.push_back(map);
      affineMaps.push_back(map);
      iteratorTypes.push_back(getParallelIteratorTypeName());
    } else {
      Value nrows = rewriter.create<graphblas::NumRowsOp>(loc, inputTensor);
      Value ncols = rewriter.create<graphblas::NumColsOp>(loc, inputTensor);
      accumulatorTensor = rewriter.create<sparse_tensor::InitOp>(
          loc, outputTensorType, ValueRange{nrows, ncols});
      affineMaps.push_back(inEncoding.getDimOrdering());
      affineMaps.push_back(outEncoding.getDimOrdering());
      iteratorTypes.push_back(getParallelIteratorTypeName());
      iteratorTypes.push_back(getParallelIteratorTypeName());
    }

    // Required blocks
    RegionRange extensions = op.extensions();
    ExtensionBlocks extBlocks;
    std::set<graphblas::YieldKind> required = {
        graphblas::YieldKind::TRANSFORM_OUT};
    LogicalResult extractResult =
        extBlocks.extractBlocks(op, extensions, required, {});

    if (extractResult.failed()) {
      return extractResult;
    }

    linalg::GenericOp linalgGenericOp = rewriter.create<linalg::GenericOp>(
        loc, outputTensorType, inputTensor, accumulatorTensor, affineMaps,
        iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          ValueRange arguments =
              nestedBuilder.getBlock()->getArguments().front();
          sparse_tensor::LinalgApplyOp linalg_apply =
              nestedBuilder.create<sparse_tensor::LinalgApplyOp>(
                  nestedLoc, outputTensorType.getElementType(), arguments);
          // Move tranform out block into linalg_apply
          Region *origRegion = extBlocks.transformOut->getParent();
          linalg_apply.getRegion().takeBody(*origRegion);
          // Replace graphblas.yield with sparse_tensor.linalg_yield
          graphblas::YieldOp graphblasYield =
              llvm::dyn_cast_or_null<graphblas::YieldOp>(
                  linalg_apply.getRegion().front().getTerminator());
          nestedBuilder.setInsertionPointAfter(graphblasYield);
          rewriter.replaceOpWithNewOp<sparse_tensor::LinalgYieldOp>(
              graphblasYield, graphblasYield.values().front());
          // Add linalg.yield
          nestedBuilder.setInsertionPointAfter(linalg_apply);
          Value result = linalg_apply.getResult();
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, result);
        });
    Value output = linalgGenericOp.getResult(0);

    rewriter.replaceOp(op, output);

    return success();
  };
};

void populateGraphBLASLinalgLoweringPatterns(RewritePatternSet &patterns) {
  patterns.add<LinalgLowerApplyGenericRewrite,
               // Common items
               LowerCommentRewrite, LowerPrintRewrite, LowerPrintTensorRewrite,
               LowerSizeRewrite, LowerNumRowsRewrite, LowerNumColsRewrite,
               LowerNumValsRewrite>(patterns.getContext());
}

struct GraphBLASLinalgLoweringPass
    : public GraphBLASLinalgLoweringBase<GraphBLASLinalgLoweringPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populateGraphBLASLinalgLoweringPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));

    // Verify that expected operations were actually handled
    ConversionTarget target(*ctx);
    target.addIllegalOp<graphblas::ApplyGenericOp>();
    // ApplyGeneric with inplace=true is not handled
    target.addDynamicallyLegalOp<graphblas::ApplyGenericOp>(
        [](graphblas::ApplyGenericOp op) { return op.in_place(); });
    RewritePatternSet noPatterns(ctx);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(noPatterns))))
      signalPassFailure();
  }
};

} // end anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createGraphBLASLinalgLoweringPass() {
  return std::make_unique<GraphBLASLinalgLoweringPass>();
}
