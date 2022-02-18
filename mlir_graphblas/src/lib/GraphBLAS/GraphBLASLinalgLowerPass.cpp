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

class LinalgLowerApplyGenericRewrite
    : public OpRewritePattern<graphblas::ApplyGenericOp> {
public:
  using OpRewritePattern<graphblas::ApplyGenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::ApplyGenericOp op,
                                PatternRewriter &rewriter) const override {

    MLIRContext *context = op.getContext();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    Value inputTensor = op.input();
    RankedTensorType outputTensorType =
        op.getResult().getType().cast<RankedTensorType>();
    // bool inPlace = op.in_place();

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value nrows = rewriter.create<tensor::DimOp>(loc, inputTensor, c0);
    Value ncols = rewriter.create<tensor::DimOp>(loc, inputTensor, c1);

    Value accumulatorTensor = rewriter.create<sparse_tensor::InitOp>(
        loc, outputTensorType, ValueRange{nrows, ncols});

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

    graphblas::YieldOp transformOutYield =
        llvm::dyn_cast_or_null<graphblas::YieldOp>(
            extBlocks.transformOut->getTerminator());

    // TODO grab these from somewhere else
    AffineMap map =
        AffineMap::getPermutationMap(ArrayRef<unsigned>{1, 0}, context);
    SmallVector<AffineMap, 2> affineMaps = {map, map};
    SmallVector<StringRef> iteratorTypes = {getParallelIteratorTypeName(),
                                            getParallelIteratorTypeName()};

    linalg::GenericOp linalgGenericOp = rewriter.create<linalg::GenericOp>(
        loc, outputTensorType, inputTensor, accumulatorTensor, affineMaps,
        iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          ValueRange arguments = nestedBuilder.getBlock()->getArguments();
          ValueRange subVals = ValueRange{arguments.front()};
          rewriter.mergeBlocks(extBlocks.transformOut, nestedBuilder.getBlock(),
                               subVals);
          Value result = transformOutYield.values().front();
          nestedBuilder.create<linalg::YieldOp>(loc, result);
        });
    rewriter.eraseOp(transformOutYield);
    Value output = linalgGenericOp.getResult(0);

    rewriter.replaceOp(op, output);

    cleanupIntermediateTensor(rewriter, module, loc, output);

    return success();
  };
};

void populateGraphBLASLinalgLoweringPatterns(RewritePatternSet &patterns) {
  patterns.add<LinalgLowerApplyGenericRewrite>(patterns.getContext());
}

struct GraphBLASLinalgLoweringPass
    : public GraphBLASLinalgLoweringBase<GraphBLASLinalgLoweringPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);
    populateGraphBLASLinalgLoweringPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // end anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createGraphBLASLinalgLoweringPass() {
  return std::make_unique<GraphBLASLinalgLoweringPass>();
}
