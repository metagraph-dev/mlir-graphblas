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
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/None.h"
#include "mlir/IR/Region.h"

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

class FuseMatrixSelectRewrite : public OpRewritePattern<graphblas::MatrixSelectOp>
{
public:
  using OpRewritePattern<graphblas::MatrixSelectOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::MatrixSelectOp op, PatternRewriter &rewriter) const
  {
    Value input = op.input();
    Location loc = op.getLoc();

    SmallVector<graphblas::MatrixSelectOp, 3> selectOps;

    for (OpOperand &inputUse : input.getUses()) {
      graphblas::MatrixSelectOp user = llvm::dyn_cast_or_null<graphblas::MatrixSelectOp>(inputUse.getOwner());
      if (user != nullptr) {
        selectOps.push_back(user);
      }
    }

    if (selectOps.size() > 1) {
      // time for some fusion
      SmallVector<StringRef, 3> selectors;
      SmallVector<Type, 3> resultTypes;

      for (graphblas::MatrixSelectOp selectOp : selectOps) {
        for (Attribute selectorStr : selectOp.selectors()) {
          selectors.push_back(selectorStr.dyn_cast<StringAttr>().getValue());
        }

        ValueTypeRange<ResultRange> opResultTypes = selectOp.getResultTypes();
        resultTypes.insert(resultTypes.end(), opResultTypes.begin(), opResultTypes.end());
      }

      NamedAttrList attrs;
      attrs.set("selectors", rewriter.getStrArrayAttr(selectors));
      graphblas::MatrixSelectOp fusedOp = rewriter.create<graphblas::MatrixSelectOp>(loc, resultTypes, input, attrs);
      ValueRange fusedResults = fusedOp.getResults();

      unsigned i = 0;
      for (graphblas::MatrixSelectOp selectOp : selectOps)
      {
        SmallVector<Value, 3> results;
        for (unsigned j=0; j < selectOp.getNumResults(); j++) {
          results.push_back(fusedResults[i]);
          i++;
        }
        rewriter.replaceOp(selectOp, results);
      }
    }

    return failure();
  };
};

class FuseMatrixMultiplyReduceRewrite : public OpRewritePattern<graphblas::MatrixReduceToScalarGenericOp>
{
public:
  using OpRewritePattern<graphblas::MatrixReduceToScalarGenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::MatrixReduceToScalarGenericOp op, PatternRewriter &rewriter) const
  {
    Value input = op.input();
    graphblas::MatrixMultiplyGenericOp predecessor = input.getDefiningOp<graphblas::MatrixMultiplyGenericOp>();
    if (predecessor != nullptr && predecessor->hasOneUse()) {
      Location loc = op->getLoc();

      if (getRank(predecessor.a()) < 2 || getRank(predecessor.b()) < 2)
        return failure();

      // Build new MatrixMultiplyReduceToScalarGeneric op with the operands and regions of the multiply,
      // then add in the aggregator from the reduce
      ValueRange operands = predecessor.getOperands();
      NamedAttrList attributes = predecessor->getAttrs();
      RegionRange multiplyExtensions = predecessor.extensions();
      unsigned newRegions = multiplyExtensions.size();

      ExtensionBlocks multiplyBlocks;
      std::set<graphblas::YieldKind> required = {
          graphblas::YieldKind::ADD_IDENTITY,
          graphblas::YieldKind::ADD,
          graphblas::YieldKind::MULT};
      std::set<graphblas::YieldKind> optional = {graphblas::YieldKind::TRANSFORM_OUT};
      LogicalResult result = multiplyBlocks.extractBlocks(op, multiplyExtensions, required, optional);
      if (result.failed())
      {
        return result;
      }

      if (multiplyBlocks.transformOut)
      {
        return failure(); // FIXME: cannot fuse with existing transform for now
      }
      else
      {
        newRegions += 2; // adding new agg and agg identity block
      }

      graphblas::MatrixMultiplyReduceToScalarGenericOp newMultOp = rewriter.create<graphblas::MatrixMultiplyReduceToScalarGenericOp>(loc,
          op->getResultTypes(), operands, attributes.getAttrs(), newRegions);

      for (unsigned i = 0; i < newRegions - 2; i++)
      {
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

class FuseMatrixMultiplyApplyRewrite : public OpRewritePattern<graphblas::MatrixApplyGenericOp>
{
public:
  using OpRewritePattern<graphblas::MatrixApplyGenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::MatrixApplyGenericOp op, PatternRewriter &rewriter) const
  {
    Value input = op.input();
    graphblas::MatrixMultiplyGenericOp predecessor = input.getDefiningOp<graphblas::MatrixMultiplyGenericOp>();

    if (predecessor != nullptr && predecessor->hasOneUse())
    {
      Location loc = op->getLoc();

      // Build new MatrixMultiplyApply op with the operands and regions of the multiply,
      // then add in the aggregator from the Apply
      ValueRange operands = predecessor.getOperands();
      NamedAttrList attributes = predecessor->getAttrs();
      RegionRange multiplyExtensions = predecessor.extensions();
      unsigned newRegions = multiplyExtensions.size();

      ExtensionBlocks multiplyBlocks;
      std::set<graphblas::YieldKind> required = {
          graphblas::YieldKind::ADD_IDENTITY,
          graphblas::YieldKind::ADD,
          graphblas::YieldKind::MULT};
      std::set<graphblas::YieldKind> optional = {graphblas::YieldKind::TRANSFORM_OUT};
      LogicalResult result = multiplyBlocks.extractBlocks(op, multiplyExtensions, required, optional);
      if (result.failed()) {
        return result;
      }

      if (multiplyBlocks.transformOut)
      {
        return failure(); // FIXME: cannot fuse with existing transform for now
      }
      else
      {
        newRegions += 1; // adding new transformOut block
      }

      RegionRange applyExtensions = op.extensions();

      graphblas::MatrixMultiplyGenericOp newMultOp = rewriter.create<graphblas::MatrixMultiplyGenericOp>(loc,
                                op->getResultTypes(), operands, attributes.getAttrs(),
                                newRegions);

      for (unsigned i=0; i < newRegions - 1; i++) {
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

void populateGraphBLASOptimizePatterns(RewritePatternSet &patterns){
  patterns.add<
      FuseMatrixSelectRewrite,
      FuseMatrixMultiplyApplyRewrite,
      FuseMatrixMultiplyReduceRewrite>(patterns.getContext());
}

struct GraphBLASOptimizePass : public GraphBLASOptimizeBase<GraphBLASOptimizePass> {
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
