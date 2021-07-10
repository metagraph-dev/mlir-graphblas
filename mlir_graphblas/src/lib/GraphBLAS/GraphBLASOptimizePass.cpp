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

class FuseMatrixMultiplyReduceRewrite : public OpRewritePattern<graphblas::MatrixReduceToScalarOp>
{
public:
  using OpRewritePattern<graphblas::MatrixReduceToScalarOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::MatrixReduceToScalarOp op, PatternRewriter &rewriter) const
  {
    Value input = op.input();
    graphblas::MatrixMultiplyOp predecessor = input.getDefiningOp<graphblas::MatrixMultiplyOp>();
    if (predecessor != nullptr && predecessor->hasOneUse()) {
      Location loc = op->getLoc();

      // Build new MatrixMultiplyReduce op with the operands and arguments of the multiply,
      // then add in the aggregator from the reduce
      ValueRange operands = predecessor.getOperands();
      NamedAttrList attributes = predecessor->getAttrs();

      StringAttr aggregator = rewriter.getStringAttr(op.aggregator());
      attributes.push_back(rewriter.getNamedAttr("aggregator", aggregator));

      Value result = rewriter.create<graphblas::MatrixMultiplyReduceToScalarOp>(loc, 
        op->getResultTypes(), operands, attributes.getAttrs());

      rewriter.replaceOp(op, result);
      
      return success();
    }
    return failure();
  };
};

class FuseMatrixMultiplyApplyRewrite : public OpRewritePattern<graphblas::MatrixApplyOp>
{
public:
  using OpRewritePattern<graphblas::MatrixApplyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::MatrixApplyOp op, PatternRewriter &rewriter) const
  {
    Value input = op.input();
    graphblas::MatrixMultiplyOp predecessor = input.getDefiningOp<graphblas::MatrixMultiplyOp>();

    if (predecessor != nullptr && predecessor->hasOneUse())
    {
      Location loc = op->getLoc();

      // Build new MatrixMultiplyApply op with the operands and arguments of the multiply,
      // then add in the aggregator from the Apply
      ValueRange operands = predecessor.getOperands();
      NamedAttrList attributes = predecessor->getAttrs();
      Value thunk = op.thunk();
      StringRef apply_operator = op.apply_operator();

      RankedTensorType tensorType = predecessor.a().getType().dyn_cast<RankedTensorType>();
      Type valueType = tensorType.getElementType();

      graphblas::MatrixMultiplyOp newMultOp = rewriter.create<graphblas::MatrixMultiplyOp>(loc,
                                op->getResultTypes(), operands, attributes.getAttrs());

      Region &region = newMultOp.getRegion();
      Block *transformBlock = rewriter.createBlock(&region, region.begin(), valueType);
      Value blockInput = transformBlock->getArgument(0);

      if (apply_operator == "min") {
        Value cmp = rewriter.create<mlir::CmpFOp>(loc, mlir::CmpFPredicate::OLT, blockInput, thunk);
        Value result = rewriter.create<mlir::SelectOp>(loc, cmp, blockInput, thunk);
        rewriter.create<graphblas::YieldOp>(loc, result);
      } else {
        return op.emitError("invalid apply_operator: " + apply_operator);
      }

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
