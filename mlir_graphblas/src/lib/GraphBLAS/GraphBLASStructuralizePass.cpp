//===- GraphBLASPasses.cpp - GraphBLAS dialect passes ---------*- C++ -*-===//
//
// TODO add documentation
//
//===--------------------------------------------------------------------===//
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
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

#include "GraphBLAS/GraphBLASArrayUtils.h"
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

class TransposeDWIMRewrite : public OpRewritePattern<graphblas::TransposeOp> {
public:
  using OpRewritePattern<graphblas::TransposeOp>::OpRewritePattern;

  static bool needsDWIM(graphblas::TransposeOp op) {

    Value inputTensor = op.input();
    RankedTensorType inputType =
        inputTensor.getType().dyn_cast<RankedTensorType>();
    RankedTensorType outputType =
        op->getResultTypes().front().dyn_cast<RankedTensorType>();

    bool inputTypeIsCSR = hasRowOrdering(inputType);
    bool outputTypeIsCSR = hasRowOrdering(outputType);

    return (inputTypeIsCSR == outputTypeIsCSR);
  };

  LogicalResult match(graphblas::TransposeOp op) const override {
    if (needsDWIM(op))
      return success();
    else
      return failure();
  };

  void rewrite(graphblas::TransposeOp op,
               PatternRewriter &rewriter) const override {
    MLIRContext *context = op.getContext();
    Location loc = op->getLoc();

    Value inputTensor = op.input();
    RankedTensorType outputType =
        op->getResultTypes().front().dyn_cast<RankedTensorType>();

    RankedTensorType flippedInputType =
        getFlippedLayoutType(context, inputTensor.getType());

    Value flippedInput = rewriter.create<graphblas::ConvertLayoutOp>(
        loc, flippedInputType, inputTensor);
    Value transposed =
        rewriter.create<graphblas::TransposeOp>(loc, outputType, flippedInput);

    rewriter.replaceOp(op, transposed);
  };
};

class ReduceToVectorDWIMRewrite
    : public OpRewritePattern<graphblas::ReduceToVectorOp> {
public:
  using OpRewritePattern<graphblas::ReduceToVectorOp>::OpRewritePattern;

  static bool needsDWIM(graphblas::ReduceToVectorOp op) {
    int axis = op.axis();
    bool isCSR = hasRowOrdering(op.input().getType());
    return ((axis == 0 && isCSR) || (axis == 1 && !isCSR));
  };

  LogicalResult matchAndRewrite(graphblas::ReduceToVectorOp op,
                                PatternRewriter &rewriter) const override {
    if (!needsDWIM(op))
      return failure();

    MLIRContext *context = op.getContext();
    Location loc = op->getLoc();

    Value input = op.input();
    RankedTensorType flippedInputType =
        getFlippedLayoutType(context, input.getType());

    rewriter.setInsertionPoint(op);
    Value flippedInput = rewriter.create<graphblas::ConvertLayoutOp>(
        loc, flippedInputType, input);
    op.inputMutable().assign(flippedInput);

    return success();
  };
};

class MatrixMultiplyGenericDWIMFirstArgRewrite
    : public OpRewritePattern<graphblas::MatrixMultiplyGenericOp> {
public:
  using OpRewritePattern<graphblas::MatrixMultiplyGenericOp>::OpRewritePattern;

  template <class T>
  static bool needsDWIM(T op) {
    return hasColumnOrdering(op.a().getType());
  };

  LogicalResult matchAndRewrite(graphblas::MatrixMultiplyGenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!needsDWIM(op))
      return failure();

    MLIRContext *context = op.getContext();
    Location loc = op->getLoc();
    Value A = op.a();
    RankedTensorType aType = A.getType().cast<RankedTensorType>();
    RankedTensorType flippedMatrixType = getFlippedLayoutType(context, aType);

    rewriter.setInsertionPoint(op);
    Value flippedA =
        rewriter.create<graphblas::ConvertLayoutOp>(loc, flippedMatrixType, A);
    op.aMutable().assign(flippedA);

    return success();
  };
};

class MatrixMultiplyGenericDWIMSecondArgRewrite
    : public OpRewritePattern<graphblas::MatrixMultiplyGenericOp> {
public:
  using OpRewritePattern<graphblas::MatrixMultiplyGenericOp>::OpRewritePattern;

  template <class T>
  static bool needsDWIM(T op) {
    return hasRowOrdering(op.b().getType());
  };

  LogicalResult matchAndRewrite(graphblas::MatrixMultiplyGenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!needsDWIM(op))
      return failure();

    MLIRContext *context = op.getContext();
    Location loc = op->getLoc();
    Value B = op.b();
    RankedTensorType bType = B.getType().cast<RankedTensorType>();
    RankedTensorType flippedMatrixType = getFlippedLayoutType(context, bType);

    rewriter.setInsertionPoint(op);
    Value flippedB =
        rewriter.create<graphblas::ConvertLayoutOp>(loc, flippedMatrixType, B);
    op.bMutable().assign(flippedB);

    return success();
  };
};

class MatrixMultiplyGenericDWIMMaskRewrite
    : public OpRewritePattern<graphblas::MatrixMultiplyGenericOp> {
public:
  using OpRewritePattern<graphblas::MatrixMultiplyGenericOp>::OpRewritePattern;

  template <class T>
  static bool needsDWIM(T op) {
    Value mask = op.mask();
    if (!mask)
      return false;
    return hasColumnOrdering(mask.getType());
  };

  LogicalResult matchAndRewrite(graphblas::MatrixMultiplyGenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!needsDWIM(op))
      return failure();

    MLIRContext *context = op.getContext();
    Location loc = op->getLoc();

    Value mask = op.mask();
    RankedTensorType maskType = mask.getType().cast<RankedTensorType>();
    RankedTensorType flippedMatrixType =
        getFlippedLayoutType(context, maskType);

    rewriter.setInsertionPoint(op);
    Value flippedMask = rewriter.create<graphblas::ConvertLayoutOp>(
        loc, flippedMatrixType, mask);
    op.maskMutable().assign(flippedMask);

    return success();
  };
};

class MatrixMultiplyReduceToScalarGenericDWIMFirstArgRewrite
    : public OpRewritePattern<
          graphblas::MatrixMultiplyReduceToScalarGenericOp> {
public:
  using OpRewritePattern<
      graphblas::MatrixMultiplyReduceToScalarGenericOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(graphblas::MatrixMultiplyReduceToScalarGenericOp op,
                  PatternRewriter &rewriter) const override {
    if (!MatrixMultiplyGenericDWIMFirstArgRewrite::needsDWIM(op))
      return failure();

    MLIRContext *context = op.getContext();
    Location loc = op->getLoc();
    Value A = op.a();
    RankedTensorType aType = A.getType().cast<RankedTensorType>();
    RankedTensorType flippedMatrixType = getFlippedLayoutType(context, aType);

    rewriter.setInsertionPoint(op);
    Value flippedA =
        rewriter.create<graphblas::ConvertLayoutOp>(loc, flippedMatrixType, A);
    op.aMutable().assign(flippedA);

    return success();
  };
};

class MatrixMultiplyReduceToScalarGenericDWIMSecondArgRewrite
    : public OpRewritePattern<
          graphblas::MatrixMultiplyReduceToScalarGenericOp> {
public:
  using OpRewritePattern<
      graphblas::MatrixMultiplyReduceToScalarGenericOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(graphblas::MatrixMultiplyReduceToScalarGenericOp op,
                  PatternRewriter &rewriter) const override {
    if (!MatrixMultiplyGenericDWIMSecondArgRewrite::needsDWIM(op))
      return failure();

    MLIRContext *context = op.getContext();
    Location loc = op->getLoc();
    Value B = op.b();
    RankedTensorType bType = B.getType().cast<RankedTensorType>();
    RankedTensorType flippedMatrixType = getFlippedLayoutType(context, bType);

    rewriter.setInsertionPoint(op);
    Value flippedB =
        rewriter.create<graphblas::ConvertLayoutOp>(loc, flippedMatrixType, B);
    op.bMutable().assign(flippedB);

    return success();
  };
};

class MatrixMultiplyReduceToScalarGenericDWIMMaskRewrite
    : public OpRewritePattern<
          graphblas::MatrixMultiplyReduceToScalarGenericOp> {
public:
  using OpRewritePattern<
      graphblas::MatrixMultiplyReduceToScalarGenericOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(graphblas::MatrixMultiplyReduceToScalarGenericOp op,
                  PatternRewriter &rewriter) const override {
    if (!MatrixMultiplyGenericDWIMMaskRewrite::needsDWIM(op))
      return failure();

    MLIRContext *context = op.getContext();
    Location loc = op->getLoc();

    Value mask = op.mask();
    RankedTensorType maskType = mask.getType().cast<RankedTensorType>();
    RankedTensorType flippedMatrixType =
        getFlippedLayoutType(context, maskType);

    rewriter.setInsertionPoint(op);
    Value flippedMask = rewriter.create<graphblas::ConvertLayoutOp>(
        loc, flippedMatrixType, mask);
    op.maskMutable().assign(flippedMask);

    return success();
  };
};

class LowerMatrixMultiplyRewrite
    : public OpRewritePattern<graphblas::MatrixMultiplyOp> {
public:
  using OpRewritePattern<graphblas::MatrixMultiplyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::MatrixMultiplyOp op,
                                PatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>(); /* ignore unused variable
                                                          for debugging */
    (void)module;
    Location loc = op->getLoc();

    // Inputs
    ValueRange operands = op.getOperands();
    StringRef semiring = op.semiring();
    bool maskComplement = op.mask_complement();

    // Types
    // Can't use result here because it might be a scalar (vector-vector)
    Type valueType =
        op.a().getType().dyn_cast<RankedTensorType>().getElementType();

    // New op
    NamedAttrList attributes = {};
    attributes.append(StringRef("mask_complement"),
                      rewriter.getBoolAttr(maskComplement));
    graphblas::MatrixMultiplyGenericOp newMultOp =
        rewriter.create<graphblas::MatrixMultiplyGenericOp>(
            loc, op->getResultTypes(), operands, attributes.getAttrs(), 3);

    if (failed(populateSemiring(rewriter, loc, semiring, valueType,
                                newMultOp.getRegions().slice(0, 3))))
      return failure();

    rewriter.setInsertionPointAfter(newMultOp);

    rewriter.replaceOp(op, newMultOp.getResult());

    return success();
  };
};

class LowerApplyRewrite : public OpRewritePattern<graphblas::ApplyOp> {
public:
  using OpRewritePattern<graphblas::ApplyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::ApplyOp op,
                                PatternRewriter &rewriter) const override {
    Value input, thunk;
    LogicalResult extractArgResult = extractApplyOpArgs(op, input, thunk);
    assert(!extractArgResult.failed() &&
           "Assumption that extractApplyOpArgs succeeded (due to verify "
           "method) has been violated.");

    StringRef apply_operator = op.apply_operator();
    if (apply_operator == "identity") {
      // This doesn't produce a copy like we do for all the other operators
      rewriter.replaceOp(op, input);
      return success();
    }

    ModuleOp module = op->getParentOfType<ModuleOp>(); /* ignore unused variable
                                                          for debugging */
    (void)module;
    Location loc = op->getLoc();

    Type valueType =
        input.getType().dyn_cast<RankedTensorType>().getElementType();

    // New op
    graphblas::ApplyGenericOp newApplyOp =
        rewriter.create<graphblas::ApplyGenericOp>(loc, op->getResultTypes(),
                                                   input, op.in_place(), 1);

    // Populate based on operator kind
    LogicalResult popResult = failure();
    if (unary1.contains(apply_operator) || unary3.contains(apply_operator)) {
      popResult = populateUnary(rewriter, loc, apply_operator, valueType,
                                newApplyOp.getRegions().slice(0, 1),
                                graphblas::YieldKind::TRANSFORM_OUT);
      if (failed(popResult))
        return failure();
    } else {
      popResult = populateBinary(rewriter, loc, apply_operator, valueType,
                                 newApplyOp.getRegions().slice(0, 1),
                                 graphblas::YieldKind::TRANSFORM_OUT);
      if (failed(popResult))
        return failure();
      // Remove thunk from populated block
      Block &block = newApplyOp.getRegion(0).front();
      int thunkPos = thunk == op.left() ? 0 : 1;
      Value thunkArg = block.getArgument(thunkPos);
      thunkArg.replaceAllUsesWith(thunk);
      block.eraseArgument(thunkPos);
    }

    rewriter.replaceOp(op, newApplyOp.getResult());

    return success();
  };
};

class LowerSelectRewrite : public OpRewritePattern<graphblas::SelectOp> {
public:
  using OpRewritePattern<graphblas::SelectOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::SelectOp op,
                                PatternRewriter &rewriter) const override {

    std::string selector = op.selector().str();
    OperandRange thunks = op.thunks();

    // Don't handle custom selectors
    if (selector == "probability")
      return failure();

    Location loc = op->getLoc();

    Value input = op.input();
    RankedTensorType inputType = input.getType().cast<RankedTensorType>();
    Type valueType = inputType.getElementType();

    // Replace with SelectGenericOp
    graphblas::SelectGenericOp newSelectOp =
        rewriter.create<graphblas::SelectGenericOp>(loc, op->getResultTypes(),
                                                    input, 1);

    // Populate based on operator kind
    LogicalResult popResult = failure();
    if (unary1.contains(selector) || unary3.contains(selector)) {
      popResult = populateUnary(rewriter, loc, selector, valueType,
                                newSelectOp.getRegions().slice(0, 1),
                                graphblas::YieldKind::SELECT_OUT,
                                /* boolAsI8 */ false);
    } else {
      popResult = populateBinary(rewriter, loc, selector, valueType,
                                 newSelectOp.getRegions().slice(0, 1),
                                 graphblas::YieldKind::SELECT_OUT,
                                 /* boolAsI8 */ false);
    }
    if (failed(popResult))
      return failure();

    // Remove thunk from populated block
    if (binary2.contains(selector) || binary4.contains(selector)) {
      Value thunk = thunks[0];
      Block &block = newSelectOp.getRegion(0).front();
      Value thunkArg = block.getArgument(1);
      thunkArg.replaceAllUsesWith(thunk);
      block.eraseArgument(1);
    }

    rewriter.setInsertionPointAfter(newSelectOp);
    rewriter.replaceOp(op, newSelectOp.getResult());

    return success();
  };
};

class LowerUnionRewrite : public OpRewritePattern<graphblas::UnionOp> {
public:
  using OpRewritePattern<graphblas::UnionOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::UnionOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    Value a = op.a();
    Value b = op.b();
    Value mask = op.mask();
    Type valueType = a.getType().cast<RankedTensorType>().getElementType();

    if (mask) {
      NamedAttrList attributes = {};
      attributes.append(StringRef("mask_complement"),
                        rewriter.getBoolAttr(op.mask_complement()));
      a = rewriter.create<graphblas::SelectMaskOp>(
          loc, a.getType(), ValueRange{a, mask}, attributes.getAttrs());
      b = rewriter.create<graphblas::SelectMaskOp>(
          loc, b.getType(), ValueRange{b, mask}, attributes.getAttrs());
    }

    // New op
    NamedAttrList attributes = {};
    graphblas::UnionGenericOp newUnionOp =
        rewriter.create<graphblas::UnionGenericOp>(loc, op->getResultTypes(),
                                                   ValueRange{a, b},
                                                   attributes.getAttrs(), 1);

    if (failed(populateBinary(rewriter, loc, op.union_operator(), valueType,
                              newUnionOp.getRegions().slice(0, 1),
                              graphblas::YieldKind::MULT)))
      return failure();

    rewriter.setInsertionPointAfter(newUnionOp);

    rewriter.replaceOp(op, newUnionOp.getResult());

    return success();
  };
};

class LowerIntersectRewrite : public OpRewritePattern<graphblas::IntersectOp> {
public:
  using OpRewritePattern<graphblas::IntersectOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::IntersectOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value a = op.a();
    Value b = op.b();
    Value mask = op.mask();
    Type valueType = a.getType().cast<RankedTensorType>().getElementType();
    StringRef opstr = op.intersect_operator();

    if (mask) {
      NamedAttrList attributes = {};
      attributes.append(StringRef("mask_complement"),
                        rewriter.getBoolAttr(op.mask_complement()));
      a = rewriter.create<graphblas::SelectMaskOp>(
          loc, a.getType(), ValueRange{a, mask}, attributes.getAttrs());
      b = rewriter.create<graphblas::SelectMaskOp>(
          loc, b.getType(), ValueRange{b, mask}, attributes.getAttrs());
    }

    // Special handling for "first" and "second"
    if (opstr == "first") {
      graphblas::SelectMaskOp newIntersectOp =
          rewriter.create<graphblas::SelectMaskOp>(loc, op->getResultTypes(),
                                                   ValueRange{a, b});
      rewriter.replaceOp(op, newIntersectOp.getResult());
    } else if (opstr == "second") {
      graphblas::SelectMaskOp newIntersectOp =
          rewriter.create<graphblas::SelectMaskOp>(loc, op->getResultTypes(),
                                                   ValueRange{b, a});
      rewriter.replaceOp(op, newIntersectOp.getResult());
    } else {
      // New op
      NamedAttrList attributes = {};
      graphblas::IntersectGenericOp newIntersectOp =
          rewriter.create<graphblas::IntersectGenericOp>(
              loc, op->getResultTypes(), ValueRange{a, b},
              attributes.getAttrs(), 1);

      if (failed(populateBinary(rewriter, loc, op.intersect_operator(),
                                valueType,
                                newIntersectOp.getRegions().slice(0, 1),
                                graphblas::YieldKind::MULT)))
        return failure();

      rewriter.setInsertionPointAfter(newIntersectOp);
      rewriter.replaceOp(op, newIntersectOp.getResult());
    }

    return success();
  };
};

class LowerUpdateRewrite : public OpRewritePattern<graphblas::UpdateOp> {
public:
  using OpRewritePattern<graphblas::UpdateOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::UpdateOp op,
                                PatternRewriter &rewriter) const override {
    // ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    Type valueType =
        op.input().getType().cast<RankedTensorType>().getElementType();
    bool maskComplement = op.mask_complement();
    bool replace = op.replace();
    llvm::Optional<llvm::StringRef> accumulateOperator =
        op.accumulate_operator();

    // Only handle lowering to generic for accumulation
    if (!accumulateOperator)
      return failure();

    // Create generic op
    NamedAttrList attributes = {};
    attributes.append(StringRef("mask_complement"),
                      rewriter.getBoolAttr(maskComplement));
    attributes.append(StringRef("replace"), rewriter.getBoolAttr(replace));
    graphblas::UpdateGenericOp newUpdateOp =
        rewriter.create<graphblas::UpdateGenericOp>(loc, op->getResultTypes(),
                                                    op.getOperands(),
                                                    attributes.getAttrs(), 1);

    if (failed(populateBinary(rewriter, loc, accumulateOperator->str(),
                              valueType, newUpdateOp.getRegions().slice(0, 1),
                              graphblas::YieldKind::ACCUMULATE)))
      return failure();

    rewriter.setInsertionPointAfter(newUpdateOp);
    rewriter.eraseOp(op);

    return success();
  };
};

class LowerReduceToVectorRewrite
    : public OpRewritePattern<graphblas::ReduceToVectorOp> {
public:
  using OpRewritePattern<graphblas::ReduceToVectorOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::ReduceToVectorOp op,
                                PatternRewriter &rewriter) const override {
    if (ReduceToVectorDWIMRewrite::needsDWIM(op))
      return failure();

    StringRef aggregator = op.aggregator();

    // Don't handle custom aggregators
    if (aggregator == "count" or aggregator == "argmin" or
        aggregator == "argmax" or aggregator == "first" or aggregator == "last")
      return failure();

    Value input = op.input();
    RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();
    Type elementType = inputType.getElementType();
    Type i64Type = rewriter.getI64Type();

    Location loc = op->getLoc();

    NamedAttrList attributes = {};
    attributes.append(StringRef("axis"),
                      rewriter.getIntegerAttr(i64Type, op.axis()));
    attributes.append(StringRef("mask_complement"),
                      rewriter.getBoolAttr(op.mask_complement()));
    graphblas::ReduceToVectorGenericOp newReduceOp =
        rewriter.create<graphblas::ReduceToVectorGenericOp>(
            loc, op->getResultTypes(), input, attributes.getAttrs(), 2);

    if (failed(populateMonoid(rewriter, loc, op.aggregator(), elementType,
                              newReduceOp.getRegions().slice(0, 2),
                              graphblas::YieldKind::AGG_IDENTITY,
                              graphblas::YieldKind::AGG)))
      return failure();

    rewriter.setInsertionPointAfter(newReduceOp);
    rewriter.replaceOp(op, newReduceOp.getResult());

    return success();
  };
};

class LowerReduceToScalarRewrite
    : public OpRewritePattern<graphblas::ReduceToScalarOp> {
public:
  using OpRewritePattern<graphblas::ReduceToScalarOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::ReduceToScalarOp op,
                                PatternRewriter &rewriter) const override {
    StringRef aggregator = op.aggregator();
    Value input = op.input();
    Location loc = op->getLoc();

    // Don't handle custom aggregators (other than "count")
    if (aggregator == "argmin" or aggregator == "argmax")
      return failure();

    if (aggregator == "count") {
      // Rewrite using graphblas.num_vals
      Type int64Type = rewriter.getIntegerType(64);

      Value countOp = rewriter.create<graphblas::NumValsOp>(loc, input);
      Value countOp_64 =
          rewriter.create<arith::IndexCastOp>(loc, countOp, int64Type);
      rewriter.replaceOp(op, countOp_64);
    } else {
      // Rewrite as generic op
      Type valueType =
          input.getType().cast<RankedTensorType>().getElementType();

      graphblas::ReduceToScalarGenericOp newReduceOp =
          rewriter.create<graphblas::ReduceToScalarGenericOp>(
              loc, op->getResultTypes(), input, 2);

      if (failed(populateMonoid(rewriter, loc, op.aggregator(), valueType,
                                newReduceOp.getRegions().slice(0, 2),
                                graphblas::YieldKind::AGG_IDENTITY,
                                graphblas::YieldKind::AGG)))
        return failure();

      rewriter.setInsertionPointAfter(newReduceOp);
      rewriter.replaceOp(op, newReduceOp.getResult());
    }

    return success();
  };
};

void populateGraphBLASDWIMPatterns(RewritePatternSet &patterns) {
  patterns.add<TransposeDWIMRewrite, ReduceToVectorDWIMRewrite,
               MatrixMultiplyGenericDWIMFirstArgRewrite,
               MatrixMultiplyGenericDWIMSecondArgRewrite,
               MatrixMultiplyGenericDWIMMaskRewrite,
               MatrixMultiplyReduceToScalarGenericDWIMFirstArgRewrite,
               MatrixMultiplyReduceToScalarGenericDWIMSecondArgRewrite,
               MatrixMultiplyReduceToScalarGenericDWIMMaskRewrite>(
      patterns.getContext());
}

struct GraphBLASDWIMPass : public GraphBLASDWIMBase<GraphBLASDWIMPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);
    populateGraphBLASDWIMPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

void populateGraphBLASStructuralizePatterns(RewritePatternSet &patterns) {
  patterns
      .add<LowerMatrixMultiplyRewrite, LowerApplyRewrite, LowerSelectRewrite,
           LowerUnionRewrite, LowerIntersectRewrite, LowerUpdateRewrite,
           LowerReduceToVectorRewrite, LowerReduceToScalarRewrite>(
          patterns.getContext());
}

struct GraphBLASStructuralizePass
    : public GraphBLASStructuralizeBase<GraphBLASStructuralizePass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);
    populateGraphBLASStructuralizePatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
} // end anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createGraphBLASDWIMPass() {
  return std::make_unique<GraphBLASDWIMPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createGraphBLASStructuralizePass() {
  return std::make_unique<GraphBLASStructuralizePass>();
}
