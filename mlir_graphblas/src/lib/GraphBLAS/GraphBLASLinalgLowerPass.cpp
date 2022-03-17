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

static void moveBlock(PatternRewriter &rewriter, Block *block, Region &region) {
  region.takeBody(*block->getParent());
  // Replace graphblas.yield with sparse_tensor.linalg_yield
  graphblas::YieldOp graphblasYield =
      llvm::dyn_cast<graphblas::YieldOp>(region.front().getTerminator());
  rewriter.setInsertionPointAfter(graphblasYield);
  rewriter.replaceOpWithNewOp<sparse_tensor::LinalgYieldOp>(
      graphblasYield, graphblasYield.values().front());
}

template <class T>
static Value buildSameShapeOutput(T op, PatternRewriter &rewriter,
                                  Value inputTensor, Type outputTensorType,
                                  SmallVector<AffineMap, 3> &affineMaps,
                                  SmallVector<StringRef, 2> &iteratorTypes,
                                  bool twoInputs = false) {
  MLIRContext *context = op.getContext();
  Location loc = op->getLoc();
  RankedTensorType inputTensorType =
      inputTensor.getType().cast<RankedTensorType>();
  unsigned rank = inputTensorType.getRank();

  for (unsigned i = 0; i < (twoInputs ? 3 : 2); i++) {
    affineMaps.push_back(AffineMap::getMultiDimIdentityMap(rank, context));
  }
  if (rank == 1) {
    Value size = rewriter.create<graphblas::SizeOp>(loc, inputTensor);
    iteratorTypes.push_back(getParallelIteratorTypeName());
    return rewriter.create<sparse_tensor::InitOp>(loc, outputTensorType,
                                                  ValueRange{size});
  } else {
    Value nrows = rewriter.create<graphblas::NumRowsOp>(loc, inputTensor);
    Value ncols = rewriter.create<graphblas::NumColsOp>(loc, inputTensor);
    iteratorTypes.push_back(getParallelIteratorTypeName());
    iteratorTypes.push_back(getParallelIteratorTypeName());
    return rewriter.create<sparse_tensor::InitOp>(loc, outputTensorType,
                                                  ValueRange{nrows, ncols});
  }
}

class LinalgLowerApplyGenericRewrite
    : public OpRewritePattern<graphblas::ApplyGenericOp> {
public:
  using OpRewritePattern<graphblas::ApplyGenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::ApplyGenericOp op,
                                PatternRewriter &rewriter) const override {
    // ModuleOp mod = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    // sparse tensor rewriting of linalg.generic does not handle overwriting
    // tensors
    if (op.in_place())
      return rewriter.notifyMatchFailure(op, "in_place=true");

    // Required blocks
    RegionRange extensions = op.extensions();
    ExtensionBlocks extBlocks;
    std::set<graphblas::YieldKind> required = {
        graphblas::YieldKind::TRANSFORM_OUT};
    LogicalResult extractResult =
        extBlocks.extractBlocks(op, extensions, required, {});

    if (extractResult.failed())
      return extractResult;

    Value inputTensor = op.input();
    RankedTensorType outputTensorType =
        op.getResult().getType().cast<RankedTensorType>();

    SmallVector<AffineMap, 3> affineMaps;
    SmallVector<StringRef, 2> iteratorTypes;
    Value accumulatorTensor = buildSameShapeOutput(
        op, rewriter, inputTensor, outputTensorType, affineMaps, iteratorTypes);

    linalg::GenericOp linalgGenericOp = rewriter.create<linalg::GenericOp>(
        loc, outputTensorType, inputTensor, accumulatorTensor, affineMaps,
        iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          ValueRange arguments =
              nestedBuilder.getBlock()->getArguments().front();
          sparse_tensor::LinalgApplyOp applyOp =
              nestedBuilder.create<sparse_tensor::LinalgApplyOp>(
                  nestedLoc, outputTensorType.getElementType(), arguments);
          moveBlock(rewriter, extBlocks.transformOut, applyOp.getRegion());
          nestedBuilder.setInsertionPointAfter(applyOp);
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, applyOp.getResult());
        });
    Value output = linalgGenericOp.getResult(0);

    rewriter.replaceOp(op, output);

    return success();
  };
};

class LinalgLowerIntersectGenericRewrite
    : public OpRewritePattern<graphblas::IntersectGenericOp> {
public:
  using OpRewritePattern<graphblas::IntersectGenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::IntersectGenericOp op,
                                PatternRewriter &rewriter) const override {
    // ModuleOp mod = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    // Required blocks
    RegionRange extensions = op.extensions();
    ExtensionBlocks extBlocks;
    std::set<graphblas::YieldKind> required = {graphblas::YieldKind::MULT};
    LogicalResult extractResult =
        extBlocks.extractBlocks(op, extensions, required, {});

    if (extractResult.failed()) {
      return extractResult;
    }

    Value a = op.a();
    Value b = op.b();
    RankedTensorType outputTensorType =
        op.getResult().getType().dyn_cast<RankedTensorType>();

    SmallVector<AffineMap, 3> affineMaps;
    SmallVector<StringRef, 2> iteratorTypes;
    Value outputTensor = buildSameShapeOutput(op, rewriter, a, outputTensorType,
                                              affineMaps, iteratorTypes, true);

    linalg::GenericOp linalgGenericOp = rewriter.create<linalg::GenericOp>(
        loc, outputTensorType, ValueRange{a, b}, outputTensor, affineMaps,
        iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          ValueRange arguments =
              nestedBuilder.getBlock()->getArguments().take_front(2);
          sparse_tensor::LinalgIntersectOp intersectOp =
              nestedBuilder.create<sparse_tensor::LinalgIntersectOp>(
                  nestedLoc, outputTensorType.getElementType(), arguments);
          moveBlock(rewriter, extBlocks.mult, intersectOp.getRegion());
          nestedBuilder.setInsertionPointAfter(intersectOp);
          nestedBuilder.create<linalg::YieldOp>(nestedLoc,
                                                intersectOp.getResult());
        });
    Value output = linalgGenericOp.getResult(0);

    rewriter.replaceOp(op, output);

    return success();
  };
};

class LinalgLowerUnionGenericRewrite
    : public OpRewritePattern<graphblas::UnionGenericOp> {
public:
  using OpRewritePattern<graphblas::UnionGenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::UnionGenericOp op,
                                PatternRewriter &rewriter) const override {
    // ModuleOp mod = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    // Required blocks
    RegionRange extensions = op.extensions();
    ExtensionBlocks extBlocks;
    std::set<graphblas::YieldKind> required = {graphblas::YieldKind::MULT};
    LogicalResult extractResult =
        extBlocks.extractBlocks(op, extensions, required, {});

    if (extractResult.failed()) {
      return extractResult;
    }

    Value a = op.a();
    Value b = op.b();
    RankedTensorType outputTensorType =
        op.getResult().getType().dyn_cast<RankedTensorType>();

    SmallVector<AffineMap, 3> affineMaps;
    SmallVector<StringRef, 2> iteratorTypes;
    Value outputTensor = buildSameShapeOutput(op, rewriter, a, outputTensorType,
                                              affineMaps, iteratorTypes, true);

    linalg::GenericOp linalgGenericOp = rewriter.create<linalg::GenericOp>(
        loc, outputTensorType, ValueRange{a, b}, outputTensor, affineMaps,
        iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          ValueRange arguments =
              nestedBuilder.getBlock()->getArguments().take_front(2);
          sparse_tensor::LinalgUnionOp unionOp =
              nestedBuilder.create<sparse_tensor::LinalgUnionOp>(
                  nestedLoc, outputTensorType.getElementType(), arguments);
          moveBlock(rewriter, extBlocks.mult, unionOp.getRegion());
          nestedBuilder.setInsertionPointAfter(unionOp);
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, unionOp.getResult());
        });
    Value output = linalgGenericOp.getResult(0);

    rewriter.replaceOp(op, output);

    return success();
  };
};

class LinalgLowerMatrixMultiplyGenericRewrite
    : public OpRewritePattern<graphblas::MatrixMultiplyGenericOp> {
public:
  using OpRewritePattern<graphblas::MatrixMultiplyGenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::MatrixMultiplyGenericOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *context = op.getContext();
    // ModuleOp mod = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    // Required blocks
    RegionRange extensions = op.extensions();
    ExtensionBlocks extBlocks;
    std::set<graphblas::YieldKind> required = {
        graphblas::YieldKind::ADD_IDENTITY, graphblas::YieldKind::ADD,
        graphblas::YieldKind::MULT};
    // TODO: handle transform_out block
    std::set<graphblas::YieldKind> optional = {
        graphblas::YieldKind::TRANSFORM_OUT};
    LogicalResult extractResult =
        extBlocks.extractBlocks(op, extensions, required, optional);

    if (extractResult.failed()) {
      return extractResult;
    }

    Value a = op.a();
    Value b = op.b();
    Value mask = op.mask();
    // TODO: handle mask complement
    // bool isMaskComplement = op.mask_complement();
    RankedTensorType maskTensorType;

    RankedTensorType aTensorType = a.getType().dyn_cast<RankedTensorType>();
    RankedTensorType bTensorType = b.getType().dyn_cast<RankedTensorType>();
    RankedTensorType outputTensorType =
        op.getResult().getType().dyn_cast<RankedTensorType>();
    // Get the output dtype, which might be the actual result object's type for
    // vec-vec
    Type outputType;
    if (outputTensorType)
      outputType = outputTensorType.getElementType();
    else
      outputType = op.getResult().getType();

    if (mask && outputTensorType) {
      maskTensorType = mask.getType().dyn_cast<RankedTensorType>();
      if (maskTensorType != outputTensorType) {
        mask = rewriter.create<sparse_tensor::ConvertOp>(loc, outputTensorType,
                                                         mask);
        maskTensorType = mask.getType().dyn_cast<RankedTensorType>();
      }
    }

    Value outputTensor;
    SmallVector<AffineMap, 2> affineMaps;
    SmallVector<StringRef, 2> iteratorTypes;

    unsigned aRank = aTensorType.getRank();
    unsigned bRank = bTensorType.getRank();
    if (aRank == 1 && bRank == 1) {
      // Vec-Vec
      assert(!mask &&
             "mask is not allowed for vector-vector matrix multiplication");
      outputTensor =
          rewriter.create<linalg::InitTensorOp>(loc, ValueRange{}, outputType);
      outputTensorType = outputTensor.getType().cast<RankedTensorType>();
      affineMaps.push_back(
          AffineMap::get(1, 0, {getAffineDimExpr(0, context)}, context));
      affineMaps.push_back(
          AffineMap::get(1, 0, {getAffineDimExpr(0, context)}, context));
      affineMaps.push_back(AffineMap::get(1, 0, {}, context));
      iteratorTypes.push_back(getReductionIteratorTypeName());
    } else if (aRank == 2 && bRank == 1) {
      // Mat-Vec
      Value nrows = rewriter.create<graphblas::NumRowsOp>(loc, a);
      outputTensor = rewriter.create<sparse_tensor::InitOp>(
          loc, outputTensorType, ValueRange{nrows});
      affineMaps.push_back(AffineMap::get(
          2, 0, {getAffineDimExpr(0, context), getAffineDimExpr(1, context)},
          context));
      affineMaps.push_back(
          AffineMap::get(2, 0, {getAffineDimExpr(1, context)}, context));
      if (mask)
        affineMaps.push_back(
            AffineMap::get(2, 0, {getAffineDimExpr(0, context)}, context));
      affineMaps.push_back(
          AffineMap::get(2, 0, {getAffineDimExpr(0, context)}, context));
      iteratorTypes.push_back(getParallelIteratorTypeName());
      iteratorTypes.push_back(getReductionIteratorTypeName());
    } else if (aRank == 1 && bRank == 2) {
      // Vec-Mat
      Value ncols = rewriter.create<graphblas::NumColsOp>(loc, b);
      outputTensor = rewriter.create<sparse_tensor::InitOp>(
          loc, outputTensorType, ValueRange{ncols});
      affineMaps.push_back(
          AffineMap::get(2, 0, {getAffineDimExpr(0, context)}, context));
      affineMaps.push_back(AffineMap::get(
          2, 0, {getAffineDimExpr(0, context), getAffineDimExpr(1, context)},
          context));
      if (mask)
        affineMaps.push_back(
            AffineMap::get(2, 0, {getAffineDimExpr(1, context)}, context));
      affineMaps.push_back(
          AffineMap::get(2, 0, {getAffineDimExpr(1, context)}, context));
      iteratorTypes.push_back(getReductionIteratorTypeName());
      iteratorTypes.push_back(getParallelIteratorTypeName());
    } else if (aRank == 2 && bRank == 2) {
      // Mat-Mat
      if (aTensorType != outputTensorType) {
        a = rewriter.create<sparse_tensor::ConvertOp>(loc, outputTensorType, a);
        aTensorType = a.getType().dyn_cast<RankedTensorType>();
      }
      Value nrows = rewriter.create<graphblas::NumRowsOp>(loc, a);
      Value ncols = rewriter.create<graphblas::NumColsOp>(loc, b);
      outputTensor = rewriter.create<sparse_tensor::InitOp>(
          loc, outputTensorType, ValueRange{nrows, ncols});
      affineMaps.push_back(AffineMap::get(
          3, 0, {getAffineDimExpr(0, context), getAffineDimExpr(2, context)},
          context));
      affineMaps.push_back(AffineMap::get(
          3, 0, {getAffineDimExpr(2, context), getAffineDimExpr(1, context)},
          context));
      if (mask)
        affineMaps.push_back(AffineMap::get(
            3, 0, {getAffineDimExpr(0, context), getAffineDimExpr(1, context)},
            context));
      affineMaps.push_back(AffineMap::get(
          3, 0, {getAffineDimExpr(0, context), getAffineDimExpr(1, context)},
          context));
      iteratorTypes.push_back(getParallelIteratorTypeName());
      iteratorTypes.push_back(getParallelIteratorTypeName());
      iteratorTypes.push_back(getReductionIteratorTypeName());
    }

    SmallVector<Value, 3> inputs = ValueRange{a, b};
    if (mask)
      inputs.push_back(mask);

    linalg::GenericOp linalgGenericOp = rewriter.create<linalg::GenericOp>(
        loc, outputTensorType, inputs, outputTensor, affineMaps, iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          ValueRange arguments = nestedBuilder.getBlock()->getArguments();
          sparse_tensor::LinalgIntersectOp intersectOp;
          if (mask) {
            // Apply the mask by intersecting it with one of the parallel index
            // values
            Value parallelArg = (aRank == 1) ? arguments[1] : arguments[0];
            Type nonMaskType = (aRank == 1) ? bTensorType.getElementType()
                                            : aTensorType.getElementType();
            sparse_tensor::LinalgIntersectOp maskOp =
                nestedBuilder.create<sparse_tensor::LinalgIntersectOp>(
                    nestedLoc, aTensorType.getElementType(),
                    ValueRange{parallelArg, arguments[2]});
            Block &maskBlock = maskOp.getRegion().emplaceBlock();
            Value leftVal = maskBlock.addArgument(nonMaskType, nestedLoc);
            maskBlock.addArgument(maskTensorType.getElementType(), nestedLoc);
            nestedBuilder.setInsertionPointToStart(&maskBlock);
            nestedBuilder.create<sparse_tensor::LinalgYieldOp>(nestedLoc,
                                                               leftVal);
            nestedBuilder.setInsertionPointAfter(maskOp);
            if (aRank == 1)
              intersectOp =
                  nestedBuilder.create<sparse_tensor::LinalgIntersectOp>(
                      nestedLoc, outputTensorType.getElementType(),
                      ValueRange{arguments[0], maskOp.getResult()});
            else
              intersectOp =
                  nestedBuilder.create<sparse_tensor::LinalgIntersectOp>(
                      nestedLoc, outputTensorType.getElementType(),
                      ValueRange{maskOp.getResult(), arguments[1]});
          } else {
            intersectOp =
                nestedBuilder.create<sparse_tensor::LinalgIntersectOp>(
                    nestedLoc, outputTensorType.getElementType(),
                    arguments.take_front(2));
          }
          moveBlock(rewriter, extBlocks.mult, intersectOp.getRegion());
          // Create reduce operation
          nestedBuilder.setInsertionPointAfter(intersectOp);
          sparse_tensor::LinalgReduceOp reduceOp =
              nestedBuilder.create<sparse_tensor::LinalgReduceOp>(
                  nestedLoc, outputTensorType.getElementType(),
                  ValueRange{arguments.back(), intersectOp.getResult()});
          moveBlock(rewriter, extBlocks.add, reduceOp.formula());
          moveBlock(rewriter, extBlocks.addIdentity, reduceOp.init());
          // Add linalg.yield
          nestedBuilder.setInsertionPointAfter(reduceOp);
          nestedBuilder.create<linalg::YieldOp>(nestedLoc,
                                                reduceOp.getResult());
        });
    Value output = linalgGenericOp.getResult(0);

    // For vec-vec scalar result, extract result from 0-dimension tensor
    if (aRank == 1 && bRank == 1) {
      rewriter.setInsertionPointAfter(linalgGenericOp);
      output = rewriter.create<tensor::ExtractOp>(loc, output, ValueRange{});
    }

    rewriter.replaceOp(op, output);

    return success();
  };
};

class LinalgLowerSelectGenericRewrite
    : public OpRewritePattern<graphblas::SelectGenericOp> {
public:
  using OpRewritePattern<graphblas::SelectGenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::SelectGenericOp op,
                                PatternRewriter &rewriter) const override {
    // ModuleOp mod = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    // Required blocks
    RegionRange extensions = op.extensions();
    ExtensionBlocks extBlocks;
    std::set<graphblas::YieldKind> required = {
        graphblas::YieldKind::SELECT_OUT};
    LogicalResult extractResult =
        extBlocks.extractBlocks(op, extensions, required, {});

    if (extractResult.failed()) {
      return extractResult;
    }

    Value input = op.input();
    RankedTensorType outputTensorType =
        op.getResult().getType().dyn_cast<RankedTensorType>();

    SmallVector<AffineMap, 3> affineMaps;
    SmallVector<StringRef, 2> iteratorTypes;
    Value outputTensor = buildSameShapeOutput(
        op, rewriter, input, outputTensorType, affineMaps, iteratorTypes);

    linalg::GenericOp linalgGenericOp = rewriter.create<linalg::GenericOp>(
        loc, outputTensorType, input, outputTensor, affineMaps, iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          sparse_tensor::LinalgMaskOp maskOp =
              nestedBuilder.create<sparse_tensor::LinalgMaskOp>(nestedLoc);
          moveBlock(rewriter, extBlocks.selectOut, maskOp.getRegion());
          // TODO: support value select; for now, only index select is working
          Block &block = maskOp.getRegion().front();
          if (block.getArgument(0).getType() != rewriter.getIndexType())
            block.eraseArgument(0);
          // Add linalg.yield, returning the input unchanged
          nestedBuilder.setInsertionPointAfter(maskOp);
          nestedBuilder.create<linalg::YieldOp>(
              nestedLoc, nestedBuilder.getBlock()->getArguments().front());
        });
    Value output = linalgGenericOp.getResult(0);

    rewriter.replaceOp(op, output);

    return success();
  };
};

class LinalgLowerReduceToVectorGenericRewrite
    : public OpRewritePattern<graphblas::ReduceToVectorGenericOp> {
public:
  using OpRewritePattern<graphblas::ReduceToVectorGenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::ReduceToVectorGenericOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *context = op.getContext();
    // ModuleOp mod = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    // Required blocks
    RegionRange extensions = op.extensions();
    ExtensionBlocks extBlocks;
    std::set<graphblas::YieldKind> required = {
        graphblas::YieldKind::AGG_IDENTITY, graphblas::YieldKind::AGG};
    LogicalResult extractResult =
        extBlocks.extractBlocks(op, extensions, required, {});

    if (extractResult.failed()) {
      return extractResult;
    }

    Value inputTensor = op.input();
    Value mask = op.mask();
    int axis = op.axis();
    // TODO: handle mask complement
    // bool maskComplement = op.mask_complement();
    RankedTensorType inputTensorType =
        inputTensor.getType().dyn_cast<RankedTensorType>();
    RankedTensorType outputTensorType =
        op.getResult().getType().dyn_cast<RankedTensorType>();
    RankedTensorType maskTensorType;

    if (mask)
      maskTensorType = mask.getType().dyn_cast<RankedTensorType>();

    Value outputTensor;
    SmallVector<AffineMap, 2> affineMaps;
    SmallVector<StringRef, 2> iteratorTypes;

    // Compute output size (depends on axis)
    Value size;
    if (axis == 1) {
      size = rewriter.create<graphblas::NumRowsOp>(loc, inputTensor);
      iteratorTypes.push_back(getParallelIteratorTypeName());
      iteratorTypes.push_back(getReductionIteratorTypeName());
    } else {
      size = rewriter.create<graphblas::NumColsOp>(loc, inputTensor);
      iteratorTypes.push_back(getReductionIteratorTypeName());
      iteratorTypes.push_back(getParallelIteratorTypeName());
    }
    outputTensor = rewriter.create<sparse_tensor::InitOp>(loc, outputTensorType,
                                                          ValueRange{size});
    affineMaps.push_back(AffineMap::getMultiDimIdentityMap(2, context));
    if (mask)
      affineMaps.push_back(
          AffineMap::get(2, 0, {getAffineDimExpr(1 - axis, context)}, context));
    affineMaps.push_back(
        AffineMap::get(2, 0, {getAffineDimExpr(1 - axis, context)}, context));

    SmallVector<Value, 2> inputs = ValueRange{inputTensor};
    if (mask)
      inputs.push_back(mask);

    linalg::GenericOp linalgGenericOp = rewriter.create<linalg::GenericOp>(
        loc, outputTensorType, inputs, outputTensor, affineMaps, iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          ValueRange arguments = nestedBuilder.getBlock()->getArguments();
          Value reduceArg = arguments[0];
          if (mask) {
            // Apply the mask by intersecting it with the input
            sparse_tensor::LinalgIntersectOp maskOp =
                nestedBuilder.create<sparse_tensor::LinalgIntersectOp>(
                    nestedLoc, inputTensorType.getElementType(),
                    ValueRange{arguments[0], arguments[1]});
            Block &maskBlock = maskOp.getRegion().emplaceBlock();
            Value leftVal = maskBlock.addArgument(
                inputTensorType.getElementType(), nestedLoc);
            maskBlock.addArgument(maskTensorType.getElementType(), nestedLoc);
            nestedBuilder.setInsertionPointToStart(&maskBlock);
            nestedBuilder.create<sparse_tensor::LinalgYieldOp>(nestedLoc,
                                                               leftVal);
            nestedBuilder.setInsertionPointAfter(maskOp);
            reduceArg = maskOp.getResult();
          }
          // Create reduce operation
          sparse_tensor::LinalgReduceOp reduceOp =
              nestedBuilder.create<sparse_tensor::LinalgReduceOp>(
                  nestedLoc, outputTensorType.getElementType(),
                  ValueRange{arguments.back(), reduceArg});
          moveBlock(rewriter, extBlocks.agg, reduceOp.formula());
          moveBlock(rewriter, extBlocks.aggIdentity, reduceOp.init());
          // Add linalg.yield
          nestedBuilder.setInsertionPointAfter(reduceOp);
          nestedBuilder.create<linalg::YieldOp>(nestedLoc,
                                                reduceOp.getResult());
        });
    Value output = linalgGenericOp.getResult(0);

    rewriter.replaceOp(op, output);

    return success();
  };
};

class LinalgLowerReduceToScalarGenericRewrite
    : public OpRewritePattern<graphblas::ReduceToScalarGenericOp> {
public:
  using OpRewritePattern<graphblas::ReduceToScalarGenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::ReduceToScalarGenericOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *context = op.getContext();
    // ModuleOp mod = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    Value inputTensor = op.input();
    RankedTensorType inputTensorType =
        inputTensor.getType().dyn_cast<RankedTensorType>();
    Type outputType = op.getResult().getType();

    // Required blocks
    RegionRange extensions = op.extensions();
    ExtensionBlocks extBlocks;
    std::set<graphblas::YieldKind> required = {
        graphblas::YieldKind::AGG_IDENTITY, graphblas::YieldKind::AGG};
    LogicalResult extractResult =
        extBlocks.extractBlocks(op, extensions, required, {});

    if (extractResult.failed()) {
      return extractResult;
    }

    // Compute output and iteration objects
    Value outputTensor =
        rewriter.create<linalg::InitTensorOp>(loc, ValueRange{}, outputType);
    RankedTensorType outputTensorType =
        outputTensor.getType().cast<RankedTensorType>();
    SmallVector<AffineMap, 2> affineMaps;
    SmallVector<StringRef, 2> iteratorTypes;
    unsigned rank = inputTensorType.getRank();
    if (rank == 1) {
      iteratorTypes.push_back(getParallelIteratorTypeName());
      affineMaps.push_back(AffineMap::getMultiDimIdentityMap(1, context));
      affineMaps.push_back(AffineMap::get(1, 0, {}, context));
    } else {
      iteratorTypes.push_back(getParallelIteratorTypeName());
      iteratorTypes.push_back(getParallelIteratorTypeName());
      affineMaps.push_back(AffineMap::getMultiDimIdentityMap(2, context));
      affineMaps.push_back(AffineMap::get(2, 0, {}, context));
    }

    linalg::GenericOp linalgGenericOp = rewriter.create<linalg::GenericOp>(
        loc, outputTensorType, inputTensor, outputTensor, affineMaps,
        iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          ValueRange arguments = nestedBuilder.getBlock()->getArguments();
          // Create reduce operation
          sparse_tensor::LinalgReduceOp reduceOp =
              nestedBuilder.create<sparse_tensor::LinalgReduceOp>(
                  nestedLoc, outputType, arguments);
          moveBlock(rewriter, extBlocks.agg, reduceOp.formula());
          moveBlock(rewriter, extBlocks.aggIdentity, reduceOp.init());
          // Add linalg.yield
          nestedBuilder.setInsertionPointAfter(reduceOp);
          nestedBuilder.create<linalg::YieldOp>(nestedLoc,
                                                reduceOp.getResult());
        });
    Value output = linalgGenericOp.getResult(0);
    rewriter.setInsertionPointAfter(linalgGenericOp);
    output = rewriter.create<tensor::ExtractOp>(loc, output, ValueRange{});

    rewriter.replaceOp(op, output);

    return success();
  };
};

class LinalgLowerSelectMaskRewrite
    : public OpRewritePattern<graphblas::SelectMaskOp> {
public:
  using OpRewritePattern<graphblas::SelectMaskOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::SelectMaskOp op,
                                PatternRewriter &rewriter) const override {
    // ModuleOp mod = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    // Inputs
    Value input = op.input();
    Value mask = op.mask();
    // TODO: handle mask complement
    // bool maskComplement = op.mask_complement();
    RankedTensorType inputTensorType =
        input.getType().dyn_cast<RankedTensorType>();
    RankedTensorType maskTensorType =
        mask.getType().dyn_cast<RankedTensorType>();
    RankedTensorType outputTensorType =
        op.getResult().getType().dyn_cast<RankedTensorType>();

    SmallVector<AffineMap, 3> affineMaps;
    SmallVector<StringRef, 2> iteratorTypes;
    Value outputTensor = buildSameShapeOutput(
        op, rewriter, input, outputTensorType, affineMaps, iteratorTypes, true);

    linalg::GenericOp linalgGenericOp = rewriter.create<linalg::GenericOp>(
        loc, outputTensorType, ValueRange{input, mask}, outputTensor,
        affineMaps, iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          ValueRange arguments =
              nestedBuilder.getBlock()->getArguments().take_front(2);
          sparse_tensor::LinalgIntersectOp maskOp =
              nestedBuilder.create<sparse_tensor::LinalgIntersectOp>(
                  nestedLoc, outputTensorType.getElementType(), arguments);
          Block &maskBlock = maskOp.getRegion().emplaceBlock();
          Value leftVal = maskBlock.addArgument(
              inputTensorType.getElementType(), nestedLoc);
          maskBlock.addArgument(maskTensorType.getElementType(), nestedLoc);
          nestedBuilder.setInsertionPointToStart(&maskBlock);
          nestedBuilder.create<sparse_tensor::LinalgYieldOp>(nestedLoc,
                                                             leftVal);
          nestedBuilder.setInsertionPointAfter(maskOp);
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, maskOp.getResult());
        });
    Value output = linalgGenericOp.getResult(0);

    rewriter.replaceOp(op, output);

    return success();
  };
};

class LinalgLowerTransposeRewrite
    : public OpRewritePattern<graphblas::TransposeOp> {
public:
  using OpRewritePattern<graphblas::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(graphblas::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *context = op.getContext();
    // ModuleOp mod = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    Value inputTensor = op.input();
    RankedTensorType inputTensorType =
        inputTensor.getType().dyn_cast<RankedTensorType>();
    RankedTensorType outputTensorType =
        op.getResult().getType().dyn_cast<RankedTensorType>();
    bool inputByRow = hasRowOrdering(inputTensorType);
    bool outputByRow = hasRowOrdering(outputTensorType);

    // Linalg logic only works for byrow->bycol or vice versa
    if (inputByRow == outputByRow) {
      RankedTensorType flippedInputType =
          getFlippedLayoutType(context, inputTensorType);
      inputTensor = rewriter.create<sparse_tensor::ConvertOp>(
          loc, flippedInputType, inputTensor);
      inputTensorType = flippedInputType;
      inputByRow = !inputByRow;
    }

    SmallVector<AffineMap, 2> affineMaps;
    SmallVector<StringRef, 2> iteratorTypes;
    Value nrows = rewriter.create<graphblas::NumRowsOp>(loc, inputTensor);
    Value ncols = rewriter.create<graphblas::NumColsOp>(loc, inputTensor);
    Value outputTensor = rewriter.create<sparse_tensor::InitOp>(
        loc, outputTensorType, ValueRange{ncols, nrows});
    iteratorTypes.push_back(getParallelIteratorTypeName());
    iteratorTypes.push_back(getParallelIteratorTypeName());
    unsigned idx = inputByRow ? 0 : 1;
    affineMaps.push_back(AffineMap::get(
        2, 0,
        {getAffineDimExpr(idx, context), getAffineDimExpr(1 - idx, context)},
        context));
    affineMaps.push_back(AffineMap::get(
        2, 0,
        {getAffineDimExpr(1 - idx, context), getAffineDimExpr(idx, context)},
        context));

    linalg::GenericOp linalgGenericOp = rewriter.create<linalg::GenericOp>(
        loc, outputTensorType, inputTensor, outputTensor, affineMaps,
        iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          Value val = nestedBuilder.getBlock()->getArguments().front();
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, val);
        });
    Value output = linalgGenericOp.getResult(0);

    rewriter.replaceOp(op, output);

    return success();
  };
};

class LinalgLowerDiagVectorToMatrixOpRewrite
    : public OpRewritePattern<graphblas::DiagOp> {
public:
  using OpRewritePattern<graphblas::DiagOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::DiagOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *context = op.getContext();
    // ModuleOp mod = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    Value input = op.input();
    RankedTensorType outputTensorType =
        op.getResult().getType().dyn_cast<RankedTensorType>();

    if (outputTensorType.getRank() != 2)
      return failure();

    SmallVector<AffineMap, 2> affineMaps;
    SmallVector<StringRef, 2> iteratorTypes;

    Value size = rewriter.create<graphblas::SizeOp>(loc, input);
    Value outputTensor = rewriter.create<sparse_tensor::InitOp>(
        loc, outputTensorType, ValueRange{size, size});
    affineMaps.push_back(
        AffineMap::get(2, 0, {getAffineDimExpr(0, context)}, context));
    int idx = hasRowOrdering(outputTensorType) ? 0 : 1;
    affineMaps.push_back(AffineMap::get(
        2, 0,
        {getAffineDimExpr(idx, context), getAffineDimExpr(1 - idx, context)},
        context));
    iteratorTypes.push_back(getParallelIteratorTypeName());
    iteratorTypes.push_back(getParallelIteratorTypeName());

    linalg::GenericOp linalgGenericOp = rewriter.create<linalg::GenericOp>(
        loc, outputTensorType, input, outputTensor, affineMaps, iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          // Only insert the value if row == col
          sparse_tensor::LinalgMaskOp maskOp =
              nestedBuilder.create<sparse_tensor::LinalgMaskOp>(nestedLoc);
          Block &maskBlock = maskOp.getRegion().emplaceBlock();
          Value row =
              maskBlock.addArgument(nestedBuilder.getIndexType(), nestedLoc);
          Value col =
              maskBlock.addArgument(nestedBuilder.getIndexType(), nestedLoc);
          nestedBuilder.setInsertionPointToStart(&maskBlock);
          Value cmp = nestedBuilder.create<arith::CmpIOp>(
              nestedLoc, arith::CmpIPredicate::eq, row, col);
          nestedBuilder.create<sparse_tensor::LinalgYieldOp>(nestedLoc, cmp);
          nestedBuilder.setInsertionPointAfter(maskOp);
          // Add linalg.yield, returning the input unchanged
          nestedBuilder.setInsertionPointAfter(maskOp);
          nestedBuilder.create<linalg::YieldOp>(
              nestedLoc, nestedBuilder.getBlock()->getArguments().front());
        });
    Value output = linalgGenericOp.getResult(0);

    rewriter.replaceOp(op, output);

    return success();
  };
};

void populateGraphBLASLinalgLoweringPatterns(RewritePatternSet &patterns) {
  patterns.add<
      LinalgLowerApplyGenericRewrite, LinalgLowerIntersectGenericRewrite,
      LinalgLowerUnionGenericRewrite, LinalgLowerMatrixMultiplyGenericRewrite,
      LinalgLowerSelectGenericRewrite, LinalgLowerReduceToVectorGenericRewrite,
      LinalgLowerReduceToScalarGenericRewrite, LinalgLowerSelectMaskRewrite,
      LinalgLowerTransposeRewrite, LinalgLowerDiagVectorToMatrixOpRewrite,
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
