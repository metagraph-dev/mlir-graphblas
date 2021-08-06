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

class LowerSizeRewrite : public OpRewritePattern<graphblas::SizeOp> {
public:
  using OpRewritePattern<graphblas::SizeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::SizeOp op, PatternRewriter &rewriter) const {
    Location loc = op->getLoc();

    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value inputTensor = op.input();
    Value size = rewriter.create<memref::DimOp>(loc, inputTensor, c0);

    rewriter.replaceOp(op, size);
    return success();
  };
};

class LowerNumRowsRewrite : public OpRewritePattern<graphblas::NumRowsOp> {
public:
  using OpRewritePattern<graphblas::NumRowsOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::NumRowsOp op, PatternRewriter &rewriter) const {
    Location loc = op->getLoc();

    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value inputTensor = op.input();
    Value nrows = rewriter.create<memref::DimOp>(loc, inputTensor, c0);

    rewriter.replaceOp(op, nrows);
    return success();
  };
};

class LowerNumColsRewrite : public OpRewritePattern<graphblas::NumColsOp> {
public:
  using OpRewritePattern<graphblas::NumColsOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::NumColsOp op, PatternRewriter &rewriter) const {
    Location loc = op->getLoc();

    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
    Value inputTensor = op.input();
    Value ncols = rewriter.create<memref::DimOp>(loc, inputTensor, c1);

    rewriter.replaceOp(op, ncols);
    return success();
  };
};

class LowerNumValsRewrite : public OpRewritePattern<graphblas::NumValsOp> {
public:
  using OpRewritePattern<graphblas::NumValsOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::NumValsOp op, PatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    Value inputTensor = op.input();
    Type inputType = inputTensor.getType();

    sparse_tensor::SparseTensorEncodingAttr sparseEncoding = sparse_tensor::getSparseTensorEncoding(inputType);
    unsigned pointerBitWidth = sparseEncoding.getPointerBitWidth();
    Type pointerType = rewriter.getIntegerType(pointerBitWidth);
    Type indexType = rewriter.getIndexType();

    // Access the pointers
    Type memref1DPointerType = MemRefType::get({-1}, pointerType);
    unsigned rank = inputType.dyn_cast<RankedTensorType>().getRank();
    Value c_rank_minus_1 = rewriter.create<ConstantIndexOp>(loc, rank - 1);
    Value ptrs = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DPointerType, inputTensor, c_rank_minus_1);

    // Find length of pointer array
    Value dimForPointers;
    if (rank == 1 || typeIsCSR(inputType)) {
      dimForPointers = rewriter.create<ConstantIndexOp>(loc, 0);
    } else {
      dimForPointers = rewriter.create<ConstantIndexOp>(loc, 1);
    }
    Value npointers = rewriter.create<memref::DimOp>(loc, inputTensor, dimForPointers);

    // The last value from the pointers is the number of nonzero values
    Value nnz_ptype = rewriter.create<memref::LoadOp>(loc, ptrs, npointers);
    Value nnz = rewriter.create<mlir::IndexCastOp>(loc, nnz_ptype, indexType);

    rewriter.replaceOp(op, nnz);
    return success();
  };
};

class LowerDupRewrite : public OpRewritePattern<graphblas::DupOp> {
public:
  using OpRewritePattern<graphblas::DupOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::DupOp op, PatternRewriter &rewriter) const {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();
    Value inputTensor = op.input();
    Type inputType = inputTensor.getType();
    unsigned rank = inputType.dyn_cast<RankedTensorType>().getRank();

    Value duplicate = callDupTensor(rewriter, module, loc, inputTensor);
    if (rank == 2) {
      if (typeIsCSC(inputType)) {
        duplicate = convertToExternalCSC(rewriter, module, loc, duplicate);
      } else {
        duplicate = convertToExternalCSR(rewriter, module, loc, duplicate);
      }
    }

    rewriter.replaceOp(op, duplicate);
    return success();
  };
};

class LowerConvertLayoutRewrite : public OpRewritePattern<graphblas::ConvertLayoutOp> {
public:
  using OpRewritePattern<graphblas::ConvertLayoutOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::ConvertLayoutOp op, PatternRewriter &rewriter) const {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    Value inputTensor = op.input();
    Type inputType = inputTensor.getType();
    Type outputType = op->getResultTypes()[0];

    // Shortcut operation if no change
    if (inputType == outputType)
    {
      rewriter.replaceOp(op, inputTensor);
      return success();
    }

    // otherwise, the rest of this function changes the data layout
    Type valueType = inputType.dyn_cast<RankedTensorType>().getElementType();
    Type int64Type = rewriter.getIntegerType(64);
    Type indexType = rewriter.getIndexType();

    // Initial constants
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
    Value c0_64 = rewriter.create<ConstantIntOp>(loc, 0, int64Type);
    Value c1_64 = rewriter.create<ConstantIntOp>(loc, 1, int64Type);

    // Get sparse tensor info
    Type memref1DI64Type = MemRefType::get({-1}, int64Type);
    Type memref1DValueType = MemRefType::get({-1}, valueType);

    Value inputPtrs = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, inputTensor, c1);
    Value inputIndices = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, inputTensor, c1);
    Value inputValues = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, inputTensor);
    Value nrow = rewriter.create<graphblas::NumRowsOp>(loc, inputTensor);
    Value ncol = rewriter.create<graphblas::NumColsOp>(loc, inputTensor);
    Value ncols_plus_one = rewriter.create<mlir::AddIOp>(loc, ncol, c1);
    Value nnz = rewriter.create<graphblas::NumValsOp>(loc, inputTensor);

    Value duplicate = callEmptyLike(rewriter, module, loc, inputTensor);
    callResizeDim(rewriter, module, loc, duplicate, c0, nrow);
    callResizeDim(rewriter, module, loc, duplicate, c1, ncol);

    callResizePointers(rewriter, module, loc, duplicate, c1, ncols_plus_one);
    callResizeIndex(rewriter, module, loc, duplicate, c1, nnz);
    callResizeValues(rewriter, module, loc, duplicate, nnz);
    
    // verify function will ensure that this is CSR->CSC or CSC->CSR
    Value output;
    if (typeIsCSR(outputType)) {
      output = convertToExternalCSR(rewriter, module, loc, duplicate);
    } else if (typeIsCSC(outputType)) {
      output = convertToExternalCSC(rewriter, module, loc, duplicate); 
    } else {
      assert(false && "Output type must be CSC or CSR.");
    }

    Value outputPtrs = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, output, c1);
    Value outputIndices = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, output, c1);
    Value outputValues = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, output);

    // compute number of non-zero entries per column of A

    // init B.pointers to zero
    scf::ForOp initLoop = rewriter.create<scf::ForOp>(loc, c0, ncol, c1);
    Value initLoopIdx = initLoop.getInductionVar();
    rewriter.setInsertionPointToStart(initLoop.getBody());
    rewriter.create<memref::StoreOp>(loc, c0_64, outputPtrs, initLoopIdx);
    rewriter.setInsertionPointAfter(initLoop);

    // store pointers
    scf::ForOp ptrLoop = rewriter.create<scf::ForOp>(loc, c0, nnz, c1);
    Value ptrLoopIdx = ptrLoop.getInductionVar();

    rewriter.setInsertionPointToStart(ptrLoop.getBody());
    Value colA64 = rewriter.create<memref::LoadOp>(loc, inputIndices, ptrLoopIdx);
    Value colA = rewriter.create<mlir::IndexCastOp>(loc, colA64, indexType);
    Value colB = rewriter.create<memref::LoadOp>(loc, outputPtrs, colA);
    Value colB1 = rewriter.create<mlir::AddIOp>(loc, colB, c1_64);
    rewriter.create<memref::StoreOp>(loc, colB1, outputPtrs, colA);

    rewriter.setInsertionPointAfter(ptrLoop);

    // cumsum the nnz per column to get Bp
    rewriter.create<memref::StoreOp>(loc, c0_64, outputPtrs, ncol);

    scf::ForOp colAccLoop = rewriter.create<scf::ForOp>(loc, c0, ncol, c1);
    Value colAccLoopIdx = colAccLoop.getInductionVar();

    rewriter.setInsertionPointToStart(colAccLoop.getBody());
    Value temp = rewriter.create<memref::LoadOp>(loc, outputPtrs, colAccLoopIdx);
    Value cumsum = rewriter.create<memref::LoadOp>(loc, outputPtrs, ncol);
    rewriter.create<memref::StoreOp>(loc, cumsum, outputPtrs, colAccLoopIdx);
    Value cumsum2 = rewriter.create<mlir::AddIOp>(loc, cumsum, temp);
    rewriter.create<memref::StoreOp>(loc, cumsum2, outputPtrs, ncol);

    rewriter.setInsertionPointAfter(colAccLoop);

    // copy values
    scf::ForOp outerLoop = rewriter.create<scf::ForOp>(loc, c0, nrow, c1);
    Value rowIdx = outerLoop.getInductionVar();

    rewriter.setInsertionPointToStart(outerLoop.getBody());
    Value row_64 = rewriter.create<mlir::IndexCastOp>(loc, rowIdx, int64Type);
    Value j_start_64 = rewriter.create<memref::LoadOp>(loc, inputPtrs, rowIdx);
    Value j_start = rewriter.create<mlir::IndexCastOp>(loc, j_start_64, indexType);
    Value row_plus1 = rewriter.create<mlir::AddIOp>(loc, rowIdx, c1);
    Value j_end_64 = rewriter.create<memref::LoadOp>(loc, inputPtrs, row_plus1);
    Value j_end = rewriter.create<mlir::IndexCastOp>(loc, j_end_64, indexType);

    scf::ForOp innerLoop = rewriter.create<scf::ForOp>(loc, j_start, j_end, c1);
    Value jj = innerLoop.getInductionVar();

    rewriter.setInsertionPointToStart(innerLoop.getBody());

    Value col_64 = rewriter.create<memref::LoadOp>(loc, inputIndices, jj);
    Value col = rewriter.create<mlir::IndexCastOp>(loc, col_64, indexType);
    Value dest_64 = rewriter.create<memref::LoadOp>(loc, outputPtrs, col);
    Value dest = rewriter.create<mlir::IndexCastOp>(loc, dest_64, indexType);
    rewriter.create<memref::StoreOp>(loc, row_64, outputIndices, dest);
    Value axjj = rewriter.create<memref::LoadOp>(loc, inputValues, jj);
    rewriter.create<memref::StoreOp>(loc, axjj, outputValues, dest);

    // Bp[col]++
    Value bp_inc = rewriter.create<memref::LoadOp>(loc, outputPtrs, col);
    Value bp_inc1 = rewriter.create<mlir::AddIOp>(loc, bp_inc, c1_64);
    rewriter.create<memref::StoreOp>(loc, bp_inc1, outputPtrs, col);

    rewriter.setInsertionPointAfter(outerLoop);

    Value last_last = rewriter.create<memref::LoadOp>(loc, outputPtrs, ncol);
    rewriter.create<memref::StoreOp>(loc, c0_64, outputPtrs, ncol);

    scf::ForOp finalLoop = rewriter.create<scf::ForOp>(loc, c0, ncol, c1);
    Value iCol = finalLoop.getInductionVar();

    rewriter.setInsertionPointToStart(finalLoop.getBody());

    Value swapTemp = rewriter.create<memref::LoadOp>(loc, outputPtrs, iCol);
    Value last = rewriter.create<memref::LoadOp>(loc, outputPtrs, ncol);
    rewriter.create<memref::StoreOp>(loc, last, outputPtrs, iCol);
    rewriter.create<memref::StoreOp>(loc, swapTemp, outputPtrs, ncol);

    rewriter.setInsertionPointAfter(finalLoop);

    rewriter.create<memref::StoreOp>(loc, last_last, outputPtrs, ncol);

    rewriter.replaceOp(op, output);

    cleanupIntermediateTensor(rewriter, module, loc, output);

    return success();
  };
};

struct MatrixSelectOutputWriter {
  MatrixSelectOutputWriter(StringRef _selector) : selector(_selector)
  {
  };

  void createConstants(PatternRewriter &rewriter, Location loc) {
    Type int64Type = rewriter.getIntegerType(64);
    FloatType float64Type = rewriter.getF64Type();

    c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    c1 = rewriter.create<ConstantIndexOp>(loc, 1);
    c0_64 = rewriter.create<ConstantIntOp>(loc, 0, int64Type);
    c1_64 = rewriter.create<ConstantIntOp>(loc, 1, int64Type);
    cf0 = rewriter.create<ConstantFloatOp>(loc, APFloat(0.0), float64Type);
  }

  void createTensor(PatternRewriter &rewriter, Location loc, ModuleOp module, Value input)
  {
    Type valueType = input.getType().dyn_cast<TensorType>().getElementType();
    Type int64Type = rewriter.getIntegerType(64);

    Type memref1DI64Type = MemRefType::get({-1}, int64Type);
    Type memref1DValueType = MemRefType::get({-1}, valueType);

    tensor = rewriter.create<graphblas::DupOp>(loc, input);
    Bp = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, tensor, c1);
    Bj = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, tensor, c1);
    Bx = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, tensor);
    rewriter.create<memref::StoreOp>(loc, c0_64, Bp, c0);
  };

  void createUpdateCurrCount(PatternRewriter &rewriter, Location loc, Value row, Value row_plus1)
  {
    Value bp_curr_count = rewriter.create<memref::LoadOp>(loc, Bp, row);
    rewriter.create<memref::StoreOp>(loc, bp_curr_count, Bp, row_plus1);
  };

  void createTestAndStore(PatternRewriter &rewriter, Location loc,
             Value row,
             Value col, Value val, Value row_plus1, Value col_64)
  {
    Type indexType = rewriter.getIndexType();

    Value keep;
    if (selector == "triu")
    {
      keep = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::ugt, col, row);
    }
    else if (selector == "tril")
    {
      keep = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::ult, col, row);
    }
    else if (selector == "gt0")
    {
      keep = rewriter.create<mlir::CmpFOp>(loc, mlir::CmpFPredicate::OGT, val, cf0);
    }
    else
    {
      // this should be impossible becasue of validation
      assert(0);
    }

    scf::IfOp ifKeep = rewriter.create<scf::IfOp>(loc, keep, false /* no else region */);

    rewriter.setInsertionPointToStart(ifKeep.thenBlock());

    Value bj_pos_64 = rewriter.create<memref::LoadOp>(loc, Bp, row_plus1);
    Value bj_pos = rewriter.create<mlir::IndexCastOp>(loc, bj_pos_64, indexType);

    rewriter.create<memref::StoreOp>(loc, col_64, Bj, bj_pos);
    rewriter.create<memref::StoreOp>(loc, val, Bx, bj_pos);

    Value bj_pos_plus1 = rewriter.create<mlir::AddIOp>(loc, bj_pos_64, c1_64);
    rewriter.create<memref::StoreOp>(loc, bj_pos_plus1, Bp, row_plus1);

    rewriter.setInsertionPointAfter(ifKeep);
  };

  void createTrimValues(PatternRewriter &rewriter, Location loc, ModuleOp module)
  {
    Value nnz = rewriter.create<graphblas::NumValsOp>(loc, tensor);

    callResizeIndex(rewriter, module, loc, tensor, c1, nnz);
    callResizeValues(rewriter, module, loc, tensor, nnz);

    assert(typeIsCSR(tensor.getType()) &&
           "tensor expected to be CSR since createTensor is expected to have been called first.");
  };

  StringRef selector;

  // frequently used values
  Value tensor;
  Value Bp;
  Value Bj;
  Value Bx;

  // frequently used constants
  Value c0;
  Value c1;
  Value cf0;
  Value c0_64;
  Value c1_64;
};

class LowerMatrixSelectRewrite : public OpRewritePattern<graphblas::MatrixSelectOp> {
public:
  using OpRewritePattern<graphblas::MatrixSelectOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::MatrixSelectOp op, PatternRewriter &rewriter) const {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    Value input = op.input();
    Type valueType = input.getType().dyn_cast<TensorType>().getElementType();
    Type int64Type = rewriter.getIntegerType(64);
    Type indexType = rewriter.getIndexType();
    Type memref1DI64Type = MemRefType::get({-1}, int64Type);
    Type memref1DValueType = MemRefType::get({-1}, valueType);

    ArrayAttr selectors = op.selectors();

    // Initial constants
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);

    // Get sparse tensor info
    Value nrow = rewriter.create<graphblas::NumRowsOp>(loc, input);
    Value Ap = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, input, c1);
    Value Aj = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, input, c1);
    Value Ax = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, input);

    SmallVector<MatrixSelectOutputWriter*, 3> outputs;
    for (Attribute selectorAttr : selectors)
    {
      StringRef selector = selectorAttr.dyn_cast_or_null<StringAttr>().getValue();
      MatrixSelectOutputWriter *output = new MatrixSelectOutputWriter(selector);
      outputs.push_back(output);

      output->createConstants(rewriter, loc);
      output->createTensor(rewriter, loc, module, input);
    }

    // Loop
    scf::ForOp outerLoop = rewriter.create<scf::ForOp>(loc, c0, nrow, c1);
    Value row = outerLoop.getInductionVar();

    rewriter.setInsertionPointToStart(outerLoop.getBody());
    Value row_plus1 = rewriter.create<mlir::AddIOp>(loc, row, c1);

    for (MatrixSelectOutputWriter* output : outputs)
    {
      output->createUpdateCurrCount(rewriter, loc, row, row_plus1);
    }

    Value j_start_64 = rewriter.create<memref::LoadOp>(loc, Ap, row);
    Value j_end_64 = rewriter.create<memref::LoadOp>(loc, Ap, row_plus1);
    Value j_start = rewriter.create<mlir::IndexCastOp>(loc, j_start_64, indexType);
    Value j_end = rewriter.create<mlir::IndexCastOp>(loc, j_end_64, indexType);

    scf::ForOp innerLoop = rewriter.create<scf::ForOp>(loc, j_start, j_end, c1);

    Value jj = innerLoop.getInductionVar();

    rewriter.setInsertionPointToStart(innerLoop.getBody());
    Value col_64 = rewriter.create<memref::LoadOp>(loc, Aj, jj);
    Value col = rewriter.create<mlir::IndexCastOp>(loc, col_64, indexType);
    Value val = rewriter.create<memref::LoadOp>(loc, Ax, jj);

    for (MatrixSelectOutputWriter* output : outputs)
    {
      output->createTestAndStore(rewriter, loc, row, col, val, row_plus1, col_64);
    }

    rewriter.setInsertionPointAfter(outerLoop);

    // trim excess values
    SmallVector<Value, 3> outputTensors;
    for (MatrixSelectOutputWriter* output : outputs) {
      output->createTrimValues(rewriter, loc, module);
      outputTensors.push_back(output->tensor);
    }
    rewriter.replaceOp(op, outputTensors);

    for (Value output : outputTensors) {
      cleanupIntermediateTensor(rewriter, module, loc, output);
    }

    return success();
  };
};

class LowerMatrixReduceToScalarRewrite : public OpRewritePattern<graphblas::MatrixReduceToScalarOp>
{
public:
  using OpRewritePattern<graphblas::MatrixReduceToScalarOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::MatrixReduceToScalarOp op, PatternRewriter &rewriter) const
  {
    Value input = op.input();
    StringRef aggregator = op.aggregator();
    Location loc = rewriter.getUnknownLoc();

    RankedTensorType operandType = op.input().getType().dyn_cast<RankedTensorType>();
    Type valueType = operandType.getElementType();

    // New op
    graphblas::MatrixReduceToScalarGenericOp newReduceOp = rewriter.create<graphblas::MatrixReduceToScalarGenericOp>(
        loc, op->getResultTypes(), input, 2);

    if (aggregator == "sum")
    {
      // Insert agg identity block
      Region &aggIdentityRegion = newReduceOp.getRegion(0);
      /*Block *aggIdentityBlock = */ rewriter.createBlock(&aggIdentityRegion, {}, {});

      Value aggIdentity = llvm::TypeSwitch<Type, Value>(valueType)
                              .Case<IntegerType>([&](IntegerType type)
                                                 { return rewriter.create<ConstantIntOp>(loc, 0, type.getWidth()); })
                              .Case<FloatType>([&](FloatType type)
                                               { return rewriter.create<ConstantFloatOp>(loc, APFloat(0.0), type); });
      rewriter.create<graphblas::YieldOp>(loc, graphblas::YieldKind::AGG_IDENTITY, aggIdentity);

      // Insert agg block
      Region &aggRegion = newReduceOp.getRegion(1);
      Block *aggBlock = rewriter.createBlock(&aggRegion, {}, {valueType, valueType});
      Value lhs = aggBlock->getArgument(0);
      Value rhs = aggBlock->getArgument(1);

      Value aggResult = llvm::TypeSwitch<Type, Value>(valueType)
                        .Case<IntegerType>([&](IntegerType type)
                                           { return rewriter.create<AddIOp>(loc, lhs, rhs).getResult(); })
                        .Case<FloatType>([&](FloatType type)
                                         { return rewriter.create<AddFOp>(loc, lhs, rhs).getResult(); });
      rewriter.create<graphblas::YieldOp>(loc, graphblas::YieldKind::AGG, aggResult);
    } else {
      return op.emitError("\"" + aggregator + "\" is not a supported aggregator.");
    }

    rewriter.replaceOp(op, newReduceOp.getResult());

    return success();
  };
};

class LowerMatrixReduceToScalarGenericRewrite : public OpRewritePattern<graphblas::MatrixReduceToScalarGenericOp> {
public:
  using OpRewritePattern<graphblas::MatrixReduceToScalarGenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::MatrixReduceToScalarGenericOp op, PatternRewriter &rewriter) const {
    Value input = op.input();
    Location loc = rewriter.getUnknownLoc();

    RankedTensorType operandType = op.input().getType().dyn_cast<RankedTensorType>();
    Type valueType = operandType.getElementType();
    Type int64Type = rewriter.getIntegerType(64); // TODO should we get this from the sparse encoding?
    Type indexType = rewriter.getIndexType();

    // Required blocks
    RegionRange extensions = op.extensions();
    ExtensionBlocks extBlocks;
    std::set<graphblas::YieldKind> required = {graphblas::YieldKind::AGG_IDENTITY, graphblas::YieldKind::AGG};
    LogicalResult extractResult = extBlocks.extractBlocks(op, extensions, required, {});

    if (extractResult.failed())
    {
      return extractResult;
    }

    // insert agg identity
    graphblas::YieldOp aggIdentityYield = llvm::dyn_cast_or_null<graphblas::YieldOp>(extBlocks.aggIdentity->getTerminator());
    rewriter.mergeBlocks(extBlocks.aggIdentity, rewriter.getBlock(), {});
    Value c0Accumulator = aggIdentityYield.values().front();
    rewriter.eraseOp(aggIdentityYield);

    // initial constants
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);

    // Get sparse tensor info
    MemRefType memref1DI64Type = MemRefType::get({-1}, int64Type);
    MemRefType memref1DValueType = MemRefType::get({-1}, valueType);

    Value nrows = rewriter.create<graphblas::NumRowsOp>(loc, input);
    sparse_tensor::ToPointersOp inputPtrs = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, input, c1);
    sparse_tensor::ToValuesOp inputValues = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, input);
    memref::LoadOp nnz64 = rewriter.create<memref::LoadOp>(loc, inputPtrs, nrows);
    IndexCastOp nnz = rewriter.create<IndexCastOp>(loc, nnz64, indexType);

    // begin loop
    scf::ParallelOp valueLoop = rewriter.create<scf::ParallelOp>(loc, c0, nnz.getResult(), c1, c0Accumulator);
    ValueRange valueLoopIdx = valueLoop.getInductionVars();

    rewriter.setInsertionPointToStart(valueLoop.getBody());
    memref::LoadOp y = rewriter.create<memref::LoadOp>(loc, inputValues, valueLoopIdx);

    scf::ReduceOp reducer = rewriter.create<scf::ReduceOp>(loc, y);
    BlockArgument lhs = reducer.getRegion().getArgument(0);
    BlockArgument rhs = reducer.getRegion().getArgument(1);

    rewriter.setInsertionPointToStart(&reducer.getRegion().front());

    graphblas::YieldOp aggYield = llvm::dyn_cast_or_null<graphblas::YieldOp>(extBlocks.agg->getTerminator());
    rewriter.mergeBlocks(extBlocks.agg, rewriter.getBlock(), {lhs, rhs});
    Value result = aggYield.values().front();
    rewriter.eraseOp(aggYield);

    rewriter.create<scf::ReduceReturnOp>(loc, result);

    rewriter.setInsertionPointAfter(reducer);

    rewriter.replaceOp(op, valueLoop.getResult(0));

    return success();
  };
};

class LowerMatrixApplyRewrite : public OpRewritePattern<graphblas::MatrixApplyOp> {
public:
  using OpRewritePattern<graphblas::MatrixApplyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::MatrixApplyOp op, PatternRewriter &rewriter) const {
    ModuleOp module = op->getParentOfType<ModuleOp>(); /* ignore unused variable for debugging */ (void)module;
    Location loc = op->getLoc();

    Type valueType = op.input().getType().dyn_cast<RankedTensorType>().getElementType();

    Value input = op.input();
    Value thunk = op.thunk();
    StringRef apply_operator = op.apply_operator();

    // New op
    graphblas::MatrixApplyGenericOp newApplyOp = rewriter.create<graphblas::MatrixApplyGenericOp>(
        loc, op->getResultTypes(), input, 1);

    // Insert transformOut block
    Region &transformOutRegion = newApplyOp.getRegion(0);
    Block *transformOutBlock = rewriter.createBlock(&transformOutRegion, {}, {valueType});

    Value transformResult;
    if (apply_operator == "min")
    {
      Value val = transformOutBlock->getArgument(0);
      Value cmp = rewriter.create<mlir::CmpFOp>(loc, mlir::CmpFPredicate::OLT, val, thunk);
      transformResult = rewriter.create<mlir::SelectOp>(loc, cmp, val, thunk);
    } else {
      return op.emitError("\"" + apply_operator + "\" is not a supported apply_operator.");
    };

    rewriter.create<graphblas::YieldOp>(loc, graphblas::YieldKind::TRANSFORM_OUT, transformResult);

    rewriter.replaceOp(op, newApplyOp.getResult());

    return success();
  };
};

class LowerMatrixApplyGenericRewrite : public OpRewritePattern<graphblas::MatrixApplyGenericOp>
{
public:
  using OpRewritePattern<graphblas::MatrixApplyGenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::MatrixApplyGenericOp op, PatternRewriter &rewriter) const
  {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    Type valueType = op.input().getType().dyn_cast<RankedTensorType>().getElementType();
    Type memref1DValueType = MemRefType::get({-1}, valueType);

    Value inputTensor = op.input();

    // Required blocks
    RegionRange extensions = op.extensions();
    ExtensionBlocks extBlocks;
    std::set<graphblas::YieldKind> required = {graphblas::YieldKind::TRANSFORM_OUT};
    LogicalResult extractResult = extBlocks.extractBlocks(op, extensions, required, {});

    if (extractResult.failed())
    {
      return extractResult;
    }

    // Initial constants
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);

    // Get sparse tensor info
    Value output = rewriter.create<graphblas::DupOp>(loc, inputTensor);
    Value inputValues = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, inputTensor);
    Value outputValues = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, output);

    Value nnz = rewriter.create<graphblas::NumValsOp>(loc, inputTensor);

    // Loop over values
    scf::ParallelOp valueLoop = rewriter.create<scf::ParallelOp>(loc, c0, nnz, c1);
    ValueRange valueLoopIdx = valueLoop.getInductionVars();

    rewriter.setInsertionPointToStart(valueLoop.getBody());
    Value val = rewriter.create<memref::LoadOp>(loc, inputValues, valueLoopIdx);

    // scf::ParallelOp automatically gets an empty scf.yield at the end which we need to insert before
    Operation *scfYield = valueLoop.getBody()->getTerminator();

    // insert transformOut block
    graphblas::YieldOp transformOutYield = llvm::dyn_cast_or_null<graphblas::YieldOp>(extBlocks.transformOut->getTerminator());

    rewriter.mergeBlockBefore(extBlocks.transformOut, scfYield, {val});
    Value result = transformOutYield.values().front();
    rewriter.eraseOp(transformOutYield);
    
    rewriter.create<memref::StoreOp>(loc, result, outputValues, valueLoopIdx);

    // end value loop
    rewriter.setInsertionPointAfter(valueLoop);

    // Add return op
    rewriter.replaceOp(op, output);

    cleanupIntermediateTensor(rewriter, module, loc, output);

    return success();
  };
};

class LowerMatrixMultiplyRewrite : public OpRewritePattern<graphblas::MatrixMultiplyOp> {
public:
  using OpRewritePattern<graphblas::MatrixMultiplyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::MatrixMultiplyOp op, PatternRewriter &rewriter) const {
    ModuleOp module = op->getParentOfType<ModuleOp>(); /* ignore unused variable for debugging */ (void)module;
    Location loc = rewriter.getUnknownLoc();

    // Inputs
    ValueRange operands = op.getOperands();
    StringRef semiring = op.semiring();

    // Types
    // Can't use result here because it might be a scalar (vector-vector)
    Type valueType = op.a().getType().dyn_cast<RankedTensorType>().getElementType();

    // New op
    ArrayRef<NamedAttribute> attributes;
    graphblas::MatrixMultiplyGenericOp newMultOp = rewriter.create<graphblas::MatrixMultiplyGenericOp>(
      loc, op->getResultTypes(), operands, attributes, 3);

    // Insert additive identity
    Region &addIdentityRegion = newMultOp.getRegion(0);
    /*Block *addIdentityBlock = */ rewriter.createBlock(&addIdentityRegion, {}, {});
    if (semiring == "plus_pair" || semiring == "plus_times" || semiring == "plus_plus") {
      // Add identity
      Value addIdentity = llvm::TypeSwitch<Type, Value>(valueType)
                              .Case<IntegerType>([&](IntegerType type)
                                                 { return rewriter.create<ConstantIntOp>(loc, 0, type.getWidth()); })
                              .Case<FloatType>([&](FloatType type)
                                               { return rewriter.create<ConstantFloatOp>(loc, APFloat(0.0), type); });
      rewriter.create<graphblas::YieldOp>(loc, graphblas::YieldKind::ADD_IDENTITY, addIdentity);
    } else {
      return op.emitError("\"" + semiring + "\" is not a supported semiring.");
    }

    // Insert additive operation
    Region &addRegion = newMultOp.getRegion(1);
    Block *addBlock = rewriter.createBlock(&addRegion, {}, {valueType, valueType});
    if (semiring == "plus_pair" || semiring == "plus_times" || semiring == "plus_plus")
    {
      // Insert add operation
      Value addBlockArg0 = addBlock->getArgument(0);
      Value addBlockArg1 = addBlock->getArgument(1);
      Value addResult = llvm::TypeSwitch<Type, Value>(valueType)
                            .Case<IntegerType>([&](IntegerType type)
                                               { return rewriter.create<AddIOp>(loc, addBlockArg0, addBlockArg1); })
                            .Case<FloatType>([&](FloatType type)
                                             { return rewriter.create<AddFOp>(loc, addBlockArg0, addBlockArg1); });

      rewriter.create<graphblas::YieldOp>(loc, graphblas::YieldKind::ADD, addResult);
    }
    else
    {
      return op.emitError("\"" + semiring + "\" is not a supported semiring.");
    }

    // Insert multiplicative operation
    Region &multRegion = newMultOp.getRegion(2);
    Block *multBlock = rewriter.createBlock(&multRegion, {}, {valueType, valueType});
    Value multBlockArg0 = multBlock->getArgument(0);
    Value multBlockArg1 = multBlock->getArgument(1);
    Value multResult;

    if (semiring == "plus_pair") {
      multResult = llvm::TypeSwitch<Type, Value>(valueType)
                       .Case<IntegerType>([&](IntegerType type)
                                          { return rewriter.create<ConstantIntOp>(loc, 1, type.getWidth()); })
                       .Case<FloatType>([&](FloatType type)
                                        { return rewriter.create<ConstantFloatOp>(loc, APFloat(1.0), type); });
    } else if (semiring == "plus_times") {
      multResult = llvm::TypeSwitch<Type, Value>(valueType)
                       .Case<IntegerType>([&](IntegerType type)
                                          { return rewriter.create<MulIOp>(loc, multBlockArg0, multBlockArg1); })
                       .Case<FloatType>([&](FloatType type)
                                        { return rewriter.create<MulFOp>(loc, multBlockArg0, multBlockArg1); });
    } else if (semiring == "plus_plus") {
      multResult = llvm::TypeSwitch<Type, Value>(valueType)
                       .Case<IntegerType>([&](IntegerType type)
                                          { return rewriter.create<AddIOp>(loc, multBlockArg0, multBlockArg1); })
                       .Case<FloatType>([&](FloatType type)
                                        { return rewriter.create<AddFOp>(loc, multBlockArg0, multBlockArg1); });
    } else {
      return op.emitError("\"" + semiring + "\" is not a supported semiring.");
    }
    rewriter.create<graphblas::YieldOp>(loc, graphblas::YieldKind::MULT, multResult);

    rewriter.setInsertionPointAfter(newMultOp);

    rewriter.replaceOp(op, newMultOp.getResult());

    return success();
  };
};


Value computeNumOverlaps(PatternRewriter &rewriter, Value nk,
                         Value fixedIndices, Value fixedIndexStart, Value fixedIndexEnd,
                         Value iterPointers, Value iterIndices,
                         // If no mask is used, set maskIndices to nullptr, and provide maskStart=c0 and maskEnd=len(iterPointers)-1
                         Value maskIndices, Value maskStart, Value maskEnd,
                         Type valueType
                         ) {
  Location loc = rewriter.getUnknownLoc();

  // Types used in this function
  Type indexType = rewriter.getIndexType();
  Type int64Type = rewriter.getIntegerType(64);
  Type boolType = rewriter.getI1Type();
  MemRefType memref1DBoolType = MemRefType::get({-1}, boolType);

  // Initial constants
  Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
  Value ci0 = rewriter.create<ConstantIntOp>(loc, 0, int64Type);
  Value ci1 = rewriter.create<ConstantIntOp>(loc, 1, int64Type);
  Value ctrue = rewriter.create<ConstantIntOp>(loc, 1, boolType);
  Value cfalse = rewriter.create<ConstantIntOp>(loc, 0, boolType);

  // Construct a dense array indicating valid kk positions within fixed index
  Value kvec_i1 = rewriter.create<memref::AllocOp>(loc, memref1DBoolType, nk);
  rewriter.create<linalg::FillOp>(loc, kvec_i1, cfalse);
  scf::ParallelOp colLoop1 = rewriter.create<scf::ParallelOp>(loc, fixedIndexStart, fixedIndexEnd, c1);
  Value jj = colLoop1.getInductionVars()[0];
  rewriter.setInsertionPointToStart(colLoop1.getBody());
  Value col64 = rewriter.create<memref::LoadOp>(loc, fixedIndices, jj);
  Value col = rewriter.create<IndexCastOp>(loc, col64, indexType);
  rewriter.create<memref::StoreOp>(loc, ctrue, kvec_i1, col);
  rewriter.setInsertionPointAfter(colLoop1);
  // Loop thru all columns; count number of resulting nonzeros in the row
  if (maskIndices != nullptr) {
    colLoop1 = rewriter.create<scf::ParallelOp>(loc, maskStart, maskEnd, c1, ci0);
    Value mm = colLoop1.getInductionVars()[0];
    rewriter.setInsertionPointToStart(colLoop1.getBody());
    col64 = rewriter.create<memref::LoadOp>(loc, maskIndices, mm);
    col = rewriter.create<IndexCastOp>(loc, col64, indexType);
  } else {
    colLoop1 = rewriter.create<scf::ParallelOp>(loc, maskStart, maskEnd, c1, ci0);
    col = colLoop1.getInductionVars()[0];
    rewriter.setInsertionPointToStart(colLoop1.getBody());
  }
  Value colPlus1 = rewriter.create<AddIOp>(loc, col, c1);
  Value rowStart64 = rewriter.create<memref::LoadOp>(loc, iterPointers, col);
  Value rowEnd64 = rewriter.create<memref::LoadOp>(loc, iterPointers, colPlus1);
  Value cmpRowSame = rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, rowStart64, rowEnd64);
  // Find overlap in column indices with kvec
  scf::IfOp ifBlock_overlap = rewriter.create<scf::IfOp>(loc, int64Type, cmpRowSame, true);
  // if cmpRowSame
  rewriter.setInsertionPointToStart(ifBlock_overlap.thenBlock());
  rewriter.create<scf::YieldOp>(loc, ci0);
  // else
  rewriter.setInsertionPointToStart(ifBlock_overlap.elseBlock());
  // Walk thru the indices; on a match yield 1, else yield 0
  scf::WhileOp whileLoop = rewriter.create<scf::WhileOp>(loc, int64Type, rowStart64);
  Block *before = rewriter.createBlock(&whileLoop.before(), {}, int64Type);
  Block *after = rewriter.createBlock(&whileLoop.after(), {}, int64Type);
  Value ii64 = before->getArgument(0);
  rewriter.setInsertionPointToStart(&whileLoop.before().front());
  // Check if ii >= rowEnd
  Value cmpEndReached = rewriter.create<CmpIOp>(loc, CmpIPredicate::uge, ii64, rowEnd64);
  scf::IfOp ifBlock_continueSearch = rewriter.create<scf::IfOp>(loc, ArrayRef<Type>{boolType, int64Type}, cmpEndReached, true);
  // if cmpEndReached
  rewriter.setInsertionPointToStart(ifBlock_continueSearch.thenBlock());
  rewriter.create<scf::YieldOp>(loc, ValueRange{cfalse, ci0});
  // else
  rewriter.setInsertionPointToStart(ifBlock_continueSearch.elseBlock());
  // Check if row has a match in kvec
  Value ii = rewriter.create<IndexCastOp>(loc, ii64, indexType);
  Value kk64 = rewriter.create<memref::LoadOp>(loc, iterIndices, ii);
  Value kk = rewriter.create<IndexCastOp>(loc, kk64, indexType);
  Value cmpPair = rewriter.create<memref::LoadOp>(loc, kvec_i1, kk);
  Value cmpResult0 = rewriter.create<SelectOp>(loc, cmpPair, cfalse, ctrue);
  Value cmpResult1 = rewriter.create<SelectOp>(loc, cmpPair, ci1, ii64);
  rewriter.create<scf::YieldOp>(loc, ValueRange{cmpResult0, cmpResult1});
  // end if cmpEndReached
  rewriter.setInsertionPointAfter(ifBlock_continueSearch);
  Value continueSearch = ifBlock_continueSearch.getResult(0);
  Value valToSend = ifBlock_continueSearch.getResult(1);
  rewriter.create<scf::ConditionOp>(loc, continueSearch, valToSend);
  // "do" portion of while loop
  rewriter.setInsertionPointToStart(&whileLoop.after().front());
  Value iiPrev = after->getArgument(0);
  Value iiNext = rewriter.create<AddIOp>(loc, iiPrev, ci1);
  rewriter.create<scf::YieldOp>(loc, iiNext);
  rewriter.setInsertionPointAfter(whileLoop);
  Value res = whileLoop.getResult(0);
  rewriter.create<scf::YieldOp>(loc, res);
  // end if cmpRowSame
  rewriter.setInsertionPointAfter(ifBlock_overlap);
  Value overlap = ifBlock_overlap.getResult(0);
  scf::ReduceOp reducer = rewriter.create<scf::ReduceOp>(loc, overlap);
  Value lhs = reducer.getRegion().getArgument(0);
  Value rhs = reducer.getRegion().getArgument(1);
  rewriter.setInsertionPointToStart(&reducer.getRegion().front());
  Value z = rewriter.create<AddIOp>(loc, lhs, rhs);
  rewriter.create<scf::ReduceReturnOp>(loc, z);
  // end col loop
  rewriter.setInsertionPointAfter(colLoop1);
  Value total = colLoop1.getResult(0);
  rewriter.create<memref::DeallocOp>(loc, kvec_i1);
  return total;
}

void computeInnerProduct(PatternRewriter &rewriter, Value nk,
                          Value fixedIndices, Value fixedValues, Value fixedIndexStart, Value fixedIndexEnd,
                          Value iterPointers, Value iterIndices, Value iterValues,
                          // If no mask is used, set maskIndices to nullptr, and provide maskStart=c0 and maskEnd=len(iterPointers)-1
                          Value maskIndices, Value maskStart, Value maskEnd,
                          Type valueType, ExtensionBlocks extBlocks,
                          Value outputIndices, Value outputValues, Value indexOffset
                          ) {
  Location loc = rewriter.getUnknownLoc();

  // Types used in this function
  Type indexType = rewriter.getIndexType();
  Type int64Type = rewriter.getIntegerType(64);
  Type boolType = rewriter.getI1Type();
  MemRefType memref1DBoolType = MemRefType::get({-1}, boolType);
  MemRefType memref1DValueType = MemRefType::get({-1}, valueType);

  // Initial constants
  Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
  Value ctrue = rewriter.create<ConstantIntOp>(loc, 1, boolType);
  Value cfalse = rewriter.create<ConstantIntOp>(loc, 0, boolType);

  // Construct a dense array of row values
  Value kvec = rewriter.create<memref::AllocOp>(loc, memref1DValueType, nk);
  Value kvec_i1 = rewriter.create<memref::AllocOp>(loc, memref1DBoolType, nk);
  rewriter.create<linalg::FillOp>(loc, kvec_i1, cfalse);
  scf::ParallelOp colLoop3p = rewriter.create<scf::ParallelOp>(loc, fixedIndexStart, fixedIndexEnd, c1);
  Value jj = colLoop3p.getInductionVars()[0];
  rewriter.setInsertionPointToStart(colLoop3p.getBody());
  Value fixedJ64 = rewriter.create<memref::LoadOp>(loc, fixedIndices, jj);
  Value fixedJ = rewriter.create<IndexCastOp>(loc, fixedJ64, indexType);
  rewriter.create<memref::StoreOp>(loc, ctrue, kvec_i1, fixedJ);
  Value val = rewriter.create<memref::LoadOp>(loc, fixedValues, jj);
  rewriter.create<memref::StoreOp>(loc, val, kvec, fixedJ);

  // end col loop 3p
  rewriter.setInsertionPointAfter(colLoop3p);

  Value col64, col;
  scf::ForOp colLoop3f;
  if (maskIndices != nullptr) {
      colLoop3f = rewriter.create<scf::ForOp>(loc, maskStart, maskEnd, c1, c0);
      Value mm = colLoop3f.getInductionVar();
      rewriter.setInsertionPointToStart(colLoop3f.getBody());
      col64 = rewriter.create<memref::LoadOp>(loc, maskIndices, mm);
      col = rewriter.create<IndexCastOp>(loc, col64, indexType);
  } else {
      colLoop3f = rewriter.create<scf::ForOp>(loc, maskStart, maskEnd, c1, c0);
      col = colLoop3f.getInductionVar();
      rewriter.setInsertionPointToStart(colLoop3f.getBody());
      col64 = rewriter.create<IndexCastOp>(loc, col, int64Type);
  }

  Value offset = colLoop3f.getLoopBody().getArgument(1);
  Value colPlus1 = rewriter.create<AddIOp>(loc, col, c1);
  Value iStart64 = rewriter.create<memref::LoadOp>(loc, iterPointers, col);
  Value iEnd64 = rewriter.create<memref::LoadOp>(loc, iterPointers, colPlus1);
  Value iStart = rewriter.create<IndexCastOp>(loc, iStart64, indexType);
  Value iEnd = rewriter.create<IndexCastOp>(loc, iEnd64, indexType);

  // insert add identity block
  graphblas::YieldOp addIdentityYield = llvm::dyn_cast_or_null<graphblas::YieldOp>(extBlocks.addIdentity->getTerminator());
  rewriter.mergeBlocks(extBlocks.addIdentity, rewriter.getBlock(), {});
  Value addIdentity = addIdentityYield.values().front();
  rewriter.eraseOp(addIdentityYield);

  scf::ForOp kLoop = rewriter.create<scf::ForOp>(loc, iStart, iEnd, c1, ValueRange{addIdentity, cfalse});
  Value ii = kLoop.getInductionVar();
  Value curr = kLoop.getLoopBody().getArgument(1);
  Value alive = kLoop.getLoopBody().getArgument(2);
  rewriter.setInsertionPointToStart(kLoop.getBody());

  Value kk64 = rewriter.create<memref::LoadOp>(loc, iterIndices, ii);
  Value kk = rewriter.create<IndexCastOp>(loc, kk64, indexType);
  Value cmpPair = rewriter.create<memref::LoadOp>(loc, kvec_i1, kk);
  scf::IfOp ifBlock_cmpPair = rewriter.create<scf::IfOp>(loc, ArrayRef<Type>{valueType, boolType}, cmpPair, true);
  // if cmpPair
  rewriter.setInsertionPointToStart(ifBlock_cmpPair.thenBlock());

  Value aVal = rewriter.create<memref::LoadOp>(loc, kvec, kk);
  Value bVal = rewriter.create<memref::LoadOp>(loc, iterValues, ii);

  // insert multiply operation block
  graphblas::YieldOp multYield = llvm::dyn_cast_or_null<graphblas::YieldOp>(extBlocks.mult->getTerminator());
  Value multResult = multYield.values().front();
  rewriter.eraseOp(multYield);
  rewriter.mergeBlocks(extBlocks.mult, rewriter.getBlock(), {aVal, bVal});

  // insert add operation block
  graphblas::YieldOp addYield = llvm::dyn_cast_or_null<graphblas::YieldOp>(extBlocks.add->getTerminator());
  Value addResult = addYield.values().front();
  rewriter.eraseOp(addYield);
  rewriter.mergeBlocks(extBlocks.add, rewriter.getBlock(), {curr, multResult});

  rewriter.create<scf::YieldOp>(loc, ValueRange{addResult, ctrue});

  // else
  rewriter.setInsertionPointToStart(ifBlock_cmpPair.elseBlock());
  rewriter.create<scf::YieldOp>(loc, ValueRange{curr, alive});

  // end if cmpPair
  rewriter.setInsertionPointAfter(ifBlock_cmpPair);
  Value newCurr = ifBlock_cmpPair.getResult(0);
  Value newAlive = ifBlock_cmpPair.getResult(1);
  rewriter.create<scf::YieldOp>(loc, ValueRange{newCurr, newAlive});

  // end k loop
  rewriter.setInsertionPointAfter(kLoop);

  Value total = kLoop.getResult(0);
  Value notEmpty = kLoop.getResult(1);

  scf::IfOp ifBlock_newOffset = rewriter.create<scf::IfOp>(loc, indexType, notEmpty, true);
  // if not empty
  rewriter.setInsertionPointToStart(ifBlock_newOffset.thenBlock());

  // Store total in Cx
  Value cjPos = rewriter.create<AddIOp>(loc, indexOffset, offset);
  rewriter.create<memref::StoreOp>(loc, col64, outputIndices, cjPos);

  // Does total need to be transformed?
  if (extBlocks.transformOut) {
    graphblas::YieldOp yield = llvm::dyn_cast_or_null<graphblas::YieldOp>(extBlocks.transformOut->getTerminator());
    Value transformResult = yield.values().front();

    rewriter.mergeBlocks(extBlocks.transformOut, rewriter.getBlock(), {total});

    rewriter.create<memref::StoreOp>(loc, transformResult, outputValues, cjPos);
    rewriter.eraseOp(yield);
  } else {
    // write total as-is
    rewriter.create<memref::StoreOp>(loc, total, outputValues, cjPos);
  }

  // Increment offset
  Value offsetPlus1 = rewriter.create<AddIOp>(loc, offset, c1);
  rewriter.create<scf::YieldOp>(loc, offsetPlus1);

  // else
  rewriter.setInsertionPointToStart(ifBlock_newOffset.elseBlock());
  rewriter.create<scf::YieldOp>(loc, offset);

  // end if not empty
  rewriter.setInsertionPointAfter(ifBlock_newOffset);

  Value newOffset = ifBlock_newOffset.getResult(0);
  rewriter.create<scf::YieldOp>(loc, newOffset);

  // end col loop 3f
  rewriter.setInsertionPointAfter(colLoop3f);
  rewriter.create<memref::DeallocOp>(loc, kvec);
  rewriter.create<memref::DeallocOp>(loc, kvec_i1);
}

class LowerMatrixMultiplyGenericRewrite : public OpRewritePattern<graphblas::MatrixMultiplyGenericOp> {
public:
  using OpRewritePattern<graphblas::MatrixMultiplyGenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::MatrixMultiplyGenericOp op, PatternRewriter &rewriter) const {
    // Required blocks
    RegionRange extensions = op.extensions();
    ExtensionBlocks extBlocks;
    std::set<graphblas::YieldKind> required = {
        graphblas::YieldKind::ADD_IDENTITY,
        graphblas::YieldKind::ADD,
        graphblas::YieldKind::MULT
    };
    std::set<graphblas::YieldKind> optional = {graphblas::YieldKind::TRANSFORM_OUT};
    LogicalResult extractResult = extBlocks.extractBlocks(op, extensions, required, optional);

    if (extractResult.failed()) {
      return extractResult;
    }

    // Inputs
    Value A = op.a();
    Value B = op.b();

    unsigned aRank = A.getType().dyn_cast<RankedTensorType>().getRank();
    unsigned bRank = B.getType().dyn_cast<RankedTensorType>().getRank();

    if (aRank == 2 && bRank == 2)
      return rewriteMatrixMatrixMultiplication(op, rewriter, extBlocks);
    else if (aRank == 2 && bRank == 1)
      return rewriteMatrixVectorMultiplication(op, rewriter, extBlocks);
    else if (aRank == 1 && bRank == 2)
      return rewriteVectorMatrixMultiplication(op, rewriter, extBlocks);
    else
      return rewriteVectorVectorMultiplication(op, rewriter, extBlocks);
  };

private:
  LogicalResult rewriteMatrixMatrixMultiplication(graphblas::MatrixMultiplyGenericOp op, PatternRewriter &rewriter, ExtensionBlocks extBlocks) const {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = rewriter.getUnknownLoc();

    // Inputs
    Value A = op.a();
    Value B = op.b();
    Value mask = op.mask();

    // Types
    Type indexType = rewriter.getIndexType();
    Type int64Type = rewriter.getIntegerType(64);
    Type valueType = op.getResult().getType().dyn_cast<RankedTensorType>().getElementType();

    MemRefType memref1DI64Type = MemRefType::get({-1}, int64Type);
    MemRefType memref1DValueType = MemRefType::get({-1}, valueType);

    // Initial constants
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
    Value ci0 = rewriter.create<ConstantIntOp>(loc, 0, int64Type);

    Value nrow = rewriter.create<graphblas::NumRowsOp>(loc, A);
    Value ncol = rewriter.create<graphblas::NumColsOp>(loc, B);
    Value nk = rewriter.create<graphblas::NumColsOp>(loc, A); // guaranteed equal to B.rows
    Value nrow_plus_one = rewriter.create<AddIOp>(loc, nrow, c1);

    Value C = callEmptyLike(rewriter, module, loc, A);
    callResizeDim(rewriter, module, loc, C, c0, nrow);
    callResizeDim(rewriter, module, loc, C, c1, ncol);
    callResizePointers(rewriter, module, loc, C, c1, nrow_plus_one);
    C = convertToExternalCSR(rewriter, module, loc, C);

    Value Ap = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, A, c1);
    Value Aj = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, A, c1);
    Value Ax = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, A);
    Value Bp = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, B, c1);
    Value Bi = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, B, c1);
    Value Bx = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, B);
    Value Cp = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, C, c1);
    Value Mp, Mj;
    if (mask)
    {
        Mp = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, mask, c1);
        Mj = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, mask, c1);
    }

    // 1st pass
    //   Compute the number of nonzero entries per row.
    //   Store results in Cp
    //   The rows in A are the fixed elements, while the columns of B are the iteration element
    scf::ParallelOp rowLoop1 = rewriter.create<scf::ParallelOp>(loc, c0, nrow, c1);
    Value row = rowLoop1.getInductionVars()[0];
    rewriter.setInsertionPointToStart(rowLoop1.getBody());

    Value colStart64 = rewriter.create<memref::LoadOp>(loc, Ap, row);
    Value rowPlus1 = rewriter.create<AddIOp>(loc, row, c1);
    Value colEnd64 = rewriter.create<memref::LoadOp>(loc, Ap, rowPlus1);
    Value cmpColSame = rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, colStart64, colEnd64);

    scf::IfOp ifBlock_rowTotal = rewriter.create<scf::IfOp>(loc, int64Type, cmpColSame, true);
    // if cmpColSame
    rewriter.setInsertionPointToStart(ifBlock_rowTotal.thenBlock());
    rewriter.create<scf::YieldOp>(loc, ci0);

    // else
    rewriter.setInsertionPointToStart(ifBlock_rowTotal.elseBlock());
    Value colStart = rewriter.create<IndexCastOp>(loc, colStart64, indexType);
    Value colEnd = rewriter.create<IndexCastOp>(loc, colEnd64, indexType);
    Value total;
    if (mask) {
      Value mcolStart64 = rewriter.create<memref::LoadOp>(loc, Mp, row);
      Value mcolEnd64 = rewriter.create<memref::LoadOp>(loc, Mp, rowPlus1);
      Value mcolStart = rewriter.create<IndexCastOp>(loc, mcolStart64, indexType);
      Value mcolEnd = rewriter.create<IndexCastOp>(loc, mcolEnd64, indexType);
      total = computeNumOverlaps(rewriter, nk, Aj, colStart, colEnd, Bp, Bi, Mj, mcolStart, mcolEnd, valueType);
    } else {
      total = computeNumOverlaps(rewriter, nk, Aj, colStart, colEnd, Bp, Bi, nullptr, c0, ncol, valueType);
    }
    rewriter.create<scf::YieldOp>(loc, total);

    // end if cmpColSame
    rewriter.setInsertionPointAfter(ifBlock_rowTotal);
    Value rowTotal = ifBlock_rowTotal.getResult(0);
    rewriter.create<memref::StoreOp>(loc, rowTotal, Cp, row);

    // end row loop
    rewriter.setInsertionPointAfter(rowLoop1);

    // 2nd pass
    //   Compute the cumsum of values in Cp to build the final Cp
    //   Then resize C's indices and values
    //   The rows in A are the fixed elements, while the columns of B are the iteration element
    scf::ForOp rowLoop2 = rewriter.create<scf::ForOp>(loc, c0, nrow, c1);
    Value cs_i = rowLoop2.getInductionVar();
    rewriter.setInsertionPointToStart(rowLoop2.getBody());

    Value csTemp = rewriter.create<memref::LoadOp>(loc, Cp, cs_i);
    Value cumsum = rewriter.create<memref::LoadOp>(loc, Cp, nrow);
    rewriter.create<memref::StoreOp>(loc, cumsum, Cp, cs_i);
    Value cumsum2 = rewriter.create<AddIOp>(loc, cumsum, csTemp);
    rewriter.create<memref::StoreOp>(loc, cumsum2, Cp, nrow);

    // end row loop
    rewriter.setInsertionPointAfter(rowLoop2);

    Value nnz = rewriter.create<graphblas::NumValsOp>(loc, C);
    callResizeIndex(rewriter, module, loc, C, c1, nnz);
    callResizeValues(rewriter, module, loc, C, nnz);
    C = convertToExternalCSR(rewriter, module, loc, C);
    Value Cj = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, C, c1);
    Value Cx = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, C);

    // 3rd pass
    //   In parallel over the rows,
    //   compute the nonzero columns and associated values.
    //   Store in Cj and Cx
    scf::ParallelOp rowLoop3 = rewriter.create<scf::ParallelOp>(loc, c0, nrow, c1);
    row = rowLoop3.getInductionVars()[0];
    rewriter.setInsertionPointToStart(rowLoop3.getBody());

    rowPlus1 = rewriter.create<AddIOp>(loc, row, c1);
    Value cpStart64 = rewriter.create<memref::LoadOp>(loc, Cp, row);
    Value cpEnd64 = rewriter.create<memref::LoadOp>(loc, Cp, rowPlus1);
    Value cmp_cpDifferent = rewriter.create<CmpIOp>(loc, CmpIPredicate::ne, cpStart64, cpEnd64);
    scf::IfOp ifBlock_cmpDiff = rewriter.create<scf::IfOp>(loc, cmp_cpDifferent);
    rewriter.setInsertionPointToStart(ifBlock_cmpDiff.thenBlock());

    Value baseIndex64 = rewriter.create<memref::LoadOp>(loc, Cp, row);
    Value baseIndex = rewriter.create<IndexCastOp>(loc, baseIndex64, indexType);

    colStart64 = rewriter.create<memref::LoadOp>(loc, Ap, row);
    colEnd64 = rewriter.create<memref::LoadOp>(loc, Ap, rowPlus1);
    colStart = rewriter.create<IndexCastOp>(loc, colStart64, indexType);
    colEnd = rewriter.create<IndexCastOp>(loc, colEnd64, indexType);

    if (mask) {
      Value mcolStart64 = rewriter.create<memref::LoadOp>(loc, Mp, row);
      Value mcolEnd64 = rewriter.create<memref::LoadOp>(loc, Mp, rowPlus1);
      Value mcolStart = rewriter.create<IndexCastOp>(loc, mcolStart64, indexType);
      Value mcolEnd = rewriter.create<IndexCastOp>(loc, mcolEnd64, indexType);
      computeInnerProduct(rewriter, nk, Aj, Ax, colStart, colEnd, Bp, Bi, Bx, Mj, mcolStart, mcolEnd, valueType, extBlocks, Cj, Cx, baseIndex);
    } else {
      computeInnerProduct(rewriter, nk, Aj, Ax, colStart, colEnd, Bp, Bi, Bx, nullptr, c0, ncol, valueType, extBlocks, Cj, Cx, baseIndex);
    }

    // end if cmpDiff
    rewriter.setInsertionPointAfter(ifBlock_cmpDiff);

    // end row loop
    rewriter.setInsertionPointAfter(rowLoop3);

    rewriter.replaceOp(op, C);

    cleanupIntermediateTensor(rewriter, module, loc, C);

    return success();
  }

  LogicalResult rewriteMatrixVectorMultiplication(graphblas::MatrixMultiplyGenericOp op, PatternRewriter &rewriter, ExtensionBlocks extBlocks) const {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = rewriter.getUnknownLoc();

    // Inputs
    Value A = op.a();
    Value B = op.b();
    Value mask = op.mask();

    // Types
    Type indexType = rewriter.getIndexType();
    Type int64Type = rewriter.getIntegerType(64);
    Type valueType = op.getResult().getType().dyn_cast<RankedTensorType>().getElementType();

    MemRefType memref1DI64Type = MemRefType::get({-1}, int64Type);
    MemRefType memref1DValueType = MemRefType::get({-1}, valueType);

    // Initial constants
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
    Value c2 = rewriter.create<ConstantIndexOp>(loc, 2);
    Value ci0 = rewriter.create<ConstantIntOp>(loc, 0, int64Type);

    Value size = rewriter.create<graphblas::NumRowsOp>(loc, A);
    Value nk = rewriter.create<graphblas::SizeOp>(loc, B);
    Value nk_check = rewriter.create<graphblas::NumColsOp>(loc, A);
    // TODO: how do I check nk == nk_check and raise an exception if they don't match?

    Value C = callEmptyLike(rewriter, module, loc, B);
    callResizeDim(rewriter, module, loc, C, c0, size);
    callResizePointers(rewriter, module, loc, C, c0, c2);

    Value Ap = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, A, c1);
    Value Aj = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, A, c1);
    Value Ax = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, A);
    Value Bp = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, B, c0);
    Value Bi = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, B, c0);
    Value Bx = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, B);
    Value Cp = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, C, c0);
    Value Mp, Mi, maskStart, maskEnd;
    if (mask)
    {
        Mp = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, mask, c0);
        Mi = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, mask, c0);
        Value maskStart64 = rewriter.create<memref::LoadOp>(loc, Mp, c0);
        Value maskEnd64 = rewriter.create<memref::LoadOp>(loc, Mp, c1);
        maskStart = rewriter.create<IndexCastOp>(loc, maskStart64, indexType);
        maskEnd = rewriter.create<IndexCastOp>(loc, maskEnd64, indexType);
    }

    // 1st pass
    //   Compute the number of nonzero entries in the result
    //   Store results in Cp
    //   The vector B is the fixed element, while the rows of A are the iteration element
    Value fixedIndexEnd64 = rewriter.create<memref::LoadOp>(loc, Bp, c1);
    Value fixedIndexEnd = rewriter.create<IndexCastOp>(loc, fixedIndexEnd64, indexType);
    Value cmpColSame = rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, c0, fixedIndexEnd);

    scf::IfOp ifBlock_rowTotal = rewriter.create<scf::IfOp>(loc, int64Type, cmpColSame, true);
    // if cmpColSame
    rewriter.setInsertionPointToStart(ifBlock_rowTotal.thenBlock());
    rewriter.create<scf::YieldOp>(loc, ci0);

    // else
    rewriter.setInsertionPointToStart(ifBlock_rowTotal.elseBlock());
    Value total;
    if (mask) {
      total = computeNumOverlaps(rewriter, nk, Bi, c0, fixedIndexEnd, Ap, Aj, Mi, maskStart, maskEnd, valueType);
    } else {
      total = computeNumOverlaps(rewriter, nk, Bi, c0, fixedIndexEnd, Ap, Aj, nullptr, c0, size, valueType);
    }
    rewriter.create<scf::YieldOp>(loc, total);

    // end if cmpColSame
    rewriter.setInsertionPointAfter(ifBlock_rowTotal);
    Value nnzTotal = ifBlock_rowTotal.getResult(0);
    Value nnz = rewriter.create<IndexCastOp>(loc, nnzTotal, indexType);
    rewriter.create<memref::StoreOp>(loc, nnzTotal, Cp, c1);

    callResizeIndex(rewriter, module, loc, C, c0, nnz);
    callResizeValues(rewriter, module, loc, C, nnz);
    Value Ci = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, C, c0);
    Value Cx = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, C);

    // 2nd pass
    //   Compute the nonzero values.
    //   Store in Ci and Cx
    //   The vector B is the fixed element, while the rows of A are the iteration element
    Value cmp_cpDifferent = rewriter.create<CmpIOp>(loc, CmpIPredicate::ne, c0, nnz);
    scf::IfOp ifBlock_cmpDiff = rewriter.create<scf::IfOp>(loc, cmp_cpDifferent);
    rewriter.setInsertionPointToStart(ifBlock_cmpDiff.thenBlock());

    if (mask) {
      computeInnerProduct(rewriter, nk, Bi, Bx, c0, fixedIndexEnd, Ap, Aj, Ax, Mi, maskStart, maskEnd, valueType, extBlocks, Ci, Cx, c0);
    } else {
      computeInnerProduct(rewriter, nk, Bi, Bx, c0, fixedIndexEnd, Ap, Aj, Ax, nullptr, c0, size, valueType, extBlocks, Ci, Cx, c0);
    }

    // end if cmpDiff
    rewriter.setInsertionPointAfter(ifBlock_cmpDiff);

    rewriter.replaceOp(op, C);

    cleanupIntermediateTensor(rewriter, module, loc, C);

    return success();
  }

  LogicalResult rewriteVectorMatrixMultiplication(graphblas::MatrixMultiplyGenericOp op, PatternRewriter &rewriter, ExtensionBlocks extBlocks) const {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = rewriter.getUnknownLoc();

    // Inputs
    Value A = op.a();
    Value B = op.b();
    Value mask = op.mask();

    // Types
    Type indexType = rewriter.getIndexType();
    Type int64Type = rewriter.getIntegerType(64);
    Type valueType = op.getResult().getType().dyn_cast<RankedTensorType>().getElementType();

    MemRefType memref1DI64Type = MemRefType::get({-1}, int64Type);
    MemRefType memref1DValueType = MemRefType::get({-1}, valueType);

    // Initial constants
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
    Value c2 = rewriter.create<ConstantIndexOp>(loc, 2);
    Value ci0 = rewriter.create<ConstantIntOp>(loc, 0, int64Type);

    Value size = rewriter.create<graphblas::NumColsOp>(loc, B);
    Value nk = rewriter.create<graphblas::SizeOp>(loc, A); // guaranteed equal to B.rows

    Value C = callEmptyLike(rewriter, module, loc, A);
    callResizeDim(rewriter, module, loc, C, c0, size);
    callResizePointers(rewriter, module, loc, C, c0, c2);

    Value Ap = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, A, c0);
    Value Ai = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, A, c0);
    Value Ax = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, A);
    Value Bp = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, B, c1);
    Value Bi = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, B, c1);
    Value Bx = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, B);
    Value Cp = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, C, c0);
    Value Mp, Mi, maskStart, maskEnd;
    if (mask)
    {
        Mp = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, mask, c0);
        Mi = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, mask, c0);
        Value maskStart64 = rewriter.create<memref::LoadOp>(loc, Mp, c0);
        Value maskEnd64 = rewriter.create<memref::LoadOp>(loc, Mp, c1);
        maskStart = rewriter.create<IndexCastOp>(loc, maskStart64, indexType);
        maskEnd = rewriter.create<IndexCastOp>(loc, maskEnd64, indexType);
    }

    // 1st pass
    //   Compute the number of nonzero entries in the result
    //   Store results in Cp
    //   The vector A is the fixed element, while the columns of B are the iteration element
    Value fixedIndexEnd64 = rewriter.create<memref::LoadOp>(loc, Ap, c1);
    Value fixedIndexEnd = rewriter.create<IndexCastOp>(loc, fixedIndexEnd64, indexType);
    Value cmpColSame = rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, c0, fixedIndexEnd);

    scf::IfOp ifBlock_rowTotal = rewriter.create<scf::IfOp>(loc, int64Type, cmpColSame, true);
    // if cmpColSame
    rewriter.setInsertionPointToStart(ifBlock_rowTotal.thenBlock());
    rewriter.create<scf::YieldOp>(loc, ci0);

    // else
    rewriter.setInsertionPointToStart(ifBlock_rowTotal.elseBlock());
    Value total;
    if (mask) {
      total = computeNumOverlaps(rewriter, nk, Ai, c0, fixedIndexEnd, Bp, Bi, Mi, maskStart, maskEnd, valueType);
    } else {
      total = computeNumOverlaps(rewriter, nk, Ai, c0, fixedIndexEnd, Bp, Bi, nullptr, c0, size, valueType);
    }
    rewriter.create<scf::YieldOp>(loc, total);

    // end if cmpColSame
    rewriter.setInsertionPointAfter(ifBlock_rowTotal);
    Value nnzTotal = ifBlock_rowTotal.getResult(0);
    Value nnz = rewriter.create<IndexCastOp>(loc, nnzTotal, indexType);
    rewriter.create<memref::StoreOp>(loc, nnzTotal, Cp, c1);

    callResizeIndex(rewriter, module, loc, C, c0, nnz);
    callResizeValues(rewriter, module, loc, C, nnz);
    Value Ci = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, C, c0);
    Value Cx = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, C);

    // 2nd pass
    //   Compute the nonzero values.
    //   Store in Ci and Cx
    //   The vector A is the fixed element, while the columns of B are the iteration element
    Value cmp_cpDifferent = rewriter.create<CmpIOp>(loc, CmpIPredicate::ne, c0, nnz);
    scf::IfOp ifBlock_cmpDiff = rewriter.create<scf::IfOp>(loc, cmp_cpDifferent);
    rewriter.setInsertionPointToStart(ifBlock_cmpDiff.thenBlock());

    if (mask) {
      computeInnerProduct(rewriter, nk, Ai, Ax, c0, fixedIndexEnd, Bp, Bi, Bx, Mi, maskStart, maskEnd, valueType, extBlocks, Ci, Cx, c0);
    } else {
      computeInnerProduct(rewriter, nk, Ai, Ax, c0, fixedIndexEnd, Bp, Bi, Bx, nullptr, c0, size, valueType, extBlocks, Ci, Cx, c0);
    }

    // end if cmpDiff
    rewriter.setInsertionPointAfter(ifBlock_cmpDiff);

    rewriter.replaceOp(op, C);

    cleanupIntermediateTensor(rewriter, module, loc, C);

    return success();
  }

  LogicalResult rewriteVectorVectorMultiplication(graphblas::MatrixMultiplyGenericOp op, PatternRewriter &rewriter, ExtensionBlocks extBlocks) const {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = rewriter.getUnknownLoc();

    // Inputs
    Value A = op.a();
    Value B = op.b();

    // Types
    Type indexType = rewriter.getIndexType();
    Type int64Type = rewriter.getIntegerType(64);
    Type valueType = A.getType().dyn_cast<RankedTensorType>().getElementType();

    MemRefType memref1DI64Type = MemRefType::get({-1}, int64Type);
    MemRefType memref1DValueType = MemRefType::get({-1}, valueType);

    // Initial constants
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
    Value c2 = rewriter.create<ConstantIndexOp>(loc, 2);

    Value size = rewriter.create<graphblas::SizeOp>(loc, A);

    Value C = callEmptyLike(rewriter, module, loc, A);
    callResizeDim(rewriter, module, loc, C, c0, c1);  // exactly one entry because this is a vector representing a scalar
    callResizePointers(rewriter, module, loc, C, c0, c2);
    callResizeIndex(rewriter, module, loc, C, c0, c1);
    callResizeValues(rewriter, module, loc, C, c1);

    Value Ap = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, A, c0);
    Value Ai = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, A, c0);
    Value Ax = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, A);
    Value Bp = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, B, c0);
    Value Bi = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, B, c0);
    Value Bx = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, B);
    Value Ci = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, C, c0);
    Value Cx = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, C);

    // Single pass
    //   Compute the nonzero values.
    //   Store in Ci and Cx (single-element vector representing a scalar)
    //   The vector A is the fixed element, while the vector B is treated as the iteration element
    Value fixedIndexEnd64 = rewriter.create<memref::LoadOp>(loc, Ap, c1);
    Value fixedIndexEnd = rewriter.create<IndexCastOp>(loc, fixedIndexEnd64, indexType);

    computeInnerProduct(rewriter, size, Ai, Ax, c0, fixedIndexEnd, Bp, Bi, Bx, nullptr, c0, c1, valueType, extBlocks, Ci, Cx, c0);

    // extract scalar from C
    Value cScalar = rewriter.create<memref::LoadOp>(loc, Cx, c0);

    rewriter.replaceOp(op, cScalar);

    cleanupIntermediateTensor(rewriter, module, loc, C);

    return success();
  }
};

class LowerMatrixMultiplyReduceToScalarGenericRewrite : public OpRewritePattern<graphblas::MatrixMultiplyReduceToScalarGenericOp> {
public:
  using OpRewritePattern<graphblas::MatrixMultiplyReduceToScalarGenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::MatrixMultiplyReduceToScalarGenericOp op, PatternRewriter &rewriter) const {
    ModuleOp module = op->getParentOfType<ModuleOp>(); /* ignore unused variable for debugging */ (void) module;
    Location loc = rewriter.getUnknownLoc();

    // Inputs
    Value A = op.a();
    Value B = op.b();
    Value mask = op.mask();

    // Required blocks
    RegionRange extensions = op.extensions();
    ExtensionBlocks extBlocks;
    std::set<graphblas::YieldKind> required = {
        graphblas::YieldKind::ADD_IDENTITY,
        graphblas::YieldKind::ADD,
        graphblas::YieldKind::MULT,
        graphblas::YieldKind::AGG_IDENTITY,
        graphblas::YieldKind::AGG};
    std::set<graphblas::YieldKind> optional = {};
    LogicalResult extractResult = extBlocks.extractBlocks(op, extensions, required, optional);

    if (extractResult.failed())
    {
      return extractResult;
    }

    // Types
    Type indexType = rewriter.getIndexType();
    Type int64Type = rewriter.getIntegerType(64);
    Type boolType = rewriter.getI1Type();
    Type valueType = A.getType().dyn_cast<RankedTensorType>().getElementType();

    MemRefType memref1DI64Type = MemRefType::get({-1}, int64Type);
    MemRefType memref1DBoolType = MemRefType::get({-1}, boolType);
    MemRefType memref1DValueType = MemRefType::get({-1}, valueType);

    // Initial constants
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
    // TODO: make cf0 value dependent on the aggregator
    Value cf0 = llvm::TypeSwitch<Type, Value>(valueType)
        .Case<IntegerType>([&](IntegerType type) { return rewriter.create<ConstantIntOp>(loc, 0, type.getWidth()); })
        .Case<FloatType>([&](FloatType type) { return rewriter.create<ConstantFloatOp>(loc, APFloat(0.0), type); });
    Value cf1 = llvm::TypeSwitch<Type, Value>(valueType)
        .Case<IntegerType>([&](IntegerType type) { return rewriter.create<ConstantIntOp>(loc, 1, type.getWidth()); })
        .Case<FloatType>([&](FloatType type) { return rewriter.create<ConstantFloatOp>(loc, APFloat(1.0), type); });
    Value ctrue = rewriter.create<ConstantIntOp>(loc, 1, boolType);
    Value cfalse = rewriter.create<ConstantIntOp>(loc, 0, boolType);

    // Get sparse tensor info
    Value Ap = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, A, c1);
    Value Aj = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, A, c1);
    Value Ax = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, A);
    Value Bp = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, B, c1);
    Value Bi = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, B, c1);
    Value Bx = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, B);

    Value nrow = rewriter.create<graphblas::NumRowsOp>(loc, A);
    Value ncol = rewriter.create<graphblas::NumColsOp>(loc, B);
    Value nk = rewriter.create<graphblas::NumColsOp>(loc, A); // guaranteed equal to B.rows

    Value Mp, Mj;
    if (mask) {
        Mp = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, mask, c1);
        Mj = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, mask, c1);
    }

    // In parallel over the rows and columns,
    //   compute the nonzero values and accumulate
    scf::ParallelOp rowLoop = rewriter.create<scf::ParallelOp>(loc, c0, nrow, c1, cf0);
    Value row = rowLoop.getInductionVars()[0];
    rewriter.setInsertionPointToStart(rowLoop.getBody());

    Value rowPlus1 = rewriter.create<AddIOp>(loc, row, c1);
    Value apStart64 = rewriter.create<memref::LoadOp>(loc, Ap, row);
    Value apEnd64 = rewriter.create<memref::LoadOp>(loc, Ap, rowPlus1);
    Value cmp_cpSame = rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, apStart64, apEnd64);

    scf::IfOp ifBlock_cmpSame = rewriter.create<scf::IfOp>(loc, valueType, cmp_cpSame, true);
    // if cmpSame
    rewriter.setInsertionPointToStart(ifBlock_cmpSame.thenBlock());
    rewriter.create<scf::YieldOp>(loc, cf0);

    // else
    rewriter.setInsertionPointToStart(ifBlock_cmpSame.elseBlock());

    // Construct a dense array of row values
    Value colStart = rewriter.create<IndexCastOp>(loc, apStart64, indexType);
    Value colEnd = rewriter.create<IndexCastOp>(loc, apEnd64, indexType);
    Value kvec = rewriter.create<memref::AllocOp>(loc, memref1DValueType, nk);
    Value kvec_i1 = rewriter.create<memref::AllocOp>(loc, memref1DBoolType, nk);
    rewriter.create<linalg::FillOp>(loc, kvec_i1, cfalse);

    scf::ParallelOp colLoop1 = rewriter.create<scf::ParallelOp>(loc, colStart, colEnd, c1);
    Value jj = colLoop1.getInductionVars()[0];
    rewriter.setInsertionPointToStart(colLoop1.getBody());
    Value col64 = rewriter.create<memref::LoadOp>(loc, Aj, jj);
    Value col = rewriter.create<IndexCastOp>(loc, col64, indexType);
    rewriter.create<memref::StoreOp>(loc, ctrue, kvec_i1, col);
    Value val = rewriter.create<memref::LoadOp>(loc, Ax, jj);
    rewriter.create<memref::StoreOp>(loc, val, kvec, col);

    // end col loop 1
    rewriter.setInsertionPointAfter(colLoop1);

    // Loop thru all columns of B; accumulate values
    scf::ParallelOp colLoop2;
    if (mask) {
        Value mcolStart64 = rewriter.create<memref::LoadOp>(loc, Mp, row);
        Value mcolEnd64 = rewriter.create<memref::LoadOp>(loc, Mp, rowPlus1);
        Value mcolStart = rewriter.create<IndexCastOp>(loc, mcolStart64, indexType);
        Value mcolEnd = rewriter.create<IndexCastOp>(loc, mcolEnd64, indexType);

        colLoop2 = rewriter.create<scf::ParallelOp>(loc, mcolStart, mcolEnd, c1, cf0);
        Value mm = colLoop2.getInductionVars()[0];
        rewriter.setInsertionPointToStart(colLoop2.getBody());
        col64 = rewriter.create<memref::LoadOp>(loc, Mj, mm);
        col = rewriter.create<IndexCastOp>(loc, col64, indexType);
    } else {
        colLoop2 = rewriter.create<scf::ParallelOp>(loc, c0, ncol, c1, cf0);
        col = colLoop2.getInductionVars()[0];
        rewriter.setInsertionPointToStart(colLoop2.getBody());
        col64 = rewriter.create<IndexCastOp>(loc, col, int64Type);
    }

    Value colPlus1 = rewriter.create<AddIOp>(loc, col, c1);
    Value iStart64 = rewriter.create<memref::LoadOp>(loc, Bp, col);
    Value iEnd64 = rewriter.create<memref::LoadOp>(loc, Bp, colPlus1);
    Value iStart = rewriter.create<IndexCastOp>(loc, iStart64, indexType);
    Value iEnd = rewriter.create<IndexCastOp>(loc, iEnd64, indexType);

    // insert add identity block
    graphblas::YieldOp addIdentityYield = llvm::dyn_cast_or_null<graphblas::YieldOp>(extBlocks.addIdentity->getTerminator());
    rewriter.mergeBlocks(extBlocks.addIdentity, rewriter.getBlock(), {});
    Value addIdentity = addIdentityYield.values().front();
    rewriter.eraseOp(addIdentityYield);

    scf::ForOp kLoop = rewriter.create<scf::ForOp>(loc, iStart, iEnd, c1, addIdentity);
    Value ii = kLoop.getInductionVar();
    Value curr = kLoop.getLoopBody().getArgument(1);
    rewriter.setInsertionPointToStart(kLoop.getBody());

    Value kk64 = rewriter.create<memref::LoadOp>(loc, Bi, ii);
    Value kk = rewriter.create<IndexCastOp>(loc, kk64, indexType);
    Value cmpPair = rewriter.create<memref::LoadOp>(loc, kvec_i1, kk);
    scf::IfOp ifBlock_cmpPair = rewriter.create<scf::IfOp>(loc, valueType, cmpPair, true);
    // if cmpPair
    rewriter.setInsertionPointToStart(ifBlock_cmpPair.thenBlock());

    Value aVal = rewriter.create<memref::LoadOp>(loc, kvec, kk);
    Value bVal = rewriter.create<memref::LoadOp>(loc, Bx, ii);

    // insert multiply operation block
    graphblas::YieldOp multYield = llvm::dyn_cast_or_null<graphblas::YieldOp>(extBlocks.mult->getTerminator());
    Value multResult = multYield.values().front();
    rewriter.eraseOp(multYield);
    rewriter.mergeBlocks(extBlocks.mult, rewriter.getBlock(), {aVal, bVal});

    // insert add operation block
    graphblas::YieldOp addYield = llvm::dyn_cast_or_null<graphblas::YieldOp>(extBlocks.add->getTerminator());
    Value addResult = addYield.values().front();
    rewriter.eraseOp(addYield);
    rewriter.mergeBlocks(extBlocks.add, rewriter.getBlock(), {curr, multResult});

    rewriter.create<scf::YieldOp>(loc, addResult);

    // else
    rewriter.setInsertionPointToStart(ifBlock_cmpPair.elseBlock());
    rewriter.create<scf::YieldOp>(loc, curr);

    // end if cmpPair
    rewriter.setInsertionPointAfter(ifBlock_cmpPair);
    Value newCurr = ifBlock_cmpPair.getResult(0);
    rewriter.create<scf::YieldOp>(loc, newCurr);

    // end k loop
    rewriter.setInsertionPointAfter(kLoop);

    Value colVal = kLoop.getResult(0);

    // FIXME: this is where transform_out goes

    scf::ReduceOp colReducer = rewriter.create<scf::ReduceOp>(loc, colVal);
    BlockArgument lhs = colReducer.getRegion().getArgument(0);
    BlockArgument rhs = colReducer.getRegion().getArgument(1);

    rewriter.setInsertionPointToStart(&colReducer.getRegion().front());


    Region *aggRegion = extBlocks.agg->getParent();
    BlockAndValueMapping mapper;
    // Clone blocks into front of region to displace existing entry block, which will be removed
    // by canonicalization later
    aggRegion->cloneInto(&colReducer.getRegion(), colReducer.getRegion().begin(), mapper);
    graphblas::YieldOp colYield = llvm::dyn_cast_or_null<graphblas::YieldOp>(colReducer.getRegion().front().getTerminator());
    Value colAggResult = colYield.values().front();
    rewriter.setInsertionPointAfter(colYield);
    rewriter.create<scf::ReduceReturnOp>(loc, colAggResult);
    rewriter.eraseOp(colYield);

    rewriter.setInsertionPointAfter(colReducer);

    // end col loop 2
    rewriter.setInsertionPointAfter(colLoop2);

    Value subtotal = colLoop2.getResult(0);
    rewriter.create<memref::DeallocOp>(loc, kvec);
    rewriter.create<memref::DeallocOp>(loc, kvec_i1);
    rewriter.create<scf::YieldOp>(loc, subtotal);

    // end if cmpSame
    rewriter.setInsertionPointAfter(ifBlock_cmpSame);

    Value rowTotal = ifBlock_cmpSame.getResult(0);

    scf::ReduceOp rowReducer = rewriter.create<scf::ReduceOp>(loc, rowTotal);
    lhs = rowReducer.getRegion().getArgument(0);
    rhs = rowReducer.getRegion().getArgument(1);

    rewriter.setInsertionPointToStart(&rowReducer.getRegion().front());

    graphblas::YieldOp yield = llvm::dyn_cast_or_null<graphblas::YieldOp>(extBlocks.agg->getTerminator());
    Value aggResult = yield.values().front();

    // we can safely merge this agg block now, since the previous agg instance was cloned above
    rewriter.mergeBlocks(extBlocks.agg, rewriter.getBlock(), {lhs, rhs});
    rewriter.create<scf::ReduceReturnOp>(loc, aggResult);
    rewriter.eraseOp(yield);

    // end row loop
    rewriter.setInsertionPointAfter(rowLoop);

    Value total = rowLoop.getResult(0);

    rewriter.replaceOp(op, total);

    return success();
  };
};

class LowerUpdateRewrite : public OpRewritePattern<graphblas::UpdateOp> {
public:
  using OpRewritePattern<graphblas::UpdateOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::UpdateOp op, PatternRewriter &rewriter) const {
    // Inputs
    Value output = op.output();
    llvm::Optional<llvm::StringRef> accumulateOperator = op.accumulate_operator();
    Value mask = op.mask();
    bool replace = op.replace();
    bool maskComplement = op.mask_complement();

    // Types
    RankedTensorType outputType = output.getType().dyn_cast<RankedTensorType>();

    unsigned rank = outputType.getRank();  // ranks guaranteed to be equal

    if (rank == 2) {
      return op.emitError("Matrix update is not yet supported.");
    } else {
      // Vector past this point
      if (accumulateOperator) {
        if (mask) {
          if (replace) {
            // input -> output(mask) { accumulate, replace }
            return op.emitError("Update with mask+accumulate+replace is not supported yet");
          } else {
            // input -> output(mask) { accumulate }
            return op.emitError("Update with mask+accumulate is not supported yet");
          }
        } else {
          // input -> output { accumulate, replace? }
          return rewriteUpdateVectorAccumulate(op, rewriter);
        }
      } else {
        if (mask) {
          if (replace) {
            // input -> output(mask) { replace }
            // Inefficient; caller should apply mask when input is created
            return op.emitError("Update with mask+replace is not supported yet");
          } else {
            // input -> output(mask)
            // Merges input into output
            return op.emitError("Update with mask and no accumulator is not supported yet");
          }
        } else {
          // input -> output { replace? }
          // Sort of pointless; caller should simply use input or call graphblas.dup if they want a copy
          return op.emitError("Update with no accumulator or mask is not supported yet");
        }
      }
    }
  };

private:
  LogicalResult rewriteUpdateVectorAccumulate(graphblas::UpdateOp op, PatternRewriter &rewriter) const {
    ModuleOp module = op->getParentOfType<ModuleOp>(); /* ignore unused variable for debugging */ (void)module;
    Location loc = rewriter.getUnknownLoc();

    // Inputs
    Value input = op.input();
    Value output = op.output();
    std::string accumulateOperator = op.accumulate_operator()->str();

    // Types
    RankedTensorType outputType = output.getType().dyn_cast<RankedTensorType>();
    Type boolType = rewriter.getI1Type();
    Type indexType = rewriter.getIndexType();
    Type int64Type = rewriter.getIntegerType(64);
    Type valueType = outputType.getElementType();
    MemRefType memref1DI64Type = MemRefType::get({-1}, int64Type);
    MemRefType memref1DValueType = MemRefType::get({-1}, valueType);

    // Initial constants
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
    Value cfalse = rewriter.create<ConstantIntOp>(loc, 0, boolType);
    Value ctrue = rewriter.create<ConstantIntOp>(loc, 1, boolType);
    Value cf0 = llvm::TypeSwitch<Type, Value>(valueType)
        .Case<IntegerType>([&](IntegerType type) { return rewriter.create<ConstantIntOp>(loc, 0, type.getWidth()); })
        .Case<FloatType>([&](FloatType type) { return rewriter.create<ConstantFloatOp>(loc, APFloat(0.0), type); });

    Value size = rewriter.create<graphblas::SizeOp>(loc, output);

    // Get sparse tensor info
    Value Ip = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, input, c0);
    Value Ii = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, input, c0);
    Value Ix = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, input);
    Value Op = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, output, c0);
    Value Oi = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, output, c0);
    Value Ox = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, output);
    Value Innz64 = rewriter.create<memref::LoadOp>(loc, Ip, c1);
    Value Onnz64 = rewriter.create<memref::LoadOp>(loc, Op, c1);
    Value Innz = rewriter.create<IndexCastOp>(loc, Innz64, indexType);
    Value Onnz = rewriter.create<IndexCastOp>(loc, Onnz64, indexType);

    // While Loop (exit when either array is exhausted)
    ArrayRef<Type> whileTypes = ArrayRef<Type>{indexType, indexType, indexType, indexType, boolType, boolType, indexType};
    scf::WhileOp whileLoop = rewriter.create<scf::WhileOp>(loc, whileTypes, ValueRange{c0, c0, c0, c0, ctrue, ctrue, c0});
    Block *before = rewriter.createBlock(&whileLoop.before(), {}, whileTypes);
    Block *after = rewriter.createBlock(&whileLoop.after(), {}, whileTypes);
    // "while" portion of the loop
    rewriter.setInsertionPointToStart(&whileLoop.before().front());
    Value posI = before->getArgument(0);
    Value posO = before->getArgument(1);
    Value validPosI = rewriter.create<CmpIOp>(loc, CmpIPredicate::ult, posI, Innz);
    Value validPosO = rewriter.create<CmpIOp>(loc, CmpIPredicate::ult, posO, Onnz);
    Value continueLoop = rewriter.create<AndOp>(loc, validPosI, validPosO);
    rewriter.create<scf::ConditionOp>(loc, continueLoop, before->getArguments());

    // "do" portion of while loop
    rewriter.setInsertionPointToStart(&whileLoop.after().front());
    posI = after->getArgument(0);
    posO = after->getArgument(1);
    Value idxI = after->getArgument(2);
    Value idxO = after->getArgument(3);
    Value needsUpdateI = after->getArgument(4);
    Value needsUpdateO = after->getArgument(5);
    Value count = after->getArgument(6);

    // Update input index based on flag
    scf::IfOp if_updateI = rewriter.create<scf::IfOp>(loc, indexType, needsUpdateI, true);
    // if updateI
    rewriter.setInsertionPointToStart(if_updateI.thenBlock());
    Value updatedIdxI64 = rewriter.create<memref::LoadOp>(loc, Ii, posI);
    Value updatedIdxI = rewriter.create<IndexCastOp>(loc, updatedIdxI64, indexType);
    rewriter.create<scf::YieldOp>(loc, updatedIdxI);
    // else
    rewriter.setInsertionPointToStart(if_updateI.elseBlock());
    rewriter.create<scf::YieldOp>(loc, idxI);
    rewriter.setInsertionPointAfter(if_updateI);

    // Update output index based on flag
    scf::IfOp if_updateO = rewriter.create<scf::IfOp>(loc, indexType, needsUpdateO, true);
    // if updateO
    rewriter.setInsertionPointToStart(if_updateO.thenBlock());
    Value updatedIdxO64 = rewriter.create<memref::LoadOp>(loc, Oi, posO);
    Value updatedIdxO = rewriter.create<IndexCastOp>(loc, updatedIdxO64, indexType);
    rewriter.create<scf::YieldOp>(loc, updatedIdxO);
    // else
    rewriter.setInsertionPointToStart(if_updateO.elseBlock());
    rewriter.create<scf::YieldOp>(loc, idxO);
    rewriter.setInsertionPointAfter(if_updateO);

    Value newIdxI = if_updateI.getResult(0);
    Value newIdxO = if_updateO.getResult(0);
    Value idxI_lt_idxO = rewriter.create<CmpIOp>(loc, CmpIPredicate::ult, newIdxI, newIdxO);
    Value idxI_gt_idxO = rewriter.create<CmpIOp>(loc, CmpIPredicate::ugt, newIdxI, newIdxO);
    Value posIplus1 = rewriter.create<AddIOp>(loc, posI, c1);
    Value posOplus1 = rewriter.create<AddIOp>(loc, posO, c1);
    Value countplus1 = rewriter.create<AddIOp>(loc, count, c1);

    ArrayRef<Type> doTypes = ArrayRef<Type>{indexType, indexType, boolType, boolType};
    scf::IfOp if_onlyI = rewriter.create<scf::IfOp>(loc, doTypes, idxI_lt_idxO, true);
    // if useIOnly
    rewriter.setInsertionPointToStart(if_onlyI.thenBlock());
    // Should include idxI here in the output
    rewriter.create<scf::YieldOp>(loc, ValueRange{posIplus1, posO, ctrue, cfalse});
    // else
    rewriter.setInsertionPointToStart(if_onlyI.elseBlock());
    scf::IfOp if_onlyO = rewriter.create<scf::IfOp>(loc, doTypes, idxI_gt_idxO, true);
    // if useOOnly
    rewriter.setInsertionPointToStart(if_onlyO.thenBlock());
    // Should include idxO here in the output
    rewriter.create<scf::YieldOp>(loc, ValueRange{posI, posOplus1, cfalse, ctrue});
    // else
    rewriter.setInsertionPointToStart(if_onlyO.elseBlock());
    // At this point, we know idxI == idxO
    // Should include idxI and idxO here in the output
    rewriter.create<scf::YieldOp>(loc, ValueRange{posIplus1, posOplus1, ctrue, ctrue});
    // end useOOnly
    rewriter.setInsertionPointAfter(if_onlyO);
    rewriter.create<scf::YieldOp>(loc, if_onlyO.getResults());
    // end useIOnly
    rewriter.setInsertionPointAfter(if_onlyI);
    Value newPosI = if_onlyI.getResult(0);
    Value newPosO = if_onlyI.getResult(1);
    needsUpdateI = if_onlyI.getResult(2);
    needsUpdateO = if_onlyI.getResult(3);

    rewriter.create<scf::YieldOp>(loc, ValueRange{newPosI, newPosO, newIdxI, newIdxO, needsUpdateI, needsUpdateO, countplus1});
    rewriter.setInsertionPointAfter(whileLoop);

    // For loop (remaining elements after other array is exhausted)
    scf::ForOp forLoop;
    count = whileLoop.getResult(6);
    posI = whileLoop.getResult(0);
    Value remainingPosI = rewriter.create<CmpIOp>(loc, CmpIPredicate::ult, posI, Innz);
    scf::IfOp if_remainingI = rewriter.create<scf::IfOp>(loc, indexType, remainingPosI, true);
    // if remainingI
    rewriter.setInsertionPointToStart(if_remainingI.thenBlock());
    forLoop = rewriter.create<scf::ForOp>(loc, posI, Innz, c1, count);
    Value ii = forLoop.getInductionVar();
    Value currCount = forLoop.getLoopBody().getArgument(1);
    rewriter.setInsertionPointToStart(forLoop.getBody());
    // Should include idxI here in the output
    Value newCount = rewriter.create<AddIOp>(loc, currCount, c1);
    rewriter.create<scf::YieldOp>(loc, newCount);
    rewriter.setInsertionPointAfter(forLoop);
    rewriter.create<scf::YieldOp>(loc, forLoop.getResult(0));
    // else
    rewriter.setInsertionPointToStart(if_remainingI.elseBlock());
    posO = whileLoop.getResult(1);
    Value remainingPosO = rewriter.create<CmpIOp>(loc, CmpIPredicate::ult, posO, Onnz);
    scf::IfOp if_remainingO = rewriter.create<scf::IfOp>(loc, indexType, remainingPosO, true);
    // if remainingO
    rewriter.setInsertionPointToStart(if_remainingO.thenBlock());
    forLoop = rewriter.create<scf::ForOp>(loc, posO, Onnz, c1, count);
    Value oo = forLoop.getInductionVar();
    currCount = forLoop.getLoopBody().getArgument(1);
    rewriter.setInsertionPointToStart(forLoop.getBody());
    // Should include idxO here in the output
    newCount = rewriter.create<AddIOp>(loc, currCount, c1);
    rewriter.create<scf::YieldOp>(loc, newCount);
    rewriter.setInsertionPointAfter(forLoop);
    rewriter.create<scf::YieldOp>(loc, forLoop.getResult(0));
    // else
    rewriter.setInsertionPointToStart(if_remainingO.elseBlock());
    rewriter.create<scf::YieldOp>(loc, count);
    // end remainingO
    rewriter.setInsertionPointAfter(if_remainingO);
    rewriter.create<scf::YieldOp>(loc, if_remainingO.getResults());
    // end remainingI
    rewriter.setInsertionPointAfter(if_remainingI);

    Value finalCount = if_remainingI.getResult(0);

    rewriter.replaceOp(op, finalCount);

    return success();
  }
};

class LowerEqualRewrite : public OpRewritePattern<graphblas::EqualOp> {
public:
  using OpRewritePattern<graphblas::EqualOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::EqualOp op, PatternRewriter &rewriter) const {
    Location loc = rewriter.getUnknownLoc();

    // Inputs
    Value A = op.a();
    Value B = op.b();
    RankedTensorType aType = A.getType().dyn_cast<RankedTensorType>();

    // Types
    Type boolType = rewriter.getI1Type();
    Type indexType = rewriter.getIndexType();
    Type int64Type = rewriter.getIntegerType(64);
    Type valueType = aType.getElementType();
    MemRefType memref1DI64Type = MemRefType::get({-1}, int64Type);
    MemRefType memref1DValueType = MemRefType::get({-1}, valueType);

    // Initial constants
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
    Value cfalse = rewriter.create<ConstantIntOp>(loc, 0, boolType);
    Value ctrue = rewriter.create<ConstantIntOp>(loc, 1, boolType);

    unsigned rank = aType.getRank();  // ranks guaranteed to be equal

    if (rank == 2) {
      // Matrix check
      return op.emitError("Matrix equality check is not yet supported.");
    } else {
      // Vector check
      // Check size
      Value aSize = rewriter.create<graphblas::SizeOp>(loc, A);
      Value bSize = rewriter.create<graphblas::SizeOp>(loc, B);
      Value cmpSize = rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, aSize, bSize);
      scf::IfOp ifOuter = rewriter.create<scf::IfOp>(loc, boolType, cmpSize, true);
      // if cmpSize
      rewriter.setInsertionPointToStart(ifOuter.thenBlock());

      // Check number of non-zeros
      Value Ap = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, A, c0);
      Value Bp = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, B, c0);
      Value aNnz = rewriter.create<memref::LoadOp>(loc, Ap, c1);
      Value bNnz = rewriter.create<memref::LoadOp>(loc, Bp, c1);
      Value cmpNnz = rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, aNnz, bNnz);
      scf::IfOp ifNnz = rewriter.create<scf::IfOp>(loc, boolType, cmpNnz, true);
      // if cmpNnz
      rewriter.setInsertionPointToStart(ifNnz.thenBlock());

      // Check index positions and values
      Value nnz = rewriter.create<IndexCastOp>(loc, aNnz, indexType);
      Value Ai = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, A, c0);
      Value Bi = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, B, c0);
      Value Ax = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, A);
      Value Bx = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, B);

      scf::ParallelOp indexLoop = rewriter.create<scf::ParallelOp>(loc, c0, nnz, c1, ctrue);
      Value loopIdx = indexLoop.getInductionVars()[0];
      rewriter.setInsertionPointToStart(indexLoop.getBody());

      Value aIndex = rewriter.create<memref::LoadOp>(loc, Ai, loopIdx);
      Value bIndex = rewriter.create<memref::LoadOp>(loc, Bi, loopIdx);
      Value aValue = rewriter.create<memref::LoadOp>(loc, Ax, loopIdx);
      Value bValue = rewriter.create<memref::LoadOp>(loc, Bx, loopIdx);
      Value cmpIndex = rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, aIndex, bIndex);
      Value cmpValue = llvm::TypeSwitch<Type, Value>(valueType)
        .Case<IntegerType>([&](IntegerType type) { return rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, aValue, bValue); })
        .Case<FloatType>([&](FloatType type) { return rewriter.create<CmpFOp>(loc, CmpFPredicate::OEQ, aValue, bValue); });
      Value cmpCombined = rewriter.create<AndOp>(loc, cmpIndex, cmpValue);

      scf::ReduceOp reducer = rewriter.create<scf::ReduceOp>(loc, cmpCombined);
      BlockArgument lhs = reducer.getRegion().getArgument(0);
      BlockArgument rhs = reducer.getRegion().getArgument(1);
      rewriter.setInsertionPointToStart(&reducer.getRegion().front());
      Value cmpFinal = rewriter.create<AndOp>(loc, lhs, rhs);
      rewriter.create<scf::ReduceReturnOp>(loc, cmpFinal);

      rewriter.setInsertionPointAfter(indexLoop);
      rewriter.create<scf::YieldOp>(loc, indexLoop.getResult(0));

      // else cmpNnz
      rewriter.setInsertionPointToStart(ifNnz.elseBlock());
      rewriter.create<scf::YieldOp>(loc, cfalse);
      // end cmpNnz
      rewriter.setInsertionPointAfter(ifNnz);
      Value nnzReturn = ifNnz.getResult(0);
      rewriter.create<scf::YieldOp>(loc, nnzReturn);

      // else cmpSize
      rewriter.setInsertionPointToStart(ifOuter.elseBlock());
      rewriter.create<scf::YieldOp>(loc, cfalse);
      // end cmpSize
      rewriter.setInsertionPointAfter(ifOuter);
      Value isEqual = ifOuter.getResult(0);

      rewriter.replaceOp(op, isEqual);

      return success();
    }
  };
};

void populateGraphBLASLoweringPatterns(RewritePatternSet &patterns) {
  patterns.add<
      LowerMatrixSelectRewrite,
      LowerMatrixReduceToScalarRewrite,
      LowerMatrixReduceToScalarGenericRewrite,
      LowerMatrixMultiplyRewrite,
      LowerConvertLayoutRewrite,
      LowerMatrixApplyRewrite,
      LowerMatrixApplyGenericRewrite,
      LowerMatrixMultiplyReduceToScalarGenericRewrite,
      LowerMatrixMultiplyGenericRewrite,
      LowerUpdateRewrite,
      LowerEqualRewrite,
      LowerSizeRewrite,
      LowerNumRowsRewrite,
      LowerNumColsRewrite,
      LowerNumValsRewrite,
      LowerDupRewrite>(patterns.getContext());
}

struct GraphBLASLoweringPass : public GraphBLASLoweringBase<GraphBLASLoweringPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);
    populateGraphBLASLoweringPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    // TODO how can we mark graphblas ops as illegal here?
  }
};

void populateGraphBLASStructuralizePatterns(RewritePatternSet &patterns)
{
  patterns.add<
      LowerMatrixMultiplyRewrite,
      LowerMatrixApplyRewrite,
      LowerMatrixReduceToScalarRewrite
      >(patterns.getContext());
}

struct GraphBLASStructuralizePass : public GraphBLASStructuralizeBase<GraphBLASStructuralizePass>
{
  void runOnOperation() override
  {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);
    populateGraphBLASStructuralizePatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
} // end anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createGraphBLASLoweringPass() {
  return std::make_unique<GraphBLASLoweringPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createGraphBLASStructuralizePass()
{
  return std::make_unique<GraphBLASStructuralizePass>();
}