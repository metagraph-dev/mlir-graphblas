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

#include "GraphBLAS/GraphBLASArrayUtils.h"
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
  LogicalResult matchAndRewrite(graphblas::SizeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value inputTensor = op.input();
    Value size = rewriter.create<tensor::DimOp>(loc, inputTensor, c0);

    rewriter.replaceOp(op, size);
    return success();
  };
};

class LowerNumRowsRewrite : public OpRewritePattern<graphblas::NumRowsOp> {
public:
  using OpRewritePattern<graphblas::NumRowsOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::NumRowsOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value inputTensor = op.input();
    Value nrows = rewriter.create<tensor::DimOp>(loc, inputTensor, c0);

    rewriter.replaceOp(op, nrows);
    return success();
  };
};

class LowerNumColsRewrite : public OpRewritePattern<graphblas::NumColsOp> {
public:
  using OpRewritePattern<graphblas::NumColsOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::NumColsOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value inputTensor = op.input();
    Value ncols = rewriter.create<tensor::DimOp>(loc, inputTensor, c1);

    rewriter.replaceOp(op, ncols);
    return success();
  };
};

class LowerNumValsRewrite : public OpRewritePattern<graphblas::NumValsOp> {
public:
  using OpRewritePattern<graphblas::NumValsOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::NumValsOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value inputTensor = op.input();
    Type inputType = inputTensor.getType();

    sparse_tensor::SparseTensorEncodingAttr sparseEncoding =
        sparse_tensor::getSparseTensorEncoding(inputType);
    unsigned pointerBitWidth = sparseEncoding.getPointerBitWidth();
    Type pointerType = rewriter.getIntegerType(pointerBitWidth);
    Type indexType = rewriter.getIndexType();

    // Access the pointers
    Type memref1DPointerType = MemRefType::get({-1}, pointerType);
    unsigned rank = inputType.dyn_cast<RankedTensorType>().getRank();
    Value c_rank_minus_1 =
        rewriter.create<arith::ConstantIndexOp>(loc, rank - 1);
    Value ptrs = rewriter.create<sparse_tensor::ToPointersOp>(
        loc, memref1DPointerType, inputTensor, c_rank_minus_1);

    // Find length of pointer array
    Value npointers;
    if (rank == 1) {
      npointers = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    } else {
      Value dimForPointers;
      if (hasRowOrdering(inputType)) {
        dimForPointers = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      } else {
        dimForPointers = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      }
      npointers =
          rewriter.create<tensor::DimOp>(loc, inputTensor, dimForPointers);
    }

    // The last value from the pointers is the number of nonzero values
    Value nnz_ptype = rewriter.create<memref::LoadOp>(loc, ptrs, npointers);
    Value nnz = rewriter.create<arith::IndexCastOp>(loc, nnz_ptype, indexType);

    rewriter.replaceOp(op, nnz);
    return success();
  };
};

class LowerDupRewrite : public OpRewritePattern<graphblas::DupOp> {
public:
  using OpRewritePattern<graphblas::DupOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::DupOp op,
                                PatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();
    Value inputTensor = op.input();

    Value duplicate = callDupTensor(rewriter, module, loc, inputTensor);
    rewriter.replaceOp(op, duplicate);

    return success();
  };
};

class LowerConvertLayoutRewrite
    : public OpRewritePattern<graphblas::ConvertLayoutOp> {
public:
  using OpRewritePattern<graphblas::ConvertLayoutOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *context = op.getContext();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    Value inputTensor = op.input();
    Type inputType = inputTensor.getType();
    Type outputType = op->getResultTypes().front();

    // Shortcut operation if no change
    if (inputType == outputType) {
      rewriter.replaceOp(op, inputTensor);
      return success();
    }

    // otherwise, the rest of this function changes the data layout
    RankedTensorType inputTensorType = inputType.dyn_cast<RankedTensorType>();
    sparse_tensor::SparseTensorEncodingAttr sparseEncoding =
        sparse_tensor::getSparseTensorEncoding(inputTensorType);
    unsigned ptrBitWidth = sparseEncoding.getPointerBitWidth();
    unsigned idxBitWidth = sparseEncoding.getIndexBitWidth();
    Type valueType = inputTensorType.getElementType();
    Type int64Type = rewriter.getIntegerType(64);
    Type indexType = rewriter.getIndexType();

    // Initial constants
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c0_64 = rewriter.create<arith::ConstantIntOp>(loc, 0, int64Type);
    Value c1_64 = rewriter.create<arith::ConstantIntOp>(loc, 1, int64Type);

    // Get sparse tensor info
    Type memref1DI64Type = MemRefType::get({-1}, int64Type);
    Type memref1DValueType = MemRefType::get({-1}, valueType);

    Value inputPtrs = rewriter.create<sparse_tensor::ToPointersOp>(
        loc, memref1DI64Type, inputTensor, c1);
    Value inputIndices = rewriter.create<sparse_tensor::ToIndicesOp>(
        loc, memref1DI64Type, inputTensor, c1);
    Value inputValues = rewriter.create<sparse_tensor::ToValuesOp>(
        loc, memref1DValueType, inputTensor);
    Value nrow = rewriter.create<graphblas::NumRowsOp>(loc, inputTensor);
    Value ncol = rewriter.create<graphblas::NumColsOp>(loc, inputTensor);
    Value nnz = rewriter.create<graphblas::NumValsOp>(loc, inputTensor);

    Value duplicate = callEmptyLike(rewriter, module, loc, inputTensor);

    // Beyond this point, the algorithm assumes csr->csc,
    // so swap nrow/ncol for csc->csr
    bool outputIsCSC = typeIsCSC(outputType);

    // update the reverse index map and dimensions for CSR or CSC
    if (outputIsCSC) {
      callAssignRev(rewriter, module, loc, duplicate, c0, c1);
      callAssignRev(rewriter, module, loc, duplicate, c1, c0);

      callResizeDim(rewriter, module, loc, duplicate, c0, ncol);
      callResizeDim(rewriter, module, loc, duplicate, c1, nrow);
    } else {
      callAssignRev(rewriter, module, loc, duplicate, c0, c0);
      callAssignRev(rewriter, module, loc, duplicate, c1, c1);

      callResizeDim(rewriter, module, loc, duplicate, c0, nrow);
      callResizeDim(rewriter, module, loc, duplicate, c1, ncol);

      Value tmp = nrow;
      nrow = ncol;
      ncol = tmp;
    }

    Value ncols_plus_one = rewriter.create<arith::AddIOp>(loc, ncol, c1);
    callResizePointers(rewriter, module, loc, duplicate, c1, ncols_plus_one);
    callResizeIndex(rewriter, module, loc, duplicate, c1, nnz);
    callResizeValues(rewriter, module, loc, duplicate, nnz);

    // the verify function will ensure that this is CSR->CSC or CSC->CSR
    Value output = castToPtr8(rewriter, module, loc, duplicate);
    RankedTensorType flippedType = getSingleCompressedMatrixType(
        context, inputTensorType.getShape(), outputIsCSC, valueType,
        ptrBitWidth, idxBitWidth);
    output = castToTensor(rewriter, module, loc, output, flippedType);

    Value outputPtrs = rewriter.create<sparse_tensor::ToPointersOp>(
        loc, memref1DI64Type, output, c1);
    Value outputIndices = rewriter.create<sparse_tensor::ToIndicesOp>(
        loc, memref1DI64Type, output, c1);
    Value outputValues = rewriter.create<sparse_tensor::ToValuesOp>(
        loc, memref1DValueType, output);

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
    Value colA64 =
        rewriter.create<memref::LoadOp>(loc, inputIndices, ptrLoopIdx);
    Value colA = rewriter.create<arith::IndexCastOp>(loc, colA64, indexType);
    Value colB = rewriter.create<memref::LoadOp>(loc, outputPtrs, colA);
    Value colB1 = rewriter.create<arith::AddIOp>(loc, colB, c1_64);
    rewriter.create<memref::StoreOp>(loc, colB1, outputPtrs, colA);

    rewriter.setInsertionPointAfter(ptrLoop);

    // cumsum the nnz per column to get Bp
    rewriter.create<memref::StoreOp>(loc, c0_64, outputPtrs, ncol);

    scf::ForOp colAccLoop = rewriter.create<scf::ForOp>(loc, c0, ncol, c1);
    Value colAccLoopIdx = colAccLoop.getInductionVar();

    rewriter.setInsertionPointToStart(colAccLoop.getBody());
    Value temp =
        rewriter.create<memref::LoadOp>(loc, outputPtrs, colAccLoopIdx);
    Value cumsum = rewriter.create<memref::LoadOp>(loc, outputPtrs, ncol);
    rewriter.create<memref::StoreOp>(loc, cumsum, outputPtrs, colAccLoopIdx);
    Value cumsum2 = rewriter.create<arith::AddIOp>(loc, cumsum, temp);
    rewriter.create<memref::StoreOp>(loc, cumsum2, outputPtrs, ncol);

    rewriter.setInsertionPointAfter(colAccLoop);

    // copy values
    scf::ForOp outerLoop = rewriter.create<scf::ForOp>(loc, c0, nrow, c1);
    Value rowIdx = outerLoop.getInductionVar();

    rewriter.setInsertionPointToStart(outerLoop.getBody());
    Value row_64 = rewriter.create<arith::IndexCastOp>(loc, rowIdx, int64Type);
    Value j_start_64 = rewriter.create<memref::LoadOp>(loc, inputPtrs, rowIdx);
    Value j_start =
        rewriter.create<arith::IndexCastOp>(loc, j_start_64, indexType);
    Value row_plus1 = rewriter.create<arith::AddIOp>(loc, rowIdx, c1);
    Value j_end_64 = rewriter.create<memref::LoadOp>(loc, inputPtrs, row_plus1);
    Value j_end = rewriter.create<arith::IndexCastOp>(loc, j_end_64, indexType);

    scf::ForOp innerLoop = rewriter.create<scf::ForOp>(loc, j_start, j_end, c1);
    Value jj = innerLoop.getInductionVar();

    rewriter.setInsertionPointToStart(innerLoop.getBody());

    Value col_64 = rewriter.create<memref::LoadOp>(loc, inputIndices, jj);
    Value col = rewriter.create<arith::IndexCastOp>(loc, col_64, indexType);
    Value dest_64 = rewriter.create<memref::LoadOp>(loc, outputPtrs, col);
    Value dest = rewriter.create<arith::IndexCastOp>(loc, dest_64, indexType);
    rewriter.create<memref::StoreOp>(loc, row_64, outputIndices, dest);
    Value axjj = rewriter.create<memref::LoadOp>(loc, inputValues, jj);
    rewriter.create<memref::StoreOp>(loc, axjj, outputValues, dest);

    // Bp[col]++
    Value bp_inc = rewriter.create<memref::LoadOp>(loc, outputPtrs, col);
    Value bp_inc1 = rewriter.create<arith::AddIOp>(loc, bp_inc, c1_64);
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

class LowerCastRewrite : public OpRewritePattern<graphblas::CastOp> {
public:
  using OpRewritePattern<graphblas::CastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::CastOp op,
                                PatternRewriter &rewriter) const override {
    //MLIRContext *context = op.getContext();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    Value input = op.input();
    Type inputType = input.getType();
    Type outputType = op->getResultTypes().front();

    // Shortcut operation if no change
    if (inputType == outputType) {
      rewriter.replaceOp(op, input);
      return success();
    }

    RankedTensorType inputTensorType = inputType.cast<RankedTensorType>();
    //sparse_tensor::SparseTensorEncodingAttr inputSparseEncoding =
    //    sparse_tensor::getSparseTensorEncoding(inputTensorType);
    //unsigned inputPtrBitWidth = inputSparseEncoding.getPointerBitWidth();
    //unsigned inputIdxBitWidth = inputSparseEncoding.getIndexBitWidth();
    Type inputValueType = inputTensorType.getElementType();

    RankedTensorType outputTensorType = outputType.cast<RankedTensorType>();
    //sparse_tensor::SparseTensorEncodingAttr outputSparseEncoding =
    //    sparse_tensor::getSparseTensorEncoding(outputTensorType);
    //unsigned outputPtrBitWidth = outputSparseEncoding.getPointerBitWidth();
    //unsigned outputIdxBitWidth = outputSparseEncoding.getIndexBitWidth();
    Type outputValueType = outputTensorType.getElementType();

    unsigned rank = inputTensorType.getRank();
    Type memref1DIValueType = MemRefType::get({-1}, inputValueType);
    Type memref1DOValueType = MemRefType::get({-1}, outputValueType);

    // Initial constants
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Get the shape as a ValueRange
    ValueRange shape;
    if (rank == 1) {
      Value size = rewriter.create<graphblas::SizeOp>(loc, input);
      shape = ValueRange{size};
    } else {
      Value nrows = rewriter.create<graphblas::NumRowsOp>(loc, input);
      Value ncols = rewriter.create<graphblas::NumColsOp>(loc, input);
      shape = ValueRange{nrows, ncols};
    }

    // Create a new tensor with the correct output value type
    Value output =
        rewriter.create<sparse_tensor::InitOp>(loc, outputType, shape);

    // Make a copy of the input so we can swap the pointers and indices
    Value duplicate = callDupTensor(rewriter, module, loc, input);
    callSwapPointers(rewriter, module, loc, duplicate, output);
    callSwapIndices(rewriter, module, loc, duplicate, output);
    rewriter.create<sparse_tensor::ReleaseOp>(loc, duplicate);

    // Cast values to new dtype
    Value nnz = rewriter.create<graphblas::NumValsOp>(loc, input);
    callResizeValues(rewriter, module, loc, output, nnz);
    Value inputValues = rewriter.create<sparse_tensor::ToValuesOp>(
        loc, memref1DIValueType, input);
    Value outputValues = rewriter.create<sparse_tensor::ToValuesOp>(
        loc, memref1DOValueType, output);
    scf::ParallelOp loop = rewriter.create<scf::ParallelOp>(loc, c0, nnz, c1);
    Value loopIdx = loop.getInductionVars().front();
    {
      rewriter.setInsertionPointToStart(loop.getBody());
      Value val = rewriter.create<memref::LoadOp>(loc, inputValues, loopIdx);
      Value newVal;
      if (auto itype = inputValueType.dyn_cast<IntegerType>()) {
        newVal = llvm::TypeSwitch<Type, Value>(outputValueType)
                     .Case<IntegerType>([&](IntegerType otype) {
                       // int -> int
                       unsigned iBitWidth = itype.getWidth();
                       unsigned oBitWidth = otype.getWidth();
                       if (iBitWidth < oBitWidth)
                         return rewriter
                             .create<arith::ExtSIOp>(loc, outputValueType, val)
                             .getResult();
                       else if (iBitWidth > oBitWidth)
                         return rewriter
                             .create<arith::TruncIOp>(loc, outputValueType, val)
                             .getResult();
                       else
                         return val;
                     })
                     .Case<FloatType>([&](FloatType otype) {
                       // int -> float
                       return rewriter.create<arith::SIToFPOp>(
                           loc, outputValueType, val);
                     });
      } else {
        newVal = llvm::TypeSwitch<Type, Value>(outputValueType)
                     .Case<IntegerType>([&](IntegerType otype) {
                       // float -> int
                       return rewriter.create<arith::FPToSIOp>(
                           loc, outputValueType, val);
                     })
                     .Case<FloatType>([&](FloatType otype) {
                       // float -> float
                       unsigned iBitWidth =
                           inputValueType.dyn_cast<FloatType>().getWidth();
                       unsigned oBitWidth = otype.getWidth();
                       if (iBitWidth < oBitWidth)
                         return rewriter
                             .create<arith::ExtFOp>(loc, outputValueType, val)
                             .getResult();
                       else if (iBitWidth > oBitWidth)
                         return rewriter
                             .create<arith::TruncFOp>(loc, outputValueType, val)
                             .getResult();
                       else
                         return val;
                     });
      }
      rewriter.create<memref::StoreOp>(loc, newVal, outputValues, loopIdx);
      rewriter.setInsertionPointAfter(loop);
    }

    rewriter.replaceOp(op, output);

    cleanupIntermediateTensor(rewriter, module, loc, output);

    return success();
  };
};

class LowerTransposeRewrite : public OpRewritePattern<graphblas::TransposeOp> {
public:
  using OpRewritePattern<graphblas::TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *context = op.getContext();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    Value inputTensor = op.input();
    RankedTensorType inputType =
        inputTensor.getType().dyn_cast<RankedTensorType>();
    llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
    sparse_tensor::SparseTensorEncodingAttr sparseEncoding =
        sparse_tensor::getSparseTensorEncoding(inputType);
    unsigned ptrBitWidth = sparseEncoding.getPointerBitWidth();
    unsigned idxBitWidth = sparseEncoding.getIndexBitWidth();
    Type inputValueType = inputType.getElementType();
    RankedTensorType outputType =
        op->getResultTypes().front().dyn_cast<RankedTensorType>();

    bool inputTypeIsCSR = typeIsCSR(inputType);
    bool outputTypeIsCSR = typeIsCSR(outputType);

    RankedTensorType flippedInputType =
        getSingleCompressedMatrixType(context, inputShape, inputTypeIsCSR,
                                      inputValueType, ptrBitWidth, idxBitWidth);

    // Add a graphblas.convert_layout op if the input and output compression
    // types are the same
    if (inputTypeIsCSR == outputTypeIsCSR) {
      // TODO consider separating this out into its own rewrite pattern
      Value flippedInput = rewriter.create<graphblas::ConvertLayoutOp>(
          loc, flippedInputType, inputTensor);
      Value transposed = rewriter.create<graphblas::TransposeOp>(
          loc, outputType, flippedInput);

      rewriter.replaceOp(op, transposed);

      return success();
    }

    // Cast types
    Value output = callDupTensor(rewriter, module, loc, inputTensor);
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    if (outputTypeIsCSR) {
      callAssignRev(rewriter, module, loc, output, c0, c0);
      callAssignRev(rewriter, module, loc, output, c1, c1);
    } else {
      callAssignRev(rewriter, module, loc, output, c0, c1);
      callAssignRev(rewriter, module, loc, output, c1, c0);
    }
    output = castToPtr8(rewriter, module, loc, output);
    output = castToTensor(rewriter, module, loc, output, flippedInputType);

    // TODO we get an error when we have hard-coded/known sizes at compile time.

    rewriter.replaceOp(op, output);

    cleanupIntermediateTensor(rewriter, module, loc, output);

    return success();
  };
};

struct MatrixSelectOutputWriter {
  MatrixSelectOutputWriter(StringRef _selector, llvm::Optional<Value> _thunk, llvm::Optional<Value> _rngContext)
      : selector(_selector), thunk(_thunk), rngContext(_rngContext){};

  void createConstants(PatternRewriter &rewriter, Location loc) {
    Type int64Type = rewriter.getIntegerType(64);

    c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    c0_64 = rewriter.create<arith::ConstantIntOp>(loc, 0, int64Type);
    c1_64 = rewriter.create<arith::ConstantIntOp>(loc, 1, int64Type);
  }

  void createTensor(PatternRewriter &rewriter, Location loc, ModuleOp module,
                    Value input) {
    RankedTensorType inputType = input.getType().cast<RankedTensorType>();
    Type valueType = inputType.getElementType();
    Type int64Type = rewriter.getIntegerType(64);

    Type memref1DI64Type = MemRefType::get({-1}, int64Type);
    Type memref1DValueType = MemRefType::get({-1}, valueType);

    rank = inputType.getRank();
    tensor = rewriter.create<graphblas::DupOp>(loc, input);
    Value indexPos = (rank == 2 ? c1 : c0);
    if (rank == 2)
      colWise = hasColumnOrdering(inputType);

    Bp = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type,
                                                      tensor, indexPos);
    Bj = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                     tensor, indexPos);
    Bx = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType,
                                                    tensor);
  };

  void createUpdateCurrCount(PatternRewriter &rewriter, Location loc, Value row,
                             Value row_plus1) {
    Value bp_curr_count = rewriter.create<memref::LoadOp>(loc, Bp, row);
    rewriter.create<memref::StoreOp>(loc, bp_curr_count, Bp, row_plus1);
  };

  void createTestAndStore(PatternRewriter &rewriter, Location loc, Value row,
                          Value col, Value val, Value row_plus1, Value col_64) {
    Type indexType = rewriter.getIndexType();

    Value keep;
    if (selector == "triu") {
      keep = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                            colWise ? row : col,
                                            colWise ? col : row);
    } else if (selector == "tril") {
      keep = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                            colWise ? row : col,
                                            colWise ? col : row);
    } else if (selector == "gt") {
      keep = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, val,
                                            thunk.getValue());
    } else if (selector == "ge") {
      keep = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGE, val,
                                            thunk.getValue());
    } else if (selector == "probability") {
      Type f64Type = rewriter.getF64Type();
      SymbolRefAttr random_double = SymbolRefAttr::get(rewriter.getContext(), "random_double");
      // Get a random double between [0, 1)
      CallOp randCall = rewriter.create<mlir::CallOp>(
        loc, random_double, TypeRange{f64Type}, ArrayRef<Value>({rngContext.getValue()})
      );
      Value rand = randCall.getResult(0);
      keep = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, rand, thunk.getValue());
    } else {
      // this should be impossible because of validation
      assert(0);
    }

    scf::IfOp ifKeep =
        rewriter.create<scf::IfOp>(loc, keep, false /* no else region */);

    rewriter.setInsertionPointToStart(ifKeep.thenBlock());

    Value bj_pos_64 = rewriter.create<memref::LoadOp>(loc, Bp, row_plus1);
    Value bj_pos =
        rewriter.create<arith::IndexCastOp>(loc, bj_pos_64, indexType);

    rewriter.create<memref::StoreOp>(loc, col_64, Bj, bj_pos);
    rewriter.create<memref::StoreOp>(loc, val, Bx, bj_pos);

    Value bj_pos_plus1 = rewriter.create<arith::AddIOp>(loc, bj_pos_64, c1_64);
    rewriter.create<memref::StoreOp>(loc, bj_pos_plus1, Bp, row_plus1);

    rewriter.setInsertionPointAfter(ifKeep);
  };

  void createTrimValues(PatternRewriter &rewriter, Location loc,
                        ModuleOp module) {
    Value nnz = rewriter.create<graphblas::NumValsOp>(loc, tensor);

    Value indexPos = (rank == 2 ? c1 : c0);
    callResizeIndex(rewriter, module, loc, tensor, indexPos, nnz);
    callResizeValues(rewriter, module, loc, tensor, nnz);
  };

  StringRef selector;
  llvm::Optional<Value> thunk;
  llvm::Optional<Value> rngContext;

  // frequently used values
  Value tensor;
  Value Bp;
  Value Bj;
  Value Bx;
  unsigned rank;
  bool colWise = false;

  // frequently used constants
  Value c0;
  Value c1;
  Value c0_64;
  Value c1_64;
};

class LowerSelectRewrite : public OpRewritePattern<graphblas::SelectOp> {
public:
  using OpRewritePattern<graphblas::SelectOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::SelectOp op,
                                PatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    Value input = op.input();
    RankedTensorType inputType = input.getType().cast<RankedTensorType>();
    Type valueType = inputType.getElementType();
    Type int64Type = rewriter.getIntegerType(64);
    Type indexType = rewriter.getIndexType();
    Type memref1DI64Type = MemRefType::get({-1}, int64Type);
    Type memref1DValueType = MemRefType::get({-1}, valueType);

    std::string selector = op.selector().str();
    OperandRange thunks = op.thunks();
    llvm::Optional<Value> thunk = llvm::None;
    if (thunks.size() > 0)
      thunk = thunks[0];
    llvm::Optional<Value> rngContext = llvm::None;
    if (thunks.size() > 1)
      rngContext = thunks[1];

    // Initial constants
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Get sparse tensor info
    unsigned rank = inputType.getRank();
    Value nrow;
    if (rank == 2)
      nrow = rewriter.create<graphblas::NumRowsOp>(loc, input);
    else
      // Vectors are stored as a 1xn matrix, so the code works correctly if we
      // assume a single row
      nrow = c1;

    Value indexPos = (rank == 2 ? c1 : c0);
    Value Ap = rewriter.create<sparse_tensor::ToPointersOp>(
        loc, memref1DI64Type, input, indexPos);
    Value Aj = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                           input, indexPos);
    Value Ax = rewriter.create<sparse_tensor::ToValuesOp>(
        loc, memref1DValueType, input);

    MatrixSelectOutputWriter *output =
        new MatrixSelectOutputWriter(selector, thunk, rngContext);

    output->createConstants(rewriter, loc);
    output->createTensor(rewriter, loc, module, input);

    // Loop
    scf::ForOp outerLoop = rewriter.create<scf::ForOp>(loc, c0, nrow, c1);
    Value row = outerLoop.getInductionVar();
    {
        rewriter.setInsertionPointToStart(outerLoop.getBody());
        Value row_plus1 = rewriter.create<arith::AddIOp>(loc, row, c1);

        output->createUpdateCurrCount(rewriter, loc, row, row_plus1);

        Value j_start_64 = rewriter.create<memref::LoadOp>(loc, Ap, row);
        Value j_end_64 = rewriter.create<memref::LoadOp>(loc, Ap, row_plus1);
        Value j_start =
            rewriter.create<arith::IndexCastOp>(loc, j_start_64, indexType);
        Value j_end = rewriter.create<arith::IndexCastOp>(loc, j_end_64, indexType);

        scf::ForOp innerLoop = rewriter.create<scf::ForOp>(loc, j_start, j_end, c1);
        Value jj = innerLoop.getInductionVar();
        {
            rewriter.setInsertionPointToStart(innerLoop.getBody());
            Value col_64 = rewriter.create<memref::LoadOp>(loc, Aj, jj);
            Value col = rewriter.create<arith::IndexCastOp>(loc, col_64, indexType);
            Value val = rewriter.create<memref::LoadOp>(loc, Ax, jj);
            output->createTestAndStore(rewriter, loc, row, col, val, row_plus1, col_64);
            // rewriter.setInsertionPointAfter(innerLoop);
        }

        rewriter.setInsertionPointAfter(outerLoop);
    }

    // trim excess values
    output->createTrimValues(rewriter, loc, module);

    rewriter.replaceOp(op, output->tensor);

    cleanupIntermediateTensor(rewriter, module, loc, output->tensor);

    return success();
  };
};

class LowerReduceToVectorRewrite
    : public OpRewritePattern<graphblas::ReduceToVectorOp> {
public:
  using OpRewritePattern<graphblas::ReduceToVectorOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::ReduceToVectorOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *context = op.getContext();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    Value matrix = op.input();
    StringRef aggregator = op.aggregator();
    int axis = op.axis();

    RankedTensorType matrixType =
        op.input().getType().dyn_cast<RankedTensorType>();
    Type elementType = matrixType.getElementType();

    bool matrixTypeIsCSR = typeIsCSR(matrixType);
    if ((axis == 0 && matrixTypeIsCSR) || (axis == 1 && !matrixTypeIsCSR)) {
      // TODO consider moving this out to its own rewrite pattern

      sparse_tensor::SparseTensorEncodingAttr sparseEncoding =
          sparse_tensor::getSparseTensorEncoding(matrixType);
      unsigned ptrBitWidth = sparseEncoding.getPointerBitWidth();
      unsigned idxBitWidth = sparseEncoding.getIndexBitWidth();

      ArrayRef<int64_t> matrixShape = matrixType.getShape();
      ArrayRef<int64_t> flippedShape =
          ArrayRef<int64_t>{matrixShape[1], matrixShape[0]};
      RankedTensorType flippedMatrixType =
          getSingleCompressedMatrixType(context, flippedShape, matrixTypeIsCSR,
                                        elementType, ptrBitWidth, idxBitWidth);
      Value convertedTensor = rewriter.create<graphblas::ConvertLayoutOp>(
          loc, flippedMatrixType, matrix);
      Type originalVectorType = op->getResultTypes().front();
      Value reducedResult = rewriter.create<graphblas::ReduceToVectorOp>(
          loc, originalVectorType, convertedTensor, aggregator, axis);

      rewriter.replaceOp(op, reducedResult);

      return success();
    }

    Type indexType = rewriter.getIndexType();
    Type int64Type = rewriter.getIntegerType(64);

    Type memref1DValueType = MemRefType::get({-1}, elementType);
    MemRefType memref1DI64Type = MemRefType::get({-1}, int64Type);

    sparse_tensor::SparseTensorEncodingAttr sparseEncoding =
        sparse_tensor::getSparseTensorEncoding(matrixType);
    unsigned pointerBitWidth = sparseEncoding.getPointerBitWidth();
    Type pointerType = rewriter.getIntegerType(pointerBitWidth);
    Type memref1DPointerType = MemRefType::get({-1}, pointerType);

    Value c0_elementType =
        llvm::TypeSwitch<Type, Value>(elementType)
            .Case<IntegerType>([&](IntegerType type) {
              return rewriter.create<ConstantOp>(
                  loc, rewriter.getIntegerAttr(elementType, 0));
            })
            .Case<FloatType>([&](FloatType type) {
              return rewriter.create<ConstantOp>(
                  loc, rewriter.getFloatAttr(elementType, 0.0));
            });
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);

    Value len_dense_dim;
    if (axis == 1)
      len_dense_dim = rewriter.create<graphblas::NumRowsOp>(loc, matrix);
    else
      len_dense_dim = rewriter.create<graphblas::NumColsOp>(loc, matrix);

    Value matrixPointers = rewriter.create<sparse_tensor::ToPointersOp>(
        loc, memref1DPointerType, matrix, c1);

    ValueRange outputShape = {len_dense_dim};
    Type vectorElementType = (aggregator == "argmin" || aggregator == "argmax")
                                 ? int64Type
                                 : elementType;
    RankedTensorType vectorType =
        getCompressedVectorType(context, vectorElementType);
    Value output =
        callNewTensor(rewriter, module, loc, outputShape, vectorType);

    scf::ForOp nnzLoop =
        rewriter.create<scf::ForOp>(loc, c0, len_dense_dim, c1, ValueRange{c0});
    {
      rewriter.setInsertionPointToStart(nnzLoop.getBody());
      Value numNonEmptyRows = nnzLoop.getLoopBody().getArgument(1);
      Value matrixRowIndex = nnzLoop.getInductionVar();
      Value nextMatrixRowIndex =
          rewriter.create<arith::AddIOp>(loc, matrixRowIndex, c1).getResult();
      Value firstPtr64 =
          rewriter.create<memref::LoadOp>(loc, matrixPointers, matrixRowIndex);
      Value secondPtr64 = rewriter.create<memref::LoadOp>(loc, matrixPointers,
                                                          nextMatrixRowIndex);
      Value rowIsEmpty = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, firstPtr64, secondPtr64);
      scf::IfOp ifRowIsEmptyBlock = rewriter.create<scf::IfOp>(
          loc, TypeRange{indexType}, rowIsEmpty, true);
      rewriter.setInsertionPointToStart(ifRowIsEmptyBlock.thenBlock());
      rewriter.create<scf::YieldOp>(loc, ValueRange{numNonEmptyRows});
      rewriter.setInsertionPointToStart(ifRowIsEmptyBlock.elseBlock());
      Value incrementedNumNonEmptyRows =
          rewriter.create<arith::AddIOp>(loc, numNonEmptyRows, c1).getResult();
      rewriter.create<scf::YieldOp>(loc,
                                    ValueRange{incrementedNumNonEmptyRows});
      rewriter.setInsertionPointAfter(ifRowIsEmptyBlock);
      Value updatedNumNonEmptyRows = ifRowIsEmptyBlock.getResult(0);
      rewriter.create<scf::YieldOp>(loc, ValueRange{updatedNumNonEmptyRows});
      rewriter.setInsertionPointAfter(nnzLoop);
    }
    Value outputNNZ = nnzLoop.getResult(0);

    callResizePointers(rewriter, module, loc, output, c0, c2);
    callResizeIndex(rewriter, module, loc, output, c0, outputNNZ);
    callResizeValues(rewriter, module, loc, output, outputNNZ);

    Value outputPointers = rewriter.create<sparse_tensor::ToPointersOp>(
        loc, memref1DPointerType, output, c0);
    Value outputNNZ_i64 =
        rewriter.create<arith::IndexCastOp>(loc, outputNNZ, int64Type);
    rewriter.create<memref::StoreOp>(loc, outputNNZ_i64, outputPointers, c1);

    Value matrixValues = rewriter.create<sparse_tensor::ToValuesOp>(
        loc, memref1DValueType, matrix);

    Value outputIndices = rewriter.create<sparse_tensor::ToIndicesOp>(
        loc, memref1DI64Type, output, c0);
    Type outputValuesType = (aggregator == "argmin" || aggregator == "argmax")
                                ? memref1DI64Type
                                : memref1DValueType;
    Value outputValues = rewriter.create<sparse_tensor::ToValuesOp>(
        loc, outputValuesType, output);

    scf::ForOp reduceLoop =
        rewriter.create<scf::ForOp>(loc, c0, len_dense_dim, c1, ValueRange{c0});
    {
      rewriter.setInsertionPointToStart(reduceLoop.getBody());
      Value outputValuesPosition = reduceLoop.getLoopBody().getArgument(1);
      Value rowIndex = reduceLoop.getInductionVar();
      Value ptr64 =
          rewriter.create<memref::LoadOp>(loc, matrixPointers, rowIndex);
      Value nextRowIndex =
          rewriter.create<arith::AddIOp>(loc, rowIndex, c1).getResult();
      Value nextPtr64 =
          rewriter.create<memref::LoadOp>(loc, matrixPointers, nextRowIndex);

      scf::IfOp ifRowIsNonEmptyBlock;
      if (aggregator == "count") {

        Value ptrDiff = rewriter.create<arith::SubIOp>(loc, nextPtr64, ptr64);

        Value c0_i64 = rewriter.create<ConstantOp>(
            loc, rewriter.getIntegerAttr(int64Type, 0));
        Value rowIsNonEmpty = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ne, ptrDiff, c0_i64);

        ifRowIsNonEmptyBlock = rewriter.create<scf::IfOp>(
            loc, TypeRange{indexType}, rowIsNonEmpty, true);
        {
          rewriter.setInsertionPointToStart(ifRowIsNonEmptyBlock.thenBlock());
          {
            Type valueType =
                output.getType().dyn_cast<RankedTensorType>().getElementType();

            // TODO should we require the output to have the element type be
            // index or some integer type?
            Value rowReduction =
                llvm::TypeSwitch<Type, Value>(valueType)
                    .Case<IntegerType>([&](IntegerType type) {
                      Value ans;
                      if (type.getWidth() < 64)
                        ans = rewriter.create<arith::TruncIOp>(loc, type,
                                                               ptrDiff);
                      else
                        ans =
                            rewriter.create<arith::ExtSIOp>(loc, type, ptrDiff);
                      return ans;
                    })
                    .Case<FloatType>([&](FloatType type) {
                      return rewriter.create<arith::SIToFPOp>(loc, type,
                                                              ptrDiff);
                    });
            rewriter.create<memref::StoreOp>(loc, rowReduction, outputValues,
                                             outputValuesPosition);
            Value rowIndex64 =
                rewriter.create<arith::IndexCastOp>(loc, rowIndex, int64Type);
            rewriter.create<memref::StoreOp>(loc, rowIndex64, outputIndices,
                                             outputValuesPosition);
            Value updatedOutputValuesPosition =
                rewriter.create<arith::AddIOp>(loc, outputValuesPosition, c1)
                    .getResult();
            rewriter.create<scf::YieldOp>(
                loc, ValueRange{updatedOutputValuesPosition});
          }

          rewriter.setInsertionPointToStart(ifRowIsNonEmptyBlock.elseBlock());
          {
            rewriter.create<scf::YieldOp>(loc,
                                          ValueRange{outputValuesPosition});
          }

          rewriter.setInsertionPointAfter(ifRowIsNonEmptyBlock);
        }
      } else if (aggregator == "argmin" || aggregator == "argmax") {
        Value rowIsNonEmpty = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ne, ptr64, nextPtr64);
        ifRowIsNonEmptyBlock = rewriter.create<scf::IfOp>(
            loc, TypeRange{indexType}, rowIsNonEmpty, true);
        {

          rewriter.setInsertionPointAfter(matrixValues.getDefiningOp());
          Value matrixIndices = rewriter.create<sparse_tensor::ToIndicesOp>(
              loc, memref1DI64Type, matrix, c1);

          rewriter.setInsertionPointToStart(ifRowIsNonEmptyBlock.thenBlock());
          {

            Value ptr =
                rewriter.create<arith::IndexCastOp>(loc, ptr64, indexType);
            Value initialExtremum =
                rewriter.create<memref::LoadOp>(loc, matrixValues, ptr);
            Value initialExtremumTensorIndex =
                rewriter.create<memref::LoadOp>(loc, matrixIndices, ptr);
            Value ptrPlusOne =
                rewriter.create<arith::AddIOp>(loc, ptr, c1).getResult();
            Value nextPtr =
                rewriter.create<arith::IndexCastOp>(loc, nextPtr64, indexType);
            scf::ForOp rowAggregationLoop = rewriter.create<scf::ForOp>(
                loc, ptrPlusOne, nextPtr, c1,
                ValueRange{initialExtremum, initialExtremumTensorIndex});
            {
              rewriter.setInsertionPointToStart(rowAggregationLoop.getBody());
              Value currentExtremum =
                  rowAggregationLoop.getLoopBody().getArgument(1);
              Value currentExtremumTensorIndex =
                  rowAggregationLoop.getLoopBody().getArgument(2);

              Value currentPtr = rowAggregationLoop.getInductionVar();
              Value rowValue = rewriter.create<memref::LoadOp>(
                  loc, matrixValues, currentPtr);

              bool useMinimum = aggregator == "argmin";
              Value mustUpdate =
                  llvm::TypeSwitch<Type, Value>(elementType)
                      .Case<IntegerType>([&](IntegerType type) {
                        return rewriter.create<arith::CmpIOp>(
                            loc,
                            useMinimum ? arith::CmpIPredicate::slt
                                       : arith::CmpIPredicate::sgt,
                            rowValue, currentExtremum);
                      })
                      .Case<FloatType>([&](FloatType type) {
                        return rewriter.create<arith::CmpFOp>(
                            loc,
                            useMinimum ? arith::CmpFPredicate::OLT
                                       : arith::CmpFPredicate::OGT,
                            rowValue, currentExtremum);
                      });

              scf::IfOp ifMustUpdateBlock = rewriter.create<scf::IfOp>(
                  loc, TypeRange{elementType, int64Type}, mustUpdate, true);
              {
                rewriter.setInsertionPointToStart(
                    ifMustUpdateBlock.thenBlock());
                Value tensorIndex = rewriter.create<memref::LoadOp>(
                    loc, matrixIndices, currentPtr);
                rewriter.create<scf::YieldOp>(
                    loc, ValueRange{rowValue, tensorIndex});
              }
              {
                rewriter.setInsertionPointToStart(
                    ifMustUpdateBlock.elseBlock());
                rewriter.create<scf::YieldOp>(
                    loc,
                    ValueRange{currentExtremum, currentExtremumTensorIndex});
                rewriter.setInsertionPointAfter(ifMustUpdateBlock);
              }
              Value updatedExtremum = ifMustUpdateBlock.getResult(0);
              Value updatedExtremumTensorIndex = ifMustUpdateBlock.getResult(1);

              rewriter.create<scf::YieldOp>(
                  loc, ValueRange{updatedExtremum, updatedExtremumTensorIndex});
              rewriter.setInsertionPointAfter(rowAggregationLoop);
            }

            Value rowAggregation = rowAggregationLoop.getResult(1);
            rewriter.create<memref::StoreOp>(loc, rowAggregation, outputValues,
                                             outputValuesPosition);
            Value rowIndex64 =
                rewriter.create<arith::IndexCastOp>(loc, rowIndex, int64Type);
            rewriter.create<memref::StoreOp>(loc, rowIndex64, outputIndices,
                                             outputValuesPosition);

            Value updatedOutputValuesPosition =
                rewriter.create<arith::AddIOp>(loc, outputValuesPosition, c1)
                    .getResult();
            rewriter.create<scf::YieldOp>(
                loc, ValueRange{updatedOutputValuesPosition});
          }

          rewriter.setInsertionPointToStart(ifRowIsNonEmptyBlock.elseBlock());
          {
            rewriter.create<scf::YieldOp>(loc,
                                          ValueRange{outputValuesPosition});
          }

          rewriter.setInsertionPointAfter(ifRowIsNonEmptyBlock);
        }
      } else if (aggregator == "plus") {
        Value rowIsNonEmpty = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ne, ptr64, nextPtr64);
        ifRowIsNonEmptyBlock = rewriter.create<scf::IfOp>(
            loc, TypeRange{indexType}, rowIsNonEmpty, true);
        {
          rewriter.setInsertionPointToStart(ifRowIsNonEmptyBlock.thenBlock());
          {
            Value ptr =
                rewriter.create<arith::IndexCastOp>(loc, ptr64, indexType);
            Value nextPtr =
                rewriter.create<arith::IndexCastOp>(loc, nextPtr64, indexType);
            scf::ForOp rowSumLoop = rewriter.create<scf::ForOp>(
                loc, ptr, nextPtr, c1, ValueRange{c0_elementType});
            {
              rewriter.setInsertionPointToStart(rowSumLoop.getBody());
              Value currentSum = rowSumLoop.getLoopBody().getArgument(1);
              Value currentPtr = rowSumLoop.getInductionVar();
              Value rowValue = rewriter.create<memref::LoadOp>(
                  loc, matrixValues, currentPtr);
              Value updatedSum =
                  llvm::TypeSwitch<Type, Value>(elementType)
                      .Case<IntegerType>([&](IntegerType type) {
                        return rewriter
                            .create<arith::AddIOp>(loc, rowValue, currentSum)
                            .getResult();
                      })
                      .Case<FloatType>([&](FloatType type) {
                        return rewriter
                            .create<arith::AddFOp>(loc, rowValue, currentSum)
                            .getResult();
                      });
              rewriter.create<scf::YieldOp>(loc, ValueRange{updatedSum});
              rewriter.setInsertionPointAfter(rowSumLoop);
            }
            Value rowSum = rowSumLoop.getResult(0);
            rewriter.create<memref::StoreOp>(loc, rowSum, outputValues,
                                             outputValuesPosition);
            Value rowIndex64 =
                rewriter.create<arith::IndexCastOp>(loc, rowIndex, int64Type);
            rewriter.create<memref::StoreOp>(loc, rowIndex64, outputIndices,
                                             outputValuesPosition);
            Value updatedOutputValuesPosition =
                rewriter.create<arith::AddIOp>(loc, outputValuesPosition, c1)
                    .getResult();
            rewriter.create<scf::YieldOp>(
                loc, ValueRange{updatedOutputValuesPosition});
          }

          rewriter.setInsertionPointToStart(ifRowIsNonEmptyBlock.elseBlock());
          {
            rewriter.create<scf::YieldOp>(loc,
                                          ValueRange{outputValuesPosition});
          }

          rewriter.setInsertionPointAfter(ifRowIsNonEmptyBlock);
        }
      }

      Value nextOutputValuesPosition = ifRowIsNonEmptyBlock.getResult(0);
      rewriter.create<scf::YieldOp>(loc, ValueRange{nextOutputValuesPosition});

      rewriter.setInsertionPointAfter(reduceLoop);
    }

    rewriter.replaceOp(op, output);

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

    if (aggregator == "count") {
      return rewriteCount(op, rewriter);
    } else if (aggregator == "argmin" or aggregator == "argmax") {
      return rewriteArgMinMax(op, rewriter);
    } else {
      return rewriteStandard(op, rewriter);
    }
  };

private:
  LogicalResult rewriteCount(graphblas::ReduceToScalarOp op,
                             PatternRewriter &rewriter) const {
    Value input = op.input();
    Location loc = op->getLoc();
    Type int64Type = rewriter.getIntegerType(64);

    Value countOp = rewriter.create<graphblas::NumValsOp>(loc, input);
    Value countOp_64 =
        rewriter.create<arith::IndexCastOp>(loc, countOp, int64Type);
    rewriter.replaceOp(op, countOp_64);

    return success();
  }

  LogicalResult rewriteArgMinMax(graphblas::ReduceToScalarOp op,
                                 PatternRewriter &rewriter) const {
    // TODO we get seg faults if given a size 0 vector or a sparse vector with
    // no non-zero values. Probably should return a -1 for these cases.
    Location loc = op->getLoc();
    StringRef aggregator = op.aggregator();

    Value input = op.input();
    RankedTensorType inputType = input.getType().cast<RankedTensorType>();

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Type indexType = rewriter.getIndexType();
    Type int64Type = rewriter.getIntegerType(64);
    Type memref1DI64Type = MemRefType::get({-1}, int64Type);

    Value pointers = rewriter.create<sparse_tensor::ToPointersOp>(
        loc, memref1DI64Type, input, c0);
    Value endPosition64 = rewriter.create<memref::LoadOp>(loc, pointers, c1);
    Value endPosition =
        rewriter.create<arith::IndexCastOp>(loc, endPosition64, indexType);

    Type inputElementType = inputType.getElementType();
    Type memref1DValueType = MemRefType::get({-1}, inputElementType);
    Value values = rewriter.create<sparse_tensor::ToValuesOp>(
        loc, memref1DValueType, input);

    Value initialExtremum = rewriter.create<memref::LoadOp>(loc, values, c0);

    scf::ForOp loop = rewriter.create<scf::ForOp>(
        loc, c1, endPosition, c1, ValueRange{initialExtremum, c0});
    Value currentValuePosition = loop.getInductionVar();
    Value currentExtremum = loop.getLoopBody().getArgument(1);
    Value currentExtremumPosition = loop.getLoopBody().getArgument(2);
    rewriter.setInsertionPointToStart(loop.getBody());

    Value currentValue =
        rewriter.create<memref::LoadOp>(loc, values, currentValuePosition);
    bool useMinimum = aggregator == "argmin";
    Value replace = llvm::TypeSwitch<Type, Value>(inputElementType)
                        .Case<IntegerType>([&](IntegerType type) {
                          return rewriter.create<arith::CmpIOp>(
                              loc,
                              useMinimum ? arith::CmpIPredicate::slt
                                         : arith::CmpIPredicate::sgt,
                              currentValue, currentExtremum);
                        })
                        .Case<FloatType>([&](FloatType type) {
                          return rewriter.create<arith::CmpFOp>(
                              loc,
                              useMinimum ? arith::CmpFPredicate::OLT
                                         : arith::CmpFPredicate::OGT,
                              currentValue, currentExtremum);
                        });

    scf::IfOp ifBlock = rewriter.create<scf::IfOp>(
        loc, TypeRange{inputElementType, indexType}, replace, true);
    rewriter.setInsertionPointToStart(ifBlock.thenBlock());
    rewriter.create<scf::YieldOp>(
        loc, ValueRange{currentValue, currentValuePosition});
    rewriter.setInsertionPointToStart(ifBlock.elseBlock());
    rewriter.create<scf::YieldOp>(
        loc, ValueRange{currentExtremum, currentExtremumPosition});
    rewriter.setInsertionPointAfter(ifBlock);

    Value nextExtremum = ifBlock.getResult(0);
    Value nextExtremumPosition = ifBlock.getResult(1);
    rewriter.create<scf::YieldOp>(
        loc, ValueRange{nextExtremum, nextExtremumPosition});

    rewriter.setInsertionPointAfter(loop);

    Value finalExtremumPosition = loop.getResult(1);
    Value indices = rewriter.create<sparse_tensor::ToIndicesOp>(
        loc, memref1DI64Type, input, c0);
    Value argExtremum =
        rewriter.create<memref::LoadOp>(loc, indices, finalExtremumPosition);
    rewriter.replaceOp(op, argExtremum);

    return success();
  }

  LogicalResult rewriteStandard(graphblas::ReduceToScalarOp op,
                                PatternRewriter &rewriter) const {
    Value input = op.input();
    StringRef aggregator = op.aggregator();
    Location loc = op->getLoc();

    RankedTensorType operandType =
        op.input().getType().cast<RankedTensorType>();
    Type valueType = operandType.getElementType();

    graphblas::ReduceToScalarGenericOp newReduceOp =
        rewriter.create<graphblas::ReduceToScalarGenericOp>(
            loc, op->getResultTypes(), input, 2);
    if (aggregator == "plus") {
      // Insert agg identity block
      Region &aggIdentityRegion = newReduceOp.getRegion(0);
      /*Block *aggIdentityBlock = */ rewriter.createBlock(&aggIdentityRegion,
                                                          {}, {});

      Value aggIdentity = llvm::TypeSwitch<Type, Value>(valueType)
                              .Case<IntegerType>([&](IntegerType type) {
                                return rewriter.create<arith::ConstantIntOp>(
                                    loc, 0, type.getWidth());
                              })
                              .Case<FloatType>([&](FloatType type) {
                                return rewriter.create<arith::ConstantFloatOp>(
                                    loc, APFloat(0.0), type);
                              });
      rewriter.create<graphblas::YieldOp>(
          loc, graphblas::YieldKind::AGG_IDENTITY, aggIdentity);

      // Insert agg block
      Region &aggRegion = newReduceOp.getRegion(1);
      Block *aggBlock =
          rewriter.createBlock(&aggRegion, {}, {valueType, valueType});
      Value lhs = aggBlock->getArgument(0);
      Value rhs = aggBlock->getArgument(1);

      Value aggResult =
          llvm::TypeSwitch<Type, Value>(valueType)
              .Case<IntegerType>([&](IntegerType type) {
                return rewriter.create<arith::AddIOp>(loc, lhs, rhs)
                    .getResult();
              })
              .Case<FloatType>([&](FloatType type) {
                return rewriter.create<arith::AddFOp>(loc, lhs, rhs)
                    .getResult();
              });
      rewriter.create<graphblas::YieldOp>(loc, graphblas::YieldKind::AGG,
                                          aggResult);

      rewriter.replaceOp(op, newReduceOp.getResult());
    } else {
      return op.emitError("\"" + aggregator +
                          "\" is not a supported aggregator.");
    }

    return success();
  }
};

class LowerReduceToScalarGenericRewrite
    : public OpRewritePattern<graphblas::ReduceToScalarGenericOp> {
public:
  using OpRewritePattern<graphblas::ReduceToScalarGenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::ReduceToScalarGenericOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.input();
    Location loc = op->getLoc();

    RankedTensorType operandType =
        op.input().getType().dyn_cast<RankedTensorType>();
    Type valueType = operandType.getElementType();

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

    // insert agg identity
    rewriter.mergeBlocks(extBlocks.aggIdentity, rewriter.getBlock(), {});
    graphblas::YieldOp aggIdentityYield =
        llvm::dyn_cast_or_null<graphblas::YieldOp>(
            rewriter.getBlock()->getTerminator());
    Value c0Accumulator = aggIdentityYield.values().front();
    rewriter.eraseOp(aggIdentityYield);

    // initial constants
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Get sparse tensor info
    MemRefType memref1DValueType = MemRefType::get({-1}, valueType);

    sparse_tensor::ToValuesOp inputValues =
        rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType,
                                                   input);

    Value nnz = rewriter.create<graphblas::NumValsOp>(loc, input);

    // begin loop
    scf::ParallelOp valueLoop =
        rewriter.create<scf::ParallelOp>(loc, c0, nnz, c1, c0Accumulator);
    ValueRange valueLoopIdx = valueLoop.getInductionVars();

    rewriter.setInsertionPointToStart(valueLoop.getBody());
    memref::LoadOp y =
        rewriter.create<memref::LoadOp>(loc, inputValues, valueLoopIdx);

    scf::ReduceOp reducer = rewriter.create<scf::ReduceOp>(loc, y);
    BlockArgument lhs = reducer.getRegion().getArgument(0);
    BlockArgument rhs = reducer.getRegion().getArgument(1);

    rewriter.setInsertionPointToStart(&reducer.getRegion().front());

    rewriter.mergeBlocks(extBlocks.agg, rewriter.getBlock(), {lhs, rhs});
    graphblas::YieldOp aggYield = llvm::dyn_cast_or_null<graphblas::YieldOp>(
        rewriter.getBlock()->getTerminator());
    Value result = aggYield.values().front();
    rewriter.eraseOp(aggYield);

    rewriter.create<scf::ReduceReturnOp>(loc, result);

    rewriter.setInsertionPointAfter(reducer);

    rewriter.replaceOp(op, valueLoop.getResult(0));

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
                                                   input, 1);

    // Insert transformOut block
    Region &transformOutRegion = newApplyOp.getRegion(0);
    Block *transformOutBlock =
        rewriter.createBlock(&transformOutRegion, {}, {valueType});

    Value val = transformOutBlock->getArgument(0);

    Value transformResult;
    if (apply_operator == "min") {

      Value cmp = llvm::TypeSwitch<Type, Value>(valueType)
                      .Case<IntegerType>([&](IntegerType type) {
                        // http://graphics.stanford.edu/~seander/bithacks.html#IntegerMinOrMax
                        // says this is best
                        return rewriter.create<arith::CmpIOp>(
                            loc, arith::CmpIPredicate::slt, val, thunk);
                      })
                      .Case<FloatType>([&](FloatType type) {
                        return rewriter.create<arith::CmpFOp>(
                            loc, arith::CmpFPredicate::OLT, val, thunk);
                      });
      transformResult = rewriter.create<SelectOp>(loc, cmp, val, thunk);

    } else if (apply_operator == "div") {

      bool thunkIsLeft = thunk == op.left();

      transformResult =
          llvm::TypeSwitch<Type, Value>(valueType)
              .Case<IntegerType>([&](IntegerType type) {
                Value quotient =
                    thunkIsLeft
                        ? rewriter.create<arith::DivSIOp>(loc, thunk, val)
                        : rewriter.create<arith::DivSIOp>(loc, val, thunk);
                return quotient;
              })
              .Case<FloatType>([&](FloatType type) {
                Value quotient =
                    thunkIsLeft
                        ? rewriter.create<arith::DivFOp>(loc, thunk, val)
                        : rewriter.create<arith::DivFOp>(loc, val, thunk);
                return quotient;
              });

    } else if (apply_operator == "fill") {

      // Always fill with the thunk, regardless of its position (left or right)
      transformResult = thunk;

    } else if (apply_operator == "minv") {

      transformResult =
          llvm::TypeSwitch<Type, Value>(valueType)
              .Case<IntegerType>([&](IntegerType type) {
                // TODO we're missing python tests for all ops when given
                // integer-typed tensors
                unsigned bitWidth = type.getWidth();
                Value shiftAmount = rewriter.create<ConstantOp>(
                    loc, rewriter.getIntegerAttr(type, bitWidth - 1));
                Value mask =
                    rewriter.create<arith::ShRSIOp>(loc, val, shiftAmount);
                Value maskPlusVal =
                    rewriter.create<arith::AddIOp>(loc, mask, val);
                Value absVal =
                    rewriter.create<arith::XOrIOp>(loc, mask, maskPlusVal);
                Value c1_type = rewriter.create<arith::ConstantOp>(
                    loc, rewriter.getIntegerAttr(type, 1));
                Value absValIsOne_i1 = rewriter.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::eq, absVal, c1_type);
                Value absValIsOne_type =
                    rewriter.create<arith::ExtSIOp>(loc, type, absValIsOne_i1);
                Value multipicativeInverse =
                    rewriter.create<arith::AndIOp>(loc, absValIsOne_type, val);
                return multipicativeInverse;
              })
              .Case<FloatType>([&](FloatType type) {
                // TODO is there a faster way? e.g. magic with logs or
                // exponents?
                Value c1_type = rewriter.create<arith::ConstantFloatOp>(
                    loc, APFloat(1.0), type);
                Value multipicativeInverse =
                    rewriter.create<arith::DivFOp>(loc, c1_type, val);
                return multipicativeInverse;
              });

    } else if (apply_operator == "ainv") {

      transformResult =
          llvm::TypeSwitch<Type, Value>(valueType)
              .Case<IntegerType>([&](IntegerType type) {
                Value c0_type = rewriter.create<ConstantOp>(
                    loc, rewriter.getIntegerAttr(type, 0));
                Value additiveInverse =
                    rewriter.create<arith::SubIOp>(loc, c0_type, val);
                return additiveInverse;
              })
              .Case<FloatType>([&](FloatType type) {
                Value additiveInverse =
                    rewriter.create<arith::NegFOp>(loc, val);
                return additiveInverse;
              });

    } else if (apply_operator == "abs") {

      transformResult =
          llvm::TypeSwitch<Type, Value>(valueType)
              .Case<IntegerType>([&](IntegerType type) {
                // http://graphics.stanford.edu/~seander/bithacks.html#IntegerAbs
                unsigned bitWidth = type.getWidth();
                Value shiftAmount = rewriter.create<ConstantOp>(
                    loc, rewriter.getIntegerAttr(type, bitWidth - 1));
                Value mask =
                    rewriter.create<arith::ShRSIOp>(loc, val, shiftAmount);
                Value maskPlusVal =
                    rewriter.create<arith::AddIOp>(loc, mask, val);
                Value absVal =
                    rewriter.create<arith::XOrIOp>(loc, mask, maskPlusVal);
                return absVal;
              })
              .Case<FloatType>([&](FloatType type) {
                return rewriter.create<math::AbsOp>(loc, val);
              });

    } else {
      return op.emitError("\"" + apply_operator +
                          "\" is not a supported apply_operator.");
    };

    rewriter.create<graphblas::YieldOp>(
        loc, graphblas::YieldKind::TRANSFORM_OUT, transformResult);

    rewriter.replaceOp(op, newApplyOp.getResult());

    return success();
  };
};

class LowerApplyGenericRewrite
    : public OpRewritePattern<graphblas::ApplyGenericOp> {
public:
  using OpRewritePattern<graphblas::ApplyGenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::ApplyGenericOp op,
                                PatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    Type valueType =
        op.input().getType().dyn_cast<RankedTensorType>().getElementType();
    Type memref1DValueType = MemRefType::get({-1}, valueType);

    Value inputTensor = op.input();

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

    // Initial constants
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Get sparse tensor info
    Value output = rewriter.create<graphblas::DupOp>(loc, inputTensor);
    Value inputValues = rewriter.create<sparse_tensor::ToValuesOp>(
        loc, memref1DValueType, inputTensor);
    Value outputValues = rewriter.create<sparse_tensor::ToValuesOp>(
        loc, memref1DValueType, output);

    Value nnz = rewriter.create<graphblas::NumValsOp>(loc, inputTensor);

    // Loop over values
    scf::ParallelOp valueLoop =
        rewriter.create<scf::ParallelOp>(loc, c0, nnz, c1);
    ValueRange valueLoopIdx = valueLoop.getInductionVars();

    rewriter.setInsertionPointToStart(valueLoop.getBody());
    Value val = rewriter.create<memref::LoadOp>(loc, inputValues, valueLoopIdx);

    // scf::ParallelOp automatically gets an empty scf.yield at the end which we
    // need to insert before
    Operation *scfYield = valueLoop.getBody()->getTerminator();

    // insert transformOut block
    graphblas::YieldOp transformOutYield =
        llvm::dyn_cast_or_null<graphblas::YieldOp>(
            extBlocks.transformOut->getTerminator());

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

    if (failed(populateSemiringRegions(rewriter, loc, semiring, valueType,
                                       newMultOp.getRegions().slice(0, 3))))
      return failure();

    rewriter.setInsertionPointAfter(newMultOp);

    rewriter.replaceOp(op, newMultOp.getResult());

    return success();
  };
};

class LowerMatrixMultiplyGenericRewrite
    : public OpRewritePattern<graphblas::MatrixMultiplyGenericOp> {
public:
  using OpRewritePattern<graphblas::MatrixMultiplyGenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::MatrixMultiplyGenericOp op,
                                PatternRewriter &rewriter) const override {
    // Required blocks
    RegionRange extensions = op.extensions();
    ExtensionBlocks extBlocks;
    std::set<graphblas::YieldKind> required = {
        graphblas::YieldKind::ADD_IDENTITY, graphblas::YieldKind::ADD,
        graphblas::YieldKind::MULT};
    std::set<graphblas::YieldKind> optional = {
        graphblas::YieldKind::TRANSFORM_OUT};
    LogicalResult extractResult =
        extBlocks.extractBlocks(op, extensions, required, optional);

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
  LogicalResult
  rewriteMatrixMatrixMultiplication(graphblas::MatrixMultiplyGenericOp op,
                                    PatternRewriter &rewriter,
                                    ExtensionBlocks extBlocks) const {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    // Inputs
    Value A = op.a();
    Value B = op.b();
    Value mask = op.mask();
    bool isMaskComplement = op.mask_complement();

    // Types
    Type indexType = rewriter.getIndexType();
    Type int64Type = rewriter.getIntegerType(64);
    Type valueType =
        op.getResult().getType().dyn_cast<RankedTensorType>().getElementType();

    MemRefType memref1DI64Type = MemRefType::get({-1}, int64Type);
    MemRefType memref1DValueType = MemRefType::get({-1}, valueType);

    // Initial constants
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value ci0 = rewriter.create<arith::ConstantIntOp>(loc, 0, int64Type);

    Value nrow = rewriter.create<graphblas::NumRowsOp>(loc, A);
    Value ncol = rewriter.create<graphblas::NumColsOp>(loc, B);
    Value nk = rewriter.create<graphblas::NumColsOp>(
        loc, A); // guaranteed equal to B.rows
    Value nrow_plus_one = rewriter.create<arith::AddIOp>(loc, nrow, c1);

    Value C = callEmptyLike(rewriter, module, loc, A);
    callResizeDim(rewriter, module, loc, C, c0, nrow);
    callResizeDim(rewriter, module, loc, C, c1, ncol);
    callResizePointers(rewriter, module, loc, C, c1, nrow_plus_one);

    Value Ap = rewriter.create<sparse_tensor::ToPointersOp>(
        loc, memref1DI64Type, A, c1);
    Value Aj = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                           A, c1);
    Value Ax =
        rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, A);
    Value Bp = rewriter.create<sparse_tensor::ToPointersOp>(
        loc, memref1DI64Type, B, c1);
    Value Bi = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                           B, c1);
    Value Bx =
        rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, B);
    Value Cp = rewriter.create<sparse_tensor::ToPointersOp>(
        loc, memref1DI64Type, C, c1);
    Value Mp, Mj;
    if (mask) {
      Mp = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type,
                                                        mask, c1);
      Mj = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                       mask, c1);
    }

    // 1st pass
    //   Compute the number of nonzero entries per row.
    //   Store results in Cp
    //   The rows in A are the fixed elements, while the columns of B are the
    //   iteration element
    scf::ParallelOp rowLoop1 =
        rewriter.create<scf::ParallelOp>(loc, c0, nrow, c1);
    Value row = rowLoop1.getInductionVars().front();
    rewriter.setInsertionPointToStart(rowLoop1.getBody());

    Value colStart64 = rewriter.create<memref::LoadOp>(loc, Ap, row);
    Value rowPlus1 = rewriter.create<arith::AddIOp>(loc, row, c1);
    Value colEnd64 = rewriter.create<memref::LoadOp>(loc, Ap, rowPlus1);
    Value cmpColSame = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, colStart64, colEnd64);

    scf::IfOp ifBlock_rowTotal =
        rewriter.create<scf::IfOp>(loc, int64Type, cmpColSame, true);
    // if cmpColSame
    rewriter.setInsertionPointToStart(ifBlock_rowTotal.thenBlock());
    rewriter.create<scf::YieldOp>(loc, ci0);

    // else
    rewriter.setInsertionPointToStart(ifBlock_rowTotal.elseBlock());
    Value colStart =
        rewriter.create<arith::IndexCastOp>(loc, colStart64, indexType);
    Value colEnd =
        rewriter.create<arith::IndexCastOp>(loc, colEnd64, indexType);
    Value total;
    if (mask) {
      Value mcolStart64 = rewriter.create<memref::LoadOp>(loc, Mp, row);
      Value mcolEnd64 = rewriter.create<memref::LoadOp>(loc, Mp, rowPlus1);
      Value mcolStart =
          rewriter.create<arith::IndexCastOp>(loc, mcolStart64, indexType);
      Value mcolEnd =
          rewriter.create<arith::IndexCastOp>(loc, mcolEnd64, indexType);
      if (isMaskComplement) {
        ValueRange mcResult =
            buildMaskComplement(rewriter, loc, ncol, Mj, mcolStart, mcolEnd);
        Value maskComplement = mcResult[0];
        Value mcSize = mcResult[1];
        total = computeNumOverlaps(rewriter, loc, nk, Aj, colStart, colEnd, Bp,
                                   Bi, maskComplement, c0, mcSize, valueType);
        rewriter.create<memref::DeallocOp>(loc, maskComplement);
      } else {
        total = computeNumOverlaps(rewriter, loc, nk, Aj, colStart, colEnd, Bp,
                                   Bi, Mj, mcolStart, mcolEnd, valueType);
      }
    } else {
      total = computeNumOverlaps(rewriter, loc, nk, Aj, colStart, colEnd, Bp,
                                 Bi, nullptr, c0, ncol, valueType);
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
    scf::ForOp rowLoop2 = rewriter.create<scf::ForOp>(loc, c0, nrow, c1);
    Value cs_i = rowLoop2.getInductionVar();
    rewriter.setInsertionPointToStart(rowLoop2.getBody());

    Value csTemp = rewriter.create<memref::LoadOp>(loc, Cp, cs_i);
    Value cumsum = rewriter.create<memref::LoadOp>(loc, Cp, nrow);
    rewriter.create<memref::StoreOp>(loc, cumsum, Cp, cs_i);
    Value cumsum2 = rewriter.create<arith::AddIOp>(loc, cumsum, csTemp);
    rewriter.create<memref::StoreOp>(loc, cumsum2, Cp, nrow);

    // end row loop
    rewriter.setInsertionPointAfter(rowLoop2);

    Value nnz = rewriter.create<graphblas::NumValsOp>(loc, C);
    callResizeIndex(rewriter, module, loc, C, c1, nnz);
    callResizeValues(rewriter, module, loc, C, nnz);
    Value Cj = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                           C, c1);
    Value Cx =
        rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, C);

    // 3rd pass
    //   In parallel over the rows,
    //   compute the nonzero columns and associated values.
    //   Store in Cj and Cx
    //   The rows in A are the fixed elements, while the columns of B are the
    //   iteration element
    scf::ParallelOp rowLoop3 =
        rewriter.create<scf::ParallelOp>(loc, c0, nrow, c1);
    row = rowLoop3.getInductionVars().front();
    rewriter.setInsertionPointToStart(rowLoop3.getBody());

    rowPlus1 = rewriter.create<arith::AddIOp>(loc, row, c1);
    Value cpStart64 = rewriter.create<memref::LoadOp>(loc, Cp, row);
    Value cpEnd64 = rewriter.create<memref::LoadOp>(loc, Cp, rowPlus1);
    Value cmp_cpDifferent = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, cpStart64, cpEnd64);
    scf::IfOp ifBlock_cmpDiff =
        rewriter.create<scf::IfOp>(loc, cmp_cpDifferent);
    rewriter.setInsertionPointToStart(ifBlock_cmpDiff.thenBlock());

    Value baseIndex64 = rewriter.create<memref::LoadOp>(loc, Cp, row);
    Value baseIndex =
        rewriter.create<arith::IndexCastOp>(loc, baseIndex64, indexType);

    colStart64 = rewriter.create<memref::LoadOp>(loc, Ap, row);
    colEnd64 = rewriter.create<memref::LoadOp>(loc, Ap, rowPlus1);
    colStart = rewriter.create<arith::IndexCastOp>(loc, colStart64, indexType);
    colEnd = rewriter.create<arith::IndexCastOp>(loc, colEnd64, indexType);

    if (mask) {
      Value mcolStart64 = rewriter.create<memref::LoadOp>(loc, Mp, row);
      Value mcolEnd64 = rewriter.create<memref::LoadOp>(loc, Mp, rowPlus1);
      Value mcolStart =
          rewriter.create<arith::IndexCastOp>(loc, mcolStart64, indexType);
      Value mcolEnd =
          rewriter.create<arith::IndexCastOp>(loc, mcolEnd64, indexType);
      if (isMaskComplement) {
        ValueRange mcResult =
            buildMaskComplement(rewriter, loc, ncol, Mj, mcolStart, mcolEnd);
        Value maskComplement = mcResult[0];
        Value mcSize = mcResult[1];
        computeInnerProduct(rewriter, loc, nk, row, Aj, Ax, colStart, colEnd,
                            Bp, Bi, Bx, maskComplement, c0, mcSize, valueType,
                            extBlocks, Cj, Cx, baseIndex, false);
        rewriter.create<memref::DeallocOp>(loc, maskComplement);
      } else {
        computeInnerProduct(rewriter, loc, nk, row, Aj, Ax, colStart, colEnd,
                            Bp, Bi, Bx, Mj, mcolStart, mcolEnd, valueType,
                            extBlocks, Cj, Cx, baseIndex, false);
      }
    } else {
      computeInnerProduct(rewriter, loc, nk, row, Aj, Ax, colStart, colEnd, Bp,
                          Bi, Bx, nullptr, c0, ncol, valueType, extBlocks, Cj,
                          Cx, baseIndex, false);
    }

    // end if cmpDiff
    rewriter.setInsertionPointAfter(ifBlock_cmpDiff);

    // end row loop
    rewriter.setInsertionPointAfter(rowLoop3);

    rewriter.replaceOp(op, C);

    cleanupIntermediateTensor(rewriter, module, loc, C);

    return success();
  }

  LogicalResult
  rewriteMatrixVectorMultiplication(graphblas::MatrixMultiplyGenericOp op,
                                    PatternRewriter &rewriter,
                                    ExtensionBlocks extBlocks) const {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    // Inputs
    Value A = op.a();
    Value B = op.b();
    Value mask = op.mask();
    bool isMaskComplement = op.mask_complement();

    // Types
    Type indexType = rewriter.getIndexType();
    Type int64Type = rewriter.getIntegerType(64);
    Type valueType =
        op.getResult().getType().dyn_cast<RankedTensorType>().getElementType();

    MemRefType memref1DI64Type = MemRefType::get({-1}, int64Type);
    MemRefType memref1DValueType = MemRefType::get({-1}, valueType);

    // Initial constants
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    Value ci0 = rewriter.create<arith::ConstantIntOp>(loc, 0, int64Type);

    Value size = rewriter.create<graphblas::NumRowsOp>(loc, A);
    Value nk = rewriter.create<graphblas::SizeOp>(loc, B);
    // TODO: how do I check nk == nk_check and raise an exception if they don't
    // match? Value nk_check = rewriter.create<graphblas::NumColsOp>(loc, A);

    Value C = callEmptyLike(rewriter, module, loc, B);
    callResizeDim(rewriter, module, loc, C, c0, size);
    callResizePointers(rewriter, module, loc, C, c0, c2);

    Value Ap = rewriter.create<sparse_tensor::ToPointersOp>(
        loc, memref1DI64Type, A, c1);
    Value Aj = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                           A, c1);
    Value Ax =
        rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, A);
    Value Bp = rewriter.create<sparse_tensor::ToPointersOp>(
        loc, memref1DI64Type, B, c0);
    Value Bi = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                           B, c0);
    Value Bx =
        rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, B);
    Value Cp = rewriter.create<sparse_tensor::ToPointersOp>(
        loc, memref1DI64Type, C, c0);
    Value Mp, Mi, maskStart, maskEnd;
    if (mask) {
      Mp = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type,
                                                        mask, c0);
      Mi = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                       mask, c0);
      Value maskStart64 = rewriter.create<memref::LoadOp>(loc, Mp, c0);
      Value maskEnd64 = rewriter.create<memref::LoadOp>(loc, Mp, c1);
      maskStart =
          rewriter.create<arith::IndexCastOp>(loc, maskStart64, indexType);
      maskEnd = rewriter.create<arith::IndexCastOp>(loc, maskEnd64, indexType);
    }

    // 1st pass
    //   Compute the number of nonzero entries in the result
    //   Store results in Cp
    //   The vector B is the fixed element, while the rows of A are the
    //   iteration element
    Value fixedIndexEnd64 = rewriter.create<memref::LoadOp>(loc, Bp, c1);
    Value fixedIndexEnd =
        rewriter.create<arith::IndexCastOp>(loc, fixedIndexEnd64, indexType);
    Value cmpColSame = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, c0, fixedIndexEnd);

    scf::IfOp ifBlock_rowTotal =
        rewriter.create<scf::IfOp>(loc, int64Type, cmpColSame, true);
    // if cmpColSame
    rewriter.setInsertionPointToStart(ifBlock_rowTotal.thenBlock());
    rewriter.create<scf::YieldOp>(loc, ci0);

    // else
    rewriter.setInsertionPointToStart(ifBlock_rowTotal.elseBlock());
    Value total;
    if (mask) {
      if (isMaskComplement) {
        ValueRange mcResult =
            buildMaskComplement(rewriter, loc, size, Mi, maskStart, maskEnd);
        Value maskComplement = mcResult[0];
        Value mcSize = mcResult[1];
        total = computeNumOverlaps(rewriter, loc, nk, Bi, c0, fixedIndexEnd, Ap,
                                   Aj, maskComplement, c0, mcSize, valueType);
        rewriter.create<memref::DeallocOp>(loc, maskComplement);
      } else {
        total = computeNumOverlaps(rewriter, loc, nk, Bi, c0, fixedIndexEnd, Ap,
                                   Aj, Mi, maskStart, maskEnd, valueType);
      }
    } else {
      total = computeNumOverlaps(rewriter, loc, nk, Bi, c0, fixedIndexEnd, Ap,
                                 Aj, nullptr, c0, size, valueType);
    }
    rewriter.create<scf::YieldOp>(loc, total);

    // end if cmpColSame
    rewriter.setInsertionPointAfter(ifBlock_rowTotal);
    Value nnzTotal = ifBlock_rowTotal.getResult(0);
    Value nnz = rewriter.create<arith::IndexCastOp>(loc, nnzTotal, indexType);
    rewriter.create<memref::StoreOp>(loc, nnzTotal, Cp, c1);

    callResizeIndex(rewriter, module, loc, C, c0, nnz);
    callResizeValues(rewriter, module, loc, C, nnz);
    Value Ci = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                           C, c0);
    Value Cx =
        rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, C);

    // 2nd pass
    //   Compute the nonzero values.
    //   Store in Ci and Cx
    //   The vector B is the fixed element, while the rows of A are the
    //   iteration element
    Value cmp_cpDifferent =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, c0, nnz);
    scf::IfOp ifBlock_cmpDiff =
        rewriter.create<scf::IfOp>(loc, cmp_cpDifferent);
    rewriter.setInsertionPointToStart(ifBlock_cmpDiff.thenBlock());

    if (mask) {
      if (isMaskComplement) {
        ValueRange mcResult =
            buildMaskComplement(rewriter, loc, size, Mi, maskStart, maskEnd);
        Value maskComplement = mcResult[0];
        Value mcSize = mcResult[1];
        computeInnerProduct(rewriter, loc, nk, c0, Bi, Bx, c0, fixedIndexEnd,
                            Ap, Aj, Ax, maskComplement, c0, mcSize, valueType,
                            extBlocks, Ci, Cx, c0, true);
        rewriter.create<memref::DeallocOp>(loc, maskComplement);
      } else {
        computeInnerProduct(rewriter, loc, nk, c0, Bi, Bx, c0, fixedIndexEnd,
                            Ap, Aj, Ax, Mi, maskStart, maskEnd, valueType,
                            extBlocks, Ci, Cx, c0, true);
      }
    } else {
      computeInnerProduct(rewriter, loc, nk, c0, Bi, Bx, c0, fixedIndexEnd, Ap,
                          Aj, Ax, nullptr, c0, size, valueType, extBlocks, Ci,
                          Cx, c0, true);
    }

    // end if cmpDiff
    rewriter.setInsertionPointAfter(ifBlock_cmpDiff);

    rewriter.replaceOp(op, C);

    cleanupIntermediateTensor(rewriter, module, loc, C);

    return success();
  }

  LogicalResult
  rewriteVectorMatrixMultiplication(graphblas::MatrixMultiplyGenericOp op,
                                    PatternRewriter &rewriter,
                                    ExtensionBlocks extBlocks) const {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    // Inputs
    Value A = op.a();
    Value B = op.b();
    Value mask = op.mask();
    bool isMaskComplement = op.mask_complement();

    // Types
    Type indexType = rewriter.getIndexType();
    Type int64Type = rewriter.getIntegerType(64);
    Type valueType =
        op.getResult().getType().dyn_cast<RankedTensorType>().getElementType();

    MemRefType memref1DI64Type = MemRefType::get({-1}, int64Type);
    MemRefType memref1DValueType = MemRefType::get({-1}, valueType);

    // Initial constants
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    Value ci0 = rewriter.create<arith::ConstantIntOp>(loc, 0, int64Type);

    Value size = rewriter.create<graphblas::NumColsOp>(loc, B);
    Value nk = rewriter.create<graphblas::SizeOp>(
        loc, A); // guaranteed equal to B.rows

    Value C = callEmptyLike(rewriter, module, loc, A);
    callResizeDim(rewriter, module, loc, C, c0, size);
    callResizePointers(rewriter, module, loc, C, c0, c2);

    Value Ap = rewriter.create<sparse_tensor::ToPointersOp>(
        loc, memref1DI64Type, A, c0);
    Value Ai = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                           A, c0);
    Value Ax =
        rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, A);
    Value Bp = rewriter.create<sparse_tensor::ToPointersOp>(
        loc, memref1DI64Type, B, c1);
    Value Bi = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                           B, c1);
    Value Bx =
        rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, B);
    Value Cp = rewriter.create<sparse_tensor::ToPointersOp>(
        loc, memref1DI64Type, C, c0);
    Value Mp, Mi, maskStart, maskEnd;
    if (mask) {
      Mp = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type,
                                                        mask, c0);
      Mi = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                       mask, c0);
      Value maskStart64 = rewriter.create<memref::LoadOp>(loc, Mp, c0);
      Value maskEnd64 = rewriter.create<memref::LoadOp>(loc, Mp, c1);
      maskStart =
          rewriter.create<arith::IndexCastOp>(loc, maskStart64, indexType);
      maskEnd = rewriter.create<arith::IndexCastOp>(loc, maskEnd64, indexType);
    }

    // 1st pass
    //   Compute the number of nonzero entries in the result
    //   Store results in Cp
    //   The vector A is the fixed element, while the columns of B are the
    //   iteration element
    Value fixedIndexEnd64 = rewriter.create<memref::LoadOp>(loc, Ap, c1);
    Value fixedIndexEnd =
        rewriter.create<arith::IndexCastOp>(loc, fixedIndexEnd64, indexType);
    Value cmpColSame = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, c0, fixedIndexEnd);

    scf::IfOp ifBlock_rowTotal =
        rewriter.create<scf::IfOp>(loc, int64Type, cmpColSame, true);
    // if cmpColSame
    rewriter.setInsertionPointToStart(ifBlock_rowTotal.thenBlock());
    rewriter.create<scf::YieldOp>(loc, ci0);

    // else
    rewriter.setInsertionPointToStart(ifBlock_rowTotal.elseBlock());
    Value total;
    if (mask) {
      if (isMaskComplement) {
        ValueRange mcResult =
            buildMaskComplement(rewriter, loc, size, Mi, maskStart, maskEnd);
        Value maskComplement = mcResult[0];
        Value mcSize = mcResult[1];
        total = computeNumOverlaps(rewriter, loc, nk, Ai, c0, fixedIndexEnd, Bp,
                                   Bi, maskComplement, c0, mcSize, valueType);
        rewriter.create<memref::DeallocOp>(loc, maskComplement);
      } else {
        total = computeNumOverlaps(rewriter, loc, nk, Ai, c0, fixedIndexEnd, Bp,
                                   Bi, Mi, maskStart, maskEnd, valueType);
      }
    } else {
      total = computeNumOverlaps(rewriter, loc, nk, Ai, c0, fixedIndexEnd, Bp,
                                 Bi, nullptr, c0, size, valueType);
    }
    rewriter.create<scf::YieldOp>(loc, total);

    // end if cmpColSame
    rewriter.setInsertionPointAfter(ifBlock_rowTotal);
    Value nnzTotal = ifBlock_rowTotal.getResult(0);
    Value nnz = rewriter.create<arith::IndexCastOp>(loc, nnzTotal, indexType);
    rewriter.create<memref::StoreOp>(loc, nnzTotal, Cp, c1);

    callResizeIndex(rewriter, module, loc, C, c0, nnz);
    callResizeValues(rewriter, module, loc, C, nnz);
    Value Ci = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                           C, c0);
    Value Cx =
        rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, C);

    // 2nd pass
    //   Compute the nonzero values.
    //   Store in Ci and Cx
    //   The vector A is the fixed element, while the columns of B are the
    //   iteration element
    Value cmp_cpDifferent =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, c0, nnz);
    scf::IfOp ifBlock_cmpDiff =
        rewriter.create<scf::IfOp>(loc, cmp_cpDifferent);
    rewriter.setInsertionPointToStart(ifBlock_cmpDiff.thenBlock());

    if (mask) {
      if (isMaskComplement) {
        ValueRange mcResult =
            buildMaskComplement(rewriter, loc, size, Mi, maskStart, maskEnd);
        Value maskComplement = mcResult[0];
        Value mcSize = mcResult[1];
        computeInnerProduct(rewriter, loc, nk, c0, Ai, Ax, c0, fixedIndexEnd,
                            Bp, Bi, Bx, maskComplement, c0, mcSize, valueType,
                            extBlocks, Ci, Cx, c0, false);
        rewriter.create<memref::DeallocOp>(loc, maskComplement);
      } else {
        computeInnerProduct(rewriter, loc, nk, c0, Ai, Ax, c0, fixedIndexEnd,
                            Bp, Bi, Bx, Mi, maskStart, maskEnd, valueType,
                            extBlocks, Ci, Cx, c0, false);
      }
    } else {
      computeInnerProduct(rewriter, loc, nk, c0, Ai, Ax, c0, fixedIndexEnd, Bp,
                          Bi, Bx, nullptr, c0, size, valueType, extBlocks, Ci,
                          Cx, c0, false);
    }

    // end if cmpDiff
    rewriter.setInsertionPointAfter(ifBlock_cmpDiff);

    rewriter.replaceOp(op, C);

    cleanupIntermediateTensor(rewriter, module, loc, C);

    return success();
  }

  LogicalResult
  rewriteVectorVectorMultiplication(graphblas::MatrixMultiplyGenericOp op,
                                    PatternRewriter &rewriter,
                                    ExtensionBlocks extBlocks) const {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

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
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);

    Value size = rewriter.create<graphblas::SizeOp>(loc, A);

    Value C = callEmptyLike(rewriter, module, loc, A);
    callResizeDim(
        rewriter, module, loc, C, c0,
        c1); // exactly one entry because this is a vector representing a scalar
    callResizePointers(rewriter, module, loc, C, c0, c2);
    callResizeIndex(rewriter, module, loc, C, c0, c1);
    callResizeValues(rewriter, module, loc, C, c1);

    Value Ap = rewriter.create<sparse_tensor::ToPointersOp>(
        loc, memref1DI64Type, A, c0);
    Value Ai = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                           A, c0);
    Value Ax =
        rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, A);
    Value Bp = rewriter.create<sparse_tensor::ToPointersOp>(
        loc, memref1DI64Type, B, c0);
    Value Bi = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                           B, c0);
    Value Bx =
        rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, B);
    Value Ci = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                           C, c0);
    Value Cx =
        rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, C);

    // Single pass
    //   Compute the nonzero values.
    //   Store in Ci and Cx (single-element vector representing a scalar)
    //   The vector A is the fixed element, while the vector B is treated as the
    //   iteration element
    Value fixedIndexEnd64 = rewriter.create<memref::LoadOp>(loc, Ap, c1);
    Value fixedIndexEnd =
        rewriter.create<arith::IndexCastOp>(loc, fixedIndexEnd64, indexType);

    computeInnerProduct(rewriter, loc, size, c0, Ai, Ax, c0, fixedIndexEnd, Bp,
                        Bi, Bx, nullptr, c0, c1, valueType, extBlocks, Ci, Cx,
                        c0, false);

    // extract scalar from C
    Value cScalar = rewriter.create<memref::LoadOp>(loc, Cx, c0);

    rewriter.replaceOp(op, cScalar);

    cleanupIntermediateTensor(rewriter, module, loc, C);

    return success();
  }
};

class LowerMatrixMultiplyReduceToScalarGenericRewrite
    : public OpRewritePattern<
          graphblas::MatrixMultiplyReduceToScalarGenericOp> {
public:
  using OpRewritePattern<
      graphblas::MatrixMultiplyReduceToScalarGenericOp>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(graphblas::MatrixMultiplyReduceToScalarGenericOp op,
                  PatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>(); /* ignore unused variable
                                                          for debugging */
    (void)module;
    Location loc = op->getLoc();

    // Inputs
    Value A = op.a();
    Value B = op.b();
    Value mask = op.mask();

    // Required blocks
    RegionRange extensions = op.extensions();
    ExtensionBlocks extBlocks;
    std::set<graphblas::YieldKind> required = {
        graphblas::YieldKind::ADD_IDENTITY, graphblas::YieldKind::ADD,
        graphblas::YieldKind::MULT, graphblas::YieldKind::AGG_IDENTITY,
        graphblas::YieldKind::AGG};
    std::set<graphblas::YieldKind> optional = {};
    LogicalResult extractResult =
        extBlocks.extractBlocks(op, extensions, required, optional);

    if (extractResult.failed()) {
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
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    // TODO: make cf0 value dependent on the aggregator
    Value cf0 = llvm::TypeSwitch<Type, Value>(valueType)
                    .Case<IntegerType>([&](IntegerType type) {
                      return rewriter.create<arith::ConstantIntOp>(
                          loc, 0, type.getWidth());
                    })
                    .Case<FloatType>([&](FloatType type) {
                      return rewriter.create<arith::ConstantFloatOp>(
                          loc, APFloat(0.0), type);
                    });
    Value ctrue = rewriter.create<arith::ConstantIntOp>(loc, 1, boolType);
    Value cfalse = rewriter.create<arith::ConstantIntOp>(loc, 0, boolType);

    // Get sparse tensor info
    Value Ap = rewriter.create<sparse_tensor::ToPointersOp>(
        loc, memref1DI64Type, A, c1);
    Value Aj = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                           A, c1);
    Value Ax =
        rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, A);
    Value Bp = rewriter.create<sparse_tensor::ToPointersOp>(
        loc, memref1DI64Type, B, c1);
    Value Bi = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                           B, c1);
    Value Bx =
        rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, B);

    Value nrow = rewriter.create<graphblas::NumRowsOp>(loc, A);
    Value ncol = rewriter.create<graphblas::NumColsOp>(loc, B);
    Value nk = rewriter.create<graphblas::NumColsOp>(
        loc, A); // guaranteed equal to B.rows

    Value Mp, Mj;
    if (mask) {
      Mp = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type,
                                                        mask, c1);
      Mj = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                       mask, c1);
    }

    // In parallel over the rows and columns,
    //   compute the nonzero values and accumulate
    scf::ParallelOp rowLoop =
        rewriter.create<scf::ParallelOp>(loc, c0, nrow, c1, cf0);
    Value row = rowLoop.getInductionVars().front();
    rewriter.setInsertionPointToStart(rowLoop.getBody());

    Value rowPlus1 = rewriter.create<arith::AddIOp>(loc, row, c1);
    Value apStart64 = rewriter.create<memref::LoadOp>(loc, Ap, row);
    Value apEnd64 = rewriter.create<memref::LoadOp>(loc, Ap, rowPlus1);
    Value cmp_cpSame = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, apStart64, apEnd64);

    scf::IfOp ifBlock_cmpSame =
        rewriter.create<scf::IfOp>(loc, valueType, cmp_cpSame, true);
    // if cmpSame
    rewriter.setInsertionPointToStart(ifBlock_cmpSame.thenBlock());
    rewriter.create<scf::YieldOp>(loc, cf0);

    // else
    rewriter.setInsertionPointToStart(ifBlock_cmpSame.elseBlock());

    // Construct a dense array of row values
    Value colStart =
        rewriter.create<arith::IndexCastOp>(loc, apStart64, indexType);
    Value colEnd = rewriter.create<arith::IndexCastOp>(loc, apEnd64, indexType);
    Value kvec = rewriter.create<memref::AllocOp>(loc, memref1DValueType, nk);
    Value kvec_i1 = rewriter.create<memref::AllocOp>(loc, memref1DBoolType, nk);
    rewriter.create<linalg::FillOp>(loc, cfalse, kvec_i1);

    scf::ParallelOp colLoop1 =
        rewriter.create<scf::ParallelOp>(loc, colStart, colEnd, c1);
    Value jj = colLoop1.getInductionVars().front();
    rewriter.setInsertionPointToStart(colLoop1.getBody());
    Value col64 = rewriter.create<memref::LoadOp>(loc, Aj, jj);
    Value col = rewriter.create<arith::IndexCastOp>(loc, col64, indexType);
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
      Value mcolStart =
          rewriter.create<arith::IndexCastOp>(loc, mcolStart64, indexType);
      Value mcolEnd =
          rewriter.create<arith::IndexCastOp>(loc, mcolEnd64, indexType);

      colLoop2 =
          rewriter.create<scf::ParallelOp>(loc, mcolStart, mcolEnd, c1, cf0);
      Value mm = colLoop2.getInductionVars().front();
      rewriter.setInsertionPointToStart(colLoop2.getBody());
      col64 = rewriter.create<memref::LoadOp>(loc, Mj, mm);
      col = rewriter.create<arith::IndexCastOp>(loc, col64, indexType);
    } else {
      colLoop2 = rewriter.create<scf::ParallelOp>(loc, c0, ncol, c1, cf0);
      col = colLoop2.getInductionVars().front();
      rewriter.setInsertionPointToStart(colLoop2.getBody());
      col64 = rewriter.create<arith::IndexCastOp>(loc, col, int64Type);
    }

    Value colPlus1 = rewriter.create<arith::AddIOp>(loc, col, c1);
    Value iStart64 = rewriter.create<memref::LoadOp>(loc, Bp, col);
    Value iEnd64 = rewriter.create<memref::LoadOp>(loc, Bp, colPlus1);
    Value iStart =
        rewriter.create<arith::IndexCastOp>(loc, iStart64, indexType);
    Value iEnd = rewriter.create<arith::IndexCastOp>(loc, iEnd64, indexType);

    // insert add identity block
    rewriter.mergeBlocks(extBlocks.addIdentity, rewriter.getBlock(), {});
    graphblas::YieldOp addIdentityYield =
        llvm::dyn_cast_or_null<graphblas::YieldOp>(
            rewriter.getBlock()->getTerminator());
    Value addIdentity = addIdentityYield.values().front();
    rewriter.eraseOp(addIdentityYield);

    scf::ForOp kLoop =
        rewriter.create<scf::ForOp>(loc, iStart, iEnd, c1, addIdentity);
    Value ii = kLoop.getInductionVar();
    Value curr = kLoop.getLoopBody().getArgument(1);
    rewriter.setInsertionPointToStart(kLoop.getBody());

    Value kk64 = rewriter.create<memref::LoadOp>(loc, Bi, ii);
    Value kk = rewriter.create<arith::IndexCastOp>(loc, kk64, indexType);
    Value cmpPair = rewriter.create<memref::LoadOp>(loc, kvec_i1, kk);
    scf::IfOp ifBlock_cmpPair =
        rewriter.create<scf::IfOp>(loc, valueType, cmpPair, true);
    // if cmpPair
    rewriter.setInsertionPointToStart(ifBlock_cmpPair.thenBlock());

    Value aVal = rewriter.create<memref::LoadOp>(loc, kvec, kk);
    Value bVal = rewriter.create<memref::LoadOp>(loc, Bx, ii);

    // insert multiply operation block
    rewriter.mergeBlocks(extBlocks.mult, rewriter.getBlock(),
                         {aVal, bVal, row, col, kk});
    graphblas::YieldOp multYield = llvm::dyn_cast_or_null<graphblas::YieldOp>(
        rewriter.getBlock()->getTerminator());
    Value multResult = multYield.values().front();
    rewriter.eraseOp(multYield);

    // insert add operation block
    rewriter.mergeBlocks(extBlocks.add, rewriter.getBlock(),
                         {curr, multResult});
    graphblas::YieldOp addYield = llvm::dyn_cast_or_null<graphblas::YieldOp>(
        rewriter.getBlock()->getTerminator());
    Value addResult = addYield.values().front();
    rewriter.eraseOp(addYield);

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
    // Clone blocks into front of region to displace existing entry block, which
    // will be removed by canonicalization later
    aggRegion->cloneInto(&colReducer.getRegion(),
                         colReducer.getRegion().begin(), mapper);
    graphblas::YieldOp colYield = llvm::dyn_cast_or_null<graphblas::YieldOp>(
        colReducer.getRegion().front().getTerminator());
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

    graphblas::YieldOp yield = llvm::dyn_cast_or_null<graphblas::YieldOp>(
        extBlocks.agg->getTerminator());
    Value aggResult = yield.values().front();

    // we can safely merge this agg block now, since the previous agg instance
    // was cloned above
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

class LowerUnionRewrite : public OpRewritePattern<graphblas::UnionOp> {
public:
  using OpRewritePattern<graphblas::UnionOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::UnionOp op,
                                PatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    // Inputs
    Value a = op.a();
    Value b = op.b();
    std::string unionOperator = op.union_operator().str();

    // Types
    RankedTensorType aType = a.getType().dyn_cast<RankedTensorType>();

    unsigned rank = aType.getRank(); // ranks guaranteed to be equal

    Value output = callEmptyLike(rewriter, module, loc, a);
    if (rank == 2) {
      computeMatrixElementWise(rewriter, loc, module, a, b, output,
                               unionOperator,
                               /* intersect */ false);
    } else {
      computeVectorElementWise(rewriter, loc, module, a, b, output,
                               unionOperator,
                               /* intersect */ false);
    }

    rewriter.replaceOp(op, output);

    cleanupIntermediateTensor(rewriter, module, loc, output);

    return success();
  };
};

class LowerIntersectRewrite : public OpRewritePattern<graphblas::IntersectOp> {
public:
  using OpRewritePattern<graphblas::IntersectOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::IntersectOp op,
                                PatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    // Inputs
    Value a = op.a();
    Value b = op.b();
    std::string intersectOperator = op.intersect_operator().str();

    // Types
    RankedTensorType aType = a.getType().dyn_cast<RankedTensorType>();

    unsigned rank = aType.getRank(); // ranks guaranteed to be equal

    Value output = callEmptyLike(rewriter, module, loc, a);
    if (rank == 2) {
      computeMatrixElementWise(rewriter, loc, module, a, b, output,
                               intersectOperator, /* intersect */ true);
    } else {
      computeVectorElementWise(rewriter, loc, module, a, b, output,
                               intersectOperator, /* intersect */ true);
    }

    rewriter.replaceOp(op, output);

    cleanupIntermediateTensor(rewriter, module, loc, output);

    return success();
  };
};

class LowerUpdateRewrite : public OpRewritePattern<graphblas::UpdateOp> {
public:
  using OpRewritePattern<graphblas::UpdateOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::UpdateOp op,
                                PatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    // Inputs
    Value input = op.input();
    Value output = op.output();
    llvm::Optional<llvm::StringRef> accumulateOperator =
        op.accumulate_operator();
    std::string accumulateString;
    if (accumulateOperator) {
      accumulateString = accumulateOperator->str();
    }
    Value mask = op.mask();
    bool maskComplement = op.mask_complement();
    bool replace = op.replace();

    // Types
    RankedTensorType outputType = output.getType().dyn_cast<RankedTensorType>();

    unsigned rank = outputType.getRank(); // ranks guaranteed to be equal
    auto computeEwise =
        rank == 2 ? computeMatrixElementWise : computeVectorElementWise;

    if (accumulateOperator) {
      if (mask) {
        auto maskString = (maskComplement ? "mask_complement" : "mask");
        if (replace) {
          // input -> output(mask) { accumulate, replace }

          // Step 1: apply the mask to the output
          Value maskedOutput = callEmptyLike(rewriter, module, loc, output);
          computeEwise(rewriter, loc, module, output, mask, maskedOutput,
                       maskString, true);
          // Step 2: apply the mask to the input
          Value maskedInput = callEmptyLike(rewriter, module, loc, input);
          computeEwise(rewriter, loc, module, input, mask, maskedInput,
                       maskString, true);
          // Step 3: union the two masked results
          computeEwise(rewriter, loc, module, maskedInput, maskedOutput, output,
                       accumulateString,
                       /* intersect */ false);
          rewriter.create<sparse_tensor::ReleaseOp>(loc, maskedOutput);
          rewriter.create<sparse_tensor::ReleaseOp>(loc, maskedInput);
        } else {
          // input -> output(mask) { accumulate }

          // Step 1: apply the mask to the input
          Value maskedInput = callEmptyLike(rewriter, module, loc, input);
          computeEwise(rewriter, loc, module, input, mask, maskedInput,
                       maskString, true);
          // Step 2: union the two masked results
          Value outputCopy = callDupTensor(rewriter, module, loc, output);
          computeEwise(rewriter, loc, module, maskedInput, outputCopy, output,
                       accumulateString,
                       /* intersect */ false);
          rewriter.create<sparse_tensor::ReleaseOp>(loc, outputCopy);
          rewriter.create<sparse_tensor::ReleaseOp>(loc, maskedInput);
        }
      } else {
        // input -> output { accumulate, replace? }

        Value outputCopy = callDupTensor(rewriter, module, loc, output);
        computeEwise(rewriter, loc, module, input, outputCopy, output,
                     accumulateString, /* intersect */ false);
        rewriter.create<sparse_tensor::ReleaseOp>(loc, outputCopy);
      }
    } else {
      if (mask) {
        auto maskString = (maskComplement ? "mask_complement" : "mask");
        if (replace) {
          // input -> output(mask) { replace }

          // Step 1: apply the mask to the input
          Value maskedInput = callEmptyLike(rewriter, module, loc, input);
          computeEwise(rewriter, loc, module, input, mask, maskedInput,
                       maskString, true);
          // Step 2: swap masked input with output
          callSwapPointers(rewriter, module, loc, maskedInput, output);
          callSwapIndices(rewriter, module, loc, maskedInput, output);
          callSwapValues(rewriter, module, loc, maskedInput, output);
          rewriter.create<sparse_tensor::ReleaseOp>(loc, maskedInput);
        } else {
          // input -> output(mask)

          // Step 1: apply the mask inverse to the output
          auto maskInverseString =
              (maskComplement ? "mask" : "mask_complement");
          Value maskedOutput = callEmptyLike(rewriter, module, loc, output);
          computeEwise(rewriter, loc, module, output, mask, maskedOutput,
                       maskInverseString, true);
          // Step 2: apply the mask to the input
          Value maskedInput = callEmptyLike(rewriter, module, loc, input);
          computeEwise(rewriter, loc, module, input, mask, maskedInput,
                       maskString, true);
          // Step 3: union the two masked results
          computeEwise(rewriter, loc, module, maskedInput, maskedOutput, output,
                       "plus", // Overlaps should never occur, so this choice
                               // doesn't matter
                       /* intersect */ false);
          rewriter.create<sparse_tensor::ReleaseOp>(loc, maskedOutput);
          rewriter.create<sparse_tensor::ReleaseOp>(loc, maskedInput);
        }
      } else {
        // input -> output { replace? }

        Value inputCopy = callDupTensor(rewriter, module, loc, input);
        callSwapPointers(rewriter, module, loc, inputCopy, output);
        callSwapIndices(rewriter, module, loc, inputCopy, output);
        callSwapValues(rewriter, module, loc, inputCopy, output);
        rewriter.create<sparse_tensor::ReleaseOp>(loc, inputCopy);
      }
    }

    // TODO: figure out how to replace an op with no return type
    rewriter.replaceOp(op, c0);

    return success();
  };
};

class LowerEqualRewrite : public OpRewritePattern<graphblas::EqualOp> {
public:
  using OpRewritePattern<graphblas::EqualOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::EqualOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    // Inputs
    Value A = op.a();
    Value B = op.b();
    RankedTensorType aType = A.getType().dyn_cast<RankedTensorType>();

    // Types
    Type boolType = rewriter.getI1Type();
    // Need to use a standard word size in AND-reduction for OpenMP
    // This could be i8, i32, or i64, but we pick i32
    Type intReduceType = rewriter.getIntegerType(32);
    Type int64Type = rewriter.getIntegerType(64);
    Type valueType = aType.getElementType();
    MemRefType memref1DI64Type = MemRefType::get({-1}, int64Type);
    MemRefType memref1DValueType = MemRefType::get({-1}, valueType);

    // Initial constants
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value cfalse = rewriter.create<arith::ConstantIntOp>(loc, 0, boolType);
    Value c1_reduce =
        rewriter.create<arith::ConstantIntOp>(loc, 1, intReduceType);

    unsigned rank = aType.getRank(); // ranks guaranteed to be equal

    Value dimIndex;
    Value cmpShape;
    if (rank == 2) {
      // Matrix check
      dimIndex = c1;
      Value aNrows = rewriter.create<graphblas::NumRowsOp>(loc, A);
      Value bNrows = rewriter.create<graphblas::NumRowsOp>(loc, B);
      Value aNcols = rewriter.create<graphblas::NumColsOp>(loc, A);
      Value bNcols = rewriter.create<graphblas::NumColsOp>(loc, B);
      Value cmpNrows = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, aNrows, bNrows);
      Value cmpNcols = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, aNcols, bNcols);
      cmpShape = rewriter.create<arith::AndIOp>(loc, cmpNrows, cmpNcols);
    } else {
      // Vector check
      dimIndex = c0;
      // Check size
      Value aSize = rewriter.create<graphblas::SizeOp>(loc, A);
      Value bSize = rewriter.create<graphblas::SizeOp>(loc, B);
      cmpShape = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                aSize, bSize);
    }

    scf::IfOp ifOuter =
        rewriter.create<scf::IfOp>(loc, boolType, cmpShape, true);
    // if cmpSize
    rewriter.setInsertionPointToStart(ifOuter.thenBlock());

    // Check number of non-zeros
    Value aNnz = rewriter.create<graphblas::NumValsOp>(loc, A);
    Value bNnz = rewriter.create<graphblas::NumValsOp>(loc, B);
    Value cmpNnz = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  aNnz, bNnz);
    scf::IfOp ifNnz = rewriter.create<scf::IfOp>(loc, boolType, cmpNnz, true);
    // if cmpNnz
    rewriter.setInsertionPointToStart(ifNnz.thenBlock());

    // Check index positions and values
    Value Ai = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                           A, dimIndex);
    Value Bi = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                           B, dimIndex);
    Value Ax =
        rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, A);
    Value Bx =
        rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, B);

    scf::ParallelOp indexLoop =
        rewriter.create<scf::ParallelOp>(loc, c0, aNnz, c1, c1_reduce);
    Value loopIdx = indexLoop.getInductionVars().front();
    rewriter.setInsertionPointToStart(indexLoop.getBody());

    Value aIndex = rewriter.create<memref::LoadOp>(loc, Ai, loopIdx);
    Value bIndex = rewriter.create<memref::LoadOp>(loc, Bi, loopIdx);
    Value aValue = rewriter.create<memref::LoadOp>(loc, Ax, loopIdx);
    Value bValue = rewriter.create<memref::LoadOp>(loc, Bx, loopIdx);
    Value cmpIndex = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, aIndex, bIndex);
    Value cmpValue = llvm::TypeSwitch<Type, Value>(valueType)
                         .Case<IntegerType>([&](IntegerType type) {
                           return rewriter.create<arith::CmpIOp>(
                               loc, arith::CmpIPredicate::eq, aValue, bValue);
                         })
                         .Case<FloatType>([&](FloatType type) {
                           return rewriter.create<arith::CmpFOp>(
                               loc, arith::CmpFPredicate::OEQ, aValue, bValue);
                         });
    Value cmpCombined = rewriter.create<arith::AndIOp>(loc, cmpIndex, cmpValue);
    // Need to do reduction with a standard word size (rather than i1) for
    // OpenMP
    Value cmpCombined_ext =
        rewriter.create<arith::ExtSIOp>(loc, cmpCombined, intReduceType);

    scf::ReduceOp reducer =
        rewriter.create<scf::ReduceOp>(loc, cmpCombined_ext);
    BlockArgument lhs = reducer.getRegion().getArgument(0);
    BlockArgument rhs = reducer.getRegion().getArgument(1);
    rewriter.setInsertionPointToStart(&reducer.getRegion().front());
    Value cmpFinal = rewriter.create<arith::AndIOp>(loc, lhs, rhs);
    rewriter.create<scf::ReduceReturnOp>(loc, cmpFinal);

    rewriter.setInsertionPointAfter(indexLoop);
    Value boolResult =
        rewriter.create<arith::TruncIOp>(loc, indexLoop.getResult(0), boolType);
    rewriter.create<scf::YieldOp>(loc, boolResult);

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
  };
};

class LowerDiagOpRewrite : public OpRewritePattern<graphblas::DiagOp> {
public:
  using OpRewritePattern<graphblas::DiagOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::DiagOp op,
                                PatternRewriter &rewriter) const override {

    RankedTensorType resultTensorType =
        op.getResult().getType().dyn_cast<RankedTensorType>();

    if (resultTensorType.getRank() == 1) {
      return lowerMatrixToVecDiagOp(op, rewriter, resultTensorType);
    } else if (resultTensorType.getRank() == 2) {
      return lowerVecToMatrixDiagOp(op, rewriter, resultTensorType);
    }

    return failure();
  };

private:
  LogicalResult
  lowerVecToMatrixDiagOp(graphblas::DiagOp op, PatternRewriter &rewriter,
                         RankedTensorType &resultTensorType) const {

    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    Value vector = op.input();

    Type valueType = resultTensorType.getElementType();

    Type indexType = rewriter.getIndexType();
    Type int64Type = rewriter.getIntegerType(64);
    Type memref1DI64Type = MemRefType::get({-1}, int64Type);
    Type memref1DValueType = MemRefType::get({-1}, valueType);

    Value c0_i64 =
        rewriter.create<ConstantOp>(loc, rewriter.getIntegerAttr(int64Type, 0));
    Value c1_i64 =
        rewriter.create<ConstantOp>(loc, rewriter.getIntegerAttr(int64Type, 1));

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    Value vectorLength = rewriter.create<graphblas::SizeOp>(loc, vector);
    Value vectorIndices = rewriter.create<sparse_tensor::ToIndicesOp>(
        loc, memref1DI64Type, vector, c0);
    Value vectorValues = rewriter.create<sparse_tensor::ToValuesOp>(
        loc, memref1DValueType, vector);

    Value output = callNewTensor(
        rewriter, module, loc, {vectorLength, vectorLength}, resultTensorType);

    Value outputNNZ = rewriter.create<graphblas::NumValsOp>(loc, vector);
    callResizeIndex(rewriter, module, loc, output, c1, outputNNZ);
    callResizeValues(rewriter, module, loc, output, outputNNZ);

    Value outputIndices = rewriter.create<sparse_tensor::ToIndicesOp>(
        loc, memref1DI64Type, output, c1);
    Value outputValues = rewriter.create<sparse_tensor::ToValuesOp>(
        loc, memref1DValueType, output);

    scf::ForOp copyValuesAndIndicesLoop =
        rewriter.create<scf::ForOp>(loc, c0, outputNNZ, c1);
    {
      rewriter.setInsertionPointToStart(copyValuesAndIndicesLoop.getBody());
      Value outputPosition = copyValuesAndIndicesLoop.getInductionVar();
      Value vectorIndex =
          rewriter.create<memref::LoadOp>(loc, vectorIndices, outputPosition);
      rewriter.create<memref::StoreOp>(loc, vectorIndex, outputIndices,
                                       outputPosition);
      Value vectorValue =
          rewriter.create<memref::LoadOp>(loc, vectorValues, outputPosition);
      rewriter.create<memref::StoreOp>(loc, vectorValue, outputValues,
                                       outputPosition);
      rewriter.setInsertionPointAfter(copyValuesAndIndicesLoop);
    }

    Value outputPointers = rewriter.create<sparse_tensor::ToPointersOp>(
        loc, memref1DI64Type, output, c1);
    Value initialVectorIndicesValue =
        rewriter.create<memref::LoadOp>(loc, vectorIndices, c0);
    Value vectorLengthMinusOne =
        rewriter.create<arith::SubIOp>(loc, vectorLength, c1);
    scf::ForOp pointersUpdateLoop = rewriter.create<scf::ForOp>(
        loc, c0, vectorLength, c1,
        ValueRange{c0_i64, c0, initialVectorIndicesValue});
    {
      rewriter.setInsertionPointToStart(pointersUpdateLoop.getBody());
      Value pointersPosition = pointersUpdateLoop.getInductionVar();
      Value ptr_i64 = pointersUpdateLoop.getLoopBody().getArgument(1);
      Value vectorIndicesPosition =
          pointersUpdateLoop.getLoopBody().getArgument(2);
      Value vectorIndicesValue =
          pointersUpdateLoop.getLoopBody().getArgument(3);

      rewriter.create<memref::StoreOp>(loc, ptr_i64, outputPointers,
                                       pointersPosition);
      Value pointersPosition_i64 =
          rewriter.create<arith::IndexCastOp>(loc, pointersPosition, int64Type);
      Value rowHasValue = rewriter.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::eq, vectorIndicesValue,
          pointersPosition_i64);
      Value notAtLastIteration = rewriter.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ne, pointersPosition,
          vectorLengthMinusOne);
      Value mustUpdate =
          rewriter.create<arith::AndIOp>(loc, notAtLastIteration, rowHasValue);

      scf::IfOp ifMustUpdateBlock = rewriter.create<scf::IfOp>(
          loc, TypeRange{int64Type, indexType, int64Type}, mustUpdate, true);
      {
        rewriter.setInsertionPointToStart(ifMustUpdateBlock.thenBlock());
        Value nextPtr_i64 =
            rewriter.create<arith::AddIOp>(loc, ptr_i64, c1_i64);
        Value nextVectorIndicesPosition =
            rewriter.create<arith::AddIOp>(loc, vectorIndicesPosition, c1);
        Value nextUpdatedVectorIndicesValue = rewriter.create<memref::LoadOp>(
            loc, vectorIndices, nextVectorIndicesPosition);

        rewriter.create<scf::YieldOp>(
            loc, ValueRange{nextPtr_i64, nextVectorIndicesPosition,
                            nextUpdatedVectorIndicesValue});
      }
      {
        rewriter.setInsertionPointToStart(ifMustUpdateBlock.elseBlock());
        rewriter.create<scf::YieldOp>(
            loc,
            ValueRange{ptr_i64, vectorIndicesPosition, vectorIndicesValue});
      }
      rewriter.setInsertionPointAfter(ifMustUpdateBlock);

      Value updatedPtr_i64 = ifMustUpdateBlock.getResult(0);
      Value updatedVectorIndicesPosition = ifMustUpdateBlock.getResult(1);
      Value updatedVectorIndicesValue = ifMustUpdateBlock.getResult(2);

      rewriter.create<scf::YieldOp>(
          loc, ValueRange{updatedPtr_i64, updatedVectorIndicesPosition,
                          updatedVectorIndicesValue});

      rewriter.setInsertionPointAfter(pointersUpdateLoop);
    }

    Value outputNNZ_i64 =
        rewriter.create<arith::IndexCastOp>(loc, outputNNZ, int64Type);
    rewriter.create<memref::StoreOp>(loc, outputNNZ_i64, outputPointers,
                                     vectorLength);

    rewriter.replaceOp(op, output);

    cleanupIntermediateTensor(rewriter, module, loc, output);

    return success();
  }

  LogicalResult
  lowerMatrixToVecDiagOp(graphblas::DiagOp op, PatternRewriter &rewriter,
                         RankedTensorType &resultTensorType) const {

    // This implementation reads as assuming the input matrix is CSR,
    // but it will work for CSC as well.

    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    Value matrix = op.input();

    Type valueType = resultTensorType.getElementType();

    Type indexType = rewriter.getIndexType();
    Type int64Type = rewriter.getIntegerType(64);
    Type int1Type = rewriter.getIntegerType(1);
    Type memref1DI64Type = MemRefType::get({-1}, int64Type);
    Type memref1DValueType = MemRefType::get({-1}, valueType);

    Value c1_i1 =
        rewriter.create<ConstantOp>(loc, rewriter.getIntegerAttr(int1Type, 1));
    Value c0_valueType = llvm::TypeSwitch<Type, Value>(valueType)
                             .Case<IntegerType>([&](IntegerType type) {
                               return rewriter.create<ConstantOp>(
                                   loc, rewriter.getIntegerAttr(valueType, 1));
                             })
                             .Case<FloatType>([&](FloatType type) {
                               return rewriter.create<ConstantOp>(
                                   loc, rewriter.getFloatAttr(valueType, 1.0));
                             });

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);

    Value nrows = rewriter.create<graphblas::NumRowsOp>(loc, matrix);

    Value matrixPointers = rewriter.create<sparse_tensor::ToPointersOp>(
        loc, memref1DI64Type, matrix, c1);
    Value matrixIndices = rewriter.create<sparse_tensor::ToIndicesOp>(
        loc, memref1DI64Type, matrix, c1);
    Value matrixValues = rewriter.create<sparse_tensor::ToValuesOp>(
        loc, memref1DValueType, matrix);

    Value output = callNewTensor(rewriter, module, loc, ValueRange{nrows},
                                 resultTensorType);
    callResizeDim(rewriter, module, loc, output, c0, nrows);

    // We do two loops, one to find the output vector's nnz
    // and one to fill up the output's indices and values.
    // We have to get the nnz first to allocate space in the
    // output vector correctly.
    // TODO We could do one loop where we do both.
    //  1) assume that the output nnz is some arbitrary size
    //  2) resize accordingly
    //  3) on each iteration,
    //        - store the values and indices
    //        - track the correct nnz
    //        - if we reach the limit of whatever size the
    //          output vector is, resize (double it or
    //          something like that)
    //  4) resize the output vector to the correct nnz
    // It's unclear which approach is more performant since
    // it depends on how accurate our arbitrary guesses are
    // for initial size and how much we should resize.

    scf::ForOp outputNNZLoop =
        rewriter.create<scf::ForOp>(loc, c0, nrows, c1, ValueRange{c0});
    {
      Value numDiagonalContainingRows =
          outputNNZLoop.getLoopBody().getArgument(1);
      rewriter.setInsertionPointToStart(outputNNZLoop.getBody());

      Value matrixRowIndex = outputNNZLoop.getInductionVar();
      Value nextMatrixRowIndex =
          rewriter.create<arith::AddIOp>(loc, matrixRowIndex, c1);

      Value firstPtr_i64 =
          rewriter.create<memref::LoadOp>(loc, matrixPointers, matrixRowIndex);
      Value secondPtr_i64 = rewriter.create<memref::LoadOp>(loc, matrixPointers,
                                                            nextMatrixRowIndex);

      Value firstPtr =
          rewriter.create<arith::IndexCastOp>(loc, firstPtr_i64, indexType);
      Value secondPtr =
          rewriter.create<arith::IndexCastOp>(loc, secondPtr_i64, indexType);

      Value matrixRowIndex_i64 =
          rewriter.create<arith::IndexCastOp>(loc, matrixRowIndex, int64Type);

      scf::WhileOp findDiagonalWhileLoop = rewriter.create<scf::WhileOp>(
          loc, TypeRange{indexType, int1Type}, ValueRange{firstPtr, c1_i1});
      Block *findDiagonalWhileLoopBefore = rewriter.createBlock(
          &findDiagonalWhileLoop.before(), {}, TypeRange{indexType, int1Type});
      Block *findDiagonalWhileLoopAfter = rewriter.createBlock(
          &findDiagonalWhileLoop.after(), {}, TypeRange{indexType, int1Type});
      Value diagonalNotFound = findDiagonalWhileLoop.getResult(1);
      {
        rewriter.setInsertionPointToStart(
            &findDiagonalWhileLoop.before().front());
        Value ptr = findDiagonalWhileLoopBefore->getArgument(0);
        Value diagonalPositionNotFound =
            findDiagonalWhileLoopBefore->getArgument(1);
        Value morePtrs = rewriter.create<arith::CmpIOp>(
            op.getLoc(), arith::CmpIPredicate::ult, ptr, secondPtr);
        Value continueCondition = rewriter.create<arith::AndIOp>(
            loc, diagonalPositionNotFound, morePtrs);
        rewriter.create<scf::ConditionOp>(
            loc, continueCondition, ValueRange{ptr, diagonalPositionNotFound});
      }
      {
        rewriter.setInsertionPointToStart(
            &findDiagonalWhileLoop.after().front());
        Value currentPtr = findDiagonalWhileLoopAfter->getArgument(0);
        Value elementColumnIndex_i64 =
            rewriter.create<memref::LoadOp>(loc, matrixIndices, currentPtr);
        Value isNotDiagonalPosition = rewriter.create<arith::CmpIOp>(
            op.getLoc(), arith::CmpIPredicate::ne, elementColumnIndex_i64,
            matrixRowIndex_i64);
        Value nextPtr = rewriter.create<arith::AddIOp>(loc, currentPtr, c1);
        rewriter.create<scf::YieldOp>(
            loc, ValueRange{nextPtr, isNotDiagonalPosition});
        rewriter.setInsertionPointAfter(findDiagonalWhileLoop);
      }

      scf::IfOp ifDiagonalNotFoundBlock = rewriter.create<scf::IfOp>(
          loc, TypeRange{indexType}, diagonalNotFound, true);
      {
        rewriter.setInsertionPointToStart(ifDiagonalNotFoundBlock.thenBlock());
        rewriter.create<scf::YieldOp>(loc,
                                      ValueRange{numDiagonalContainingRows});
      }
      {
        rewriter.setInsertionPointToStart(ifDiagonalNotFoundBlock.elseBlock());
        Value nextNumDiagonalContainingRows =
            rewriter.create<arith::AddIOp>(loc, numDiagonalContainingRows, c1);
        rewriter.create<scf::YieldOp>(
            loc, ValueRange{nextNumDiagonalContainingRows});
      }
      rewriter.setInsertionPointAfter(ifDiagonalNotFoundBlock);
      Value updatedNumDiagonalContainingRows =
          ifDiagonalNotFoundBlock.getResult(0);

      rewriter.create<scf::YieldOp>(
          loc, ValueRange{updatedNumDiagonalContainingRows});

      rewriter.setInsertionPointAfter(outputNNZLoop);
    }
    Value outputNNZ = outputNNZLoop.getResult(0);

    callResizePointers(rewriter, module, loc, output, c0, c2);
    callResizeIndex(rewriter, module, loc, output, c0, outputNNZ);
    callResizeValues(rewriter, module, loc, output, outputNNZ);

    Value outputPointers = rewriter.create<sparse_tensor::ToPointersOp>(
        loc, memref1DI64Type, output, c0);
    Value outputNNZ_i64 =
        rewriter.create<arith::IndexCastOp>(loc, outputNNZ, int64Type);
    rewriter.create<memref::StoreOp>(loc, outputNNZ_i64, outputPointers, c1);

    Value outputIndices = rewriter.create<sparse_tensor::ToIndicesOp>(
        loc, memref1DI64Type, output, c0);
    Value outputValues = rewriter.create<sparse_tensor::ToValuesOp>(
        loc, memref1DValueType, output);

    scf::ForOp outputValueAndIncidesFillingLoop =
        rewriter.create<scf::ForOp>(loc, c0, nrows, c1, ValueRange{c0});
    {
      Value outputValuesPosition =
          outputValueAndIncidesFillingLoop.getLoopBody().getArgument(1);
      Value rowIndex = outputValueAndIncidesFillingLoop.getInductionVar();
      rewriter.setInsertionPointToStart(
          outputValueAndIncidesFillingLoop.getBody());

      Value nextRowIndex = rewriter.create<arith::AddIOp>(loc, rowIndex, c1);
      Value firstPtr_i64 =
          rewriter.create<memref::LoadOp>(loc, matrixPointers, rowIndex);
      Value secondPtr_i64 =
          rewriter.create<memref::LoadOp>(loc, matrixPointers, nextRowIndex);

      Value firstPtr =
          rewriter.create<arith::IndexCastOp>(loc, firstPtr_i64, indexType);
      Value secondPtr =
          rewriter.create<arith::IndexCastOp>(loc, secondPtr_i64, indexType);

      Value rowIndex_i64 =
          rewriter.create<arith::IndexCastOp>(loc, rowIndex, int64Type);

      // instead of having a var for whether or not a diagonal value was found
      // and the value itself, we could just track whether or not the diagonal
      // value (see the C++ variable diagonalValue) is zero (or whatever the
      // missing value represents).
      // This will cause bugs with malformed sparse tensors that have the
      // missing value in the values array.

      // c0_valueType is just used as a dummmy initial value here ; any garbage
      // value would work
      scf::WhileOp findDiagonalWhileLoop = rewriter.create<scf::WhileOp>(
          loc, TypeRange{indexType, int1Type, valueType},
          ValueRange{firstPtr, c1_i1, c0_valueType});
      Block *findDiagonalWhileLoopBefore =
          rewriter.createBlock(&findDiagonalWhileLoop.before(), {},
                               TypeRange{indexType, int1Type, valueType});
      Block *findDiagonalWhileLoopAfter =
          rewriter.createBlock(&findDiagonalWhileLoop.after(), {},
                               TypeRange{indexType, int1Type, valueType});
      Value diagonalNotFound = findDiagonalWhileLoop.getResult(1);
      Value diagonalValue = findDiagonalWhileLoop.getResult(2);
      {
        Value ptr = findDiagonalWhileLoopBefore->getArgument(0);
        Value diagonalPositionNotFound =
            findDiagonalWhileLoopBefore->getArgument(1);
        Value currentDiagonalValue =
            findDiagonalWhileLoopBefore->getArgument(2);
        rewriter.setInsertionPointToStart(
            &findDiagonalWhileLoop.before().front());
        Value morePtrs = rewriter.create<arith::CmpIOp>(
            op.getLoc(), arith::CmpIPredicate::ult, ptr, secondPtr);
        Value continueCondition = rewriter.create<arith::AndIOp>(
            loc, diagonalPositionNotFound, morePtrs);
        rewriter.create<scf::ConditionOp>(
            loc, continueCondition,
            ValueRange{ptr, diagonalPositionNotFound, currentDiagonalValue});
      }
      {
        rewriter.setInsertionPointToStart(
            &findDiagonalWhileLoop.after().front());
        Value currentPtr = findDiagonalWhileLoopAfter->getArgument(0);
        Value previousDiagonalValue =
            findDiagonalWhileLoopAfter->getArgument(2);
        Value elementColumnIndex_i64 =
            rewriter.create<memref::LoadOp>(loc, matrixIndices, currentPtr);
        Value isNotDiagonalPosition = rewriter.create<arith::CmpIOp>(
            op.getLoc(), arith::CmpIPredicate::ne, elementColumnIndex_i64,
            rowIndex_i64);

        scf::IfOp ifDiagonalNotFoundBlock = rewriter.create<scf::IfOp>(
            loc, TypeRange{valueType}, isNotDiagonalPosition, true);
        {
          rewriter.setInsertionPointToStart(
              ifDiagonalNotFoundBlock.thenBlock());
          // TODO yielding a dummy value (e.g. c0_valueType) works as well ;
          // unsure which creates more optimal code
          rewriter.create<scf::YieldOp>(loc, ValueRange{previousDiagonalValue});
        }
        {
          rewriter.setInsertionPointToStart(
              ifDiagonalNotFoundBlock.elseBlock());
          Value actualDiagonalValue =
              rewriter.create<memref::LoadOp>(loc, matrixValues, currentPtr);
          rewriter.create<scf::YieldOp>(loc, ValueRange{actualDiagonalValue});
        }
        rewriter.setInsertionPointAfter(ifDiagonalNotFoundBlock);
        Value updatedDiagonalValue = ifDiagonalNotFoundBlock.getResult(0);

        Value nextPtr = rewriter.create<arith::AddIOp>(loc, currentPtr, c1);
        rewriter.create<scf::YieldOp>(
            loc,
            ValueRange{nextPtr, isNotDiagonalPosition, updatedDiagonalValue});
        rewriter.setInsertionPointAfter(findDiagonalWhileLoop);
      }

      scf::IfOp ifDiagonalNotFoundBlock = rewriter.create<scf::IfOp>(
          loc, TypeRange{indexType}, diagonalNotFound, true);
      {
        rewriter.setInsertionPointToStart(ifDiagonalNotFoundBlock.thenBlock());
        rewriter.create<scf::YieldOp>(loc, ValueRange{outputValuesPosition});
      }
      {
        rewriter.setInsertionPointToStart(ifDiagonalNotFoundBlock.elseBlock());

        rewriter.create<memref::StoreOp>(loc, diagonalValue, outputValues,
                                         outputValuesPosition);
        rewriter.create<memref::StoreOp>(loc, rowIndex_i64, outputIndices,
                                         outputValuesPosition);

        Value nextOutputValuesPosition =
            rewriter.create<arith::AddIOp>(loc, outputValuesPosition, c1);
        rewriter.create<scf::YieldOp>(loc,
                                      ValueRange{nextOutputValuesPosition});
      }
      rewriter.setInsertionPointAfter(ifDiagonalNotFoundBlock);
      Value nextOutputValuesPosition = ifDiagonalNotFoundBlock.getResult(0);

      rewriter.create<scf::YieldOp>(loc, ValueRange{nextOutputValuesPosition});
      rewriter.setInsertionPointAfter(outputValueAndIncidesFillingLoop);
    }

    rewriter.replaceOp(op, output);

    return success();
  }
};

class LowerCommentRewrite : public OpRewritePattern<graphblas::CommentOp> {
public:
  using OpRewritePattern<graphblas::CommentOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::CommentOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  };
};

class LowerPrintRewrite : public OpRewritePattern<graphblas::PrintOp> {
public:
  using OpRewritePattern<graphblas::PrintOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::PrintOp op,
                                PatternRewriter &rewriter) const override {

    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    for (auto enumerated_pair :
         llvm::enumerate(llvm::zip_longest(op.strings(), op.values()))) {
      auto pair = enumerated_pair.value();
      Optional<Attribute> stringAttribute = std::get<0>(pair);
      Optional<Value> val = std::get<1>(pair);

      if (stringAttribute) {
        StringRef currentString =
            stringAttribute.getValue().dyn_cast<StringAttr>().getValue();
        callPrintString(rewriter, module, loc, currentString);
      } else if (enumerated_pair.index() != 0)
        callPrintString(rewriter, module, loc, " ");

      if (val)
        callPrintValue(rewriter, module, loc, val.getValue());
      else
        callPrintString(rewriter, module, loc, " ");
    }
    callPrintString(rewriter, module, loc, "\n");

    rewriter.eraseOp(op);

    return success();
  };
};

class LowerMatrixSelectRandomRewrite
    : public OpRewritePattern<graphblas::MatrixSelectRandomOp> {
public:
  using OpRewritePattern<graphblas::MatrixSelectRandomOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::MatrixSelectRandomOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    ModuleOp module = op->getParentOfType<ModuleOp>();

    Value input = op.input();
    Value n = op.n();
    Value rngContext = op.rng_context();
    SymbolRefAttr chooseNSymbol = op.choose_n();

    Type valueType = input.getType().dyn_cast<TensorType>().getElementType();
    Type int64Type = rewriter.getIntegerType(64);
    Type indexType = rewriter.getIndexType();
    Type memref1DI64Type = MemRefType::get({-1}, int64Type);
    Type memref1DValueType = MemRefType::get({-1}, valueType);

    // Initial constants
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c0_64 = rewriter.create<arith::ConstantIntOp>(loc, 0, int64Type);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Get sparse tensor info
    Value nrow = rewriter.create<graphblas::NumRowsOp>(loc, input);
    Value Ap = rewriter.create<sparse_tensor::ToPointersOp>(
        loc, memref1DI64Type, input, c1);
    Value Aj = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                           input, c1);
    Value Ax = rewriter.create<sparse_tensor::ToValuesOp>(
        loc, memref1DValueType, input);

    // Create output tensor
    Value output = rewriter.create<graphblas::DupOp>(loc, input);
    Value Bp = rewriter.create<sparse_tensor::ToPointersOp>(
        loc, memref1DI64Type, output, c1);
    Value Bj = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                           output, c1);
    Value Bx = rewriter.create<sparse_tensor::ToValuesOp>(
        loc, memref1DValueType, output);
    rewriter.create<memref::StoreOp>(loc, c0_64, Bp, c0);

    // Pass 1: Scan input tensor to compute offsets
    scf::ForOp scanLoop = rewriter.create<scf::ForOp>(loc, c0, nrow, c1);
    Value row = scanLoop.getInductionVar();

    rewriter.setInsertionPointToStart(scanLoop.getBody());
    Value row_plus1 = rewriter.create<arith::AddIOp>(loc, row, c1);
    Value Aj_start_64 = rewriter.create<memref::LoadOp>(loc, Ap, row);
    Value Aj_end_64 = rewriter.create<memref::LoadOp>(loc, Ap, row_plus1);

    // Limit number of row values in output to n
    Value Aj_size_64 =
        rewriter.create<arith::SubIOp>(loc, Aj_end_64, Aj_start_64);
    Value isRowSmall = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ule, Aj_size_64, n);
    Value Bj_size_64 =
        rewriter.create<SelectOp>(loc, isRowSmall, Aj_size_64, n);

    Value Bj_start_64 = rewriter.create<memref::LoadOp>(loc, Bp, row);
    Value Bj_end_64 =
        rewriter.create<arith::AddIOp>(loc, Bj_start_64, Bj_size_64);
    rewriter.create<memref::StoreOp>(loc, Bj_end_64, Bp, row_plus1);

    rewriter.setInsertionPointAfter(scanLoop);

    // Pass 2: Parallel select and compute output
    scf::ParallelOp rowLoop =
        rewriter.create<scf::ParallelOp>(loc, c0, nrow, c1);
    row = rowLoop.getInductionVars()[0];

    rewriter.setInsertionPointToStart(rowLoop.getBody());

    row_plus1 = rewriter.create<arith::AddIOp>(loc, row, c1);
    Aj_start_64 = rewriter.create<memref::LoadOp>(loc, Ap, row);
    Value Aj_start =
        rewriter.create<arith::IndexCastOp>(loc, Aj_start_64, indexType);
    Aj_end_64 = rewriter.create<memref::LoadOp>(loc, Ap, row_plus1);
    Value Aj_end =
        rewriter.create<arith::IndexCastOp>(loc, Aj_end_64, indexType);
    Bj_start_64 = rewriter.create<memref::LoadOp>(loc, Bp, row);
    Value Bj_start =
        rewriter.create<arith::IndexCastOp>(loc, Bj_start_64, indexType);
    Bj_end_64 = rewriter.create<memref::LoadOp>(loc, Bp, row_plus1);
    Value Bj_end =
        rewriter.create<arith::IndexCastOp>(loc, Bj_end_64, indexType);

    Value Aj_size = rewriter.create<arith::SubIOp>(loc, Aj_end, Aj_start);
    Aj_size_64 = rewriter.create<arith::IndexCastOp>(loc, Aj_size, int64Type);
    Value Bj_size = rewriter.create<arith::SubIOp>(loc, Bj_end, Bj_start);
    Bj_size_64 = rewriter.create<arith::IndexCastOp>(loc, Bj_size, int64Type);
    Value copyRow = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, Aj_size, Bj_size);

    // Create output subviews
    Value Bj_view =
        rewriter.create<memref::SubViewOp>(loc, Bj, Bj_start, Bj_size, c1);
    Value Bx_view =
        rewriter.create<memref::SubViewOp>(loc, Bx, Bj_start, Bj_size, c1);
    Value Aj_view =
        rewriter.create<memref::SubViewOp>(loc, Aj, Aj_start, Aj_size, c1);
    Value Ax_view =
        rewriter.create<memref::SubViewOp>(loc, Ax, Aj_start, Aj_size, c1);

    // If number of row values less than or equal to n, copy row directly
    scf::IfOp ifCopy = rewriter.create<scf::IfOp>(loc, copyRow, true);

    rewriter.setInsertionPointToStart(ifCopy.thenBlock());

    // copy contents
    rewriter.create<memref::CopyOp>(loc, Aj_view, Bj_view);
    rewriter.create<memref::CopyOp>(loc, Ax_view, Bx_view);

    // Else, fill output row with random selection from input row
    rewriter.setInsertionPointToStart(ifCopy.elseBlock());

    // These are unused. Should they be removed?
    // MLIRContext *context = module.getContext();
    // FuncOp chooseFunc = module.lookupSymbol<FuncOp>(chooseNSymbol);

    // TODO: Verify signature of this function is what we expect

    // Call function using output Bj row as temporary storage
    rewriter.create<mlir::CallOp>(
        loc, chooseNSymbol, TypeRange(),
        ArrayRef<Value>(
            {rngContext, Bj_size_64, Aj_size_64, Bj_view, Ax_view}));

    // Loop over randomly selected offsets
    scf::ParallelOp colLoop =
        rewriter.create<scf::ParallelOp>(loc, c0, Bj_size, c1);
    Value offset = colLoop.getInductionVars()[0];

    rewriter.setInsertionPointToStart(colLoop.getBody());

    Value sourceOffset_64 =
        rewriter.create<memref::LoadOp>(loc, Bj_view, offset);
    Value sourceOffset =
        rewriter.create<arith::IndexCastOp>(loc, sourceOffset_64, indexType);
    Value colIndex =
        rewriter.create<memref::LoadOp>(loc, Aj_view, sourceOffset);
    Value colValue =
        rewriter.create<memref::LoadOp>(loc, Ax_view, sourceOffset);
    // overwrite the randomly selected offset with the actual column index
    rewriter.create<memref::StoreOp>(loc, colIndex, Bj_view, offset);
    // write the corresponding value from source matrix
    rewriter.create<memref::StoreOp>(loc, colValue, Bx_view, offset);

    // end loop over columns

    // end loop over rows

    // Output array is populated
    rewriter.setInsertionPointAfter(rowLoop);
    // Resize output index and values to match total number of elements
    Value outputNNZ_64 = rewriter.create<memref::LoadOp>(loc, Bp, nrow);
    Value outputNNZ =
        rewriter.create<arith::IndexCastOp>(loc, outputNNZ_64, indexType);
    callResizeIndex(rewriter, module, loc, output, c1, outputNNZ);
    callResizeValues(rewriter, module, loc, output, outputNNZ);

    rewriter.replaceOp(op, output);

    return success();
  };
};

void populateGraphBLASLoweringPatterns(RewritePatternSet &patterns) {
  patterns
      .add<LowerMatrixSelectRandomRewrite, LowerSelectRewrite,
           LowerReduceToVectorRewrite, LowerReduceToScalarRewrite,
           LowerReduceToScalarGenericRewrite, LowerMatrixMultiplyRewrite,
           LowerConvertLayoutRewrite, LowerCastRewrite, LowerTransposeRewrite,
           LowerApplyRewrite, LowerApplyGenericRewrite,
           LowerMatrixMultiplyReduceToScalarGenericRewrite,
           LowerMatrixMultiplyGenericRewrite, LowerUnionRewrite,
           LowerIntersectRewrite, LowerUpdateRewrite, LowerEqualRewrite,
           LowerDiagOpRewrite, LowerCommentRewrite, LowerPrintRewrite,
           LowerSizeRewrite, LowerNumRowsRewrite, LowerNumColsRewrite,
           LowerNumValsRewrite, LowerDupRewrite>(patterns.getContext());
}

struct GraphBLASLoweringPass
    : public GraphBLASLoweringBase<GraphBLASLoweringPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);
    populateGraphBLASLoweringPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    // TODO how can we mark graphblas ops as illegal here?
  }
};

void populateGraphBLASStructuralizePatterns(RewritePatternSet &patterns) {
  patterns.add<LowerMatrixMultiplyRewrite, LowerApplyRewrite,
               LowerReduceToScalarRewrite>(patterns.getContext());
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

std::unique_ptr<OperationPass<ModuleOp>> mlir::createGraphBLASLoweringPass() {
  return std::make_unique<GraphBLASLoweringPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createGraphBLASStructuralizePass() {
  return std::make_unique<GraphBLASStructuralizePass>();
}
