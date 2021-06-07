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

class LowerTransposeRewrite : public OpRewritePattern<graphblas::TransposeOp> {
public:
  using OpRewritePattern<graphblas::TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::TransposeOp op, PatternRewriter &rewriter) const {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();
    
    Value inputTensor = op.input();
    Type valueType = inputTensor.getType().dyn_cast<RankedTensorType>().getElementType();
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
    Value nrow = rewriter.create<memref::DimOp>(loc, inputTensor, c0);
    Value ncol = rewriter.create<memref::DimOp>(loc, inputTensor, c1);
    Value ncols_plus_one = rewriter.create<mlir::AddIOp>(loc, ncol, c1);

    Value nnz_64 = rewriter.create<memref::LoadOp>(loc, inputPtrs, nrow);
    Value nnz = rewriter.create<mlir::IndexCastOp>(loc, nnz_64, indexType);

    Value output = callEmptyLike(rewriter, module, loc, inputTensor).getResult(0);
    bool swap_sizes = op->getAttr("swap_sizes").dyn_cast<BoolAttr>().getValue();
    if (swap_sizes)
    {
      callResizeDim(rewriter, module, loc, output, c0, ncol);
      callResizeDim(rewriter, module, loc, output, c1, nrow);
    }
    else
    {
      callResizeDim(rewriter, module, loc, output, c0, nrow);
      callResizeDim(rewriter, module, loc, output, c1, ncol);
    }

    callResizePointers(rewriter, module, loc, output, c1, ncols_plus_one);
    callResizeIndex(rewriter, module, loc, output, c1, nnz);
    callResizeValues(rewriter, module, loc, output, nnz);

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

    return success();
  };
};

class LowerMatrixSelectRewrite : public OpRewritePattern<graphblas::MatrixSelectOp> {
public:
  using OpRewritePattern<graphblas::MatrixSelectOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::MatrixSelectOp op, PatternRewriter &rewriter) const {
    // TODO sanity check that the sparse encoding is sane
    return failure();
  };
};

class LowerMatrixReduceToScalarRewrite : public OpRewritePattern<graphblas::MatrixReduceToScalarOp> {
public:
  using OpRewritePattern<graphblas::MatrixReduceToScalarOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::MatrixReduceToScalarOp op, PatternRewriter &rewriter) const {
    MLIRContext *context = op->getContext();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = rewriter.getUnknownLoc();

    RankedTensorType operandType = op.input().getType().dyn_cast<RankedTensorType>();
    ArrayRef<int64_t> operandShape = operandType.getShape();
    Type valueType = operandType.getElementType();
    Type int64Type = rewriter.getIntegerType(64); // TODO should we get this from the sparse encoding?
    Type indexType = rewriter.getIndexType();
    RankedTensorType csrTensorType = getCSRTensorType(context, operandShape, valueType);
    
    // TODO should this name also account for the dimensions of the input? Or should we fail upon certain dimensions/rank?
    std::string funcName = "matrix_reduce_to_scalar_";
    llvm::raw_string_ostream stream(funcName);
    std::string aggregator = op.aggregator().str();
    stream <<  aggregator << "_elem_";
    valueType.print(stream);
    stream.flush();
    
    FuncOp func = module.lookupSymbol<FuncOp>(funcName);
    if (!func) {
      OpBuilder moduleBuilder(module.getBodyRegion());
      
      FunctionType funcType = FunctionType::get(context, {csrTensorType}, valueType);
      moduleBuilder.create<FuncOp>(op->getLoc(), funcName, funcType).setPrivate();
      func = module.lookupSymbol<FuncOp>(funcName);
      Block &entry_block = *func.addEntryBlock();
      moduleBuilder.setInsertionPointToStart(&entry_block);
      BlockArgument input = entry_block.getArgument(0);
      
      // Initial constants
      llvm::Optional<ConstantOp> c0Accumulator = llvm::TypeSwitch<Type, llvm::Optional<ConstantOp>>(valueType)
        .Case<IntegerType>([&](IntegerType type) {
                             return moduleBuilder.create<ConstantIntOp>(loc, 0, type.getWidth());
                           })
        .Case<FloatType>([&](FloatType type) {
                           return moduleBuilder.create<ConstantFloatOp>(loc, APFloat(type.getFloatSemantics()), type);
                         })
        .Default([&](Type type) { return llvm::None; });
      if (!c0Accumulator.hasValue()) {
        return failure(); // TODO test this case
      }
      ConstantIndexOp c0 = moduleBuilder.create<ConstantIndexOp>(loc, 0);
      ConstantIndexOp c1 = moduleBuilder.create<ConstantIndexOp>(loc, 1);
      
      // Get sparse tensor info
      MemRefType memref1DI64Type = MemRefType::get({-1}, int64Type);
      MemRefType memref1DValueType = MemRefType::get({-1}, valueType);

      memref::DimOp nrows = moduleBuilder.create<memref::DimOp>(loc, input, c0.getResult());
      sparse_tensor::ToPointersOp inputPtrs = moduleBuilder.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, input, c1);
      sparse_tensor::ToValuesOp inputValues = moduleBuilder.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, input);
      memref::LoadOp nnz64 = moduleBuilder.create<memref::LoadOp>(loc, inputPtrs, nrows.getResult());
      IndexCastOp nnz = moduleBuilder.create<IndexCastOp>(loc, nnz64, indexType);

      // begin loop
      scf::ParallelOp valueLoop = moduleBuilder.create<scf::ParallelOp>(loc, c0.getResult(), nnz.getResult(), c1.getResult(), c0Accumulator.getValue().getResult());
      ValueRange valueLoopIdx = valueLoop.getInductionVars();

      moduleBuilder.setInsertionPointToStart(valueLoop.getBody());
      memref::LoadOp y = moduleBuilder.create<memref::LoadOp>(loc, inputValues, valueLoopIdx);

      scf::ReduceOp reducer = moduleBuilder.create<scf::ReduceOp>(loc, y);
      BlockArgument lhs = reducer.getRegion().getArgument(0);
      BlockArgument rhs = reducer.getRegion().getArgument(1);

      moduleBuilder.setInsertionPointToStart(&reducer.getRegion().front());

      llvm::Optional<Value> z;
      if (aggregator == "sum") {
         z = llvm::TypeSwitch<Type, llvm::Optional<Value>>(valueType)
          .Case<IntegerType>([&](IntegerType type) { return moduleBuilder.create<AddIOp>(loc, lhs, rhs).getResult(); })
          .Case<FloatType>([&](FloatType type) { return moduleBuilder.create<AddFOp>(loc, lhs, rhs).getResult(); })
          .Default([&](Type type) { return llvm::None; });
        if (!z.hasValue()) {
          return failure();
        }
      } else {
        return failure(); // TODO test this
      }
      moduleBuilder.create<scf::ReduceReturnOp>(loc, z.getValue());

      moduleBuilder.setInsertionPointAfter(reducer);

      // end loop
      moduleBuilder.setInsertionPointAfter(valueLoop);

      // Add return op
      moduleBuilder.create<ReturnOp>(loc, valueLoop.getResult(0));
    }
    FlatSymbolRefAttr funcSymbol = SymbolRefAttr::get(context, funcName);
    
    Value inputTensor = op.input();
    CallOp callOp = rewriter.create<CallOp>(loc,
                                            funcSymbol,
                                            valueType,
                                            llvm::ArrayRef<Value>({inputTensor})
                                            );    
    rewriter.replaceOp(op, callOp->getResults());
    
    return success();
  };
};

class LowerMatrixApplyRewrite : public OpRewritePattern<graphblas::MatrixApplyOp> {
public:
  using OpRewritePattern<graphblas::MatrixApplyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::MatrixApplyOp op, PatternRewriter &rewriter) const {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    Type valueType = op.input().getType().dyn_cast<RankedTensorType>().getElementType();
    Type int64Type = rewriter.getIntegerType(64);
    Type indexType = rewriter.getIndexType();

    Type memref1DI64Type = MemRefType::get({-1}, int64Type);
    Type memref1DValueType = MemRefType::get({-1}, valueType);

    Value inputTensor = op.input();
    Value thunk = op.thunk();
    StringRef apply_operator = op.apply_operator();

    // Initial constants
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);

    // Get sparse tensor info
    Value output = callDupTensor(rewriter, module, loc, inputTensor).getResult(0);
    Value inputPtrs = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, inputTensor, c1);
    Value inputValues = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, inputTensor);
    Value outputValues = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, output);

    Value nrows = rewriter.create<memref::DimOp>(loc, inputTensor, c0);
    Value nnz64 = rewriter.create<memref::LoadOp>(loc, inputPtrs, nrows);
    Value nnz = rewriter.create<mlir::IndexCastOp>(loc, nnz64, indexType);

    // Loop over values
    scf::ParallelOp valueLoop = rewriter.create<scf::ParallelOp>(loc, c0, nnz, c1);
    ValueRange valueLoopIdx = valueLoop.getInductionVars();

    rewriter.setInsertionPointToStart(valueLoop.getBody());
    Value val = rewriter.create<memref::LoadOp>(loc, inputValues, valueLoopIdx);

    Value result;
    if (apply_operator == "min")
    {
      Value cmp = rewriter.create<mlir::CmpFOp>(loc, mlir::CmpFPredicate::OLT, val, thunk);
      result = rewriter.create<mlir::SelectOp>(loc, cmp, val, thunk);
    };

    rewriter.create<memref::StoreOp>(loc, result, outputValues, valueLoopIdx);

    // end value loop
    rewriter.setInsertionPointAfter(valueLoop);

    // Add return op
    rewriter.replaceOp(op, output);

    return success();
  };
};

class LowerMatrixMultiplyRewrite : public OpRewritePattern<graphblas::MatrixMultiplyOp> {
public:
  using OpRewritePattern<graphblas::MatrixMultiplyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::MatrixMultiplyOp op, PatternRewriter &rewriter) const {
    MLIRContext *context = op->getContext();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = rewriter.getUnknownLoc();

    // Inputs
    Value A = op.a();
    Value B = op.b();
    Value mask = op.mask();
    StringRef semiring = op.semiring();

    // Types
    Type indexType = rewriter.getIndexType();
    Type int64Type = rewriter.getIntegerType(64);
    Type boolType = rewriter.getI1Type();
    Type valueType = op.getResult().getType().dyn_cast<RankedTensorType>().getElementType();
    ArrayRef<int64_t> shape = {-1, -1};
    RankedTensorType csrTensorType = getCSRTensorType(context, shape, valueType);
    RankedTensorType cscTensorType = getCSCTensorType(context, shape, valueType);
    MemRefType memref1DI64Type = MemRefType::get({-1}, int64Type);
    MemRefType memref1DBoolType = MemRefType::get({-1}, boolType);
    MemRefType memref1DValueType = MemRefType::get({-1}, valueType);

    // Initial constants
    Value cf0, cf1;
    cf0 = llvm::TypeSwitch<Type, Value>(valueType)
        .Case<IntegerType>([&](IntegerType type) { return rewriter.create<ConstantIntOp>(loc, 0, type.getWidth()); })
        .Case<FloatType>([&](FloatType type) { return rewriter.create<ConstantFloatOp>(loc, APFloat(0.0), type); });
    cf1 = llvm::TypeSwitch<Type, Value>(valueType)
        .Case<IntegerType>([&](IntegerType type) { return rewriter.create<ConstantIntOp>(loc, 1, type.getWidth()); })
        .Case<FloatType>([&](FloatType type) { return rewriter.create<ConstantFloatOp>(loc, APFloat(1.0), type); });
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
    Value ci0 = rewriter.create<ConstantIntOp>(loc, 0, int64Type);
    Value ci1 = rewriter.create<ConstantIntOp>(loc, 1, int64Type);
    Value cfalse = rewriter.create<ConstantIntOp>(loc, 0, boolType);
    Value ctrue = rewriter.create<ConstantIntOp>(loc, 1, boolType);

    // Get sparse tensor info
    Value Ap = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, A, c1);
    Value Aj = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, A, c1);
    Value Ax = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, A);
    Value Bp = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, B, c1);
    Value Bi = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, B, c1);
    Value Bx = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, B);

    Value nrow = rewriter.create<memref::DimOp>(loc, A, c0);
    Value ncol = rewriter.create<memref::DimOp>(loc, B, c1);
    Value nk = rewriter.create<memref::DimOp>(loc, A, c1);
    Value nrow_plus_one = rewriter.create<AddIOp>(loc, nrow, c1);

    Value Mp, Mj;
    if (mask) {
        Mp = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, mask, c1);
        Mj = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, mask, c1);
    }

    Value C = callEmptyLike(rewriter, module, loc, A).getResult(0);
    callResizeDim(rewriter, module, loc, C, c0, nrow);
    callResizeDim(rewriter, module, loc, C, c1, ncol);
    callResizePointers(rewriter, module, loc, C, c1, nrow_plus_one);

    Value Cp = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, C, c1);

    // 1st pass
    //   Using nested parallel loops for each row and column,
    //   compute the number of nonzero entries per row.
    //   Store results in Cp
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

    // Construct a dense array indicating valid row positions
    Value colStart = rewriter.create<IndexCastOp>(loc, colStart64, indexType);
    Value colEnd = rewriter.create<IndexCastOp>(loc, colEnd64, indexType);
    Value kvec_i1 = rewriter.create<memref::AllocOp>(loc, memref1DBoolType, nk);
    rewriter.create<linalg::FillOp>(loc, kvec_i1, cfalse);
    scf::ParallelOp colLoop1 = rewriter.create<scf::ParallelOp>(loc, colStart, colEnd, c1);
    Value jj = colLoop1.getInductionVars()[0];
    rewriter.setInsertionPointToStart(colLoop1.getBody());
    Value col64 = rewriter.create<memref::LoadOp>(loc, Aj, jj);
    Value col = rewriter.create<IndexCastOp>(loc, col64, indexType);
    rewriter.create<memref::StoreOp>(loc, ctrue, kvec_i1, col);
    rewriter.setInsertionPointAfter(colLoop1);

    // Loop thru all columns; count number of resulting nonzeros in the row
    if (mask) {
        Value mcolStart64 = rewriter.create<memref::LoadOp>(loc, Mp, row);
        Value mcolEnd64 = rewriter.create<memref::LoadOp>(loc, Mp, rowPlus1);
        Value mcolStart = rewriter.create<IndexCastOp>(loc, mcolStart64, indexType);
        Value mcolEnd = rewriter.create<IndexCastOp>(loc, mcolEnd64, indexType);

        colLoop1 = rewriter.create<scf::ParallelOp>(loc, mcolStart, mcolEnd, c1, ci0);
        Value mm = colLoop1.getInductionVars()[0];
        rewriter.setInsertionPointToStart(colLoop1.getBody());
        col64 = rewriter.create<memref::LoadOp>(loc, Mj, mm);
        col = rewriter.create<IndexCastOp>(loc, col64, indexType);
    } else {
        colLoop1 = rewriter.create<scf::ParallelOp>(loc, c0, ncol, c1, ci0);
        col = colLoop1.getInductionVars()[0];
        rewriter.setInsertionPointToStart(colLoop1.getBody());
    }

    Value colPlus1 = rewriter.create<AddIOp>(loc, col, c1);
    Value rowStart64 = rewriter.create<memref::LoadOp>(loc, Bp, col);
    Value rowEnd64 = rewriter.create<memref::LoadOp>(loc, Bp, colPlus1);
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
    Value kk64 = rewriter.create<memref::LoadOp>(loc, Bi, ii);
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

    auto reducer = rewriter.create<scf::ReduceOp>(loc, overlap);
    Value lhs = reducer.getRegion().getArgument(0);
    Value rhs = reducer.getRegion().getArgument(1);
    rewriter.setInsertionPointToStart(&reducer.getRegion().front());
    Value z = rewriter.create<AddIOp>(loc, lhs, rhs);
    rewriter.create<scf::ReduceReturnOp>(loc, z);

    // end col loop
    rewriter.setInsertionPointAfter(colLoop1);
    Value total = colLoop1.getResult(0);
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
    Value cumsum2 = rewriter.create<AddIOp>(loc, cumsum, csTemp);
    rewriter.create<memref::StoreOp>(loc, cumsum2, Cp, nrow);

    // end row loop
    rewriter.setInsertionPointAfter(rowLoop2);

    Value nnz64 = rewriter.create<memref::LoadOp>(loc, Cp, nrow);
    Value nnz = rewriter.create<IndexCastOp>(loc, nnz64, indexType);
    callResizeIndex(rewriter, module, loc, C, c1, nnz);
    callResizeValues(rewriter, module, loc, C, nnz);
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

    // Construct a dense array of row values
    colStart64 = rewriter.create<memref::LoadOp>(loc, Ap, row);
    colEnd64 = rewriter.create<memref::LoadOp>(loc, Ap, rowPlus1);
    colStart = rewriter.create<IndexCastOp>(loc, colStart64, indexType);
    colEnd = rewriter.create<IndexCastOp>(loc, colEnd64, indexType);
    Value kvec = rewriter.create<memref::AllocOp>(loc, memref1DValueType, nk);
    kvec_i1 = rewriter.create<memref::AllocOp>(loc, memref1DBoolType, nk);
    rewriter.create<linalg::FillOp>(loc, kvec_i1, cfalse);
    scf::ParallelOp colLoop3p = rewriter.create<scf::ParallelOp>(loc, colStart, colEnd, c1);
    jj = colLoop3p.getInductionVars()[0];
    rewriter.setInsertionPointToStart(colLoop3p.getBody());
    col64 = rewriter.create<memref::LoadOp>(loc, Aj, jj);
    col = rewriter.create<IndexCastOp>(loc, col64, indexType);
    rewriter.create<memref::StoreOp>(loc, ctrue, kvec_i1, col);
    Value val = rewriter.create<memref::LoadOp>(loc, Ax, jj);
    rewriter.create<memref::StoreOp>(loc, val, kvec, col);

    // end col loop 3p
    rewriter.setInsertionPointAfter(colLoop3p);

    scf::ForOp colLoop3f;
    if (mask) {
        Value mcolStart64 = rewriter.create<memref::LoadOp>(loc, Mp, row);
        Value mcolEnd64 = rewriter.create<memref::LoadOp>(loc, Mp, rowPlus1);
        Value mcolStart = rewriter.create<IndexCastOp>(loc, mcolStart64, indexType);
        Value mcolEnd = rewriter.create<IndexCastOp>(loc, mcolEnd64, indexType);

        colLoop3f = rewriter.create<scf::ForOp>(loc, mcolStart, mcolEnd, c1, c0);
        Value mm = colLoop3f.getInductionVar();
        rewriter.setInsertionPointToStart(colLoop3f.getBody());
        col64 = rewriter.create<memref::LoadOp>(loc, Mj, mm);
        col = rewriter.create<IndexCastOp>(loc, col64, indexType);
    } else {
        colLoop3f = rewriter.create<scf::ForOp>(loc, c0, ncol, c1, c0);
        col = colLoop3f.getInductionVar();
        rewriter.setInsertionPointToStart(colLoop3f.getBody());
        col64 = rewriter.create<IndexCastOp>(loc, col, int64Type);
    }

    Value offset = colLoop3f.getLoopBody().getArgument(1);
    colPlus1 = rewriter.create<AddIOp>(loc, col, c1);
    Value iStart64 = rewriter.create<memref::LoadOp>(loc, Bp, col);
    Value iEnd64 = rewriter.create<memref::LoadOp>(loc, Bp, colPlus1);
    Value iStart = rewriter.create<IndexCastOp>(loc, iStart64, indexType);
    Value iEnd = rewriter.create<IndexCastOp>(loc, iEnd64, indexType);

    scf::ForOp kLoop = rewriter.create<scf::ForOp>(loc, iStart, iEnd, c1, ValueRange{cf0, cfalse});
    ii = kLoop.getInductionVar();
    Value curr = kLoop.getLoopBody().getArgument(1);
    Value alive = kLoop.getLoopBody().getArgument(2);
    rewriter.setInsertionPointToStart(kLoop.getBody());

    kk64 = rewriter.create<memref::LoadOp>(loc, Bi, ii);
    kk = rewriter.create<IndexCastOp>(loc, kk64, indexType);
    cmpPair = rewriter.create<memref::LoadOp>(loc, kvec_i1, kk);
    scf::IfOp ifBlock_cmpPair = rewriter.create<scf::IfOp>(loc, ArrayRef<Type>{valueType, boolType}, cmpPair, true);
    // if cmpPair
    rewriter.setInsertionPointToStart(ifBlock_cmpPair.thenBlock());
    Value newVal;
    if (semiring == "plus_pair") {
        newVal = rewriter.create<AddFOp>(loc, curr, cf1);
    } else {
        Value aVal = rewriter.create<memref::LoadOp>(loc, kvec, kk);
        Value bVal = rewriter.create<memref::LoadOp>(loc, Bx, ii);
        if (semiring == "plus_times") {
            val = rewriter.create<MulFOp>(loc, aVal, bVal);
            newVal = rewriter.create<AddFOp>(loc, curr, val);
        } else if (semiring == "plus_plus") {
            val = rewriter.create<AddFOp>(loc, aVal, bVal);
            newVal = rewriter.create<AddFOp>(loc, curr, val);
        }
    }
    rewriter.create<scf::YieldOp>(loc, ValueRange{newVal, ctrue});

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

    total = kLoop.getResult(0);
    Value notEmpty = kLoop.getResult(1);

    scf::IfOp ifBlock_newOffset = rewriter.create<scf::IfOp>(loc, indexType, notEmpty, true);
    // if not empty
    rewriter.setInsertionPointToStart(ifBlock_newOffset.thenBlock());

    // Store total in Cx
    Value cjPos = rewriter.create<AddIOp>(loc, baseIndex, offset);
    rewriter.create<memref::StoreOp>(loc, col64, Cj, cjPos);
    rewriter.create<memref::StoreOp>(loc, total, Cx, cjPos);
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

    // end if cmpDiff
    rewriter.setInsertionPointAfter(ifBlock_cmpDiff);

    // end row loop
    rewriter.setInsertionPointAfter(rowLoop3);

    rewriter.replaceOp(op, C);

    return success();
  };
};

void populateGraphBLASLoweringPatterns(RewritePatternSet &patterns) {
  patterns.add<
    LowerMatrixReduceToScalarRewrite,
    LowerMatrixMultiplyRewrite,
    LowerTransposeRewrite,
    LowerMatrixApplyRewrite
    >(patterns.getContext());
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

} // end anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createGraphBLASLoweringPass() {
  return std::make_unique<GraphBLASLoweringPass>();
}
