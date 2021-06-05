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
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();

    Value input = op.input();
    Type valueType = input.getType().dyn_cast<TensorType>().getElementType();
    Type int64Type = rewriter.getIntegerType(64);
    FloatType float64Type = rewriter.getF64Type();
    Type indexType = rewriter.getIndexType();
    Type memref1DI64Type = MemRefType::get({-1}, int64Type);
    Type memref1DValueType = MemRefType::get({-1}, valueType);

    StringRef selector = op.selector();

    bool needs_col = false, needs_val = false;
    if (selector == "triu")
    {
      needs_col = true;
      needs_val = false;
    }
    else if (selector == "tril")
    {
      needs_col = true;
      needs_val = false;
    }
    else if (selector == "gt0")
    {
      needs_col = false;
      needs_val = true;
    }
    else
    {
      return failure();
    }

    // Initial constants
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
    Value c0_64 = rewriter.create<ConstantIntOp>(loc, 0, int64Type);
    Value c1_64 = rewriter.create<ConstantIntOp>(loc, 1, int64Type);
    Value cf0 = rewriter.create<ConstantFloatOp>(loc, APFloat(0.0), float64Type);

    // Get sparse tensor info
    Value nrow = rewriter.create<memref::DimOp>(loc, input, c0);
    Value Ap = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, input, c1);
    Value Aj = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, input, c1);
    Value Ax = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, input);

    Value output = callDupTensor(rewriter, module, loc, input).getResult(0);
    Value Bp = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, output, c1);
    Value Bj = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, output, c1);
    Value Bx = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, output);

    rewriter.create<memref::StoreOp>(loc, c0_64, Bp, c0);
    // Loop
    scf::ForOp outerLoop = rewriter.create<scf::ForOp>(loc, c0, nrow, c1);
    Value row = outerLoop.getInductionVar();

    rewriter.setInsertionPointToStart(outerLoop.getBody());
    Value row_plus1 = rewriter.create<mlir::AddIOp>(loc, row, c1);
    Value bp_curr_count = rewriter.create<memref::LoadOp>(loc, Bp, row);
    rewriter.create<memref::StoreOp>(loc, bp_curr_count, Bp, row_plus1);

    Value j_start_64 = rewriter.create<memref::LoadOp>(loc, Ap, row);
    Value j_end_64 = rewriter.create<memref::LoadOp>(loc, Ap, row_plus1);
    Value j_start = rewriter.create<mlir::IndexCastOp>(loc, j_start_64, indexType);
    Value j_end = rewriter.create<mlir::IndexCastOp>(loc, j_end_64, indexType);

    scf::ForOp innerLoop = rewriter.create<scf::ForOp>(loc, j_start, j_end, c1);

    Value jj = innerLoop.getInductionVar();

    rewriter.setInsertionPointToStart(innerLoop.getBody());
    Value col_64, col, val, keep;
    if (needs_col)
    {
      col_64 = rewriter.create<memref::LoadOp>(loc, Aj, jj);
      col = rewriter.create<mlir::IndexCastOp>(loc, col_64, indexType);
    }
    if (needs_val)
    {
      val = rewriter.create<memref::LoadOp>(loc, Ax, jj);
    }
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
      return failure();
    }

    scf::IfOp ifKeep = rewriter.create<scf::IfOp>(loc, keep, false /* no else region */);

    rewriter.setInsertionPointToStart(ifKeep.thenBlock());

    Value bj_pos_64 = rewriter.create<memref::LoadOp>(loc, Bp, row_plus1);
    Value bj_pos = rewriter.create<mlir::IndexCastOp>(loc, bj_pos_64, indexType);

    if (!needs_col)
    {
      col_64 = rewriter.create<memref::LoadOp>(loc, Aj, jj);
    }
    rewriter.create<memref::StoreOp>(loc, col_64, Bj, bj_pos);

    if (!needs_val)
    {
      val = rewriter.create<memref::LoadOp>(loc, Ax, jj);
    }
    rewriter.create<memref::StoreOp>(loc, val, Bx, bj_pos);

    Value bj_pos_plus1 = rewriter.create<mlir::AddIOp>(loc, bj_pos_64, c1_64);
    rewriter.create<memref::StoreOp>(loc, bj_pos_plus1, Bp, row_plus1);

    rewriter.setInsertionPointAfter(outerLoop);

    // trim excess values
    Value nnz_64 = rewriter.create<memref::LoadOp>(loc, Bp, nrow);
    Value nnz = rewriter.create<mlir::IndexCastOp>(loc, nnz_64, indexType);

    callResizeIndex(rewriter, module, loc, output, c1, nnz);
    callResizeValues(rewriter, module, loc, output, nnz);

    rewriter.replaceOp(op, output);

    return success();
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
    // TODO sanity check that the sparse encoding is sane
    // TODO handle the mask
    // TODO there should be a "return failure();" somewhere
    
    MLIRContext *context = op->getContext();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    
    Type valueType = rewriter.getI64Type();
    ArrayRef<int64_t> shape = {-1, -1};
    RankedTensorType csrTensorType = getCSRTensorType(context, shape, valueType);

    std::string funcName = "matrix_multiply_" + op.semiring().str();
    FuncOp func = module.lookupSymbol<FuncOp>(funcName);
    if (!func) {
      OpBuilder moduleBuilder(module.getBodyRegion());
      FunctionType funcType = FunctionType::get(context, {csrTensorType, csrTensorType}, csrTensorType);
      moduleBuilder.create<FuncOp>(op->getLoc(), funcName, funcType).setPrivate();
    }
    FlatSymbolRefAttr funcSymbol = SymbolRefAttr::get(context, funcName);
    
    Value a = op.a();
    Value b = op.b();
    Location loc = rewriter.getUnknownLoc();
    
    CallOp callOp = rewriter.create<CallOp>(loc,
                                            funcSymbol,
                                            csrTensorType,
                                            llvm::ArrayRef<Value>({a, b})
                                            );
    
    rewriter.replaceOp(op, callOp->getResults());
    
    return success();
  };
};

void populateGraphBLASLoweringPatterns(RewritePatternSet &patterns) {
  patterns.add<
    LowerMatrixSelectRewrite,
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
