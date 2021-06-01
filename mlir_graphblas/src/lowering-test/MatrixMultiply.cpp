#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"

#include "lowering.h"

using namespace std;
using namespace mlir;
using namespace mlir::sparse_tensor;

void addMatrixMultiplyFunc(mlir::ModuleOp mod, const std::string &semiring, bool mask)
{
    MLIRContext *context = mod.getContext();
    OpBuilder builder(mod.getBodyRegion());
    auto loc = builder.getUnknownLoc();

    builder.setInsertionPointToStart(mod.getBody());

    // Types
    auto valueType = builder.getF64Type();
    auto int64Type = builder.getI64Type();
    auto boolType = builder.getI1Type();
    auto indexType = builder.getIndexType();
    auto noneType = builder.getNoneType();
    RankedTensorType csrTensorType = getCSRTensorType(context, valueType);
    auto memref1DI64Type = MemRefType::get({-1}, int64Type);
    auto memref1DValueType = MemRefType::get({-1}, valueType);
    auto memref1DBoolType = MemRefType::get({-1}, boolType);

    // Create function signature
    string func_name;
    FunctionType func_type;

    if (mask) {
        func_name += "matrix_multiply_mask_" + semiring;
        func_type = FunctionType::get(context, {csrTensorType, csrTensorType, csrTensorType}, csrTensorType);
    } else {
        func_name += "matrix_multiply_" + semiring;
        func_type = FunctionType::get(context, {csrTensorType, csrTensorType}, csrTensorType);
    }

    auto func = builder.create<FuncOp>(loc, func_name, func_type);

    // Move to function body
    auto &entry_block = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entry_block);

    auto A = entry_block.getArgument(0);
    auto B = entry_block.getArgument(1);
    Value M;
    if (mask) {
        M = entry_block.getArgument(2);
    }

    // Initial constants
    Value cf0 = builder.create<ConstantFloatOp>(loc, APFloat(0.0), valueType);
    Value cf1 = builder.create<ConstantFloatOp>(loc, APFloat(1.0), valueType);
    Value c0 = builder.create<ConstantIndexOp>(loc, 0);
    Value c1 = builder.create<ConstantIndexOp>(loc, 1);
    Value ci0 = builder.create<ConstantIntOp>(loc, 0, int64Type);
    Value ci1 = builder.create<ConstantIntOp>(loc, 1, int64Type);
    Value cfalse = builder.create<ConstantIntOp>(loc, 0, boolType);
    Value ctrue = builder.create<ConstantIntOp>(loc, 1, boolType);

    // Get sparse tensor info
    Value Ap = builder.create<ToPointersOp>(loc, memref1DI64Type, A, c1);
    Value Aj = builder.create<ToIndicesOp>(loc, memref1DI64Type, A, c1);
    Value Ax = builder.create<ToValuesOp>(loc, memref1DValueType, A);
    Value Bp = builder.create<ToPointersOp>(loc, memref1DI64Type, B, c1);
    Value Bi = builder.create<ToIndicesOp>(loc, memref1DI64Type, B, c1);
    Value Bx = builder.create<ToValuesOp>(loc, memref1DValueType, B);

    Value nrow = builder.create<memref::DimOp>(loc, A, c0);
    Value ncol = builder.create<memref::DimOp>(loc, B, c1);
    Value nk = builder.create<memref::DimOp>(loc, A, c1);
    Value nrow_plus_one = builder.create<AddIOp>(loc, nrow, c1);

    Value Mp, Mj;
    if (mask) {
        Mp = builder.create<ToPointersOp>(loc, memref1DI64Type, M, c1);
        Mj = builder.create<ToIndicesOp>(loc, memref1DValueType, M, c1);
    }

    Value C = callDupTensor(builder, mod, loc, A).getResult(0);
    callResizeDim(builder, mod, loc, C, c0, nrow);
    callResizeDim(builder, mod, loc, C, c1, ncol);
    callResizePointers(builder, mod, loc, C, c1, nrow_plus_one);

    Value Cp = builder.create<ToPointersOp>(loc, memref1DI64Type, C, c1);

    // 1st pass
    //   Using nested parallel loops for each row and column,
    //   compute the number of nonzero entries per row.
    //   Store results in Cp
    scf::ParallelOp rowLoop1 = builder.create<scf::ParallelOp>(loc, c0, nrow, c1);
    Value row = rowLoop1.getInductionVars()[0];
    builder.setInsertionPointToStart(rowLoop1.getBody());

    Value colStart64 = builder.create<memref::LoadOp>(loc, Ap, row);
    Value rowPlus1 = builder.create<AddIOp>(loc, row, c1);
    Value colEnd64 = builder.create<memref::LoadOp>(loc, Ap, rowPlus1);
    Value cmpColSame = builder.create<CmpIOp>(loc, CmpIPredicate::eq, colStart64, colEnd64);

    scf::IfOp ifBlock_rowTotal = builder.create<scf::IfOp>(loc, indexType, cmpColSame, true);
    // if cmpColSame
    builder.setInsertionPointToStart(ifBlock_rowTotal.thenBlock());
    builder.create<scf::YieldOp>(loc, ci0);

    // else
    builder.setInsertionPointToStart(ifBlock_rowTotal.elseBlock());

    // Construct a dense array indicating valid row positions
    Value colStart = builder.create<IndexCastOp>(loc, colStart64, indexType);
    Value colEnd = builder.create<IndexCastOp>(loc, colEnd64, indexType);
    Value kvec_i1 = builder.create<memref::AllocOp>(loc, memref1DBoolType, nk);
    builder.create<linalg::FillOp>(loc, kvec_i1, cfalse);
    scf::ParallelOp colLoop1 = builder.create<scf::ParallelOp>(loc, colStart, colEnd, c1);
    Value jj = colLoop1.getInductionVars()[0];
    builder.setInsertionPointToStart(colLoop1.getBody());
    Value col64 = builder.create<memref::LoadOp>(loc, Aj, jj);
    Value col = builder.create<IndexCastOp>(loc, col64, indexType);
    builder.create<memref::StoreOp>(loc, ctrue, kvec_i1, col);
    builder.setInsertionPointAfter(colLoop1);

    // Loop thru all columns; count number of resulting nonzeros in the row
    if (mask) {
        Value mcolStart64 = builder.create<memref::LoadOp>(loc, Mp, row);
        Value mcolEnd64 = builder.create<memref::LoadOp>(loc, Mp, rowPlus1);
        Value mcolStart = builder.create<IndexCastOp>(loc, mcolStart64, indexType);
        Value mcolEnd = builder.create<IndexCastOp>(loc, mcolEnd64, indexType);

        colLoop1 = builder.create<scf::ParallelOp>(loc, mcolStart, mcolEnd, c1, ci0);
        Value mm = colLoop1.getInductionVars()[0];
        builder.setInsertionPointToStart(colLoop1.getBody());
        col64 = builder.create<memref::LoadOp>(loc, Mj, mm);
        col = builder.create<IndexCastOp>(loc, col64, indexType);
    } else {
        colLoop1 = builder.create<scf::ParallelOp>(loc, c0, ncol, c1, ci0);
        col = colLoop1.getInductionVars()[0];
        builder.setInsertionPointToStart(colLoop1.getBody());
    }

    Value colPlus1 = builder.create<AddIOp>(loc, col, c1);
    Value rowStart64 = builder.create<memref::LoadOp>(loc, Bp, col);
    Value rowEnd64 = builder.create<memref::LoadOp>(loc, Bp, colPlus1);
    Value cmpRowSame = builder.create<CmpIOp>(loc, CmpIPredicate::eq, rowStart64, rowEnd64);

    // Find overlap in column indices with kvec
    scf::IfOp ifBlock_overlap = builder.create<scf::IfOp>(loc, int64Type, cmpRowSame, true);
    // if cmpRowSame
    builder.setInsertionPointToStart(ifBlock_overlap.thenBlock());
    builder.create<scf::YieldOp>(loc, ci0);

    // else
    builder.setInsertionPointToStart(ifBlock_overlap.elseBlock());

    // Walk thru the indices; on a match yield 1, else yield 0
    scf::WhileOp whileLoop = builder.create<scf::WhileOp>(loc, int64Type, rowStart64);
    Block *before = builder.createBlock(&whileLoop.before(), {}, int64Type);
    Block *after = builder.createBlock(&whileLoop.after(), {}, int64Type);
    Value ii64 = before->getArgument(0);
    builder.setInsertionPointToStart(&whileLoop.before().front());

    // Check if ii >= rowEnd
    Value cmpEndReached = builder.create<CmpIOp>(loc, CmpIPredicate::uge, ii64, rowEnd64);
    scf::IfOp ifBlock_continueSearch = builder.create<scf::IfOp>(loc, ArrayRef<Type>{boolType, int64Type}, cmpEndReached, true);

    // if cmpEndReached
    builder.setInsertionPointToStart(ifBlock_continueSearch.thenBlock());
    builder.create<scf::YieldOp>(loc, ValueRange{cfalse, ci0});

    // else
    builder.setInsertionPointToStart(ifBlock_continueSearch.elseBlock());
    // Check if row has a match in kvec
    Value ii = builder.create<IndexCastOp>(loc, ii64, indexType);
    Value kk64 = builder.create<memref::LoadOp>(loc, Bi, ii);
    Value kk = builder.create<IndexCastOp>(loc, kk64, indexType);
    Value cmpPair = builder.create<memref::LoadOp>(loc, kvec_i1, kk);
    Value cmpResult0 = builder.create<SelectOp>(loc, cmpPair, cfalse, ctrue);
    Value cmpResult1 = builder.create<SelectOp>(loc, cmpPair, ci1, ii64);
    builder.create<scf::YieldOp>(loc, ValueRange{cmpResult0, cmpResult1});

    // end if cmpEndReached
    builder.setInsertionPointAfter(ifBlock_continueSearch);
    Value continueSearch = ifBlock_continueSearch.getResult(0);
    Value valToSend = ifBlock_continueSearch.getResult(1);
    builder.create<scf::ConditionOp>(loc, continueSearch, valToSend);

    // "do" portion of while loop
    builder.setInsertionPointToStart(&whileLoop.after().front());
    Value iiPrev = after->getArgument(0);
    Value iiNext = builder.create<AddIOp>(loc, iiPrev, ci1);
    builder.create<scf::YieldOp>(loc, iiNext);

    builder.setInsertionPointAfter(whileLoop);
    Value res = whileLoop.getResult(0);
    builder.create<scf::YieldOp>(loc, res);

    // end if cmpRowSame
    builder.setInsertionPointAfter(ifBlock_overlap);
    Value overlap = ifBlock_overlap.getResult(0);

    auto reducer = builder.create<scf::ReduceOp>(loc, overlap);
    Value lhs = reducer.getRegion().getArgument(0);
    Value rhs = reducer.getRegion().getArgument(1);
    builder.setInsertionPointToStart(&reducer.getRegion().front());
    Value z = builder.create<AddIOp>(loc, lhs, rhs);
    builder.create<scf::ReduceReturnOp>(loc, z);

    // end col loop
    builder.setInsertionPointAfter(colLoop1);
    Value total = colLoop1.getResult(0);
    builder.create<scf::YieldOp>(loc, total);

    // end if cmpColSame
    builder.setInsertionPointAfter(ifBlock_rowTotal);
    Value rowTotal = ifBlock_rowTotal.getResult(0);
    builder.create<memref::StoreOp>(loc, rowTotal, Cp, row);

    // end row loop
    builder.setInsertionPointAfter(rowLoop1);

    // 2nd pass
    //   Compute the cumsum of values in Cp to build the final Cp
    //   Then resize C's indices and values
    scf::ForOp rowLoop2 = builder.create<scf::ForOp>(loc, c0, nrow, c1);
    Value cs_i = rowLoop2.getInductionVar();
    builder.setInsertionPointToStart(rowLoop1.getBody());

    Value csTemp = builder.create<memref::LoadOp>(loc, Cp, cs_i);
    Value cumsum = builder.create<memref::LoadOp>(loc, Cp, nrow);
    builder.create<memref::StoreOp>(loc, cumsum, Cp, cs_i);
    Value cumsum2 = builder.create<AddIOp>(loc, cumsum, csTemp);
    builder.create<memref::StoreOp>(loc, cumsum2, Cp, nrow);

    // end row loop
    builder.setInsertionPointAfter(rowLoop2);

    Value nnz64 = builder.create<memref::LoadOp>(loc, Cp, nrow);
    Value nnz = builder.create<IndexCastOp>(loc, nnz64, indexType);
    callResizeIndex(builder, mod, loc, C, c1, nnz);
    callResizeValues(builder, mod, loc, C, nnz);
    Value Cj = builder.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, C, c1);
    Value Cx = builder.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, C);

    // 3rd pass
    //   In parallel over the rows,
    //   compute the nonzero columns and associated values.
    //   Store in Cj and Cx
    scf::ParallelOp rowLoop3 = builder.create<scf::ParallelOp>(loc, c0, nrow, c1);
    row = rowLoop3.getInductionVars()[0];
    builder.setInsertionPointToStart(rowLoop3.getBody());

    rowPlus1 = builder.create<AddIOp>(loc, row, c1);
    Value cpStart64 = builder.create<memref::LoadOp>(loc, Cp, row);
    Value cpEnd64 = builder.create<memref::LoadOp>(loc, Cp, rowPlus1);
    Value cmp_cpDifferent = builder.create<CmpIOp>(loc, CmpIPredicate::ne, cpStart64, cpEnd64);
    scf::IfOp ifBlock_cmpDiff = builder.create<scf::IfOp>(loc, cmp_cpDifferent);
    builder.setInsertionPointToStart(ifBlock_cmpDiff.thenBlock());

    Value baseIndex64 = builder.create<memref::LoadOp>(loc, Ap, row);
    Value baseIndex = builder.create<IndexCastOp>(loc, baseIndex64, indexType);

    // Construct a dense array of row values
    colStart64 = builder.create<memref::LoadOp>(loc, Ap, row);
    colEnd64 = builder.create<memref::LoadOp>(loc, Ap, rowPlus1);
    colStart = builder.create<IndexCastOp>(loc, colStart64, indexType);
    colEnd = builder.create<IndexCastOp>(loc, colEnd64, indexType);
    Value kvec = builder.create<memref::AllocOp>(loc, memref1DValueType, nk);
    kvec_i1 = builder.create<memref::AllocOp>(loc, memref1DBoolType, nk);
    builder.create<linalg::FillOp>(loc, kvec_i1, cfalse);
    scf::ParallelOp colLoop3p = builder.create<scf::ParallelOp>(loc, colStart, colEnd, c1);
    jj = colLoop3p.getInductionVars()[0];
    builder.setInsertionPointToStart(colLoop3p.getBody());
    col64 = builder.create<memref::LoadOp>(loc, Aj, jj);
    col = builder.create<IndexCastOp>(loc, col64, indexType);
    builder.create<memref::StoreOp>(loc, ctrue, kvec_i1);
    Value val = builder.create<memref::LoadOp>(loc, Ax, jj);
    builder.create<memref::StoreOp>(loc, val, kvec, col);

    // end col loop 3p
    builder.setInsertionPointAfter(colLoop3p);

    scf::ForOp colLoop3f;
    if (mask) {
        Value mcolStart64 = builder.create<memref::LoadOp>(loc, Mp, row);
        Value mcolEnd64 = builder.create<memref::LoadOp>(loc, Mp, rowPlus1);
        Value mcolStart = builder.create<IndexCastOp>(loc, mcolStart64, indexType);
        Value mcolEnd = builder.create<IndexCastOp>(loc, mcolEnd64, indexType);

        colLoop3f = builder.create<scf::ForOp>(loc, mcolStart, mcolEnd, c1, c0);
        Value mm = colLoop3f.getInductionVar();
        builder.setInsertionPointToStart(colLoop3f.getBody());
        col64 = builder.create<memref::LoadOp>(loc, Mj, mm);
        col = builder.create<IndexCastOp>(loc, col64, indexType);
    } else {
        colLoop3f = builder.create<scf::ForOp>(loc, c0, ncol, c1, c0);
        col = colLoop3f.getInductionVar();
        builder.setInsertionPointToStart(colLoop3f.getBody());
        col64 = builder.create<IndexCastOp>(loc, col, int64Type);
    }

    Value offset = colLoop3f.getLoopBody().getArgument(1);
    colPlus1 = builder.create<AddIOp>(loc, col, c1);
    Value iStart64 = builder.create<memref::LoadOp>(loc, Bp, col);
    Value iEnd64 = builder.create<memref::LoadOp>(loc, Bp, colPlus1);
    Value iStart = builder.create<IndexCastOp>(loc, iStart64, indexType);
    Value iEnd = builder.create<IndexCastOp>(loc, iEnd64, indexType);

    scf::ForOp kLoop = builder.create<scf::ForOp>(loc, iStart, iEnd, c1, ValueRange{cf0, cfalse});
    ii = kLoop.getInductionVar();
    Value curr = kLoop.getLoopBody().getArgument(1);
    Value alive = kLoop.getLoopBody().getArgument(2);
    builder.setInsertionPointToStart(kLoop.getBody());

    kk64 = builder.create<memref::LoadOp>(loc, Bi, ii);
    kk = builder.create<IndexCastOp>(loc, kk64, indexType);
    cmpPair = builder.create<memref::LoadOp>(loc, kvec_i1, kk);
    scf::IfOp ifBlock_cmpPair = builder.create<scf::IfOp>(loc, ArrayRef<Type>{valueType, boolType}, cmpPair, true);
    // if cmpPair
    builder.setInsertionPointToStart(ifBlock_cmpPair.thenBlock());
    Value newVal;
    if (semiring == "plus_pair") {
        newVal = builder.create<AddFOp>(loc, curr, cf1);
    } else {
        Value aVal = builder.create<memref::LoadOp>(loc, kvec, kk);
        Value bVal = builder.create<memref::LoadOp>(loc, Bx, ii);
        if (semiring == "plus_times") {
            val = builder.create<MulFOp>(loc, aVal, bVal);
            newVal = builder.create<AddFOp>(loc, curr, val);
        } else if (semiring == "plus_plus") {
            val = builder.create<AddFOp>(loc, aVal, bVal);
            newVal = builder.create<AddFOp>(loc, curr, val);
        }
    }
    builder.create<scf::YieldOp>(loc, ValueRange{newVal, ctrue});

    // else
    builder.setInsertionPointToStart(ifBlock_cmpPair.elseBlock());
    builder.create<scf::YieldOp>(loc, ValueRange{curr, alive});

    // end if cmpPair
    builder.setInsertionPointAfter(ifBlock_cmpPair);
    Value newCurr = ifBlock_cmpPair.getResult(0);
    Value newAlive = ifBlock_cmpPair.getResult(1);
    builder.create<scf::YieldOp>(loc, ValueRange{newCurr, newAlive});

    // end k loop
    builder.setInsertionPointAfter(kLoop);

    total = kLoop.getResult(0);
    Value notEmpty = kLoop.getResult(1);

    scf::IfOp ifBlock_newOffset = builder.create<scf::IfOp>(loc, indexType, notEmpty, true);
    // if not empty
    builder.setInsertionPointToStart(ifBlock_newOffset.thenBlock());

    // Store total in Cx
    Value cjPos = builder.create<AddIOp>(loc, baseIndex, offset);
    builder.create<memref::StoreOp>(loc, col64, Cj, cjPos);
    builder.create<memref::StoreOp>(loc, total, Cx, cjPos);
    // Increment offset
    Value offsetPlus1 = builder.create<AddIOp>(loc, offset, c1);
    builder.create<scf::YieldOp>(loc, offsetPlus1);

    // else
    builder.setInsertionPointToStart(ifBlock_newOffset.elseBlock());
    builder.create<scf::YieldOp>(loc, offset);

    // end if not empty
    builder.setInsertionPointAfter(ifBlock_newOffset);

    Value newOffset = ifBlock_newOffset.getResult(0);
    builder.create<scf::YieldOp>(loc, newOffset);

    // end col loop 3f
    builder.setInsertionPointAfter(colLoop3f);

    // end if cmpDiff
    builder.setInsertionPointAfter(ifBlock_cmpDiff);

    // end row loop
    builder.setInsertionPointAfter(rowLoop3);

    // Add return op
    builder.create<ReturnOp>(loc, C);
}