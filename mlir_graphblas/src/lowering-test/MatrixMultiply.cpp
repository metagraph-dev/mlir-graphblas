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

void addMatrixMultiplyFunc(mlir::ModuleOp mod, const std::string &semi_ring, bool mask)
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
        func_name += "matrix_multiply_mask_" + semi_ring;
        func_type = FunctionType::get(context, {csrTensorType, csrTensorType, csrTensorType}, csrTensorType);
    } else {
        func_name += "matrix_multiply_" + semi_ring;
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
    auto rowLoop = builder.create<scf::ParallelOp>(loc, c0, nrow, c1);
    Value row = rowLoop.getInductionVars()[0];
    builder.setInsertionPointToStart(rowLoop.getBody());

    Value colStart64 = builder.create<memref::LoadOp>(loc, Ap, row);
    Value rowPlus1 = builder.create<AddIOp>(loc, row, c1);
    Value colEnd64 = builder.create<memref::LoadOp>(loc, Ap, rowPlus1);
    Value cmpColSame = builder.create<CmpIOp>(loc, CmpIPredicate::eq, colStart64, colEnd64);

    auto ifBlock_rowTotal = builder.create<scf::IfOp>(loc, indexType, cmpColSame, true);
    // if cmpColSame
    builder.setInsertionPointToStart(&ifBlock_rowTotal.getRegion(0).front());
    builder.create<scf::YieldOp>(loc, ci0);

    // else
    builder.setInsertionPointToStart(&ifBlock_rowTotal.getRegion(1).front());

    // Construct a dense array indicating valid row positions
    Value colStart = builder.create<IndexCastOp>(loc, colStart64, indexType);
    Value colEnd = builder.create<IndexCastOp>(loc, colEnd64, indexType);
    Value kvec_i1 = builder.create<memref::AllocOp>(loc, memref1DBoolType, nk);
    builder.create<linalg::FillOp>(loc, kvec_i1, cfalse);
    auto colLoop = builder.create<scf::ParallelOp>(loc, colStart, colEnd, c1);
    Value jj = colLoop.getInductionVars()[0];
    builder.setInsertionPointToStart(colLoop.getBody());
    Value col64 = builder.create<memref::LoadOp>(loc, Aj, jj);
    Value col = builder.create<IndexCastOp>(loc, col64, indexType);
    builder.create<memref::StoreOp>(loc, ctrue, kvec_i1, col);
    builder.setInsertionPointAfter(colLoop);

    // Loop thru all columns; count number of resulting nonzeros in the row
    if (mask) {
        Value mcolStart64 = builder.create<memref::LoadOp>(loc, Mp, row);
        Value mcolEnd64 = builder.create<memref::LoadOp>(loc, Mp, rowPlus1);
        Value mcolStart = builder.create<IndexCastOp>(loc, mcolStart64, indexType);
        Value mcolEnd = builder.create<IndexCastOp>(loc, mcolEnd64, indexType);

        colLoop = builder.create<scf::ParallelOp>(loc, mcolStart, mcolEnd, c1, ci0);
        Value mm = colLoop.getInductionVars()[0];
        col64 = builder.create<memref::LoadOp>(loc, Mj, mm);
        col = builder.create<IndexCastOp>(loc, col64, indexType);
    } else {
        colLoop = builder.create<scf::ParallelOp>(loc, c0, ncol, c1, ci0);
        col = colLoop.getInductionVars()[0];
    }
    builder.setInsertionPointToStart(colLoop.getBody());

    Value colPlus1 = builder.create<AddIOp>(loc, col, c1);
    Value rowStart64 = builder.create<memref::LoadOp>(loc, Bp, col);
    Value rowEnd64 = builder.create<memref::LoadOp>(loc, Bp, colPlus1);
    Value cmpRowSame = builder.create<CmpIOp>(loc, CmpIPredicate::eq, rowStart64, rowEnd64);

    // Find overlap in column indices with kvec
    auto ifBlock_overlap = builder.create<scf::IfOp>(loc, int64Type, cmpRowSame, true);
    // if cmpRowSame
    builder.setInsertionPointToStart(&ifBlock_overlap.getRegion(0).front());
    builder.create<scf::YieldOp>(loc, ci0);

    // else
    builder.setInsertionPointToStart(&ifBlock_overlap.getRegion(1).front());

    // Walk thru the indices; on a match yield 1, else yield 0
    auto whileLoop = builder.create<scf::WhileOp>(loc, int64Type, rowStart64);
    Block *before = builder.createBlock(&whileLoop.before(), {}, int64Type);
    Block *after = builder.createBlock(&whileLoop.after(), {}, int64Type);
    Value ii64 = before->getArgument(0);
    builder.setInsertionPointToStart(&whileLoop.before().front());

    // Check if ii >= rowEnd
    Value cmpEndReached = builder.create<CmpIOp>(loc, CmpIPredicate::uge, ii64, rowEnd64);
    auto ifBlock_continueSearch = builder.create<scf::IfOp>(loc, ArrayRef<Type>{boolType, int64Type}, cmpEndReached, true);

    // if cmpEndReached
    builder.setInsertionPointToStart(&ifBlock_continueSearch.getRegion(0).front());
    builder.create<scf::YieldOp>(loc, ValueRange{cfalse, ci0});

    // else
    builder.setInsertionPointToStart(&ifBlock_continueSearch.getRegion(1).front());
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
    builder.setInsertionPointAfter(colLoop);
    Value total = colLoop.getResult(0);
    builder.create<scf::YieldOp>(loc, total);

    // end if cmpColSame
    builder.setInsertionPointAfter(ifBlock_rowTotal);
    Value rowTotal = ifBlock_rowTotal.getResult(0);
    builder.create<memref::StoreOp>(loc, rowTotal, Cp, row);

    // end row loop
    builder.setInsertionPointAfter(rowLoop);

    // 2nd pass
    //   Compute the cumsum of values in Cp to build the final Cp
    //   Then resize output indices and values


    // 3rd pass
    //   In parallel over the rows,
    //   compute the nonzero columns and associated values.
    //   Store in Cj and Cx
    

    // Add return op
    builder.create<ReturnOp>(loc, C);
}