#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "lowering.h"

using namespace std;
using namespace mlir;
using namespace mlir::sparse_tensor;

void addTransposeFunc(mlir::ModuleOp mod, bool swap_sizes)
{
    MLIRContext *context = mod.getContext();
    OpBuilder builder(mod.getBodyRegion());
    auto loc = builder.getUnknownLoc(); /* eventually this will get the location of the op being lowered */

    builder.setInsertionPointToStart(mod.getBody());

    // Create function signature
    auto valueType = builder.getF64Type();
    RankedTensorType csrTensor = getCSRTensorType(context, valueType);

    string func_name = swap_sizes ? "transpose_swap" : "transpose_noswap";
    auto func = builder.create<FuncOp>(builder.getUnknownLoc(),
                                       func_name,
                                       FunctionType::get(context, csrTensor, csrTensor));

    // Move to function body
    auto &entry_block = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entry_block);

    auto input = entry_block.getArgument(0);

    // Initial constants
    auto int64Type = builder.getIntegerType(64);
    Value c0 = builder.create<ConstantIndexOp>(loc, 0);
    Value c1 = builder.create<ConstantIndexOp>(loc, 1);
    Value c0_64 = builder.create<ConstantIntOp>(loc, 0, int64Type);
    Value c1_64 = builder.create<ConstantIntOp>(loc, 1, int64Type);

    // Get sparse tensor info
    auto memref1DI64Type = MemRefType::get({-1}, int64Type);
    auto memref1DValueType = MemRefType::get({-1}, valueType);

    auto inputPtrs = builder.create<ToPointersOp>(loc, memref1DI64Type, input, c1);
    auto inputIndices = builder.create<ToIndicesOp>(loc, memref1DI64Type, input, c1);
    auto inputValues = builder.create<ToValuesOp>(loc, memref1DValueType, input);
    Value nrow = builder.create<memref::DimOp>(loc, input, c0);
    Value ncol = builder.create<memref::DimOp>(loc, input, c1);
    Value ncols_plus_one = builder.create<mlir::AddIOp>(loc, nrow, c1);

    auto indexType = builder.getIndexType();
    Value nnz_64 = builder.create<memref::LoadOp>(loc, inputPtrs, nrow);
    Value nnz = builder.create<mlir::IndexCastOp>(loc, nnz_64, indexType);

    auto output = callEmptyLike(builder, mod, loc, input).getResult(0);
    if (swap_sizes) {
        callResizeDim(builder, mod, loc, output, c0, ncol);
        callResizeDim(builder, mod, loc, output, c1, nrow);
    } else {
        callResizeDim(builder, mod, loc, output, c0, nrow);
        callResizeDim(builder, mod, loc, output, c1, ncol);
    }

    callResizePointers(builder, mod, loc, output, c1, ncols_plus_one);
    callResizeIndex(builder, mod, loc, output, c1, nnz);
    callResizeValues(builder, mod, loc, output, nnz);

    auto outputPtrs = builder.create<ToPointersOp>(loc, memref1DI64Type, output, c1);
    auto outputIndices = builder.create<ToIndicesOp>(loc, memref1DI64Type, output, c1);
    auto outputValues = builder.create<ToValuesOp>(loc, memref1DValueType, output);

    // compute number of non-zero entries per column of A

    // init B.pointers to zero
    auto initLoop = builder.create<scf::ForOp>(loc, c0, ncol, c1);
    auto initLoopIdx = initLoop.getInductionVar();
    builder.setInsertionPointToStart(initLoop.getBody());
    builder.create<memref::StoreOp>(loc, c0_64, outputPtrs, initLoopIdx);
    builder.setInsertionPointAfter(initLoop);

    // store pointers
    auto ptrLoop = builder.create<scf::ForOp>(loc, c0, nnz, c1);
    auto ptrLoopIdx = ptrLoop.getInductionVar();

    builder.setInsertionPointToStart(ptrLoop.getBody());
    Value colA64 = builder.create<memref::LoadOp>(loc, inputIndices, ptrLoopIdx);
    Value colA = builder.create<mlir::IndexCastOp>(loc, colA64, indexType);
    Value colB = builder.create<memref::LoadOp>(loc, outputPtrs, colA);
    Value colB1 = builder.create<mlir::AddIOp>(loc, colB, c1_64);
    builder.create<memref::StoreOp>(loc, colB1, outputPtrs, colA);

    builder.setInsertionPointAfter(ptrLoop);

    // cumsum the nnz per column to get Bp
    builder.create<memref::StoreOp>(loc, c0_64, outputPtrs, ncol);

    auto colAccLoop = builder.create<scf::ForOp>(loc, c0, ncol, c1);
    auto colAccLoopIdx = colAccLoop.getInductionVar();

    builder.setInsertionPointToStart(colAccLoop.getBody());
    Value temp = builder.create<memref::LoadOp>(loc, outputPtrs, colAccLoopIdx);
    Value cumsum = builder.create<memref::LoadOp>(loc, outputPtrs, ncol);
    builder.create<memref::StoreOp>(loc, cumsum, outputPtrs, colAccLoopIdx);
    Value cumsum2 = builder.create<mlir::AddIOp>(loc, cumsum, temp);
    builder.create<memref::StoreOp>(loc, cumsum2, outputPtrs, ncol);

    builder.setInsertionPointAfter(colAccLoop);

    // copy values
    auto outerLoop = builder.create<scf::ForOp>(loc, c0, nrow, c1);
    Value rowIdx = outerLoop.getInductionVar();

    builder.setInsertionPointToStart(outerLoop.getBody());
    Value row_64 = builder.create<mlir::IndexCastOp>(loc, rowIdx, indexType);
    Value j_start_64 = builder.create<memref::LoadOp>(loc, inputPtrs, rowIdx);
    Value j_start = builder.create<mlir::IndexCastOp>(loc, j_start_64, indexType);
    Value row_plus1 = builder.create<mlir::AddIOp>(loc, rowIdx, c1);
    Value j_end_64 = builder.create<memref::LoadOp>(loc, inputPtrs, row_plus1);
    Value j_end = builder.create<mlir::IndexCastOp>(loc, j_end_64, indexType);

    auto innerLoop = builder.create<scf::ForOp>(loc, j_start, j_end, c1);
    Value jj = innerLoop.getInductionVar();

    builder.setInsertionPointToStart(innerLoop.getBody());

    Value col_64 = builder.create<memref::LoadOp>(loc, inputIndices, jj);
    Value col = builder.create<mlir::IndexCastOp>(loc, col_64, indexType);
    Value dest_64 = builder.create<memref::LoadOp>(loc, outputPtrs, col);
    Value dest = builder.create<mlir::IndexCastOp>(loc, dest_64, indexType);
    builder.create<memref::StoreOp>(loc, row_64, outputIndices, dest);
    Value axjj = builder.create<memref::LoadOp>(loc, inputValues, jj);
    builder.create<memref::StoreOp>(loc, axjj, outputValues, dest);

    // Bp[col]++
    Value bp_inc = builder.create<memref::LoadOp>(loc, outputPtrs, col);
    Value bp_inc1 = builder.create<mlir::AddIOp>(loc, bp_inc, c1_64);
    builder.create<memref::StoreOp>(loc, bp_inc1, outputPtrs, col);

    builder.setInsertionPointAfter(outerLoop);

    Value last_last = builder.create<memref::LoadOp>(loc, outputPtrs, ncol);
    builder.create<memref::StoreOp>(loc, c0_64, outputPtrs, ncol);

    auto finalLoop = builder.create<scf::ForOp>(loc, c0, ncol, c1);
    Value iCol = finalLoop.getInductionVar();

    builder.setInsertionPointToStart(finalLoop.getBody());

    Value swapTemp = builder.create<memref::LoadOp>(loc, outputPtrs, iCol);
    Value last = builder.create<memref::LoadOp>(loc, outputPtrs, ncol);
    builder.create<memref::StoreOp>(loc, last, outputPtrs, iCol);
    builder.create<memref::StoreOp>(loc, swapTemp, outputPtrs, ncol);

    builder.setInsertionPointAfter(finalLoop);

    builder.create<memref::StoreOp>(loc, last_last, outputPtrs, ncol);

    builder.create<ReturnOp>(loc, output);
}