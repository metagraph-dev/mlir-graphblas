#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "lowering.h"

using namespace std;
using namespace mlir;
using namespace mlir::sparse_tensor;

void addMatrixSelectFunc(mlir::ModuleOp mod, const std::string &selector)
{
    MLIRContext *context = mod.getContext();
    OpBuilder builder(mod.getBodyRegion());
    auto loc = builder.getUnknownLoc();
    builder.setInsertionPointToStart(mod.getBody());

    bool needs_col = false, needs_val = false;
    if (selector == "triu") {
        needs_col = true;
        needs_val = false;
    } else if (selector == "tril") {
        needs_col = true;
        needs_val = false;
    } else if (selector == "gt0") {
        needs_col = false;
        needs_val = true;
    } else {
        assert(!"invalid selector");
    }

    // Types
    auto valueType = builder.getF64Type();
    auto i64Type = builder.getI64Type();
    auto f64Type = builder.getF64Type();
    auto indexType = builder.getIndexType();
    auto memref1DI64Type = MemRefType::get({-1}, i64Type);
    auto memref1DValueType = MemRefType::get({-1}, valueType);

    // Create function signature
    RankedTensorType csrTensor = getCSRTensorType(context, valueType);

    string func_name = "matrix_select_" + selector;
    auto func = builder.create<FuncOp>(builder.getUnknownLoc(),
                                       func_name,
                                       FunctionType::get(context, csrTensor, csrTensor));

    // Move to function body
    auto &entry_block = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entry_block);

    auto input = entry_block.getArgument(0);

    // add function body ops here
    // Initial constants
    Value c0 = builder.create<ConstantIndexOp>(loc, 0);
    Value c1 = builder.create<ConstantIndexOp>(loc, 1);
    Value c0_64 = builder.create<ConstantIntOp>(loc, 0, i64Type);
    Value c1_64 = builder.create<ConstantIntOp>(loc, 1, i64Type);
    Value cf0 = builder.create<ConstantFloatOp>(loc, APFloat(0.0), f64Type);

    // Get sparse tensor info
    Value nrow = builder.create<memref::DimOp>(loc, input, c0);
    Value ncol = builder.create<memref::DimOp>(loc, input, c1);
    Value Ap = builder.create<ToPointersOp>(loc, memref1DI64Type, input, c1);
    Value Aj = builder.create<ToIndicesOp>(loc, memref1DI64Type, input, c1);
    Value Ax = builder.create<ToValuesOp>(loc, memref1DValueType, input);

    Value output = callDupTensor(builder, mod, loc, input).getResult(0);
    Value Bp = builder.create<ToPointersOp>(loc, memref1DI64Type, output, c1);
    Value Bj = builder.create<ToIndicesOp>(loc, memref1DI64Type, output, c1);
    Value Bx = builder.create<ToValuesOp>(loc, memref1DValueType, output);

    builder.create<memref::StoreOp>(loc, c0_64, Bp, c0);
    // Loop
    auto outerLoop = builder.create<scf::ForOp>(loc, c0, nrow, c1);
    Value row = outerLoop.getInductionVar();

    builder.setInsertionPointToStart(outerLoop.getBody());
    Value row_plus1 = builder.create<mlir::AddIOp>(loc, row, c1);
    Value bp_curr_count = builder.create<memref::LoadOp>(loc, Bp, row);
    builder.create<memref::StoreOp>(loc, bp_curr_count, Bp, row_plus1);

    Value j_start_64 = builder.create<memref::LoadOp>(loc, Ap, row);
    Value j_end_64 = builder.create<memref::LoadOp>(loc, Ap, row_plus1);
    Value j_start = builder.create<mlir::IndexCastOp>(loc, j_start_64, indexType);
    Value j_end = builder.create<mlir::IndexCastOp>(loc, j_end_64, indexType);

    auto innerLoop = builder.create<scf::ForOp>(loc, j_start, j_end, c1);

    Value jj = innerLoop.getInductionVar();

    builder.setInsertionPointToStart(innerLoop.getBody());
    Value col_64, col, val, keep;
    if (needs_col) {
        col_64 = builder.create<memref::LoadOp>(loc, Aj, jj);
        col = builder.create<mlir::IndexCastOp>(loc, col_64, indexType);
    }
    if (needs_val) {
        val = builder.create<memref::LoadOp>(loc, Ax, jj);
    }
    if (selector == "triu") {
        keep = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::ugt, col, row);
    }
    else if (selector == "tril") {
        keep = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::ult, col, row);
    }
    else if (selector == "gt0") {
        keep = builder.create<mlir::CmpFOp>(loc, mlir::CmpFPredicate::OGT, val, cf0);
    }
    else {
        assert(!"invalid selector");
    }

    scf::IfOp ifKeep = builder.create<scf::IfOp>(loc, keep, false /* no else region */);

    builder.setInsertionPointToStart(ifKeep.thenBlock());

    Value bj_pos_64 = builder.create<memref::LoadOp>(loc, Bp, row_plus1);
    Value bj_pos = builder.create<mlir::IndexCastOp>(loc, bj_pos_64, indexType);

    if (!needs_col) {
        col_64 = builder.create<memref::LoadOp>(loc, Aj, jj);
    }
    builder.create<memref::StoreOp>(loc, col_64, Bj, bj_pos);

    if (!needs_val) {
        val = builder.create<memref::LoadOp>(loc, Ax, jj);
    }
    builder.create<memref::StoreOp>(loc, val, Bx, bj_pos);

    Value bj_pos_plus1 = builder.create<mlir::AddIOp>(loc, bj_pos_64, c1_64);
    builder.create<memref::StoreOp>(loc, bj_pos_plus1, Bp, row_plus1);

    builder.setInsertionPointAfter(outerLoop);

    Value nnz_64 = builder.create<memref::LoadOp>(loc, Bp, nrow);
    Value nnz = builder.create<mlir::IndexCastOp>(loc, nnz_64, indexType);

    callResizeIndex(builder, mod, loc, output, c1, nnz);
    callResizeValues(builder, mod, loc, output, nnz);

    // Add return op
    builder.create<ReturnOp>(builder.getUnknownLoc(), output);
}