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
    auto c0 = builder.create<ConstantIndexOp>(loc, 0);
    auto c1 = builder.create<ConstantIndexOp>(loc, 1);
    auto c0_64 = builder.create<ConstantIntOp>(loc, 0, int64Type);
    auto c1_64 = builder.create<ConstantIntOp>(loc, 1, int64Type);

    // Get sparse tensor info
    auto memref1DI64Type = MemRefType::get({-1}, int64Type);
    auto memref1DValueType = MemRefType::get({-1}, valueType);

    auto inputPtrs = builder.create<ToPointersOp>(loc, memref1DI64Type, input, c1);
    auto inputIndices = builder.create<ToIndicesOp>(loc, memref1DI64Type, input, c1);
    auto inputValues = builder.create<ToValuesOp>(loc, memref1DValueType, input);
    auto nrow = builder.create<memref::DimOp>(loc, input, c0.getResult());
    auto ncol = builder.create<memref::DimOp>(loc, input, c1.getResult());
    auto ncols_plus_one = builder.create<mlir::AddIOp>(loc, nrow, c1);

    auto indexType = builder.getIndexType();
    auto nnz_64 = builder.create<memref::LoadOp>(loc, inputPtrs, nrow.getResult());
    auto nnz = builder.create<mlir::IndexCastOp>(loc, nnz_64, indexType);

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

    // FIXME: More to put here

    builder.create<ReturnOp>(loc, output);
}