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

void addMatrixApplyFunc(mlir::ModuleOp mod, const std::string &operation)
{
    MLIRContext *context = mod.getContext();
    OpBuilder builder(mod.getBodyRegion());
    auto loc = builder.getUnknownLoc();

    // Types
    auto valueType = builder.getF64Type();
    auto i64Type = builder.getI64Type();
    auto indexType = builder.getIndexType();
    auto memref1DI64Type = MemRefType::get({-1}, i64Type);
    auto memref1DValueType = MemRefType::get({-1}, valueType);
    RankedTensorType csrTensorType = getCSRTensorType(context, valueType);

    builder.setInsertionPointToStart(mod.getBody());

    // Create function signature
    string func_name = "matrix_apply_" + operation;
    auto func = builder.create<FuncOp>(loc,
                           func_name,
                           FunctionType::get(context, {csrTensorType, valueType}, csrTensorType));

    // Move to function body
    auto &entry_block = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entry_block);

    auto input = entry_block.getArgument(0);
    auto thunk = entry_block.getArgument(1);

    // Initial constants
    auto c0 = builder.create<ConstantIndexOp>(loc, 0);
    auto c1 = builder.create<ConstantIndexOp>(loc, 1);

    // Get sparse tensor info
    auto output = callDupTensor(builder, mod, loc, input).getResult(0);
    auto inputPtrs = builder.create<ToPointersOp>(loc, memref1DI64Type, input, c1);
    auto inputValues = builder.create<ToValuesOp>(loc, memref1DValueType, input);
    auto outputValues = builder.create<ToValuesOp>(loc, memref1DValueType, output);

    auto nrows = builder.create<memref::DimOp>(loc, input, c0.getResult());
    auto nnz64 = builder.create<memref::LoadOp>(loc, inputPtrs, nrows.getResult());
    auto nnz = builder.create<mlir::IndexCastOp>(loc, nnz64, indexType);

    // Loop over values
    auto valueLoop = builder.create<scf::ParallelOp>(loc, c0.getResult(), nnz.getResult(), c1.getResult());
    auto valueLoopIdx = valueLoop.getInductionVars();

    builder.setInsertionPointToStart(valueLoop.getBody());
    auto val = builder.create<memref::LoadOp>(loc, inputValues, valueLoopIdx);

    Value result;
    if (operation == "min") {
        auto cmp = builder.create<mlir::CmpFOp>(loc, mlir::CmpFPredicate::OLT, val, thunk);
        result = builder.create<mlir::SelectOp>(loc, cmp, val, thunk);
    };

    builder.create<memref::StoreOp>(loc, result, outputValues, valueLoopIdx);

    // end value loop
    builder.setInsertionPointAfter(valueLoop);

    // Add return op
    builder.create<ReturnOp>(loc, output);
}