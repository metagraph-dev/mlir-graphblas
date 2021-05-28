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

void addMatrixReduceToScalarFunc(mlir::ModuleOp mod, const std::string &aggregator)
{
    MLIRContext *context = mod.getContext();
    OpBuilder builder(mod.getBodyRegion());
    auto loc = builder.getUnknownLoc();  /* eventually this will get the location of the op being lowered */

    auto valueType = builder.getF64Type();
    auto int64Type = builder.getIntegerType(64);
    auto indexType = builder.getIndexType();
    string func_name = "matrix_reduce_to_scalar_" + aggregator;

    RankedTensorType csrTensor = getCSRTensorType(context, valueType);

    builder.setInsertionPointToStart(mod.getBody());

    // Create function signature
    auto func = builder.create<FuncOp>(loc,
                           func_name,
                           FunctionType::get(context, csrTensor, valueType));

    // Move to function body
    auto &entry_block = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entry_block);

    auto input = entry_block.getArgument(0);

    // Initial constants
    auto cf0 = builder.create<ConstantFloatOp>(loc, APFloat(0.0), valueType);
    auto c0 = builder.create<ConstantIndexOp>(loc, 0);
    auto c1 = builder.create<ConstantIndexOp>(loc, 1);

    // Get sparse tensor info
    auto memref1DI64Type = MemRefType::get({-1}, int64Type);
    auto memref1DValueType = MemRefType::get({-1}, valueType);

    auto nrows = builder.create<memref::DimOp>(loc, input, c0.getResult());
    auto inputPtrs = builder.create<ToPointersOp>(loc, memref1DI64Type, input, c1);
    auto inputValues = builder.create<ToValuesOp>(loc, memref1DValueType, input);
    auto nnz64 = builder.create<memref::LoadOp>(loc, inputPtrs, nrows.getResult());
    auto nnz = builder.create<mlir::IndexCastOp>(loc, nnz64, indexType);

    // begin loop
    auto valueLoop = builder.create<scf::ParallelOp>(loc, c0.getResult(), nnz.getResult(), c1.getResult(), cf0.getResult());
    auto valueLoopIdx = valueLoop.getInductionVars();

    builder.setInsertionPointToStart(valueLoop.getBody());
    auto y = builder.create<memref::LoadOp>(loc, inputValues, valueLoopIdx);

    auto reducer = builder.create<scf::ReduceOp>(loc, y);
    auto lhs = reducer.getRegion().getArgument(0);
    auto rhs = reducer.getRegion().getArgument(1);

    builder.setInsertionPointToStart(&reducer.getRegion().front());

    Value z;
    if (aggregator == "sum") {
        auto zOp = builder.create<mlir::AddFOp>(loc, lhs, rhs);
        z = zOp.getResult();
    }
    builder.create<scf::ReduceReturnOp>(loc, z);

    builder.setInsertionPointAfter(reducer);

    // end loop
    builder.setInsertionPointAfter(valueLoop);

    // Add return op
    builder.create<ReturnOp>(loc, valueLoop.getResult(0));
}
