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
    auto unLoc = builder.getUnknownLoc();

    auto valueType = builder.getF64Type();
    string func_name = "matrix_reduce_to_scalar_" + aggregator;

    RankedTensorType csrTensor = getCSRTensorType(context, valueType);

    builder.setInsertionPointToStart(mod.getBody());

    // Create function signature
    auto func = builder.create<FuncOp>(unLoc,
                           func_name,
                           FunctionType::get(context, csrTensor, valueType));

    // Move to function body
    auto &entry_block = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entry_block);

    auto input = entry_block.getArgument(0);

    // Initial constants
    auto int64Type = builder.getIntegerType(64);
    auto cf0 = builder.create<ConstantFloatOp>(unLoc, APFloat(0.0), valueType);
    auto c0 = builder.create<ConstantIndexOp>(unLoc, 0);
    auto c1 = builder.create<ConstantIndexOp>(unLoc, 1);

    // Get sparse tensor info
    auto memref1DI64Type = MemRefType::get({-1}, int64Type);
    auto memref1DValueType = MemRefType::get({-1}, valueType);

    auto nrows = builder.create<memref::DimOp>(unLoc, input, c0.getResult());
    auto inputPtrs = builder.create<ToPointersOp>(unLoc, memref1DI64Type, input, c1);
    auto inputIndices = builder.create<ToIndicesOp>(unLoc, memref1DI64Type, input, c1);
    auto inputValues = builder.create<ToValuesOp>(unLoc, memref1DValueType, input);

    // Allocate temporary storage
    auto memrefF64 = MemRefType::get({}, valueType);
    auto acc = builder.create<memref::AllocOp>(unLoc, memrefF64);
    builder.create<memref::StoreOp>(unLoc, cf0, acc);

    // outer row loop
    auto rowLoop = builder.create<scf::ForOp>(unLoc, c0, nrows, c1);
    auto rowLoopIdx = rowLoop.getInductionVar();

    builder.setInsertionPointToStart(rowLoop.getBody());
    auto col64 = builder.create<memref::LoadOp>(unLoc, inputPtrs, rowLoopIdx);
    auto indexType = builder.getIndexType();
    auto col = builder.create<mlir::IndexCastOp>(unLoc, col64, indexType);

    auto colEndIdx = builder.create<mlir::AddIOp>(unLoc, rowLoopIdx, c1);
    auto colEnd64 = builder.create<memref::LoadOp>(unLoc, inputPtrs, colEndIdx.getResult());
    auto colEnd = builder.create<mlir::IndexCastOp>(unLoc, colEnd64, indexType);
    auto curValue = builder.create<memref::LoadOp>(unLoc, acc);

    // begin inner col loop
    auto valueLoop = builder.create<scf::ForOp>(unLoc, col, colEnd, c1, curValue.getResult());
    auto valueLoopIdx = valueLoop.getInductionVar();
    auto x = valueLoop.getLoopBody().getArgument(1); /* this seems weird, but works */

    builder.setInsertionPointToStart(valueLoop.getBody());
    auto y = builder.create<memref::LoadOp>(unLoc, inputValues, valueLoopIdx);
    Value z;
    if (aggregator == "sum") {
        auto zOp = builder.create<mlir::AddFOp>(unLoc, x, y);
        z = zOp.getResult();
    }
    builder.create<scf::YieldOp>(unLoc, z);
    // end inner col loop

    builder.setInsertionPointAfter(valueLoop);
    builder.create<memref::StoreOp>(unLoc, valueLoop.getResult(0), acc);

    // end outer row loop
    builder.setInsertionPointAfter(rowLoop);
    auto finalResult = builder.create<memref::LoadOp>(unLoc, acc);

    // Add return op
    builder.create<ReturnOp>(builder.getUnknownLoc(), finalResult.getResult());
}