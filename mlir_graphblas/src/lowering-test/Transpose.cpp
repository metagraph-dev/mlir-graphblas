#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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

    // add function body ops here

    // Add return op
    builder.create<ReturnOp>(builder.getUnknownLoc());
}