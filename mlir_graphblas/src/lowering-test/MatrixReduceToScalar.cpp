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

    // Initial constants
    auto cf0 = builder.create<ConstantFloatOp>(unLoc, APFloat(0.0), valueType);
    auto c0 = builder.create<ConstantIndexOp>(unLoc, 0);
    auto c1 = builder.create<ConstantIndexOp>(unLoc, 1);

    // Add return op
    builder.create<ReturnOp>(builder.getUnknownLoc());
}