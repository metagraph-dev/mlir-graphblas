#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

    // Get sparse tensor info
    //auto nrows = builder.create<CallOp>()
    /*         %0 = call @sparseDimSize(%input, %c0) : (!llvm.ptr<i8>, index) -> index
        %1 = call @sparsePointers64(%input, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %2 = call @sparseIndices64(%input, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %3 = call @sparseValuesF64(%input) : (!llvm.ptr<i8>) -> memref<?xf64>
        */
    // TBD
    
    // Allocate temporary storage
    auto memrefF64 = MemRefType::get({}, valueType);
    auto acc = builder.create<memref::AllocOp>(unLoc, memrefF64);
    builder.create<memref::StoreOp>(unLoc, cf0, acc);

    // Add return op
    builder.create<ReturnOp>(builder.getUnknownLoc());
}