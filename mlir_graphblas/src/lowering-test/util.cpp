#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "lowering.h"

using namespace std;
using namespace mlir;
using namespace mlir::sparse_tensor;

// make CSR tensor type
mlir::RankedTensorType getCSRTensorType(mlir::MLIRContext *context, mlir::FloatType valueType) {
    SmallVector<SparseTensorEncodingAttr::DimLevelType, 2> dlt;
    dlt.push_back(SparseTensorEncodingAttr::DimLevelType::Dense);
    dlt.push_back(SparseTensorEncodingAttr::DimLevelType::Compressed);
    unsigned ptr = 64;
    unsigned ind = 64;
    AffineMap map = {};

    RankedTensorType csrTensor = RankedTensorType::get(
        {-1, -1}, /* 2D, unknown size */
        valueType,
        SparseTensorEncodingAttr::get(context, dlt, map, ptr, ind));

    return csrTensor;
}

/// Returns function reference (first hit also inserts into module).
// from: llvm/llvm-project/mlir/lib/Dialect/SparseTensor/Transforms/SparseTensorConversion.cpp
static FlatSymbolRefAttr getFunc(mlir::ModuleOp &mod, mlir::Location &loc, StringRef name, Type result,
                                 TypeRange operands)
{
    MLIRContext *context = mod.getContext();
    auto func = mod.lookupSymbol<FuncOp>(name);
    if (!func)
    {
        OpBuilder moduleBuilder(mod.getBodyRegion());
        moduleBuilder
            .create<FuncOp>(loc, name,
                            FunctionType::get(context, operands, result))
            .setPrivate();
    }
    return SymbolRefAttr::get(context, name);
}

mlir::CallOp callEmptyLike(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc, mlir::Value tensor) {
    auto tensorType = tensor.getType();

    auto func = getFunc(mod, loc, "empty_like", tensorType, tensorType);
    auto result = builder.create<mlir::CallOp>(loc, func, tensorType, tensor);

    return result;
}

mlir::CallOp callDupTensor(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc, mlir::Value tensor) {
    auto tensorType = tensor.getType();

    auto func = getFunc(mod, loc, "dup_tensor", tensorType, tensorType);
    auto result = builder.create<mlir::CallOp>(loc, func, tensorType, tensor);

    return result;
}

mlir::CallOp callResizeDim(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc,
                           mlir::Value tensor, mlir::Value d, mlir::Value size)
{
    auto tensorType = tensor.getType();

    auto indexType = builder.getIndexType();
    auto noneType = builder.getNoneType();
    auto func = getFunc(mod, loc, "resize_dim", noneType, {tensorType, indexType, indexType});
    auto result = builder.create<mlir::CallOp>(loc, func, noneType, ArrayRef<Value>({tensor, d, size}));

    return result;
}

mlir::CallOp callResizePointers(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc,
                                mlir::Value tensor, mlir::Value d, mlir::Value size)
{
    auto tensorType = tensor.getType();

    auto indexType = builder.getIndexType();
    auto noneType = builder.getNoneType();
    auto func = getFunc(mod, loc, "resize_pointers", noneType, {tensorType, indexType, indexType});
    auto result = builder.create<mlir::CallOp>(loc, func, noneType, ArrayRef<Value>({tensor, d, size}));

    return result;
}

mlir::CallOp callResizeIndex(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc,
                                mlir::Value tensor, mlir::Value d, mlir::Value size)
{
    auto tensorType = tensor.getType();

    auto indexType = builder.getIndexType();
    auto noneType = builder.getNoneType();
    auto func = getFunc(mod, loc, "resize_index", noneType, {tensorType, indexType, indexType});
    auto result = builder.create<mlir::CallOp>(loc, func, noneType, ArrayRef<Value>({tensor, d, size}));

    return result;
}

mlir::CallOp callResizeValues(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc,
                              mlir::Value tensor, mlir::Value size)
{
    auto tensorType = tensor.getType();

    auto indexType = builder.getIndexType();
    auto noneType = builder.getNoneType();
    auto func = getFunc(mod, loc, "resize_values", noneType, {tensorType, indexType});
    auto result = builder.create<mlir::CallOp>(loc, func, noneType, ArrayRef<Value>({tensor, size}));

    return result;
}