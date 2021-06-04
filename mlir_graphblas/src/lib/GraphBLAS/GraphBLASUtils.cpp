#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace std;
using namespace mlir;
using namespace mlir::sparse_tensor;

// make CSR tensor type
RankedTensorType getCSRTensorType(MLIRContext *context, ArrayRef<int64_t> shape, Type valueType)
{
    SmallVector<sparse_tensor::SparseTensorEncodingAttr::DimLevelType, 2> dlt;
    dlt.push_back(sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Dense);
    dlt.push_back(sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Compressed);
    unsigned ptr = 64;
    unsigned ind = 64;
    AffineMap map = AffineMap::getMultiDimIdentityMap(2, context);

    RankedTensorType csrTensor = RankedTensorType::get(
        shape,
        valueType,
        sparse_tensor::SparseTensorEncodingAttr::get(context, dlt, map, ptr, ind));

    return csrTensor;
}

/// Returns function reference (first hit also inserts into module).
// from: llvm/llvm-project/mlir/lib/Dialect/SparseTensor/Transforms/SparseTensorConversion.cpp
static FlatSymbolRefAttr getFunc(ModuleOp &mod, Location &loc, StringRef name, TypeRange result,
                                 TypeRange operands)
{
    MLIRContext *context = mod.getContext();
    FuncOp func = mod.lookupSymbol<FuncOp>(name);
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

mlir::CallOp callEmptyLike(OpBuilder &builder, ModuleOp &mod, Location loc, Value tensor) {
    Type tensorType = tensor.getType();

    FlatSymbolRefAttr func = getFunc(mod, loc, "empty_like", tensorType, tensorType);
    mlir::CallOp result = builder.create<mlir::CallOp>(loc, func, tensorType, tensor);

    return result;
}

mlir::CallOp callDupTensor(OpBuilder &builder, ModuleOp &mod, Location loc, Value tensor) {
    Type tensorType = tensor.getType();

    FlatSymbolRefAttr func = getFunc(mod, loc, "dup_tensor", tensorType, tensorType);
    mlir::CallOp result = builder.create<mlir::CallOp>(loc, func, tensorType, tensor);

    return result;
}

mlir::CallOp callResizeDim(OpBuilder &builder, ModuleOp &mod, Location loc,
                           Value tensor, Value d, Value size)
{
    Type tensorType = tensor.getType();

    Type indexType = builder.getIndexType();
    FlatSymbolRefAttr func = getFunc(mod, loc, "resize_dim", TypeRange(), {tensorType, indexType, indexType});
    mlir::CallOp result = builder.create<mlir::CallOp>(loc, func, TypeRange(), ArrayRef<Value>({tensor, d, size}));

    return result;
}

mlir::CallOp callResizePointers(OpBuilder &builder, ModuleOp &mod, Location loc,
                                Value tensor, Value d, Value size)
{
    Type tensorType = tensor.getType();

    Type indexType = builder.getIndexType();
    FlatSymbolRefAttr func = getFunc(mod, loc, "resize_pointers", TypeRange(), {tensorType, indexType, indexType});
    mlir::CallOp result = builder.create<mlir::CallOp>(loc, func, TypeRange(), ArrayRef<Value>({tensor, d, size}));

    return result;
}

mlir::CallOp callResizeIndex(OpBuilder &builder, ModuleOp &mod, Location loc,
                             Value tensor, Value d, Value size)
{
    Type tensorType = tensor.getType();

    Type indexType = builder.getIndexType();
    FlatSymbolRefAttr func = getFunc(mod, loc, "resize_index", TypeRange(), {tensorType, indexType, indexType});
    mlir::CallOp result = builder.create<mlir::CallOp>(loc, func, TypeRange(), ArrayRef<Value>({tensor, d, size}));

    return result;
}

mlir::CallOp callResizeValues(OpBuilder &builder, ModuleOp &mod, Location loc,
                              Value tensor, Value size)
{
    Type tensorType = tensor.getType();

    Type indexType = builder.getIndexType();
    FlatSymbolRefAttr func = getFunc(mod, loc, "resize_values", TypeRange(), {tensorType, indexType});
    mlir::CallOp result = builder.create<mlir::CallOp>(loc, func, TypeRange(), ArrayRef<Value>({tensor, size}));

    return result;
}
