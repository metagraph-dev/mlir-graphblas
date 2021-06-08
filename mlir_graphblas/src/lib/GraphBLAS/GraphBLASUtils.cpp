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

RankedTensorType getCSCTensorType(MLIRContext *context, ArrayRef<int64_t> shape, Type valueType)
{
    SmallVector<sparse_tensor::SparseTensorEncodingAttr::DimLevelType, 2> dlt;
    dlt.push_back(sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Dense);
    dlt.push_back(sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Compressed);
    unsigned ptr = 64;
    unsigned ind = 64;
    AffineMap map = AffineMap::getPermutationMap({1, 0}, context);

    RankedTensorType cscTensor = RankedTensorType::get(
        shape,
        valueType,
        sparse_tensor::SparseTensorEncodingAttr::get(context, dlt, map, ptr, ind));

    return cscTensor;
}

Value convertToExternalCSR(OpBuilder &builder, ModuleOp &mod, Location loc, Value input)
{
    // Cast the tensor to the CSR type that our external functions expect
    // since these are passed via opaque pointer (ultimately) to these functions
    // this is OK.
    MLIRContext *context = mod.getContext();

    RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();
    // all external calls are currently for unknown size float64 tensors
    RankedTensorType csrType = getCSRTensorType(context, {-1, -1}, builder.getF64Type());

    if (inputType == csrType) {
        return input;
    } else {
        Value result = builder.create<tensor::CastOp>(loc, csrType, input);
        return result;
    }
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

mlir::Value callEmptyLike(OpBuilder &builder, ModuleOp &mod, Location loc, Value tensor) {
    Value csrTensor = convertToExternalCSR(builder, mod, loc, tensor);
    Type tensorType = csrTensor.getType();

    FlatSymbolRefAttr func = getFunc(mod, loc, "empty_like", tensorType, tensorType);
    mlir::CallOp result = builder.create<mlir::CallOp>(loc, func, tensorType, csrTensor);
    Value castResult = builder.create<tensor::CastOp>(loc, tensor.getType(), result.getResult(0));

    return castResult;
}

mlir::Value callDupTensor(OpBuilder &builder, ModuleOp &mod, Location loc, Value tensor) {
    Value csrTensor = convertToExternalCSR(builder, mod, loc, tensor);
    Type tensorType = csrTensor.getType();

    FlatSymbolRefAttr func = getFunc(mod, loc, "dup_tensor", tensorType, tensorType);
    mlir::CallOp result = builder.create<mlir::CallOp>(loc, func, tensorType, csrTensor);
    Value castResult = builder.create<tensor::CastOp>(loc, tensor.getType(), result.getResult(0));

    return castResult;
}

mlir::CallOp callResizeDim(OpBuilder &builder, ModuleOp &mod, Location loc,
                           Value tensor, Value d, Value size)
{
    Value csrTensor = convertToExternalCSR(builder, mod, loc, tensor);
    Type tensorType = csrTensor.getType();

    Type indexType = builder.getIndexType();
    FlatSymbolRefAttr func = getFunc(mod, loc, "resize_dim", TypeRange(), {tensorType, indexType, indexType});
    mlir::CallOp result = builder.create<mlir::CallOp>(loc, func, TypeRange(), ArrayRef<Value>({csrTensor, d, size}));

    return result;
}

mlir::CallOp callResizePointers(OpBuilder &builder, ModuleOp &mod, Location loc,
                                Value tensor, Value d, Value size)
{
    Value csrTensor = convertToExternalCSR(builder, mod, loc, tensor);
    Type tensorType = csrTensor.getType();

    Type indexType = builder.getIndexType();
    FlatSymbolRefAttr func = getFunc(mod, loc, "resize_pointers", TypeRange(), {tensorType, indexType, indexType});
    mlir::CallOp result = builder.create<mlir::CallOp>(loc, func, TypeRange(), ArrayRef<Value>({csrTensor, d, size}));

    return result;
}

mlir::CallOp callResizeIndex(OpBuilder &builder, ModuleOp &mod, Location loc,
                             Value tensor, Value d, Value size)
{
    Value csrTensor = convertToExternalCSR(builder, mod, loc, tensor);
    Type tensorType = csrTensor.getType();

    Type indexType = builder.getIndexType();
    FlatSymbolRefAttr func = getFunc(mod, loc, "resize_index", TypeRange(), {tensorType, indexType, indexType});
    mlir::CallOp result = builder.create<mlir::CallOp>(loc, func, TypeRange(), ArrayRef<Value>({csrTensor, d, size}));

    return result;
}

mlir::CallOp callResizeValues(OpBuilder &builder, ModuleOp &mod, Location loc,
                              Value tensor, Value size)
{
    Value csrTensor = convertToExternalCSR(builder, mod, loc, tensor);
    Type tensorType = csrTensor.getType();

    Type indexType = builder.getIndexType();
    FlatSymbolRefAttr func = getFunc(mod, loc, "resize_values", TypeRange(), {tensorType, indexType});
    mlir::CallOp result = builder.create<mlir::CallOp>(loc, func, TypeRange(), ArrayRef<Value>({csrTensor, size}));

    return result;
}
