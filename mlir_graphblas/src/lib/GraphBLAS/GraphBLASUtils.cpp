#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace std;
using namespace mlir;
using namespace mlir::sparse_tensor;

bool typeIsCSR(Type inputType) {
  sparse_tensor::SparseTensorEncodingAttr inputSparseEncoding =
    sparse_tensor::getSparseTensorEncoding(inputType);
  if (inputSparseEncoding) {
    AffineMap inputDimOrdering = inputSparseEncoding.getDimOrdering();
    if (inputDimOrdering.getNumDims() == 2) {
      unsigned inputDimOrdering0 = inputDimOrdering.getDimPosition(0);
      unsigned inputDimOrdering1 = inputDimOrdering.getDimPosition(1);
      bool inputTypeIsCSR = (inputDimOrdering0 == 0 &&  inputDimOrdering1 == 1);
      return inputTypeIsCSR;
    }
  }
  return false;
}

bool typeIsCSC(Type inputType) {
  sparse_tensor::SparseTensorEncodingAttr inputSparseEncoding =
    sparse_tensor::getSparseTensorEncoding(inputType);
  if (inputSparseEncoding) {
    AffineMap inputDimOrdering = inputSparseEncoding.getDimOrdering();
    if (inputDimOrdering.getNumDims() == 2) {
      unsigned inputDimOrdering0 = inputDimOrdering.getDimPosition(0);
      unsigned inputDimOrdering1 = inputDimOrdering.getDimPosition(1);
      bool inputTypeIsCSC = (inputDimOrdering0 == 1 &&  inputDimOrdering1 == 0);
      return inputTypeIsCSC;
    }
  }
  return false;
}

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

// make CSC tensor type
RankedTensorType getCSCTensorType(MLIRContext *context, ArrayRef<int64_t> shape, Type valueType)
{
    SmallVector<sparse_tensor::SparseTensorEncodingAttr::DimLevelType, 2> dlt;
    dlt.push_back(sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Dense);
    dlt.push_back(sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Compressed);
    unsigned ptr = 64;
    unsigned ind = 64;
    AffineMap map = AffineMap::getPermutationMap(ArrayRef<unsigned>{1, 0}, context);

    RankedTensorType cscTensor = RankedTensorType::get(
        shape,
        valueType,
        sparse_tensor::SparseTensorEncodingAttr::get(context, dlt, map, ptr, ind));

    return cscTensor;
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

Value convertToExternalCSR(OpBuilder &builder, ModuleOp &mod, Location loc, Value input)
{
  // Cast the tensor to the CSR type that our external functions expect
  // since these are passed via opaque pointer (ultimately) to these functions
  // this is OK.
  MLIRContext *context = mod.getContext();
  RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();
  if (typeIsCSR(inputType)) {
    return input;
  }
  
  // all external calls are currently for unknown size float64 tensors
  RankedTensorType csrType = getCSRTensorType(context, {-1, -1}, builder.getF64Type());
  FlatSymbolRefAttr castFuncSymbol = getFunc(mod, loc, "cast_csc_to_csr", csrType, inputType);
  CallOp castCallOp = builder.create<CallOp>(loc,
					     castFuncSymbol,
					     csrType,
					     llvm::ArrayRef<Value>({input})
					     );

  Value result = castCallOp->getResult(0);
  return result;
}

Value convertToExternalCSC(OpBuilder &builder, ModuleOp &mod, Location loc, Value input)
{
  MLIRContext *context = mod.getContext();
  RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();
  if (typeIsCSC(inputType)) {
    return input;
  }
  
  // all external calls are currently for unknown size float64 tensors
  RankedTensorType csrType = getCSCTensorType(context, {-1, -1}, builder.getF64Type());
  FlatSymbolRefAttr castFuncSymbol = getFunc(mod, loc, "cast_csr_to_csc", csrType, inputType);
  CallOp castCallOp = builder.create<CallOp>(loc,
					     castFuncSymbol,
					     csrType,
					     llvm::ArrayRef<Value>({input})
					     );

  Value result = castCallOp->getResult(0);
  return result;
}

mlir::Value callEmptyLike(OpBuilder &builder, ModuleOp &mod, Location loc, Value tensor) {
    Type tensorType = tensor.getType();
    Value csrTensor = convertToExternalCSR(builder, mod, loc, tensor);
    Type csrTensorType = csrTensor.getType();

    FlatSymbolRefAttr func = getFunc(mod, loc, "empty_like", csrTensorType, csrTensorType);
    mlir::CallOp result = builder.create<mlir::CallOp>(loc, func, csrTensorType, csrTensor);
    if (typeIsCSR(tensorType)) {
      Value castResult = convertToExternalCSR(builder, mod, loc, result.getResult(0));
      return castResult;
    } else if (typeIsCSC(tensorType)) {
      Value castResult = convertToExternalCSC(builder, mod, loc, result.getResult(0));
      return castResult;
    } else {
      assert(false && "Unexpected tensor type.");
    }
}

mlir::Value callDupTensor(OpBuilder &builder, ModuleOp &mod, Location loc, Value tensor) {
  Type tensorType = tensor.getType();
  Value csrTensor = convertToExternalCSR(builder, mod, loc, tensor);
  Type csrTensorType = csrTensor.getType();

  FlatSymbolRefAttr func = getFunc(mod, loc, "dup_tensor", csrTensorType, csrTensorType);
  mlir::CallOp result = builder.create<mlir::CallOp>(loc, func, csrTensorType, csrTensor);
  if (typeIsCSR(tensorType)) {
    Value castResult = convertToExternalCSR(builder, mod, loc, result.getResult(0));
    return castResult;
  } else if (typeIsCSC(tensorType)) {
    Value castResult = convertToExternalCSC(builder, mod, loc, result.getResult(0));
    return castResult;
  } else {
    assert(false && "Unexpected tensor type.");
  }
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
