#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace std;
using namespace mlir;
using namespace mlir::sparse_tensor;

bool typeIsCSX(Type inputType) {
  sparse_tensor::SparseTensorEncodingAttr inputSparseEncoding =
    sparse_tensor::getSparseTensorEncoding(inputType);

  if (!inputSparseEncoding)
    return false;
  
  AffineMap inputDimOrdering = inputSparseEncoding.getDimOrdering();
  if (inputDimOrdering) 
    return false;
  
  llvm::ArrayRef<SparseTensorEncodingAttr::DimLevelType> dlt = inputSparseEncoding.getDimLevelType();
  if (dlt.size() != 2)
    return false;
  
  return true;
}

bool typeIsCSR(Type inputType) {

  sparse_tensor::SparseTensorEncodingAttr inputSparseEncoding =
    sparse_tensor::getSparseTensorEncoding(inputType);
  
  if (!inputSparseEncoding)
    return false;
  
  AffineMap inputDimOrdering = inputSparseEncoding.getDimOrdering();
  if (!inputDimOrdering) // if inputDimOrdering.map != nullptr ; i.e. if the dimOrdering exists
    return false;
  if (inputDimOrdering.getNumDims() != 2) {
    return false;
  } else {
    unsigned inputDimOrdering0 = inputDimOrdering.getDimPosition(0);
    unsigned inputDimOrdering1 = inputDimOrdering.getDimPosition(1);
    if (inputDimOrdering0 != 0 || inputDimOrdering1 != 1)
      return false;
  }
  llvm::ArrayRef<SparseTensorEncodingAttr::DimLevelType> dlt = inputSparseEncoding.getDimLevelType();
  if (dlt.size() != 2) {
    return false;
  } else {
    if (dlt[0] != sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Dense
        || dlt[1] != sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Compressed)
      return false;
  }

  return true;
}

bool typeIsCSC(Type inputType) {
  sparse_tensor::SparseTensorEncodingAttr inputSparseEncoding =
    sparse_tensor::getSparseTensorEncoding(inputType);
  
  if (!inputSparseEncoding)
    return false;
  
  AffineMap inputDimOrdering = inputSparseEncoding.getDimOrdering();
  if (!inputDimOrdering) // if inputDimOrdering.map != nullptr ; i.e. if the dimOrdering exists
    return false;
  if (inputDimOrdering.getNumDims() != 2) {
    return false;
  } else {
    unsigned inputDimOrdering0 = inputDimOrdering.getDimPosition(0);
    unsigned inputDimOrdering1 = inputDimOrdering.getDimPosition(1);
    if (inputDimOrdering0 != 1 || inputDimOrdering1 != 0)
      return false;
  }
  
  llvm::ArrayRef<SparseTensorEncodingAttr::DimLevelType> dlt = inputSparseEncoding.getDimLevelType();
  if (dlt.size() != 2) {
    return false;
  } else {
    if (dlt[0] != sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Dense
        || dlt[1] != sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Compressed)
      return false;
  }
  
  return true;
}

// make CSX tensor type
RankedTensorType getCSXTensorType(MLIRContext *context, ArrayRef<int64_t> shape, Type valueType)
{
    SmallVector<sparse_tensor::SparseTensorEncodingAttr::DimLevelType, 2> dlt;
    dlt.push_back(sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Dense);
    dlt.push_back(sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Compressed);
    unsigned ptr = 64;
    unsigned ind = 64;
    AffineMap map = {};
    
    RankedTensorType csrTensor = RankedTensorType::get(
        shape,
        valueType,
        sparse_tensor::SparseTensorEncodingAttr::get(context, dlt, map, ptr, ind));
    
    return csrTensor;
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

Value convertToExternalCSX(OpBuilder &builder, ModuleOp &mod, Location loc, Value input)
{
  MLIRContext *context = mod.getContext();
  RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();
  if (typeIsCSX(inputType))
    return input;
  
  RankedTensorType csxType = getCSXTensorType(context, {-1, -1}, builder.getF64Type()); 
  FlatSymbolRefAttr castFuncSymbol;
  if (typeIsCSC(inputType))
    castFuncSymbol = getFunc(mod, loc, "cast_csc_to_csx", csxType, inputType);
  else if (typeIsCSR(inputType))
    castFuncSymbol = getFunc(mod, loc, "cast_csr_to_csx", csxType, inputType);
  else 
    assert(false && "Unexpected tensor type.");
  CallOp castCallOp = builder.create<CallOp>(loc,
                                             castFuncSymbol,
                                             csxType,
                                             llvm::ArrayRef<Value>({input})
                                             );

  Value result = castCallOp->getResult(0);
  return result;
}

Value convertToExternalCSR(OpBuilder &builder, ModuleOp &mod, Location loc, Value input)
{
  // Cast the tensor to the CSR type that our external functions expect
  // since these are passed via opaque pointer (ultimately) to these functions
  // this is OK.
  MLIRContext *context = mod.getContext();
  RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();
  if (typeIsCSR(inputType))
    return input;

  if (typeIsCSC(inputType)) {
    input = convertToExternalCSX(builder, mod, loc, input);
    inputType = input.getType().dyn_cast<RankedTensorType>();
  }
  
  // all external calls are currently for unknown size float64 tensors
  RankedTensorType csrType = getCSRTensorType(context, {-1, -1}, builder.getF64Type()); 
  FlatSymbolRefAttr castFuncSymbol = getFunc(mod, loc, "cast_csx_to_csr", csrType, inputType);
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
  if (typeIsCSC(inputType))
    return input;

  if (typeIsCSR(inputType)) {
    input = convertToExternalCSX(builder, mod, loc, input);
    inputType = input.getType().dyn_cast<RankedTensorType>();
  }
  
  RankedTensorType cscType = getCSCTensorType(context, {-1, -1}, builder.getF64Type()); 
  FlatSymbolRefAttr castFuncSymbol = getFunc(mod, loc, "cast_csx_to_csc", cscType, inputType);
  CallOp castCallOp = builder.create<CallOp>(loc,
                                             castFuncSymbol,
                                             cscType,
                                             llvm::ArrayRef<Value>({input})
                                             );

  Value result = castCallOp->getResult(0);
  return result;
}

void callDelSparseTensor(OpBuilder &builder, ModuleOp &mod, Location loc, Value tensor) {
    Value csxTensor = convertToExternalCSX(builder, mod, loc, tensor);
    Type csxTensorType = csxTensor.getType();

    FlatSymbolRefAttr func = getFunc(mod, loc, "delSparseTensor", ArrayRef<Type>({}), csxTensorType);
    builder.create<mlir::CallOp>(loc, func, TypeRange(), csxTensor);
    return;
}

mlir::Value callEmptyLike(OpBuilder &builder, ModuleOp &mod, Location loc, Value tensor) {
    Value csxTensor = convertToExternalCSX(builder, mod, loc, tensor);
    Type csxTensorType = csxTensor.getType();

    FlatSymbolRefAttr func = getFunc(mod, loc, "empty_like", csxTensorType, csxTensorType);
    mlir::CallOp callOpResult = builder.create<mlir::CallOp>(loc, func, csxTensorType, csxTensor);
    Value result = callOpResult->getResult(0);
    return result;
}

mlir::Value callDupTensor(OpBuilder &builder, ModuleOp &mod, Location loc, Value tensor) {
  Value csxTensor = convertToExternalCSX(builder, mod, loc, tensor);
  Type csxTensorType = csxTensor.getType();

  FlatSymbolRefAttr func = getFunc(mod, loc, "dup_tensor", csxTensorType, csxTensorType);
    mlir::CallOp callOpResult = builder.create<mlir::CallOp>(loc, func, csxTensorType, csxTensor);
    Value result = callOpResult->getResult(0);
    return result;
  
}

mlir::CallOp callResizeDim(OpBuilder &builder, ModuleOp &mod, Location loc,
                           Value tensor, Value d, Value size)
{
  Value csxTensor = convertToExternalCSX(builder, mod, loc, tensor);
  Type csxTensorType = csxTensor.getType();

  Type indexType = builder.getIndexType();
  FlatSymbolRefAttr func = getFunc(mod, loc, "resize_dim", TypeRange(), {csxTensorType, indexType, indexType});
  mlir::CallOp result = builder.create<mlir::CallOp>(loc, func, TypeRange(), ArrayRef<Value>({csxTensor, d, size}));

  return result;
}

mlir::CallOp callResizePointers(OpBuilder &builder, ModuleOp &mod, Location loc,
                                Value tensor, Value d, Value size)
{
    Value csxTensor = convertToExternalCSX(builder, mod, loc, tensor);
    Type tensorType = csxTensor.getType();

    Type indexType = builder.getIndexType();
    FlatSymbolRefAttr func = getFunc(mod, loc, "resize_pointers", TypeRange(), {tensorType, indexType, indexType});
    mlir::CallOp result = builder.create<mlir::CallOp>(loc, func, TypeRange(), ArrayRef<Value>({csxTensor, d, size}));

    return result;
}

mlir::CallOp callResizeIndex(OpBuilder &builder, ModuleOp &mod, Location loc,
                             Value tensor, Value d, Value size)
{
    Value csxTensor = convertToExternalCSX(builder, mod, loc, tensor);
    Type tensorType = csxTensor.getType();

    Type indexType = builder.getIndexType();
    FlatSymbolRefAttr func = getFunc(mod, loc, "resize_index", TypeRange(), {tensorType, indexType, indexType});
    mlir::CallOp result = builder.create<mlir::CallOp>(loc, func, TypeRange(), ArrayRef<Value>({csxTensor, d, size}));

    return result;
}

mlir::CallOp callResizeValues(OpBuilder &builder, ModuleOp &mod, Location loc,
                              Value tensor, Value size)
{
    Value csxTensor = convertToExternalCSX(builder, mod, loc, tensor);
    Type tensorType = csxTensor.getType();

    Type indexType = builder.getIndexType();
    FlatSymbolRefAttr func = getFunc(mod, loc, "resize_values", TypeRange(), {tensorType, indexType});
    mlir::CallOp result = builder.create<mlir::CallOp>(loc, func, TypeRange(), ArrayRef<Value>({csxTensor, size}));

    return result;
}

void cleanupIntermediateTensor(OpBuilder &builder, ModuleOp &mod, Location loc, Value tensor) {
  // Clean up sparse tensor unless it is returned by the function
  Block * outputBlock = tensor.getParentBlock();
  ReturnOp lastStatement = llvm::dyn_cast_or_null<ReturnOp>(outputBlock->getTerminator());
  bool toDelete = true;
  if (lastStatement != nullptr) {
    for (Value result : lastStatement->getOperands()) {
      if (result == tensor) {
        toDelete = false;
        break;
      }
    }
    if (toDelete) {
      builder.setInsertionPoint(lastStatement);
      callDelSparseTensor(builder, mod, loc, tensor);
    }
  }
}