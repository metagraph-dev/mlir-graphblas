#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "GraphBLAS/GraphBLASOps.h"
#include "GraphBLAS/GraphBLASUtils.h"

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

int64_t getRank(Type inputType)
{
  mlir::sparse_tensor::SparseTensorEncodingAttr sparseEncoding =
      mlir::sparse_tensor::getSparseTensorEncoding(inputType);
  if (!sparseEncoding)
    return -1;

  RankedTensorType inputTensorType = inputType.dyn_cast<RankedTensorType>();
  return inputTensorType.getRank();
}

int64_t getRank(Value inputValue)
{
  Type inputType = inputValue.getType();
  return getRank(inputType);
}

// make Compressed Vector type
RankedTensorType getCompressedVectorType(MLIRContext *context, ArrayRef<int64_t> shape, Type valueType)
{
    SmallVector<sparse_tensor::SparseTensorEncodingAttr::DimLevelType, 1> dlt;
    dlt.push_back(sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Compressed);
    unsigned ptr = 64;
    unsigned ind = 64;
    AffineMap map = {};

    RankedTensorType rtt = RankedTensorType::get(
        shape,
        valueType,
        sparse_tensor::SparseTensorEncodingAttr::get(context, dlt, map, ptr, ind));

    return rtt;
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
    
    RankedTensorType rtt = RankedTensorType::get(
        shape,
        valueType,
        sparse_tensor::SparseTensorEncodingAttr::get(context, dlt, map, ptr, ind));
    
    return rtt;
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

    RankedTensorType rtt = RankedTensorType::get(
        shape,
        valueType,
        sparse_tensor::SparseTensorEncodingAttr::get(context, dlt, map, ptr, ind));

    return rtt;
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

    RankedTensorType rtt = RankedTensorType::get(
        shape,
        valueType,
        sparse_tensor::SparseTensorEncodingAttr::get(context, dlt, map, ptr, ind));

    return rtt;
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
  
  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  Type inputTensorValueType = inputType.getElementType();
  RankedTensorType csxType = getCSXTensorType(context, inputShape, inputTensorValueType);
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
  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  Type inputTensorValueType = inputType.getElementType();
  RankedTensorType csrType = getCSRTensorType(context, inputShape, inputTensorValueType); 
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
  
  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  Type inputTensorValueType = inputType.getElementType();
  RankedTensorType cscType = getCSCTensorType(context, inputShape, inputTensorValueType); 
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
  RankedTensorType inputTensorType = tensor.getType().dyn_cast<RankedTensorType>();
  int64_t rank = inputTensorType.getRank();

  std::string funcName;
  tensor = convertToExternalCSX(builder, mod, loc, tensor);
  funcName = "delSparseTensor";
  Type tensorType = tensor.getType();

  FlatSymbolRefAttr func = getFunc(mod, loc, funcName, ArrayRef<Type>({}), tensorType);
  builder.create<mlir::CallOp>(loc, func, TypeRange(), tensor);
  return;
}

mlir::Value callEmptyLike(OpBuilder &builder, ModuleOp &mod, Location loc, Value tensor) {
  RankedTensorType inputTensorType = tensor.getType().dyn_cast<RankedTensorType>();
  int64_t rank = inputTensorType.getRank();

  std::string funcName;
  if (rank == 2)
  {
    tensor = convertToExternalCSX(builder, mod, loc, tensor);
    funcName = "matrix_empty_like";
  }
  else
  {
    funcName = "vector_empty_like";
  }
  Type tensorType = tensor.getType();

  FlatSymbolRefAttr func = getFunc(mod, loc, funcName, tensorType, tensorType);
  mlir::CallOp callOpResult = builder.create<mlir::CallOp>(loc, func, tensorType, tensor);
  Value result = callOpResult->getResult(0);
  return result;
}

mlir::Value callDupTensor(OpBuilder &builder, ModuleOp &mod, Location loc, Value tensor) {
  RankedTensorType inputTensorType = tensor.getType().dyn_cast<RankedTensorType>();
  int64_t rank = inputTensorType.getRank();

  std::string funcName;
  if (rank == 2)
  {
    tensor = convertToExternalCSX(builder, mod, loc, tensor);
    funcName = "dup_matrix";
  }
  else
  {
    funcName = "dup_vector";
  }
  Type tensorType = tensor.getType();

  FlatSymbolRefAttr func = getFunc(mod, loc, funcName, tensorType, tensorType);
    mlir::CallOp callOpResult = builder.create<mlir::CallOp>(loc, func, tensorType, tensor);
    Value result = callOpResult->getResult(0);
    return result;
  
}

mlir::CallOp callResizeDim(OpBuilder &builder, ModuleOp &mod, Location loc,
                           Value tensor, Value d, Value size)
{
  RankedTensorType inputTensorType = tensor.getType().dyn_cast<RankedTensorType>();
  int64_t rank = inputTensorType.getRank();

  std::string funcName;
  if (rank == 2)
  {
    tensor = convertToExternalCSX(builder, mod, loc, tensor);
    funcName = "matrix_resize_dim";
  }
  else
  {
    funcName = "vector_resize_dim";
  }
  Type tensorType = tensor.getType();

  Type indexType = builder.getIndexType();
  FlatSymbolRefAttr func = getFunc(mod, loc, funcName, TypeRange(), {tensorType, indexType, indexType});
  mlir::CallOp result = builder.create<mlir::CallOp>(loc, func, TypeRange(), ArrayRef<Value>({tensor, d, size}));

  return result;
}

mlir::CallOp callResizePointers(OpBuilder &builder, ModuleOp &mod, Location loc,
                                Value tensor, Value d, Value size)
{
  RankedTensorType inputTensorType = tensor.getType().dyn_cast<RankedTensorType>();
  int64_t rank = inputTensorType.getRank();

  std::string funcName;
  if (rank == 2)
  {
    tensor = convertToExternalCSX(builder, mod, loc, tensor);
    funcName = "matrix_resize_pointers";
  }
  else
  {
    funcName = "vector_resize_pointers";
  }
  Type tensorType = tensor.getType();

  Type indexType = builder.getIndexType();
  FlatSymbolRefAttr func = getFunc(mod, loc, funcName, TypeRange(), {tensorType, indexType, indexType});
  mlir::CallOp result = builder.create<mlir::CallOp>(loc, func, TypeRange(), ArrayRef<Value>({tensor, d, size}));

  return result;
}

mlir::CallOp callResizeIndex(OpBuilder &builder, ModuleOp &mod, Location loc,
                             Value tensor, Value d, Value size)
{
  RankedTensorType inputTensorType = tensor.getType().dyn_cast<RankedTensorType>();
  int64_t rank = inputTensorType.getRank();

  std::string funcName;
  if (rank == 2)
  {
    tensor = convertToExternalCSX(builder, mod, loc, tensor);
    funcName = "matrix_resize_index";
  }
  else
  {
    funcName = "vector_resize_index";
  }
  Type tensorType = tensor.getType();

  Type indexType = builder.getIndexType();
  FlatSymbolRefAttr func = getFunc(mod, loc, funcName, TypeRange(), {tensorType, indexType, indexType});
  mlir::CallOp result = builder.create<mlir::CallOp>(loc, func, TypeRange(), ArrayRef<Value>({tensor, d, size}));

  return result;
}

mlir::CallOp callResizeValues(OpBuilder &builder, ModuleOp &mod, Location loc,
                              Value tensor, Value size)
{
  RankedTensorType inputTensorType = tensor.getType().dyn_cast<RankedTensorType>();
  int64_t rank = inputTensorType.getRank();

  std::string funcName;
  if (rank == 2)
  {
    tensor = convertToExternalCSX(builder, mod, loc, tensor);
    funcName = "matrix_resize_values";
  }
  else
  {
    funcName = "vector_resize_values";
  }
  Type tensorType = tensor.getType();

  Type indexType = builder.getIndexType();
  FlatSymbolRefAttr func = getFunc(mod, loc, funcName, TypeRange(), {tensorType, indexType});
  mlir::CallOp result = builder.create<mlir::CallOp>(loc, func, TypeRange(), ArrayRef<Value>({tensor, size}));

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

LogicalResult ExtensionBlocks::extractBlocks(Operation *op, RegionRange &regions,
                                             const std::set<graphblas::YieldKind> &required,
                                             const std::set<graphblas::YieldKind> &optional)
{
  std::set<graphblas::YieldKind> allowed(required);
  allowed.insert(optional.begin(), optional.end());
  std::set<graphblas::YieldKind> seen;

  for (Region *ext : regions)
  {
    Block &block = ext->front();
    graphblas::YieldOp yield = llvm::dyn_cast_or_null<graphblas::YieldOp>(block.getTerminator());
    if (yield == nullptr)
    {
      // must have graphblas.yield as terminator
      return op->emitError("extension blocks must have a graphblas.yield terminator");
    }

    graphblas::YieldKind kind = yield.kind();
    if (allowed.count(kind) == 0) {
      return op->emitError("extension block " + stringifyYieldKind(kind) + " not allowed for this op");
    }

    switch (kind)
    {
    case graphblas::YieldKind::TRANSFORM_IN_A:
      this->transformInA = &block;
      break;
    case graphblas::YieldKind::TRANSFORM_IN_B:
      this->transformInB = &block;
      break;
    case graphblas::YieldKind::TRANSFORM_OUT:
      this->transformOut = &block;
      break;
    case graphblas::YieldKind::SELECT_IN_A:
      this->selectInA = &block;
      break;
    case graphblas::YieldKind::SELECT_IN_B:
      this->selectInB = &block;
      break;
    case graphblas::YieldKind::SELECT_OUT:
      this->selectOut = &block;
      break;
    case graphblas::YieldKind::ADD_IDENTITY:
      this->addIdentity = &block;
      break;
    case graphblas::YieldKind::ADD:
      this->add = &block;
      break;
    case graphblas::YieldKind::MULT:
      this->mult = &block;
      break;
    case graphblas::YieldKind::AGG_IDENTITY:
      this->aggIdentity = &block;
      break;
    case graphblas::YieldKind::AGG:
      this->agg = &block;
      break;
    default:
      return op->emitError("unsupported graphblas extension block type");
    }
    seen.insert(kind);
  }

  for (auto requiredKind : required) {
    if (seen.count(requiredKind) != 1) {
      return op->emitError("required extension block " + stringifyYieldKind(requiredKind) + " not found");
    }
  }

  return success();
};
