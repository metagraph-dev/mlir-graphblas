#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/TypeSwitch.h"

#include "GraphBLAS/GraphBLASOps.h"
#include "GraphBLAS/GraphBLASUtils.h"

using namespace std;
using namespace mlir;
using namespace mlir::sparse_tensor;

bool hasRowOrdering(Type inputType) {
  sparse_tensor::SparseTensorEncodingAttr sparseEncoding =
      sparse_tensor::getSparseTensorEncoding(inputType);
  AffineMap dimOrdering = sparseEncoding.getDimOrdering();
  unsigned dimSize = dimOrdering.getNumResults();
  for (unsigned i = 0; i < dimSize; i++) {
    if (dimOrdering.getDimPosition(i) != i)
      return false;
  }
  return true;
}

bool hasColumnOrdering(Type inputType) {
  sparse_tensor::SparseTensorEncodingAttr sparseEncoding =
      sparse_tensor::getSparseTensorEncoding(inputType);
  AffineMap dimOrdering = sparseEncoding.getDimOrdering();
  unsigned dimSize = dimOrdering.getNumResults();
  for (unsigned i = 0; i < dimSize; i++) {
    if (dimOrdering.getDimPosition(i) != dimSize - (i + 1))
      return false;
  }
  return true;
}

// TODO: this is very heavyweight; we already check for CSR/CSC/EITHER in the
// verify methods; prefer hasRow/ColumnOrdering methods above
bool typeIsCSR(Type inputType) {

  sparse_tensor::SparseTensorEncodingAttr inputSparseEncoding =
      sparse_tensor::getSparseTensorEncoding(inputType);

  if (!inputSparseEncoding)
    return false;

  AffineMap inputDimOrdering = inputSparseEncoding.getDimOrdering();
  if (!inputDimOrdering) // if inputDimOrdering.map != nullptr ; i.e. if the
                         // dimOrdering exists
    return false;
  if (inputDimOrdering.getNumDims() != 2) {
    return false;
  } else {
    unsigned inputDimOrdering0 = inputDimOrdering.getDimPosition(0);
    unsigned inputDimOrdering1 = inputDimOrdering.getDimPosition(1);
    if (inputDimOrdering0 != 0 || inputDimOrdering1 != 1)
      return false;
  }
  llvm::ArrayRef<SparseTensorEncodingAttr::DimLevelType> dlt =
      inputSparseEncoding.getDimLevelType();
  if (dlt.size() != 2) {
    return false;
  } else {
    if (dlt[0] !=
            sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Dense ||
        dlt[1] !=
            sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Compressed)
      return false;
  }

  return true;
}

// TODO: this is very heavyweight; we already check for CSR/CSC/EITHER in the
// verify methods; prefer hasRow/ColumnOrdering methods above
bool typeIsCSC(Type inputType) {
  sparse_tensor::SparseTensorEncodingAttr inputSparseEncoding =
      sparse_tensor::getSparseTensorEncoding(inputType);

  if (!inputSparseEncoding)
    return false;

  AffineMap inputDimOrdering = inputSparseEncoding.getDimOrdering();
  if (!inputDimOrdering) // if inputDimOrdering.map != nullptr ; i.e. if the
                         // dimOrdering exists
    return false;
  if (inputDimOrdering.getNumDims() != 2) {
    return false;
  } else {
    unsigned inputDimOrdering0 = inputDimOrdering.getDimPosition(0);
    unsigned inputDimOrdering1 = inputDimOrdering.getDimPosition(1);
    if (inputDimOrdering0 != 1 || inputDimOrdering1 != 0)
      return false;
  }

  llvm::ArrayRef<SparseTensorEncodingAttr::DimLevelType> dlt =
      inputSparseEncoding.getDimLevelType();
  if (dlt.size() != 2) {
    return false;
  } else {
    if (dlt[0] !=
            sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Dense ||
        dlt[1] !=
            sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Compressed)
      return false;
  }

  return true;
}

int64_t getRank(Type inputType) {
  mlir::sparse_tensor::SparseTensorEncodingAttr sparseEncoding =
      mlir::sparse_tensor::getSparseTensorEncoding(inputType);
  if (!sparseEncoding)
    return -1;

  RankedTensorType inputTensorType = inputType.cast<RankedTensorType>();
  return inputTensorType.getRank();
}

int64_t getRank(Value inputValue) {
  Type inputType = inputValue.getType();
  return getRank(inputType);
}

// make Compressed Vector type
RankedTensorType getCompressedVectorType(MLIRContext *context,
                                         ArrayRef<int64_t> shape,
                                         Type valueType, unsigned ptrBitWidth,
                                         unsigned idxBitWidth) {
  SmallVector<sparse_tensor::SparseTensorEncodingAttr::DimLevelType, 1> dlt;
  dlt.push_back(
      sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Compressed);
  AffineMap map = {};

  RankedTensorType rtt =
      RankedTensorType::get(shape, valueType,
                            sparse_tensor::SparseTensorEncodingAttr::get(
                                context, dlt, map, ptrBitWidth, idxBitWidth));

  return rtt;
}

RankedTensorType getCompressedVectorType(MLIRContext *context, Type valueType) {
  return getCompressedVectorType(context, ArrayRef<int64_t>{-1}, valueType, 64,
                                 64);
}

RankedTensorType
getSingleCompressedMatrixType(MLIRContext *context, ArrayRef<int64_t> shape,
                              bool columnOriented, Type valueType,
                              unsigned ptrBitWidth, unsigned idxBitWidth) {
  SmallVector<sparse_tensor::SparseTensorEncodingAttr::DimLevelType, 2> dlt;
  dlt.push_back(sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Dense);
  dlt.push_back(
      sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Compressed);
  AffineMap map;
  if (columnOriented)
    map = AffineMap::getPermutationMap(ArrayRef<unsigned>{1, 0}, context);
  else
    map = AffineMap::getPermutationMap(ArrayRef<unsigned>{0, 1}, context);

  RankedTensorType rtt =
      RankedTensorType::get(shape, valueType,
                            sparse_tensor::SparseTensorEncodingAttr::get(
                                context, dlt, map, ptrBitWidth, idxBitWidth));

  return rtt;
}

RankedTensorType getCSRType(MLIRContext *context, Type valueType) {
  return getSingleCompressedMatrixType(context, ArrayRef<int64_t>{-1, -1},
                                       false, valueType, 64, 64);
}

RankedTensorType getCSCType(MLIRContext *context, Type valueType) {
  return getSingleCompressedMatrixType(context, ArrayRef<int64_t>{-1, -1}, true,
                                       valueType, 64, 64);
}

/// Returns function reference (first hit also inserts into module).
// from:
// llvm/llvm-project/mlir/lib/Dialect/SparseTensor/Transforms/SparseTensorConversion.cpp
static FlatSymbolRefAttr getFunc(ModuleOp &mod, Location &loc, StringRef name,
                                 TypeRange result, TypeRange operands) {
  MLIRContext *context = mod.getContext();
  FuncOp func = mod.lookupSymbol<FuncOp>(name);
  if (!func) {
    OpBuilder moduleBuilder(mod.getBodyRegion());
    moduleBuilder
        .create<FuncOp>(loc, name, FunctionType::get(context, operands, result))
        .setPrivate();
  }
  return SymbolRefAttr::get(context, name);
}

std::string buildSparseTypeString(RankedTensorType tensorType) {
  int64_t rank = tensorType.getRank();
  Type valueType = tensorType.getElementType();
  sparse_tensor::SparseTensorEncodingAttr sparseEncoding =
      sparse_tensor::getSparseTensorEncoding(tensorType);
  std::string piString =
      "p" + std::to_string(sparseEncoding.getPointerBitWidth()) + "i" +
      std::to_string(sparseEncoding.getIndexBitWidth());
  std::string dtype = llvm::TypeSwitch<Type, std::string>(valueType)
                          .Case<IntegerType>([&](IntegerType type) {
                            return "i" + std::to_string(type.getWidth());
                          })
                          .Case<FloatType>([&](FloatType type) {
                            return "f" + std::to_string(type.getWidth());
                          });

  if (rank == 2) {
    AffineMap dimOrdering = sparseEncoding.getDimOrdering();
    unsigned dimOrdering0 = dimOrdering.getDimPosition(0);
    unsigned dimOrdering1 = dimOrdering.getDimPosition(1);
    bool isCSC = (dimOrdering0 == 1 && dimOrdering1 == 0);
    std::string compressType = isCSC ? "csc" : "csr";
    return "matrix_" + compressType + "_" + dtype + "_" + piString;
  } else if (rank == 1) {
    return "vector_" + dtype + "_" + piString;
  } else
    assert(false && "Unexpected tensor type.");
}

void callPrintString(OpBuilder &builder, ModuleOp &mod, Location loc,
                     StringRef string) {
  Type int64Type = builder.getIntegerType(64);
  FlatSymbolRefAttr funcSymbol =
      getFunc(mod, loc, "print_int_as_char", TypeRange(), int64Type);
  for (char character : string) {
    int64_t character_int = (int64_t)character;
    Value character_int_i64 =
        builder.create<arith::ConstantIntOp>(loc, character_int, int64Type);
    builder.create<CallOp>(loc, funcSymbol, TypeRange(),
                           llvm::ArrayRef<Value>({character_int_i64}));
  }
  return;
}

void callPrintValue(OpBuilder &builder, ModuleOp &mod, Location loc,
                    Value input) {
  std::string funcName = "print_";
  llvm::raw_string_ostream stream(funcName);
  Type type = input.getType();
  type.print(stream);
  stream.flush();
  FlatSymbolRefAttr funcSymbol = getFunc(mod, loc, funcName, TypeRange(), type);
  builder.create<CallOp>(loc, funcSymbol, TypeRange(),
                         llvm::ArrayRef<Value>({input}));
  return;
}

Value castToPtr8(OpBuilder &builder, ModuleOp &mod, Location loc, Value input) {
  MLIRContext *context = mod.getContext();
  RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();
  if (!inputType) // Assume this means it is already !llvm.ptr<i8>
    return input;

  Type ptr8Type = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));

  std::string funcName = buildSparseTypeString(inputType) + "_to_ptr8";
  FlatSymbolRefAttr castFuncSymbol =
      getFunc(mod, loc, funcName, ptr8Type, inputType);
  CallOp castCallOp = builder.create<CallOp>(loc, castFuncSymbol, ptr8Type,
                                             llvm::ArrayRef<Value>({input}));
  Value result = castCallOp->getResult(0);
  return result;
}

Value castToTensor(OpBuilder &builder, ModuleOp &mod, Location loc,
                   Value ptrInput, RankedTensorType tensorType) {
  Type inputType = ptrInput.getType();

  std::string funcName = "ptr8_to_" + buildSparseTypeString(tensorType);
  FlatSymbolRefAttr castFuncSymbol =
      getFunc(mod, loc, funcName, tensorType, inputType);
  CallOp castCallOp = builder.create<CallOp>(loc, castFuncSymbol, tensorType,
                                             llvm::ArrayRef<Value>({ptrInput}));

  Value result = castCallOp->getResult(0);
  return result;
}

Value callNewTensor(OpBuilder &builder, ModuleOp &mod, Location loc,
                    ValueRange shape, RankedTensorType tensorType) {
  Value result = builder.create<sparse_tensor::InitOp>(loc, tensorType, shape);
  unsigned rank = tensorType.getRank();
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value dim, npointers;
  if (rank == 1) {
    dim = builder.create<arith::ConstantIndexOp>(loc, 0);
    npointers = c1;
  } else {
    dim = c1;
    if (hasRowOrdering(tensorType))
      npointers = builder.create<graphblas::NumRowsOp>(loc, result);
    else
      npointers = builder.create<graphblas::NumColsOp>(loc, result);
  }
  Value npointers_plus_1 = builder.create<arith::AddIOp>(loc, npointers, c1);
  callResizePointers(builder, mod, loc, result, dim, npointers_plus_1);

  return result;
}

Value callEmptyLike(OpBuilder &builder, ModuleOp &mod, Location loc,
                    Value tensor) {
  RankedTensorType tensorType = tensor.getType().dyn_cast<RankedTensorType>();
  Value ptr = castToPtr8(builder, mod, loc, tensor);
  Type ptr8Type = ptr.getType();

  FlatSymbolRefAttr func = getFunc(mod, loc, "empty_like", ptr8Type, ptr8Type);
  CallOp callOpResult = builder.create<mlir::CallOp>(loc, func, ptr8Type, ptr);
  Value result = callOpResult->getResult(0);
  tensor = castToTensor(builder, mod, loc, result, tensorType);
  return tensor;
}

Value callDupTensor(OpBuilder &builder, ModuleOp &mod, Location loc,
                    Value tensor) {
  RankedTensorType tensorType = tensor.getType().dyn_cast<RankedTensorType>();
  Value ptr = castToPtr8(builder, mod, loc, tensor);
  Type ptr8Type = ptr.getType();

  FlatSymbolRefAttr func = getFunc(mod, loc, "dup_tensor", ptr8Type, ptr8Type);
  CallOp callOpResult = builder.create<mlir::CallOp>(loc, func, ptr8Type, ptr);
  Value result = callOpResult->getResult(0);
  tensor = castToTensor(builder, mod, loc, result, tensorType);
  return tensor;
}

CallOp callAssignRev(OpBuilder &builder, ModuleOp &mod, Location loc,
                     Value tensor, Value d, Value newIndexValue) {
  Value ptr = castToPtr8(builder, mod, loc, tensor);
  Type ptr8Type = ptr.getType();

  Type indexType = builder.getIndexType();
  FlatSymbolRefAttr func = getFunc(mod, loc, "assign_rev", TypeRange(),
                                   {ptr8Type, indexType, indexType});
  CallOp result = builder.create<mlir::CallOp>(
      loc, func, TypeRange(), ArrayRef<Value>({ptr, d, newIndexValue}));

  return result;
}

CallOp callResizeDim(OpBuilder &builder, ModuleOp &mod, Location loc,
                     Value tensor, Value d, Value size) {
  Value ptr = castToPtr8(builder, mod, loc, tensor);
  Type ptr8Type = ptr.getType();

  Type indexType = builder.getIndexType();
  FlatSymbolRefAttr func = getFunc(mod, loc, "resize_dim", TypeRange(),
                                   {ptr8Type, indexType, indexType});
  CallOp result = builder.create<mlir::CallOp>(loc, func, TypeRange(),
                                               ArrayRef<Value>({ptr, d, size}));

  return result;
}

CallOp callResizePointers(OpBuilder &builder, ModuleOp &mod, Location loc,
                          Value tensor, Value d, Value size) {
  Value ptr = castToPtr8(builder, mod, loc, tensor);
  Type ptr8Type = ptr.getType();

  Type indexType = builder.getIndexType();
  FlatSymbolRefAttr func = getFunc(mod, loc, "resize_pointers", TypeRange(),
                                   {ptr8Type, indexType, indexType});
  CallOp result = builder.create<mlir::CallOp>(loc, func, TypeRange(),
                                               ArrayRef<Value>({ptr, d, size}));

  return result;
}

mlir::CallOp callResizeIndex(OpBuilder &builder, ModuleOp &mod, Location loc,
                             Value tensor, Value d, Value size) {
  Value ptr = castToPtr8(builder, mod, loc, tensor);
  Type ptr8Type = ptr.getType();

  Type indexType = builder.getIndexType();
  FlatSymbolRefAttr func = getFunc(mod, loc, "resize_index", TypeRange(),
                                   {ptr8Type, indexType, indexType});
  mlir::CallOp result = builder.create<mlir::CallOp>(
      loc, func, TypeRange(), ArrayRef<Value>({ptr, d, size}));

  return result;
}

CallOp callResizeValues(OpBuilder &builder, ModuleOp &mod, Location loc,
                        Value tensor, Value size) {
  Value ptr = castToPtr8(builder, mod, loc, tensor);
  Type ptr8Type = ptr.getType();

  Type indexType = builder.getIndexType();
  FlatSymbolRefAttr func =
      getFunc(mod, loc, "resize_values", TypeRange(), {ptr8Type, indexType});
  CallOp result = builder.create<mlir::CallOp>(loc, func, TypeRange(),
                                               ArrayRef<Value>({ptr, size}));

  return result;
}

CallOp callSwapPointers(OpBuilder &builder, ModuleOp &mod, Location loc,
                        Value tensor, Value other) {
  Value tPtr = castToPtr8(builder, mod, loc, tensor);
  Value oPtr = castToPtr8(builder, mod, loc, other);
  Type ptr8Type = oPtr.getType();

  FlatSymbolRefAttr ptrFunc =
      getFunc(mod, loc, "get_pointers_ptr", ptr8Type, ptr8Type);
  FlatSymbolRefAttr swapFunc =
      getFunc(mod, loc, "swap_pointers", TypeRange(), {ptr8Type, ptr8Type});
  CallOp oPointers = builder.create<mlir::CallOp>(loc, ptrFunc, ptr8Type, oPtr);
  CallOp result = builder.create<mlir::CallOp>(
      loc, swapFunc, TypeRange(),
      ArrayRef<Value>({tPtr, oPointers.getResult(0)}));

  return result;
}

CallOp callSwapIndices(OpBuilder &builder, ModuleOp &mod, Location loc,
                       Value tensor, Value other) {
  Value tPtr = castToPtr8(builder, mod, loc, tensor);
  Value oPtr = castToPtr8(builder, mod, loc, other);
  Type ptr8Type = oPtr.getType();

  FlatSymbolRefAttr ptrFunc =
      getFunc(mod, loc, "get_indices_ptr", ptr8Type, ptr8Type);
  FlatSymbolRefAttr swapFunc =
      getFunc(mod, loc, "swap_indices", TypeRange(), {ptr8Type, ptr8Type});
  CallOp oIndices = builder.create<mlir::CallOp>(loc, ptrFunc, ptr8Type, oPtr);
  CallOp result = builder.create<mlir::CallOp>(
      loc, swapFunc, TypeRange(),
      ArrayRef<Value>({tPtr, oIndices.getResult(0)}));

  return result;
}

CallOp callSwapValues(OpBuilder &builder, ModuleOp &mod, Location loc,
                      Value tensor, Value other) {
  Value tPtr = castToPtr8(builder, mod, loc, tensor);
  Value oPtr = castToPtr8(builder, mod, loc, other);
  Type ptr8Type = oPtr.getType();

  FlatSymbolRefAttr ptrFunc =
      getFunc(mod, loc, "get_values_ptr", ptr8Type, ptr8Type);
  FlatSymbolRefAttr swapFunc =
      getFunc(mod, loc, "swap_values", TypeRange(), {ptr8Type, ptr8Type});
  CallOp oValues = builder.create<mlir::CallOp>(loc, ptrFunc, ptr8Type, oPtr);
  CallOp result = builder.create<mlir::CallOp>(
      loc, swapFunc, TypeRange(),
      ArrayRef<Value>({tPtr, oValues.getResult(0)}));

  return result;
}

void cleanupIntermediateTensor(OpBuilder &builder, ModuleOp &mod, Location loc,
                               Value tensor) {
  // Clean up sparse tensor unless it is returned by the function
  Block *outputBlock = tensor.getParentBlock();
  ReturnOp lastStatement =
      llvm::dyn_cast_or_null<ReturnOp>(outputBlock->getTerminator());
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
      builder.create<sparse_tensor::ReleaseOp>(loc, tensor);
    }
  }
}

LogicalResult
ExtensionBlocks::extractBlocks(Operation *op, RegionRange &regions,
                               const std::set<graphblas::YieldKind> &required,
                               const std::set<graphblas::YieldKind> &optional) {
  std::set<graphblas::YieldKind> allowed(required);
  allowed.insert(optional.begin(), optional.end());
  std::set<graphblas::YieldKind> seen;

  for (Region *ext : regions) {
    Block &block = ext->front();
    graphblas::YieldOp yield =
        llvm::dyn_cast_or_null<graphblas::YieldOp>(block.getTerminator());
    if (yield == nullptr) {
      // must have graphblas.yield as terminator
      return op->emitError(
          "extension blocks must have a graphblas.yield terminator");
    }

    graphblas::YieldKind kind = yield.kind();
    if (allowed.count(kind) == 0) {
      return op->emitError("extension block " + stringifyYieldKind(kind) +
                           " not allowed for this op");
    }

    switch (kind) {
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
      return op->emitError("required extension block " +
                           stringifyYieldKind(requiredKind) + " not found");
    }
  }

  return success();
};

LogicalResult populateSemiringAdd(OpBuilder &builder, Location loc,
                                  StringRef addName, Type valueType,
                                  RegionRange regions) {
  // This function must match the options supported by
  // GraphBLASOps.cpp::checkSemiringAdd()

  // Insert additive identity
  Region *addIdentityRegion = regions[0];
  Value addIdentity;
  /*Block *addIdentityBlock = */ builder.createBlock(addIdentityRegion, {}, {});
  if (addName == "plus" || addName == "any") {
    // Add identity
    addIdentity = llvm::TypeSwitch<Type, Value>(valueType)
                      .Case<IntegerType>([&](IntegerType type) {
                        return builder.create<arith::ConstantIntOp>(
                            loc, 0, type.getWidth());
                      })
                      .Case<FloatType>([&](FloatType type) {
                        return builder.create<arith::ConstantFloatOp>(
                            loc, APFloat(0.0), type);
                      });
  } else if (addName == "min") {
    addIdentity =
        llvm::TypeSwitch<Type, Value>(valueType)
            .Case<IntegerType>([&](IntegerType type) {
              return builder.create<ConstantOp>(
                  loc,
                  builder.getIntegerAttr(
                      valueType, APInt::getSignedMaxValue(type.getWidth())));
            })
            .Case<FloatType>([&](FloatType type) {
              return builder.create<ConstantOp>(
                  loc, builder.getFloatAttr(
                           valueType, APFloat::getLargest(
                                          type.getFloatSemantics(), false)));
            });
  } else {
    return addIdentityRegion->getParentOp()->emitError(
        "\"" + addName + "\" is not a supported semiring add.");
  }
  builder.create<graphblas::YieldOp>(loc, graphblas::YieldKind::ADD_IDENTITY,
                                     addIdentity);

  // Insert additive operation
  Region *addRegion = regions[1];
  Block *addBlock = builder.createBlock(addRegion, {}, {valueType, valueType});
  Value addBlockArg0 = addBlock->getArgument(0);
  Value addBlockArg1 = addBlock->getArgument(1);
  Value addResult;
  if (addName == "plus") {
    // Insert add operation
    addResult = llvm::TypeSwitch<Type, Value>(valueType)
                    .Case<IntegerType>([&](IntegerType type) {
                      return builder.create<arith::AddIOp>(loc, addBlockArg0,
                                                           addBlockArg1);
                    })
                    .Case<FloatType>([&](FloatType type) {
                      return builder.create<arith::AddFOp>(loc, addBlockArg0,
                                                           addBlockArg1);
                    });
  } else if (addName == "min") {
    Value cmp =
        llvm::TypeSwitch<Type, Value>(valueType)
            .Case<IntegerType>([&](IntegerType type) {
              return builder.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::slt, addBlockArg0, addBlockArg1);
            })
            .Case<FloatType>([&](FloatType type) {
              return builder.create<arith::CmpFOp>(
                  loc, arith::CmpFPredicate::OLT, addBlockArg0, addBlockArg1);
            });
    addResult = builder.create<SelectOp>(loc, cmp, addBlockArg0, addBlockArg1);
  } else if (addName == "any") {
    // Same as "second" for multiplicative op?
    // https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/master/GraphBLAS/Source/Template/GB_binop_factory.c#L243-L244
    addResult = addBlockArg1;
  } else {
    return addRegion->getParentOp()->emitError(
        "\"" + addName + "\" is not a supported semiring add.");
  }
  builder.create<graphblas::YieldOp>(loc, graphblas::YieldKind::ADD, addResult);

  return success();
}

LogicalResult populateSemiringMultiply(OpBuilder &builder, Location loc,
                                       StringRef multiplyName, Type valueType,
                                       RegionRange regions) {
  // This function must match the options supported by
  // GraphBLASOps.cpp::checkSemiringMultiply()

  // Insert multiplicative operation
  Region *multRegion = regions[0];
  Block *multBlock =
      builder.createBlock(multRegion, {}, {valueType, valueType});
  Value multBlockArg0 = multBlock->getArgument(0);
  Value multBlockArg1 = multBlock->getArgument(1);
  Value multResult;

  if (multiplyName == "pair") {
    multResult = llvm::TypeSwitch<Type, Value>(valueType)
                     .Case<IntegerType>([&](IntegerType type) {
                       return builder.create<arith::ConstantIntOp>(
                           loc, 1, type.getWidth());
                     })
                     .Case<FloatType>([&](FloatType type) {
                       return builder.create<arith::ConstantFloatOp>(
                           loc, APFloat(1.0), type);
                     });
  } else if (multiplyName == "times") {
    multResult = llvm::TypeSwitch<Type, Value>(valueType)
                     .Case<IntegerType>([&](IntegerType type) {
                       return builder.create<arith::MulIOp>(loc, multBlockArg0,
                                                            multBlockArg1);
                     })
                     .Case<FloatType>([&](FloatType type) {
                       return builder.create<arith::MulFOp>(loc, multBlockArg0,
                                                            multBlockArg1);
                     });
  } else if (multiplyName == "plus") {
    multResult = llvm::TypeSwitch<Type, Value>(valueType)
                     .Case<IntegerType>([&](IntegerType type) {
                       return builder.create<arith::AddIOp>(loc, multBlockArg0,
                                                            multBlockArg1);
                     })
                     .Case<FloatType>([&](FloatType type) {
                       return builder.create<arith::AddFOp>(loc, multBlockArg0,
                                                            multBlockArg1);
                     });
  } else if (multiplyName == "first") {
    multResult = multBlockArg0;
  } else if (multiplyName == "second") {
    multResult = multBlockArg1;
  } else {
    return multRegion->getParentOp()->emitError(
        "\"" + multiplyName + "\" is not a supported semiring multiply.");
  }
  builder.create<graphblas::YieldOp>(loc, graphblas::YieldKind::MULT,
                                     multResult);

  return success();
}

LogicalResult populateSemiringRegions(OpBuilder &builder, Location loc,
                                      StringRef semiring, Type valueType,
                                      RegionRange regions) {
  auto semiringParts = semiring.split('_');

  if (failed(populateSemiringAdd(builder, loc, semiringParts.first, valueType,
                                 regions.slice(0, 2))))
    return failure();

  if (failed(populateSemiringMultiply(
          builder, loc, semiringParts.second, valueType,
          regions.slice(2, 1) /* no multiply identity block currently */)))
    return failure();

  return success();
}

LogicalResult extractApplyOpArgs(mlir::graphblas::ApplyOp op, Value &input,
                                 Value &thunk) {
  Value left = op.left();
  Value right = op.right();

  bool left_is_tensor = (bool)left.getType().dyn_cast<RankedTensorType>();
  bool right_is_tensor = right && right.getType().dyn_cast<RankedTensorType>();

  if (left_is_tensor == right_is_tensor) {
    return op.emitError("Exactly one operand must be a ranked tensor.");
  }

  if (left_is_tensor) {
    input = left;
    thunk = right;
  } else {
    input = right;
    thunk = left;
  }

  return success();
}
