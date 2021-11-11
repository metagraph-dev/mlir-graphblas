#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
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

MemRefType getMemrefPointerType(Type tensorType) {
  sparse_tensor::SparseTensorEncodingAttr sparseEncoding =
      sparse_tensor::getSparseTensorEncoding(tensorType);
  unsigned pointerBitWidth = sparseEncoding.getPointerBitWidth();
  Type pointerType = IntegerType::get(tensorType.getContext(), pointerBitWidth);
  return MemRefType::get({-1}, pointerType);
}

MemRefType getMemrefIndexType(Type tensorType) {
  sparse_tensor::SparseTensorEncodingAttr sparseEncoding =
      sparse_tensor::getSparseTensorEncoding(tensorType);
  unsigned indexBitWidth = sparseEncoding.getIndexBitWidth();
  Type indexType = IntegerType::get(tensorType.getContext(), indexBitWidth);
  return MemRefType::get({-1}, indexType);
}

MemRefType getMemrefValueType(Type tensorType) {
  RankedTensorType rtt = tensorType.dyn_cast<RankedTensorType>();
  return MemRefType::get({-1}, rtt.getElementType());
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

RankedTensorType getFlippedLayoutType(MLIRContext *context,
                                      Type inputOriginalType) {
  RankedTensorType inputType = inputOriginalType.dyn_cast<RankedTensorType>();
  sparse_tensor::SparseTensorEncodingAttr sparseEncoding =
      sparse_tensor::getSparseTensorEncoding(inputType);
  unsigned ptrBitWidth = sparseEncoding.getPointerBitWidth();
  unsigned idxBitWidth = sparseEncoding.getIndexBitWidth();
  Type inputValueType = inputType.getElementType();
  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();

  bool inputTypeIsCSR = hasRowOrdering(inputType);

  RankedTensorType flippedInputType =
      getSingleCompressedMatrixType(context, inputShape, inputTypeIsCSR,
                                    inputValueType, ptrBitWidth, idxBitWidth);

  return flippedInputType;
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

void callPrintTensor(OpBuilder &builder, ModuleOp &mod, Location loc,
                     Value input) {
  callPrintString(builder, mod, loc, "[");

  sparse_tensor::SparseTensorEncodingAttr sparseEncoding =
      sparse_tensor::getSparseTensorEncoding(input.getType());
  bool inputIsDense = !sparseEncoding;
  StringRef missingValueString = inputIsDense ? "0" : "_";

  RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();
  Type inputValueType = inputType.getElementType();

  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Type indexType = builder.getIndexType();
  Type int64Type = builder.getIntegerType(64);
  Type boolType = builder.getIntegerType(1);
  Type memref1DI64Type = MemRefType::get({-1}, int64Type);
  Type memref1DValueType = MemRefType::get({-1}, inputValueType);

  if (inputType.getRank() == 1) {
    if (inputIsDense) {
      MLIRContext *context = mod.getContext();
      inputType = getCompressedVectorType(context, inputValueType);
      // TODO: this is a hack and is very slow ; convert to an MLIR vector and
      // use vector.print
      input = builder.create<sparse_tensor::ConvertOp>(loc, inputType, input);
    }

    Value inputIndices = builder.create<sparse_tensor::ToIndicesOp>(
        loc, memref1DI64Type, input, c0);
    Value inputValues = builder.create<sparse_tensor::ToValuesOp>(
        loc, memref1DValueType, input);
    Value vectorLength = builder.create<graphblas::SizeOp>(loc, input);
    Value vectorLengthMinusOne =
        builder.create<arith::SubIOp>(loc, vectorLength, c1);

    Value firstValuesValue =
        builder.create<memref::LoadOp>(loc, inputValues, c0);
    Value firstIndicesValue_i64 =
        builder.create<memref::LoadOp>(loc, inputIndices, c0);
    Value firstIndicesValue = builder.create<arith::IndexCastOp>(
        loc, firstIndicesValue_i64, indexType);

    scf::ForOp forEachLoop = builder.create<scf::ForOp>(
        loc, c0, vectorLength, c1,
        ValueRange{c0, firstIndicesValue, firstValuesValue});
    {
      builder.setInsertionPointToStart(forEachLoop.getBody());
      Value toPrintPosition = forEachLoop.getInductionVar();
      Value inputValuesAndIndicesPosition =
          forEachLoop.getLoopBody().getArgument(1);
      Value inputIndicesValue = forEachLoop.getLoopBody().getArgument(2);
      Value inputValuesValue = forEachLoop.getLoopBody().getArgument(3);

      Value printPresentValue = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, inputIndicesValue, toPrintPosition);
      scf::IfOp ifPrintPresentValue = builder.create<scf::IfOp>(
          loc, TypeRange{indexType, indexType, inputValueType},
          printPresentValue, true);
      {
        builder.setInsertionPointToStart(ifPrintPresentValue.thenBlock());
        {
          callPrintValue(builder, mod, loc, inputValuesValue);
          Value updatedInputValuesAndIndicesPosition =
              builder.create<arith::AddIOp>(loc, inputValuesAndIndicesPosition,
                                            c1);
          Value updatedInputIndicesValue_i64 = builder.create<memref::LoadOp>(
              loc, inputIndices, updatedInputValuesAndIndicesPosition);
          Value updatedInputIndicesValue = builder.create<arith::IndexCastOp>(
              loc, updatedInputIndicesValue_i64, indexType);
          Value updatedInputValuesValue = builder.create<memref::LoadOp>(
              loc, inputValues, updatedInputValuesAndIndicesPosition);
          builder.create<scf::YieldOp>(
              loc,
              ValueRange{updatedInputValuesAndIndicesPosition,
                         updatedInputIndicesValue, updatedInputValuesValue});
        }
        builder.setInsertionPointToStart(ifPrintPresentValue.elseBlock());
        {
          callPrintString(builder, mod, loc, missingValueString);
          builder.create<scf::YieldOp>(
              loc, ValueRange{inputValuesAndIndicesPosition, inputIndicesValue,
                              inputValuesValue});
        }
        builder.setInsertionPointAfter(ifPrintPresentValue);
      }

      Value notLastPosition = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ne, vectorLengthMinusOne, toPrintPosition);
      scf::IfOp ifNotLastPosition =
          builder.create<scf::IfOp>(loc, TypeRange{}, notLastPosition, false);
      {
        builder.setInsertionPointToStart(ifNotLastPosition.thenBlock());
        callPrintString(builder, mod, loc, ", ");
        builder.setInsertionPointAfter(ifNotLastPosition);
      }

      Value nextInputValuesAndIndicesPosition =
          ifPrintPresentValue.getResult(0);
      Value nextInputIndicesValue = ifPrintPresentValue.getResult(1);
      Value nextInputValuesValue = ifPrintPresentValue.getResult(2);
      builder.create<scf::YieldOp>(
          loc, ValueRange{nextInputValuesAndIndicesPosition,
                          nextInputIndicesValue, nextInputValuesValue});
      builder.setInsertionPointAfter(forEachLoop);
    }
  } else {
    if (inputIsDense) {
      MLIRContext *context = mod.getContext();
      inputType = getCSRType(context, inputValueType);
      // TODO: this is a hack and is very slow ; convert to an MLIR vector and
      // use vector.print
      input = builder.create<sparse_tensor::ConvertOp>(loc, inputType, input);
    } else if (!hasRowOrdering(inputType)) {
      MLIRContext *context = mod.getContext();
      inputType = getCSRType(context, inputValueType);
      input = builder.create<graphblas::ConvertLayoutOp>(loc, inputType, input);
    }
    Value nrows = builder.create<graphblas::NumRowsOp>(loc, input);
    Value ncols = builder.create<graphblas::NumColsOp>(loc, input);
    Value ncolsMinusOne = builder.create<arith::SubIOp>(loc, ncols, c1);

    sparse_tensor::SparseTensorEncodingAttr sparseEncoding =
        sparse_tensor::getSparseTensorEncoding(inputType);
    unsigned pointerBitWidth = sparseEncoding.getPointerBitWidth();
    Type pointerType = builder.getIntegerType(pointerBitWidth);
    Type memref1DPointerType = MemRefType::get({-1}, pointerType);

    Value matrixPointers = builder.create<sparse_tensor::ToPointersOp>(
        loc, memref1DPointerType, input, c1);
    Value matrixIndices = builder.create<sparse_tensor::ToIndicesOp>(
        loc, memref1DI64Type, input, c1);
    Value matrixValues = builder.create<sparse_tensor::ToValuesOp>(
        loc, memref1DValueType, input);
    scf::ForOp forRowLoop = builder.create<scf::ForOp>(loc, c0, nrows, c1);
    {
      builder.setInsertionPointToStart(forRowLoop.getBody());

      Value matrixRowIndex = forRowLoop.getInductionVar();
      Value nextMatrixRowIndex =
          builder.create<arith::AddIOp>(loc, matrixRowIndex, c1).getResult();
      Value firstPtr64 =
          builder.create<memref::LoadOp>(loc, matrixPointers, matrixRowIndex);
      Value firstPtr =
          builder.create<arith::IndexCastOp>(loc, firstPtr64, indexType);
      Value secondPtr64 = builder.create<memref::LoadOp>(loc, matrixPointers,
                                                         nextMatrixRowIndex);
      Value secondPtr =
          builder.create<arith::IndexCastOp>(loc, secondPtr64, indexType);

      callPrintString(builder, mod, loc, "\n  [");
      scf::ForOp forColLoop =
          builder.create<scf::ForOp>(loc, c0, ncols, c1, ValueRange{firstPtr});
      {
        builder.setInsertionPointToStart(forColLoop.getBody());
        Value toPrintPosition = forColLoop.getInductionVar();
        Value ptr = forColLoop.getLoopBody().getArgument(1);

        Value ptrIsValid = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ult, ptr, secondPtr);
        scf::IfOp ifPtrIsValid = builder.create<scf::IfOp>(
            loc, TypeRange{boolType}, ptrIsValid, true);
        {
          builder.setInsertionPointToStart(ifPtrIsValid.thenBlock());
          {
            Value col_i64 =
                builder.create<memref::LoadOp>(loc, matrixIndices, ptr);
            Value col =
                builder.create<arith::IndexCastOp>(loc, col_i64, indexType);
            Value colEqualsToPrintPosition = builder.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::eq, col, toPrintPosition);
            builder.create<scf::YieldOp>(loc,
                                         ValueRange{colEqualsToPrintPosition});
          }
          builder.setInsertionPointToStart(ifPtrIsValid.elseBlock());
          {
            Value false_i1 =
                builder.create<arith::ConstantIntOp>(loc, 0, boolType);
            builder.create<scf::YieldOp>(loc, ValueRange{false_i1});
          }
          builder.setInsertionPointAfter(ifPtrIsValid);
        }
        Value printPresentValue = ifPtrIsValid.getResult(0);

        scf::IfOp ifPrintPresentValue = builder.create<scf::IfOp>(
            loc, TypeRange{indexType}, printPresentValue, true);
        {
          builder.setInsertionPointToStart(ifPrintPresentValue.thenBlock());
          {
            Value matrixValue =
                builder.create<memref::LoadOp>(loc, matrixValues, ptr);
            callPrintValue(builder, mod, loc, matrixValue);
            Value updatedPtr = builder.create<arith::AddIOp>(loc, ptr, c1);
            builder.create<scf::YieldOp>(loc, ValueRange{updatedPtr});
          }
          builder.setInsertionPointToStart(ifPrintPresentValue.elseBlock());
          {
            callPrintString(builder, mod, loc, missingValueString);
            builder.create<scf::YieldOp>(loc, ValueRange{ptr});
          }
          builder.setInsertionPointAfter(ifPrintPresentValue);
        }

        Value notLastCol = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ne, ncolsMinusOne, toPrintPosition);
        scf::IfOp ifNotLastCol =
            builder.create<scf::IfOp>(loc, TypeRange{}, notLastCol, true);
        {
          builder.setInsertionPointToStart(ifNotLastCol.thenBlock());
          callPrintString(builder, mod, loc, ", ");
          builder.setInsertionPointToStart(ifNotLastCol.elseBlock());
          callPrintString(builder, mod, loc, "],");
          builder.setInsertionPointAfter(ifNotLastCol);
        }

        Value nextPtr = ifPrintPresentValue.getResult(0);
        builder.create<scf::YieldOp>(loc, ValueRange{nextPtr});
        builder.setInsertionPointAfter(forColLoop);
      }

      builder.setInsertionPointAfter(forRowLoop);
    }
    callPrintString(builder, mod, loc, "\n");
  }
  callPrintString(builder, mod, loc, "]");

  return;
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
                    Value tensor, Type valueType) {
  // Variant on callNewTensor which uses information from an existing tensor
  // but allows for a different valueType
  Type inputType = tensor.getType();
  RankedTensorType inputTensorType = inputType.cast<RankedTensorType>();
  Type inputValueType = inputTensorType.getElementType();

  if (!valueType)
    valueType = inputValueType;

  sparse_tensor::SparseTensorEncodingAttr sparseEncoding =
      sparse_tensor::getSparseTensorEncoding(inputTensorType);
  AffineMap map = sparseEncoding.getDimOrdering();
  if (!map)
    map = {};
  RankedTensorType rtt = RankedTensorType::get(
      inputTensorType.getShape(), valueType,
      sparse_tensor::SparseTensorEncodingAttr::get(
          builder.getContext(), sparseEncoding.getDimLevelType(), map,
          sparseEncoding.getPointerBitWidth(),
          sparseEncoding.getIndexBitWidth()));

  unsigned rank = inputTensorType.getRank();

  // Get the shape as a ValueRange
  ValueRange shape;
  if (rank == 1) {
    Value size = builder.create<graphblas::SizeOp>(loc, tensor);
    shape = ValueRange{size};
  } else {
    Value nrows = builder.create<graphblas::NumRowsOp>(loc, tensor);
    Value ncols = builder.create<graphblas::NumColsOp>(loc, tensor);
    shape = ValueRange{nrows, ncols};
  }

  return callNewTensor(builder, mod, loc, shape, rtt);
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
    case graphblas::YieldKind::ACCUMULATE:
      this->accumulate = &block;
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

LogicalResult populateUnary(OpBuilder &builder, Location loc, StringRef unaryOp,
                            Type valueType, RegionRange regions,
                            graphblas::YieldKind yieldKind) {
  // This function must match the options supported by
  // GraphBLASOps.cpp::checkUnaryOp()

  Type indexType = builder.getIndexType();

  // Insert unary operation
  Region *unaryRegion = regions[0];
  TypeRange inputTypes;
  if (unary1.contains(unaryOp))
    inputTypes = TypeRange{valueType};
  else if (unary3.contains(unaryOp))
    inputTypes = TypeRange{valueType, indexType, indexType};

  Block *unaryBlock = builder.createBlock(unaryRegion, {}, inputTypes);
  int numArgs = unaryBlock->getArguments().size();

  Value val = unaryBlock->getArgument(0);
  Value row, col;
  if (numArgs >= 3) {
    row = unaryBlock->getArgument(1);
    col = unaryBlock->getArgument(2);
  }

  Value opResult;

  ////////////////////////////////////
  // Unary with one argument
  ////////////////////////////////////
  if (unaryOp == "abs") {
    opResult =
        llvm::TypeSwitch<Type, Value>(valueType)
            .Case<IntegerType>([&](IntegerType type) {
              // http://graphics.stanford.edu/~seander/bithacks.html#IntegerAbs
              unsigned bitWidth = type.getWidth();
              Value shiftAmount = builder.create<ConstantOp>(
                  loc, builder.getIntegerAttr(type, bitWidth - 1));
              Value mask =
                  builder.create<arith::ShRSIOp>(loc, val, shiftAmount);
              Value maskPlusVal = builder.create<arith::AddIOp>(loc, mask, val);
              return builder.create<arith::XOrIOp>(loc, mask, maskPlusVal);
            })
            .Case<FloatType>([&](FloatType type) {
              return builder.create<math::AbsOp>(loc, val);
            });
  } else if (unaryOp == "ainv") {
    opResult = llvm::TypeSwitch<Type, Value>(valueType)
                   .Case<IntegerType>([&](IntegerType type) {
                     Value c0_type = builder.create<ConstantOp>(
                         loc, builder.getIntegerAttr(type, 0));
                     return builder.create<arith::SubIOp>(loc, c0_type, val);
                   })
                   .Case<FloatType>([&](FloatType type) {
                     return builder.create<arith::NegFOp>(loc, val);
                   });
  } else if (unaryOp == "acos") {
    if (!valueType.isa<FloatType>())
      return unaryRegion->getParentOp()->emitError(
          "acos requires float type input");
    // acos = atan2(sqrt(1-val**2), val)
    Value cf1 = builder.create<arith::ConstantFloatOp>(
        loc, APFloat(1.0), valueType.cast<FloatType>());
    Value tmp = builder.create<arith::MulFOp>(loc, val, val);
    tmp = builder.create<arith::SubFOp>(loc, cf1, tmp);
    tmp = builder.create<math::SqrtOp>(loc, tmp);
    opResult = builder.create<math::Atan2Op>(loc, tmp, val);
  } else if (unaryOp == "asin") {
    if (!valueType.isa<FloatType>())
      return unaryRegion->getParentOp()->emitError(
          "asin requires float type input");
    // asin = atan2(val, sqrt(1-val**2))
    Value cf1 = builder.create<arith::ConstantFloatOp>(
        loc, APFloat(1.0), valueType.cast<FloatType>());
    Value tmp = builder.create<arith::MulFOp>(loc, val, val);
    tmp = builder.create<arith::SubFOp>(loc, cf1, tmp);
    tmp = builder.create<math::SqrtOp>(loc, tmp);
    opResult = builder.create<math::Atan2Op>(loc, val, tmp);
  } else if (unaryOp == "tan") {
    if (!valueType.isa<FloatType>())
      return unaryRegion->getParentOp()->emitError(
          "tan requires float type input");
    Value tmp = builder.create<math::CosOp>(loc, val);
    opResult = builder.create<math::SinOp>(loc, val);
    opResult = builder.create<arith::DivFOp>(loc, opResult, tmp);
  } else if (unaryOp == "cos") {
    if (!valueType.isa<FloatType>())
      return unaryRegion->getParentOp()->emitError(
          "cos requires float type input");
    opResult = builder.create<math::CosOp>(loc, val);
  } else if (unaryOp == "minv") {
    opResult =
        llvm::TypeSwitch<Type, Value>(valueType)
            .Case<IntegerType>([&](IntegerType type) {
              // TODO we're missing python tests for all ops when given
              // integer-typed tensors
              unsigned bitWidth = type.getWidth();
              Value shiftAmount = builder.create<ConstantOp>(
                  loc, builder.getIntegerAttr(type, bitWidth - 1));
              Value mask =
                  builder.create<arith::ShRSIOp>(loc, val, shiftAmount);
              Value maskPlusVal = builder.create<arith::AddIOp>(loc, mask, val);
              Value absVal =
                  builder.create<arith::XOrIOp>(loc, mask, maskPlusVal);
              Value c1_type = builder.create<arith::ConstantOp>(
                  loc, builder.getIntegerAttr(type, 1));
              Value absValIsOne_i1 = builder.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::eq, absVal, c1_type);
              Value absValIsOne_type =
                  builder.create<arith::ExtSIOp>(loc, type, absValIsOne_i1);
              return builder.create<arith::AndIOp>(loc, absValIsOne_type, val);
            })
            .Case<FloatType>([&](FloatType type) {
              Value c1_type = builder.create<arith::ConstantFloatOp>(
                  loc, APFloat(1.0), type);
              return builder.create<arith::DivFOp>(loc, c1_type, val);
            });
  } else if (unaryOp == "sin") {
    if (!valueType.isa<FloatType>())
      return unaryRegion->getParentOp()->emitError(
          "sin requires float type input");
    opResult = builder.create<math::SinOp>(loc, val);
  } else if (unaryOp == "sqrt") {
    if (!valueType.isa<FloatType>())
      return unaryRegion->getParentOp()->emitError(
          "sqrt requires float type input");
    opResult = builder.create<math::SqrtOp>(loc, val);
  }

  ////////////////////////////////////
  // Unary with three arguments
  ////////////////////////////////////
  else if (unaryOp == "column") {
    opResult = col;
  } else if (unaryOp == "index") {
    // For Vectors, (row, col) is set as (index, index), so choose either
    opResult = row;
  } else if (unaryOp == "row") {
    opResult = row;
  } else if (unaryOp == "triu") {
    opResult =
        builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, col, row);
  } else if (unaryOp == "tril") {
    opResult =
        builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, col, row);
  }

  builder.create<graphblas::YieldOp>(loc, yieldKind, opResult);

  return success();
}

LogicalResult populateBinary(OpBuilder &builder, Location loc,
                             StringRef binaryOp, Type valueType,
                             RegionRange regions,
                             graphblas::YieldKind yieldKind, bool boolAsI8) {
  // This function must match the options supported by
  // GraphBLASOps.cpp::checkBinaryOp()

  Type indexType = builder.getIndexType();
  Type i1Type = builder.getI1Type();
  Type i8Type = builder.getI8Type();
  Value false8 = builder.create<arith::ConstantIntOp>(loc, 0, i8Type);
  Value true8 = builder.create<arith::ConstantIntOp>(loc, 1, i8Type);

  // Insert binary operation
  Region *binaryRegion = regions[0];
  TypeRange inputTypes;
  if (binary2.contains(binaryOp))
    inputTypes = TypeRange{valueType, valueType};
  else if (binary4.contains(binaryOp))
    inputTypes = TypeRange{valueType, valueType, indexType, indexType};
  else if (binary5.contains(binaryOp))
    inputTypes =
        TypeRange{valueType, valueType, indexType, indexType, indexType};
  else
    return binaryRegion->getParentOp()->emitError(
        "\"" + binaryOp + "\" is not a supported binary operation.");

  Block *binaryBlock = builder.createBlock(binaryRegion, {}, inputTypes);
  int numArgs = binaryBlock->getArguments().size();

  Value aVal = binaryBlock->getArgument(0);
  Value bVal = binaryBlock->getArgument(1);
  Value row, col, overlap;
  if (numArgs >= 4) {
    row = binaryBlock->getArgument(2);
    col = binaryBlock->getArgument(3);
  }
  if (numArgs >= 5)
    overlap = binaryBlock->getArgument(4);

  Value opResult;

  ////////////////////////////////////
  // Binary with two arguments
  ////////////////////////////////////
  if (binaryOp == "div") {
    opResult = llvm::TypeSwitch<Type, Value>(valueType)
                   .Case<IntegerType>([&](IntegerType type) {
                     return builder.create<arith::DivSIOp>(loc, aVal, bVal);
                   })
                   .Case<FloatType>([&](FloatType type) {
                     return builder.create<arith::DivFOp>(loc, aVal, bVal);
                   });
  } else if (binaryOp == "eq") {
    opResult = llvm::TypeSwitch<Type, Value>(valueType)
                   .Case<IntegerType>([&](IntegerType type) {
                     return builder.create<arith::CmpIOp>(
                         loc, arith::CmpIPredicate::eq, aVal, bVal);
                   })
                   .Case<FloatType>([&](FloatType type) {
                     return builder.create<arith::CmpFOp>(
                         loc, arith::CmpFPredicate::OEQ, aVal, bVal);
                   });
  } else if (binaryOp == "first") {
    opResult = aVal;
  } else if (binaryOp == "ge") {
    opResult = llvm::TypeSwitch<Type, Value>(valueType)
                   .Case<IntegerType>([&](IntegerType type) {
                     return builder.create<arith::CmpIOp>(
                         loc, arith::CmpIPredicate::sge, aVal, bVal);
                   })
                   .Case<FloatType>([&](FloatType type) {
                     return builder.create<arith::CmpFOp>(
                         loc, arith::CmpFPredicate::OGE, aVal, bVal);
                   });
  } else if (binaryOp == "gt") {
    opResult = llvm::TypeSwitch<Type, Value>(valueType)
                   .Case<IntegerType>([&](IntegerType type) {
                     return builder.create<arith::CmpIOp>(
                         loc, arith::CmpIPredicate::sgt, aVal, bVal);
                   })
                   .Case<FloatType>([&](FloatType type) {
                     return builder.create<arith::CmpFOp>(
                         loc, arith::CmpFPredicate::OGT, aVal, bVal);
                   });
  } else if (binaryOp == "le") {
    opResult = llvm::TypeSwitch<Type, Value>(valueType)
                   .Case<IntegerType>([&](IntegerType type) {
                     return builder.create<arith::CmpIOp>(
                         loc, arith::CmpIPredicate::sle, aVal, bVal);
                   })
                   .Case<FloatType>([&](FloatType type) {
                     return builder.create<arith::CmpFOp>(
                         loc, arith::CmpFPredicate::OLE, aVal, bVal);
                   });
  } else if (binaryOp == "lt") {
    opResult = llvm::TypeSwitch<Type, Value>(valueType)
                   .Case<IntegerType>([&](IntegerType type) {
                     return builder.create<arith::CmpIOp>(
                         loc, arith::CmpIPredicate::slt, aVal, bVal);
                   })
                   .Case<FloatType>([&](FloatType type) {
                     return builder.create<arith::CmpFOp>(
                         loc, arith::CmpFPredicate::OLT, aVal, bVal);
                   });
  } else if (binaryOp == "max") {
    Value cmp = llvm::TypeSwitch<Type, Value>(valueType)
                    .Case<IntegerType>([&](IntegerType type) {
                      return builder.create<arith::CmpIOp>(
                          loc, arith::CmpIPredicate::sgt, aVal, bVal);
                    })
                    .Case<FloatType>([&](FloatType type) {
                      return builder.create<arith::CmpFOp>(
                          loc, arith::CmpFPredicate::OGT, aVal, bVal);
                    });
    opResult = builder.create<SelectOp>(loc, cmp, aVal, bVal);
  } else if (binaryOp == "min") {
    Value cmp = llvm::TypeSwitch<Type, Value>(valueType)
                    .Case<IntegerType>([&](IntegerType type) {
                      return builder.create<arith::CmpIOp>(
                          loc, arith::CmpIPredicate::slt, aVal, bVal);
                    })
                    .Case<FloatType>([&](FloatType type) {
                      return builder.create<arith::CmpFOp>(
                          loc, arith::CmpFPredicate::OLT, aVal, bVal);
                    });
    opResult = builder.create<SelectOp>(loc, cmp, aVal, bVal);
  } else if (binaryOp == "minus") {
    opResult = llvm::TypeSwitch<Type, Value>(valueType)
                   .Case<IntegerType>([&](IntegerType type) {
                     return builder.create<arith::SubIOp>(loc, aVal, bVal);
                   })
                   .Case<FloatType>([&](FloatType type) {
                     return builder.create<arith::SubFOp>(loc, aVal, bVal);
                   });
  } else if (binaryOp == "ne") {
    opResult = llvm::TypeSwitch<Type, Value>(valueType)
                   .Case<IntegerType>([&](IntegerType type) {
                     return builder.create<arith::CmpIOp>(
                         loc, arith::CmpIPredicate::ne, aVal, bVal);
                   })
                   .Case<FloatType>([&](FloatType type) {
                     return builder.create<arith::CmpFOp>(
                         loc, arith::CmpFPredicate::ONE, aVal, bVal);
                   });
  } else if (binaryOp == "pair") {
    opResult = llvm::TypeSwitch<Type, Value>(valueType)
                   .Case<IntegerType>([&](IntegerType type) {
                     return builder.create<arith::ConstantIntOp>(
                         loc, 1, type.getWidth());
                   })
                   .Case<FloatType>([&](FloatType type) {
                     return builder.create<arith::ConstantFloatOp>(
                         loc, APFloat(1.0), type);
                   });
  } else if (binaryOp == "plus") {
    opResult = llvm::TypeSwitch<Type, Value>(valueType)
                   .Case<IntegerType>([&](IntegerType type) {
                     return builder.create<arith::AddIOp>(loc, aVal, bVal);
                   })
                   .Case<FloatType>([&](FloatType type) {
                     return builder.create<arith::AddFOp>(loc, aVal, bVal);
                   });
  } else if (binaryOp == "pow") {
    if (!valueType.isa<FloatType>())
      return binaryRegion->getParentOp()->emitError(
          "pow requires float type input");
    opResult = builder.create<math::PowFOp>(loc, aVal, bVal);
  } else if (binaryOp == "second") {
    opResult = bVal;
  } else if (binaryOp == "times") {
    opResult = llvm::TypeSwitch<Type, Value>(valueType)
                   .Case<IntegerType>([&](IntegerType type) {
                     return builder.create<arith::MulIOp>(loc, aVal, bVal);
                   })
                   .Case<FloatType>([&](FloatType type) {
                     return builder.create<arith::MulFOp>(loc, aVal, bVal);
                   });
  }

  ////////////////////////////////////
  // Binary with four arguments
  ////////////////////////////////////
  else if (binaryOp == "column") {
    opResult =
        llvm::TypeSwitch<Type, Value>(valueType)
            .Case<IntegerType>([&](IntegerType type) {
              return builder.create<arith::IndexCastOp>(loc, col, valueType);
            })
            .Case<FloatType>([&](FloatType type) {
              Type intType = builder.getIntegerType(type.getWidth());
              Value intValue =
                  builder.create<arith::IndexCastOp>(loc, col, intType);
              return builder.create<arith::SIToFPOp>(loc, type, intValue);
            });
  } else if (binaryOp == "row") {
    opResult =
        llvm::TypeSwitch<Type, Value>(valueType)
            .Case<IntegerType>([&](IntegerType type) {
              return builder.create<arith::IndexCastOp>(loc, row, valueType);
            })
            .Case<FloatType>([&](FloatType type) {
              unsigned bitWidth = type.getWidth();
              Type intType = builder.getIntegerType(bitWidth);
              Value intValue =
                  builder.create<arith::IndexCastOp>(loc, row, intType);
              return builder.create<arith::SIToFPOp>(loc, type, intValue);
            });
  }

  ////////////////////////////////////
  // Binary with five arguments
  ////////////////////////////////////
  else if (binaryOp == "overlapi") {
    opResult =
        llvm::TypeSwitch<Type, Value>(valueType)
            .Case<IntegerType>([&](IntegerType type) {
              return builder.create<arith::IndexCastOp>(loc, overlap,
                                                        valueType);
            })
            .Case<FloatType>([&](FloatType type) {
              unsigned bitWidth = type.getWidth();
              Type intType = builder.getIntegerType(bitWidth);
              Value intValue =
                  builder.create<arith::IndexCastOp>(loc, overlap, intType);
              return builder.create<arith::SIToFPOp>(loc, type, intValue);
            });
  }

  // Cannot store i1 in sparse tensor, so convert to i8 if needed
  if (boolAsI8 && opResult.getType() == i1Type)
    opResult = builder.create<SelectOp>(loc, opResult, true8, false8);

  builder.create<graphblas::YieldOp>(loc, yieldKind, opResult);

  return success();
}

LogicalResult populateMonoid(OpBuilder &builder, Location loc,
                             StringRef monoidOp, Type valueType,
                             RegionRange regions,
                             graphblas::YieldKind yieldIdentity,
                             graphblas::YieldKind yieldKind) {
  // This function must match the options supported by
  // GraphBLASOps.cpp::checkMonoidOp()

  Region *identityRegion = regions[0];
  Region *opRegion = regions[1];

  TypeRange inputTypes;
  if (monoid2.contains(monoidOp))
    inputTypes = TypeRange{valueType, valueType};
  else
    return identityRegion->getParentOp()->emitError(
        "\"" + monoidOp + "\" is not a supported monoid.");

  // Insert monoid identity
  /*Block *identityBlock = */ builder.createBlock(identityRegion, {}, {});
  Value identity;
  if (monoidOp == "plus" || monoidOp == "any") {
    // Add identity
    identity = llvm::TypeSwitch<Type, Value>(valueType)
                   .Case<IntegerType>([&](IntegerType type) {
                     return builder.create<arith::ConstantIntOp>(
                         loc, 0, type.getWidth());
                   })
                   .Case<FloatType>([&](FloatType type) {
                     return builder.create<arith::ConstantFloatOp>(
                         loc, APFloat(0.0), type);
                   });
  } else if (monoidOp == "min") {
    identity =
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
  }

  builder.create<graphblas::YieldOp>(loc, yieldIdentity, identity);

  // Insert operation
  Block *opBlock = builder.createBlock(opRegion, {}, inputTypes);
  Value aVal = opBlock->getArgument(0);
  Value bVal = opBlock->getArgument(1);
  Value opResult;
  if (monoidOp == "plus") {
    opResult = llvm::TypeSwitch<Type, Value>(valueType)
                   .Case<IntegerType>([&](IntegerType type) {
                     return builder.create<arith::AddIOp>(loc, aVal, bVal);
                   })
                   .Case<FloatType>([&](FloatType type) {
                     return builder.create<arith::AddFOp>(loc, aVal, bVal);
                   });
  } else if (monoidOp == "min") {
    Value cmp = llvm::TypeSwitch<Type, Value>(valueType)
                    .Case<IntegerType>([&](IntegerType type) {
                      return builder.create<arith::CmpIOp>(
                          loc, arith::CmpIPredicate::slt, aVal, bVal);
                    })
                    .Case<FloatType>([&](FloatType type) {
                      return builder.create<arith::CmpFOp>(
                          loc, arith::CmpFPredicate::OLT, aVal, bVal);
                    });
    opResult = builder.create<SelectOp>(loc, cmp, aVal, bVal);
  } else if (monoidOp == "any") {
    // Same as "second" for multiplicative op?
    // https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/master/GraphBLAS/Source/Template/GB_binop_factory.c#L243-L244
    opResult = bVal;
  }

  builder.create<graphblas::YieldOp>(loc, yieldKind, opResult);

  return success();
}

LogicalResult populateSemiring(OpBuilder &builder, Location loc,
                               StringRef semiringOp, Type valueType,
                               RegionRange regions) {
  auto semiringParts = semiringOp.split('_');

  if (failed(populateMonoid(
          builder, loc, semiringParts.first, valueType, regions.slice(0, 2),
          graphblas::YieldKind::ADD_IDENTITY, graphblas::YieldKind::ADD)))
    return failure();

  if (failed(populateBinary(builder, loc, semiringParts.second, valueType,
                            regions.slice(2, 1), graphblas::YieldKind::MULT)))
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
