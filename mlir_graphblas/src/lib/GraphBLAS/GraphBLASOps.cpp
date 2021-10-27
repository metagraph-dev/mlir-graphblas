//===- GraphBLASOps.cpp - GraphBLAS dialect ops ---------------*- C++ -*-===//
//
// TODO add documentation
//
//===--------------------------------------------------------------------===//

#include "GraphBLAS/GraphBLASOps.h"
#include "GraphBLAS/GraphBLASDialect.h"
#include "GraphBLAS/GraphBLASUtils.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include "GraphBLAS/GraphBLASOpsDialect.cpp.inc"
#include "GraphBLAS/GraphBLASOpsEnums.cpp.inc"
#include "GraphBLAS/GraphBLASUtils.h"

#include <numeric>
#include <unordered_set>

using namespace mlir;
using namespace mlir::graphblas;

//===--------------------------------------------------------------------===//
// Helpers
//===--------------------------------------------------------------------===//

/// Utility function to check encoding attribute.
static LogicalResult hasSparseEncodingAttr(RankedTensorType t) {
  if (!sparse_tensor::getSparseTensorEncoding(t))
    return failure();
  return success();
}

static LogicalResult verifySameShape(ArrayRef<int64_t> aShape,
                                     ArrayRef<int64_t> bShape) {
  if (aShape.size() != bShape.size())
    return failure();
  for (unsigned i = 0; i < aShape.size(); i++) {
    if (aShape[i] != bShape[i])
      return failure();
  }
  return success();
}

static LogicalResult verifySameShape(RankedTensorType aType,
                                     RankedTensorType bType) {
  ArrayRef<int64_t> aShape = aType.getShape();
  ArrayRef<int64_t> bShape = bType.getShape();
  return verifySameShape(aShape, bShape);
}

static llvm::Optional<std::string>
checkMatrixEncoding(RankedTensorType tensorType,
                    CompressionType compressionType) {
  /*
     Returns llvm::None if the given matrix is valid.
     Returns a string explaining the problem otherwise.
   */
  // Must have sparse encoding
  sparse_tensor::SparseTensorEncodingAttr sparseEncoding =
      sparse_tensor::getSparseTensorEncoding(tensorType);
  if (!sparseEncoding)
    return std::string("must be a sparse tensor.");
  // Must be rank 2
  if (tensorType.getRank() != 2)
    return std::string("must have rank 2.");
  // Must be [dense, compressed] dimLevelType (for currently available
  // compression types)
  ArrayRef<mlir::sparse_tensor::SparseTensorEncodingAttr::DimLevelType>
      compression = sparseEncoding.getDimLevelType();
  if (compression[0] !=
          mlir::sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Dense ||
      compression[1] != mlir::sparse_tensor::SparseTensorEncodingAttr::
                            DimLevelType::Compressed)
    return std::string("must have CSR or CSC compression, i.e. must have "
                       "dimLevelType = [ \"dense\", \"compressed\" ] in the "
                       "sparse encoding.");

  // Even if (compressionType == EITHER), we still need to check that the
  // dim ordering is present
  AffineMap dimOrdering = sparseEncoding.getDimOrdering();
  if (!dimOrdering)
    return std::string("dimOrdering must be present in sparse tensor encoding");
  if (dimOrdering.getNumResults() != 2)
    return std::string("dimOrdering must have exactly 2 outputs");
  unsigned dimOrdering0 = dimOrdering.getDimPosition(0);
  unsigned dimOrdering1 = dimOrdering.getDimPosition(1);

  if (compressionType == CSR) {
    if (dimOrdering0 != 0 || dimOrdering1 != 1)
      return std::string("must have CSR compression.");
  } else if (compressionType == CSC) {
    if (dimOrdering0 != 1 || dimOrdering1 != 0)
      return std::string("must have CSC compression.");
  }

  return llvm::None;
}

static llvm::Optional<std::string>
checkVectorEncoding(RankedTensorType tensorType) {
  /*
     Returns llvm::None if the given matrix is valid.
     Returns a string explaining the problem otherwise.
   */
  // Must have sparse encoding
  sparse_tensor::SparseTensorEncodingAttr sparseEncoding =
      sparse_tensor::getSparseTensorEncoding(tensorType);
  if (!sparseEncoding)
    return std::string("must be a sparse tensor.");
  // Must be rank 1
  if (tensorType.getRank() != 1)
    return std::string("must have rank 1.");
  // Must be [compressed] dimLevelType (until we decide to support dense
  // vectors)
  ArrayRef<mlir::sparse_tensor::SparseTensorEncodingAttr::DimLevelType>
      compression = sparseEncoding.getDimLevelType();
  if (compression[0] !=
      mlir::sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Compressed)
    return std::string(
        "must be sparse, i.e. must have "
        "dimLevelType = [ \"compressed\" ] in the sparse encoding.");

  return llvm::None;
}

static llvm::Optional<std::string> checkBitWidthMatch(RankedTensorType a,
                                                      RankedTensorType b) {
  sparse_tensor::SparseTensorEncodingAttr aEncoding =
      sparse_tensor::getSparseTensorEncoding(a);
  sparse_tensor::SparseTensorEncodingAttr bEncoding =
      sparse_tensor::getSparseTensorEncoding(b);
  unsigned aPtr = aEncoding.getPointerBitWidth();
  unsigned bPtr = bEncoding.getPointerBitWidth();
  if (aPtr != bPtr)
    return "pointer bit widths do not match: " + std::to_string(aPtr) +
           "!=" + std::to_string(bPtr);

  unsigned aIdx = aEncoding.getIndexBitWidth();
  unsigned bIdx = bEncoding.getIndexBitWidth();
  if (aIdx != bIdx)
    return "index bit widths do not match: " + std::to_string(aIdx) +
           "!=" + std::to_string(bIdx);

  return llvm::None;
}

static llvm::Optional<std::string> checkSemiringAdd(StringRef addName) {
  if (!supportedSemiringAddNames.contains(addName))
    return "\"" + addName.str() + "\" is not a supported semiring add.";
  else
    return llvm::None;
}

static llvm::Optional<std::string>
checkSemiringMultiply(StringRef multiplyName) {
  if (!supportedSemiringMultiplyNames.contains(multiplyName))
    return "\"" + multiplyName.str() +
           "\" is not a supported semiring multiply.";
  else
    return llvm::None;
}

static llvm::Optional<std::string> checkSemiring(StringRef semiring) {
  auto semiringParts = semiring.split('_');
  auto addCheck = checkSemiringAdd(semiringParts.first);
  if (addCheck != llvm::None)
    return addCheck;

  auto multiplyCheck = checkSemiringMultiply(semiringParts.second);
  if (multiplyCheck != llvm::None)
    return multiplyCheck;

  return llvm::None;
}

//===--------------------------------------------------------------------===//
// GraphBLAS Ops Methods
//===--------------------------------------------------------------------===//

void SizeOp::build(OpBuilder &builder, OperationState &result, Value tensor) {
  Type indexType = builder.getIndexType();
  build(builder, result, indexType, tensor);
}

void NumRowsOp::build(OpBuilder &builder, OperationState &result,
                      Value tensor) {
  Type indexType = builder.getIndexType();
  build(builder, result, indexType, tensor);
}

void NumColsOp::build(OpBuilder &builder, OperationState &result,
                      Value tensor) {
  Type indexType = builder.getIndexType();
  build(builder, result, indexType, tensor);
}

static LogicalResult verify(NumValsOp op) {
  RankedTensorType inputType = op.input().getType().cast<RankedTensorType>();
  llvm::Optional<std::string> errMsg;
  if (inputType.getRank() == 2) {
    errMsg = checkMatrixEncoding(inputType, EITHER);
    if (errMsg)
      op.emitError("operand " + errMsg.getValue());
  } else {
    errMsg = checkVectorEncoding(inputType);
    if (errMsg)
      op.emitError("operand " + errMsg.getValue());
  }
  return success();
}

void NumValsOp::build(OpBuilder &builder, OperationState &result,
                      Value tensor) {
  Type indexType = builder.getIndexType();
  build(builder, result, indexType, tensor);
}

static LogicalResult verify(DupOp op) {
  // Dup needs a sparse tensor, but otherwise works for any rank or compression
  // type
  RankedTensorType inputType = op.input().getType().cast<RankedTensorType>();
  if (failed(hasSparseEncodingAttr(inputType)))
    return op.emitError("operand must be a sparse tensor");
  return success();
}

void DupOp::build(OpBuilder &builder, OperationState &result, Value tensor) {
  Type inputType = tensor.getType();
  build(builder, result, inputType, tensor);
}

template <class T>
static LogicalResult verifyApplyArgs(T op, Value input) {
  Type inputType = input.getType();
  Type resultType = op.getResult().getType();

  RankedTensorType inputTensorType = inputType.cast<RankedTensorType>();
  RankedTensorType resultTensorType = resultType.cast<RankedTensorType>();

  ArrayRef<int64_t> inputShape = inputTensorType.getShape();
  ArrayRef<int64_t> resultShape = resultTensorType.getShape();
  if (inputShape.size() != resultShape.size())
    return op.emitError("operand and result must have the same rank");
  for (unsigned i = 0; i < inputShape.size(); i++) {
    if (inputShape[i] != resultShape[i])
      return op.emitError("operand and result shapes must match");
  }

  llvm::Optional<std::string> errMsg;
  if (resultTensorType.getRank() == 2) {
    errMsg = checkMatrixEncoding(inputTensorType, EITHER);
    if (errMsg)
      return op.emitError("operand " + errMsg.getValue());

    // Result must be ordered the same as input
    if (hasRowOrdering(inputTensorType))
      errMsg = checkMatrixEncoding(resultTensorType, CSR);
    else
      errMsg = checkMatrixEncoding(resultTensorType, CSC);
    if (errMsg)
      return op.emitError("result " + errMsg.getValue());
  } else {
    errMsg = checkVectorEncoding(inputTensorType);
    if (errMsg)
      return op.emitError("operand " + errMsg.getValue());

    errMsg = checkVectorEncoding(resultTensorType);
    if (errMsg)
      return op.emitError("result " + errMsg.getValue());
  }
  return success();
}

static LogicalResult verify(ApplyOp op) {
  Value input, thunk;
  LogicalResult extractArgResult = extractApplyOpArgs(op, input, thunk);
  if (extractArgResult.failed())
    return extractArgResult;

  LogicalResult argResult = verifyApplyArgs(op, input);

  if (argResult.failed())
    return argResult;

  Type inputType = input.getType();
  Type resultType = op.getResult().getType();

  std::string applyOperator = op.apply_operator().str();
  if (supportedBinaryApplyOperators.contains(applyOperator)) {

    if (!thunk)
      return op.emitError("\"" + applyOperator +
                          "\""
                          " requires a thunk.");

    RankedTensorType inputTensorType = inputType.dyn_cast<RankedTensorType>();
    RankedTensorType resultTensorType = resultType.dyn_cast<RankedTensorType>();

    Type thunkType = thunk.getType();

    if (inputTensorType.getElementType() != thunkType)
      return op.emitError(
          "Element type of input tensor does not match type of thunk.");

    if (resultTensorType.getElementType() != thunkType)
      // TODO this is not always correct, e.g.
      // apply_less_than(tensor<f64>, 2.3) -> tensor<i1>.
      return op.emitError(
          "Element type of result tensor does not match type of thunk.");

  } else if (supportedUnaryApplyOperators.contains(applyOperator)) {

    if (thunk)
      return op.emitError("\"" + applyOperator +
                          "\""
                          " is a unary operator, but was given a thunk.");

  } else {
    return op.emitError("\"" + applyOperator +
                        "\" is not a supported operator.");
  }

  return success();
}

static LogicalResult verify(ApplyGenericOp op) {
  LogicalResult argResult = verifyApplyArgs(op, op.input());

  if (argResult.failed())
    return argResult;

  RegionRange extensions = op.extensions();
  if (extensions.size() < 1) {
    return op.emitError("Must have at least 1 region: transform_out.");
  }

  return success();
}

template <class T>
static LogicalResult verifyMatrixMultiplyArgs(T op,
                                              bool checkResultTensorType) {
  Type aOrigType = op.a().getType();
  Type bOrigType = op.b().getType();
  Type resultOrigType = op.getResult().getType();

  RankedTensorType aType = aOrigType.cast<RankedTensorType>();
  RankedTensorType bType = bOrigType.cast<RankedTensorType>();

  ArrayRef<int64_t> aShape = aType.getShape();
  ArrayRef<int64_t> bShape = bType.getShape();

  int64_t aRank = aType.getRank();
  int64_t bRank = bType.getRank();

  int64_t resultRank = 0;
  ArrayRef<int64_t> resultShape;
  Type resultElementType = resultOrigType;
  RankedTensorType resultType;

  // Vector-vector result is a scalar; Otherwise, get the tensor properties of
  // the result
  if (checkResultTensorType) {
    if (aRank == 2 || bRank == 2) {
      resultType = resultOrigType.cast<RankedTensorType>();
      resultShape = resultType.getShape();
      resultRank = resultType.getRank();
      resultElementType = resultType.getElementType();
    } else if (!resultElementType.isIntOrFloat()) {
      assert(aRank == 1 && bRank == 1);
      return op.emitError(
          "Vector-vector multiplication must result in a scalar result-type.");
    }

    if (aType.getElementType() != resultElementType)
      return op.emitError(
          "Result element type differs from the input element types.");
  }

  if (aType.getElementType() != bType.getElementType())
    return op.emitError("Operand element types must be identical.");

  llvm::Optional<std::string> errMsg;
  if (aRank == 2 && bRank == 2) {
    // Matrix-Matrix
    errMsg = checkMatrixEncoding(aType, CSR);
    if (errMsg)
      return op.emitError("1st operand " + errMsg.getValue());

    errMsg = checkMatrixEncoding(bType, CSC);
    if (errMsg)
      return op.emitError("2nd operand " + errMsg.getValue());

    if (checkResultTensorType) {
      errMsg = checkMatrixEncoding(resultType, CSR);
      if (errMsg)
        return op.emitError("result " + errMsg.getValue());
      if (resultShape[0] != aShape[0] || resultShape[1] != bShape[1])
        return op.emitError("Operand shapes incompatible with output shape.");
    }

    // TODO intelligently handle arbitrarily shaped tensors, i.e. tensors with
    // shapes using "?"
    if (aShape[1] != bShape[0])
      return op.emitError("Operand shapes are incompatible.");
  } else if (aRank == 2 && bRank == 1) {
    // Matrix-Vector
    errMsg = checkMatrixEncoding(aType, CSR);
    if (errMsg)
      return op.emitError("1st operand " + errMsg.getValue());

    errMsg = checkVectorEncoding(bType);
    if (errMsg)
      return op.emitError("2nd operand " + errMsg.getValue());

    if (checkResultTensorType) {
      errMsg = checkVectorEncoding(resultType);
      if (errMsg)
        return op.emitError("result " + errMsg.getValue());

      if (resultShape[0] != aShape[0])
        return op.emitError("Operand shapes incompatible with output shape.");
    }
    if (aShape[1] != bShape[0])
      return op.emitError("Operand shapes are incompatible.");
  } else if (aRank == 1 && bRank == 2) {
    // Vector-Matrix
    errMsg = checkVectorEncoding(aType);
    if (errMsg)
      return op.emitError("1st operand " + errMsg.getValue());

    errMsg = checkMatrixEncoding(bType, CSC);
    if (errMsg)
      return op.emitError("2nd operand " + errMsg.getValue());

    if (aShape[0] != bShape[0])
      return op.emitError("Operand shapes are incompatible.");

    if (checkResultTensorType) {
      errMsg = checkVectorEncoding(resultType);
      if (errMsg)
        return op.emitError("result " + errMsg.getValue());
      if (resultShape[0] != bShape[1])
        return op.emitError("Operand shapes incompatible with output shape.");
    }
  } else {
    // Vector-Vector
    errMsg = checkVectorEncoding(aType);
    if (errMsg)
      return op.emitError("1st operand " + errMsg.getValue());

    errMsg = checkVectorEncoding(bType);
    if (errMsg)
      return op.emitError("2nd operand " + errMsg.getValue());

    if (aShape[0] != bShape[0])
      return op.emitError("Operand shapes are incompatible.");
  }

  Value mask = op.mask();
  if (mask) {
    RankedTensorType maskType = mask.getType().cast<RankedTensorType>();

    if (checkResultTensorType) {
      ArrayRef<int64_t> maskShape = maskType.getShape();
      if (resultRank == 2) {
        errMsg = checkMatrixEncoding(maskType, CSR);
        if (errMsg)
          return op.emitError("3rd operand (mask) " + errMsg.getValue());

        if (resultShape[0] != maskShape[0] || resultShape[1] != maskShape[1])
          return op.emitError(
              "Mask shape must match shape of matrix multiply result.");
      } else if (resultRank == 1) {
        errMsg = checkVectorEncoding(maskType);
        if (errMsg)
          return op.emitError("3rd operand (mask) " + errMsg.getValue());

        if (resultShape[0] != maskShape[0])
          return op.emitError(
              "Mask shape must match shape of matrix multiply result.");
      } else {
        return op.emitError(
            "Mask not allowed for vector times vector multiplication.");
      }
    }
  }

  return success();
}

static LogicalResult verify(MatrixMultiplyOp op) {
  LogicalResult argResult = verifyMatrixMultiplyArgs(op, true);

  if (argResult.failed())
    return argResult;

  llvm::Optional<std::string> semiringError = checkSemiring(op.semiring());
  if (semiringError != llvm::None) {
    return op.emitError(semiringError.getValue());
  }

  return success();
}

static LogicalResult verify(MatrixMultiplyGenericOp op) {
  LogicalResult argResult = verifyMatrixMultiplyArgs(op, true);

  if (argResult.failed())
    return argResult;

  RegionRange extensions = op.extensions();
  if (extensions.size() < 3) {
    return op.emitError(
        "Must have at least 3 regions: add_identity, add, mult.");
  }
  // TODO add more verification of the blocks (possibly use
  // ExtensionBlocks::extractBlocks)

  return success();
}

static LogicalResult verify(MatrixMultiplyReduceToScalarGenericOp op) {
  LogicalResult argResult =
      verifyMatrixMultiplyArgs(op, false /* no result tensor */);

  if (argResult.failed())
    return argResult;

  RegionRange extensions = op.extensions();
  if (extensions.size() < 4) {
    return op.emitError(
        "Must have at least 4 regions: add_identity, add, mult, agg.");
  }

  return success();
}

static LogicalResult verify(DiagOp op) {
  RankedTensorType inputType = op.input().getType().cast<RankedTensorType>();
  RankedTensorType resultType =
      op.getResult().getType().cast<RankedTensorType>();

  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> resultShape = resultType.getShape();

  // TODO intelligently handle arbitrarily shaped tensors, i.e. tensors with
  // shapes using "?"
  llvm::Optional<std::string> errMsg;
  if (resultType.getRank() == 1) {
    errMsg = checkMatrixEncoding(inputType, EITHER);
    if (errMsg)
      return op.emitError("operand " + errMsg.getValue());

    errMsg = checkVectorEncoding(resultType);
    if (errMsg)
      return op.emitError("result " + errMsg.getValue());

    if (inputShape[0] != inputShape[1])
      return op.emitError("Input shape must be square.");

    if (inputShape[0] != resultShape[0])
      return op.emitError("Input shape is not compatible with output shape.");
  } else {
    errMsg = checkVectorEncoding(inputType);
    if (errMsg)
      return op.emitError("operand " + errMsg.getValue());

    errMsg = checkMatrixEncoding(resultType, EITHER);
    if (errMsg)
      return op.emitError("result " + errMsg.getValue());

    if (resultShape[0] != resultShape[1])
      return op.emitError("Output shape must be square.");

    if (inputShape[0] != resultShape[0])
      return op.emitError("Input shape is not compatible with output shape.");
  }

  return success();
}

static LogicalResult verify(UpdateOp op) {
  RankedTensorType iType = op.input().getType().cast<RankedTensorType>();
  RankedTensorType oType = op.output().getType().cast<RankedTensorType>();
  Value mask = op.mask();

  int64_t iRank = iType.getRank();
  RankedTensorType maskType;
  if (mask) {
    maskType = mask.getType().cast<RankedTensorType>();
  }

  if (failed(verifySameShape(iType, oType)))
    return op.emitError(
        "Input and Output arguments must have identical shapes.");
  if (mask && failed(verifySameShape(oType, maskType)))
    return op.emitError(
        "Mask and Output arguments must have identical shapes.");

  llvm::Optional<std::string> errMsg;
  if (iRank == 1) {
    errMsg = checkVectorEncoding(iType);
    if (errMsg)
      return op.emitError("input " + errMsg.getValue());
  } else if (iRank == 2) {
    errMsg = checkMatrixEncoding(iType, EITHER);
    if (errMsg)
      return op.emitError("input " + errMsg.getValue());
  }

  if (iType != oType)
    return op.emitError("input and output must have identical types.");
  if (mask && iType != maskType)
    return op.emitError("mask and input must have identical types.");

  llvm::Optional<llvm::StringRef> accumulateOperator = op.accumulate_operator();
  if (accumulateOperator) {
    if (!supportedUpdateAccumulateOperators.contains(accumulateOperator->str()))
      return op.emitError("\"" + accumulateOperator->str() +
                          "\" is not a supported accumulate operator.");
  }

  return success();
}

template <class T>
static LogicalResult verifyEwise(T op) {
  Type aOrigType = op.a().getType();
  Type bOrigType = op.b().getType();

  RankedTensorType aType = aOrigType.cast<RankedTensorType>();
  RankedTensorType bType = bOrigType.cast<RankedTensorType>();

  if (failed(verifySameShape(aType, bType)))
    return op.emitError("Inputs must have identical shapes.");

  int64_t aRank = aType.getRank();

  llvm::Optional<std::string> errMsg;
  if (aRank == 1) {
    errMsg = checkVectorEncoding(aType);
    if (errMsg)
      return op.emitError("1st operand " + errMsg.getValue());
  } else if (aRank == 2) {
    errMsg = checkMatrixEncoding(aType, EITHER);
    if (errMsg)
      return op.emitError("1st operand " + errMsg.getValue());
  }
  if (aType != bType)
    return op.emitError("operands must have identical types.");

  return success();
}

static LogicalResult verify(UnionOp op) {
  llvm::Optional<llvm::StringRef> unionOperator = op.union_operator();
  if (unionOperator) {
    if (!supportedUnionOperators.contains(unionOperator->str()))
      return op.emitError("\"" + unionOperator->str() +
                          "\" is not a supported union operator.");
  }

  return verifyEwise(op);
}

static LogicalResult verify(IntersectOp op) {
  llvm::Optional<llvm::StringRef> intersectOperator = op.intersect_operator();
  if (intersectOperator) {
    if (!supportedIntersectOperators.contains(intersectOperator->str()))
      return op.emitError("\"" + intersectOperator->str() +
                          "\" is not a supported intersect operator.");
  }

  return verifyEwise(op);
}

static LogicalResult verify(EqualOp op) {
  // TODO: this might need to be separate once masks are available for union and
  // intersect
  return verifyEwise(op);
}

static LogicalResult verify(ReduceToVectorOp op) {
  std::string aggregator = op.aggregator().str();
  if (!supportedReduceAggregators.contains(aggregator))
    return op.emitError("\"" + aggregator +
                        "\" is not a supported aggregator.");

  RankedTensorType inputType = op.input().getType().cast<RankedTensorType>();

  llvm::Optional<std::string> errMsg;
  errMsg = checkMatrixEncoding(inputType, EITHER);
  if (errMsg)
    return op.emitError("operand " + errMsg.getValue());

  RankedTensorType resultType =
      op.getResult().getType().cast<RankedTensorType>();

  errMsg = checkVectorEncoding(resultType);
  if (errMsg)
    return op.emitError("result " + errMsg.getValue());

  if (aggregator == "argmin" or aggregator == "argmax") {
    Type valueType = resultType.getElementType();
    bool valueTypeIsI64 = llvm::TypeSwitch<Type, bool>(valueType)
                              .Case<IntegerType>([&](IntegerType type) {
                                unsigned bitWidth = type.getWidth();
                                return bitWidth == 64;
                              })
                              .Default([&](Type type) { return false; });
    if (!valueTypeIsI64)
      return op.emitError(
          "\"" + aggregator +
          "\" requires the output vector to have i64 elements.");
  } else if (resultType.getElementType() != inputType.getElementType())
    return op.emitError("Operand and output types are incompatible.");

  ArrayRef<int64_t> inputShape = inputType.getShape();

  int axis = op.axis();
  int expectedResultLength;
  if (axis == 0) {
    expectedResultLength = inputShape[1];
  } else if (axis == 1) {
    expectedResultLength = inputShape[0];
  } else {
    return op.emitError("The axis attribute is expected to be 0 or 1.");
  }

  ArrayRef<int64_t> resultShape = resultType.getShape();
  if (resultShape[0] != expectedResultLength) {
    return op.emitError("Operand and output shapes are incompatible.");
  }

  return success();
}

template <class T>
static LogicalResult verifyReduceToScalarArgs(T op) {
  Type operandOrigType = op.input().getType();

  RankedTensorType operandType = operandOrigType.cast<RankedTensorType>();

  int64_t rank = operandType.getRank();

  llvm::Optional<std::string> errMsg;
  if (rank == 1) {
    errMsg = checkVectorEncoding(operandType);
    if (errMsg)
      return op.emitError("operand " + errMsg.getValue());
  } else {
    errMsg = checkMatrixEncoding(operandType, EITHER);
    if (errMsg)
      return op.emitError("operand " + errMsg.getValue());
  }

  return success();
}

static LogicalResult verify(ReduceToScalarOp op) {
  LogicalResult argResult = verifyReduceToScalarArgs(op);

  if (argResult.failed())
    return argResult;

  std::string aggregator = op.aggregator().str();
  if (!supportedReduceAggregators.contains(aggregator))
    return op.emitError("\"" + aggregator +
                        "\" is not a supported aggregator.");

  Type operandOrigType = op.input().getType();
  RankedTensorType operandType = operandOrigType.cast<RankedTensorType>();
  Type resultType = op.getResult().getType();
  if (aggregator == "argmin" or aggregator == "argmax" or
      aggregator == "count") {
    if (operandType.getRank() != 1 and aggregator != "count")
      return op.emitError("\"" + aggregator + "\" only supported for vectors.");
    bool resultTypeIsI64 = llvm::TypeSwitch<Type, bool>(resultType)
                               .Case<IntegerType>([&](IntegerType type) {
                                 unsigned bitWidth = type.getWidth();
                                 return bitWidth == 64;
                               })
                               .Default([&](Type type) { return false; });
    if (!resultTypeIsI64)
      return op.emitError("\"" + aggregator +
                          "\" requires the output type to be i64.");
  } else if (resultType != operandType.getElementType())
    return op.emitError("Operand and output types are incompatible.");

  return success();
}

static LogicalResult verify(ReduceToScalarGenericOp op) {
  LogicalResult argResult = verifyReduceToScalarArgs(op);

  if (argResult.failed())
    return argResult;

  RegionRange extensions = op.extensions();
  if (extensions.size() < 1) {
    return op.emitError("Must have at least 2 regions: agg_identity, agg.");
  }

  return success();
}

static LogicalResult verify(graphblas::SelectOp op) {
  RankedTensorType inputType = op.input().getType().cast<RankedTensorType>();
  unsigned rank = inputType.getRank();

  llvm::Optional<std::string> errMsg;
  if (rank == 2)
    errMsg = checkMatrixEncoding(inputType, EITHER);
  else
    errMsg = checkVectorEncoding(inputType);
  if (errMsg)
    return op.emitError("input " + errMsg.getValue());

  for (OpResult result : op.getResults()) {
    RankedTensorType resultType = result.getType().cast<RankedTensorType>();

    if (inputType != resultType)
      return op.emitError(
          "At least 1 result type does not match that of the input matrix.");
  }

  std::vector<std::string> selectorsNeedingThunk;
  ArrayAttr selectors = op.selectors();

  for (Attribute selectorAttr : selectors) {
    std::string selector =
        selectorAttr.dyn_cast_or_null<StringAttr>().getValue().str();
    if (!supportedSelectors.contains(selector))
      return op.emitError("\"" + selector + "\" is not a supported selector.");
    if (rank == 1 && !supportedSelectorsComparingValues.contains(selector))
      return op.emitError("Selector '" + selector +
                          "' not allowed for vectors.");

    if (supportedSelectorsNeedingThunk.contains(selector))
      selectorsNeedingThunk.push_back(selector);
  }

  OperandRange thunks = op.thunks();

  if (thunks.size() != selectorsNeedingThunk.size()) {
    if (selectorsNeedingThunk.size() == 0) {
      return op.emitError()
             << "No selectors need thunks, but "
             << std::to_string(thunks.size()) << " thunks were given.";
    } else {
      return op.emitError()
             << "Some selectors ("
             << std::accumulate(++selectorsNeedingThunk.begin(),
                                selectorsNeedingThunk.end(),
                                "\"" + selectorsNeedingThunk[0] + "\"",
                                [](const std::string &a, std::string b) {
                                  return a + ", \"" + b + "\"";
                                })
             << ") need thunks, but " << std::to_string(thunks.size())
             << " thunks were given.";
    }
  }

  for (auto indexed_pair :
       llvm::enumerate(llvm::zip(selectorsNeedingThunk, thunks))) {
    std::tuple<std::string, Value> pair = indexed_pair.value();
    std::string selector = std::get<0>(pair);
    if (supportedSelectorsComparingValues.contains(selector)) {
      Value thunk = std::get<1>(pair);
      Type thunkType = thunk.getType();
      if (thunkType != inputType.getElementType()) {
        return op.emitError() << "Operand #" << indexed_pair.index() + 1
                              << " is associated with the selector "
                              << "\"" << selector << "\""
                              << ", but has a different type than the input "
                                 "tensor's element type.";
      }
    }
  }

  return success();
}

static LogicalResult verify(ConvertLayoutOp op) {
  RankedTensorType inputType = op.input().getType().cast<RankedTensorType>();
  RankedTensorType resultType =
      op.getResult().getType().cast<RankedTensorType>();

  llvm::Optional<std::string> errMsg;
  errMsg = checkMatrixEncoding(inputType, EITHER);
  if (errMsg)
    return op.emitError("operand " + errMsg.getValue());

  errMsg = checkMatrixEncoding(resultType, EITHER);
  if (errMsg)
    return op.emitError("result " + errMsg.getValue());

  // TODO intelligently handle arbitrarily shaped tensors, i.e. tensors with
  // shapes using "?"

  if (inputType.getElementType() != resultType.getElementType())
    return op.emitError(
        "Input and output tensors must have same element type.");

  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> resultShape = resultType.getShape();

  if (inputShape != resultShape)
    return op.emitError("Input and output shapes are expected to be the same.");

  errMsg = checkBitWidthMatch(inputType, resultType);
  if (errMsg)
    return op.emitError("Input and output " + errMsg.getValue());

  return success();
}

static LogicalResult verify(TransposeOp op) {
  RankedTensorType inputType = op.input().getType().cast<RankedTensorType>();
  RankedTensorType resultType =
      op.getResult().getType().cast<RankedTensorType>();

  llvm::Optional<std::string> errMsg;
  errMsg = checkMatrixEncoding(inputType, EITHER);
  if (errMsg)
    return op.emitError("operand " + errMsg.getValue());

  errMsg = checkMatrixEncoding(resultType, EITHER);
  if (errMsg)
    return op.emitError("result " + errMsg.getValue());

  if (inputType.getElementType() != resultType.getElementType())
    return op.emitError(
        "Input and output tensors have different element types.");

  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> resultShape = resultType.getShape();

  if (inputShape[0] != resultShape[1] || inputShape[1] != resultShape[0])
    return op.emitError("Input and output shapes are expected to be swapped.");

  errMsg = checkBitWidthMatch(inputType, resultType);
  if (errMsg)
    return op.emitError("Input and output " + errMsg.getValue());

  return success();
}

static LogicalResult verify(MatrixSelectRandomOp op) {
  RankedTensorType inputType = op.input().getType().cast<RankedTensorType>();
  IntegerType nType = op.n().getType().cast<IntegerType>();
  RankedTensorType resultType =
      op.getResult().getType().cast<RankedTensorType>();

  llvm::Optional<std::string> errMsg = checkMatrixEncoding(inputType, CSR);
  if (errMsg)
    return op.emitError("input " + errMsg.getValue());

  if (inputType != resultType)
    return op.emitError("Input and output tensor type must be identical.");

  mlir::sparse_tensor::SparseTensorEncodingAttr inputSparseEncoding =
      mlir::sparse_tensor::getSparseTensorEncoding(inputType);

  if (nType.getWidth() != inputSparseEncoding.getIndexBitWidth())
    return op.emitError(
        "n must match bit width of input sparse tensor index type");

  return success();
}

static LogicalResult verify(PrintOp op) {
  for (OpOperand &opOperand : op->getOpOperands()) {
    Type operandType = opOperand.get().getType();

    llvm::Optional<std::string> errorString =
        llvm::TypeSwitch<Type, llvm::Optional<std::string>>(operandType)
            .Case<IndexType>(
                [&](IndexType type) -> llvm::Optional<std::string> {
                  return llvm::None;
                })
            .Case<IntegerType>(
                [&](IntegerType type) -> llvm::Optional<std::string> {
                  unsigned bitWidth = type.getWidth();
                  static const std::unordered_set<int> allowedBitWidths = {
                      1, 8, 16, 32, 64};
                  if (allowedBitWidths.find(bitWidth) ==
                      allowedBitWidths.end()) {
                    std::string errorMessageString =
                        "Cannot print integer with bit width of ";
                    errorMessageString += bitWidth;
                    errorMessageString += ".";
                    return errorMessageString;
                  }
                  return llvm::None;
                })
            .Case<FloatType>([&](FloatType type)
                                 -> llvm::Optional<std::string> {
              unsigned bitWidth = type.getWidth();
              static const std::unordered_set<int> allowedBitWidths = {32, 64};
              if (allowedBitWidths.find(bitWidth) == allowedBitWidths.end()) {
                std::string errorMessageString =
                    "Cannot print float with bit width of ";
                errorMessageString += bitWidth;
                errorMessageString += ".";
                return errorMessageString;
              }
              return llvm::None;
            })
            .Default([&](Type type) -> llvm::Optional<std::string> {
              std::string errorMessageString = "Printing for the type ";
              llvm::raw_string_ostream stream(errorMessageString);
              type.print(stream);
              stream << " is not yet supported.";
              stream.flush();
              return errorMessageString;
            });

    if (errorString)
      return op.emitError(errorString.getValue());
  }

  return success();
}

#define GET_OP_CLASSES
#include "GraphBLAS/GraphBLASOps.cpp.inc"
