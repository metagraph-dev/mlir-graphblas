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

// TODO currently unused
// static llvm::Optional<std::string> checkUnaryOp(StringRef unaryOp) {
//   if (!unary1.contains(unaryOp) && !unary3.contains(unaryOp))
//     return "\"" + unaryOp.str() + "\" is not a supported unary operator.";
//   else
//     return llvm::None;
// }

static llvm::Optional<std::string> checkBinaryOp(StringRef binaryOp) {
  if (!binary2.contains(binaryOp) && !binary4.contains(binaryOp) &&
      !binary5.contains(binaryOp))
    return "\"" + binaryOp.str() + "\" is not a supported binary operator.";
  else
    return llvm::None;
}

static llvm::Optional<std::string> checkMonoidOp(StringRef monoidOp) {
  if (!monoid2.contains(monoidOp))
    return "\"" + monoidOp.str() + "\" is not a supported monoid.";
  else
    return llvm::None;
}

static llvm::Optional<std::string> checkSemiringOp(StringRef semiringOp) {
  auto semiringParts = semiringOp.split('_');
  auto monoidCheck = checkMonoidOp(semiringParts.first);
  if (monoidCheck != llvm::None)
    return monoidCheck;

  auto binaryCheck = checkBinaryOp(semiringParts.second);
  if (binaryCheck != llvm::None)
    return binaryCheck;

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
static LogicalResult verifyApplyArgs(T op, Type inputType) {
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

  Type inputType = input.getType();
  LogicalResult argResult = verifyApplyArgs(op, inputType);

  if (argResult.failed())
    return argResult;

  std::string applyOperator = op.apply_operator().str();
  if (!supportedForApply.contains(applyOperator))
    return op.emitError("\"" + applyOperator +
                        "\" is not a supported operator.");

  if (binary2.contains(applyOperator) || binary4.contains(applyOperator)) {

    if (!thunk)
      return op.emitError("\"" + applyOperator +
                          "\""
                          " requires a thunk.");

    RankedTensorType inputTensorType = inputType.dyn_cast<RankedTensorType>();

    Type thunkType = thunk.getType();

    // For binary2 ops, thunk type must match input type
    // for binary4 ops, the thunk may be used to compare against the index
    // values, so no check is made
    if (inputTensorType.getElementType() != thunkType &&
        binary2.contains(applyOperator))
      return op.emitError(
          "Element type of input tensor does not match type of thunk.");

  } else if (unary1.contains(applyOperator) || unary3.contains(applyOperator)) {

    if (thunk)
      return op.emitError("\"" + applyOperator +
                          "\""
                          " is a unary operator, but was given a thunk.");
  }

  return success();
}

static LogicalResult verify(ApplyGenericOp op) {
  LogicalResult argResult = verifyApplyArgs(op, op.input().getType());

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
    errMsg = checkMatrixEncoding(aType, EITHER);
    if (errMsg)
      return op.emitError("1st operand " + errMsg.getValue());

    errMsg = checkMatrixEncoding(bType, EITHER);
    if (errMsg)
      return op.emitError("2nd operand " + errMsg.getValue());

    if (checkResultTensorType) {
      errMsg = checkMatrixEncoding(resultType, EITHER);
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
    errMsg = checkMatrixEncoding(aType, EITHER);
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

    errMsg = checkMatrixEncoding(bType, EITHER);
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
        errMsg = checkMatrixEncoding(maskType, EITHER);
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

  llvm::Optional<std::string> semiringError = checkSemiringOp(op.semiring());
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

  if (inputType.getElementType() != resultType.getElementType())
    return op.emitError("input and output types must match.");

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

template <class T>
static LogicalResult verifyEwise(T op, Value a, Value b, std::string aName,
                                 std::string bName, bool verifyType) {
  Type aOrigType = a.getType();
  Type bOrigType = b.getType();

  RankedTensorType aType = aOrigType.cast<RankedTensorType>();
  RankedTensorType bType = bOrigType.cast<RankedTensorType>();

  if (failed(verifySameShape(aType, bType)))
    return op.emitError("\"" + aName + "\" and \"" + bName +
                        "\" must have identical shapes.");

  int64_t aRank = aType.getRank();

  llvm::Optional<std::string> errMsg;
  if (aRank == 1) {
    errMsg = checkVectorEncoding(aType);
    if (errMsg)
      return op.emitError("\"" + aName + "\" " + errMsg.getValue());
    errMsg = checkVectorEncoding(bType);
    if (errMsg)
      return op.emitError("\"" + bName + "\" " + errMsg.getValue());
  } else if (aRank == 2) {
    errMsg = checkMatrixEncoding(aType, EITHER);
    if (errMsg)
      return op.emitError("\"" + aName + "\" " + errMsg.getValue());
    // B must be ordered the same as A
    if (hasRowOrdering(aType))
      errMsg = checkMatrixEncoding(bType, CSR);
    else
      errMsg = checkMatrixEncoding(bType, CSC);
    if (errMsg)
      return op.emitError("\"" + bName + "\" " + errMsg.getValue());
  }
  if (verifyType && aType != bType)
    return op.emitError("\"" + aName + "\" and \"" + bName +
                        "\" must have identical types.");

  return success();
}

static LogicalResult verify(UpdateOp op) {
  llvm::Optional<llvm::StringRef> accumulateOperator = op.accumulate_operator();
  if (accumulateOperator) {
    if (!supportedForUpdate.contains(accumulateOperator->str()))
      return op.emitError("\"" + accumulateOperator->str() +
                          "\" is not a supported accumulate operator.");
  }

  if (failed(verifyEwise<UpdateOp>(op, op.input(), op.output(), "input",
                                   "output",
                                   /* verifyType */ true)))
    return failure();

  Value mask = op.mask();
  if (mask &&
      failed(verifyEwise<UpdateOp>(op, op.output(), mask, "output", "mask",
                                   /* verifyType */ false)))
    return failure();

  return success();
}

static LogicalResult verify(UpdateGenericOp op) {
  RegionRange extensions = op.extensions();
  if (extensions.size() < 1)
    return op.emitError("Must have at least 1 region: accumulate.");

  if (failed(verifyEwise<UpdateGenericOp>(op, op.input(), op.output(), "input",
                                          "output",
                                          /* verifyType */ true)))
    return failure();

  Value mask = op.mask();
  if (mask && failed(verifyEwise<UpdateGenericOp>(op, op.output(), mask,
                                                  "output", "mask",
                                                  /* verifyType */ false)))
    return failure();

  return success();
}

static LogicalResult verify(UnionOp op) {
  llvm::Optional<llvm::StringRef> unionOperator = op.union_operator();
  if (unionOperator) {
    if (!supportedForUnion.contains(unionOperator->str()))
      return op.emitError("\"" + unionOperator->str() +
                          "\" is not a supported union operator.");
  }

  if (failed(verifyEwise<UnionOp>(op, op.a(), op.b(), "a", "b",
                                  /* verifyType */ true)))
    return failure();
  if (failed(verifyEwise<UnionOp>(op, op.a(), op.getResult(), "input", "output",
                                  /* verifyType */ true)))
    return failure();

  return success();
}

static LogicalResult verify(UnionGenericOp op) {
  RegionRange extensions = op.extensions();
  if (extensions.size() < 1)
    return op.emitError("Must have at least 1 region: mult.");

  if (failed(verifyEwise<UnionGenericOp>(op, op.a(), op.b(), "a", "b",
                                         /* verifyType */ true)))
    return failure();
  if (failed(verifyEwise<UnionGenericOp>(op, op.a(), op.getResult(), "input",
                                         "output",
                                         /* verifyType */ true)))
    return failure();

  return success();
}

static LogicalResult verify(IntersectOp op) {
  llvm::Optional<llvm::StringRef> intersectOperator = op.intersect_operator();
  if (intersectOperator) {
    if (!supportedForIntersect.contains(intersectOperator->str()))
      return op.emitError("\"" + intersectOperator->str() +
                          "\" is not a supported intersect operator.");
  }

  if (failed(verifyEwise<IntersectOp>(op, op.a(), op.b(), "a", "b",
                                      /* verifyType */ true)))
    return failure();
  if (failed(verifyEwise<IntersectOp>(op, op.a(), op.getResult(), "input",
                                      "output",
                                      /* verifyType */ false)))
    return failure();

  return success();
}

static LogicalResult verify(IntersectGenericOp op) {
  RegionRange extensions = op.extensions();
  if (extensions.size() < 1)
    return op.emitError("Must have at least 1 region: mult.");

  if (failed(verifyEwise<IntersectGenericOp>(op, op.a(), op.b(), "a", "b",
                                             /* verifyType */ true)))
    return failure();
  if (failed(verifyEwise<IntersectGenericOp>(op, op.a(), op.getResult(),
                                             "input", "output",
                                             /* verifyType */ false)))
    return failure();

  return success();
}

static LogicalResult verify(EqualOp op) {
  return verifyEwise<EqualOp>(op, op.a(), op.b(), "a", "b",
                              /* verifyType */ true);
}

template <class T>
static LogicalResult verifyReduceToVectorArgs(T op) {
  Type inputType = op.input().getType();
  RankedTensorType inputTensorType = inputType.cast<RankedTensorType>();

  llvm::Optional<std::string> errMsg;
  errMsg = checkMatrixEncoding(inputTensorType, EITHER);
  if (errMsg)
    return op.emitError("operand " + errMsg.getValue());

  Type resultType = op.getResult().getType();
  RankedTensorType resultTensorType = resultType.cast<RankedTensorType>();

  errMsg = checkVectorEncoding(resultTensorType);
  if (errMsg)
    return op.emitError("result " + errMsg.getValue());

  ArrayRef<int64_t> inputShape = inputTensorType.getShape();

  int axis = op.axis();
  int expectedResultLength;
  if (axis == 0) {
    expectedResultLength = inputShape[1];
  } else if (axis == 1) {
    expectedResultLength = inputShape[0];
  } else {
    return op.emitError("The axis attribute is expected to be 0 or 1.");
  }

  ArrayRef<int64_t> resultShape = resultTensorType.getShape();
  if (resultShape[0] != expectedResultLength) {
    return op.emitError("Operand and output shapes are incompatible.");
  }

  return success();
}

static LogicalResult verify(ReduceToVectorOp op) {
  std::string aggregator = op.aggregator().str();
  if (!supportedForReduce.contains(aggregator))
    return op.emitError("\"" + aggregator +
                        "\" is not a supported aggregator.");

  LogicalResult argResult = verifyReduceToVectorArgs(op);

  if (argResult.failed())
    return argResult;

  Type resultType =
      op.getResult().getType().cast<RankedTensorType>().getElementType();

  StringSet<> i64Aggs{"argmax", "argmin", "count"};
  if (i64Aggs.contains(aggregator)) {
    if (!resultType.isa<IntegerType>() ||
        resultType.cast<IntegerType>().getWidth() != 64)
      return op.emitError(
          "\"" + aggregator +
          "\" requires the output vector to have i64 elements.");
  } else {
    Type inputType =
        op.input().getType().cast<RankedTensorType>().getElementType();
    if (resultType != inputType)
      return op.emitError("Operand and output types are incompatible.");
  }

  return success();
}

static LogicalResult verify(ReduceToVectorGenericOp op) {
  LogicalResult argResult = verifyReduceToVectorArgs(op);

  if (argResult.failed())
    return argResult;

  RegionRange extensions = op.extensions();
  if (extensions.size() < 2) {
    return op.emitError("Must have at least 2 regions: agg_identity, agg.");
  }

  // Enforce reasonable iteration direction for axis
  bool isCSR = hasRowOrdering(op.input().getType());
  int axis = op.axis();
  if (axis == 0 && isCSR)
    return op.emitError("Reducing with axis=0 requires CSC matrix.");
  if (axis == 1 && !isCSR)
    return op.emitError("Reducing with axis=1 requires CSR matrix.");

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
  if (!supportedForReduce.contains(aggregator))
    return op.emitError("\"" + aggregator +
                        "\" is not a supported aggregator.");

  StringSet<> i64Aggs{"argmax", "argmin", "count"};
  StringSet<> vectorOnlyAggs{"argmax", "argmin", "first", "last"};

  Type operandOrigType = op.input().getType();
  RankedTensorType operandType = operandOrigType.cast<RankedTensorType>();
  Type resultType = op.getResult().getType();

  if (i64Aggs.contains(aggregator)) {
    if (!resultType.isa<IntegerType>() ||
        resultType.cast<IntegerType>().getWidth() != 64)
      return op.emitError("\"" + aggregator +
                          "\" requires the output type to be i64.");
  } else {
    if (resultType != operandType.getElementType())
      return op.emitError("Operand and output types are incompatible.");
  }

  if (vectorOnlyAggs.contains(aggregator) && operandType.getRank() != 1)
    return op.emitError("\"" + aggregator + "\" only supported for vectors.");

  return success();
}

static LogicalResult verify(ReduceToScalarGenericOp op) {
  LogicalResult argResult = verifyReduceToScalarArgs(op);

  if (argResult.failed())
    return argResult;

  RegionRange extensions = op.extensions();
  if (extensions.size() < 2) {
    return op.emitError("Must have at least 2 regions: agg_identity, agg.");
  }

  return success();
}

static LogicalResult verify(graphblas::SelectOp op) {
  Type inputType = op.input().getType();
  LogicalResult argResult = verifyApplyArgs(op, inputType);

  if (argResult.failed())
    return argResult;

  RankedTensorType inputTensorType = inputType.cast<RankedTensorType>();
  RankedTensorType resultTensorType =
      op.getResult().getType().cast<RankedTensorType>();

  if (inputTensorType != resultTensorType)
    return op.emitError("result type must match input type.");

  unsigned rank = inputTensorType.getRank();

  std::string selector = op.selector().str();
  std::vector<std::string> selectorsNeedingThunk;
  ValueRange thunks = op.thunks();

  if (!supportedForSelect.contains(selector))
    return op.emitError("\"" + selector + "\" is not a supported selector.");
  if (rank == 1 && (unary3.contains(selector) || binary4.contains(selector)))
    return op.emitError("Selector '" + selector + "' not allowed for vectors.");

  if (thunks.size() <= 0) {
    if (binary2.contains(selector) || binary4.contains(selector))
      return op.emitError("Selector '" + selector + "' requires a thunk.");
  } else if (thunks.size() > 2) {
    return op.emitError("Too many thunk values provided.");
  } else {
    if (unary3.contains(selector))
      return op.emitError("Selector '" + selector + "' cannot take a thunk.");

    Value thunk = thunks[0];
    Type thunkType = thunk.getType();

    if (selector == "probability") {
      // Ensure thunk type is f64
      if (!thunkType.isa<FloatType>() ||
          thunkType.cast<FloatType>().getWidth() != 64)
        return op.emitError("Select 'probability' requires f64 thunk.");
      if (thunks.size() != 2)
        return op.emitError("Selector 'probability' requires a RNG context");
    } else {
      // All other selectors
      if (thunks.size() > 1)
        return op.emitError("Too many thunks provided for selector '" +
                            selector + "'");
      // For binary2 ops, thunk type must match input type
      // for binary4 ops, the thunk may be used to compare against the index
      // values, so no check is made
      if (binary2.contains(selector) &&
          thunkType != inputTensorType.getElementType())
        return op.emitError("Thunk type must match operand type.");
    }
  }

  return success();
}

static LogicalResult verify(graphblas::SelectGenericOp op) {
  Value input = op.input();
  Type inputType = input.getType();
  LogicalResult argResult = verifyApplyArgs(op, inputType);

  if (argResult.failed())
    return argResult;

  RegionRange extensions = op.extensions();
  if (extensions.size() < 1) {
    return op.emitError("Must have at least 1 region: select_out.");
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

static LogicalResult verify(CastOp op) {
  RankedTensorType inputType = op.input().getType().cast<RankedTensorType>();
  RankedTensorType resultType =
      op.getResult().getType().cast<RankedTensorType>();

  unsigned rank = inputType.getRank();
  if (resultType.getRank() != rank)
    return op.emitError("Input and output ranks must match.");

  ArrayRef<int64_t> shape = inputType.getShape();
  if (resultType.getShape() != shape)
    return op.emitError("Input and output shapes must match.");

  llvm::Optional<std::string> errMsg;
  if (rank == 2) {
    errMsg = checkMatrixEncoding(inputType, EITHER);
    if (errMsg)
      return op.emitError("operand " + errMsg.getValue());

    // Result must be ordered the same as input
    errMsg =
        checkMatrixEncoding(resultType, hasRowOrdering(inputType) ? CSR : CSC);
    if (errMsg)
      return op.emitError("result " + errMsg.getValue());
  } else {
    errMsg = checkVectorEncoding(inputType);
    if (errMsg)
      return op.emitError("operand " + errMsg.getValue());

    errMsg = checkVectorEncoding(resultType);
    if (errMsg)
      return op.emitError("operand " + errMsg.getValue());
  }

  // TODO: remove this check once we support bit width changes
  errMsg = checkBitWidthMatch(inputType, resultType);
  if (errMsg)
    return op.emitError(
        "Changing bit width is not yet supported. Input and output " +
        errMsg.getValue());

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
            .Case<RankedTensorType>(
                [&](RankedTensorType type) -> llvm::Optional<std::string> {
                  sparse_tensor::SparseTensorEncodingAttr sparseEncoding =
                      sparse_tensor::getSparseTensorEncoding(type);
                  bool inputIsDense = !sparseEncoding;
                  if (inputIsDense)
                    return llvm::None;

                  unsigned rank = type.getRank();
                  if (rank == 1) {
                    llvm::ArrayRef<
                        sparse_tensor::SparseTensorEncodingAttr::DimLevelType>
                        dlt = sparseEncoding.getDimLevelType();
                    if (dlt[0] != sparse_tensor::SparseTensorEncodingAttr::
                                      DimLevelType::Compressed)
                      return std::string("Vectors must be dense or sparse.");
                    return llvm::None;
                  } else if (rank == 2) {
                    return checkMatrixEncoding(type, EITHER);
                  } else {
                    return std::string(
                        "Can only print sparse tensors with rank 1 or 2.");
                  }
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

static LogicalResult verify(PrintTensorOp op) {
  Type operandOrigType = op.input().getType();
  RankedTensorType operandType = operandOrigType.cast<RankedTensorType>();
  int64_t rank = operandType.getRank();

  llvm::Optional<std::string> errMsg;
  if (rank == 1) {
    errMsg = checkVectorEncoding(operandType);
    if (errMsg)
      return op.emitError("input " + errMsg.getValue());
  } else {
    errMsg = checkMatrixEncoding(operandType, EITHER);
    if (errMsg)
      return op.emitError("input " + errMsg.getValue());
  }

  return success();
}

static LogicalResult verify(FromCoordinatesOp op) {
  Type indicesType = op.indices().getType();
  Type valuesType = op.values().getType();
  auto encIdx = sparse_tensor::getSparseTensorEncoding(indicesType);
  auto encVal = sparse_tensor::getSparseTensorEncoding(valuesType);

  if (encIdx)
    return op.emitError("Indices must be a dense tensor.");
  if (encVal)
    return op.emitError("Values must be a dense tensor.");

  ValueRange sizes = op.sizes();
  Type resultType = op.getResult().getType();
  RankedTensorType resultTensorType = resultType.cast<RankedTensorType>();
  size_t rank = resultTensorType.getRank();

  if (sizes.size() != rank)
    return op.emitError("Length of sizes must match result.");

  llvm::Optional<std::string> errMsg;
  if (rank == 1) {
    errMsg = checkVectorEncoding(resultTensorType);
    if (errMsg)
      return op.emitError("result " + errMsg.getValue());
  } else {
    errMsg = checkMatrixEncoding(resultTensorType, CSR);
    if (errMsg)
      return op.emitError("result " + errMsg.getValue());
  }

  Type valueType = valuesType.cast<RankedTensorType>().getElementType();
  Type rvalType = resultTensorType.getElementType();
  if (rvalType != valueType)
    return op.emitError("Value type must match return type");

  return success();
}

static LogicalResult verify(ToCoordinatesOp op) {
  Type indicesType = op.getResult(0).getType();
  Type valuesType = op.getResult(1).getType();
  auto encIdx = sparse_tensor::getSparseTensorEncoding(indicesType);
  auto encVal = sparse_tensor::getSparseTensorEncoding(valuesType);

  if (encIdx)
    return op.emitError("Returned indices must be a dense tensor.");
  if (encVal)
    return op.emitError("Returned values must be a dense tensor.");

  Type inputType = op.input().getType();
  RankedTensorType inputTensorType = inputType.cast<RankedTensorType>();
  int64_t rank = inputTensorType.getRank();

  llvm::Optional<std::string> errMsg;
  if (rank == 1) {
    errMsg = checkVectorEncoding(inputTensorType);
    if (errMsg)
      return op.emitError("input " + errMsg.getValue());
  } else {
    errMsg = checkMatrixEncoding(inputTensorType, CSR);
    if (errMsg)
      return op.emitError("input " + errMsg.getValue());
  }

  Type valueType = valuesType.cast<RankedTensorType>().getElementType();
  Type ivalType = inputTensorType.getElementType();
  if (ivalType != valueType)
    return op.emitError("Input type must match return value type");

  return success();
}

#define GET_OP_CLASSES
#include "GraphBLAS/GraphBLASOps.cpp.inc"
