//===- GraphBLASOps.cpp - GraphBLAS dialect ops ---------------*- C++ -*-===//
//
// TODO add documentation
//
//===--------------------------------------------------------------------===//

#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "GraphBLAS/GraphBLASOps.h"
#include "GraphBLAS/GraphBLASDialect.h"
#include "GraphBLAS/GraphBLASUtils.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/None.h"

#include "GraphBLAS/GraphBLASOpsEnums.cpp.inc"
#include "GraphBLAS/GraphBLASOpsDialect.cpp.inc"
#include "GraphBLAS/GraphBLASUtils.h"

using namespace mlir;
using namespace mlir::graphblas;

//===--------------------------------------------------------------------===//
// Helpers
//===--------------------------------------------------------------------===//

enum CompressionType { CSR, CSC, EITHER };

static llvm::Optional<std::string> checkCompressedMatrix(
        Type inputType,
        int inputIndex,
        CompressionType compressionType
    ) {
  /*
     Negative values for inputIndex indicate that the input type is the return type.
     Otherwise, inputIndex indicates which arg inputType corresponds to.

     Returns llvm::None if the given tensor is valid.
     Returns a string explaining the problem otherwise.
   */

  std::string inputName = inputIndex < 0 ? "Return value" : "Operand #"+std::to_string(inputIndex);

  mlir::sparse_tensor::SparseTensorEncodingAttr sparseEncoding =
    mlir::sparse_tensor::getSparseTensorEncoding(inputType);
  if (!sparseEncoding)
    return inputName+" must be a sparse tensor.";

  RankedTensorType inputTensorType = inputType.dyn_cast<RankedTensorType>();
  if (inputTensorType.getRank() != 2)
    return inputName+" must have rank 2.";

  ArrayRef<mlir::sparse_tensor::SparseTensorEncodingAttr::DimLevelType> compression =
    sparseEncoding.getDimLevelType();
  if (compression[0] != mlir::sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Dense ||
      compression[1] != mlir::sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Compressed)
    return inputName+" must have CSR or CSC compression, i.e. must have "
      "dimLevelType = [ \"dense\", \"compressed\" ] in the sparse encoding.";

  // Even if (compressionType == EITHER), we still need to check that the
  // dim ordering is present, i.e. not CSX
  AffineMap dimOrdering = sparseEncoding.getDimOrdering();
  unsigned dimOrdering0 = dimOrdering.getDimPosition(0); 
  unsigned dimOrdering1 = dimOrdering.getDimPosition(1);
  
  if (compressionType != EITHER) {

    assert(compressionType == CSR || compressionType == CSC);
    
    if (compressionType == CSR) {
      if (dimOrdering0 != 0 || dimOrdering1 != 1)
        return inputName+" must have CSR compression.";
    } else if (compressionType == CSC) {
      if (dimOrdering0 != 1 || dimOrdering1 != 0)
        return inputName+" must have CSC compression.";
    }
  }

  return llvm::None;
}

static llvm::Optional<std::string> checkCompressedVector(
        Type inputType,
        int inputIndex
    ) {
  
  std::string inputName = inputIndex < 0 ? "Return value" : "Operand #"+std::to_string(inputIndex);
  
  mlir::sparse_tensor::SparseTensorEncodingAttr sparseEncoding =
    mlir::sparse_tensor::getSparseTensorEncoding(inputType);
  if (!sparseEncoding)
    return inputName+" must be a sparse tensor.";
  
  RankedTensorType inputTensorType = inputType.dyn_cast<RankedTensorType>();
  if (inputTensorType.getRank() != 1)
    return inputName+" must have rank 1.";
  
  ArrayRef<mlir::sparse_tensor::SparseTensorEncodingAttr::DimLevelType> compression =
    sparseEncoding.getDimLevelType();
  if (compression[0] != mlir::sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Compressed)
    return inputName+" must be sparse, i.e. must have "
      "dimLevelType = [ \"dense\", \"compressed\" ] in the sparse encoding.";
  
  return llvm::None;
}

//===--------------------------------------------------------------------===//
// GraphBLAS Ops Methods
//===--------------------------------------------------------------------===//

// TODO: remove me. Cannot do now as all the ops require a verifier (see:
// GraphBLAS_Op) a follow-up commit will address this issue.
static LogicalResult verify(SizeOp op) {
  return success();
}

void SizeOp::build(OpBuilder &builder, OperationState &result, Value tensor) {
  Type indexType = builder.getIndexType();
  build(builder, result, indexType, tensor);
}

// TODO: originally this verifier was calling 'checkCompressedMatrix'
// why do we need to check the sparse attribute? This ops should work
// also without passing the sparese encoding attribute.
static LogicalResult verify(NumRowsOp op) {
  return success();
}

void NumRowsOp::build(OpBuilder &builder, OperationState &result, Value tensor) {
  Type indexType = builder.getIndexType();
  build(builder, result, indexType, tensor);
}

// TODO: remove me. see SizeOp.
static LogicalResult verify(NumColsOp op) {
  return success();
}

void NumColsOp::build(OpBuilder &builder, OperationState &result, Value tensor) {
  Type indexType = builder.getIndexType();
  build(builder, result, indexType, tensor);
}

static LogicalResult verify(NumValsOp op) {
  Type inputType = op.input().getType();
  int64_t rank = getRank(inputType);

  // Require sparse matrices to be either CSR or CSC
  if (rank == 2) {
    llvm::Optional<std::string> inputCompressionErrorMessage = checkCompressedMatrix(inputType, 0, EITHER);
    if (inputCompressionErrorMessage)
      return op.emitError(inputCompressionErrorMessage.getValue());
  }

  return success();
}

void NumValsOp::build(OpBuilder &builder, OperationState &result, Value tensor) {
  Type indexType = builder.getIndexType();
  build(builder, result, indexType, tensor);
}

/// Utility function to check encoding attribute.
static LogicalResult hasSparseEncodingAttr(RankedTensorType t) {
  if (!sparse_tensor::getSparseTensorEncoding(t))
    return failure();
  return success();
}

/// Utility function to check if a 2d-tensor is in CSR or
/// CSC format.
static LogicalResult hasCSRorCSCEncoding(RankedTensorType t) {
  assert(t.getRank() == 2 && "expect a 2d ranked tensor");
  if (auto encoding = sparse_tensor::getSparseTensorEncoding(t)) {
    auto compression = encoding.getDimLevelType();
    if (((compression[0] ==
          sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Dense) &&
         (compression[1] ==
          sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Compressed)) ||
        ((compression[0] ==
          sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Compressed) &&
         (compression[1] ==
          sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Dense)))
      return success();
  }
  return failure();
}

static LogicalResult verify(DupOp op) {
  auto inputType = op.input().getType().cast<RankedTensorType>();
  auto rank = inputType.getRank();
  if (failed(hasSparseEncodingAttr(inputType)))
    return op.emitError("operand #0 must have sparse tensor attribute");
  if (rank == 2 && failed(hasCSRorCSCEncoding(inputType)))
    return op.emitError("operand #0 must be in CSR or in CSC compression");
  return success();
}

void DupOp::build(OpBuilder &builder, OperationState &result, Value tensor) {
  Type inputType = tensor.getType();
  build(builder, result, inputType, tensor);
}

template <class T>
static LogicalResult verifyMatrixApplyArgs(T op)
{
  Type inputType = op.input().getType();
  Type resultType = op.getResult().getType();

  llvm::Optional<std::string> inputCompressionErrorMessage = checkCompressedMatrix(inputType, 0, EITHER);
  if (inputCompressionErrorMessage)
    return op.emitError(inputCompressionErrorMessage.getValue());

  llvm::Optional<std::string> resultCompressionErrorMessage = checkCompressedMatrix(resultType, -1, EITHER);
  if (resultCompressionErrorMessage)
    return op.emitError(resultCompressionErrorMessage.getValue());
  
  RankedTensorType inputTensorType = inputType.dyn_cast<RankedTensorType>();
  RankedTensorType resultTensorType = resultType.dyn_cast<RankedTensorType>();

  ArrayRef<int64_t> inputShape = inputTensorType.getShape();
  ArrayRef<int64_t> resultShape = resultTensorType.getShape();

  // TODO intelligently handle arbitrarily shaped tensors, i.e. tensors with shapes using "?"
  if (inputShape[0] != resultShape[0] || inputShape[1] != resultShape[1])
    return op.emitError("Input shape does not match output shape.");

  return success();
}

static LogicalResult verify(MatrixApplyOp op) {
  LogicalResult argResult = verifyMatrixApplyArgs(op);

  if (argResult.failed())
    return argResult;

  Type inputType = op.input().getType();
  Type thunkType = op.thunk().getType();
  Type resultType = op.getResult().getType();

  RankedTensorType inputTensorType = inputType.dyn_cast<RankedTensorType>();
  RankedTensorType resultTensorType = resultType.dyn_cast<RankedTensorType>();

  if (inputTensorType.getElementType() != thunkType)
    return op.emitError("Element type of input tensor does not match type of thunk.");

  if (resultTensorType.getElementType() != thunkType)
    // TODO this is not always correct, e.g. matrix_apply_less_than(tensor<f64>, 2.3) -> tensor<i1>.
    return op.emitError("Element type of result tensor does not match type of thunk.");

  static const std::vector<std::string> supportedOperators{"min"};
  std::string applyOperator = op.apply_operator().str();
  bool operatorSupported = std::find(supportedOperators.begin(), supportedOperators.end(), applyOperator)
    != supportedOperators.end();
  if (!operatorSupported)
    return op.emitError("\""+applyOperator+"\" is not a supported operator.");

  return success();
}

static LogicalResult verify(MatrixApplyGenericOp op)
{
  LogicalResult argResult = verifyMatrixApplyArgs(op);

  if (argResult.failed())
    return argResult;

  RegionRange extensions = op.extensions();
  if (extensions.size() < 1)
  {
    return op.emitError("Must have at least 1 region: transform_out.");
  }

  return success();
}


template <class T>
static LogicalResult verifyMatrixMultiplyArgs(T op, bool checkResultTensorType)
{
  Type aType = op.a().getType();
  Type bType = op.b().getType();
  Type resultType = op.getResult().getType();

  int64_t aRank = getRank(aType);
  int64_t bRank = getRank(bType);

  if (aRank < 1 || aRank > 2)
    return op.emitError("First argument must be a sparse vector or sparse matrix.");
  if (bRank < 1 || bRank > 2)
    return op.emitError("Second argument must be a sparse vector or sparse matrix.");

  RankedTensorType aTensorType = aType.dyn_cast<RankedTensorType>();
  RankedTensorType bTensorType = bType.dyn_cast<RankedTensorType>();

  ArrayRef<int64_t> aShape = aTensorType.getShape();
  ArrayRef<int64_t> bShape = bTensorType.getShape();

  int64_t resultRank = 0;
  ArrayRef<int64_t> resultShape;
  Type resultElementType = resultType;
  RankedTensorType resultTensorType;

  // Vector-vector result is a scalar; Otherwise, get the tensor properties of the result
  if (checkResultTensorType)
  {
    if (aRank == 2 || bRank == 2)
    {
      resultTensorType = resultType.dyn_cast<RankedTensorType>();
      resultShape = resultTensorType.getShape();
      resultRank = getRank(resultType);
      resultElementType = resultTensorType.getElementType();
    }
  }

  if (aTensorType.getElementType() != bTensorType.getElementType())
    return op.emitError("Operand element types must be identical.");

  llvm::Optional<std::string> errMsg;
  if (aRank == 2 && bRank == 2)
  {
    // Matrix-Matrix
    errMsg = checkCompressedMatrix(aType, 0, CSR);
    if (errMsg)
      return op.emitError(errMsg.getValue());

    errMsg = checkCompressedMatrix(bType, 1, CSC);
    if (errMsg)
      return op.emitError(errMsg.getValue());

    if (checkResultTensorType) {
      errMsg = checkCompressedMatrix(resultType, -1, CSR);
      if (errMsg)
        return op.emitError(errMsg.getValue());
      if (resultShape[0] != aShape[0] || resultShape[1] != bShape[1])
        return op.emitError("Operand shapes incompatible with output shape.");
      if (aTensorType.getElementType() != resultElementType)
        return op.emitError("Result element type differs from the input element types.");
    }

    // TODO intelligently handle arbitrarily shaped tensors, i.e. tensors with shapes using "?"
    if (aShape[1] != bShape[0])
      return op.emitError("Operand shapes are incompatible.");
  }
  else if (aRank == 2 && bRank == 1)
  {
    // Matrix-Vector
    errMsg = checkCompressedMatrix(aType, 0, CSR);
    if (errMsg)
      return op.emitError(errMsg.getValue());

    errMsg = checkCompressedVector(bType, 1);
    if (errMsg)
      return op.emitError(errMsg.getValue());

    if (checkResultTensorType) {
      errMsg = checkCompressedVector(resultType, -1);
      if (errMsg)
        return op.emitError(errMsg.getValue());

      if (resultShape[0] != aShape[0])
        return op.emitError("Operand shapes incompatible with output shape.");
      if (aTensorType.getElementType() != resultElementType)
        return op.emitError("Result element type differs from the input element types.");
    }
    if (aShape[1] != bShape[0])
      return op.emitError("Operand shapes are incompatible.");
  }
  else if (aRank == 1 && bRank == 2)
  {
    // Vector-Matrix
    errMsg = checkCompressedVector(aType, 0);
    if (errMsg)
      return op.emitError(errMsg.getValue());

    errMsg = checkCompressedMatrix(bType, 1, CSC);
    if (errMsg)
      return op.emitError(errMsg.getValue());

    if (aShape[0] != bShape[0])
      return op.emitError("Operand shapes are incompatible.");

    if (checkResultTensorType)
    {
      errMsg = checkCompressedVector(resultType, -1);
      if (errMsg)
        return op.emitError(errMsg.getValue());
      if (resultShape[0] != bShape[1])
        return op.emitError("Operand shapes incompatible with output shape.");
      if (aTensorType.getElementType() != resultElementType)
        return op.emitError("Result element type differs from the input element types.");
    }
  }
  else
  {
    // Vector-Vector
    errMsg = checkCompressedVector(aType, 0);
    if (errMsg)
      return op.emitError(errMsg.getValue());

    errMsg = checkCompressedVector(bType, 1);
    if (errMsg)
      return op.emitError(errMsg.getValue());

    if (aShape[0] != bShape[0])
      return op.emitError("Operand shapes are incompatible.");
    if (aTensorType.getElementType() != resultElementType)
      return op.emitError("Result element type differs from the input element types.");
  }

  Value mask = op.mask();
  if (mask)
  {
    Type maskType = mask.getType();

    if (checkResultTensorType) {
      if (resultRank == 2)
      {
        errMsg = checkCompressedMatrix(maskType, 2, CSR);
        if (errMsg)
          return op.emitError(errMsg.getValue());

        RankedTensorType maskTensorType = maskType.dyn_cast<RankedTensorType>();
        ArrayRef<int64_t> maskShape = maskTensorType.getShape();
        if (resultShape[0] != maskShape[0] || resultShape[1] != maskShape[1])
          return op.emitError("Mask shape must match shape of matrix multiply result.");
      }
      else if (resultRank == 1)
      {
        errMsg = checkCompressedVector(maskType, 2);
        if (errMsg)
          return op.emitError(errMsg.getValue());

        RankedTensorType maskTensorType = maskType.dyn_cast<RankedTensorType>();
        ArrayRef<int64_t> maskShape = maskTensorType.getShape();
        if (resultShape[0] != maskShape[0])
          return op.emitError("Mask shape must match shape of matrix multiply result.");
      }
      else {
        return op.emitError("Mask not allowed for vector times vector multiplication.");
      }
    }
  }

  return success();
}

static const std::vector<std::string> supportedSemirings{"plus_times", "plus_pair", "plus_plus", "min_plus"};

static LogicalResult verify(MatrixMultiplyOp op) {
  LogicalResult argResult = verifyMatrixMultiplyArgs(op, true);

  if (argResult.failed())
    return argResult;

  std::string semiring = op.semiring().str();
  bool semiringSupported = std::find(supportedSemirings.begin(), supportedSemirings.end(), semiring)
    != supportedSemirings.end();
  if (!semiringSupported)
    return op.emitError("\""+semiring+"\" is not a supported semiring.");

  Region &body = op.body();
  auto numBlocks = body.getBlocks().size();
  if (numBlocks > 0) {
    return op.emitError("graphblas.matrix_multiply should have no blocks.  Did you mean graphblas.matrix_multiply_generic?");
  }

  return success();
}

static LogicalResult verify(MatrixMultiplyGenericOp op)
{
  LogicalResult argResult = verifyMatrixMultiplyArgs(op, true);

  if (argResult.failed())
    return argResult;

  RegionRange extensions = op.extensions();
  if (extensions.size() < 3)
  {
    return op.emitError("Must have at least 3 regions: add_identity, add, mult.");
  }

  return success();
}

static LogicalResult verify(MatrixMultiplyReduceToScalarGenericOp op) {
  LogicalResult argResult = verifyMatrixMultiplyArgs(op, false /* no result tensor */);

  if (argResult.failed())
    return argResult;

  RegionRange extensions = op.extensions();
  if (extensions.size() < 4)
  {
    return op.emitError("Must have at least 4 regions: add_identity, add, mult, agg.");
  }

  return success();
}

static LogicalResult verify(VectorArgMinMaxOp op) {
  Type vecType = op.vec().getType();

  llvm::Optional<std::string> vecCompressionErrorMessage = checkCompressedVector(vecType, 0);
  if (vecCompressionErrorMessage)
    return op.emitError(vecCompressionErrorMessage.getValue());

  std::string  minmax = op.minmax().str();
  if (minmax != "min" && minmax != "max")
    return op.emitError("The minmax attribute is expected to be \"min\" or \"max\"; got \""+minmax+"\" instead.");
  
  return success();
}

static LogicalResult verify(VectorArgMinOp op) {
  Type vecType = op.vec().getType();

  llvm::Optional<std::string> vecCompressionErrorMessage = checkCompressedVector(vecType, 0);
  if (vecCompressionErrorMessage)
    return op.emitError(vecCompressionErrorMessage.getValue());

  return success();
}

static LogicalResult verify(VectorArgMaxOp op) {
  Type vecType = op.vec().getType();

  llvm::Optional<std::string> vecCompressionErrorMessage = checkCompressedVector(vecType, 0);
  if (vecCompressionErrorMessage)
    return op.emitError(vecCompressionErrorMessage.getValue());

  return success();
}

template <class T>
static LogicalResult verifyMatrixReduceToScalarArgs(T op)
{
  Type operandType = op.input().getType();

  llvm::Optional<std::string> compressionErrorMessage = checkCompressedMatrix(operandType, 0, EITHER);
  if (compressionErrorMessage)
    return op.emitError(compressionErrorMessage.getValue());

  Type resultType = op.getResult().getType();
  RankedTensorType operandTensorType = operandType.dyn_cast<RankedTensorType>();
  if (resultType != operandTensorType.getElementType())
    return op.emitError("Operand and output types are incompatible.");

  return success();
}

static const std::vector<std::string> supportedUpdateAccumulateOperators{"plus", "min"};

static LogicalResult verify(UpdateOp op) {
  Type iType = op.input().getType();
  Type oType = op.output().getType();
  Value mask = op.mask();

  int64_t iRank = getRank(iType);
  int64_t oRank = getRank(oType);
  int64_t mRank = -1;
  if (mask)
    mRank = getRank(mask.getType());

  if (iRank < 1 || iRank > 2)
    return op.emitError("Input argument must be a sparse vector or sparse matrix.");
  if (oRank < 1 || oRank > 2)
    return op.emitError("Output argument must be a sparse vector or sparse matrix.");
  if (mask && (mRank < 1 || mRank > 2))
    return op.emitError("Mask argument must be a sparse vector or sparse matrix.");

  if (failed(verifyCompatibleShape(iType, oType)))
    return op.emitError("Input and Output arguments must have compatible shapes.");
  if (mask and failed(verifyCompatibleShape(oType, mask.getType())))
    return op.emitError("Mask and Output arguments must have compatible shapes.");

  llvm::Optional<std::string> errMsg;
  if (iRank == 1) {
    errMsg = checkCompressedVector(iType, 0);
    if (errMsg)
      return op.emitError(errMsg.getValue());
    errMsg = checkCompressedVector(oType, 1);
    if (errMsg)
      return op.emitError(errMsg.getValue());
    if (mask) {
      errMsg = checkCompressedVector(mask.getType(), 2);
      if (errMsg)
        return op.emitError(errMsg.getValue());
    }
  } else if (iRank == 2) {
    errMsg = checkCompressedMatrix(iType, 0, EITHER);
    if (errMsg)
      return op.emitError(errMsg.getValue());
    // Determine output sparse encoding
    CompressionType sparseEncoding = CSR;
    if (typeIsCSC(iType)) {
      sparseEncoding = CSC;
    }
    errMsg = checkCompressedMatrix(oType, 1, sparseEncoding);
    if (errMsg)
      return op.emitError(errMsg.getValue());
    if (mask) {
      errMsg = checkCompressedMatrix(mask.getType(), 2, sparseEncoding);
      if (errMsg)
        return op.emitError(errMsg.getValue());
    }
  }

  if (iType != oType)
    return op.emitError("Arguments must have the same type.");

  llvm::Optional<llvm::StringRef> accumulateOperator = op.accumulate_operator();
  if (accumulateOperator) {
    bool operatorSupported = std::find(supportedUpdateAccumulateOperators.begin(),
                                       supportedUpdateAccumulateOperators.end(),
                                       accumulateOperator->str())
      != supportedUpdateAccumulateOperators.end();
    if (!operatorSupported)
      return op.emitError("\""+accumulateOperator->str()+"\" is not a supported accumulate operator.");
  }

  return success();
}

template <class T>
static LogicalResult verifyEwise(T op) {
  Type aType = op.a().getType();
  Type bType = op.b().getType();

  int64_t aRank = getRank(aType);
  int64_t bRank = getRank(bType);

  if (aRank < 1 || aRank > 2)
    return op.emitError("Input a must be a sparse vector or sparse matrix.");
  if (bRank < 1 || bRank > 2)
    return op.emitError("Input b must be a sparse vector or sparse matrix.");

  if (failed(verifyCompatibleShape(aType, bType)))
    return op.emitError("Inputs must have compatible shapes.");

  llvm::Optional<std::string> errMsg;
  if (aRank == 1) {
    errMsg = checkCompressedVector(aType, 0);
    if (errMsg)
      return op.emitError(errMsg.getValue());
    errMsg = checkCompressedVector(bType, 1);
    if (errMsg)
      return op.emitError(errMsg.getValue());
  } else if (aRank == 2) {
    errMsg = checkCompressedMatrix(aType, 0, EITHER);
    if (errMsg)
      return op.emitError(errMsg.getValue());
    // Determine output sparse encoding
    CompressionType sparseEncoding = CSR;
    if (typeIsCSC(aType)) {
      sparseEncoding = CSC;
    }
    errMsg = checkCompressedMatrix(bType, 1, sparseEncoding);
    if (errMsg)
      return op.emitError(errMsg.getValue());
  }

  return success();
}

static const std::vector<std::string> supportedUnionOperators{"plus", "min", "times"};

static LogicalResult verify(UnionOp op) {
  llvm::Optional<llvm::StringRef> unionOperator = op.union_operator();
  if (unionOperator) {
    bool operatorSupported = std::find(supportedUnionOperators.begin(),
                                       supportedUnionOperators.end(),
                                       unionOperator->str())
      != supportedUnionOperators.end();
    if (!operatorSupported)
      return op.emitError("\""+unionOperator->str()+"\" is not a supported union operator.");
  }

  return verifyEwise(op);
}

static const std::vector<std::string> supportedIntersectOperators{"plus", "min", "times"};

static LogicalResult verify(IntersectOp op) {
  llvm::Optional<llvm::StringRef> intersectOperator = op.intersect_operator();
  if (intersectOperator) {
    bool operatorSupported = std::find(supportedIntersectOperators.begin(),
                                       supportedIntersectOperators.end(),
                                       intersectOperator->str())
      != supportedIntersectOperators.end();
    if (!operatorSupported)
      return op.emitError("\""+intersectOperator->str()+"\" is not a supported intersect operator.");
  }

  return verifyEwise(op);
}

static LogicalResult verify(EqualOp op) {
  Type aType = op.a().getType();
  Type bType = op.b().getType();

  if (failed(verifyCompatibleShape(aType, bType)))
    return op.emitError("Input vectors must have compatible shapes.");

  if (aType != bType)
    return op.emitError("Arguments must have the same type.");

  int64_t aRank = getRank(aType);
  int64_t bRank = getRank(bType);

  if (aRank < 1 || aRank > 2)
    return op.emitError("First argument must be a sparse vector or sparse matrix.");
  if (bRank < 1 || bRank > 2)
    return op.emitError("Second argument must be a sparse vector or sparse matrix.");

  if (aRank != bRank)
    return op.emitError("Arguments must have same rank.");

  llvm::Optional<std::string> errMsg;
  if (aRank == 1) {
    errMsg = checkCompressedVector(aType, 0);
    if (errMsg)
      return op.emitError(errMsg.getValue());
    errMsg = checkCompressedVector(bType, 1);
    if (errMsg)
      return op.emitError(errMsg.getValue());
  } else if (aRank == 2) {
    errMsg = checkCompressedMatrix(aType, 0, EITHER);
    if (errMsg)
      return op.emitError(errMsg.getValue());
    errMsg = checkCompressedMatrix(bType, 1, EITHER);
    if (errMsg)
      return op.emitError(errMsg.getValue());
  }

  return success();
}

static LogicalResult verify(MatrixReduceToScalarOp op) {
  LogicalResult argResult = verifyMatrixReduceToScalarArgs(op);

  if (argResult.failed())
    return argResult;

  static const std::vector<std::string> supportedAggregators{"sum"};
  std::string aggregator = op.aggregator().str();
  bool aggregatorSupported = std::find(supportedAggregators.begin(), supportedAggregators.end(), aggregator)
    != supportedAggregators.end();
  if (!aggregatorSupported)
    return op.emitError("\""+aggregator+"\" is not a supported aggregator.");

  return success();
}

static LogicalResult verify(MatrixReduceToScalarGenericOp op)
{
  LogicalResult argResult = verifyMatrixReduceToScalarArgs(op);

  if (argResult.failed())
    return argResult;

  RegionRange extensions = op.extensions();
  if (extensions.size() < 1)
  {
    return op.emitError("Must have at least 2 regions: agg_identity, agg.");
  }

  return success();
}

static LogicalResult verify(MatrixSelectOp op) {
  // input and result types are already guaranteed to be the same
  for (auto result : op.getResults()) {
    Type resultType = result.getType();

    llvm::Optional<std::string> resultCompressionErrorMessage = checkCompressedMatrix(resultType, -1, EITHER);
    if (resultCompressionErrorMessage)
      return op.emitError(resultCompressionErrorMessage.getValue());
  }

  static const std::vector<std::string> supportedSelectors{"triu", "tril", "gt0"};
  for (auto selectorAttr : op.selectors()) {
    std::string selector = selectorAttr.dyn_cast_or_null<StringAttr>().getValue().str();
    bool selectorSupported = std::find(supportedSelectors.begin(), supportedSelectors.end(), selector)
      != supportedSelectors.end();
    if (!selectorSupported)
      return op.emitError("\""+selector+"\" is not a supported selector.");
  }
  return success();
}

static LogicalResult verify(ConvertLayoutOp op) {
  Type inputType = op.input().getType();
  Type resultType = op.getResult().getType();

  llvm::Optional<std::string> inputCompressionErrorMessage = checkCompressedMatrix(inputType, 0, EITHER);
  if (inputCompressionErrorMessage)
    return op.emitError(inputCompressionErrorMessage.getValue());

  llvm::Optional<std::string> resultCompressionErrorMessage = checkCompressedMatrix(resultType, -1, EITHER);
  if (resultCompressionErrorMessage)
    return op.emitError(resultCompressionErrorMessage.getValue());

  // TODO intelligently handle arbitrarily shaped tensors, i.e. tensors with shapes using "?"

  RankedTensorType inputTensorType = inputType.dyn_cast<RankedTensorType>();
  RankedTensorType resultTensorType = resultType.dyn_cast<RankedTensorType>();

  if (inputTensorType.getElementType() != resultTensorType.getElementType())
    return op.emitError("Input and output tensors have different element types.");

  ArrayRef<int64_t> inputShape = inputTensorType.getShape();
  ArrayRef<int64_t> resultShape = resultTensorType.getShape();

  mlir::sparse_tensor::SparseTensorEncodingAttr inputSparseEncoding =
    mlir::sparse_tensor::getSparseTensorEncoding(inputType);

  mlir::sparse_tensor::SparseTensorEncodingAttr resultSparseEncoding =
    mlir::sparse_tensor::getSparseTensorEncoding(resultType);

  if (inputShape[0] != resultShape[0] || inputShape[1] != resultShape[1])
    return op.emitError("Input and output shapes are expected to be the same.");

  if (inputSparseEncoding != resultSparseEncoding) {
    AffineMap inputDimOrdering = inputSparseEncoding.getDimOrdering();
    AffineMap resultDimOrdering = resultSparseEncoding.getDimOrdering();
    unsigned inputDimOrdering0 = inputDimOrdering.getDimPosition(0);
    unsigned inputDimOrdering1 = inputDimOrdering.getDimPosition(1);
    unsigned resultDimOrdering0 = resultDimOrdering.getDimPosition(0);
    unsigned resultDimOrdering1 = resultDimOrdering.getDimPosition(1);
    if (inputDimOrdering0 != resultDimOrdering1 || inputDimOrdering1 != resultDimOrdering0)
      return op.emitError("Sparse encoding dimension orderings of input and result tensors "
        "expected to be swapped or encodings must be identical.");

    // TODO should we be more lenient like the sparse tensor dialect is via isMatchingWidth?
    // see llvm-project/mlir/lib/Dialect/SparseTensor/IR/SparseTensorDialect.cpp
    unsigned inputPointerBitWidth = inputSparseEncoding.getPointerBitWidth();
    unsigned resultPointerBitWidth = resultSparseEncoding.getPointerBitWidth();
    if (inputPointerBitWidth != resultPointerBitWidth)
      return op.emitError("Sparse encoding pointer bit widths of input and result tensors do not match.");

    unsigned inputIndexBitWidth = inputSparseEncoding.getIndexBitWidth();
    unsigned resultIndexBitWidth = resultSparseEncoding.getIndexBitWidth();
    if (inputIndexBitWidth != resultIndexBitWidth)
      return op.emitError("Sparse encoding index bit widths of input and result tensors do not match.");
    // dimLevelType values guaranteed to be the same since we already checked earlier
  }

  return success();
}

static LogicalResult verify(TransposeOp op) {
  RankedTensorType inputType = op.input().getType().cast<RankedTensorType>();
  RankedTensorType resultType =
      op.getResult().getType().cast<RankedTensorType>();

  if (failed(hasSparseEncodingAttr(inputType)) ||
      failed(hasCSRorCSCEncoding(inputType)))
    return op.emitError("input: Missing sparse tensor encoding or 2d-tensor "
                        "not in CSR or CSC form");

  if (failed(hasSparseEncodingAttr(resultType)) ||
      failed(hasCSRorCSCEncoding(resultType)))
    return op.emitError("result: Missing sparse tensor encoding or 2d-tensor "
                        "not in CSR or CSC form");

  // TODO intelligently handle arbitrarily shaped tensors, i.e. tensors with shapes using "?"
  if (inputType.getElementType() != resultType.getElementType())
    return op.emitError("Input and output tensors have different element types.");

  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> resultShape = resultType.getShape();

  mlir::sparse_tensor::SparseTensorEncodingAttr inputSparseEncoding =
    mlir::sparse_tensor::getSparseTensorEncoding(inputType);

  mlir::sparse_tensor::SparseTensorEncodingAttr resultSparseEncoding =
    mlir::sparse_tensor::getSparseTensorEncoding(resultType);

  if (inputShape[0] != resultShape[1] || inputShape[1] != resultShape[0])
    return op.emitError("Input and output shapes are expected to be swapped.");

  if (inputSparseEncoding != resultSparseEncoding) {
    AffineMap inputDimOrdering = inputSparseEncoding.getDimOrdering();
    AffineMap resultDimOrdering = resultSparseEncoding.getDimOrdering();
    unsigned inputDimOrdering0 = inputDimOrdering.getDimPosition(0);
    unsigned inputDimOrdering1 = inputDimOrdering.getDimPosition(1);
    unsigned resultDimOrdering0 = resultDimOrdering.getDimPosition(0);
    unsigned resultDimOrdering1 = resultDimOrdering.getDimPosition(1);
    if (inputDimOrdering0 != resultDimOrdering1 || inputDimOrdering1 != resultDimOrdering0)
      return op.emitError("Sparse encoding dimension orderings of input and result tensors "
        "expected to be swapped or encodings must be identical.");

    // TODO should we be more lenient like the sparse tensor dialect is via isMatchingWidth?
    // see llvm-project/mlir/lib/Dialect/SparseTensor/IR/SparseTensorDialect.cpp
    unsigned inputPointerBitWidth = inputSparseEncoding.getPointerBitWidth();
    unsigned resultPointerBitWidth = resultSparseEncoding.getPointerBitWidth();
    if (inputPointerBitWidth != resultPointerBitWidth)
      return op.emitError("Sparse encoding pointer bit widths of input and result tensors do not match.");

    unsigned inputIndexBitWidth = inputSparseEncoding.getIndexBitWidth();
    unsigned resultIndexBitWidth = resultSparseEncoding.getIndexBitWidth();
    if (inputIndexBitWidth != resultIndexBitWidth)
      return op.emitError("Sparse encoding index bit widths of input and result tensors do not match.");
    // dimLevelType values guaranteed to be the same since we already checked earlier
  }

  return success();
}

static LogicalResult verify(YieldOp op)
{
  // no additional verification needed yet
  return success();
}

static LogicalResult verify(CommentOp op)
{
  // no additional verification needed yet
  return success();
}

#define GET_OP_CLASSES
#include "GraphBLAS/GraphBLASOps.cpp.inc"
