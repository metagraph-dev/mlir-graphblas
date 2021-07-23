//===- GraphBLASOps.cpp - GraphBLAS dialect ops ---------------*- C++ -*-===//
//
// TODO add documentation
//
//===--------------------------------------------------------------------===//

#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "GraphBLAS/GraphBLASOps.h"
#include "GraphBLAS/GraphBLASDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/None.h"

#include "GraphBLAS/GraphBLASOpsEnums.cpp.inc"

using namespace mlir;
using namespace mlir::graphblas;

//===--------------------------------------------------------------------===//
// Helpers
//===--------------------------------------------------------------------===//

enum CompressionType { CSR, CSC, EITHER };

static llvm::Optional<std::string> checkCompressedSparseTensor(
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

  if (compressionType != EITHER) {
    
    AffineMap dimOrdering = sparseEncoding.getDimOrdering();
    unsigned dimOrdering0 = dimOrdering.getDimPosition(0); 
    unsigned dimOrdering1 = dimOrdering.getDimPosition(1);

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

static LogicalResult verify(SizeOp op) {
  Type inputType = op.input().getType();

  mlir::sparse_tensor::SparseTensorEncodingAttr sparseEncoding =
    mlir::sparse_tensor::getSparseTensorEncoding(inputType);
  if (!sparseEncoding)
    op.emitError("Input must be a sparse tensor.");

  RankedTensorType inputTensorType = inputType.dyn_cast<RankedTensorType>();
  int64_t rank = inputTensorType.getRank();

  if (rank != 1)
    return op.emitError("Input must be a vector (rank 1 tensor).");

  return success();
}

void SizeOp::build(OpBuilder &builder, OperationState &result, Value tensor) {
  Type indexType = builder.getIndexType();
  build(builder, result, indexType, tensor);
}

static LogicalResult verify(NumRowsOp op) {
  Type inputType = op.input().getType();

  llvm::Optional<std::string> inputCompressionErrorMessage = checkCompressedSparseTensor(inputType, 0, EITHER);
  if (inputCompressionErrorMessage)
    return op.emitError(inputCompressionErrorMessage.getValue());

  RankedTensorType inputTensorType = inputType.dyn_cast<RankedTensorType>();
  int64_t rank = inputTensorType.getRank();

  if (rank != 2)
    return op.emitError("Input must be a matrix (rank 2 tensor).");

  return success();
}

void NumRowsOp::build(OpBuilder &builder, OperationState &result, Value tensor) {
  Type indexType = builder.getIndexType();
  build(builder, result, indexType, tensor);
}

static LogicalResult verify(NumColsOp op) {
  Type inputType = op.input().getType();

  llvm::Optional<std::string> inputCompressionErrorMessage = checkCompressedSparseTensor(inputType, 0, EITHER);
  if (inputCompressionErrorMessage)
    return op.emitError(inputCompressionErrorMessage.getValue());

  RankedTensorType inputTensorType = inputType.dyn_cast<RankedTensorType>();
  int64_t rank = inputTensorType.getRank();

  if (rank != 2)
    return op.emitError("Input must be a matrix (rank 2 tensor).");

  return success();
}

void NumColsOp::build(OpBuilder &builder, OperationState &result, Value tensor) {
  Type indexType = builder.getIndexType();
  build(builder, result, indexType, tensor);
}

static LogicalResult verify(NumValsOp op) {
  Type inputType = op.input().getType();

  mlir::sparse_tensor::SparseTensorEncodingAttr sparseEncoding =
    mlir::sparse_tensor::getSparseTensorEncoding(inputType);
  if (!sparseEncoding)
    op.emitError("Input must be a sparse tensor.");

  RankedTensorType inputTensorType = inputType.dyn_cast<RankedTensorType>();
  int64_t rank = inputTensorType.getRank();

  // Require sparse matrices to be either CSR or CSC
  if (rank == 2) {
    llvm::Optional<std::string> inputCompressionErrorMessage = checkCompressedSparseTensor(inputType, 0, EITHER);
    if (inputCompressionErrorMessage)
      return op.emitError(inputCompressionErrorMessage.getValue());
  }

  return success();
}

void NumValsOp::build(OpBuilder &builder, OperationState &result, Value tensor) {
  Type indexType = builder.getIndexType();
  build(builder, result, indexType, tensor);
}

static LogicalResult verify(DupOp op) {
  Type inputType = op.input().getType();

  mlir::sparse_tensor::SparseTensorEncodingAttr sparseEncoding =
    mlir::sparse_tensor::getSparseTensorEncoding(inputType);
  if (!sparseEncoding)
    op.emitError("Input must be a sparse tensor.");

  RankedTensorType inputTensorType = inputType.dyn_cast<RankedTensorType>();
  int64_t rank = inputTensorType.getRank();

  // Require sparse matrices to be either CSR or CSC
  if (rank == 2) {
    llvm::Optional<std::string> inputCompressionErrorMessage = checkCompressedSparseTensor(inputType, 0, EITHER);
    if (inputCompressionErrorMessage)
      return op.emitError(inputCompressionErrorMessage.getValue());
  }

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

  llvm::Optional<std::string> inputCompressionErrorMessage = checkCompressedSparseTensor(inputType, 0, EITHER);
  if (inputCompressionErrorMessage)
    return op.emitError(inputCompressionErrorMessage.getValue());

  llvm::Optional<std::string> resultCompressionErrorMessage = checkCompressedSparseTensor(resultType, -1, EITHER);
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
static LogicalResult verifyMatrixMultiplyArgs(T op)
{
  Type aType = op.a().getType();
  Type bType = op.b().getType();
  Type resultType = op.getResult().getType();

  llvm::Optional<std::string> aCompressionErrorMessage = checkCompressedSparseTensor(aType, 0, CSR);
  if (aCompressionErrorMessage)
    return op.emitError(aCompressionErrorMessage.getValue());

  llvm::Optional<std::string> bCompressionErrorMessage = checkCompressedSparseTensor(bType, 1, CSC);
  if (bCompressionErrorMessage)
    return op.emitError(bCompressionErrorMessage.getValue());

  llvm::Optional<std::string> resultCompressionErrorMessage = checkCompressedSparseTensor(resultType, -1, CSR);
  if (resultCompressionErrorMessage)
    return op.emitError(resultCompressionErrorMessage.getValue());

  RankedTensorType aTensorType = aType.dyn_cast<RankedTensorType>();
  RankedTensorType bTensorType = bType.dyn_cast<RankedTensorType>();
  RankedTensorType resultTensorType = resultType.dyn_cast<RankedTensorType>();

  ArrayRef<int64_t> aShape = aTensorType.getShape();
  ArrayRef<int64_t> bShape = bTensorType.getShape();
  ArrayRef<int64_t> resultShape = resultTensorType.getShape();
  // TODO intelligently handle arbitrarily shaped tensors, i.e. tensors with shapes using "?"
  if (aShape[1] != bShape[0])
    return op.emitError("Operand shapes are incompatible.");
  if (resultShape[0] != aShape[0] || resultShape[1] != bShape[1])
    return op.emitError("Operand shapes incompatible with output shape.");

  if (aTensorType.getElementType() != bTensorType.getElementType())
    return op.emitError("Operand element types must be identical.");
  if (aTensorType.getElementType() != resultTensorType.getElementType())
    return op.emitError("Result element type differs from the input element types.");

  Value mask = op.mask();
  if (mask)
  {
    Type maskType = mask.getType();
    llvm::Optional<std::string> maskCompressionErrorMessage = checkCompressedSparseTensor(maskType, 2, CSR);
    if (maskCompressionErrorMessage)
      return op.emitError(maskCompressionErrorMessage.getValue());

    RankedTensorType maskTensorType = maskType.dyn_cast<RankedTensorType>();
    ArrayRef<int64_t> maskShape = maskTensorType.getShape();
    if (resultShape[0] != maskShape[0] || resultShape[1] != maskShape[1])
      return op.emitError("Mask shape must match output shape.");
  }

  return success();
}

static const std::vector<std::string> supportedSemirings{"plus_times", "plus_pair", "plus_plus"};

static LogicalResult verify(MatrixMultiplyOp op) {
  LogicalResult argResult = verifyMatrixMultiplyArgs(op);

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
  LogicalResult argResult = verifyMatrixMultiplyArgs(op);

  if (argResult.failed())
    return argResult;

  RegionRange extensions = op.extensions();
  if (extensions.size() < 3)
  {
    return op.emitError("Must have at least 3 regions: add_identity, add, mult.");
  }

  return success();
}

static LogicalResult verify(MatrixMultiplyReduceToScalarOp op) {
  Type aType = op.a().getType();
  Type bType = op.b().getType();
  Type resultType = op.getResult().getType();

  llvm::Optional<std::string> aCompressionErrorMessage = checkCompressedSparseTensor(aType, 0, CSR);
  if (aCompressionErrorMessage)
    return op.emitError(aCompressionErrorMessage.getValue());

  llvm::Optional<std::string> bCompressionErrorMessage = checkCompressedSparseTensor(bType, 1, CSC);
  if (bCompressionErrorMessage)
    return op.emitError(bCompressionErrorMessage.getValue());

  static const std::vector<std::string> supportedAggregators{"sum"};
  std::string aggregator = op.aggregator().str();
  bool aggregatorSupported = std::find(supportedAggregators.begin(), supportedAggregators.end(), aggregator)
    != supportedAggregators.end();
  if (!aggregatorSupported)
    return op.emitError("\""+aggregator+"\" is not a supported aggregator.");

  RankedTensorType operandTensorType = aType.dyn_cast<RankedTensorType>();
  if (resultType != operandTensorType.getElementType())
    return op.emitError("Operand and output types are incompatible.");

  std::string semiring = op.semiring().str();
  bool semiringSupported = std::find(supportedSemirings.begin(), supportedSemirings.end(), semiring)
    != supportedSemirings.end();
  if (!semiringSupported)
    return op.emitError("\""+semiring+"\" is not a supported semiring.");

  RankedTensorType aTensorType = aType.dyn_cast<RankedTensorType>();
  RankedTensorType bTensorType = bType.dyn_cast<RankedTensorType>();

  ArrayRef<int64_t> aShape = aTensorType.getShape();
  ArrayRef<int64_t> bShape = bTensorType.getShape();
  // TODO intelligently handle arbitrarily shaped tensors, i.e. tensors with shapes using "?"
  if (aShape[1] != bShape[0])
    return op.emitError("Operand shapes are incompatible.");

  if (aTensorType.getElementType() != bTensorType.getElementType())
    return op.emitError("Operand element types must be identical.");

  Value mask = op.mask();
  if (mask) {
    Type maskType = mask.getType();
    llvm::Optional<std::string> maskCompressionErrorMessage = checkCompressedSparseTensor(maskType, 2, CSR);
    if (maskCompressionErrorMessage)
      return op.emitError(maskCompressionErrorMessage.getValue());

    RankedTensorType maskTensorType = maskType.dyn_cast<RankedTensorType>();
    ArrayRef<int64_t> maskShape = maskTensorType.getShape();
    if (aShape[0] != maskShape[0] || bShape[1] != maskShape[1])
      return op.emitError("Mask shape must match shape from result of matrix multiply.");
  }

  return success();
}

static LogicalResult verify(MatrixVectorMultiplyOp op) {
  Type matType = op.mat().getType();
  Type vecType = op.vec().getType();
  Type resultType = op.getResult().getType();

  llvm::Optional<std::string> matCompressionErrorMessage = checkCompressedSparseTensor(matType, 0, CSR);
  if (matCompressionErrorMessage)
    return op.emitError(matCompressionErrorMessage.getValue());

  llvm::Optional<std::string> vecCompressionErrorMessage = checkCompressedVector(vecType, 1);
  if (vecCompressionErrorMessage)
    return op.emitError(vecCompressionErrorMessage.getValue());

  llvm::Optional<std::string> resultCompressionErrorMessage = checkCompressedSparseTensor(resultType, -1, CSR);
  if (resultCompressionErrorMessage)
    return op.emitError(resultCompressionErrorMessage.getValue());

  std::string semiring = op.semiring().str();
  bool semiringSupported = std::find(supportedSemirings.begin(), supportedSemirings.end(), semiring)
    != supportedSemirings.end();
  if (!semiringSupported)
    return op.emitError("\""+semiring+"\" is not a supported semiring.");

  RankedTensorType matTensorType = matType.dyn_cast<RankedTensorType>();
  RankedTensorType vecTensorType = vecType.dyn_cast<RankedTensorType>();
  RankedTensorType resultTensorType = resultType.dyn_cast<RankedTensorType>();

  ArrayRef<int64_t> matShape = matTensorType.getShape();
  ArrayRef<int64_t> vecShape = vecTensorType.getShape();
  ArrayRef<int64_t> resultShape = resultTensorType.getShape();
  // TODO intelligently handle arbitrarily shaped tensors, i.e. tensors with shapes using "?"
  if (matShape[1] != vecShape[0])
    return op.emitError("Operand shapes are incompatible.");
  if (resultShape[0] != matShape[0] || resultShape[1] != matShape[0])
    return op.emitError("Operand shapes incompatible with output shape.");

  if (matTensorType.getElementType() != vecTensorType.getElementType())
    return op.emitError("Operand element types must be identical.");
  if (matTensorType.getElementType() != resultTensorType.getElementType())
    return op.emitError("Result element type differs from the input element types.");

  Value mask = op.mask();
  if (mask) {
    Type maskType = mask.getType();
    llvm::Optional<std::string> maskCompressionErrorMessage = checkCompressedSparseTensor(maskType, 2, CSR);
    if (maskCompressionErrorMessage)
      return op.emitError(maskCompressionErrorMessage.getValue());

    RankedTensorType maskTensorType = maskType.dyn_cast<RankedTensorType>();
    ArrayRef<int64_t> maskShape = maskTensorType.getShape();
    if (resultShape[0] != maskShape[0] || resultShape[1] != maskShape[1])
      return op.emitError("Mask shape must match output shape.");
  }

  Region &body = op.body();
  auto numBlocks = body.getBlocks().size();
  if (numBlocks > 1) {
    return op.emitError("Region must have at most one block.");
  }

  return success();
}

static const std::vector<std::string> supportedVectorAccumulateOperators{"plus"};

static LogicalResult verify(VectorAccumulateOp op) {  
  Type aType = op.a().getType();
  Type bType = op.b().getType();
  
  llvm::Optional<std::string> aCompressionErrorMessage = checkCompressedVector(aType, 0);
  if (aCompressionErrorMessage)
    return op.emitError(aCompressionErrorMessage.getValue());
  
  if (failed(verifyCompatibleShape(aType, bType)))
    return op.emitError("Input vectors must have compatible shapes.");
 
  if (aType != bType)
    return op.emitError("Input vectors must have the same type.");
  
  std::string accumulateOperator = op.accumulate_operator().str();
  bool operatorSupported = std::find(supportedVectorAccumulateOperators.begin(),
				     supportedVectorAccumulateOperators.end(),
				     accumulateOperator)
    != supportedVectorAccumulateOperators.end();
  if (!operatorSupported)
    return op.emitError("\""+accumulateOperator+"\" is not a supported accumulate operator.");

  return success();
}

static LogicalResult verify(VectorDotProductOp op) {
  Type aType = op.a().getType();
  Type bType = op.b().getType();
  
  llvm::Optional<std::string> aCompressionErrorMessage = checkCompressedVector(aType, 0);
  if (aCompressionErrorMessage)
    return op.emitError(aCompressionErrorMessage.getValue());

  if (failed(verifyCompatibleShape(aType, bType)))
    return op.emitError("Input vectors must have compatible shapes.");
  
  if (aType != bType)
    return op.emitError("Input vectors must have the same type.");

  Type resultType = op.getResult().getType();
  RankedTensorType aTensorType = aType.dyn_cast<RankedTensorType>();
  if (aTensorType.getElementType() != resultType)
    return op.emitError("Result type must have same type as the element type of the input vectors.");
    
  std::string semiring = op.semiring().str();
  bool semiringSupported = std::find(supportedSemirings.begin(),
   				     supportedSemirings.end(),
   				     semiring)
    != supportedSemirings.end();
  if (!semiringSupported)
    return op.emitError("\""+semiring+"\" is not a supported semiring.");
  
  return success();
}

static LogicalResult verify(VectorEqualsOp op) {
  Type aType = op.a().getType();
  Type bType = op.b().getType();
  
  llvm::Optional<std::string> aCompressionErrorMessage = checkCompressedVector(aType, 0);
  if (aCompressionErrorMessage)
    return op.emitError(aCompressionErrorMessage.getValue());
  
  if (failed(verifyCompatibleShape(aType, bType)))
    return op.emitError("Input vectors must have compatible shapes.");

  if (aType != bType)
    return op.emitError("Input vectors must have the same type.");
    
  return success();
}

static LogicalResult verify(MatrixReduceToScalarOp op) {
  Type operandType = op.input().getType();

  llvm::Optional<std::string> compressionErrorMessage = checkCompressedSparseTensor(operandType, 0, EITHER);
  if (compressionErrorMessage)
    return op.emitError(compressionErrorMessage.getValue());

  static const std::vector<std::string> supportedAggregators{"sum"};
  std::string aggregator = op.aggregator().str();
  bool aggregatorSupported = std::find(supportedAggregators.begin(), supportedAggregators.end(), aggregator)
    != supportedAggregators.end();
  if (!aggregatorSupported)
    return op.emitError("\""+aggregator+"\" is not a supported aggregator.");

  Type resultType = op.getResult().getType();
  RankedTensorType operandTensorType = operandType.dyn_cast<RankedTensorType>();
  if (resultType != operandTensorType.getElementType())
    return op.emitError("Operand and output types are incompatible.");

  return success();
}

static LogicalResult verify(MatrixSelectOp op) {
  // input and result types are already guaranteed to be the same
  for (auto result : op.getResults()) {
    Type resultType = result.getType();

    llvm::Optional<std::string> resultCompressionErrorMessage = checkCompressedSparseTensor(resultType, -1, EITHER);
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

  llvm::Optional<std::string> inputCompressionErrorMessage = checkCompressedSparseTensor(inputType, 0, EITHER);
  if (inputCompressionErrorMessage)
    return op.emitError(inputCompressionErrorMessage.getValue());

  llvm::Optional<std::string> resultCompressionErrorMessage = checkCompressedSparseTensor(resultType, -1, EITHER);
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

static LogicalResult verify(YieldOp op)
{
  // no additional verification needed yet
  return success();
}

#define GET_OP_CLASSES
#include "GraphBLAS/GraphBLASOps.cpp.inc"
