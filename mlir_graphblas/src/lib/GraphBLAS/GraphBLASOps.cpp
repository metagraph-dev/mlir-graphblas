//===- GraphBLASOps.cpp - GraphBLAS dialect ops ---------------*- C++ -*-===//
//
// TODO add documentation
//
//===--------------------------------------------------------------------===//

#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "GraphBLAS/GraphBLASOps.h"
#include "GraphBLAS/GraphBLASDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/None.h"

using namespace mlir;
using namespace mlir::graphblas;

//===--------------------------------------------------------------------===//
// Helpers
//===--------------------------------------------------------------------===//

static llvm::Optional<std::string> checkCSROrCSCTensor(Type inputType, int inputIndex) {
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

  return llvm::None;
}

//===--------------------------------------------------------------------===//
// GraphBLAS Ops Methods
//===--------------------------------------------------------------------===//

static LogicalResult verify(MatrixApplyOp op) {
  Type inputType = op.input().getType();
  Type thunkType = op.thunk().getType();
  Type resultType = op.getResult().getType();

  llvm::Optional<std::string> inputCompressionErrorMessage = checkCSROrCSCTensor(inputType, 0);
  if (inputCompressionErrorMessage)
    return op.emitError(inputCompressionErrorMessage.getValue());
  
  llvm::Optional<std::string> resultCompressionErrorMessage = checkCSROrCSCTensor(resultType, -1);
  if (resultCompressionErrorMessage)
    return op.emitError(resultCompressionErrorMessage.getValue());

  RankedTensorType inputTensorType = inputType.dyn_cast<RankedTensorType>();
  RankedTensorType resultTensorType = resultType.dyn_cast<RankedTensorType>();
  
  if (inputTensorType.getElementType() != thunkType)
    return op.emitError("Element type of input tensor does not match type of thunk.");

  if (resultTensorType.getElementType() != thunkType)
    return op.emitError("Element type of result tensor does not match type of thunk.");
  
  ArrayRef<int64_t> inputShape = inputTensorType.getShape();
  ArrayRef<int64_t> resultShape = resultTensorType.getShape();
  
  // TODO intelligently handle arbitrarily shaped tensors, i.e. tensors with shapes using "?"
  if (inputShape[0] != resultShape[0] || inputShape[1] != resultShape[1])
    return op.emitError("Input shape does not match output shape.");
  
  static const std::vector<std::string> supportedOperators{"min"};
  std::string applyOperator = op.apply_operator().str();
  bool operatorSupported = std::find(supportedOperators.begin(), supportedOperators.end(), applyOperator)
    != supportedOperators.end();
  if (!operatorSupported)
    return op.emitError("\""+applyOperator+"\" is not a supported semiring.");
  
  return success();
}

static LogicalResult verify(MatrixMultiplyOp op) {
  Type aType = op.a().getType();
  Type bType = op.b().getType();
  Type resultType = op.getResult().getType();
  
  llvm::Optional<std::string> aCompressionErrorMessage = checkCSROrCSCTensor(aType, 0);
  if (aCompressionErrorMessage)
    return op.emitError(aCompressionErrorMessage.getValue());

  llvm::Optional<std::string> bCompressionErrorMessage = checkCSROrCSCTensor(bType, 1);
  if (bCompressionErrorMessage)
    return op.emitError(bCompressionErrorMessage.getValue());

  llvm::Optional<std::string> resultCompressionErrorMessage = checkCSROrCSCTensor(resultType, -1);
  if (resultCompressionErrorMessage)
    return op.emitError(resultCompressionErrorMessage.getValue());
  
  static const std::vector<std::string> supportedSemirings{"plus_times", "plus_pair", "plus_plus"};
  std::string semiring = op.semiring().str();
  bool semiringSupported = std::find(supportedSemirings.begin(), supportedSemirings.end(), semiring)
    != supportedSemirings.end();
  if (!semiringSupported)
    return op.emitError("\""+semiring+"\" is not a supported semiring.");

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
  
  // TODO sanity check the mask shape
  // TODO sanity check the mask shape element type
  
  return success();
}

static LogicalResult verify(MatrixReduceToScalarOp op) {
  Type operandType = op.input().getType();
  
  llvm::Optional<std::string> compressionErrorMessage = checkCSROrCSCTensor(operandType, 0);
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
  Type resultType = op.getResult().getType();
  
  llvm::Optional<std::string> resultCompressionErrorMessage = checkCSROrCSCTensor(resultType, -1);
  if (resultCompressionErrorMessage)
    return op.emitError(resultCompressionErrorMessage.getValue());
    
  static const std::vector<std::string> supportedSelectors{"triu", "tril", "gt0"};
  std::string selector = op.selector().str();
  bool selectorSupported = std::find(supportedSelectors.begin(), supportedSelectors.end(), selector)
    != supportedSelectors.end();
  if (!selectorSupported)
    return op.emitError("\""+selector+"\" is not a supported selector.");
  
  return success();
}

static LogicalResult verify(TransposeOp op) {
  Type inputType = op.input().getType();
  Type resultType = op.getResult().getType();

  llvm::Optional<std::string> inputCompressionErrorMessage = checkCSROrCSCTensor(inputType, 0);
  if (inputCompressionErrorMessage)
    return op.emitError(inputCompressionErrorMessage.getValue());

  llvm::Optional<std::string> resultCompressionErrorMessage = checkCSROrCSCTensor(resultType, -1);
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
  
  bool swapSizes = op.swap_sizes();
  if (swapSizes) {
    if (inputShape[0] != resultShape[1] || inputShape[1] != resultShape[0])
      return op.emitError("Input and output shapes are expected to be swapped.");
    if (inputSparseEncoding != resultSparseEncoding)
      return op.emitError("Input and output tensors are expected to have identical sparse encodings.");
  } else {
    if (inputShape[0] != resultShape[0] || inputShape[1] != resultShape[1])
      return op.emitError("Input and output shapes are expected to be the same.");

    // TODO Check that the dim ordering is swapped.
    AffineMap inputDimOrdering = inputSparseEncoding.getDimOrdering();
    AffineMap resultDimOrdering = resultSparseEncoding.getDimOrdering();
    unsigned inputDimOrdering0 = inputDimOrdering.getDimPosition(0); 
    unsigned inputDimOrdering1 = inputDimOrdering.getDimPosition(1);
    unsigned resultDimOrdering0 = resultDimOrdering.getDimPosition(0); 
    unsigned resultDimOrdering1 = resultDimOrdering.getDimPosition(1);
    if (inputDimOrdering0 != resultDimOrdering1 || inputDimOrdering1 != resultDimOrdering0)
      return op.emitError("Sparse encoding dimension orderings of input and result tensors expected to be swapped.");
    
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

#define GET_OP_CLASSES
#include "GraphBLAS/GraphBLASOps.cpp.inc"
