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
  return success();
}

static LogicalResult verify(TransposeOp op) {
  return success();
}

#define GET_OP_CLASSES
#include "GraphBLAS/GraphBLASOps.cpp.inc"
