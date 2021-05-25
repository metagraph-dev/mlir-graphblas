// Lowering test functions

#ifndef GRAPHBLAS_LOWERING_H
#define GRAPHBLAS_LOWERING_H

#include <string>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

void addMatrixApplyFunc(mlir::ModuleOp mod, const std::string &operation);
void addMatrixMultiplyFunc(mlir::ModuleOp mod, const std::string &semi_ring, bool mask);
void addMatrixReduceToScalarFunc(mlir::ModuleOp mod, const std::string &aggregator);
void addMatrixSelectFunc(mlir::ModuleOp mod, const std::string &selector);
void addTransposeFunc(mlir::ModuleOp mod, bool swap_sizes);

// util.cpp
mlir::RankedTensorType getCSRTensorType(mlir::MLIRContext *context, mlir::FloatType valueType);
mlir::CallOp callEmptyLike(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc, mlir::Value tensor);
mlir::CallOp callResizeDim(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc,
                           mlir::Value tensor, mlir::Value d, mlir::Value size);

mlir::CallOp callResizePointers(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc,
                                mlir::Value tensor, mlir::Value d, mlir::Value size);
mlir::CallOp callResizeIndex(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc,
                             mlir::Value tensor, mlir::Value d, mlir::Value size);
mlir::CallOp callResizeValues(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc,
                              mlir::Value tensor, mlir::Value size);

#endif // GRAPHBLAS_LOWERING_FUNCTIONS_H