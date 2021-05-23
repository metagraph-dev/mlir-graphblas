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

#endif // GRAPHBLAS_LOWERING_FUNCTIONS_H