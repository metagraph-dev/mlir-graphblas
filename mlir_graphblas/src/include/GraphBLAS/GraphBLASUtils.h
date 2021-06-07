#ifndef GRAPHBLAS_GRAPHBLASUTILS_H
#define GRAPHBLAS_GRAPHBLASUTILS_H

#include <string>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/APInt.h"


mlir::RankedTensorType getCSRTensorType(mlir::MLIRContext *context, llvm::ArrayRef<int64_t> shape, mlir::Type valueType);
mlir::RankedTensorType getCSCTensorType(mlir::MLIRContext *context, llvm::ArrayRef<int64_t> shape, mlir::Type valueType);

mlir::CallOp callEmptyLike(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc, mlir::Value tensor);
mlir::CallOp callDupTensor(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc, mlir::Value tensor);

mlir::CallOp callResizeDim(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc,
                           mlir::Value tensor, mlir::Value d, mlir::Value size);
mlir::CallOp callResizePointers(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc,
                                mlir::Value tensor, mlir::Value d, mlir::Value size);
mlir::CallOp callResizeIndex(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc,
                             mlir::Value tensor, mlir::Value d, mlir::Value size);
mlir::CallOp callResizeValues(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc,
                              mlir::Value tensor, mlir::Value size);

#endif // GRAPHBLAS_GRAPHBLASUTILS_H
