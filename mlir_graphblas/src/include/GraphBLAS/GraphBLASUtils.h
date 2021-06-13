#ifndef GRAPHBLAS_GRAPHBLASUTILS_H
#define GRAPHBLAS_GRAPHBLASUTILS_H

#include <string>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/APInt.h"


bool typeIsCSR(mlir::Type inputType);
bool typeIsCSC(mlir::Type inputType);
mlir::RankedTensorType getCSRTensorType(mlir::MLIRContext *context, llvm::ArrayRef<int64_t> shape, mlir::Type valueType);
mlir::RankedTensorType getCSCTensorType(mlir::MLIRContext *context, llvm::ArrayRef<int64_t> shape, mlir::Type valueType);

mlir::Value convertToExternalCSR(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc, mlir::Value input);
mlir::Value convertToExternalCSC(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc, mlir::Value input);
mlir::Value callEmptyLike(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc, mlir::Value tensor);
mlir::Value callDupTensor(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc, mlir::Value tensor);

mlir::CallOp callResizeDim(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc,
                           mlir::Value tensor, mlir::Value d, mlir::Value size);
mlir::CallOp callResizePointers(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc,
                                mlir::Value tensor, mlir::Value d, mlir::Value size);
mlir::CallOp callResizeIndex(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc,
                             mlir::Value tensor, mlir::Value d, mlir::Value size);
mlir::CallOp callResizeValues(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc,
                              mlir::Value tensor, mlir::Value size);

#endif // GRAPHBLAS_GRAPHBLASUTILS_H
