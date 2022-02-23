#ifndef GRAPHBLAS_GRAPHBLASCOMMONPASSES_H
#define GRAPHBLAS_GRAPHBLASCOMMONPASSES_H

#include "GraphBLAS/GraphBLASDialect.h"
#include "GraphBLAS/GraphBLASUtils.h"

class LowerPrintRewrite
    : public mlir::OpRewritePattern<mlir::graphblas::PrintOp> {
public:
  using mlir::OpRewritePattern<mlir::graphblas::PrintOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::graphblas::PrintOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

class LowerPrintTensorRewrite
    : public mlir::OpRewritePattern<mlir::graphblas::PrintTensorOp> {
public:
  using mlir::OpRewritePattern<
      mlir::graphblas::PrintTensorOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::graphblas::PrintTensorOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

#endif // GRAPHBLAS_GRAPHBLASCOMMONPASSES_H
