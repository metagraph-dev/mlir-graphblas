#ifndef GRAPHBLAS_GRAPHBLASCOMMONPASSES_H
#define GRAPHBLAS_GRAPHBLASCOMMONPASSES_H

#include "GraphBLAS/GraphBLASDialect.h"
#include "GraphBLAS/GraphBLASUtils.h"

class LowerCommentRewrite
    : public mlir::OpRewritePattern<mlir::graphblas::CommentOp> {
public:
  using mlir::OpRewritePattern<mlir::graphblas::CommentOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::graphblas::CommentOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

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

class LowerSizeRewrite
    : public mlir::OpRewritePattern<mlir::graphblas::SizeOp> {
public:
  using OpRewritePattern<mlir::graphblas::SizeOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::graphblas::SizeOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

class LowerNumRowsRewrite
    : public mlir::OpRewritePattern<mlir::graphblas::NumRowsOp> {
public:
  using OpRewritePattern<mlir::graphblas::NumRowsOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::graphblas::NumRowsOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

class LowerNumColsRewrite
    : public mlir::OpRewritePattern<mlir::graphblas::NumColsOp> {
public:
  using OpRewritePattern<mlir::graphblas::NumColsOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::graphblas::NumColsOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

class LowerNumValsRewrite
    : public mlir::OpRewritePattern<mlir::graphblas::NumValsOp> {
public:
  using OpRewritePattern<mlir::graphblas::NumValsOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::graphblas::NumValsOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

#endif // GRAPHBLAS_GRAPHBLASCOMMONPASSES_H
