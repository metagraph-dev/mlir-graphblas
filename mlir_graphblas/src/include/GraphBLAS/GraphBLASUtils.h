#ifndef GRAPHBLAS_GRAPHBLASUTILS_H
#define GRAPHBLAS_GRAPHBLASUTILS_H

#include <string>
#include <set>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringSet.h"

static const llvm::StringSet<> supportedIntersectOperators{"plus", "min", "times"};

static const llvm::StringSet<> supportedUnionOperators{"plus", "min", "times"};

static const llvm::StringSet<> supportedUpdateAccumulateOperators{"plus", "min"};

static const llvm::StringSet<> supportedSemirings{"plus_times", "plus_pair", "plus_plus", "min_plus"};

static const llvm::StringSet<> supportedReduceAggregators{"plus"};

static const llvm::StringSet<> supportedSelectors{"triu", "tril", "gt"};
static const llvm::StringSet<> supportedThunkNeedingSelectors{"gt"};

static const llvm::StringSet<> supportedApplyOperators{"min"};

bool typeIsCSR(mlir::Type inputType);
bool typeIsCSC(mlir::Type inputType);
mlir::RankedTensorType getCompressedVectorType(mlir::MLIRContext *context, mlir::ArrayRef<int64_t> shape, mlir::Type valueType);
mlir::RankedTensorType getCSRTensorType(mlir::MLIRContext *context, llvm::ArrayRef<int64_t> shape, mlir::Type valueType);
mlir::RankedTensorType getCSCTensorType(mlir::MLIRContext *context, llvm::ArrayRef<int64_t> shape, mlir::Type valueType);

int64_t getRank(mlir::Type inputType);
int64_t getRank(mlir::Value inputValue);

mlir::Value convertToExternalCSR(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc, mlir::Value input);
mlir::Value convertToExternalCSC(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc, mlir::Value input);
mlir::Value callEmpty(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc, mlir::Value inputTensor, llvm::ArrayRef<int64_t> resultShape);
mlir::Value callEmptyLike(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc, mlir::Value tensor);
mlir::Value callDupTensor(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc, mlir::Value tensor);
void callDelSparseTensor(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc, mlir::Value tensor);

mlir::CallOp callResizeDim(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc,
                           mlir::Value tensor, mlir::Value d, mlir::Value size);
mlir::CallOp callResizePointers(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc,
                                mlir::Value tensor, mlir::Value d, mlir::Value size);
mlir::CallOp callResizeIndex(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc,
                             mlir::Value tensor, mlir::Value d, mlir::Value size);
mlir::CallOp callResizeValues(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc,
                              mlir::Value tensor, mlir::Value size);

void cleanupIntermediateTensor(mlir::OpBuilder &builder, mlir::ModuleOp &mod, mlir::Location loc, mlir::Value tensor);

struct ExtensionBlocks {
    mlir::Block *transformInA = nullptr;
    mlir::Block *transformInB = nullptr;
    mlir::Block *transformOut = nullptr;
    mlir::Block *selectInA = nullptr;
    mlir::Block *selectInB = nullptr;
    mlir::Block *selectOut = nullptr;
    mlir::Block *addIdentity = nullptr;
    mlir::Block *add = nullptr;
    mlir::Block *multIdentity = nullptr;
    mlir::Block *mult = nullptr;
    mlir::Block *aggIdentity = nullptr;
    mlir::Block *agg = nullptr;

    ExtensionBlocks() { };
    mlir::LogicalResult extractBlocks(mlir::Operation *op, mlir::RegionRange &regions,
                                      const std::set<mlir::graphblas::YieldKind> &required,
                                      const std::set<mlir::graphblas::YieldKind> &optional);
};

mlir::LogicalResult populateSemiringRegions(mlir::OpBuilder &builder, mlir::Location loc,
                                            mlir::StringRef semiring, mlir::Type valueType,
                                            mlir::RegionRange regions);

#endif // GRAPHBLAS_GRAPHBLASUTILS_H
