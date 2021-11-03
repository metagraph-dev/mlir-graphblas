#ifndef GRAPHBLAS_GRAPHBLASUTILS_H
#define GRAPHBLAS_GRAPHBLASUTILS_H

#include "GraphBLAS/GraphBLASOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringSet.h"
#include <set>
#include <string>

enum CompressionType { CSR, CSC, EITHER };

static const llvm::StringSet<> supportedUnionOperators{
    "plus", "times", "min", "max", "first", "second"};

static const llvm::StringSet<> supportedIntersectOperators{
    "plus", "minus", "times", "div", "min", "max", "first", "second"};
//"eq", "ne", "lt", "le", "gt", "ge"};

static const llvm::StringSet<> supportedUpdateAccumulateOperators{
    "plus", "times", "min", "max"};

// These must match the options supported by
// GraphBLASUtils.cpp::populateSemiringAdd()
static const llvm::StringSet<> supportedSemiringAddNames{"plus", "any", "min"};
// These must match the options supported by
// GraphBLASUtils.cpp::populateSemiringMultiply()
static const llvm::StringSet<> supportedSemiringMultiplyNames{
    "pair",    "times",    "plus",  "firsti",
    "secondi", "overlapi", "first", "second"};

static const llvm::StringSet<> supportedReduceAggregators{"plus", "count",
                                                          "argmin", "argmax"};

static const llvm::StringSet<> supportedSelectors{"triu", "tril", "gt", "ge",
                                                  "probability"};
static const llvm::StringSet<> supportedSelectorsNeedingThunk{"gt", "ge",
                                                              "probability"};
static const llvm::StringSet<> supportedSelectorsComparingValues{"gt", "ge",
                                                                 "probability"};

static const llvm::StringSet<> supportedBinaryApplyOperators{"min", "div",
                                                             "pow", "fill"};
static const llvm::StringSet<> supportedUnaryApplyOperators{"abs", "minv",
                                                            "ainv", "identity"};

bool hasRowOrdering(mlir::Type inputType);
bool hasColumnOrdering(mlir::Type inputType);
bool typeIsCSR(mlir::Type inputType);
bool typeIsCSC(mlir::Type inputType);
mlir::RankedTensorType getCompressedVectorType(mlir::MLIRContext *context,
                                               mlir::ArrayRef<int64_t> shape,
                                               mlir::Type valueType,
                                               unsigned ptrBitWidth,
                                               unsigned idxBitWidth);
mlir::RankedTensorType getCompressedVectorType(mlir::MLIRContext *context,
                                               mlir::Type valueType);
mlir::RankedTensorType
getSingleCompressedMatrixType(mlir::MLIRContext *context,
                              mlir::ArrayRef<int64_t> shape,
                              bool columnOriented, mlir::Type valueType,
                              unsigned ptrBitWidth, unsigned idxBitWidth);
mlir::RankedTensorType getCSRType(mlir::MLIRContext *context,
                                  mlir::Type valueType);
mlir::RankedTensorType getCSCType(mlir::MLIRContext *context,
                                  mlir::Type valueType);

int64_t getRank(mlir::Type inputType);
int64_t getRank(mlir::Value inputValue);

void callPrintTensor(mlir::OpBuilder &builder, mlir::ModuleOp &mod,
                     mlir::Location loc, mlir::Value input);
void callPrintString(mlir::OpBuilder &builder, mlir::ModuleOp &mod,
                     mlir::Location loc, mlir::StringRef string);
void callPrintValue(mlir::OpBuilder &builder, mlir::ModuleOp &mod,
                    mlir::Location loc, mlir::Value input);
mlir::Value castToPtr8(mlir::OpBuilder &builder, mlir::ModuleOp &mod,
                       mlir::Location loc, mlir::Value input);
mlir::Value castToTensor(mlir::OpBuilder &builder, mlir::ModuleOp &mod,
                         mlir::Location loc, mlir::Value input,
                         mlir::RankedTensorType tensorType);
mlir::Value callNewTensor(mlir::OpBuilder &builder, mlir::ModuleOp &mod,
                          mlir::Location loc, mlir::ValueRange shape,
                          mlir::RankedTensorType tensorType);
mlir::Value callEmptyLike(mlir::OpBuilder &builder, mlir::ModuleOp &mod,
                          mlir::Location loc, mlir::Value tensor);
mlir::Value callDupTensor(mlir::OpBuilder &builder, mlir::ModuleOp &mod,
                          mlir::Location loc, mlir::Value tensor);

mlir::CallOp callAssignRev(mlir::OpBuilder &builder, mlir::ModuleOp &mod,
                           mlir::Location loc, mlir::Value tensor,
                           mlir::Value d, mlir::Value newIndexValue);
mlir::CallOp callResizeDim(mlir::OpBuilder &builder, mlir::ModuleOp &mod,
                           mlir::Location loc, mlir::Value tensor,
                           mlir::Value d, mlir::Value size);
mlir::CallOp callResizePointers(mlir::OpBuilder &builder, mlir::ModuleOp &mod,
                                mlir::Location loc, mlir::Value tensor,
                                mlir::Value d, mlir::Value size);
mlir::CallOp callResizeIndex(mlir::OpBuilder &builder, mlir::ModuleOp &mod,
                             mlir::Location loc, mlir::Value tensor,
                             mlir::Value d, mlir::Value size);
mlir::CallOp callResizeValues(mlir::OpBuilder &builder, mlir::ModuleOp &mod,
                              mlir::Location loc, mlir::Value tensor,
                              mlir::Value size);
mlir::CallOp callSwapPointers(mlir::OpBuilder &builder, mlir::ModuleOp &mod,
                              mlir::Location loc, mlir::Value tensor,
                              mlir::Value other);
mlir::CallOp callSwapIndices(mlir::OpBuilder &builder, mlir::ModuleOp &mod,
                             mlir::Location loc, mlir::Value tensor,
                             mlir::Value other);
mlir::CallOp callSwapValues(mlir::OpBuilder &builder, mlir::ModuleOp &mod,
                            mlir::Location loc, mlir::Value tensor,
                            mlir::Value other);

void cleanupIntermediateTensor(mlir::OpBuilder &builder, mlir::ModuleOp &mod,
                               mlir::Location loc, mlir::Value tensor);

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

  ExtensionBlocks(){};
  mlir::LogicalResult
  extractBlocks(mlir::Operation *op, mlir::RegionRange &regions,
                const std::set<mlir::graphblas::YieldKind> &required,
                const std::set<mlir::graphblas::YieldKind> &optional);
};

mlir::LogicalResult populateSemiringRegions(mlir::OpBuilder &builder,
                                            mlir::Location loc,
                                            mlir::StringRef semiring,
                                            mlir::Type valueType,
                                            mlir::RegionRange regions);

mlir::LogicalResult extractApplyOpArgs(mlir::graphblas::ApplyOp op,
                                       mlir::Value &input, mlir::Value &thunk);

#endif // GRAPHBLAS_GRAPHBLASUTILS_H
