#ifndef GRAPHBLAS_GRAPHBLASUTILS_H
#define GRAPHBLAS_GRAPHBLASUTILS_H

#include "GraphBLAS/GraphBLASOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringSet.h"
#include <set>
#include <string>

enum CompressionType { CSR, CSC, EITHER };

// Note about custom ops:
// Operators which have highly customized handling within an operation
// are not listed here. Below are those which can be described independently
// of a specific operation.
// Examples of custom ops: identity, fill, count, probability, argmin, argmax

// Note about vector index usage:
// Whenever (row, column) is indicated below, vectors can also use these
// operators The vector index will be filled in as *both* the row and column.

// Unary operators which take a single argument (value)
static const llvm::StringSet<> unary1{"abs", "ainv", "minv"};

// Unary operators which take 3 arguments (value, row, column)
static const llvm::StringSet<> unary3{"column", "index", "row", "tril", "triu"};

// Binary operators which take 2 arguments (leftValue, rightValue)
static const llvm::StringSet<> binary2{"div",
                                       "eq",
                                       "first",
                                       "ge",
                                       "gt",
                                       "le",
                                       "lt",
                                       "max",
                                       "min",
                                       "minus",
                                       "ne",
                                       "plus",
                                       "second",
                                       "times"
                                       "first",
                                       "pair",
                                       "plus",
                                       "second",
                                       "times"};

// Binary operators which take 4 arguments (leftValue, rightValue, row, column)
static const llvm::StringSet<> binary4{"column", "index", "row"};

// Binary operators which take 5 arguments (leftValue, rightValue, row, column,
// overlap) This is only for binary operators used in a semiring during matrix
// multiplication
static const llvm::StringSet<> binary5{"overlapi"};

// Monoids always take 2 arguments (leftValue, rightValue)
static const llvm::StringSet<> monoid2{"any", "min", "plus"};

// TODO: create allUnary, allBinary, allMonoid

static const llvm::StringSet<> supportedForUnion{"first", "max",    "min",
                                                 "plus",  "second", "times"};

static const llvm::StringSet<> supportedForIntersect{
    "div", "first", "max", "min", "minus", "plus", "second", "times"};
//"eq", "ne", "lt", "le", "gt", "ge"};

static const llvm::StringSet<> supportedForUpdate{"max", "min", "plus",
                                                  "times"};

static const llvm::StringSet<> supportedForReduce{
    // List custom aggregators first
    "argmax", "argmin", "count",
    // Normal aggregators in alphabetical order
    "plus"};

static const llvm::StringSet<> supportedForSelect{
    // List custom selectors first
    "probability",
    // Normal selectors in alphabetical order
    "ge", "gt", "tril", "triu"};

static const llvm::StringSet<> supportedForApply{
    // List custom operators first
    "fill", "identity",
    // Normal operators in alphabetical order
    "abs", "ainv", "column", "div", "index", "min", "minv", "row"};

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
  mlir::Block *transformInA = nullptr; // not used
  mlir::Block *transformInB = nullptr; // not used
  mlir::Block *transformOut = nullptr;
  mlir::Block *selectInA = nullptr; // not used
  mlir::Block *selectInB = nullptr; // not used
  mlir::Block *selectOut = nullptr;
  mlir::Block *addIdentity = nullptr;
  mlir::Block *add = nullptr;
  mlir::Block *mult = nullptr;
  mlir::Block *aggIdentity = nullptr;
  mlir::Block *agg = nullptr;
  mlir::Block *accumulate = nullptr;

  ExtensionBlocks(){};
  mlir::LogicalResult
  extractBlocks(mlir::Operation *op, mlir::RegionRange &regions,
                const std::set<mlir::graphblas::YieldKind> &required,
                const std::set<mlir::graphblas::YieldKind> &optional);
};

mlir::LogicalResult populateUnary(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::StringRef unaryOp, mlir::Type valueType,
                                  mlir::RegionRange regions,
                                  mlir::graphblas::YieldKind yieldKind);

mlir::LogicalResult populateBinary(mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::StringRef binaryOp,
                                   mlir::Type valueType,
                                   mlir::RegionRange regions,
                                   mlir::graphblas::YieldKind yieldKind);

mlir::LogicalResult populateMonoid(mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::StringRef monoidOp,
                                   mlir::Type valueType,
                                   mlir::RegionRange regions,
                                   mlir::graphblas::YieldKind yieldIdentity,
                                   mlir::graphblas::YieldKind yieldKind);

mlir::LogicalResult populateSemiring(mlir::OpBuilder &builder,
                                     mlir::Location loc,
                                     mlir::StringRef semiringOp,
                                     mlir::Type valueType,
                                     mlir::RegionRange regions);

mlir::LogicalResult extractApplyOpArgs(mlir::graphblas::ApplyOp op,
                                       mlir::Value &input, mlir::Value &thunk);

#endif // GRAPHBLAS_GRAPHBLASUTILS_H
