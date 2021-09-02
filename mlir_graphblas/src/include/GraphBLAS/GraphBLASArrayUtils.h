#ifndef GRAPHBLAS_GRAPHBLASARRAYUTILS_H
#define GRAPHBLAS_GRAPHBLASARRAYUTILS_H

#include "GraphBLAS/GraphBLASUtils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/APInt.h"
#include <set>
#include <string>

using namespace mlir;

ValueRange buildMaskComplement(PatternRewriter &rewriter, Value fullSize,
                               Value maskIndices, Value maskStart,
                               Value maskEnd);

Value computeNumOverlaps(PatternRewriter &rewriter, Value nk,
                         Value fixedIndices, Value fixedIndexStart,
                         Value fixedIndexEnd, Value iterPointers,
                         Value iterIndices, Value maskIndices, Value maskStart,
                         Value maskEnd, Type valueType);

void computeInnerProduct(PatternRewriter &rewriter, Value nk,
                         Value fixedIndices, Value fixedValues,
                         Value fixedIndexStart, Value fixedIndexEnd,
                         Value iterPointers, Value iterIndices,
                         Value iterValues, Value maskIndices, Value maskStart,
                         Value maskEnd, Type valueType,
                         ExtensionBlocks extBlocks, Value outputIndices,
                         Value outputValues, Value indexOffset,
                         bool swapMultOps);

Value computeIndexOverlapSize(PatternRewriter &rewriter, bool intersect,
                              Value aPosStart, Value aPosEnd, Value Ai,
                              Value bPosStart, Value bPosEnd, Value Bi);

Value computeUnionAggregation(PatternRewriter &rewriter, bool intersect,
                              std::string agg, Type valueType, Value aPosStart,
                              Value aPosEnd, Value Ai, Value Ax,
                              Value bPosStart, Value bPosEnd, Value Bi,
                              Value Bx, Value oPosStart, Value Oi, Value Ox);

void computeVectorElementWise(PatternRewriter &rewriter, ModuleOp module,
                              Value lhs, Value rhs, Value output,
                              std::string op, bool intersect);
void computeMatrixElementWise(PatternRewriter &rewriter, ModuleOp module,
                              Value lhs, Value rhs, Value output,
                              std::string op, bool intersect);

#endif // GRAPHBLAS_GRAPHBLASARRAYUTILS_H
