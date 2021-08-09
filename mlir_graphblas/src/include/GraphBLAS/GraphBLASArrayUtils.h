#ifndef GRAPHBLAS_GRAPHBLASARRAYUTILS_H
#define GRAPHBLAS_GRAPHBLASARRAYUTILS_H

#include <string>
#include <set>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/APInt.h"

using namespace mlir;

Value computeNumOverlaps(PatternRewriter &rewriter, Value nk,
                         Value fixedIndices, Value fixedIndexStart, Value fixedIndexEnd,
                         Value iterPointers, Value iterIndices,
                         Value maskIndices, Value maskStart, Value maskEnd,
                         Type valueType
                         );

void computeInnerProduct(PatternRewriter &rewriter, Value nk,
                         Value fixedIndices, Value fixedValues, Value fixedIndexStart, Value fixedIndexEnd,
                         Value iterPointers, Value iterIndices, Value iterValues,
                         Value maskIndices, Value maskStart, Value maskEnd,
                         Type valueType, ExtensionBlocks extBlocks,
                         Value outputIndices, Value outputValues, Value indexOffset
                         );

Value computeIndexOverlapSize(PatternRewriter &rewriter, bool intersect,
                              Value aPosStart, Value aPosEnd, Value Ai,
                              Value bPosStart, Value bPosEnd, Value Bi);

Value computeUnionAggregation(PatternRewriter &rewriter, bool intersect, std::string agg, Type valueType,
                              Value aPosStart, Value aPosEnd, Value Ai, Value Ax,
                              Value bPosStart, Value bPosEnd, Value Bi, Value Bx,
                              Value oPosStart, Value Oi, Value Ox);

#endif // GRAPHBLAS_GRAPHBLASARRAYUTILS_H
