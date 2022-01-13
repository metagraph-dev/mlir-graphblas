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

enum EwiseBehavior {
  UNION,
  INTERSECT,
  MASK,
  MASK_COMPLEMENT,
};

ValueRange buildMaskComplement(PatternRewriter &rewriter, Location loc,
                               Value fullSize, Value maskIndices,
                               Value maskStart, Value maskEnd);

ValueRange sparsifyDensePointers(PatternRewriter &rewriter, Location loc,
                                 Value size, Value pointers);

ValueRange buildIndexOverlap(PatternRewriter &rewriter, Location loc,
                             Value aSize, Value a, Value bSize, Value b);

Value computeNumOverlaps(PatternRewriter &rewriter, Location loc, Value nk,
                         Value fixedIndices, Value fixedIndexStart,
                         Value fixedIndexEnd, Value iterPointers,
                         Value iterIndices, Value maskIndices, Value maskStart,
                         Value maskEnd, Type valueType);

void computeInnerProduct(PatternRewriter &rewriter, Location loc, Value nk,
                         Value fixedRowIndex, Value fixedIndices,
                         Value fixedValues, Value fixedIndexStart,
                         Value fixedIndexEnd, Value iterPointers,
                         Value iterIndices, Value iterValues, Value maskIndices,
                         Value maskStart, Value maskEnd, Type valueType,
                         ExtensionBlocks extBlocks, Value outputIndices,
                         Value outputValues, Value indexOffset,
                         bool swapMultOps);

Value computeIndexOverlapSize(PatternRewriter &rewriter, Location loc,
                              bool intersect, Value aPosStart, Value aPosEnd,
                              Value Ai, Value bPosStart, Value bPosEnd,
                              Value Bi);

Value computeUnionAggregation(PatternRewriter &rewriter, Location loc,
                              bool intersect, Block *binaryBlock,
                              Type valueType, Value aPosStart, Value aPosEnd,
                              Value Ai, Value Ax, Value bPosStart,
                              Value bPosEnd, Value Bi, Value Bx,
                              Value oPosStart, Value Oi, Value Ox);

void computeVectorElementWise(PatternRewriter &rewriter, Location loc,
                              ModuleOp module, Value lhs, Value rhs,
                              Value output, Block *binaryBlock,
                              EwiseBehavior behavior);
void computeMatrixElementWise(PatternRewriter &rewriter, Location loc,
                              ModuleOp module, Value lhs, Value rhs,
                              Value output, Block *binaryBlock,
                              EwiseBehavior behavior);

#endif // GRAPHBLAS_GRAPHBLASARRAYUTILS_H
