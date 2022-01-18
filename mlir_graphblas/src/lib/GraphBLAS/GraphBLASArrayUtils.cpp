#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include "GraphBLAS/GraphBLASArrayUtils.h"
#include "GraphBLAS/GraphBLASOps.h"
#include "GraphBLAS/GraphBLASUtils.h"

using namespace ::mlir;

ValueRange buildMaskComplement(PatternRewriter &rewriter, Location loc,
                               Value fullSize, Value maskIndices,
                               Value maskStart, Value maskEnd) {
  // Operates on a vector or on a single row/column of a matrix
  //
  // Returns:
  // 1. a memref containing the indices of the mask complement
  // 2. the size of the memref

  // Types used in this function
  Type indexType = rewriter.getIndexType();
  Type int64Type = rewriter.getIntegerType(64);
  MemRefType memref1DI64Type = MemRefType::get({-1}, int64Type);

  // Initial constants
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  // Compute the size of the complemented mask
  Value maskSize = rewriter.create<arith::SubIOp>(loc, maskEnd, maskStart);
  Value compSize = rewriter.create<arith::SubIOp>(loc, fullSize, maskSize);

  // Allocate memref
  Value compIndices =
      rewriter.create<memref::AllocOp>(loc, memref1DI64Type, compSize);

  // Force firstMaskIndex to be unreachable for the case of empty row
  // This will cause every idx in the loop to be included in the output
  Value rowIsEmpty = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, maskStart, maskEnd);

  scf::IfOp if_empty_row =
      rewriter.create<scf::IfOp>(loc, indexType, rowIsEmpty, true);
  {
    rewriter.setInsertionPointToStart(if_empty_row.thenBlock());
    rewriter.create<scf::YieldOp>(loc, fullSize);
  }
  {
    rewriter.setInsertionPointToStart(if_empty_row.elseBlock());
    Value startIndex64 =
        rewriter.create<memref::LoadOp>(loc, maskIndices, maskStart);
    Value startIndex =
        rewriter.create<arith::IndexCastOp>(loc, startIndex64, indexType);
    rewriter.create<scf::YieldOp>(loc, startIndex);
    rewriter.setInsertionPointAfter(if_empty_row);
  }
  Value firstMaskIndex = if_empty_row.getResult(0);

  // Populate memref
  scf::ForOp loop = rewriter.create<scf::ForOp>(
      loc, c0, fullSize, c1, ValueRange{c0, maskStart, firstMaskIndex});
  Value idx = loop.getInductionVar();
  Value compPos = loop.getLoopBody().getArgument(1);
  Value maskPos = loop.getLoopBody().getArgument(2);
  Value maskIndex = loop.getLoopBody().getArgument(3);
  {
    rewriter.setInsertionPointToStart(loop.getBody());
    Value maskMatch = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, idx, maskIndex);
    scf::IfOp if_match = rewriter.create<scf::IfOp>(
        loc, TypeRange{indexType, indexType, indexType}, maskMatch, true);
    {
      rewriter.setInsertionPointToStart(if_match.thenBlock());
      // Increment maskPos
      Value nextMaskPos = rewriter.create<arith::AddIOp>(loc, maskPos, c1);
      // Update maskIndex (unless we're at the end)
      Value isAtEnd = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::uge, nextMaskPos, maskEnd);
      scf::IfOp if_atEnd =
          rewriter.create<scf::IfOp>(loc, indexType, isAtEnd, true);
      {
        rewriter.setInsertionPointToStart(if_atEnd.thenBlock());
        rewriter.create<scf::YieldOp>(loc, maskIndex);
      }
      {
        rewriter.setInsertionPointToStart(if_atEnd.elseBlock());
        Value newMaskIndex64 =
            rewriter.create<memref::LoadOp>(loc, maskIndices, nextMaskPos);
        Value newMaskIndex =
            rewriter.create<arith::IndexCastOp>(loc, newMaskIndex64, indexType);
        rewriter.create<scf::YieldOp>(loc, newMaskIndex);
      }
      rewriter.setInsertionPointAfter(if_atEnd);
      rewriter.create<scf::YieldOp>(
          loc, ValueRange{compPos, nextMaskPos, if_atEnd.getResult(0)});
    }
    {
      rewriter.setInsertionPointToStart(if_match.elseBlock());
      // Add i to compIndices
      Value idx64 = rewriter.create<arith::IndexCastOp>(loc, idx, int64Type);
      rewriter.create<memref::StoreOp>(loc, idx64, compIndices, compPos);
      // Increment compPos
      Value nextCompPos = rewriter.create<arith::AddIOp>(loc, compPos, c1);
      rewriter.create<scf::YieldOp>(
          loc, ValueRange{nextCompPos, maskPos, maskIndex});
    }
    rewriter.setInsertionPointAfter(if_match);
    rewriter.create<scf::YieldOp>(loc, if_match.getResults());
    rewriter.setInsertionPointAfter(loop);
  }

  return ValueRange{compIndices, compSize};
}

ValueRange sparsifyDensePointers(PatternRewriter &rewriter, Location loc,
                                 Value size, Value pointers) {
  // From a memref of dense pointers (length size+1),
  // Returns:
  // 1. a memref containing the indices of the non-empty rows (or columns)
  // 2. the size of the memref

  // Types used in this function
  Type indexType = rewriter.getIndexType();
  Type int64Type = rewriter.getIntegerType(64);
  MemRefType memref1DI64Type = MemRefType::get({-1}, int64Type);

  // Initial constants
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  // Step 1: compute the number of non-empty rows
  scf::ForOp nnzCountLoop =
      rewriter.create<scf::ForOp>(loc, c0, size, c1, ValueRange{c0});
  {
    rewriter.setInsertionPointToStart(nnzCountLoop.getBody());
    Value count = nnzCountLoop.getLoopBody().getArgument(1);
    Value rowIndex = nnzCountLoop.getInductionVar();
    Value nextRowIndex = rewriter.create<arith::AddIOp>(loc, rowIndex, c1);
    Value firstPtr64 = rewriter.create<memref::LoadOp>(loc, pointers, rowIndex);
    Value secondPtr64 =
        rewriter.create<memref::LoadOp>(loc, pointers, nextRowIndex);
    Value rowIsEmpty = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, firstPtr64, secondPtr64);
    scf::IfOp ifRowIsEmptyBlock =
        rewriter.create<scf::IfOp>(loc, TypeRange{indexType}, rowIsEmpty, true);
    {
      rewriter.setInsertionPointToStart(ifRowIsEmptyBlock.thenBlock());
      rewriter.create<scf::YieldOp>(loc, ValueRange{count});
    }
    {
      rewriter.setInsertionPointToStart(ifRowIsEmptyBlock.elseBlock());
      Value count_plus1 = rewriter.create<arith::AddIOp>(loc, count, c1);
      rewriter.create<scf::YieldOp>(loc, ValueRange{count_plus1});
    }
    rewriter.setInsertionPointAfter(ifRowIsEmptyBlock);
    Value newCount = ifRowIsEmptyBlock.getResult(0);
    rewriter.create<scf::YieldOp>(loc, ValueRange{newCount});
    rewriter.setInsertionPointAfter(nnzCountLoop);
  }
  Value nnz = nnzCountLoop.getResult(0);

  // Step 2: build the indices of the non-empty rows
  Value indices = rewriter.create<memref::AllocOp>(loc, memref1DI64Type, nnz);
  scf::ForOp nnzIdxLoop =
      rewriter.create<scf::ForOp>(loc, c0, size, c1, ValueRange{c0});
  {
    rewriter.setInsertionPointToStart(nnzIdxLoop.getBody());
    Value pos = nnzIdxLoop.getLoopBody().getArgument(1);
    Value rowIndex = nnzIdxLoop.getInductionVar();
    Value nextRowIndex = rewriter.create<arith::AddIOp>(loc, rowIndex, c1);
    Value firstPtr64 = rewriter.create<memref::LoadOp>(loc, pointers, rowIndex);
    Value secondPtr64 =
        rewriter.create<memref::LoadOp>(loc, pointers, nextRowIndex);
    Value rowIsEmpty = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, firstPtr64, secondPtr64);
    scf::IfOp ifRowIsEmptyBlock =
        rewriter.create<scf::IfOp>(loc, TypeRange{indexType}, rowIsEmpty, true);
    {
      rewriter.setInsertionPointToStart(ifRowIsEmptyBlock.thenBlock());
      rewriter.create<scf::YieldOp>(loc, ValueRange{pos});
    }
    {
      rewriter.setInsertionPointToStart(ifRowIsEmptyBlock.elseBlock());
      Value rowIndex64 =
          rewriter.create<arith::IndexCastOp>(loc, rowIndex, int64Type);
      rewriter.create<memref::StoreOp>(loc, rowIndex64, indices, pos);
      Value pos_plus1 = rewriter.create<arith::AddIOp>(loc, pos, c1);
      rewriter.create<scf::YieldOp>(loc, ValueRange{pos_plus1});
    }
    rewriter.setInsertionPointAfter(ifRowIsEmptyBlock);
    Value newPos = ifRowIsEmptyBlock.getResult(0);
    rewriter.create<scf::YieldOp>(loc, ValueRange{newPos});
    rewriter.setInsertionPointAfter(nnzIdxLoop);
  }

  return ValueRange{indices, nnz};
}

ValueRange buildIndexOverlap(PatternRewriter &rewriter, Location loc,
                             Value aSize, Value a, Value bSize, Value b) {
  // Takes two memrefs containing a list of indices and performs an intersection
  // Returns:
  // 1. a memref containing the overlapping indices
  // 2. the size of the memref

  // Types used in this function
  Type boolType = rewriter.getIntegerType(1);
  Type indexType = rewriter.getIndexType();
  Type int64Type = rewriter.getIntegerType(64);
  MemRefType memref1DI64Type = MemRefType::get({-1}, int64Type);

  // Initial constants
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value ci0 = rewriter.create<arith::ConstantIntOp>(loc, 0, int64Type);
  Value cfalse = rewriter.create<arith::ConstantIntOp>(loc, 0, boolType);
  Value ctrue = rewriter.create<arith::ConstantIntOp>(loc, 1, boolType);

  // Allocate a memref matching the smaller size
  // This will sacrifice memory for less computation
  Value aIsSmaller = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ult, aSize, bSize);
  Value smallerSize = rewriter.create<SelectOp>(loc, aIsSmaller, aSize, bSize);
  Value output =
      rewriter.create<memref::AllocOp>(loc, memref1DI64Type, smallerSize);

  // Find matching indices
  // While Loop (exit when either array is exhausted)
  scf::WhileOp whileLoop = rewriter.create<scf::WhileOp>(
      loc,
      TypeRange{indexType, indexType, indexType, int64Type, int64Type, boolType,
                boolType},
      ValueRange{c0, c0, c0, ci0, ci0, ctrue, ctrue});
  Block *before =
      rewriter.createBlock(&whileLoop.getBefore(), {},
                           TypeRange{indexType, indexType, indexType, int64Type,
                                     int64Type, boolType, boolType});
  Block *after =
      rewriter.createBlock(&whileLoop.getAfter(), {},
                           TypeRange{indexType, indexType, indexType, int64Type,
                                     int64Type, boolType, boolType});
  // "while" portion of the loop
  rewriter.setInsertionPointToStart(&whileLoop.getBefore().front());
  Value posA = before->getArgument(0);
  Value posB = before->getArgument(1);
  Value validPosA = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ult, posA, aSize);
  Value validPosB = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ult, posB, bSize);
  Value continueLoop =
      rewriter.create<arith::AndIOp>(loc, validPosA, validPosB);
  rewriter.create<scf::ConditionOp>(loc, continueLoop, before->getArguments());

  // "do" portion of while loop
  rewriter.setInsertionPointToStart(&whileLoop.getAfter().front());
  posA = after->getArgument(0);
  posB = after->getArgument(1);
  Value posO = after->getArgument(2);
  Value idxA = after->getArgument(3);
  Value idxB = after->getArgument(4);
  Value needsUpdateA = after->getArgument(5);
  Value needsUpdateB = after->getArgument(6);

  // Update A's index based on flag
  scf::IfOp if_updateA =
      rewriter.create<scf::IfOp>(loc, int64Type, needsUpdateA, true);
  {
    rewriter.setInsertionPointToStart(if_updateA.thenBlock());
    Value updatedIdxA = rewriter.create<memref::LoadOp>(loc, a, posA);
    rewriter.create<scf::YieldOp>(loc, updatedIdxA);
  }
  {
    rewriter.setInsertionPointToStart(if_updateA.elseBlock());
    rewriter.create<scf::YieldOp>(loc, idxA);
  }
  rewriter.setInsertionPointAfter(if_updateA);

  // Update B's index based on flag
  scf::IfOp if_updateB =
      rewriter.create<scf::IfOp>(loc, int64Type, needsUpdateB, true);
  {
    rewriter.setInsertionPointToStart(if_updateB.thenBlock());
    Value updatedIdxB = rewriter.create<memref::LoadOp>(loc, b, posB);
    rewriter.create<scf::YieldOp>(loc, updatedIdxB);
  }
  {
    rewriter.setInsertionPointToStart(if_updateB.elseBlock());
    rewriter.create<scf::YieldOp>(loc, idxB);
  }
  rewriter.setInsertionPointAfter(if_updateB);

  Value newIdxA = if_updateA.getResult(0);
  Value newIdxB = if_updateB.getResult(0);
  Value idxA_lt_idxB = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ult, newIdxA, newIdxB);
  Value idxA_gt_idxB = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ugt, newIdxA, newIdxB);
  Value posAplus1 = rewriter.create<arith::AddIOp>(loc, posA, c1);
  Value posBplus1 = rewriter.create<arith::AddIOp>(loc, posB, c1);

  Value posOplus1 = rewriter.create<arith::AddIOp>(loc, posO, c1);

  scf::IfOp if_onlyA = rewriter.create<scf::IfOp>(
      loc, TypeRange{indexType, indexType, indexType, boolType, boolType},
      idxA_lt_idxB, true);
  {
    rewriter.setInsertionPointToStart(if_onlyA.thenBlock());
    rewriter.create<scf::YieldOp>(
        loc, ValueRange{posAplus1, posB, posO, ctrue, cfalse});
  }
  {
    rewriter.setInsertionPointToStart(if_onlyA.elseBlock());
    scf::IfOp if_onlyB = rewriter.create<scf::IfOp>(
        loc, TypeRange{indexType, indexType, indexType, boolType, boolType},
        idxA_gt_idxB, true);
    {
      rewriter.setInsertionPointToStart(if_onlyB.thenBlock());
      rewriter.create<scf::YieldOp>(
          loc, ValueRange{posA, posBplus1, posO, cfalse, ctrue});
    }
    {
      rewriter.setInsertionPointToStart(if_onlyB.elseBlock());
      // At this point, we know newIdxA == newIdxB
      rewriter.create<memref::StoreOp>(loc, newIdxA, output, posO);
      rewriter.create<scf::YieldOp>(
          loc, ValueRange{posAplus1, posBplus1, posOplus1, ctrue, ctrue});
    }
    rewriter.setInsertionPointAfter(if_onlyB);
    rewriter.create<scf::YieldOp>(loc, if_onlyB.getResults());
  }
  rewriter.setInsertionPointAfter(if_onlyA);
  Value newPosA = if_onlyA.getResult(0);
  Value newPosB = if_onlyA.getResult(1);
  Value newPosO = if_onlyA.getResult(2);
  needsUpdateA = if_onlyA.getResult(3);
  needsUpdateB = if_onlyA.getResult(4);

  rewriter.create<scf::YieldOp>(loc, ValueRange{newPosA, newPosB, newPosO,
                                                newIdxA, newIdxB, needsUpdateA,
                                                needsUpdateB});
  rewriter.setInsertionPointAfter(whileLoop);

  Value finalPosO = whileLoop.getResult(2);
  return ValueRange{output, finalPosO};
}

Value computeNumOverlaps(PatternRewriter &rewriter, Location loc, Value nk,
                         Value fixedIndices, Value fixedIndexStart,
                         Value fixedIndexEnd, Value iterPointers,
                         Value iterIndices,
                         // If no mask is used, set maskIndices to nullptr, and
                         // provide maskStart=c0 and maskEnd=len(iterPointers)-1
                         Value maskIndices, Value maskStart, Value maskEnd,
                         Type valueType) {
  // Types used in this function
  Type indexType = rewriter.getIndexType();
  Type int64Type = rewriter.getIntegerType(64);
  Type boolType = rewriter.getI1Type();
  MemRefType memref1DBoolType = MemRefType::get({-1}, boolType);

  // Initial constants
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value ci0 = rewriter.create<arith::ConstantIntOp>(loc, 0, int64Type);
  Value ci1 = rewriter.create<arith::ConstantIntOp>(loc, 1, int64Type);
  Value ctrue = rewriter.create<arith::ConstantIntOp>(loc, 1, boolType);
  Value cfalse = rewriter.create<arith::ConstantIntOp>(loc, 0, boolType);

  // Construct a dense array indicating valid kk positions within fixed index
  Value kvec_i1 = rewriter.create<memref::AllocOp>(loc, memref1DBoolType, nk);
  rewriter.create<linalg::FillOp>(loc, cfalse, kvec_i1);
  scf::ParallelOp colLoop1 =
      rewriter.create<scf::ParallelOp>(loc, fixedIndexStart, fixedIndexEnd, c1);
  Value jj = colLoop1.getInductionVars()[0];
  rewriter.setInsertionPointToStart(colLoop1.getBody());
  Value col64 = rewriter.create<memref::LoadOp>(loc, fixedIndices, jj);
  Value col = rewriter.create<arith::IndexCastOp>(loc, col64, indexType);
  rewriter.create<memref::StoreOp>(loc, ctrue, kvec_i1, col);
  rewriter.setInsertionPointAfter(colLoop1);
  // Loop thru all columns; count number of resulting nonzeros in the row
  if (maskIndices != nullptr) {
    colLoop1 =
        rewriter.create<scf::ParallelOp>(loc, maskStart, maskEnd, c1, ci0);
    Value mm = colLoop1.getInductionVars()[0];
    rewriter.setInsertionPointToStart(colLoop1.getBody());
    col64 = rewriter.create<memref::LoadOp>(loc, maskIndices, mm);
    col = rewriter.create<arith::IndexCastOp>(loc, col64, indexType);
  } else {
    colLoop1 =
        rewriter.create<scf::ParallelOp>(loc, maskStart, maskEnd, c1, ci0);
    col = colLoop1.getInductionVars()[0];
    rewriter.setInsertionPointToStart(colLoop1.getBody());
  }
  Value colPlus1 = rewriter.create<arith::AddIOp>(loc, col, c1);
  Value rowStart64 = rewriter.create<memref::LoadOp>(loc, iterPointers, col);
  Value rowEnd64 = rewriter.create<memref::LoadOp>(loc, iterPointers, colPlus1);
  Value cmpRowSame = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, rowStart64, rowEnd64);
  // Find overlap in column indices with kvec
  scf::IfOp ifBlock_overlap =
      rewriter.create<scf::IfOp>(loc, int64Type, cmpRowSame, true);
  // if cmpRowSame
  rewriter.setInsertionPointToStart(ifBlock_overlap.thenBlock());
  rewriter.create<scf::YieldOp>(loc, ci0);
  // else
  rewriter.setInsertionPointToStart(ifBlock_overlap.elseBlock());
  // Walk thru the indices; on a match yield 1, else yield 0
  scf::WhileOp whileLoop =
      rewriter.create<scf::WhileOp>(loc, int64Type, rowStart64);
  Block *before = rewriter.createBlock(&whileLoop.getBefore(), {}, int64Type);
  Block *after = rewriter.createBlock(&whileLoop.getAfter(), {}, int64Type);
  Value ii64 = before->getArgument(0);
  rewriter.setInsertionPointToStart(&whileLoop.getBefore().front());
  // Check if ii >= rowEnd
  Value cmpEndReached = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::uge, ii64, rowEnd64);
  scf::IfOp ifBlock_continueSearch = rewriter.create<scf::IfOp>(
      loc, TypeRange{boolType, int64Type}, cmpEndReached, true);
  // if cmpEndReached
  rewriter.setInsertionPointToStart(ifBlock_continueSearch.thenBlock());
  rewriter.create<scf::YieldOp>(loc, ValueRange{cfalse, ci0});
  // else
  rewriter.setInsertionPointToStart(ifBlock_continueSearch.elseBlock());
  // Check if row has a match in kvec
  Value ii = rewriter.create<arith::IndexCastOp>(loc, ii64, indexType);
  Value kk64 = rewriter.create<memref::LoadOp>(loc, iterIndices, ii);
  Value kk = rewriter.create<arith::IndexCastOp>(loc, kk64, indexType);
  Value cmpPair = rewriter.create<memref::LoadOp>(loc, kvec_i1, kk);
  Value cmpResult0 = rewriter.create<SelectOp>(loc, cmpPair, cfalse, ctrue);
  Value cmpResult1 = rewriter.create<SelectOp>(loc, cmpPair, ci1, ii64);
  rewriter.create<scf::YieldOp>(loc, ValueRange{cmpResult0, cmpResult1});
  // end if cmpEndReached
  rewriter.setInsertionPointAfter(ifBlock_continueSearch);
  Value continueSearch = ifBlock_continueSearch.getResult(0);
  Value valToSend = ifBlock_continueSearch.getResult(1);
  rewriter.create<scf::ConditionOp>(loc, continueSearch, valToSend);
  // "do" portion of while loop
  rewriter.setInsertionPointToStart(&whileLoop.getAfter().front());
  Value iiPrev = after->getArgument(0);
  Value iiNext = rewriter.create<arith::AddIOp>(loc, iiPrev, ci1);
  rewriter.create<scf::YieldOp>(loc, iiNext);
  rewriter.setInsertionPointAfter(whileLoop);
  Value res = whileLoop.getResult(0);
  rewriter.create<scf::YieldOp>(loc, res);
  // end if cmpRowSame
  rewriter.setInsertionPointAfter(ifBlock_overlap);
  Value overlap = ifBlock_overlap.getResult(0);
  scf::ReduceOp reducer = rewriter.create<scf::ReduceOp>(loc, overlap);
  Value lhs = reducer.getRegion().getArgument(0);
  Value rhs = reducer.getRegion().getArgument(1);
  rewriter.setInsertionPointToStart(&reducer.getRegion().front());
  Value z = rewriter.create<arith::AddIOp>(loc, lhs, rhs);
  rewriter.create<scf::ReduceReturnOp>(loc, z);
  // end col loop
  rewriter.setInsertionPointAfter(colLoop1);
  Value total = colLoop1.getResult(0);
  rewriter.create<memref::DeallocOp>(loc, kvec_i1);
  return total;
}

void computeInnerProduct(PatternRewriter &rewriter, Location loc, Value nk,
                         Value fixedRowIndex, Value fixedIndices,
                         Value fixedValues, Value fixedIndexStart,
                         Value fixedIndexEnd, Value iterPointers,
                         Value iterIndices, Value iterValues,
                         // If no mask is used, set maskIndices to nullptr, and
                         // provide maskStart=c0 and maskEnd=len(iterPointers)-1
                         Value maskIndices, Value maskStart, Value maskEnd,
                         Type valueType, ExtensionBlocks extBlocks,
                         Value outputIndices, Value outputValues,
                         Value indexOffset, bool swapMultOps) {
  // Types used in this function
  Type indexType = rewriter.getIndexType();
  Type int64Type = rewriter.getIntegerType(64);
  Type boolType = rewriter.getI1Type();
  MemRefType memref1DBoolType = MemRefType::get({-1}, boolType);
  MemRefType memref1DValueType = MemRefType::get({-1}, valueType);

  // Initial constants
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value ctrue = rewriter.create<arith::ConstantIntOp>(loc, 1, boolType);
  Value cfalse = rewriter.create<arith::ConstantIntOp>(loc, 0, boolType);

  // Construct a dense array of row values
  Value kvec = rewriter.create<memref::AllocOp>(loc, memref1DValueType, nk);
  Value kvec_i1 = rewriter.create<memref::AllocOp>(loc, memref1DBoolType, nk);
  rewriter.create<linalg::FillOp>(loc, cfalse, kvec_i1);
  scf::ParallelOp colLoop3p =
      rewriter.create<scf::ParallelOp>(loc, fixedIndexStart, fixedIndexEnd, c1);
  Value jj = colLoop3p.getInductionVars()[0];
  rewriter.setInsertionPointToStart(colLoop3p.getBody());
  Value fixedJ64 = rewriter.create<memref::LoadOp>(loc, fixedIndices, jj);
  Value fixedJ = rewriter.create<arith::IndexCastOp>(loc, fixedJ64, indexType);
  rewriter.create<memref::StoreOp>(loc, ctrue, kvec_i1, fixedJ);
  Value val = rewriter.create<memref::LoadOp>(loc, fixedValues, jj);
  rewriter.create<memref::StoreOp>(loc, val, kvec, fixedJ);

  // end col loop 3p
  rewriter.setInsertionPointAfter(colLoop3p);

  Value col64, col;
  scf::ForOp colLoop3f;
  if (maskIndices != nullptr) {
    colLoop3f = rewriter.create<scf::ForOp>(loc, maskStart, maskEnd, c1, c0);
    Value mm = colLoop3f.getInductionVar();
    rewriter.setInsertionPointToStart(colLoop3f.getBody());
    col64 = rewriter.create<memref::LoadOp>(loc, maskIndices, mm);
    col = rewriter.create<arith::IndexCastOp>(loc, col64, indexType);
  } else {
    colLoop3f = rewriter.create<scf::ForOp>(loc, maskStart, maskEnd, c1, c0);
    col = colLoop3f.getInductionVar();
    rewriter.setInsertionPointToStart(colLoop3f.getBody());
    col64 = rewriter.create<arith::IndexCastOp>(loc, col, int64Type);
  }

  Value offset = colLoop3f.getLoopBody().getArgument(1);
  Value colPlus1 = rewriter.create<arith::AddIOp>(loc, col, c1);
  Value iStart64 = rewriter.create<memref::LoadOp>(loc, iterPointers, col);
  Value iEnd64 = rewriter.create<memref::LoadOp>(loc, iterPointers, colPlus1);
  Value iStart = rewriter.create<arith::IndexCastOp>(loc, iStart64, indexType);
  Value iEnd = rewriter.create<arith::IndexCastOp>(loc, iEnd64, indexType);

  // insert add identity block
  rewriter.mergeBlocks(extBlocks.addIdentity, rewriter.getBlock(), {});
  graphblas::YieldOp addIdentityYield =
      llvm::dyn_cast_or_null<graphblas::YieldOp>(
          rewriter.getBlock()->getTerminator());
  Value addIdentity = addIdentityYield.values().front();
  rewriter.eraseOp(addIdentityYield);

  scf::ForOp kLoop = rewriter.create<scf::ForOp>(
      loc, iStart, iEnd, c1, ValueRange{addIdentity, cfalse});
  Value ii = kLoop.getInductionVar();
  Value curr = kLoop.getLoopBody().getArgument(1);
  Value alive = kLoop.getLoopBody().getArgument(2);
  rewriter.setInsertionPointToStart(kLoop.getBody());

  Value kk64 = rewriter.create<memref::LoadOp>(loc, iterIndices, ii);
  Value kk = rewriter.create<arith::IndexCastOp>(loc, kk64, indexType);
  Value cmpPair = rewriter.create<memref::LoadOp>(loc, kvec_i1, kk);
  scf::IfOp ifBlock_cmpPair = rewriter.create<scf::IfOp>(
      loc, TypeRange{valueType, boolType}, cmpPair, true);
  // if cmpPair
  rewriter.setInsertionPointToStart(ifBlock_cmpPair.thenBlock());

  Value aVal = rewriter.create<memref::LoadOp>(loc, kvec, kk);
  Value bVal = rewriter.create<memref::LoadOp>(loc, iterValues, ii);

  // insert multiply operation block
  ValueRange injectVals;
  if (swapMultOps)
    injectVals = ValueRange{bVal, aVal, col, fixedRowIndex, kk};
  else
    injectVals = ValueRange{aVal, bVal, fixedRowIndex, col, kk};
  rewriter.mergeBlocks(
      extBlocks.mult, rewriter.getBlock(),
      injectVals.slice(0, extBlocks.mult->getArguments().size()));
  // NOTE: Need to do this after merge, in case the yield is one of the block
  // arguments, as is the case with "first" and "second" binops
  graphblas::YieldOp multYield = llvm::dyn_cast_or_null<graphblas::YieldOp>(
      rewriter.getBlock()->getTerminator());
  Value multResult = multYield.values().front();
  rewriter.eraseOp(multYield);

  // insert add operation block
  rewriter.mergeBlocks(extBlocks.add, rewriter.getBlock(), {curr, multResult});
  graphblas::YieldOp addYield = llvm::dyn_cast_or_null<graphblas::YieldOp>(
      rewriter.getBlock()->getTerminator());
  Value addResult = addYield.values().front();
  rewriter.eraseOp(addYield);

  rewriter.create<scf::YieldOp>(loc, ValueRange{addResult, ctrue});

  // else
  rewriter.setInsertionPointToStart(ifBlock_cmpPair.elseBlock());
  rewriter.create<scf::YieldOp>(loc, ValueRange{curr, alive});

  // end if cmpPair
  rewriter.setInsertionPointAfter(ifBlock_cmpPair);
  Value newCurr = ifBlock_cmpPair.getResult(0);
  Value newAlive = ifBlock_cmpPair.getResult(1);
  rewriter.create<scf::YieldOp>(loc, ValueRange{newCurr, newAlive});

  // end k loop
  rewriter.setInsertionPointAfter(kLoop);

  Value total = kLoop.getResult(0);
  Value notEmpty = kLoop.getResult(1);

  scf::IfOp ifBlock_newOffset =
      rewriter.create<scf::IfOp>(loc, indexType, notEmpty, true);
  // if not empty
  rewriter.setInsertionPointToStart(ifBlock_newOffset.thenBlock());

  // Store total in Cx
  Value cjPos = rewriter.create<arith::AddIOp>(loc, indexOffset, offset);
  rewriter.create<memref::StoreOp>(loc, col64, outputIndices, cjPos);

  // Does total need to be transformed?
  if (extBlocks.transformOut) {
    rewriter.mergeBlocks(extBlocks.transformOut, rewriter.getBlock(), {total});
    graphblas::YieldOp yield = llvm::dyn_cast_or_null<graphblas::YieldOp>(
        rewriter.getBlock()->getTerminator());
    Value transformResult = yield.values().front();

    rewriter.create<memref::StoreOp>(loc, transformResult, outputValues, cjPos);
    rewriter.eraseOp(yield);
  } else {
    // write total as-is
    rewriter.create<memref::StoreOp>(loc, total, outputValues, cjPos);
  }

  // Increment offset
  Value offsetPlus1 = rewriter.create<arith::AddIOp>(loc, offset, c1);
  rewriter.create<scf::YieldOp>(loc, offsetPlus1);

  // else
  rewriter.setInsertionPointToStart(ifBlock_newOffset.elseBlock());
  rewriter.create<scf::YieldOp>(loc, offset);

  // end if not empty
  rewriter.setInsertionPointAfter(ifBlock_newOffset);

  Value newOffset = ifBlock_newOffset.getResult(0);
  rewriter.create<scf::YieldOp>(loc, newOffset);

  // end col loop 3f
  rewriter.setInsertionPointAfter(colLoop3f);
  rewriter.create<memref::DeallocOp>(loc, kvec);
  rewriter.create<memref::DeallocOp>(loc, kvec_i1);
}

// Given two index arrays and positions within those arrays,
// computes the resulting number of indices based on:
// intersect=true -> the intersection of indices
// intersect=false -> the union of indices
Value computeIndexOverlapSize(PatternRewriter &rewriter, Location loc,
                              bool intersect, Value aPosStart, Value aPosEnd,
                              Value Ai, Value bPosStart, Value bPosEnd,
                              Value Bi) {

  // Types used in this function
  Type boolType = rewriter.getI1Type();
  Type indexType = rewriter.getIndexType();

  // Initial constants
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value cfalse = rewriter.create<arith::ConstantIntOp>(loc, 0, boolType);
  Value ctrue = rewriter.create<arith::ConstantIntOp>(loc, 1, boolType);

  // While Loop (exit when either array is exhausted)
  scf::WhileOp whileLoop = rewriter.create<scf::WhileOp>(
      loc,
      TypeRange{indexType, indexType, indexType, indexType, boolType, boolType,
                indexType},
      ValueRange{aPosStart, bPosStart, c0, c0, ctrue, ctrue, c0});
  Block *before =
      rewriter.createBlock(&whileLoop.getBefore(), {},
                           TypeRange{indexType, indexType, indexType, indexType,
                                     boolType, boolType, indexType});
  Block *after =
      rewriter.createBlock(&whileLoop.getAfter(), {},
                           TypeRange{indexType, indexType, indexType, indexType,
                                     boolType, boolType, indexType});
  // "while" portion of the loop
  rewriter.setInsertionPointToStart(&whileLoop.getBefore().front());
  Value posA = before->getArgument(0);
  Value posB = before->getArgument(1);
  Value validPosA = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ult, posA, aPosEnd);
  Value validPosB = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ult, posB, bPosEnd);
  Value continueLoop =
      rewriter.create<arith::AndIOp>(loc, validPosA, validPosB);
  rewriter.create<scf::ConditionOp>(loc, continueLoop, before->getArguments());

  // "do" portion of while loop
  rewriter.setInsertionPointToStart(&whileLoop.getAfter().front());
  posA = after->getArgument(0);
  posB = after->getArgument(1);
  Value idxA = after->getArgument(2);
  Value idxB = after->getArgument(3);
  Value needsUpdateA = after->getArgument(4);
  Value needsUpdateB = after->getArgument(5);
  Value count = after->getArgument(6);

  // Update input index based on flag
  scf::IfOp if_updateA =
      rewriter.create<scf::IfOp>(loc, indexType, needsUpdateA, true);
  // if updateA
  rewriter.setInsertionPointToStart(if_updateA.thenBlock());
  Value updatedIdxA64 = rewriter.create<memref::LoadOp>(loc, Ai, posA);
  Value updatedIdxA =
      rewriter.create<arith::IndexCastOp>(loc, updatedIdxA64, indexType);
  rewriter.create<scf::YieldOp>(loc, updatedIdxA);
  // else
  rewriter.setInsertionPointToStart(if_updateA.elseBlock());
  rewriter.create<scf::YieldOp>(loc, idxA);
  rewriter.setInsertionPointAfter(if_updateA);

  // Update output index based on flag
  scf::IfOp if_updateB =
      rewriter.create<scf::IfOp>(loc, indexType, needsUpdateB, true);
  // if updateB
  rewriter.setInsertionPointToStart(if_updateB.thenBlock());
  Value updatedIdxB64 = rewriter.create<memref::LoadOp>(loc, Bi, posB);
  Value updatedIdxB =
      rewriter.create<arith::IndexCastOp>(loc, updatedIdxB64, indexType);
  rewriter.create<scf::YieldOp>(loc, updatedIdxB);
  // else
  rewriter.setInsertionPointToStart(if_updateB.elseBlock());
  rewriter.create<scf::YieldOp>(loc, idxB);
  rewriter.setInsertionPointAfter(if_updateB);

  Value newIdxA = if_updateA.getResult(0);
  Value newIdxB = if_updateB.getResult(0);
  Value idxA_lt_idxB = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ult, newIdxA, newIdxB);
  Value idxA_gt_idxB = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ugt, newIdxA, newIdxB);
  Value posAplus1 = rewriter.create<arith::AddIOp>(loc, posA, c1);
  Value posBplus1 = rewriter.create<arith::AddIOp>(loc, posB, c1);

  Value countplus1 = rewriter.create<arith::AddIOp>(loc, count, c1);
  Value countForUnion;
  if (intersect) {
    countForUnion = count;
  } else {
    countForUnion = countplus1;
  }

  scf::IfOp if_onlyA = rewriter.create<scf::IfOp>(
      loc, TypeRange{indexType, indexType, boolType, boolType, indexType},
      idxA_lt_idxB, true);
  // if onlyA
  rewriter.setInsertionPointToStart(if_onlyA.thenBlock());
  rewriter.create<scf::YieldOp>(
      loc, ValueRange{posAplus1, posB, ctrue, cfalse, countForUnion});
  // else
  rewriter.setInsertionPointToStart(if_onlyA.elseBlock());
  scf::IfOp if_onlyB = rewriter.create<scf::IfOp>(
      loc, TypeRange{indexType, indexType, boolType, boolType, indexType},
      idxA_gt_idxB, true);
  // if onlyB
  rewriter.setInsertionPointToStart(if_onlyB.thenBlock());
  rewriter.create<scf::YieldOp>(
      loc, ValueRange{posA, posBplus1, cfalse, ctrue, countForUnion});
  // else (At this point, we know idxA == idxB)
  rewriter.setInsertionPointToStart(if_onlyB.elseBlock());
  rewriter.create<scf::YieldOp>(
      loc, ValueRange{posAplus1, posBplus1, ctrue, ctrue, countplus1});
  // end onlyB
  rewriter.setInsertionPointAfter(if_onlyB);
  rewriter.create<scf::YieldOp>(loc, if_onlyB.getResults());
  // end onlyA
  rewriter.setInsertionPointAfter(if_onlyA);
  Value newPosA = if_onlyA.getResult(0);
  Value newPosB = if_onlyA.getResult(1);
  needsUpdateA = if_onlyA.getResult(2);
  needsUpdateB = if_onlyA.getResult(3);
  Value newCount = if_onlyA.getResult(4);

  rewriter.create<scf::YieldOp>(loc, ValueRange{newPosA, newPosB, newIdxA,
                                                newIdxB, needsUpdateA,
                                                needsUpdateB, newCount});
  rewriter.setInsertionPointAfter(whileLoop);

  Value finalCount;
  if (!intersect) {
    // Handle remaining elements after other array is exhausted
    scf::ForOp forLoop;
    count = whileLoop.getResult(6);
    posA = whileLoop.getResult(0);
    Value remainingPosA = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, posA, aPosEnd);
    scf::IfOp if_remainingA =
        rewriter.create<scf::IfOp>(loc, indexType, remainingPosA, true);
    // if remainingA
    rewriter.setInsertionPointToStart(if_remainingA.thenBlock());
    Value extraIndices = rewriter.create<arith::SubIOp>(loc, aPosEnd, posA);
    Value newCount = rewriter.create<arith::AddIOp>(loc, count, extraIndices);
    rewriter.create<scf::YieldOp>(loc, newCount);
    // else
    rewriter.setInsertionPointToStart(if_remainingA.elseBlock());
    posB = whileLoop.getResult(1);
    Value remainingPosB = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, posB, bPosEnd);
    scf::IfOp if_remainingB =
        rewriter.create<scf::IfOp>(loc, indexType, remainingPosB, true);
    // if remainingB
    rewriter.setInsertionPointToStart(if_remainingB.thenBlock());
    extraIndices = rewriter.create<arith::SubIOp>(loc, bPosEnd, posB);
    newCount = rewriter.create<arith::AddIOp>(loc, count, extraIndices);
    rewriter.create<scf::YieldOp>(loc, newCount);
    // else
    rewriter.setInsertionPointToStart(if_remainingB.elseBlock());
    rewriter.create<scf::YieldOp>(loc, count);
    // end remainingB
    rewriter.setInsertionPointAfter(if_remainingB);
    rewriter.create<scf::YieldOp>(loc, if_remainingB.getResults());
    // end remainingA
    rewriter.setInsertionPointAfter(if_remainingA);
    finalCount = if_remainingA.getResult(0);
  } else {
    finalCount = whileLoop.getResult(6);
  }

  return finalCount;
}

// Updates Oi and Ox with indices and values
// intersect flag determines whether this is an intersection or union operation
// Returns the final position in Oi (one more than the last value inserted)
Value computeUnionAggregation(PatternRewriter &rewriter, Location loc,
                              bool intersect, Block *binaryBlock,
                              Type valueType, Value aPosStart, Value aPosEnd,
                              Value Ai, Value Ax, Value bPosStart,
                              Value bPosEnd, Value Bi, Value Bx,
                              Value oPosStart, Value Oi, Value Ox) {
  // Types used in this function
  Type boolType = rewriter.getI1Type();
  Type int64Type = rewriter.getI64Type();
  Type indexType = rewriter.getIndexType();

  // Initial constants
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value ci0 = rewriter.create<arith::ConstantIntOp>(loc, 0, int64Type);
  Value cfalse = rewriter.create<arith::ConstantIntOp>(loc, 0, boolType);
  Value ctrue = rewriter.create<arith::ConstantIntOp>(loc, 1, boolType);
  Value typedPlaceholder = llvm::TypeSwitch<Type, Value>(valueType)
                               .Case<IntegerType>([&](IntegerType type) {
                                 return rewriter.create<arith::ConstantIntOp>(
                                     loc, -1, type.getWidth());
                               })
                               .Case<FloatType>([&](FloatType type) {
                                 return rewriter.create<arith::ConstantFloatOp>(
                                     loc, APFloat(-1.0), type);
                               });

  // While Loop (exit when either array is exhausted)
  scf::WhileOp whileLoop = rewriter.create<scf::WhileOp>(
      loc,
      TypeRange{indexType, indexType, indexType, int64Type, int64Type,
                valueType, valueType, boolType, boolType},
      ValueRange{aPosStart, bPosStart, oPosStart, ci0, ci0, typedPlaceholder,
                 typedPlaceholder, ctrue, ctrue});
  Block *before = rewriter.createBlock(
      &whileLoop.getBefore(), {},
      TypeRange{indexType, indexType, indexType, int64Type, int64Type,
                valueType, valueType, boolType, boolType});
  Block *after = rewriter.createBlock(&whileLoop.getAfter(), {},
                                      TypeRange{indexType, indexType, indexType,
                                                int64Type, int64Type, valueType,
                                                valueType, boolType, boolType});
  // "while" portion of the loop
  rewriter.setInsertionPointToStart(&whileLoop.getBefore().front());
  Value posA = before->getArgument(0);
  Value posB = before->getArgument(1);
  Value validPosA = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ult, posA, aPosEnd);
  Value validPosB = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ult, posB, bPosEnd);
  Value continueLoop =
      rewriter.create<arith::AndIOp>(loc, validPosA, validPosB);
  rewriter.create<scf::ConditionOp>(loc, continueLoop, before->getArguments());

  // "do" portion of while loop
  rewriter.setInsertionPointToStart(&whileLoop.getAfter().front());
  posA = after->getArgument(0);
  posB = after->getArgument(1);
  Value posO = after->getArgument(2);
  Value idxA = after->getArgument(3);
  Value idxB = after->getArgument(4);
  Value valA = after->getArgument(5);
  Value valB = after->getArgument(6);
  Value needsUpdateA = after->getArgument(7);
  Value needsUpdateB = after->getArgument(8);

  // Update A's index and value based on flag
  scf::IfOp if_updateA = rewriter.create<scf::IfOp>(
      loc, TypeRange{int64Type, valueType}, needsUpdateA, true);
  // if updateA
  rewriter.setInsertionPointToStart(if_updateA.thenBlock());
  Value updatedIdxA = rewriter.create<memref::LoadOp>(loc, Ai, posA);
  Value updatedValA = rewriter.create<memref::LoadOp>(loc, Ax, posA);
  rewriter.create<scf::YieldOp>(loc, ValueRange{updatedIdxA, updatedValA});
  // else
  rewriter.setInsertionPointToStart(if_updateA.elseBlock());
  rewriter.create<scf::YieldOp>(loc, ValueRange{idxA, valA});
  rewriter.setInsertionPointAfter(if_updateA);

  // Update B's index and value based on flag
  scf::IfOp if_updateB = rewriter.create<scf::IfOp>(
      loc, TypeRange{int64Type, valueType}, needsUpdateB, true);
  // if updateB
  rewriter.setInsertionPointToStart(if_updateB.thenBlock());
  Value updatedIdxB = rewriter.create<memref::LoadOp>(loc, Bi, posB);
  Value updatedValB = rewriter.create<memref::LoadOp>(loc, Bx, posB);
  rewriter.create<scf::YieldOp>(loc, ValueRange{updatedIdxB, updatedValB});
  // else
  rewriter.setInsertionPointToStart(if_updateB.elseBlock());
  rewriter.create<scf::YieldOp>(loc, ValueRange{idxB, valB});
  rewriter.setInsertionPointAfter(if_updateB);

  Value newIdxA = if_updateA.getResult(0);
  Value newValA = if_updateA.getResult(1);
  Value newIdxB = if_updateB.getResult(0);
  Value newValB = if_updateB.getResult(1);
  Value idxA_lt_idxB = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ult, newIdxA, newIdxB);
  Value idxA_gt_idxB = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ugt, newIdxA, newIdxB);
  Value posAplus1 = rewriter.create<arith::AddIOp>(loc, posA, c1);
  Value posBplus1 = rewriter.create<arith::AddIOp>(loc, posB, c1);

  Value posOplus1 = rewriter.create<arith::AddIOp>(loc, posO, c1);
  Value posOForUnion;
  if (intersect) {
    posOForUnion = posO;
  } else {
    posOForUnion = posOplus1;
  }

  scf::IfOp if_onlyA = rewriter.create<scf::IfOp>(
      loc, TypeRange{indexType, indexType, indexType, boolType, boolType},
      idxA_lt_idxB, true);
  // if onlyA
  rewriter.setInsertionPointToStart(if_onlyA.thenBlock());
  if (!intersect) {
    rewriter.create<memref::StoreOp>(loc, newIdxA, Oi, posO);
    rewriter.create<memref::StoreOp>(loc, newValA, Ox, posO);
  }
  rewriter.create<scf::YieldOp>(
      loc, ValueRange{posAplus1, posB, posOForUnion, ctrue, cfalse});
  // else
  rewriter.setInsertionPointToStart(if_onlyA.elseBlock());
  scf::IfOp if_onlyB = rewriter.create<scf::IfOp>(
      loc, TypeRange{indexType, indexType, indexType, boolType, boolType},
      idxA_gt_idxB, true);
  // if onlyB
  rewriter.setInsertionPointToStart(if_onlyB.thenBlock());
  if (!intersect) {
    rewriter.create<memref::StoreOp>(loc, newIdxB, Oi, posO);
    rewriter.create<memref::StoreOp>(loc, newValB, Ox, posO);
  }
  rewriter.create<scf::YieldOp>(
      loc, ValueRange{posA, posBplus1, posOForUnion, cfalse, ctrue});
  // else
  rewriter.setInsertionPointToStart(if_onlyB.elseBlock());
  // At this point, we know newIdxA == newIdxB
  rewriter.create<memref::StoreOp>(loc, newIdxA, Oi, posO);

  if (binaryBlock) {
    // Insert binary block
    // TODO we might need a swapArgs boolean flag to handle CSC indices
    ValueRange injectVals = ValueRange{newValA, newValB, newIdxA, newIdxB};
    rewriter.mergeBlocks(
        binaryBlock, rewriter.getBlock(),
        injectVals.slice(0, binaryBlock->getArguments().size()));

    graphblas::YieldOp binaryYield = llvm::dyn_cast_or_null<graphblas::YieldOp>(
        rewriter.getBlock()->getTerminator());
    Value aggVal = binaryYield.values().front();
    rewriter.eraseOp(binaryYield);

    rewriter.create<memref::StoreOp>(loc, aggVal, Ox, posO);
  } else {
    // Arriving here is a mistake; fill with -1
    rewriter.create<memref::StoreOp>(loc, typedPlaceholder, Ox, posO);
  }
  rewriter.create<scf::YieldOp>(
      loc, ValueRange{posAplus1, posBplus1, posOplus1, ctrue, ctrue});
  // end onlyB
  rewriter.setInsertionPointAfter(if_onlyB);
  rewriter.create<scf::YieldOp>(loc, if_onlyB.getResults());
  // end onlyA
  rewriter.setInsertionPointAfter(if_onlyA);
  Value newPosA = if_onlyA.getResult(0);
  Value newPosB = if_onlyA.getResult(1);
  Value newPosO = if_onlyA.getResult(2);
  needsUpdateA = if_onlyA.getResult(3);
  needsUpdateB = if_onlyA.getResult(4);

  rewriter.create<scf::YieldOp>(
      loc, ValueRange{newPosA, newPosB, newPosO, newIdxA, newIdxB, newValA,
                      newValB, needsUpdateA, needsUpdateB});
  rewriter.setInsertionPointAfter(whileLoop);

  Value finalPosO;
  if (!intersect) {
    // For loop (remaining elements after other array is exhausted)
    scf::ForOp forLoop;
    posO = whileLoop.getResult(2);
    posA = whileLoop.getResult(0);
    Value remainingPosA = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, posA, aPosEnd);
    scf::IfOp if_remainingA =
        rewriter.create<scf::IfOp>(loc, indexType, remainingPosA, true);
    // if remainingA
    rewriter.setInsertionPointToStart(if_remainingA.thenBlock());
    forLoop = rewriter.create<scf::ForOp>(loc, posA, aPosEnd, c1, posO);
    Value aa = forLoop.getInductionVar();
    Value currPosO = forLoop.getLoopBody().getArgument(1);
    rewriter.setInsertionPointToStart(forLoop.getBody());
    idxA = rewriter.create<memref::LoadOp>(loc, Ai, aa);
    valA = rewriter.create<memref::LoadOp>(loc, Ax, aa);
    rewriter.create<memref::StoreOp>(loc, idxA, Oi, currPosO);
    rewriter.create<memref::StoreOp>(loc, valA, Ox, currPosO);
    Value newPosO = rewriter.create<arith::AddIOp>(loc, currPosO, c1);
    rewriter.create<scf::YieldOp>(loc, newPosO);
    rewriter.setInsertionPointAfter(forLoop);
    rewriter.create<scf::YieldOp>(loc, forLoop.getResult(0));
    // else
    rewriter.setInsertionPointToStart(if_remainingA.elseBlock());
    posB = whileLoop.getResult(1);
    Value remainingPosB = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, posB, bPosEnd);
    scf::IfOp if_remainingB =
        rewriter.create<scf::IfOp>(loc, indexType, remainingPosB, true);
    // if remainingB
    rewriter.setInsertionPointToStart(if_remainingB.thenBlock());
    forLoop = rewriter.create<scf::ForOp>(loc, posB, bPosEnd, c1, posO);
    Value bb = forLoop.getInductionVar();
    currPosO = forLoop.getLoopBody().getArgument(1);
    rewriter.setInsertionPointToStart(forLoop.getBody());
    idxB = rewriter.create<memref::LoadOp>(loc, Bi, bb);
    valB = rewriter.create<memref::LoadOp>(loc, Bx, bb);
    rewriter.create<memref::StoreOp>(loc, idxB, Oi, currPosO);
    rewriter.create<memref::StoreOp>(loc, valB, Ox, currPosO);
    newPosO = rewriter.create<arith::AddIOp>(loc, currPosO, c1);
    rewriter.create<scf::YieldOp>(loc, newPosO);
    rewriter.setInsertionPointAfter(forLoop);
    rewriter.create<scf::YieldOp>(loc, forLoop.getResult(0));
    // else
    rewriter.setInsertionPointToStart(if_remainingB.elseBlock());
    rewriter.create<scf::YieldOp>(loc, posO);
    // end remainingB
    rewriter.setInsertionPointAfter(if_remainingB);
    rewriter.create<scf::YieldOp>(loc, if_remainingB.getResults());
    // end remainingA
    rewriter.setInsertionPointAfter(if_remainingA);
    finalPosO = if_remainingA.getResult(0);
  } else {
    finalPosO = whileLoop.getResult(2);
  }

  return finalPosO;
}

// Updates Oi and Ox with indices and values
// Value in A which not not overlap the mask M are not included in O
// Returns the final position in Oi (one more than the last value inserted)
Value applyMask(PatternRewriter &rewriter, Location loc, Type valueType,
                Value aPosStart, Value aPosEnd, Value Ai, Value Ax,
                Value mPosStart, Value mPosEnd, Value Mi, Value oPosStart,
                Value Oi, Value Ox) {
  // Types used in this function
  Type boolType = rewriter.getI1Type();
  Type int64Type = rewriter.getI64Type();
  Type indexType = rewriter.getIndexType();

  // Initial constants
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value ci0 = rewriter.create<arith::ConstantIntOp>(loc, 0, int64Type);
  Value cfalse = rewriter.create<arith::ConstantIntOp>(loc, 0, boolType);
  Value ctrue = rewriter.create<arith::ConstantIntOp>(loc, 1, boolType);
  Value cf0 = llvm::TypeSwitch<Type, Value>(valueType)
                  .Case<IntegerType>([&](IntegerType type) {
                    return rewriter.create<arith::ConstantIntOp>(
                        loc, 0, type.getWidth());
                  })
                  .Case<FloatType>([&](FloatType type) {
                    return rewriter.create<arith::ConstantFloatOp>(
                        loc, APFloat(0.0), type);
                  });

  // While Loop (exit when either array is exhausted)
  scf::WhileOp whileLoop = rewriter.create<scf::WhileOp>(
      loc,
      TypeRange{indexType, indexType, indexType, int64Type, int64Type,
                valueType, boolType, boolType},
      ValueRange{aPosStart, mPosStart, oPosStart, ci0, ci0, cf0, ctrue, ctrue});
  Block *before =
      rewriter.createBlock(&whileLoop.getBefore(), {},
                           TypeRange{indexType, indexType, indexType, int64Type,
                                     int64Type, valueType, boolType, boolType});
  Block *after =
      rewriter.createBlock(&whileLoop.getAfter(), {},
                           TypeRange{indexType, indexType, indexType, int64Type,
                                     int64Type, valueType, boolType, boolType});
  // "while" portion of the loop
  rewriter.setInsertionPointToStart(&whileLoop.getBefore().front());
  Value posA = before->getArgument(0);
  Value posM = before->getArgument(1);
  Value validPosA = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ult, posA, aPosEnd);
  Value validPosM = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ult, posM, mPosEnd);
  Value continueLoop =
      rewriter.create<arith::AndIOp>(loc, validPosA, validPosM);
  rewriter.create<scf::ConditionOp>(loc, continueLoop, before->getArguments());

  // "do" portion of while loop
  rewriter.setInsertionPointToStart(&whileLoop.getAfter().front());
  posA = after->getArgument(0);
  posM = after->getArgument(1);
  Value posO = after->getArgument(2);
  Value idxA = after->getArgument(3);
  Value idxM = after->getArgument(4);
  Value valA = after->getArgument(5);
  Value needsUpdateA = after->getArgument(6);
  Value needsUpdateM = after->getArgument(7);

  // Update A's index and value based on flag
  scf::IfOp if_updateA = rewriter.create<scf::IfOp>(
      loc, TypeRange{int64Type, valueType}, needsUpdateA, true);
  // if updateA
  rewriter.setInsertionPointToStart(if_updateA.thenBlock());
  Value updatedIdxA = rewriter.create<memref::LoadOp>(loc, Ai, posA);
  Value updatedValA = rewriter.create<memref::LoadOp>(loc, Ax, posA);
  rewriter.create<scf::YieldOp>(loc, ValueRange{updatedIdxA, updatedValA});
  // else
  rewriter.setInsertionPointToStart(if_updateA.elseBlock());
  rewriter.create<scf::YieldOp>(loc, ValueRange{idxA, valA});
  rewriter.setInsertionPointAfter(if_updateA);

  // Update M's index based on flag
  scf::IfOp if_updateM =
      rewriter.create<scf::IfOp>(loc, int64Type, needsUpdateM, true);
  // if updateM
  rewriter.setInsertionPointToStart(if_updateM.thenBlock());
  Value updatedIdxM = rewriter.create<memref::LoadOp>(loc, Mi, posM);
  rewriter.create<scf::YieldOp>(loc, updatedIdxM);
  // else
  rewriter.setInsertionPointToStart(if_updateM.elseBlock());
  rewriter.create<scf::YieldOp>(loc, idxM);
  rewriter.setInsertionPointAfter(if_updateM);

  Value newIdxA = if_updateA.getResult(0);
  Value newValA = if_updateA.getResult(1);
  Value newIdxM = if_updateM.getResult(0);
  Value idxA_lt_idxM = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ult, newIdxA, newIdxM);
  Value idxA_gt_idxM = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ugt, newIdxA, newIdxM);
  Value posAplus1 = rewriter.create<arith::AddIOp>(loc, posA, c1);
  Value posMplus1 = rewriter.create<arith::AddIOp>(loc, posM, c1);

  Value posOplus1 = rewriter.create<arith::AddIOp>(loc, posO, c1);

  scf::IfOp if_onlyA = rewriter.create<scf::IfOp>(
      loc, TypeRange{indexType, indexType, indexType, boolType, boolType},
      idxA_lt_idxM, true);
  // if onlyA
  rewriter.setInsertionPointToStart(if_onlyA.thenBlock());
  rewriter.create<scf::YieldOp>(
      loc, ValueRange{posAplus1, posM, posO, ctrue, cfalse});
  // else
  rewriter.setInsertionPointToStart(if_onlyA.elseBlock());
  scf::IfOp if_onlyM = rewriter.create<scf::IfOp>(
      loc, TypeRange{indexType, indexType, indexType, boolType, boolType},
      idxA_gt_idxM, true);
  // if onlyM
  rewriter.setInsertionPointToStart(if_onlyM.thenBlock());
  rewriter.create<scf::YieldOp>(
      loc, ValueRange{posA, posMplus1, posO, cfalse, ctrue});
  // else
  rewriter.setInsertionPointToStart(if_onlyM.elseBlock());
  // At this point, we know newIdxA == newIdxM
  rewriter.create<memref::StoreOp>(loc, newIdxA, Oi, posO);
  rewriter.create<memref::StoreOp>(loc, newValA, Ox, posO);
  rewriter.create<scf::YieldOp>(
      loc, ValueRange{posAplus1, posMplus1, posOplus1, ctrue, ctrue});
  // end onlyM
  rewriter.setInsertionPointAfter(if_onlyM);
  rewriter.create<scf::YieldOp>(loc, if_onlyM.getResults());
  // end onlyA
  rewriter.setInsertionPointAfter(if_onlyA);
  Value newPosA = if_onlyA.getResult(0);
  Value newPosM = if_onlyA.getResult(1);
  Value newPosO = if_onlyA.getResult(2);
  needsUpdateA = if_onlyA.getResult(3);
  needsUpdateM = if_onlyA.getResult(4);

  rewriter.create<scf::YieldOp>(loc, ValueRange{newPosA, newPosM, newPosO,
                                                newIdxA, newIdxM, newValA,
                                                needsUpdateA, needsUpdateM});
  rewriter.setInsertionPointAfter(whileLoop);

  Value finalPosO = whileLoop.getResult(2);

  return finalPosO;
}

void computeVectorElementWise(PatternRewriter &rewriter, Location loc,
                              ModuleOp module, Value lhs, Value rhs,
                              Value output, Block *binaryBlock,
                              EwiseBehavior behavior) {
  // aggBlock is ignored if behavior is MASK or MASK_COMPLEMENT

  bool intersect = behavior != UNION;

  // Types
  RankedTensorType outputType = output.getType().dyn_cast<RankedTensorType>();
  Type int64Type = rewriter.getIntegerType(64);
  Type inputElementType =
      lhs.getType().cast<RankedTensorType>().getElementType();
  MemRefType memref1DI64Type = MemRefType::get({-1}, int64Type);
  MemRefType memrefIValueType = getMemrefValueType(lhs.getType());
  MemRefType memrefOValueType = getMemrefValueType(outputType);

  // Initial constants
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  // Get sparse tensor info
  Value size = rewriter.create<graphblas::SizeOp>(loc, output);
  Value lhsNnz = rewriter.create<graphblas::NumValsOp>(loc, lhs);
  Value rhsNnz = rewriter.create<graphblas::NumValsOp>(loc, rhs);
  Value Li = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                         lhs, c0);
  Value Lx =
      rewriter.create<sparse_tensor::ToValuesOp>(loc, memrefIValueType, lhs);
  Value Ri = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                         rhs, c0);
  Value Rx =
      rewriter.create<sparse_tensor::ToValuesOp>(loc, memrefIValueType, rhs);

  // Special handling for complemented mask
  Value ewiseSize, maskComplement, maskComplementSize;
  if (behavior == MASK_COMPLEMENT) {
    ValueRange results =
        buildMaskComplement(rewriter, loc, size, Ri, c0, rhsNnz);
    maskComplement = results[0];
    maskComplementSize = results[1];
    ewiseSize = computeIndexOverlapSize(rewriter, loc, true, c0, lhsNnz, Li, c0,
                                        maskComplementSize, maskComplement);
  } else {
    ewiseSize = computeIndexOverlapSize(rewriter, loc, intersect, c0, lhsNnz,
                                        Li, c0, rhsNnz, Ri);
  }

  Value ewiseSize64 =
      rewriter.create<arith::IndexCastOp>(loc, ewiseSize, int64Type);

  callResizeIndex(rewriter, module, loc, output, c0, ewiseSize);
  callResizeValues(rewriter, module, loc, output, ewiseSize);

  Value Op = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type,
                                                          output, c0);
  rewriter.create<memref::StoreOp>(loc, ewiseSize64, Op, c1);
  Value Oi = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                         output, c0);
  Value Ox =
      rewriter.create<sparse_tensor::ToValuesOp>(loc, memrefOValueType, output);

  if (behavior == MASK) {
    applyMask(rewriter, loc, inputElementType, c0, lhsNnz, Li, Lx, c0, rhsNnz,
              Ri, c0, Oi, Ox);
  } else if (behavior == MASK_COMPLEMENT) {
    applyMask(rewriter, loc, inputElementType, c0, lhsNnz, Li, Lx, c0,
              maskComplementSize, maskComplement, c0, Oi, Ox);
    rewriter.create<memref::DeallocOp>(loc, maskComplement);
  } else {
    computeUnionAggregation(rewriter, loc, intersect, binaryBlock,
                            inputElementType, c0, lhsNnz, Li, Lx, c0, rhsNnz,
                            Ri, Rx, c0, Oi, Ox);
  }
}

void computeMatrixElementWise(PatternRewriter &rewriter, Location loc,
                              ModuleOp module, Value lhs, Value rhs,
                              Value output, Block *binaryBlock,
                              EwiseBehavior behavior) {
  // aggBlock is ignored if behavior is MASK or MASK_COMPLEMENT

  bool intersect = behavior != UNION;

  // Types
  RankedTensorType outputType = output.getType().cast<RankedTensorType>();
  Type indexType = rewriter.getIndexType();
  Type boolType = rewriter.getI1Type();
  Type int64Type = rewriter.getIntegerType(64);
  Type inputElementType =
      lhs.getType().cast<RankedTensorType>().getElementType();
  MemRefType memref1DI64Type = MemRefType::get({-1}, int64Type);
  MemRefType memrefIValueType = getMemrefValueType(lhs.getType());
  MemRefType memrefOValueType = getMemrefValueType(outputType);

  // Initial constants
  Value cfalse = rewriter.create<arith::ConstantIntOp>(loc, 0, boolType);
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value ci0 = rewriter.create<arith::ConstantIntOp>(loc, 0, int64Type);

  Value nrows, ncols;
  if (hasRowOrdering(outputType)) {
    nrows = rewriter.create<graphblas::NumRowsOp>(loc, output);
    ncols = rewriter.create<graphblas::NumColsOp>(loc, output);
  } else {
    // Swap nrows and ncols so logic works
    ncols = rewriter.create<graphblas::NumRowsOp>(loc, output);
    nrows = rewriter.create<graphblas::NumColsOp>(loc, output);
  }

  // Get sparse tensor info
  Value Lp = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type,
                                                          lhs, c1);
  Value Li = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                         lhs, c1);
  Value Lx =
      rewriter.create<sparse_tensor::ToValuesOp>(loc, memrefIValueType, lhs);
  Value Rp = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type,
                                                          rhs, c1);
  Value Ri = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                         rhs, c1);
  Value Rx =
      rewriter.create<sparse_tensor::ToValuesOp>(loc, memrefIValueType, rhs);
  Value Op = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type,
                                                          output, c1);

  // 1st pass
  //   Compute overlap size for each row
  //   Store results in Op
  scf::ParallelOp rowLoop1 =
      rewriter.create<scf::ParallelOp>(loc, c0, nrows, c1);
  Value row = rowLoop1.getInductionVars()[0];
  rewriter.setInsertionPointToStart(rowLoop1.getBody());

  Value rowPlus1 = rewriter.create<arith::AddIOp>(loc, row, c1);
  Value lhsColStart64 = rewriter.create<memref::LoadOp>(loc, Lp, row);
  Value lhsColEnd64 = rewriter.create<memref::LoadOp>(loc, Lp, rowPlus1);
  Value rhsColStart64 = rewriter.create<memref::LoadOp>(loc, Rp, row);
  Value rhsColEnd64 = rewriter.create<memref::LoadOp>(loc, Rp, rowPlus1);
  Value LcmpColSame = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, lhsColStart64, lhsColEnd64);
  Value RcmpColSame = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, rhsColStart64, rhsColEnd64);
  Value emptyRow;
  if (behavior == MASK_COMPLEMENT) {
    emptyRow = cfalse;
  } else if (intersect) {
    emptyRow = rewriter.create<arith::OrIOp>(loc, LcmpColSame, RcmpColSame);
  } else {
    emptyRow = rewriter.create<arith::AndIOp>(loc, LcmpColSame, RcmpColSame);
  }

  scf::IfOp ifBlock_rowTotal =
      rewriter.create<scf::IfOp>(loc, int64Type, emptyRow, true);
  // if cmpColSame
  rewriter.setInsertionPointToStart(ifBlock_rowTotal.thenBlock());
  rewriter.create<scf::YieldOp>(loc, ci0);

  // else
  rewriter.setInsertionPointToStart(ifBlock_rowTotal.elseBlock());
  Value lhsColStart =
      rewriter.create<arith::IndexCastOp>(loc, lhsColStart64, indexType);
  Value lhsColEnd =
      rewriter.create<arith::IndexCastOp>(loc, lhsColEnd64, indexType);
  Value rhsColStart =
      rewriter.create<arith::IndexCastOp>(loc, rhsColStart64, indexType);
  Value rhsColEnd =
      rewriter.create<arith::IndexCastOp>(loc, rhsColEnd64, indexType);

  // Special handling for complemented mask
  Value unionSize;
  if (behavior == MASK_COMPLEMENT) {
    ValueRange results =
        buildMaskComplement(rewriter, loc, ncols, Ri, rhsColStart, rhsColEnd);
    Value maskComplement = results[0];
    Value maskComplementSize = results[1];
    unionSize =
        computeIndexOverlapSize(rewriter, loc, true, lhsColStart, lhsColEnd, Li,
                                c0, maskComplementSize, maskComplement);
    rewriter.create<memref::DeallocOp>(loc, maskComplement);
  } else {
    unionSize =
        computeIndexOverlapSize(rewriter, loc, intersect, lhsColStart,
                                lhsColEnd, Li, rhsColStart, rhsColEnd, Ri);
  }

  Value unionSize64 =
      rewriter.create<arith::IndexCastOp>(loc, unionSize, int64Type);
  rewriter.create<scf::YieldOp>(loc, unionSize64);

  // end if cmpColSame
  rewriter.setInsertionPointAfter(ifBlock_rowTotal);
  Value rowSize = ifBlock_rowTotal.getResult(0);
  rewriter.create<memref::StoreOp>(loc, rowSize, Op, row);

  // end row loop
  rewriter.setInsertionPointAfter(rowLoop1);

  // 2nd pass
  //   Compute the cumsum of values in Op to build the final Op
  //   Then resize output indices and values
  rewriter.create<memref::StoreOp>(loc, ci0, Op, nrows);
  scf::ForOp rowLoop2 = rewriter.create<scf::ForOp>(loc, c0, nrows, c1);
  Value cs_i = rowLoop2.getInductionVar();
  rewriter.setInsertionPointToStart(rowLoop2.getBody());

  Value csTemp = rewriter.create<memref::LoadOp>(loc, Op, cs_i);
  Value cumsum = rewriter.create<memref::LoadOp>(loc, Op, nrows);
  rewriter.create<memref::StoreOp>(loc, cumsum, Op, cs_i);
  Value cumsum2 = rewriter.create<arith::AddIOp>(loc, cumsum, csTemp);
  rewriter.create<memref::StoreOp>(loc, cumsum2, Op, nrows);

  // end row loop
  rewriter.setInsertionPointAfter(rowLoop2);

  Value nnz = rewriter.create<graphblas::NumValsOp>(loc, output);
  callResizeIndex(rewriter, module, loc, output, c1, nnz);
  callResizeValues(rewriter, module, loc, output, nnz);

  Value Oi = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type,
                                                         output, c1);
  Value Ox =
      rewriter.create<sparse_tensor::ToValuesOp>(loc, memrefOValueType, output);

  // 3rd pass
  //   In parallel over the rows, compute the aggregation
  //   Store in Oi and Ox
  scf::ParallelOp rowLoop3 =
      rewriter.create<scf::ParallelOp>(loc, c0, nrows, c1);
  row = rowLoop3.getInductionVars()[0];
  rewriter.setInsertionPointToStart(rowLoop3.getBody());

  rowPlus1 = rewriter.create<arith::AddIOp>(loc, row, c1);
  Value opStart64 = rewriter.create<memref::LoadOp>(loc, Op, row);
  Value opEnd64 = rewriter.create<memref::LoadOp>(loc, Op, rowPlus1);
  Value cmp_opDifferent = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ne, opStart64, opEnd64);
  scf::IfOp ifBlock_cmpDiff = rewriter.create<scf::IfOp>(loc, cmp_opDifferent);
  rewriter.setInsertionPointToStart(ifBlock_cmpDiff.thenBlock());

  Value OcolStart64 = rewriter.create<memref::LoadOp>(loc, Op, row);
  Value OcolStart =
      rewriter.create<arith::IndexCastOp>(loc, OcolStart64, indexType);

  lhsColStart64 = rewriter.create<memref::LoadOp>(loc, Lp, row);
  lhsColEnd64 = rewriter.create<memref::LoadOp>(loc, Lp, rowPlus1);
  lhsColStart =
      rewriter.create<arith::IndexCastOp>(loc, lhsColStart64, indexType);
  lhsColEnd = rewriter.create<arith::IndexCastOp>(loc, lhsColEnd64, indexType);
  rhsColStart64 = rewriter.create<memref::LoadOp>(loc, Rp, row);
  rhsColEnd64 = rewriter.create<memref::LoadOp>(loc, Rp, rowPlus1);
  rhsColStart =
      rewriter.create<arith::IndexCastOp>(loc, rhsColStart64, indexType);
  rhsColEnd = rewriter.create<arith::IndexCastOp>(loc, rhsColEnd64, indexType);

  if (behavior == MASK) {
    applyMask(rewriter, loc, inputElementType, lhsColStart, lhsColEnd, Li, Lx,
              rhsColStart, rhsColEnd, Ri, OcolStart, Oi, Ox);
  } else if (behavior == MASK_COMPLEMENT) {
    // Need to recompute maskComplement because previous use was inside a
    // different loop
    ValueRange results =
        buildMaskComplement(rewriter, loc, ncols, Ri, rhsColStart, rhsColEnd);
    Value maskComplement = results[0];
    Value maskComplementSize = results[1];
    applyMask(rewriter, loc, inputElementType, lhsColStart, lhsColEnd, Li, Lx,
              c0, maskComplementSize, maskComplement, OcolStart, Oi, Ox);
    rewriter.create<memref::DeallocOp>(loc, maskComplement);
  } else {
    computeUnionAggregation(rewriter, loc, intersect, binaryBlock,
                            inputElementType, lhsColStart, lhsColEnd, Li, Lx,
                            rhsColStart, rhsColEnd, Ri, Rx, OcolStart, Oi, Ox);
  }

  // end if cmpDiff
  rewriter.setInsertionPointAfter(ifBlock_cmpDiff);

  // end row loop
  rewriter.setInsertionPointAfter(rowLoop3);
}
