#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/None.h"
#include "mlir/IR/Region.h"

#include "GraphBLAS/GraphBLASOps.h"
#include "GraphBLAS/GraphBLASUtils.h"
#include "GraphBLAS/GraphBLASArrayUtils.h"

using namespace ::mlir;

Value computeNumOverlaps(PatternRewriter &rewriter, Value nk,
                         Value fixedIndices, Value fixedIndexStart, Value fixedIndexEnd,
                         Value iterPointers, Value iterIndices,
                         // If no mask is used, set maskIndices to nullptr, and provide maskStart=c0 and maskEnd=len(iterPointers)-1
                         Value maskIndices, Value maskStart, Value maskEnd,
                         Type valueType
                         ) {
  Location loc = rewriter.getUnknownLoc();

  // Types used in this function
  Type indexType = rewriter.getIndexType();
  Type int64Type = rewriter.getIntegerType(64);
  Type boolType = rewriter.getI1Type();
  MemRefType memref1DBoolType = MemRefType::get({-1}, boolType);

  // Initial constants
  Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
  Value ci0 = rewriter.create<ConstantIntOp>(loc, 0, int64Type);
  Value ci1 = rewriter.create<ConstantIntOp>(loc, 1, int64Type);
  Value ctrue = rewriter.create<ConstantIntOp>(loc, 1, boolType);
  Value cfalse = rewriter.create<ConstantIntOp>(loc, 0, boolType);

  // Construct a dense array indicating valid kk positions within fixed index
  Value kvec_i1 = rewriter.create<memref::AllocOp>(loc, memref1DBoolType, nk);
  rewriter.create<linalg::FillOp>(loc, kvec_i1, cfalse);
  scf::ParallelOp colLoop1 = rewriter.create<scf::ParallelOp>(loc, fixedIndexStart, fixedIndexEnd, c1);
  Value jj = colLoop1.getInductionVars()[0];
  rewriter.setInsertionPointToStart(colLoop1.getBody());
  Value col64 = rewriter.create<memref::LoadOp>(loc, fixedIndices, jj);
  Value col = rewriter.create<IndexCastOp>(loc, col64, indexType);
  rewriter.create<memref::StoreOp>(loc, ctrue, kvec_i1, col);
  rewriter.setInsertionPointAfter(colLoop1);
  // Loop thru all columns; count number of resulting nonzeros in the row
  if (maskIndices != nullptr) {
    colLoop1 = rewriter.create<scf::ParallelOp>(loc, maskStart, maskEnd, c1, ci0);
    Value mm = colLoop1.getInductionVars()[0];
    rewriter.setInsertionPointToStart(colLoop1.getBody());
    col64 = rewriter.create<memref::LoadOp>(loc, maskIndices, mm);
    col = rewriter.create<IndexCastOp>(loc, col64, indexType);
  } else {
    colLoop1 = rewriter.create<scf::ParallelOp>(loc, maskStart, maskEnd, c1, ci0);
    col = colLoop1.getInductionVars()[0];
    rewriter.setInsertionPointToStart(colLoop1.getBody());
  }
  Value colPlus1 = rewriter.create<AddIOp>(loc, col, c1);
  Value rowStart64 = rewriter.create<memref::LoadOp>(loc, iterPointers, col);
  Value rowEnd64 = rewriter.create<memref::LoadOp>(loc, iterPointers, colPlus1);
  Value cmpRowSame = rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, rowStart64, rowEnd64);
  // Find overlap in column indices with kvec
  scf::IfOp ifBlock_overlap = rewriter.create<scf::IfOp>(loc, int64Type, cmpRowSame, true);
  // if cmpRowSame
  rewriter.setInsertionPointToStart(ifBlock_overlap.thenBlock());
  rewriter.create<scf::YieldOp>(loc, ci0);
  // else
  rewriter.setInsertionPointToStart(ifBlock_overlap.elseBlock());
  // Walk thru the indices; on a match yield 1, else yield 0
  scf::WhileOp whileLoop = rewriter.create<scf::WhileOp>(loc, int64Type, rowStart64);
  Block *before = rewriter.createBlock(&whileLoop.before(), {}, int64Type);
  Block *after = rewriter.createBlock(&whileLoop.after(), {}, int64Type);
  Value ii64 = before->getArgument(0);
  rewriter.setInsertionPointToStart(&whileLoop.before().front());
  // Check if ii >= rowEnd
  Value cmpEndReached = rewriter.create<CmpIOp>(loc, CmpIPredicate::uge, ii64, rowEnd64);
  scf::IfOp ifBlock_continueSearch = rewriter.create<scf::IfOp>(loc, TypeRange{boolType, int64Type}, cmpEndReached, true);
  // if cmpEndReached
  rewriter.setInsertionPointToStart(ifBlock_continueSearch.thenBlock());
  rewriter.create<scf::YieldOp>(loc, ValueRange{cfalse, ci0});
  // else
  rewriter.setInsertionPointToStart(ifBlock_continueSearch.elseBlock());
  // Check if row has a match in kvec
  Value ii = rewriter.create<IndexCastOp>(loc, ii64, indexType);
  Value kk64 = rewriter.create<memref::LoadOp>(loc, iterIndices, ii);
  Value kk = rewriter.create<IndexCastOp>(loc, kk64, indexType);
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
  rewriter.setInsertionPointToStart(&whileLoop.after().front());
  Value iiPrev = after->getArgument(0);
  Value iiNext = rewriter.create<AddIOp>(loc, iiPrev, ci1);
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
  Value z = rewriter.create<AddIOp>(loc, lhs, rhs);
  rewriter.create<scf::ReduceReturnOp>(loc, z);
  // end col loop
  rewriter.setInsertionPointAfter(colLoop1);
  Value total = colLoop1.getResult(0);
  rewriter.create<memref::DeallocOp>(loc, kvec_i1);
  return total;
}

void computeInnerProduct(PatternRewriter &rewriter, Value nk,
                          Value fixedIndices, Value fixedValues, Value fixedIndexStart, Value fixedIndexEnd,
                          Value iterPointers, Value iterIndices, Value iterValues,
                          // If no mask is used, set maskIndices to nullptr, and provide maskStart=c0 and maskEnd=len(iterPointers)-1
                          Value maskIndices, Value maskStart, Value maskEnd,
                          Type valueType, ExtensionBlocks extBlocks,
                          Value outputIndices, Value outputValues, Value indexOffset
                          ) {
  Location loc = rewriter.getUnknownLoc();

  // Types used in this function
  Type indexType = rewriter.getIndexType();
  Type int64Type = rewriter.getIntegerType(64);
  Type boolType = rewriter.getI1Type();
  MemRefType memref1DBoolType = MemRefType::get({-1}, boolType);
  MemRefType memref1DValueType = MemRefType::get({-1}, valueType);

  // Initial constants
  Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
  Value ctrue = rewriter.create<ConstantIntOp>(loc, 1, boolType);
  Value cfalse = rewriter.create<ConstantIntOp>(loc, 0, boolType);

  // Construct a dense array of row values
  Value kvec = rewriter.create<memref::AllocOp>(loc, memref1DValueType, nk);
  Value kvec_i1 = rewriter.create<memref::AllocOp>(loc, memref1DBoolType, nk);
  rewriter.create<linalg::FillOp>(loc, kvec_i1, cfalse);
  scf::ParallelOp colLoop3p = rewriter.create<scf::ParallelOp>(loc, fixedIndexStart, fixedIndexEnd, c1);
  Value jj = colLoop3p.getInductionVars()[0];
  rewriter.setInsertionPointToStart(colLoop3p.getBody());
  Value fixedJ64 = rewriter.create<memref::LoadOp>(loc, fixedIndices, jj);
  Value fixedJ = rewriter.create<IndexCastOp>(loc, fixedJ64, indexType);
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
      col = rewriter.create<IndexCastOp>(loc, col64, indexType);
  } else {
      colLoop3f = rewriter.create<scf::ForOp>(loc, maskStart, maskEnd, c1, c0);
      col = colLoop3f.getInductionVar();
      rewriter.setInsertionPointToStart(colLoop3f.getBody());
      col64 = rewriter.create<IndexCastOp>(loc, col, int64Type);
  }

  Value offset = colLoop3f.getLoopBody().getArgument(1);
  Value colPlus1 = rewriter.create<AddIOp>(loc, col, c1);
  Value iStart64 = rewriter.create<memref::LoadOp>(loc, iterPointers, col);
  Value iEnd64 = rewriter.create<memref::LoadOp>(loc, iterPointers, colPlus1);
  Value iStart = rewriter.create<IndexCastOp>(loc, iStart64, indexType);
  Value iEnd = rewriter.create<IndexCastOp>(loc, iEnd64, indexType);

  // insert add identity block
  graphblas::YieldOp addIdentityYield = llvm::dyn_cast_or_null<graphblas::YieldOp>(extBlocks.addIdentity->getTerminator());
  rewriter.mergeBlocks(extBlocks.addIdentity, rewriter.getBlock(), {});
  Value addIdentity = addIdentityYield.values().front();
  rewriter.eraseOp(addIdentityYield);

  scf::ForOp kLoop = rewriter.create<scf::ForOp>(loc, iStart, iEnd, c1, ValueRange{addIdentity, cfalse});
  Value ii = kLoop.getInductionVar();
  Value curr = kLoop.getLoopBody().getArgument(1);
  Value alive = kLoop.getLoopBody().getArgument(2);
  rewriter.setInsertionPointToStart(kLoop.getBody());

  Value kk64 = rewriter.create<memref::LoadOp>(loc, iterIndices, ii);
  Value kk = rewriter.create<IndexCastOp>(loc, kk64, indexType);
  Value cmpPair = rewriter.create<memref::LoadOp>(loc, kvec_i1, kk);
  scf::IfOp ifBlock_cmpPair = rewriter.create<scf::IfOp>(loc, TypeRange{valueType, boolType}, cmpPair, true);
  // if cmpPair
  rewriter.setInsertionPointToStart(ifBlock_cmpPair.thenBlock());

  Value aVal = rewriter.create<memref::LoadOp>(loc, kvec, kk);
  Value bVal = rewriter.create<memref::LoadOp>(loc, iterValues, ii);

  // insert multiply operation block
  graphblas::YieldOp multYield = llvm::dyn_cast_or_null<graphblas::YieldOp>(extBlocks.mult->getTerminator());
  Value multResult = multYield.values().front();
  rewriter.eraseOp(multYield);
  rewriter.mergeBlocks(extBlocks.mult, rewriter.getBlock(), {aVal, bVal});

  // insert add operation block
  graphblas::YieldOp addYield = llvm::dyn_cast_or_null<graphblas::YieldOp>(extBlocks.add->getTerminator());
  Value addResult = addYield.values().front();
  rewriter.eraseOp(addYield);
  rewriter.mergeBlocks(extBlocks.add, rewriter.getBlock(), {curr, multResult});

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

  scf::IfOp ifBlock_newOffset = rewriter.create<scf::IfOp>(loc, indexType, notEmpty, true);
  // if not empty
  rewriter.setInsertionPointToStart(ifBlock_newOffset.thenBlock());

  // Store total in Cx
  Value cjPos = rewriter.create<AddIOp>(loc, indexOffset, offset);
  rewriter.create<memref::StoreOp>(loc, col64, outputIndices, cjPos);

  // Does total need to be transformed?
  if (extBlocks.transformOut) {
    graphblas::YieldOp yield = llvm::dyn_cast_or_null<graphblas::YieldOp>(extBlocks.transformOut->getTerminator());
    Value transformResult = yield.values().front();

    rewriter.mergeBlocks(extBlocks.transformOut, rewriter.getBlock(), {total});

    rewriter.create<memref::StoreOp>(loc, transformResult, outputValues, cjPos);
    rewriter.eraseOp(yield);
  } else {
    // write total as-is
    rewriter.create<memref::StoreOp>(loc, total, outputValues, cjPos);
  }

  // Increment offset
  Value offsetPlus1 = rewriter.create<AddIOp>(loc, offset, c1);
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
Value computeIndexOverlapSize(PatternRewriter &rewriter, bool intersect,
                              Value aPosStart, Value aPosEnd, Value Ai,
                              Value bPosStart, Value bPosEnd, Value Bi) {
  Location loc = rewriter.getUnknownLoc();

  // Types used in this function
  Type boolType = rewriter.getI1Type();
  Type indexType = rewriter.getIndexType();

  // Initial constants
  Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
  Value cfalse = rewriter.create<ConstantIntOp>(loc, 0, boolType);
  Value ctrue = rewriter.create<ConstantIntOp>(loc, 1, boolType);

  // While Loop (exit when either array is exhausted)
  scf::WhileOp whileLoop = rewriter.create<scf::WhileOp>(loc,
                                                         TypeRange{indexType, indexType, indexType, indexType, boolType, boolType, indexType},
                                                         ValueRange{aPosStart, bPosStart, c0, c0, ctrue, ctrue, c0});
  Block *before = rewriter.createBlock(&whileLoop.before(), {}, TypeRange{indexType, indexType, indexType, indexType, boolType, boolType, indexType});
  Block *after = rewriter.createBlock(&whileLoop.after(), {}, TypeRange{indexType, indexType, indexType, indexType, boolType, boolType, indexType});
  // "while" portion of the loop
  rewriter.setInsertionPointToStart(&whileLoop.before().front());
  Value posA = before->getArgument(0);
  Value posB = before->getArgument(1);
  Value validPosA = rewriter.create<CmpIOp>(loc, CmpIPredicate::ult, posA, aPosEnd);
  Value validPosB = rewriter.create<CmpIOp>(loc, CmpIPredicate::ult, posB, bPosEnd);
  Value continueLoop = rewriter.create<AndOp>(loc, validPosA, validPosB);
  rewriter.create<scf::ConditionOp>(loc, continueLoop, before->getArguments());

  // "do" portion of while loop
  rewriter.setInsertionPointToStart(&whileLoop.after().front());
  posA = after->getArgument(0);
  posB = after->getArgument(1);
  Value idxA = after->getArgument(2);
  Value idxB = after->getArgument(3);
  Value needsUpdateA = after->getArgument(4);
  Value needsUpdateB = after->getArgument(5);
  Value count = after->getArgument(6);

  // Update input index based on flag
  scf::IfOp if_updateA = rewriter.create<scf::IfOp>(loc, indexType, needsUpdateA, true);
  // if updateA
  rewriter.setInsertionPointToStart(if_updateA.thenBlock());
  Value updatedIdxA64 = rewriter.create<memref::LoadOp>(loc, Ai, posA);
  Value updatedIdxA = rewriter.create<IndexCastOp>(loc, updatedIdxA64, indexType);
  rewriter.create<scf::YieldOp>(loc, updatedIdxA);
  // else
  rewriter.setInsertionPointToStart(if_updateA.elseBlock());
  rewriter.create<scf::YieldOp>(loc, idxA);
  rewriter.setInsertionPointAfter(if_updateA);

  // Update output index based on flag
  scf::IfOp if_updateB = rewriter.create<scf::IfOp>(loc, indexType, needsUpdateB, true);
  // if updateB
  rewriter.setInsertionPointToStart(if_updateB.thenBlock());
  Value updatedIdxB64 = rewriter.create<memref::LoadOp>(loc, Bi, posB);
  Value updatedIdxB = rewriter.create<IndexCastOp>(loc, updatedIdxB64, indexType);
  rewriter.create<scf::YieldOp>(loc, updatedIdxB);
  // else
  rewriter.setInsertionPointToStart(if_updateB.elseBlock());
  rewriter.create<scf::YieldOp>(loc, idxB);
  rewriter.setInsertionPointAfter(if_updateB);

  Value newIdxA = if_updateA.getResult(0);
  Value newIdxB = if_updateB.getResult(0);
  Value idxA_lt_idxB = rewriter.create<CmpIOp>(loc, CmpIPredicate::ult, newIdxA, newIdxB);
  Value idxA_gt_idxB = rewriter.create<CmpIOp>(loc, CmpIPredicate::ugt, newIdxA, newIdxB);
  Value posAplus1 = rewriter.create<AddIOp>(loc, posA, c1);
  Value posBplus1 = rewriter.create<AddIOp>(loc, posB, c1);

  Value countplus1 = rewriter.create<AddIOp>(loc, count, c1);
  Value countForUnion;
  if (intersect) {
    countForUnion = count;
  } else {
    countForUnion = countplus1;
  }

  scf::IfOp if_onlyA = rewriter.create<scf::IfOp>(loc, TypeRange{indexType, indexType, boolType, boolType, indexType}, idxA_lt_idxB, true);
  // if onlyA
  rewriter.setInsertionPointToStart(if_onlyA.thenBlock());
  rewriter.create<scf::YieldOp>(loc, ValueRange{posAplus1, posB, ctrue, cfalse, countForUnion});
  // else
  rewriter.setInsertionPointToStart(if_onlyA.elseBlock());
  scf::IfOp if_onlyB = rewriter.create<scf::IfOp>(loc, TypeRange{indexType, indexType, boolType, boolType, indexType}, idxA_gt_idxB, true);
  // if onlyB
  rewriter.setInsertionPointToStart(if_onlyB.thenBlock());
  rewriter.create<scf::YieldOp>(loc, ValueRange{posA, posBplus1, cfalse, ctrue, countForUnion});
  // else (At this point, we know idxA == idxB)
  rewriter.setInsertionPointToStart(if_onlyB.elseBlock());
  rewriter.create<scf::YieldOp>(loc, ValueRange{posAplus1, posBplus1, ctrue, ctrue, countplus1});
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

  rewriter.create<scf::YieldOp>(loc, ValueRange{newPosA, newPosB, newIdxA, newIdxB, needsUpdateA, needsUpdateB, newCount});
  rewriter.setInsertionPointAfter(whileLoop);

  Value finalCount;
  if (!intersect) {
    // Handle remaining elements after other array is exhausted
    scf::ForOp forLoop;
    count = whileLoop.getResult(6);
    posA = whileLoop.getResult(0);
    Value remainingPosA = rewriter.create<CmpIOp>(loc, CmpIPredicate::ult, posA, aPosEnd);
    scf::IfOp if_remainingA = rewriter.create<scf::IfOp>(loc, indexType, remainingPosA, true);
    // if remainingA
    rewriter.setInsertionPointToStart(if_remainingA.thenBlock());
    Value extraIndices = rewriter.create<SubIOp>(loc, aPosEnd, posA);
    Value newCount = rewriter.create<AddIOp>(loc, count, extraIndices);
    rewriter.create<scf::YieldOp>(loc, newCount);
    // else
    rewriter.setInsertionPointToStart(if_remainingA.elseBlock());
    posB = whileLoop.getResult(1);
    Value remainingPosB = rewriter.create<CmpIOp>(loc, CmpIPredicate::ult, posB, bPosEnd);
    scf::IfOp if_remainingB = rewriter.create<scf::IfOp>(loc, indexType, remainingPosB, true);
    // if remainingB
    rewriter.setInsertionPointToStart(if_remainingB.thenBlock());
    extraIndices = rewriter.create<SubIOp>(loc, bPosEnd, posB);
    newCount = rewriter.create<AddIOp>(loc, count, extraIndices);
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
Value computeUnionAggregation(PatternRewriter &rewriter, bool intersect, std::string agg, Type valueType,
                             Value aPosStart, Value aPosEnd, Value Ai, Value Ax,
                             Value bPosStart, Value bPosEnd, Value Bi, Value Bx,
                             Value oPosStart, Value Oi, Value Ox) {
  Location loc = rewriter.getUnknownLoc();

  // Types used in this function
  Type boolType = rewriter.getI1Type();
  Type int64Type = rewriter.getI64Type();
  Type indexType = rewriter.getIndexType();

  // Initial constants
  Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
  Value ci0 = rewriter.create<ConstantIntOp>(loc, 0, int64Type);
  Value cfalse = rewriter.create<ConstantIntOp>(loc, 0, boolType);
  Value ctrue = rewriter.create<ConstantIntOp>(loc, 1, boolType);
  // TODO: change this out for AGG_IDENTITY
  Value cf0 = llvm::TypeSwitch<Type, Value>(valueType)
        .Case<IntegerType>([&](IntegerType type) { return rewriter.create<ConstantIntOp>(loc, 0, type.getWidth()); })
        .Case<FloatType>([&](FloatType type) { return rewriter.create<ConstantFloatOp>(loc, APFloat(0.0), type); });

  // While Loop (exit when either array is exhausted)
  scf::WhileOp whileLoop = rewriter.create<scf::WhileOp>(loc,
                                                         TypeRange{indexType, indexType, indexType, int64Type, int64Type, valueType, valueType, boolType, boolType},
                                                         ValueRange{aPosStart, bPosStart, oPosStart, ci0, ci0, cf0, cf0, ctrue, ctrue});
  Block *before = rewriter.createBlock(&whileLoop.before(), {}, TypeRange{indexType, indexType, indexType, int64Type, int64Type, valueType, valueType, boolType, boolType});
  Block *after = rewriter.createBlock(&whileLoop.after(), {}, TypeRange{indexType, indexType, indexType, int64Type, int64Type, valueType, valueType, boolType, boolType});
  // "while" portion of the loop
  rewriter.setInsertionPointToStart(&whileLoop.before().front());
  Value posA = before->getArgument(0);
  Value posB = before->getArgument(1);
  Value validPosA = rewriter.create<CmpIOp>(loc, CmpIPredicate::ult, posA, aPosEnd);
  Value validPosB = rewriter.create<CmpIOp>(loc, CmpIPredicate::ult, posB, bPosEnd);
  Value continueLoop = rewriter.create<AndOp>(loc, validPosA, validPosB);
  rewriter.create<scf::ConditionOp>(loc, continueLoop, before->getArguments());

  // "do" portion of while loop
  rewriter.setInsertionPointToStart(&whileLoop.after().front());
  posA = after->getArgument(0);
  posB = after->getArgument(1);
  Value posO = after->getArgument(2);
  Value idxA = after->getArgument(3);
  Value idxB = after->getArgument(4);
  Value valA = after->getArgument(5);
  Value valB = after->getArgument(6);
  Value needsUpdateA = after->getArgument(7);
  Value needsUpdateB = after->getArgument(8);

  // Update input index based on flag
  scf::IfOp if_updateA = rewriter.create<scf::IfOp>(loc, TypeRange{int64Type, valueType}, needsUpdateA, true);
  // if updateA
  rewriter.setInsertionPointToStart(if_updateA.thenBlock());
  Value updatedIdxA = rewriter.create<memref::LoadOp>(loc, Ai, posA);
  Value updatedValA = rewriter.create<memref::LoadOp>(loc, Ax, posA);
  rewriter.create<scf::YieldOp>(loc, ValueRange{updatedIdxA, updatedValA});
  // else
  rewriter.setInsertionPointToStart(if_updateA.elseBlock());
  rewriter.create<scf::YieldOp>(loc, ValueRange{idxA, valA});
  rewriter.setInsertionPointAfter(if_updateA);

  // Update output index based on flag
  scf::IfOp if_updateB = rewriter.create<scf::IfOp>(loc, TypeRange{int64Type, valueType}, needsUpdateB, true);
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
  Value idxA_lt_idxB = rewriter.create<CmpIOp>(loc, CmpIPredicate::ult, newIdxA, newIdxB);
  Value idxA_gt_idxB = rewriter.create<CmpIOp>(loc, CmpIPredicate::ugt, newIdxA, newIdxB);
  Value posAplus1 = rewriter.create<AddIOp>(loc, posA, c1);
  Value posBplus1 = rewriter.create<AddIOp>(loc, posB, c1);

  Value posOplus1 = rewriter.create<AddIOp>(loc, posO, c1);
  Value posOForUnion;
  if (intersect) {
    posOForUnion = posO;
  } else {
    posOForUnion = posOplus1;
  }

  scf::IfOp if_onlyA = rewriter.create<scf::IfOp>(loc, TypeRange{indexType, indexType, indexType, boolType, boolType}, idxA_lt_idxB, true);
  // if onlyA
  rewriter.setInsertionPointToStart(if_onlyA.thenBlock());
  if (!intersect) {
    rewriter.create<memref::StoreOp>(loc, newIdxA, Oi, posO);
    rewriter.create<memref::StoreOp>(loc, newValA, Ox, posO);
  }
  rewriter.create<scf::YieldOp>(loc, ValueRange{posAplus1, posB, posOForUnion, ctrue, cfalse});
  // else
  rewriter.setInsertionPointToStart(if_onlyA.elseBlock());
  scf::IfOp if_onlyB = rewriter.create<scf::IfOp>(loc, TypeRange{indexType, indexType, indexType, boolType, boolType}, idxA_gt_idxB, true);
  // if onlyB
  rewriter.setInsertionPointToStart(if_onlyB.thenBlock());
  if (!intersect) {
    rewriter.create<memref::StoreOp>(loc, newIdxB, Oi, posO);
    rewriter.create<memref::StoreOp>(loc, newValB, Ox, posO);
  }
  rewriter.create<scf::YieldOp>(loc, ValueRange{posA, posBplus1, posOForUnion, cfalse, ctrue});
  // else
  rewriter.setInsertionPointToStart(if_onlyB.elseBlock());
  // At this point, we know newIdxA == newIdxB
  rewriter.create<memref::StoreOp>(loc, newIdxA, Oi, posO);
  // TODO: replace this with AGG
  Value aggVal;
  if (agg == "plus") {
    aggVal = llvm::TypeSwitch<Type, Value>(valueType)
        .Case<IntegerType>([&](IntegerType type) { return rewriter.create<AddIOp>(loc, newValA, newValB); })
        .Case<FloatType>([&](FloatType type) { return rewriter.create<AddFOp>(loc, newValA, newValB); });
  } else if (agg == "min") {
    Value cmp = llvm::TypeSwitch<Type, Value>(valueType)
                    .Case<IntegerType>([&](IntegerType type)
                                       { return rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, newValA, newValB); })
                    .Case<FloatType>([&](FloatType type)
                                       { return rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT, newValA, newValB); });
    aggVal = rewriter.create<SelectOp>(loc, cmp, newValA, newValB);
  } else if (agg == "times") {
    aggVal = llvm::TypeSwitch<Type, Value>(valueType)
        .Case<IntegerType>([&](IntegerType type) { return rewriter.create<MulIOp>(loc, newValA, newValB); })
        .Case<FloatType>([&](FloatType type) { return rewriter.create<MulFOp>(loc, newValA, newValB); });
  }
  rewriter.create<memref::StoreOp>(loc, aggVal, Ox, posO);
  rewriter.create<scf::YieldOp>(loc, ValueRange{posAplus1, posBplus1, posOplus1, ctrue, ctrue});
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

  rewriter.create<scf::YieldOp>(loc, ValueRange{newPosA, newPosB, newPosO, newIdxA, newIdxB, newValA, newValB, needsUpdateA, needsUpdateB});
  rewriter.setInsertionPointAfter(whileLoop);

  Value finalPosO;
  if (!intersect) {
    // For loop (remaining elements after other array is exhausted)
    scf::ForOp forLoop;
    posO = whileLoop.getResult(2);
    posA = whileLoop.getResult(0);
    Value remainingPosA = rewriter.create<CmpIOp>(loc, CmpIPredicate::ult, posA, aPosEnd);
    scf::IfOp if_remainingA = rewriter.create<scf::IfOp>(loc, indexType, remainingPosA, true);
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
    Value newPosO = rewriter.create<AddIOp>(loc, currPosO, c1);
    rewriter.create<scf::YieldOp>(loc, newPosO);
    rewriter.setInsertionPointAfter(forLoop);
    rewriter.create<scf::YieldOp>(loc, forLoop.getResult(0));
    // else
    rewriter.setInsertionPointToStart(if_remainingA.elseBlock());
    posB = whileLoop.getResult(1);
    Value remainingPosB = rewriter.create<CmpIOp>(loc, CmpIPredicate::ult, posB, bPosEnd);
    scf::IfOp if_remainingB = rewriter.create<scf::IfOp>(loc, indexType, remainingPosB, true);
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
    newPosO = rewriter.create<AddIOp>(loc, currPosO, c1);
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

void computeVectorElementWise(PatternRewriter &rewriter, ModuleOp module, Value lhs, Value rhs, Value output, std::string op, bool intersect) {
  Location loc = rewriter.getUnknownLoc();

  // Types
  RankedTensorType outputType = output.getType().dyn_cast<RankedTensorType>();
  Type int64Type = rewriter.getIntegerType(64);
  Type valueType = outputType.getElementType();
  MemRefType memref1DI64Type = MemRefType::get({-1}, int64Type);
  MemRefType memref1DValueType = MemRefType::get({-1}, valueType);

  // Initial constants
  Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);

  // Get sparse tensor info
  Value lhsNnz = rewriter.create<graphblas::NumValsOp>(loc, lhs);
  Value rhsNnz = rewriter.create<graphblas::NumValsOp>(loc, rhs);
  Value Li = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, lhs, c0);
  Value Lx = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, lhs);
  Value Ri = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, rhs, c0);
  Value Rx = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, rhs);

  Value ewiseSize = computeIndexOverlapSize(rewriter, intersect, c0, lhsNnz, Li, c0, rhsNnz, Ri);
  Value ewiseSize64 = rewriter.create<IndexCastOp>(loc, ewiseSize, int64Type);

  callResizeIndex(rewriter, module, loc, output, c0, ewiseSize);
  callResizeValues(rewriter, module, loc, output, ewiseSize);

  Value Op = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, output, c0);
  rewriter.create<memref::StoreOp>(loc, ewiseSize64, Op, c1);
  Value Oi = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, output, c0);
  Value Ox = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, output);

  computeUnionAggregation(rewriter, intersect, op, valueType,
                          c0, lhsNnz, Li, Lx, c0, rhsNnz, Ri, Rx, c0, Oi, Ox);
}

void computeMatrixElementWise(PatternRewriter &rewriter, ModuleOp module, Value lhs, Value rhs, Value output, std::string op, bool intersect) {
  Location loc = rewriter.getUnknownLoc();

  // Types
  RankedTensorType outputType = output.getType().dyn_cast<RankedTensorType>();
  Type indexType = rewriter.getIndexType();
  Type int64Type = rewriter.getIntegerType(64);
  Type valueType = outputType.getElementType();
  MemRefType memref1DI64Type = MemRefType::get({-1}, int64Type);
  MemRefType memref1DValueType = MemRefType::get({-1}, valueType);

  // Initial constants
  Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
  Value ci0 = rewriter.create<ConstantIntOp>(loc, 0, int64Type);

  Value nrows = rewriter.create<graphblas::NumRowsOp>(loc, output);

  // Get sparse tensor info
  Value Lp = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, lhs, c1);
  Value Li = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, lhs, c1);
  Value Lx = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, lhs);
  Value Rp = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, rhs, c1);
  Value Ri = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, rhs, c1);
  Value Rx = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, rhs);
  Value Op = rewriter.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, output, c1);

  // 1st pass
  //   Compute overlap size for each row
  //   Store results in Op
  scf::ParallelOp rowLoop1 = rewriter.create<scf::ParallelOp>(loc, c0, nrows, c1);
  Value row = rowLoop1.getInductionVars()[0];
  rewriter.setInsertionPointToStart(rowLoop1.getBody());

  Value rowPlus1 = rewriter.create<AddIOp>(loc, row, c1);
  Value lhsColStart64 = rewriter.create<memref::LoadOp>(loc, Lp, row);
  Value lhsColEnd64 = rewriter.create<memref::LoadOp>(loc, Lp, rowPlus1);
  Value rhsColStart64 = rewriter.create<memref::LoadOp>(loc, Rp, row);
  Value rhsColEnd64 = rewriter.create<memref::LoadOp>(loc, Rp, rowPlus1);
  Value LcmpColSame = rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, lhsColStart64, lhsColEnd64);
  Value RcmpColSame = rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, rhsColStart64, rhsColEnd64);
  Value emptyRow;
  if (intersect) {
    emptyRow = rewriter.create<OrOp>(loc, LcmpColSame, RcmpColSame);
  } else {
    emptyRow = rewriter.create<AndOp>(loc, LcmpColSame, RcmpColSame);
  }

  scf::IfOp ifBlock_rowTotal = rewriter.create<scf::IfOp>(loc, int64Type, emptyRow, true);
  // if cmpColSame
  rewriter.setInsertionPointToStart(ifBlock_rowTotal.thenBlock());
  rewriter.create<scf::YieldOp>(loc, ci0);

  // else
  rewriter.setInsertionPointToStart(ifBlock_rowTotal.elseBlock());
  Value lhsColStart = rewriter.create<IndexCastOp>(loc, lhsColStart64, indexType);
  Value lhsColEnd = rewriter.create<IndexCastOp>(loc, lhsColEnd64, indexType);
  Value rhsColStart = rewriter.create<IndexCastOp>(loc, rhsColStart64, indexType);
  Value rhsColEnd = rewriter.create<IndexCastOp>(loc, rhsColEnd64, indexType);
  Value unionSize = computeIndexOverlapSize(rewriter, intersect, lhsColStart, lhsColEnd, Li, rhsColStart, rhsColEnd, Ri);
  Value unionSize64 = rewriter.create<IndexCastOp>(loc, unionSize, int64Type);
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
  Value cumsum2 = rewriter.create<AddIOp>(loc, cumsum, csTemp);
  rewriter.create<memref::StoreOp>(loc, cumsum2, Op, nrows);

  // end row loop
  rewriter.setInsertionPointAfter(rowLoop2);

  Value nnz = rewriter.create<graphblas::NumValsOp>(loc, output);
  callResizeIndex(rewriter, module, loc, output, c1, nnz);
  callResizeValues(rewriter, module, loc, output, nnz);

  Value Oi = rewriter.create<sparse_tensor::ToIndicesOp>(loc, memref1DI64Type, output, c1);
  Value Ox = rewriter.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, output);

  // 3rd pass
  //   In parallel over the rows, compute the aggregation
  //   Store in Oi and Ox
  scf::ParallelOp rowLoop3 = rewriter.create<scf::ParallelOp>(loc, c0, nrows, c1);
  row = rowLoop3.getInductionVars()[0];
  rewriter.setInsertionPointToStart(rowLoop3.getBody());

  rowPlus1 = rewriter.create<AddIOp>(loc, row, c1);
  Value opStart64 = rewriter.create<memref::LoadOp>(loc, Op, row);
  Value opEnd64 = rewriter.create<memref::LoadOp>(loc, Op, rowPlus1);
  Value cmp_opDifferent = rewriter.create<CmpIOp>(loc, CmpIPredicate::ne, opStart64, opEnd64);
  scf::IfOp ifBlock_cmpDiff = rewriter.create<scf::IfOp>(loc, cmp_opDifferent);
  rewriter.setInsertionPointToStart(ifBlock_cmpDiff.thenBlock());

  Value OcolStart64 = rewriter.create<memref::LoadOp>(loc, Op, row);
  Value OcolStart = rewriter.create<IndexCastOp>(loc, OcolStart64, indexType);

  lhsColStart64 = rewriter.create<memref::LoadOp>(loc, Lp, row);
  lhsColEnd64 = rewriter.create<memref::LoadOp>(loc, Lp, rowPlus1);
  lhsColStart = rewriter.create<IndexCastOp>(loc, lhsColStart64, indexType);
  lhsColEnd = rewriter.create<IndexCastOp>(loc, lhsColEnd64, indexType);
  rhsColStart64 = rewriter.create<memref::LoadOp>(loc, Rp, row);
  rhsColEnd64 = rewriter.create<memref::LoadOp>(loc, Rp, rowPlus1);
  rhsColStart = rewriter.create<IndexCastOp>(loc, rhsColStart64, indexType);
  rhsColEnd = rewriter.create<IndexCastOp>(loc, rhsColEnd64, indexType);

  computeUnionAggregation(rewriter, intersect, op, valueType,
                          lhsColStart, lhsColEnd, Li, Lx, rhsColStart, rhsColEnd, Ri, Rx, OcolStart, Oi, Ox);

  // end if cmpDiff
  rewriter.setInsertionPointAfter(ifBlock_cmpDiff);

  // end row loop
  rewriter.setInsertionPointAfter(rowLoop3);
}
