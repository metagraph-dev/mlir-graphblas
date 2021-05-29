//===- GraphBLASPasses.cpp - GraphBLAS dialect passes ---------*- C++ -*-===//
//
// TODO add documentation
//
//===--------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "GraphBLAS/GraphBLASPasses.h"

#include <iostream> // TODO remove this

using namespace ::mlir;

namespace {

//===----------------------------------------------------------------------===//
// Passes Implementation Helpers.
//===----------------------------------------------------------------------===//

mlir::RankedTensorType getCSRTensorType(MLIRContext *context, Type valueType) {
    SmallVector<sparse_tensor::SparseTensorEncodingAttr::DimLevelType, 2> dlt;
    dlt.push_back(sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Dense);
    dlt.push_back(sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Compressed);
    unsigned ptr = 64;
    unsigned ind = 64;
    AffineMap map = AffineMap::getMultiDimIdentityMap(2, context);

    RankedTensorType csrTensor = RankedTensorType::get(
        {-1, -1}, /* 2D, unknown size */
        valueType,
        sparse_tensor::SparseTensorEncodingAttr::get(context, dlt, map, ptr, ind));

    return csrTensor;
}

//===----------------------------------------------------------------------===//
// Passes declaration.
//===----------------------------------------------------------------------===//

#define GEN_PASS_CLASSES
#include "GraphBLAS/GraphBLASPasses.h.inc"

//===----------------------------------------------------------------------===//
// Passes implementation.
//===----------------------------------------------------------------------===//

class LowerMatrixMultiplyRewrite : public OpRewritePattern<graphblas::MatrixMultiplyOp> {
public:
  using OpRewritePattern<graphblas::MatrixMultiplyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::MatrixMultiplyOp op, PatternRewriter &rewriter) const {
    // TODO there should be a failure  case here
    
    MLIRContext *context = op->getContext();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    
    // TODO get the types from the inputs during the "match" part of this func
    Type valueType = rewriter.getI64Type();
    RankedTensorType csrTensorType = getCSRTensorType(context, valueType);

    llvm::StringRef semi_ring = op.semiring();
    std::string func_name = "matrix_multiply_" + semi_ring.str();
    FuncOp func = module.lookupSymbol<FuncOp>(func_name);
    if (!func) {
      OpBuilder moduleBuilder(module.getBodyRegion());
      FunctionType func_type = FunctionType::get(context, {csrTensorType, csrTensorType}, csrTensorType);
      moduleBuilder.create<FuncOp>(op->getLoc(), func_name, func_type).setPrivate();
    }
    FlatSymbolRefAttr funcSymbol = SymbolRefAttr::get(context, func_name);
    
    Value a = op.a();
    Value b = op.b();
    Location loc = rewriter.getUnknownLoc();
    
    CallOp callOp = rewriter.create<CallOp>(loc,
						funcSymbol,
						csrTensorType,
						llvm::ArrayRef<Value>({a, b})
						);
    
    rewriter.replaceOp(op, callOp->getResults());
    
    return success();
  };
};

class LowerMatrixReduceToScalarRewrite : public OpRewritePattern<graphblas::MatrixReduceToScalarOp> {
public:
  using OpRewritePattern<graphblas::MatrixReduceToScalarOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(graphblas::MatrixReduceToScalarOp op, PatternRewriter &rewriter) const {
    // TODO there should be a "return failure();" somewhere
    MLIRContext *context = op->getContext();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = rewriter.getUnknownLoc();

    // TODO get the types from the inputs during the "match" part of this func
    auto valueType = rewriter.getF64Type();
    Type int64Type = rewriter.getIntegerType(64);
    Type indexType = rewriter.getIndexType();
    RankedTensorType csrTensorType = getCSRTensorType(context, valueType);

    std::string aggregator = op.aggregator().str();
    std::string func_name = "matrix_reduce_to_scalar_" + aggregator;
    FuncOp func = module.lookupSymbol<FuncOp>(func_name);
    if (!func) {
      OpBuilder moduleBuilder(module.getBodyRegion());
      
      FunctionType func_type = FunctionType::get(context, {csrTensorType}, valueType);
      moduleBuilder.create<FuncOp>(op->getLoc(), func_name, func_type).setPrivate();
      func = module.lookupSymbol<FuncOp>(func_name);
      Block &entry_block = *func.addEntryBlock();
      moduleBuilder.setInsertionPointToStart(&entry_block);
      BlockArgument input = entry_block.getArgument(0);
      
      // Initial constants
      ConstantFloatOp cf0 = moduleBuilder.create<ConstantFloatOp>(loc, APFloat(0.0), valueType); // TODO this should change according to valueType
      ConstantIndexOp c0 = moduleBuilder.create<ConstantIndexOp>(loc, 0);
      ConstantIndexOp c1 = moduleBuilder.create<ConstantIndexOp>(loc, 1);
      
      // Get sparse tensor info
      MemRefType memref1DI64Type = MemRefType::get({-1}, int64Type);
      MemRefType memref1DValueType = MemRefType::get({-1}, valueType);

      memref::DimOp nrows = moduleBuilder.create<memref::DimOp>(loc, input, c0.getResult());
      sparse_tensor::ToPointersOp inputPtrs = moduleBuilder.create<sparse_tensor::ToPointersOp>(loc, memref1DI64Type, input, c1);
      sparse_tensor::ToValuesOp inputValues = moduleBuilder.create<sparse_tensor::ToValuesOp>(loc, memref1DValueType, input);
      memref::LoadOp nnz64 = moduleBuilder.create<memref::LoadOp>(loc, inputPtrs, nrows.getResult());
      IndexCastOp nnz = moduleBuilder.create<IndexCastOp>(loc, nnz64, indexType);

      // begin loop
      scf::ParallelOp valueLoop = moduleBuilder.create<scf::ParallelOp>(loc, c0.getResult(), nnz.getResult(), c1.getResult(), cf0.getResult());
      ValueRange valueLoopIdx = valueLoop.getInductionVars();

      moduleBuilder.setInsertionPointToStart(valueLoop.getBody());
      memref::LoadOp y = moduleBuilder.create<memref::LoadOp>(loc, inputValues, valueLoopIdx);

      scf::ReduceOp reducer = moduleBuilder.create<scf::ReduceOp>(loc, y);
      BlockArgument lhs = reducer.getRegion().getArgument(0);
      BlockArgument rhs = reducer.getRegion().getArgument(1);

      moduleBuilder.setInsertionPointToStart(&reducer.getRegion().front());

      Value z;
      if (aggregator == "sum") {
        AddFOp zOp = moduleBuilder.create<AddFOp>(loc, lhs, rhs);
        z = zOp.getResult();
      }
      moduleBuilder.create<scf::ReduceReturnOp>(loc, z);

      moduleBuilder.setInsertionPointAfter(reducer);

      // end loop
      moduleBuilder.setInsertionPointAfter(valueLoop);

      // Add return op
      moduleBuilder.create<ReturnOp>(loc, valueLoop.getResult(0));
    }
    FlatSymbolRefAttr funcSymbol = SymbolRefAttr::get(context, func_name);
    
    Value inputTensor = op.input();
    CallOp callOp = rewriter.create<CallOp>(loc,
    						funcSymbol,
    						valueType,
    						llvm::ArrayRef<Value>({inputTensor})
    						);    
    rewriter.replaceOp(op, callOp->getResults());
    
    return success();
  };
};

void populateGraphBLASLoweringPatterns(RewritePatternSet &patterns) {
  patterns.add<
    LowerMatrixReduceToScalarRewrite,
    LowerMatrixMultiplyRewrite
    >(patterns.getContext());
}

struct GraphBLASLoweringPass : public GraphBLASLoweringBase<GraphBLASLoweringPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);
    populateGraphBLASLoweringPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // end anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createGraphBLASLoweringPass() {
  return std::make_unique<GraphBLASLoweringPass>();
}
