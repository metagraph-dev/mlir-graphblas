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

mlir::RankedTensorType getCSRTensorType(mlir::MLIRContext *context, mlir::Type valueType) {
    SmallVector<mlir::sparse_tensor::SparseTensorEncodingAttr::DimLevelType, 2> dlt;
    dlt.push_back(mlir::sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Dense);
    dlt.push_back(mlir::sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Compressed);
    unsigned ptr = 64;
    unsigned ind = 64;
    AffineMap map = AffineMap::getMultiDimIdentityMap(2, context);

    RankedTensorType csrTensor = RankedTensorType::get(
        {-1, -1}, /* 2D, unknown size */
        valueType,
        mlir::sparse_tensor::SparseTensorEncodingAttr::get(context, dlt, map, ptr, ind));

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

class LowerMatrixMultiplyRewrite : public OpRewritePattern<mlir::graphblas::MatrixMultiplyOp> {
public:
  using OpRewritePattern<mlir::graphblas::MatrixMultiplyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::graphblas::MatrixMultiplyOp op, PatternRewriter &rewriter) const {
    // TODO there should be a failure  case here
    
    MLIRContext *context = op->getContext();
    auto module = op->getParentOfType<ModuleOp>();
    
    // TODO get the types from the inputs during the "match" part of this func
    auto valueType = rewriter.getI64Type();
    RankedTensorType csrTensorType = getCSRTensorType(context, valueType);

    llvm::StringRef semi_ring = op.semiring();
    std::string func_name = "matrix_multiply_" + semi_ring.str();
    auto func = module.lookupSymbol<FuncOp>(func_name);
    if (!func) {
      OpBuilder moduleBuilder(module.getBodyRegion());
      auto func_type = FunctionType::get(context, {csrTensorType, csrTensorType}, csrTensorType);
      moduleBuilder.create<FuncOp>(op->getLoc(), func_name, func_type).setPrivate();
    }
    auto funcSymbol = SymbolRefAttr::get(context, func_name);
    
    mlir::Value a = op.a();
    mlir::Value b = op.b();
    auto loc = rewriter.getUnknownLoc();
    
    auto callOp = rewriter.create<mlir::CallOp>(loc,
						funcSymbol,
						csrTensorType,
						llvm::ArrayRef<mlir::Value>({a, b})
						);
    
    rewriter.replaceOp(op, callOp->getResults());
    
    return success();
  };
};

class LowerMatrixReduceToScalarRewrite : public OpRewritePattern<mlir::graphblas::MatrixReduceToScalarOp> {
public:
  using OpRewritePattern<mlir::graphblas::MatrixReduceToScalarOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::graphblas::MatrixReduceToScalarOp op, PatternRewriter &rewriter) const {
    // TODO there should be a "return failure();" somewhere
    MLIRContext *context = op->getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto loc = rewriter.getUnknownLoc();

    // TODO get the types from the inputs during the "match" part of this func
    auto valueType = rewriter.getF64Type();
    auto int64Type = rewriter.getIntegerType(64);
    auto indexType = rewriter.getIndexType();
    RankedTensorType csrTensorType = getCSRTensorType(context, valueType);

    std::string aggregator = op.aggregator().str();
    std::string func_name = "matrix_reduce_to_scalar_" + aggregator;
    auto funcExists = module.lookupSymbol<FuncOp>(func_name);
    if (!funcExists) {
      OpBuilder moduleBuilder(module.getBodyRegion());
      
      auto func_type = FunctionType::get(context, {csrTensorType}, valueType);
      moduleBuilder.create<FuncOp>(op->getLoc(), func_name, func_type).setPrivate();
      auto func = module.lookupSymbol<FuncOp>(func_name);
      auto &entry_block = *func.addEntryBlock();
      moduleBuilder.setInsertionPointToStart(&entry_block);
      auto input = entry_block.getArgument(0);
      
      // Initial constants
      auto cf0 = moduleBuilder.create<ConstantFloatOp>(loc, APFloat(0.0), valueType);
      auto c0 = moduleBuilder.create<ConstantIndexOp>(loc, 0);
      auto c1 = moduleBuilder.create<ConstantIndexOp>(loc, 1);
      
      // Get sparse tensor info
      auto memref1DI64Type = MemRefType::get({-1}, int64Type);
      auto memref1DValueType = MemRefType::get({-1}, valueType);

      auto nrows = moduleBuilder.create<memref::DimOp>(loc, input, c0.getResult());
      auto inputPtrs = moduleBuilder.create<mlir::sparse_tensor::ToPointersOp>(loc, memref1DI64Type, input, c1);
      auto inputValues = moduleBuilder.create<mlir::sparse_tensor::ToValuesOp>(loc, memref1DValueType, input);
      auto nnz64 = moduleBuilder.create<memref::LoadOp>(loc, inputPtrs, nrows.getResult());
      auto nnz = moduleBuilder.create<mlir::IndexCastOp>(loc, nnz64, indexType);

      // begin loop
      auto valueLoop = moduleBuilder.create<scf::ParallelOp>(loc, c0.getResult(), nnz.getResult(), c1.getResult(), cf0.getResult());
      auto valueLoopIdx = valueLoop.getInductionVars();

      moduleBuilder.setInsertionPointToStart(valueLoop.getBody());
      auto y = moduleBuilder.create<memref::LoadOp>(loc, inputValues, valueLoopIdx);

      auto reducer = moduleBuilder.create<scf::ReduceOp>(loc, y);
      auto lhs = reducer.getRegion().getArgument(0);
      auto rhs = reducer.getRegion().getArgument(1);

      moduleBuilder.setInsertionPointToStart(&reducer.getRegion().front());

      Value z;
      if (aggregator == "sum") {
        auto zOp = moduleBuilder.create<mlir::AddFOp>(loc, lhs, rhs);
        z = zOp.getResult();
      }
      moduleBuilder.create<scf::ReduceReturnOp>(loc, z);

      moduleBuilder.setInsertionPointAfter(reducer);

      // end loop
      moduleBuilder.setInsertionPointAfter(valueLoop);

      // Add return op
      moduleBuilder.create<ReturnOp>(loc, valueLoop.getResult(0));
    }
    auto funcSymbol = SymbolRefAttr::get(context, func_name);
    
    mlir::Value inputTensor = op.input();
    mlir::Value outputTensor = op.output();
    
    auto callOp = rewriter.create<mlir::CallOp>(loc,
    						funcSymbol,
    						valueType,
    						llvm::ArrayRef<mlir::Value>({inputTensor})
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
    auto *ctx = &getContext();
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
