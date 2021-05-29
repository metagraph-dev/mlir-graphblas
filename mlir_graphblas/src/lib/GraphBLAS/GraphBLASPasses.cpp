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
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/None.h"

#include "GraphBLAS/GraphBLASPasses.h"

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
    
    MLIRContext *context = op->getContext();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    
    Type valueType = rewriter.getI64Type();
    RankedTensorType csrTensorType = getCSRTensorType(context, valueType);

    std::string funcName = "matrix_multiply_" + op.semiring().str();
    FuncOp func = module.lookupSymbol<FuncOp>(funcName);
    if (!func) {
      OpBuilder moduleBuilder(module.getBodyRegion());
      FunctionType funcType = FunctionType::get(context, {csrTensorType, csrTensorType}, csrTensorType);
      moduleBuilder.create<FuncOp>(op->getLoc(), funcName, funcType).setPrivate();
    }
    FlatSymbolRefAttr funcSymbol = SymbolRefAttr::get(context, funcName);
    
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

    ValueTypeRange<OperandRange> operandTypes  = op->getOperandTypes();
    Type valueType = operandTypes.front().dyn_cast<TensorType>().getElementType(); 
    Type int64Type = rewriter.getIntegerType(64);
    Type indexType = rewriter.getIndexType();
    RankedTensorType csrTensorType = getCSRTensorType(context, valueType);

    std::string funcName = "matrix_reduce_to_scalar_";
    llvm::raw_string_ostream stream(funcName);
    std::string aggregator = op.aggregator().str();
    stream <<  aggregator << "_elem_";
    valueType.print(stream);
    stream.flush();
    
    FuncOp func = module.lookupSymbol<FuncOp>(funcName);
    if (!func) {
      OpBuilder moduleBuilder(module.getBodyRegion());
      
      FunctionType funcType = FunctionType::get(context, {csrTensorType}, valueType);
      moduleBuilder.create<FuncOp>(op->getLoc(), funcName, funcType).setPrivate();
      func = module.lookupSymbol<FuncOp>(funcName);
      Block &entry_block = *func.addEntryBlock();
      moduleBuilder.setInsertionPointToStart(&entry_block);
      BlockArgument input = entry_block.getArgument(0);
      
      // Initial constants
      llvm::Optional<ConstantOp> c0Accumulator = llvm::TypeSwitch<Type, llvm::Optional<ConstantOp>>(valueType)
        .Case<IntegerType>([&](IntegerType type) {
                             return moduleBuilder.create<ConstantIntOp>(loc, 0, type.getWidth());
                           })
        .Case<FloatType>([&](FloatType type) {
                           return moduleBuilder.create<ConstantFloatOp>(loc, APFloat(type.getFloatSemantics()), type);
                         })
        .Default([&](Type type) { return llvm::None; });
      if (!c0Accumulator.hasValue()) {
        return failure(); // TODO test this case
      }
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
      scf::ParallelOp valueLoop = moduleBuilder.create<scf::ParallelOp>(loc, c0.getResult(), nnz.getResult(), c1.getResult(), c0Accumulator.getValue().getResult());
      ValueRange valueLoopIdx = valueLoop.getInductionVars();

      moduleBuilder.setInsertionPointToStart(valueLoop.getBody());
      memref::LoadOp y = moduleBuilder.create<memref::LoadOp>(loc, inputValues, valueLoopIdx);

      scf::ReduceOp reducer = moduleBuilder.create<scf::ReduceOp>(loc, y);
      BlockArgument lhs = reducer.getRegion().getArgument(0);
      BlockArgument rhs = reducer.getRegion().getArgument(1);

      moduleBuilder.setInsertionPointToStart(&reducer.getRegion().front());

      llvm::Optional<Value> z;
      if (aggregator == "sum") {
         z = llvm::TypeSwitch<Type, llvm::Optional<Value>>(valueType)
          .Case<IntegerType>([&](IntegerType type) { return moduleBuilder.create<AddIOp>(loc, lhs, rhs).getResult(); })
          .Case<FloatType>([&](FloatType type) { return moduleBuilder.create<AddFOp>(loc, lhs, rhs).getResult(); })
          .Default([&](Type type) { return llvm::None; });
        if (!z.hasValue()) {
          return failure();
        }
      } else {
        return failure(); // TODO test this
      }
      moduleBuilder.create<scf::ReduceReturnOp>(loc, z.getValue());

      moduleBuilder.setInsertionPointAfter(reducer);

      // end loop
      moduleBuilder.setInsertionPointAfter(valueLoop);

      // Add return op
      moduleBuilder.create<ReturnOp>(loc, valueLoop.getResult(0));
    }
    FlatSymbolRefAttr funcSymbol = SymbolRefAttr::get(context, funcName);
    
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
