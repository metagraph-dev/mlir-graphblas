#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "lowering.h"

using namespace std;
using namespace mlir;
using namespace mlir::sparse_tensor;

void addMatrixMultiplyFunc(mlir::ModuleOp mod, const std::string &semi_ring, bool mask)
{
    MLIRContext *context = mod.getContext();
    OpBuilder builder(mod.getBodyRegion());
    auto loc = builder.getUnknownLoc();

    builder.setInsertionPointToStart(mod.getBody());

    // Types
    auto valueType = builder.getF64Type();
    auto int64Type = builder.getI64Type();
    auto boolType = builder.getI1Type();
    auto indexType = builder.getIndexType();
    auto noneType = builder.getNoneType();
    RankedTensorType csrTensorType = getCSRTensorType(context, valueType);
    auto memref1DI64Type = MemRefType::get({-1}, int64Type);
    auto memref1DValueType = MemRefType::get({-1}, valueType);

    // Create function signature
    string func_name;
    FunctionType func_type;

    if (mask) {
        func_name += "matrix_multiply_mask_" + semi_ring;
        func_type = FunctionType::get(context, {csrTensorType, csrTensorType, csrTensorType}, csrTensorType);
    } else {
        func_name += "matrix_multiply_" + semi_ring;
        func_type = FunctionType::get(context, {csrTensorType, csrTensorType}, csrTensorType);
    }

    auto func = builder.create<FuncOp>(loc, func_name, func_type);

    // Move to function body
    auto &entry_block = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entry_block);

    auto A = entry_block.getArgument(0);
    auto B = entry_block.getArgument(1);
    Value M;
    if (mask) {
        M = entry_block.getArgument(2);
    }

    // Initial constants
    Value cf0 = builder.create<ConstantFloatOp>(loc, APFloat(0.0), valueType);
    Value cf1 = builder.create<ConstantFloatOp>(loc, APFloat(1.0), valueType);
    Value c0 = builder.create<ConstantIndexOp>(loc, 0);
    Value c1 = builder.create<ConstantIndexOp>(loc, 1);
    Value ci0 = builder.create<ConstantIntOp>(loc, 0, int64Type);
    Value ci1 = builder.create<ConstantIntOp>(loc, 1, int64Type);
    Value cfalse = builder.create<ConstantIntOp>(loc, 0, boolType);
    Value ctrue = builder.create<ConstantIntOp>(loc, 1, boolType);

    // Get sparse tensor info
    Value Ap = builder.create<ToPointersOp>(loc, memref1DI64Type, A, c1);
    Value Aj = builder.create<ToIndicesOp>(loc, memref1DI64Type, A, c1);
    Value Ax = builder.create<ToValuesOp>(loc, memref1DValueType, A);
    Value Bp = builder.create<ToPointersOp>(loc, memref1DI64Type, B, c1);
    Value Bi = builder.create<ToIndicesOp>(loc, memref1DI64Type, B, c1);
    Value Bx = builder.create<ToValuesOp>(loc, memref1DValueType, B);

    Value nrow = builder.create<memref::DimOp>(loc, A, c0);
    Value ncol = builder.create<memref::DimOp>(loc, B, c1);
    Value nk = builder.create<memref::DimOp>(loc, A, c1);
    Value nrow_plus_one = builder.create<AddIOp>(loc, nrow, c1);

    Value Mp, Mx;
    if (mask) {
        Mp = builder.create<ToPointersOp>(loc, memref1DI64Type, M, c1);
        Mx = builder.create<ToValuesOp>(loc, memref1DValueType, M);
    }

    Value C = callDupTensor(builder, mod, loc, A).getResult(0);
    callResizeDim(builder, mod, loc, C, c0, nrow);
    callResizeDim(builder, mod, loc, C, c1, ncol);
    callResizePointers(builder, mod, loc, C, c1, nrow_plus_one);

    Value Cp = builder.create<ToPointersOp>(loc, memref1DI64Type, C, c1);

    // 1st pass
    //   Using nested parallel loops for each row and column,
    //   compute the number of nonzero entries per row.
    //   Store results in Cp
    auto rowLoop = builder.create<scf::ParallelOp>(loc, c0, nrow, c1);
    Value row = rowLoop.getInductionVars()[0];
    builder.setInsertionPointToStart(rowLoop.getBody());

    Value colStart64 = builder.create<memref::LoadOp>(loc, Ap, row);
    Value rowPlus1 = builder.create<AddIOp>(loc, row, c1);
    Value colEnd64 = builder.create<memref::LoadOp>(loc, Ap, rowPlus1);
    Value cmpColSame = builder.create<CmpIOp>(loc, mlir::CmpIPredicate::eq, colStart64, colEnd64);

    auto ifBlock = builder.create<scf::IfOp>(loc, indexType, cmpColSame, true);
    // if cmpColSame
    builder.setInsertionPointToStart(&ifBlock.getRegion(0).front());
    builder.create<scf::YieldOp>(loc, ci0);

    // else
    builder.setInsertionPointToStart(&ifBlock.getRegion(1).front());

    // Construct a dense array indicating valid row positions
    Value colStart = builder.create<IndexCastOp>(loc, colStart64, indexType);
    Value colEnd = builder.create<IndexCastOp>(loc, colEnd64, indexType);



    // end if cmpColSame
    builder.setInsertionPointAfter(ifBlock);


    // end row loop
    builder.setInsertionPointAfter(rowLoop);

    // 2nd pass
    //   Compute the cumsum of values in Cp to build the final Cp
    //   Then resize output indices and values


    // 3rd pass
    //   In parallel over the rows,
    //   compute the nonzero columns and associated values.
    //   Store in Cj and Cx
    

    // Add return op
    builder.create<ReturnOp>(loc, C);
}