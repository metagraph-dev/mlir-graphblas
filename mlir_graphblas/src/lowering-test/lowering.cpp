//  g++ -g $(llvm-config --cxxflags --ldflags) -Wl,-rpath $CONDA_PREFIX/lib -lLLVM -lMLIR -o lower lower.cpp && lldb -o r ./lower
#include <iostream>

#include "mlir/IR/Builders.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"

#include "lowering.h"

using namespace std;

int main()
{
    cerr << "Loading dialects..." << endl;

    auto *context = new mlir::MLIRContext();
    context->loadAllAvailableDialects();
    context->getOrLoadDialect<mlir::sparse_tensor::SparseTensorDialect>();
    context->getOrLoadDialect<mlir::StandardOpsDialect>();

    cerr << "Building empty module..." << endl;

    auto builder = mlir::OpBuilder(context);
    auto mod = mlir::ModuleOp::create(builder.getUnknownLoc());

    // Transpose
    cerr << "Adding transpose[swap_sizes=True] function..." << endl;
    addTransposeFunc(mod, true);

    cerr << "Adding transpose[swap_sizes=False] function..." << endl;
    addTransposeFunc(mod, false);

    // Matrix Select
    cerr << "Adding matrix_select[triu] function..." << endl;
    addMatrixSelectFunc(mod, "triu");

    cerr << "Adding matrix_select[tril] function..." << endl;
    addMatrixSelectFunc(mod, "tril");

    cerr << "Adding matrix_select[gt0] function..." << endl;
    addMatrixSelectFunc(mod, "gt0");

    // Matrix Reduce To Scalar
    cerr << "Adding matrix_reduce_to_scalar[sum] function..." << endl;
    addMatrixReduceToScalarFunc(mod, "sum");

    // Matrix Apply
    cerr << "Adding matrix_apply[min] function..." << endl;
    addMatrixApplyFunc(mod, "min");

    // Matrix Multiply
    cerr << "Adding matrix_multiply[plus_times, mask=False] function..." << endl;
    addMatrixMultiplyFunc(mod, "plus_times", false);

    cerr << "Adding matrix_multiply[plus_pair, mask=False] function..." << endl;
    addMatrixMultiplyFunc(mod, "plus_pair", false);

    cerr << "Adding matrix_multiply[plus_plus, mask=False] function..." << endl;
    addMatrixMultiplyFunc(mod, "plus_plus", false);

    cerr << "Adding matrix_multiply[plus_times, mask=True] function..." << endl;
    addMatrixMultiplyFunc(mod, "plus_times", true);

    cerr << "Adding matrix_multiply[plus_pair, mask=True] function..." << endl;
    addMatrixMultiplyFunc(mod, "plus_pair", true);

    cerr << "Adding matrix_multiply[plus_plus, mask=True] function..." << endl;
    addMatrixMultiplyFunc(mod, "plus_plus", true);

    cerr << "Dumping completed module..." << endl;

    mod.dump();
}
