//===- graphblas-opt.cpp ---------------------------------------*- C++ -*-===//
//
// TODO add documentation
//
//===---------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "GraphBLAS/GraphBLASDialect.h"
#include "GraphBLAS/GraphBLASPasses.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  registerGraphBLASPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::graphblas::GraphBLASDialect>();
  registry.insert<mlir::StandardOpsDialect>();
  registry.insert<mlir::sparse_tensor::SparseTensorDialect>();
  registry.insert<mlir::scf::SCFDialect>(); // TODO is this needed?
  
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);

  return failed(
      mlir::MlirOptMain(argc, argv, "GraphBLAS optimizer driver\n", registry));
}
