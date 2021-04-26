//===- GraphBLASDialect.cpp - GraphBLAS dialect ---------------*- C++ -*-===//
//
// TODO add documentation
//
//===--------------------------------------------------------------------===//

#include "GraphBLAS/GraphBLASDialect.h"
#include "GraphBLAS/GraphBLASOps.h"

using namespace mlir;
using namespace mlir::graphblas;

//===--------------------------------------------------------------------===//
// GraphBLAS dialect.
//===--------------------------------------------------------------------===//

void GraphBLASDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "GraphBLAS/GraphBLASOps.cpp.inc"
      >();
}
