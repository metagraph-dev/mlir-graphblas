//===- GraphBLASDialect.td - GraphBLAS dialect -----------*- tablegen -*-===//
//
// TODO add documentation
//
//===--------------------------------------------------------------------===//

#ifndef GRAPHBLAS_DIALECT
#define GRAPHBLAS_DIALECT

include "mlir/IR/OpBase.td"

//===--------------------------------------------------------------------===//
// GraphBLAS dialect definition.
//===--------------------------------------------------------------------===//

def GraphBLAS_Dialect : Dialect {
    let name = "graphblas";
    let summary = "TODO add documentation";
    let description = [{
        TODO add documentation
    }];
    let cppNamespace = "::mlir::graphblas";
}

#endif // GRAPHBLAS_DIALECT
