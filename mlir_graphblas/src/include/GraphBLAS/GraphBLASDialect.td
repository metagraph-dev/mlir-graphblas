//===- GraphBLASDialect.td - GraphBLAS dialect -----------*- tablegen -*-===//
//
// TODO add doc string
//
//===--------------------------------------------------------------------===//

#ifndef GRAPHBLAS_DIALECT
#define GRAPHBLAS_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// GraphBLAS dialect definition.
//===----------------------------------------------------------------------===//

def GraphBLAS_Dialect : Dialect {
    let name = "graphblas";
    let summary = "An MLIR dialect for GraphBLAS.";
    let description = [{
        This dialect is designed to support GraphBLAS.
    }];
    let cppNamespace = "::mlir::graphblas";
}

//===----------------------------------------------------------------------===//
// Base graphblas operation definition.
//===----------------------------------------------------------------------===//

class GraphBLAS_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<GraphBLAS_Dialect, mnemonic, traits>;

#endif // GRAPHBLAS_DIALECT
