//===- GraphBLASOps.td - GraphBLAS dialect ops -----------*- tablegen -*-===//
//
// TODO add documentation
//
//===--------------------------------------------------------------------===//

#ifndef GRAPHBLAS_OPS
#define GRAPHBLAS_OPS

include "GraphBLASDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def GraphBLAS_FooOp : GraphBLAS_Op<"foo", [NoSideEffect,
                                             SameOperandsAndResultType]> {
    let summary = "TODO add documentation";
    let description = [{
        TODO add documentation
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

#endif // GRAPHBLAS_OPS
