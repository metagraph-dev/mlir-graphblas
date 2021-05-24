//===- GraphBLASOps.td - GraphBLAS dialect ops -----------*- tablegen -*-===//
//
// TODO add documentation
//
//===--------------------------------------------------------------------===//

#ifndef GRAPHBLAS_OPS
#define GRAPHBLAS_OPS

include "GraphBLASDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def GraphBLAS_FooOp : GraphBLAS_Op<"foo", [//NoSideEffect,
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

def GraphBLAS_TransposeOp : GraphBLAS_Op<"transpose", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "transpose operation";
    let description = [{
        Returns a new sparse tensor that's the transpose of the original sparse tensor.

        TODO Add example
    }];

    let arguments = (ins AnyTensor:$input, BoolAttr:$swap_sizes);
    let results = (outs AnyTensor:$output);
    
    let assemblyFormat = [{
           $input attr-dict `:` type($input) `to` type($output)
    }];

    // TODO add custsom verifier sanity checking the input and output types are sane
    // let verifier = 
}

#endif // GRAPHBLAS_OPS
