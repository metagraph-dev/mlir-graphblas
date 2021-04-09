//===- GraphBLASOps.td - GraphBLAS dialect ops -----------*- tablegen -*-===//
//
// TODO add doc string
//
//===--------------------------------------------------------------------===//

#ifndef GRAPHBLAS_OPS
#define GRAPHBLAS_OPS

include "GraphBLASDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def GraphBLAS_FooOp : GraphBLAS_Op<"foo", [NoSideEffect,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `graphblas.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = constant 2 : i32
        // Apply the foo operation to %0
        %1 = graphblas.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

#endif // GRAPHBLAS_OPS
