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
    let summary = "The `graphblas` dialect describes standard sparse tensor operations that are found in the [GraphBLAS spec](http://people.eecs.berkeley.edu/~aydin/GraphBLAS_API_C_v13.pdf).";
    let description = [{
        The ops are not one-to-one equivalents of GraphBLAS function calls in order to fit into MLIR’s SSA requirements.

        This document is not intended to be a tutorial and acts more as a reference manual for the ops in the GraphBLAS dialect. Tutorials can be found in later sections of our documentation.

        ## Assumptions
    
        Although the [sparse tensor encoding](https://mlir.llvm.org/docs/Dialects/SparseTensorOps/#sparsetensorencodingattr)
        in MLIR is extremely flexible, the ``graphblas`` dialect and associated
        lowering pass only supports three encodings currently.

        The *CSR64* encoding is usually defined with the alias:

        ```mlir
        #CSR64 = #sparse_tensor.encoding<{
          dimLevelType = [ "dense", "compressed" ],
          dimOrdering = affine_map<(i,j) -> (i,j)>,
          pointerBitWidth = 64,
          indexBitWidth = 64
        }>
        ```

        The *CSC64* encoding can be defined with the alias:

        ```mlir
        #CSC64 = #sparse_tensor.encoding<{
          dimLevelType = [ "dense", "compressed" ],
          dimOrdering = affine_map<(i,j) -> (j,i)>,
          pointerBitWidth = 64,
          indexBitWidth = 64
        }>
        ```

        In terms of data structure contents CSR and CSC are identical (with index,
        pointer, and value arrays), just the indexing is reversed for CSC.  The sparse
        tensor is then defined in the same way as a regular MLIR tensor, but with this
        additional encoding attribute:

        ```mlir
        tensor<?x?xf64, #CSC64>
        ```

        The *CV64* encoding (for sparse vectors) is usually defined with the alias:

        ```mlir
        #CV64 = #sparse_tensor.encoding<{
          dimLevelType = [ "compressed" ],
          pointerBitWidth = 64,
          indexBitWidth = 64
        }>
        ```

        Note that the `--graphblas-lower` pass only supports tensors with unknown
        dimensions (indicated by the ``?``).
    }];
    let cppNamespace = "::mlir::graphblas";
}

#endif // GRAPHBLAS_DIALECT
