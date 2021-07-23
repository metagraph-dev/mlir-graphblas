
set(LLVM_TARGET_DEFINITIONS GraphBLASPasses.td)
mlir_tablegen(GraphBLASPasses.h.inc -gen-pass-decls -name GraphBLAS)
add_public_tablegen_target(MLIRGraphBLASIncGen)
add_mlir_doc(GraphBLASPasses GraphBLASPasses ./ -gen-pass-doc)

set(LLVM_TARGET_DEFINITIONS GraphBLASOps.td)
mlir_tablegen(GraphBLASOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(GraphBLASOpsEnums.cpp.inc -gen-enum-defs)

add_mlir_dialect(GraphBLASOps graphblas)

add_mlir_doc(GraphBLASDialect -gen-dialect-doc GraphBLASDialect GraphBLAS/)
add_mlir_doc(GraphBLASOps -gen-op-doc GraphBLASOps GraphBLAS/)
