get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRGraphBLAS
        )
add_llvm_executable(graphblas-opt graphblas-opt.cpp)

llvm_update_compile_flags(graphblas-opt)
target_link_libraries(graphblas-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(graphblas-opt)
