get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
	MLIRIR
        MLIROptLib
        MLIRGraphBLAS
        )
add_llvm_executable(lowering-test 
        lowering.cpp
        MatrixApply.cpp
        MatrixMultiply.cpp 
        MatrixReduceToScalar.cpp
        MatrixSelect.cpp
        Transpose.cpp
        util.cpp
        )

llvm_update_compile_flags(lowering-test)
target_link_libraries(lowering-test PRIVATE ${LIBS})

mlir_check_all_link_libraries(lowering-test lowering-test)

