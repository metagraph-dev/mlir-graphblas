add_mlir_dialect_library(MLIRGraphBLAS
        GraphBLASDialect.cpp
        GraphBLASOps.cpp
        GraphBLASLowerPass.cpp
        GraphBLASLinalgLowerPass.cpp
        GraphBLASStructuralizePass.cpp
        GraphBLASOptimizePass.cpp
        GraphBLASUtils.cpp
        GraphBLASArrayUtils.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/GraphBLAS

        DEPENDS
        MLIRGraphBLASOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        )
