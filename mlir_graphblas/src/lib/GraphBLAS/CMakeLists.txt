add_mlir_dialect_library(MLIRGraphBLAS
        GraphBLASDialect.cpp
        GraphBLASOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/GraphBLAS

        DEPENDS
        MLIRGraphBLASOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
