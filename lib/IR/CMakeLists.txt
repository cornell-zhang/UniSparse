add_mlir_dialect_library(MLIRSparlay
        SparlayDialect.cpp
        SparlayOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Sparlay

        DEPENDS
        MLIRSparlayOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
