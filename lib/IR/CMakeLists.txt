add_mlir_dialect_library(MLIRUniSparse
        UniSparseDialect.cpp
        UniSparseOps.cpp
        UniSparseTypes.cpp
        UniSparseParser.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/IR

        DEPENDS
        MLIRUniSparseOpsIncGen
        MLIRUniSparseAttrIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRDialect
        MLIRParser
        MLIRInferTypeOpInterface
        MLIRSupport
	)
