add_mlir_dialect_library(MLIRSparlay
        SparlayDialect.cpp
        SparlayOps.cpp
        SparlayTypes.cpp
        SparlayParser.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/IR

        DEPENDS
        MLIRSparlayOpsIncGen
        MLIRSparselayAttrIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRDialect
        MLIRParser
        MLIRInferTypeOpInterface
        MLIRSupport
	)
