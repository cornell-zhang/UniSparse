//===- unisparse-translate.cpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that translates a file from/to MLIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//

#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

#include "IR/UniSparseDialect.h"
#include "Translation/EmitTapaHLS.h"

int main(int argc, char **argv) {
  mlir::registerAllTranslations();

  // TODO: Register unisparse translations here.
  mlir::unisparse::registerEmitTapaHLSTranslation();

  return failed(
      mlir::mlirTranslateMain(argc, argv, 
        "UniSparse MLIR Dialect Translation Tool"));
}
