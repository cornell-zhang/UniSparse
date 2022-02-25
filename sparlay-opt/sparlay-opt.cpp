//===- sparlay-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorOr.h"

#include "IR/SparlayDialect.h"
#include "Transforms/Passes.h"

// using namespace sparlay;
// using namespace mlir;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input sparlay file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));
static cl::opt<std::string> outputFilename("o", 
                                           cl::desc("Output filename"), 
                                           cl::value_desc("filename"), 
                                           cl::init("-"));
static cl::opt<bool> lowerFormatConversion("lower-format-conversion",
                                           cl::init(false),
                                           cl::desc("Enable Format Lowering"));

int loadMLIR(mlir::MLIRContext &context, mlir::OwningModuleRef &module) {
  // Read the input mlir.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code EC = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << EC.message() << "\n";
    return 1;
  }

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't parse file " << inputFilename << "\n";
    return 2;
  }
  return 0;
}


int loadAndProcessMLIR(mlir::MLIRContext &context,
                       mlir::OwningModuleRef &module) {
  if (int error = loadMLIR(context, module))
    return error;
  
  mlir::PassManager pm(&context);
  // Apply any generic pass manager command line options and run the pipeline.
  applyPassManagerCLOptions(pm);
  mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();

  if (lowerFormatConversion) {
    // pm.addPass(mlir::createLowerFormatConversionPass());
    optPM.addPass(mlir::sparlay::createLowerFormatConversionPass());
  }

  if (mlir::failed(pm.run(*module)))
    return 3;
  return 0;
}

int main(int argc, char **argv) {
  // mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "MLIR Sparlay modular optimizer\n");

  mlir::MLIRContext context;

  // mlir::DialectRegistry registry = context.getDialectRegistry();
  registerAllDialects(context);
  context.loadAllAvailableDialects();

  context.getOrLoadDialect<mlir::sparlay::SparlayDialect>();

  // --- original pass registry  --- //
  mlir::registerAllPasses();
  // TODO: Register sparlay passes here.

  // mlir::DialectRegistry registry;
  // registerAllDialects(registry);
  // registry.insert<mlir::sparlay::SparlayDialect>();
  // registry.insert<mlir::StandardOpsDialect>();
  // registry.insert<mlir::tensor::TensorDialect>();
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);
  // --- original pass registry  --- //

  mlir::OwningModuleRef module;
  if (int error = loadAndProcessMLIR(context, module))
    return error;
  
  // print output
  std::string errorMessage;
  auto outfile = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!outfile) {
    llvm::errs() << errorMessage << "\n";
    return 4;
  }

  // if (failed(MlirOptMain(outfile->os(), std::move(file), passPipeline, registry,
  //                        splitInputFile, verifyDiagnostics, verifyPasses,
  //                        allowUnregisteredDialects, preloadDialectsInContext)))
  //   return failure();
  module->print(outfile->os());
  outfile->keep();

  // return mlir::asMainReturnCode(
  //     mlir::MlirOptMain(argc, argv, "Sparlay optimizer driver\n", registry));
  return 0;
}
