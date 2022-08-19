//===- SparlayDialect.td - Sparlay dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SPARLAY_DIALECT
#define SPARLAY_DIALECT

include "mlir/IR/OpBase.td"

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Sparlay dialect definition.
//===----------------------------------------------------------------------===//

def Sparlay_Dialect : Dialect {
    let name = "sparlay";
    let summary = "A sparlay out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::sparlay";
    let extraClassDeclaration = [{
        void registerTypes();
    }];
    let useDefaultAttributePrinterParser = 1;
    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base sparlay type definition.
//===----------------------------------------------------------------------===//

def Sparlay_StructType :
    DialectType<Sparlay_Dialect, CPred<"$_self.isa<StructType>()">,
                "Sparlay struct type">;

def Sparlay_Type : AnyTypeOf<[AnyType, AnyTensor, AnyMemRef, Sparlay_StructType]>;

#endif // SPARLAY_DIALECT
