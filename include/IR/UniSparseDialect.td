//===- UniSparseDialect.td - UniSparse dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef UNISPARSE_DIALECT
#define UNISPARSE_DIALECT

include "mlir/IR/OpBase.td"

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// UniSparse dialect definition.
//===----------------------------------------------------------------------===//

def UniSparse_Dialect : Dialect {
    let name = "unisparse";
    let summary = "A unisparse out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::unisparse";
    let extraClassDeclaration = [{
        void registerTypes();
    }];
    let useDefaultAttributePrinterParser = 1;
    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base unisparse type definition.
//===----------------------------------------------------------------------===//

def UniSparse_StructType :
    DialectType<UniSparse_Dialect, CPred<"$_self.isa<StructType>()">,
                "UniSparse struct type">;

def UniSparse_Type : AnyTypeOf<[AnyType, AnyTensor, AnyMemRef, UniSparse_StructType]>;

#endif // UNISPARSE_DIALECT
