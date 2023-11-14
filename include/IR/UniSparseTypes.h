//===- UniSparseTypes.h - UniSparse Type Classes --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef UNISPARSE_UNISPARSETYPES_H
#define UNISPARSE_UNISPARSETYPES_H

#include "mlir/IR/Types.h"

//===----------------------------------------------------------------------===//
// UniSparse Types
//===----------------------------------------------------------------------===//
namespace mlir {
namespace unisparse {
  
namespace detail {
struct StructTypeStorage;
} // end namespace detail

//===----------------------------------------------------------------------===//
// UniSparse Types
//===----------------------------------------------------------------------===//

/// This class defines the Toy struct type. It represents a collection of
/// element types. All derived types in MLIR must inherit from the CRTP class
/// 'Type::TypeBase'. It takes as template parameters the concrete type
/// (StructType), the base class to use (Type), and the storage class
/// (StructTypeStorage).
class StructType : public mlir::Type::TypeBase<StructType, mlir::Type,
                                               detail::StructTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// Create an instance of a `StructType` with the given element types. There
  /// *must* be atleast one element type.
  static StructType get(llvm::ArrayRef<int64_t> dimSizes,
                        llvm::ArrayRef<mlir::Type> elementTypes,
                        llvm::StringRef identifier,
                        llvm::ArrayRef<AffineMap> order);
  llvm::ArrayRef<int64_t> getDimSizes();

  /// Returns the element types of this struct type.
  llvm::ArrayRef<mlir::Type> getElementTypes();

  llvm::StringRef getIdentifier();

  llvm::ArrayRef<AffineMap> getOrder();

  /// Returns the number of element type held by this struct.
  size_t getNumElementTypes() { return getElementTypes().size(); }
};
} // end namespace unisparse
} // end namespace mlir

#define GET_TYPEDEF_CLASSES
#include "IR/UniSparseOpsTypes.h.inc"

#endif // UNISPARSE_UNISPARSETYPES_H
