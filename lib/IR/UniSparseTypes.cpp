//===- UniSparseDialect.cpp - UniSparse dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IR/UniSparseDialect.h"
#include "IR/UniSparseOps.h"
#include "IR/UniSparseTypes.h"
#include "IR/UniSparseAttr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/TypeSwitch.h"
#include <iostream>

#define DEBUG_TYPE "struct-parsing"

using namespace mlir;
using namespace mlir::unisparse;

//===----------------------------------------------------------------------===//
// UniSparse Types
//===----------------------------------------------------------------------===//

namespace mlir {
namespace unisparse {
namespace detail {
/// This class represents the internal storage of the UniSparse `StructType`.
struct StructTypeStorage : public mlir::TypeStorage {
  /// The `KeyTy` is a required type that provides an interface for the storage
  /// instance. This type will be used when uniquing an instance of the type
  /// storage. For our struct type, we will unique each instance structurally on
  /// the elements that it contains.
  using KeyTy = std::tuple<llvm::ArrayRef<int64_t>, llvm::ArrayRef<mlir::Type>, 
                            llvm::StringRef, llvm::ArrayRef<AffineMap>>;

  /// A constructor for the type storage instance.
  // StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
  //     : elementTypes(elementTypes) {}
  
  // StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes, 
  //                   Attribute identifier)
  //     : elementTypes(elementTypes), identifier(identifier) {}
  
  StructTypeStorage(llvm::ArrayRef<int64_t> dimSizes,
                    llvm::ArrayRef<mlir::Type> elementTypes, 
                    llvm::StringRef identifier,
                    llvm::ArrayRef<AffineMap> order)
      : dimSizes(dimSizes), elementTypes(elementTypes), 
        identifier(identifier), order(order) {}

  /// Define the comparison function for the key type with the current storage
  /// instance. This is used when constructing a new instance to ensure that we
  /// haven't already uniqued an instance of the given key.
  bool operator==(const KeyTy &key) const {
    if (!(dimSizes == std::get<0>(key)))
      return false;
    if (!(elementTypes == std::get<1>(key)))
      return false;
    if (!(identifier == std::get<2>(key)))
      return false;
    if (!(order == std::get<3>(key)))
      return false;
    return true;
    // if ( !hasIdentifier() && !hasOrder() ) {
    //   return getElementTypes() == std::get<0>(key);
    // } else if ( hasIdentifier() && !hasOrder() ) {
    //   return getElementTypes() == std::get<0>(key) && getIdentifier() == std::get<1>(key);
    // } else
    //   return key == KeyTy(getElementTypes(), getIdentifier(), getOrder()); 
  }

  /// Define a hash function for the key type. This is used when uniquing
  /// instances of the storage, see the `StructType::get` method.
  /// Note: This method isn't necessary as both llvm::ArrayRef and mlir::Type
  /// have hash functions available, so we could just omit this entirely.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key), 
                              std::get<2>(key), std::get<3>(key));
  }

  /// Define a construction function for the key type from a set of parameters.
  /// These parameters will be provided when constructing the storage instance
  /// itself.
  /// Note: This method isn't necessary because KeyTy can be directly
  /// constructed with the given parameters.
  // static KeyTy getKey(llvm::ArrayRef<mlir::Type> elementTypes, 
  //                     StringRef identifier,
  //                     AffineMap order) {
  //   return KeyTy(elementTypes, identifier, order);
  // }

  /// Define a construction method for creating a new instance of this storage.
  /// This method takes an instance of a storage allocator, and an instance of a
  /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
  /// allocations used to create the type storage and its internal.
  static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    // Copy the elements from the provided `KeyTy` into the allocator.
    auto dimSizes = std::get<0>(key);
    auto elementTypes = std::get<1>(key);
    auto identifier = std::get<2>(key);
    auto order = std::get<3>(key);

    dimSizes = allocator.copyInto(dimSizes);
    elementTypes = allocator.copyInto(elementTypes);
    identifier = allocator.copyInto(identifier);
    order = allocator.copyInto(order);

    // if ( !std::get<1>(key).empty() ) {
    //   identifier = allocator.copyInto(std::get<1>(key));
    //   if ( !std::get<2>(key).isEmpty() ) {
    //     order = std::get<2>(key);
    //     return new (allocator.allocate<StructTypeStorage>())
    //     StructTypeStorage(elementTypes, identifier, order);
    //   }
    //   return new (allocator.allocate<StructTypeStorage>())
    //     StructTypeStorage(elementTypes, identifier);
    // }

    // Allocate the storage instance and construct it.
    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(dimSizes, elementTypes, identifier, order);
  }

  // bool hasOrder() const { return !order.isEmpty(); }

  // bool hasIdentifier() const { return !identifier.empty(); }

  // llvm::ArrayRef<mlir::Type> getElementTypes() const { return elementTypes; }

  // llvm::StringRef getIdentifier() const { return identifier; }

  // llvm::ArrayRef<AffineMap> getOrder() const { return order; }

  /// The following field contains the element types of the struct.
  llvm::ArrayRef<int64_t> dimSizes;
  llvm::ArrayRef<mlir::Type> elementTypes;
  llvm::StringRef identifier;
  llvm::ArrayRef<AffineMap> order;
};
} // end namespace detail
} // end namespace unisparse
} // end namespace mlir

/// Create an instance of a `StructType` with the given element types. There
/// *must* be at least one element type.
StructType StructType::get(llvm::ArrayRef<int64_t> dimSizes,
                           llvm::ArrayRef<mlir::Type> elementTypes,
                           llvm::StringRef identifier,
                           llvm::ArrayRef<AffineMap> order) {
  assert(!elementTypes.empty() && "expected at least 1 element type");

  // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
  // of this type. The first parameter is the context to unique in. The
  // parameters after the context are forwarded to the storage instance.
  mlir::MLIRContext *ctx = elementTypes.front().getContext();
  // if (identifier.empty() && order.isEmpty())
  //   return Base::get(ctx, elementTypes);
  // else if (order.isEmpty())
  //   return Base::get(ctx, elementTypes, identifier);
  // else
  return Base::get(ctx, dimSizes, elementTypes, identifier, order);
}

llvm::ArrayRef<int64_t> StructType::getDimSizes() {
  return getImpl()->dimSizes;
}

/// Returns the element types of this struct type.
llvm::ArrayRef<mlir::Type> StructType::getElementTypes() {
  // 'getImpl' returns a pointer to the internal storage instance.
  return getImpl()->elementTypes;
}

llvm::StringRef StructType::getIdentifier() {
  return getImpl()->identifier;
}

llvm::ArrayRef<AffineMap> StructType::getOrder() {
  return getImpl()->order;
}

/// Parse an instance of a type registered to the toy dialect.
mlir::Type UniSparseDialect::parseType(mlir::DialectAsmParser &parser) const {
  // Parse a struct type in the following form:
  //   struct-type ::= `struct` `<` type (`,` type)* `>`

  // NOTE: All MLIR parser function return a ParseResult. This is a
  // specialization of LogicalResult that auto-converts to a `true` boolean
  // value on failure to allow for chaining, but may be used with explicit
  // `mlir::failed/mlir::succeeded` as desired.

  // Parse: `struct` `<`
  if (parser.parseKeyword("struct") || parser.parseLess())
    return Type();

  // Parse the element types of the struct.
  SmallVector<mlir::Type, 1> elementTypes;
  std::string getIdentifier;
  SmallVector<AffineMap, 1> elmOrders;
  bool isParsingType = true;

  SmallVector<int64_t, 1> dimSizes;
  LLVM_DEBUG(llvm::dbgs() << "before parsing [\n");
  // parse index list
  if (!(parser.parseOptionalLSquare())) {
    llvm::SMLoc typeLoc = parser.getCurrentLocation();
    do {
      int64_t dim;
      auto parseInt = parser.parseOptionalInteger(dim);
      if (parseInt.hasValue())
        dimSizes.push_back(dim);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseOptionalRSquare()) {
      parser.emitError(typeLoc, "Missing right square.\n");
      return nullptr;
    }
    if (failed(parser.parseOptionalComma())) 
    return nullptr;
  }

  do {
    // Parse the current element type.
    llvm::SMLoc typeLoc = parser.getCurrentLocation();
    mlir::Type elementType;
    AffineMap elmOrder;

    if (isParsingType) {
      auto parseType = parser.parseOptionalType(elementType);
      if (!parseType.hasValue()) {
        if (parser.parseOptionalString(&getIdentifier)) {
          return nullptr;
        } else {
          isParsingType = false;
          LLVM_DEBUG(llvm::dbgs() << "is a string type: " << getIdentifier << "\n");
        }
      } else {
        LLVM_DEBUG(llvm::dbgs() << "is a memref type\n");
        if (elementType.isa<mlir::TensorType, mlir::MemRefType, StructType>())
          elementTypes.push_back(elementType);
        else {
          parser.emitError(typeLoc, "element type for a struct must either "
                                "be a TensorType, a MemrefType or a StructType, StringRef, AffineMap, got: ")
              << elementType;
          return Type();
        }
      }
    } else {
      if (parser.parseAffineMap(elmOrder)) {
        return nullptr;
      } else {
        LLVM_DEBUG(llvm::dbgs() << "is an affine_map type\n");
        elmOrders.push_back(elmOrder);
      }
    }

    // if (!parseType.hasValue()) {
    //   if (parser.parseOptionalString(&getIdentifier)) {
    //     if (parser.parseAffineMap(&elmOrder)) {
    //       return nullptr;
    //     } else {
    //       LLVM_DEBUG(llvm::dbgs() << "is an affine_map type\n");
    //       elmOrders.push_back(elmOrder);
    //     }
    //   } else {
    //     LLVM_DEBUG(llvm::dbgs() << "is a string type: " << getIdentifier << "\n");
    //   }
    // } else {
    //   LLVM_DEBUG(llvm::dbgs() << "is a memref type\n");
    //   if (elementType.isa<mlir::TensorType, mlir::MemRefType, StructType>())
    //     elementTypes.push_back(elementType);
    //   else {
    //     parser.emitError(typeLoc, "element type for a struct must either "
    //                           "be a TensorType, a MemrefType or a StructType, StringRef, AffineMap, got: ")
    //         << elementType;
    //     return Type();
    //   }
    // }
    
    // Parse the optional: `,`
  } while (succeeded(parser.parseOptionalComma()));

  // Parse: `>`
  if (parser.parseGreater())
    return Type();
  
  llvm::StringRef elmIdentifier(getIdentifier);
  return StructType::get(dimSizes, elementTypes, elmIdentifier, elmOrders);
}

/// Print an instance of a type registered to the toy dialect.
void UniSparseDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  // Currently the only toy type is a struct type.
  StructType structType = type.cast<StructType>();

  // Print the struct type according to the parser format.
  printer << "struct<";
  if (!structType.getDimSizes().empty()) {
    printer << "[";
    llvm::interleaveComma(structType.getDimSizes(), printer);
    printer << "], ";
  }
  llvm::interleaveComma(structType.getElementTypes(), printer);
  if (!structType.getIdentifier().empty())
    printer << ", " << "\"" << structType.getIdentifier() << "\"";
  // printer << ", " << structType.getOrder();
  if (!structType.getOrder().empty()) {
    printer << ", ";
    llvm::interleaveComma(structType.getOrder(), printer);
  }
  printer << '>';
}

//===----------------------------------------------------------------------===//
// UniSparse Dialect
//===----------------------------------------------------------------------===//

void UniSparseDialect::registerTypes() {
  addTypes<StructType>();
}
