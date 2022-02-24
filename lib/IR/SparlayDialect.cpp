//===- SparlayDialect.cpp - Sparlay dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IR/SparlayDialect.h"
#include "IR/SparlayOps.h"
#include "IR/SparlayTypes.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "struct-parsing"

using namespace mlir;
using namespace mlir::sparlay;

#include "IR/SparlayOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Sparlay Types
//===----------------------------------------------------------------------===//

namespace mlir {
namespace sparlay {
namespace detail {
/// This class represents the internal storage of the Sparlay `StructType`.
struct StructTypeStorage : public mlir::TypeStorage {
  /// The `KeyTy` is a required type that provides an interface for the storage
  /// instance. This type will be used when uniquing an instance of the type
  /// storage. For our struct type, we will unique each instance structurally on
  /// the elements that it contains.
  using KeyTy = std::tuple<llvm::ArrayRef<mlir::Type>, llvm::StringRef, llvm::ArrayRef<AffineMap>>;

  /// A constructor for the type storage instance.
  // StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
  //     : elementTypes(elementTypes) {}
  
  // StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes, 
  //                   Attribute identifier)
  //     : elementTypes(elementTypes), identifier(identifier) {}
  
  StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes, 
                    llvm::StringRef identifier,
                    llvm::ArrayRef<AffineMap> order)
      : elementTypes(elementTypes), identifier(identifier), order(order) {}

  /// Define the comparison function for the key type with the current storage
  /// instance. This is used when constructing a new instance to ensure that we
  /// haven't already uniqued an instance of the given key.
  bool operator==(const KeyTy &key) const {
    if (!(elementTypes == std::get<0>(key)))
      return false;
    if (!(identifier == std::get<1>(key)))
      return false;
    if (!(order == std::get<2>(key)))
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
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key), std::get<2>(key));
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
    auto elementTypes = std::get<0>(key);
    auto identifier = std::get<1>(key);
    auto order = std::get<2>(key);

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
        StructTypeStorage(elementTypes, identifier, order);
  }

  // bool hasOrder() const { return !order.isEmpty(); }

  // bool hasIdentifier() const { return !identifier.empty(); }

  // llvm::ArrayRef<mlir::Type> getElementTypes() const { return elementTypes; }

  // llvm::StringRef getIdentifier() const { return identifier; }

  // llvm::ArrayRef<AffineMap> getOrder() const { return order; }

  /// The following field contains the element types of the struct.
  llvm::ArrayRef<mlir::Type> elementTypes;
  llvm::StringRef identifier;
  llvm::ArrayRef<AffineMap> order;
};
} // end namespace detail
} // end namespace sparlay
} // end namespace mlir

/// Create an instance of a `StructType` with the given element types. There
/// *must* be at least one element type.
StructType StructType::get(llvm::ArrayRef<mlir::Type> elementTypes,
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
  return Base::get(ctx, elementTypes, identifier, order);
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
mlir::Type SparlayDialect::parseType(mlir::DialectAsmParser &parser) const {
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
  do {
    // Parse the current element type.
    llvm::SMLoc typeLoc = parser.getCurrentLocation();
    mlir::Type elementType;
    AffineMap elmOrder;

    auto parseType = parser.parseOptionalType(elementType);

    if (!parseType.hasValue()) {
      if (parser.parseOptionalString(&getIdentifier)) {
        if (parser.parseAffineMap(elmOrder)) {
          return nullptr;
        } else {
          LLVM_DEBUG(llvm::dbgs() << "is an affine_map type\n");
          elmOrders.push_back(elmOrder);
        }
      } else {
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
    
    // if (parser.parseAffineMap(elmOrder)) {
    //   if (parser.parseOptionalString(&getIdentifier)) {
    //     if (parser.parseOptionalType(elementType)) {
    //       return nullptr;
    //     } else {
    //       LLVM_DEBUG(llvm::dbgs() << "is a memref type\n");
    //       if (elementType.isa<mlir::TensorType, mlir::MemRefType, StructType>())
    //         elementTypes.push_back(elementType);
    //       else {
    //         parser.emitError(typeLoc, "element type for a struct must either "
    //                               "be a TensorType, a MemrefType or a StructType, StringRef, AffineMap, got: ")
    //             << elementType;
    //         return Type();
    //       }
    //     }
    //   } else {
    //   LLVM_DEBUG(llvm::dbgs() << "is a string type: " << getIdentifier << "\n");
    //   }
    // } else {
    //   LLVM_DEBUG(llvm::dbgs() << "is an affine_map type\n");
    //   elmOrders.push_back(elmOrder);
    // }

    // if (parser.parseType(elementType)) {
    //   if (parser.parseString(&elmIdentifier)) {
    //     LLVM_DEBUG(llvm::dbgs() << "is a string type\n");
    //     if (parser.parseAffineMap(elmOrder)) {
    //     LLVM_DEBUG(llvm::dbgs() << "is an affine_map type\n");
    //       return nullptr;
    //     }
    //   }
    //   // elmIdentifier = StringRef(elmIdentifier_copy);
    // } else {
    //   // Check that the type is a TensorType, MemRefType or another StructType.
    //   if (elementType.isa<mlir::TensorType, mlir::MemRefType, StructType>()) {
    //     LLVM_DEBUG(llvm::dbgs() << "is a memref type\n");
    //     elementTypes.push_back(elementType);
    //   } else {
    //     parser.emitError(typeLoc, "element type for a struct must either "
    //                               "be a TensorType, a MemrefType or a StructType, StringRef, AffineMap, got: ")
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
  return StructType::get(elementTypes, elmIdentifier, elmOrders);
}

/// Print an instance of a type registered to the toy dialect.
void SparlayDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  // Currently the only toy type is a struct type.
  StructType structType = type.cast<StructType>();

  // Print the struct type according to the parser format.
  printer << "struct<";
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
// Sparlay dialect.
//===----------------------------------------------------------------------===//

void SparlayDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "IR/SparlayOps.cpp.inc"
      >();
  addTypes<StructType>();
}
