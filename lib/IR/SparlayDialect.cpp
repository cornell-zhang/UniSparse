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
#include "IR/SparlayAttr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser.h"

#include "mlir/Transforms/InliningUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"
#include <iostream>

#define DEBUG_TYPE "struct-parsing"

using namespace mlir;
using namespace mlir::sparlay;

#include "IR/SparlayOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "IR/SparlayAttr.cpp.inc"

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
} // end namespace sparlay
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
void SparlayDialect::printType(mlir::Type type,
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

ParseResult parseAffineExprGetDim(
  DialectAsmParser& parser, AffineExpr& expr, const SmallVector<std::string>& mapDimtoID
) {
  StringRef key;
  auto ret = parser.parseKeyword(&key);
  if (succeeded(ret)) {
    bool mark = 0;
    for (size_t dim = 0; dim < mapDimtoID.size(); ++dim) {
      if (mapDimtoID[dim] == key.str()) {
        mark = 1;
        expr = getAffineDimExpr(dim, parser.getContext());
      }
    }
    if (!mark) ret = ParseResult(failure());
  }
  return ret;
}

Attribute SparlayAffineAttr::parse(DialectAsmParser &parser, Type type) {
  SmallVector<int> fuseIndex;
  SmallVector<int> trimIndex;
  SmallVector<AffineExpr> secondaryExprs;
  static std::string tok_trim = "trim", tok_fuse = "fuse";
  SmallVector<StringRef, 2> op_token{tok_trim, tok_fuse};
  SmallVector<std::string> mapDimtoID;
  int tot_dim = 0;
  auto parseElt = [&]() -> ParseResult {
    AffineExpr expr;
    StringRef key;
    auto ret = parser.parseKeyword(&key);
    if (succeeded(ret)) {
      expr = getAffineDimExpr(tot_dim++, parser.getContext());
      mapDimtoID.push_back(key.str());
    }
    return ret;
  };
  int image_idx = 0;
  auto parseElt2 = [&]() -> ParseResult {
    StringRef op_key_ref;
    bool fused = 0;
    bool trimmed = 0;
    while (1) {
      auto ret = parser.parseOptionalKeyword(&op_key_ref, ArrayRef<StringRef>(op_token));
      if (succeeded(ret)) {
        if (std::string(op_key_ref.str()) == "fuse") {
          if (!fused) fuseIndex.push_back(image_idx), fused = 1;
        } else {
          if (!trimmed) trimIndex.push_back(image_idx), trimmed = 1;
        }
      } else {
        break;
      }
    }
    AffineExpr expr;
    auto ret = parseAffineExprGetDim(parser, expr, mapDimtoID);
    if (succeeded(ret)) {
      image_idx++;
      secondaryExprs.push_back(expr);
    }
    return ret;
  };
  if (failed(parser.parseLess())) {
    return {};
  }
  if (failed(parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren, parseElt))) {
    return {};
  }
  if (failed(parser.parseArrow())) {
    return {};
  }
  if (failed(parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren, parseElt2))) {
    return {};
  }
  if (failed(parser.parseGreater())) {
    return {};
  }
  return SparlayAffineAttr::get(
    parser.getContext(), SparlayAffineMap(AffineMap::get(tot_dim, 0, secondaryExprs, parser.getContext()), trimIndex, fuseIndex)
  );
}

Attribute SparlayEncodingAttr::parse(DialectAsmParser &parser, Type type) {
  if (failed(parser.parseLess()))
    return {};
  // Parse the data as a dictionary.
  DictionaryAttr dict;
  if (failed(parser.parseAttribute(dict)))
    return {};
  AffineMap primaryMap{};
  SparlayAffineMap secondaryMap{};
  unsigned bitWidth = 8;
  for (const auto& attr: dict) {
    if (attr.first == "primaryMap") {
      auto affineAttr = attr.second.dyn_cast<AffineMapAttr>();
      if (!affineAttr) {
        return {};
      }
      primaryMap = affineAttr.getValue();
    } else if (attr.first == "secondaryMap") {
      auto sparlayAffineAttr = attr.second.dyn_cast<SparlayAffineAttr>();
      if (!sparlayAffineAttr) {
        return {};
      }
      secondaryMap = sparlayAffineAttr.getValue();
    } else if (attr.first == "bitWidth") {
      auto intAttr = attr.second.dyn_cast<IntegerAttr>();
      if (!intAttr) return {};
      bitWidth = intAttr.getInt();
    } else {
      return {};
    }
  }
  if (failed(parser.parseGreater()))
    return {};
  return SparlayEncodingAttr::get(parser.getContext(), primaryMap, secondaryMap, bitWidth);
}

void SparlayEncodingAttr::print(DialectAsmPrinter &printer) const {
  const SparlayAffineMap& secondaryMap = getSecondaryMap();
  const AffineMap& primaryMap = getPrimaryMap();
  printer << "primaryMap: " << primaryMap << ", ";
  printer << "secondaryMap: " << AffineMap::get(secondaryMap.getNumDims(), secondaryMap.getNumSymbols(), secondaryMap.getResults(), secondaryMap.getContext()) << ", ";
  printer << "trim_level(";
  const auto& trimIndex = secondaryMap.getTrimIndex();
  for (size_t i = 0; i < trimIndex.size(); ++i) {
    printer << trimIndex[i];
    if (i != trimIndex.size()-1) printer << ',';
  }
  printer << "), ";
  printer << "fuse_level(";
  const auto& fuseIndex = secondaryMap.getFuseIndex();
  for (size_t i = 0; i < fuseIndex.size(); ++i) {
    printer << fuseIndex[i];
    if (i != fuseIndex.size()-1) printer << ',';
  }
  printer << "), ";
  printer << "bitWidth: " << getBitWidth();
}

void SparlayAffineAttr::print(DialectAsmPrinter &printer) const {
  return;
}

LogicalResult SparlayEncodingAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    AffineMap primaryMap, SparlayAffineMap secondaryMap, unsigned bitWidth
) {
  return success();
}

LogicalResult SparlayEncodingAttr::verifyEncoding(
  ArrayRef<int64_t> shape, Type elementType, function_ref<InFlightDiagnostic()> emitError
) const {
  return success();
}

Attribute SparlayDialect::parseAttribute(DialectAsmParser &parser,
                                              Type type) const {
  StringRef attrTag;
  if (failed(parser.parseKeyword(&attrTag)))
    return Attribute();
  Attribute attr;
  auto parseResult = generatedAttributeParser(parser, attrTag, type, attr);
  if (parseResult.hasValue())
    return attr;
  parser.emitError(parser.getNameLoc(), "unknown sparse tensor layout attribute");
  return Attribute();
}

void SparlayDialect::printAttribute(Attribute attr,
                                         DialectAsmPrinter &printer) const {
  if (succeeded(generatedAttributePrinter(attr, printer)))
    return;
}

//===----------------------------------------------------------------------===//
// Sparlay dialect.
//===----------------------------------------------------------------------===//

void SparlayDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "IR/SparlayAttr.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "IR/SparlayOps.cpp.inc"
      >();
  addTypes<StructType>();
}