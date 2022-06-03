/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::sparlay::SparlayEncodingAttr,
::mlir::sparlay::SparlayAffineAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


static ::mlir::OptionalParseResult generatedAttributeParser(
                                      ::mlir::DialectAsmParser &parser,
                                      ::llvm::StringRef mnemonic, ::mlir::Type type,
                                      ::mlir::Attribute &value) {
  if (mnemonic == ::mlir::sparlay::SparlayEncodingAttr::getMnemonic()) { 
    value = ::mlir::sparlay::SparlayEncodingAttr::parse(parser, type);
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::sparlay::SparlayAffineAttr::getMnemonic()) { 
    value = ::mlir::sparlay::SparlayAffineAttr::parse(parser, type);
    return ::mlir::success(!!value);
  }
  return {};
}


static ::mlir::LogicalResult generatedAttributePrinter(
                         ::mlir::Attribute def, ::mlir::DialectAsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::mlir::LogicalResult>(def)
    .Case<::mlir::sparlay::SparlayEncodingAttr>([&](::mlir::sparlay::SparlayEncodingAttr t) {
      t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::sparlay::SparlayAffineAttr>([&](::mlir::sparlay::SparlayAffineAttr t) {
      t.print(printer);
      return ::mlir::success();
    })
    .Default([](::mlir::Attribute) { return ::mlir::failure(); });
}

namespace mlir {
namespace sparlay {

namespace detail {
  struct SparlayEncodingAttrStorage : public ::mlir::AttributeStorage {
    SparlayEncodingAttrStorage (AffineMap primaryMap, SparlayAffineMap secondaryMap, unsigned bitWidth)
      : ::mlir::AttributeStorage(), primaryMap(primaryMap), secondaryMap(secondaryMap), bitWidth(bitWidth) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<AffineMap, SparlayAffineMap, unsigned>;
  bool operator==(const KeyTy &tblgenKey) const {
    if (!(primaryMap == std::get<0>(tblgenKey)))
      return false;
    if (!(secondaryMap == std::get<1>(tblgenKey)))
      return false;
    if (!(bitWidth == std::get<2>(tblgenKey)))
      return false;
    return true;
  }
  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
      return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static SparlayEncodingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator,
                          const KeyTy &tblgenKey) {
      auto primaryMap = std::get<0>(tblgenKey);
      auto secondaryMap = std::get<1>(tblgenKey);
      auto bitWidth = std::get<2>(tblgenKey);

      return new (allocator.allocate<SparlayEncodingAttrStorage>())
          SparlayEncodingAttrStorage(primaryMap, secondaryMap, bitWidth);
    }
      AffineMap primaryMap;
      SparlayAffineMap secondaryMap;
      unsigned bitWidth;
  };
} // namespace detail
SparlayEncodingAttr SparlayEncodingAttr::get(::mlir::MLIRContext *context, AffineMap primaryMap, SparlayAffineMap secondaryMap, unsigned bitWidth) {
  return Base::get(context, primaryMap, secondaryMap, bitWidth);
}
SparlayEncodingAttr SparlayEncodingAttr::getChecked(llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, AffineMap primaryMap, SparlayAffineMap secondaryMap, unsigned bitWidth) {
  return Base::getChecked(emitError, context, primaryMap, secondaryMap, bitWidth);
}
AffineMap SparlayEncodingAttr::getPrimaryMap() const { return getImpl()->primaryMap; }
SparlayAffineMap SparlayEncodingAttr::getSecondaryMap() const { return getImpl()->secondaryMap; }
unsigned SparlayEncodingAttr::getBitWidth() const { return getImpl()->bitWidth; }
} // namespace sparlay
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::sparlay::SparlayEncodingAttr)
namespace mlir {
namespace sparlay {

namespace detail {
  struct SparlayAffineAttrStorage : public ::mlir::AttributeStorage {
    SparlayAffineAttrStorage (SparlayAffineMap value)
      : ::mlir::AttributeStorage(), value(value) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<SparlayAffineMap>;
  bool operator==(const KeyTy &tblgenKey) const {
    if (!(value == std::get<0>(tblgenKey)))
      return false;
    return true;
  }
  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
      return ::llvm::hash_combine(std::get<0>(tblgenKey));
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static SparlayAffineAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator,
                          const KeyTy &tblgenKey) {
      auto value = std::get<0>(tblgenKey);

      return new (allocator.allocate<SparlayAffineAttrStorage>())
          SparlayAffineAttrStorage(value);
    }
      SparlayAffineMap value;
  };
} // namespace detail
SparlayAffineAttr SparlayAffineAttr::get(::mlir::MLIRContext *context, SparlayAffineMap value) {
  return Base::get(context, value);
}
SparlayAffineMap SparlayAffineAttr::getValue() const { return getImpl()->value; }
} // namespace sparlay
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::sparlay::SparlayAffineAttr)

#endif  // GET_ATTRDEF_CLASSES

