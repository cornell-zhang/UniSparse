/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


namespace mlir {
class DialectAsmParser;
class DialectAsmPrinter;
} // namespace mlir
namespace mlir {
namespace sparlay {
  class SparlayEncodingAttr;
  class SparlayAffineAttr;

  namespace detail {
    struct SparlayEncodingAttrStorage;
  } // end namespace detail
  class SparlayEncodingAttr : public ::mlir::Attribute::AttrBase<SparlayEncodingAttr, ::mlir::Attribute,
                                         detail::SparlayEncodingAttrStorage, ::mlir::VerifiableTensorEncoding::Trait> {
  public:
    /// Inherit some necessary constructors from 'AttrBase'.
    using Base::Base;

    static SparlayEncodingAttr get(::mlir::MLIRContext *context, AffineMap primaryMap, SparlayAffineMap secondaryMap, unsigned bitWidth);
    static SparlayEncodingAttr getChecked(llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, AffineMap primaryMap, SparlayAffineMap secondaryMap, unsigned bitWidth);

    using Base::getChecked;
    static ::mlir::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, AffineMap primaryMap, SparlayAffineMap secondaryMap, unsigned bitWidth);
    static constexpr ::llvm::StringLiteral getMnemonic() {
      return ::llvm::StringLiteral("encoding");
    }

    static ::mlir::Attribute parse(::mlir::DialectAsmParser &parser, ::mlir::Type type);
    void print(::mlir::DialectAsmPrinter &printer) const;
    AffineMap getPrimaryMap() const;
    SparlayAffineMap getSecondaryMap() const;
    unsigned getBitWidth() const;
    ::mlir::LogicalResult verifyEncoding(ArrayRef<int64_t> shape, Type elementType, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) const;
  };

  namespace detail {
    struct SparlayAffineAttrStorage;
  } // end namespace detail
  class SparlayAffineAttr : public ::mlir::Attribute::AttrBase<SparlayAffineAttr, ::mlir::Attribute,
                                         detail::SparlayAffineAttrStorage> {
  public:
    /// Inherit some necessary constructors from 'AttrBase'.
    using Base::Base;

    static SparlayAffineAttr get(::mlir::MLIRContext *context, SparlayAffineMap value);
    static constexpr ::llvm::StringLiteral getMnemonic() {
      return ::llvm::StringLiteral("affine");
    }

    static ::mlir::Attribute parse(::mlir::DialectAsmParser &parser, ::mlir::Type type);
    void print(::mlir::DialectAsmPrinter &printer) const;
    SparlayAffineMap getValue() const;
  };
} // namespace sparlay
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::sparlay::SparlayEncodingAttr)
DECLARE_EXPLICIT_TYPE_ID(::mlir::sparlay::SparlayAffineAttr)

#endif  // GET_ATTRDEF_CLASSES

