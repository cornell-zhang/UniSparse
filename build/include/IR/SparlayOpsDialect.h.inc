/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace sparlay {

class SparlayDialect : public ::mlir::Dialect {
  explicit SparlayDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context,
      ::mlir::TypeID::get<SparlayDialect>()) {
    
    initialize();
  }

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~SparlayDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("sparlay");
  }

  /// Parse an attribute registered to this dialect.
  ::mlir::Attribute parseAttribute(::mlir::DialectAsmParser &parser,
                                   ::mlir::Type type) const override;

  /// Print an attribute registered to this dialect.
  void printAttribute(::mlir::Attribute attr,
                      ::mlir::DialectAsmPrinter &os) const override;

  /// Parse a type registered to this dialect.
  ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect.
  void printType(::mlir::Type type,
                 ::mlir::DialectAsmPrinter &os) const override;
};
} // namespace sparlay
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::sparlay::SparlayDialect)
