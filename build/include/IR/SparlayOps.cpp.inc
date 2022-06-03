/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::sparlay::CompressOp,
::mlir::sparlay::FooOp,
::mlir::sparlay::MultiplyOp,
::mlir::sparlay::NewOp,
::mlir::sparlay::PackOp,
::mlir::sparlay::StructAccessOp,
::mlir::sparlay::StructConstructOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace sparlay {
static ::mlir::LogicalResult __mlir_ods_local_type_constraint_SparlayOps0(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::TensorType>())) && ((true))) || (((type.isa<::mlir::MemRefType>())) && ((true))) || ((type.isa<StructType>())))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be tensor of any type values or memref of any type values or Sparlay struct type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_SparlayOps1(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isSignlessInteger(32)))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be 32-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_SparlayOps3(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((true))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_SparlayOps4(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<StructType>()))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be Sparlay struct type, but got " << type;
  }
  return ::mlir::success();
}

} // namespace sparlay
} // namespace mlir
namespace mlir {
namespace sparlay {

//===----------------------------------------------------------------------===//
// ::mlir::sparlay::CompressOp definitions
//===----------------------------------------------------------------------===//

CompressOpAdaptor::CompressOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CompressOpAdaptor::CompressOpAdaptor(CompressOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CompressOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CompressOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CompressOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CompressOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr CompressOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr CompressOpAdaptor::compress_dim() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("compress_dim").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::AffineMapAttr CompressOpAdaptor::storage_order() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::AffineMapAttr attr = odsAttrs.get("storage_order").cast<::mlir::AffineMapAttr>();
  return attr;
}

::mlir::LogicalResult CompressOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_compress_dim = odsAttrs.get("compress_dim");
  if (!tblgen_compress_dim) return emitError(loc, "'sparlay.compress' op ""requires attribute 'compress_dim'");
    if (!(((tblgen_compress_dim.isa<::mlir::IntegerAttr>())) && ((tblgen_compress_dim.cast<::mlir::IntegerAttr>().getType().isa<::mlir::IndexType>())))) return emitError(loc, "'sparlay.compress' op ""attribute 'compress_dim' failed to satisfy constraint: index attribute");
  }
  {
  auto tblgen_storage_order = odsAttrs.get("storage_order");
  if (!tblgen_storage_order) return emitError(loc, "'sparlay.compress' op ""requires attribute 'storage_order'");
    if (!((tblgen_storage_order.isa<::mlir::AffineMapAttr>()))) return emitError(loc, "'sparlay.compress' op ""attribute 'storage_order' failed to satisfy constraint: AffineMap attribute");
  }
  return ::mlir::success();
}

















std::pair<unsigned, unsigned> CompressOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CompressOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CompressOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CompressOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CompressOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CompressOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CompressOp::output() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr CompressOp::compress_dimAttr() {
  return (*this)->getAttr(compress_dimAttrName()).template cast<::mlir::IntegerAttr>();
}

::llvm::APInt CompressOp::compress_dim() {
  auto attr = compress_dimAttr();
  return attr.getValue();
}

::mlir::AffineMapAttr CompressOp::storage_orderAttr() {
  return (*this)->getAttr(storage_orderAttrName()).template cast<::mlir::AffineMapAttr>();
}

::mlir::AffineMap CompressOp::storage_order() {
  auto attr = storage_orderAttr();
  return attr.getValue();
}

void CompressOp::compress_dimAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(compress_dimAttrName(), attr);
}

void CompressOp::storage_orderAttr(::mlir::AffineMapAttr attr) {
  (*this)->setAttr(storage_orderAttrName(), attr);
}

void CompressOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr compress_dim, ::mlir::AffineMapAttr storage_order) {
  odsState.addOperands(input);
  odsState.addAttribute(compress_dimAttrName(odsState.name), compress_dim);
  odsState.addAttribute(storage_orderAttrName(odsState.name), storage_order);
  odsState.addTypes(output);
}

void CompressOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr compress_dim, ::mlir::AffineMapAttr storage_order) {
  odsState.addOperands(input);
  odsState.addAttribute(compress_dimAttrName(odsState.name), compress_dim);
  odsState.addAttribute(storage_orderAttrName(odsState.name), storage_order);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CompressOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::llvm::APInt compress_dim, ::mlir::AffineMap storage_order) {
  odsState.addOperands(input);
  odsState.addAttribute(compress_dimAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIndexType(), compress_dim));
  odsState.addAttribute(storage_orderAttrName(odsState.name), ::mlir::AffineMapAttr::get(storage_order));
  odsState.addTypes(output);
}

void CompressOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::llvm::APInt compress_dim, ::mlir::AffineMap storage_order) {
  odsState.addOperands(input);
  odsState.addAttribute(compress_dimAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIndexType(), compress_dim));
  odsState.addAttribute(storage_orderAttrName(odsState.name), ::mlir::AffineMapAttr::get(storage_order));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CompressOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CompressOp::verify() {
  if (::mlir::failed(CompressOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparlayOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparlayOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult CompressOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType inputRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> inputOperands(inputRawOperands);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::Type inputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> inputTypes(inputRawTypes);
  ::mlir::Type outputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> outputTypes(outputRawTypes);
  if (parser.parseLParen())
    return ::mlir::failure();

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(inputRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseType(outputRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(outputTypes);
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CompressOp::print(::mlir::OpAsmPrinter &p) {
  p << "(";
  p << input();
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(input().getType());
  p << ' ' << "to";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(output().getType());
}

void CompressOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace sparlay
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::sparlay::CompressOp)

namespace mlir {
namespace sparlay {

//===----------------------------------------------------------------------===//
// ::mlir::sparlay::FooOp definitions
//===----------------------------------------------------------------------===//

FooOpAdaptor::FooOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

FooOpAdaptor::FooOpAdaptor(FooOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange FooOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FooOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FooOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FooOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr FooOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FooOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> FooOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FooOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FooOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange FooOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FooOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FooOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FooOp::res() {
  return *getODSResults(0).begin();
}

void FooOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value input) {
  odsState.addOperands(input);
  odsState.addTypes(res);
}

void FooOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FooOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FooOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input) {
  odsState.addOperands(input);
  odsState.addTypes({input.getType()});

}

void FooOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult FooOp::verify() {
  if (::mlir::failed(FooOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparlayOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparlayOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult FooOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType inputRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> inputOperands(inputRawOperands);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::Type inputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> inputTypes(inputRawTypes);

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(inputRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(inputTypes);
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FooOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << input();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(input().getType());
}

void FooOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace sparlay
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::sparlay::FooOp)

namespace mlir {
namespace sparlay {

//===----------------------------------------------------------------------===//
// ::mlir::sparlay::MultiplyOp definitions
//===----------------------------------------------------------------------===//

MultiplyOpAdaptor::MultiplyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MultiplyOpAdaptor::MultiplyOpAdaptor(MultiplyOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MultiplyOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MultiplyOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange MultiplyOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange MultiplyOpAdaptor::input() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr MultiplyOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr MultiplyOpAdaptor::target() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("target").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::StringAttr MultiplyOpAdaptor::pattern() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("pattern").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::LogicalResult MultiplyOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_target = odsAttrs.get("target");
  if (!tblgen_target) return emitError(loc, "'sparlay.multiply' op ""requires attribute 'target'");
    if (!((tblgen_target.isa<::mlir::StringAttr>()))) return emitError(loc, "'sparlay.multiply' op ""attribute 'target' failed to satisfy constraint: string attribute");
  }
  {
  auto tblgen_pattern = odsAttrs.get("pattern");
  if (!tblgen_pattern) return emitError(loc, "'sparlay.multiply' op ""requires attribute 'pattern'");
    if (!((tblgen_pattern.isa<::mlir::StringAttr>()))) return emitError(loc, "'sparlay.multiply' op ""attribute 'pattern' failed to satisfy constraint: string attribute");
  }
  return ::mlir::success();
}

















std::pair<unsigned, unsigned> MultiplyOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range MultiplyOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MultiplyOp::input() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange MultiplyOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MultiplyOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MultiplyOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MultiplyOp::output() {
  return *getODSResults(0).begin();
}

::mlir::StringAttr MultiplyOp::targetAttr() {
  return (*this)->getAttr(targetAttrName()).template cast<::mlir::StringAttr>();
}

::llvm::StringRef MultiplyOp::target() {
  auto attr = targetAttr();
  return attr.getValue();
}

::mlir::StringAttr MultiplyOp::patternAttr() {
  return (*this)->getAttr(patternAttrName()).template cast<::mlir::StringAttr>();
}

::llvm::StringRef MultiplyOp::pattern() {
  auto attr = patternAttr();
  return attr.getValue();
}

void MultiplyOp::targetAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(targetAttrName(), attr);
}

void MultiplyOp::patternAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(patternAttrName(), attr);
}

void MultiplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::ValueRange input, ::mlir::StringAttr target, ::mlir::StringAttr pattern) {
  odsState.addOperands(input);
  odsState.addAttribute(targetAttrName(odsState.name), target);
  odsState.addAttribute(patternAttrName(odsState.name), pattern);
  odsState.addTypes(output);
}

void MultiplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange input, ::mlir::StringAttr target, ::mlir::StringAttr pattern) {
  odsState.addOperands(input);
  odsState.addAttribute(targetAttrName(odsState.name), target);
  odsState.addAttribute(patternAttrName(odsState.name), pattern);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MultiplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::ValueRange input, ::llvm::StringRef target, ::llvm::StringRef pattern) {
  odsState.addOperands(input);
  odsState.addAttribute(targetAttrName(odsState.name), odsBuilder.getStringAttr(target));
  odsState.addAttribute(patternAttrName(odsState.name), odsBuilder.getStringAttr(pattern));
  odsState.addTypes(output);
}

void MultiplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange input, ::llvm::StringRef target, ::llvm::StringRef pattern) {
  odsState.addOperands(input);
  odsState.addAttribute(targetAttrName(odsState.name), odsBuilder.getStringAttr(target));
  odsState.addAttribute(patternAttrName(odsState.name), odsBuilder.getStringAttr(pattern));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MultiplyOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MultiplyOp::verify() {
  if (::mlir::failed(MultiplyOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparlayOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparlayOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult MultiplyOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> inputOperands;
  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> inputTypes;
  ::mlir::Type outputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> outputTypes(outputRawTypes);
  if (parser.parseLParen())
    return ::mlir::failure();

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(inputOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(inputTypes))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseType(outputRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(outputTypes);
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MultiplyOp::print(::mlir::OpAsmPrinter &p) {
  p << "(";
  p << input();
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << input().getTypes();
  p << ' ' << "to";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(output().getType());
}

void MultiplyOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace sparlay
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::sparlay::MultiplyOp)

namespace mlir {
namespace sparlay {

//===----------------------------------------------------------------------===//
// ::mlir::sparlay::NewOp definitions
//===----------------------------------------------------------------------===//

NewOpAdaptor::NewOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

NewOpAdaptor::NewOpAdaptor(NewOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange NewOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> NewOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange NewOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value NewOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr NewOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult NewOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> NewOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range NewOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value NewOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange NewOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> NewOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range NewOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value NewOp::output() {
  return *getODSResults(0).begin();
}

void NewOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input) {
  odsState.addOperands(input);
  odsState.addTypes(output);
}

void NewOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void NewOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult NewOp::verify() {
  if (::mlir::failed(NewOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparlayOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparlayOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult NewOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType inputRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> inputOperands(inputRawOperands);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::Type inputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> inputTypes(inputRawTypes);
  ::mlir::Type outputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> outputTypes(outputRawTypes);
  if (parser.parseLParen())
    return ::mlir::failure();

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(inputRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseType(outputRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(outputTypes);
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void NewOp::print(::mlir::OpAsmPrinter &p) {
  p << "(";
  p << input();
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(input().getType());
  p << ' ' << "to";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(output().getType());
}

void NewOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace sparlay
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::sparlay::NewOp)

namespace mlir {
namespace sparlay {

//===----------------------------------------------------------------------===//
// ::mlir::sparlay::PackOp definitions
//===----------------------------------------------------------------------===//

PackOpAdaptor::PackOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

PackOpAdaptor::PackOpAdaptor(PackOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange PackOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PackOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PackOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PackOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr PackOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr PackOpAdaptor::reduce_dim() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("reduce_dim").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::StringAttr PackOpAdaptor::padding() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("padding").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::AffineMapAttr PackOpAdaptor::storage_order() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::AffineMapAttr attr = odsAttrs.get("storage_order").cast<::mlir::AffineMapAttr>();
  return attr;
}

::mlir::LogicalResult PackOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_reduce_dim = odsAttrs.get("reduce_dim");
  if (!tblgen_reduce_dim) return emitError(loc, "'sparlay.pack' op ""requires attribute 'reduce_dim'");
    if (!(((tblgen_reduce_dim.isa<::mlir::IntegerAttr>())) && ((tblgen_reduce_dim.cast<::mlir::IntegerAttr>().getType().isa<::mlir::IndexType>())))) return emitError(loc, "'sparlay.pack' op ""attribute 'reduce_dim' failed to satisfy constraint: index attribute");
  }
  {
  auto tblgen_padding = odsAttrs.get("padding");
  if (!tblgen_padding) return emitError(loc, "'sparlay.pack' op ""requires attribute 'padding'");
    if (!((tblgen_padding.isa<::mlir::StringAttr>()))) return emitError(loc, "'sparlay.pack' op ""attribute 'padding' failed to satisfy constraint: string attribute");
  }
  {
  auto tblgen_storage_order = odsAttrs.get("storage_order");
  if (!tblgen_storage_order) return emitError(loc, "'sparlay.pack' op ""requires attribute 'storage_order'");
    if (!((tblgen_storage_order.isa<::mlir::AffineMapAttr>()))) return emitError(loc, "'sparlay.pack' op ""attribute 'storage_order' failed to satisfy constraint: AffineMap attribute");
  }
  return ::mlir::success();
}





















std::pair<unsigned, unsigned> PackOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PackOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PackOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange PackOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> PackOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PackOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PackOp::output() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr PackOp::reduce_dimAttr() {
  return (*this)->getAttr(reduce_dimAttrName()).template cast<::mlir::IntegerAttr>();
}

::llvm::APInt PackOp::reduce_dim() {
  auto attr = reduce_dimAttr();
  return attr.getValue();
}

::mlir::StringAttr PackOp::paddingAttr() {
  return (*this)->getAttr(paddingAttrName()).template cast<::mlir::StringAttr>();
}

::llvm::StringRef PackOp::padding() {
  auto attr = paddingAttr();
  return attr.getValue();
}

::mlir::AffineMapAttr PackOp::storage_orderAttr() {
  return (*this)->getAttr(storage_orderAttrName()).template cast<::mlir::AffineMapAttr>();
}

::mlir::AffineMap PackOp::storage_order() {
  auto attr = storage_orderAttr();
  return attr.getValue();
}

void PackOp::reduce_dimAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(reduce_dimAttrName(), attr);
}

void PackOp::paddingAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(paddingAttrName(), attr);
}

void PackOp::storage_orderAttr(::mlir::AffineMapAttr attr) {
  (*this)->setAttr(storage_orderAttrName(), attr);
}

void PackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr reduce_dim, ::mlir::StringAttr padding, ::mlir::AffineMapAttr storage_order) {
  odsState.addOperands(input);
  odsState.addAttribute(reduce_dimAttrName(odsState.name), reduce_dim);
  odsState.addAttribute(paddingAttrName(odsState.name), padding);
  odsState.addAttribute(storage_orderAttrName(odsState.name), storage_order);
  odsState.addTypes(output);
}

void PackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr reduce_dim, ::mlir::StringAttr padding, ::mlir::AffineMapAttr storage_order) {
  odsState.addOperands(input);
  odsState.addAttribute(reduce_dimAttrName(odsState.name), reduce_dim);
  odsState.addAttribute(paddingAttrName(odsState.name), padding);
  odsState.addAttribute(storage_orderAttrName(odsState.name), storage_order);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::llvm::APInt reduce_dim, ::llvm::StringRef padding, ::mlir::AffineMap storage_order) {
  odsState.addOperands(input);
  odsState.addAttribute(reduce_dimAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIndexType(), reduce_dim));
  odsState.addAttribute(paddingAttrName(odsState.name), odsBuilder.getStringAttr(padding));
  odsState.addAttribute(storage_orderAttrName(odsState.name), ::mlir::AffineMapAttr::get(storage_order));
  odsState.addTypes(output);
}

void PackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::llvm::APInt reduce_dim, ::llvm::StringRef padding, ::mlir::AffineMap storage_order) {
  odsState.addOperands(input);
  odsState.addAttribute(reduce_dimAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIndexType(), reduce_dim));
  odsState.addAttribute(paddingAttrName(odsState.name), odsBuilder.getStringAttr(padding));
  odsState.addAttribute(storage_orderAttrName(odsState.name), ::mlir::AffineMapAttr::get(storage_order));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PackOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PackOp::verify() {
  if (::mlir::failed(PackOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparlayOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparlayOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult PackOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType inputRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> inputOperands(inputRawOperands);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::Type inputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> inputTypes(inputRawTypes);
  ::mlir::Type outputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> outputTypes(outputRawTypes);
  if (parser.parseLParen())
    return ::mlir::failure();

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(inputRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseType(outputRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(outputTypes);
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PackOp::print(::mlir::OpAsmPrinter &p) {
  p << "(";
  p << input();
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(input().getType());
  p << ' ' << "to";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(output().getType());
}

void PackOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace sparlay
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::sparlay::PackOp)

namespace mlir {
namespace sparlay {

//===----------------------------------------------------------------------===//
// ::mlir::sparlay::StructAccessOp definitions
//===----------------------------------------------------------------------===//

StructAccessOpAdaptor::StructAccessOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

StructAccessOpAdaptor::StructAccessOpAdaptor(StructAccessOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange StructAccessOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> StructAccessOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange StructAccessOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StructAccessOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr StructAccessOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr StructAccessOpAdaptor::index() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("index").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult StructAccessOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_index = odsAttrs.get("index");
  if (!tblgen_index) return emitError(loc, "'sparlay.struct_access' op ""requires attribute 'index'");
    if (!(((tblgen_index.isa<::mlir::IntegerAttr>())) && ((tblgen_index.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'sparlay.struct_access' op ""attribute 'index' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  return ::mlir::success();
}













std::pair<unsigned, unsigned> StructAccessOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range StructAccessOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StructAccessOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange StructAccessOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StructAccessOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range StructAccessOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StructAccessOp::output() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr StructAccessOp::indexAttr() {
  return (*this)->getAttr(indexAttrName()).template cast<::mlir::IntegerAttr>();
}

uint64_t StructAccessOp::index() {
  auto attr = indexAttr();
  return attr.getValue().getZExtValue();
}

void StructAccessOp::indexAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(indexAttrName(), attr);
}



void StructAccessOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr index) {
  odsState.addOperands(input);
  odsState.addAttribute(indexAttrName(odsState.name), index);
  odsState.addTypes(output);
}

void StructAccessOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr index) {
  odsState.addOperands(input);
  odsState.addAttribute(indexAttrName(odsState.name), index);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StructAccessOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t index) {
  odsState.addOperands(input);
  odsState.addAttribute(indexAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), index));
  odsState.addTypes(output);
}

void StructAccessOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint64_t index) {
  odsState.addOperands(input);
  odsState.addAttribute(indexAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), index));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StructAccessOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult StructAccessOp::verify() {
  if (::mlir::failed(StructAccessOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparlayOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparlayOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}



::mlir::ParseResult StructAccessOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType inputRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> inputOperands(inputRawOperands);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::IntegerAttr indexAttr;
  ::mlir::Type inputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> inputTypes(inputRawTypes);
  ::mlir::Type outputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> outputTypes(outputRawTypes);

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  if (parser.parseAttribute(indexAttr, parser.getBuilder().getIntegerType(64), "index", result.attributes))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(inputRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseType(outputRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(outputTypes);
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void StructAccessOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << input();
  p << "[";
  p.printAttributeWithoutType(indexAttr());
  p << "]";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"index"});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(input().getType());
  p << ' ' << "to";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(output().getType());
}

void StructAccessOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace sparlay
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::sparlay::StructAccessOp)

namespace mlir {
namespace sparlay {

//===----------------------------------------------------------------------===//
// ::mlir::sparlay::StructConstructOp definitions
//===----------------------------------------------------------------------===//

StructConstructOpAdaptor::StructConstructOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

StructConstructOpAdaptor::StructConstructOpAdaptor(StructConstructOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange StructConstructOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> StructConstructOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange StructConstructOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange StructConstructOpAdaptor::input() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr StructConstructOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult StructConstructOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> StructConstructOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range StructConstructOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range StructConstructOp::input() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange StructConstructOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StructConstructOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range StructConstructOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StructConstructOp::output() {
  return *getODSResults(0).begin();
}

void StructConstructOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::ValueRange input) {
  odsState.addOperands(input);
  odsState.addTypes(output);
}

void StructConstructOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult StructConstructOp::verify() {
  if (::mlir::failed(StructConstructOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparlayOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparlayOps4(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult StructConstructOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> inputOperands;
  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> inputTypes;
  ::mlir::Type outputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> outputTypes(outputRawTypes);
  if (parser.parseLParen())
    return ::mlir::failure();

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(inputOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(inputTypes))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseType(outputRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(outputTypes);
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void StructConstructOp::print(::mlir::OpAsmPrinter &p) {
  p << "(";
  p << input();
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << input().getTypes();
  p << ' ' << "to";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(output().getType());
}

void StructConstructOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace sparlay
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::sparlay::StructConstructOp)


#endif  // GET_OP_CLASSES

