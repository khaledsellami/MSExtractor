# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: parse/v2/parse.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x14parse/v2/parse.proto\x12\x07parsing\"\x95\x01\n\x0cNamesRequest\x12\x0f\n\x07\x61ppName\x18\x01 \x01(\t\x12\x0f\n\x07\x61ppRepo\x18\x02 \x01(\t\x12\x10\n\x08language\x18\x03 \x01(\t\x12#\n\x05level\x18\x04 \x01(\x0e\x32\x14.parsing.Granularity\x12\x1a\n\risDistributed\x18\x05 \x01(\x08H\x00\x88\x01\x01\x42\x10\n\x0e_isDistributed\"\xd5\x01\n\x0cParseRequest\x12\x0f\n\x07\x61ppName\x18\x01 \x01(\t\x12\x0f\n\x07\x61ppRepo\x18\x02 \x01(\t\x12\x10\n\x08language\x18\x03 \x01(\t\x12$\n\x06\x66ormat\x18\x04 \x01(\x0e\x32\x0f.parsing.FormatH\x00\x88\x01\x01\x12(\n\x05level\x18\x05 \x01(\x0e\x32\x14.parsing.GranularityH\x01\x88\x01\x01\x12\x1a\n\risDistributed\x18\x06 \x01(\x08H\x02\x88\x01\x01\x42\t\n\x07_formatB\x08\n\x06_levelB\x10\n\x0e_isDistributed\"-\n\nParseReply\x12\x1f\n\x06status\x18\x01 \x01(\x0e\x32\x0f.parsing.Status\"\x16\n\x05Names\x12\r\n\x05names\x18\x01 \x03(\t\"\xd1\x01\n\x08MetaData\x12\x1f\n\x06status\x18\x01 \x01(\x0e\x32\x0f.parsing.Status\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x1f\n\x06\x66ormat\x18\x03 \x01(\x0e\x32\x0f.parsing.Format\x12\x14\n\x0c\x63olumn_index\x18\x04 \x01(\x03\x12\x11\n\trow_index\x18\x05 \x01(\x03\x12\x18\n\x0b\x63olumn_name\x18\x06 \x01(\tH\x00\x88\x01\x01\x12\x15\n\x08row_name\x18\x07 \x01(\tH\x01\x88\x01\x01\x42\x0e\n\x0c_column_nameB\x0b\n\t_row_name\"\x17\n\x04\x46ile\x12\x0f\n\x07\x63ontent\x18\x01 \x01(\x0c\"e\n\x11ParseDataResponse\x12%\n\x08metadata\x18\x01 \x01(\x0b\x32\x11.parsing.MetaDataH\x00\x12\x1d\n\x04\x66ile\x18\x02 \x01(\x0b\x32\r.parsing.FileH\x00\x42\n\n\x08response*$\n\x0bGranularity\x12\t\n\x05\x43LASS\x10\x00\x12\n\n\x06METHOD\x10\x01*?\n\x06Status\x12\x0b\n\x07PENDING\x10\x00\x12\x0f\n\x0bIN_PROGRESS\x10\x01\x12\x0b\n\x07SUCCESS\x10\x02\x12\n\n\x06\x46\x41ILED\x10\x03*4\n\x06\x46ormat\x12\x0b\n\x07PARQUET\x10\x00\x12\x07\n\x03\x43SV\x10\x01\x12\x08\n\x04JSON\x10\x02\x12\n\n\x06PICKLE\x10\x03\x32\x8f\x03\n\x06Parser\x12\x38\n\x08parseAll\x12\x15.parsing.ParseRequest\x1a\x13.parsing.ParseReply\"\x00\x12\x33\n\x08getNames\x12\x15.parsing.NamesRequest\x1a\x0e.parsing.Names\"\x00\x12H\n\x0fgetInteractions\x12\x15.parsing.ParseRequest\x1a\x1a.parsing.ParseDataResponse\"\x00\x30\x01\x12\x41\n\x08getCalls\x12\x15.parsing.ParseRequest\x1a\x1a.parsing.ParseDataResponse\"\x00\x30\x01\x12\x41\n\x08getTFIDF\x12\x15.parsing.ParseRequest\x1a\x1a.parsing.ParseDataResponse\"\x00\x30\x01\x12\x46\n\rgetWordCounts\x12\x15.parsing.ParseRequest\x1a\x1a.parsing.ParseDataResponse\"\x00\x30\x01\x42*\n\x12\x63om.decomp.parsingB\x0cParsingProtoP\x01\xa2\x02\x03JPDb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'parse.v2.parse_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  _globals['DESCRIPTOR']._options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\022com.decomp.parsingB\014ParsingProtoP\001\242\002\003JPD'
  _globals['_GRANULARITY']._serialized_start=812
  _globals['_GRANULARITY']._serialized_end=848
  _globals['_STATUS']._serialized_start=850
  _globals['_STATUS']._serialized_end=913
  _globals['_FORMAT']._serialized_start=915
  _globals['_FORMAT']._serialized_end=967
  _globals['_NAMESREQUEST']._serialized_start=34
  _globals['_NAMESREQUEST']._serialized_end=183
  _globals['_PARSEREQUEST']._serialized_start=186
  _globals['_PARSEREQUEST']._serialized_end=399
  _globals['_PARSEREPLY']._serialized_start=401
  _globals['_PARSEREPLY']._serialized_end=446
  _globals['_NAMES']._serialized_start=448
  _globals['_NAMES']._serialized_end=470
  _globals['_METADATA']._serialized_start=473
  _globals['_METADATA']._serialized_end=682
  _globals['_FILE']._serialized_start=684
  _globals['_FILE']._serialized_end=707
  _globals['_PARSEDATARESPONSE']._serialized_start=709
  _globals['_PARSEDATARESPONSE']._serialized_end=810
  _globals['_PARSER']._serialized_start=970
  _globals['_PARSER']._serialized_end=1369
# @@protoc_insertion_point(module_scope)