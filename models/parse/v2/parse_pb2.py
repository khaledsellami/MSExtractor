# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: parse/v2/parse.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='parse/v2/parse.proto',
  package='parsing',
  syntax='proto3',
  serialized_options=b'\n\022com.decomp.parsingB\014ParsingProtoP\001\242\002\003JPD',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x14parse/v2/parse.proto\x12\x07parsing\"\x95\x01\n\x0cNamesRequest\x12\x0f\n\x07\x61ppName\x18\x01 \x01(\t\x12\x0f\n\x07\x61ppRepo\x18\x02 \x01(\t\x12\x10\n\x08language\x18\x03 \x01(\t\x12#\n\x05level\x18\x04 \x01(\x0e\x32\x14.parsing.Granularity\x12\x1a\n\risDistributed\x18\x05 \x01(\x08H\x00\x88\x01\x01\x42\x10\n\x0e_isDistributed\"\xd5\x01\n\x0cParseRequest\x12\x0f\n\x07\x61ppName\x18\x01 \x01(\t\x12\x0f\n\x07\x61ppRepo\x18\x02 \x01(\t\x12\x10\n\x08language\x18\x03 \x01(\t\x12$\n\x06\x66ormat\x18\x04 \x01(\x0e\x32\x0f.parsing.FormatH\x00\x88\x01\x01\x12(\n\x05level\x18\x05 \x01(\x0e\x32\x14.parsing.GranularityH\x01\x88\x01\x01\x12\x1a\n\risDistributed\x18\x06 \x01(\x08H\x02\x88\x01\x01\x42\t\n\x07_formatB\x08\n\x06_levelB\x10\n\x0e_isDistributed\"-\n\nParseReply\x12\x1f\n\x06status\x18\x01 \x01(\x0e\x32\x0f.parsing.Status\"\x16\n\x05Names\x12\r\n\x05names\x18\x01 \x03(\t\"\xd1\x01\n\x08MetaData\x12\x1f\n\x06status\x18\x01 \x01(\x0e\x32\x0f.parsing.Status\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x1f\n\x06\x66ormat\x18\x03 \x01(\x0e\x32\x0f.parsing.Format\x12\x14\n\x0c\x63olumn_index\x18\x04 \x01(\x03\x12\x11\n\trow_index\x18\x05 \x01(\x03\x12\x18\n\x0b\x63olumn_name\x18\x06 \x01(\tH\x00\x88\x01\x01\x12\x15\n\x08row_name\x18\x07 \x01(\tH\x01\x88\x01\x01\x42\x0e\n\x0c_column_nameB\x0b\n\t_row_name\"\x17\n\x04\x46ile\x12\x0f\n\x07\x63ontent\x18\x01 \x01(\x0c\"e\n\x11ParseDataResponse\x12%\n\x08metadata\x18\x01 \x01(\x0b\x32\x11.parsing.MetaDataH\x00\x12\x1d\n\x04\x66ile\x18\x02 \x01(\x0b\x32\r.parsing.FileH\x00\x42\n\n\x08response*$\n\x0bGranularity\x12\t\n\x05\x43LASS\x10\x00\x12\n\n\x06METHOD\x10\x01*?\n\x06Status\x12\x0b\n\x07PENDING\x10\x00\x12\x0f\n\x0bIN_PROGRESS\x10\x01\x12\x0b\n\x07SUCCESS\x10\x02\x12\n\n\x06\x46\x41ILED\x10\x03*4\n\x06\x46ormat\x12\x0b\n\x07PARQUET\x10\x00\x12\x07\n\x03\x43SV\x10\x01\x12\x08\n\x04JSON\x10\x02\x12\n\n\x06PICKLE\x10\x03\x32\x8f\x03\n\x06Parser\x12\x38\n\x08parseAll\x12\x15.parsing.ParseRequest\x1a\x13.parsing.ParseReply\"\x00\x12\x33\n\x08getNames\x12\x15.parsing.NamesRequest\x1a\x0e.parsing.Names\"\x00\x12H\n\x0fgetInteractions\x12\x15.parsing.ParseRequest\x1a\x1a.parsing.ParseDataResponse\"\x00\x30\x01\x12\x41\n\x08getCalls\x12\x15.parsing.ParseRequest\x1a\x1a.parsing.ParseDataResponse\"\x00\x30\x01\x12\x41\n\x08getTFIDF\x12\x15.parsing.ParseRequest\x1a\x1a.parsing.ParseDataResponse\"\x00\x30\x01\x12\x46\n\rgetWordCounts\x12\x15.parsing.ParseRequest\x1a\x1a.parsing.ParseDataResponse\"\x00\x30\x01\x42*\n\x12\x63om.decomp.parsingB\x0cParsingProtoP\x01\xa2\x02\x03JPDb\x06proto3'
)

_GRANULARITY = _descriptor.EnumDescriptor(
  name='Granularity',
  full_name='parsing.Granularity',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='CLASS', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='METHOD', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=812,
  serialized_end=848,
)
_sym_db.RegisterEnumDescriptor(_GRANULARITY)

Granularity = enum_type_wrapper.EnumTypeWrapper(_GRANULARITY)
_STATUS = _descriptor.EnumDescriptor(
  name='Status',
  full_name='parsing.Status',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='PENDING', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='IN_PROGRESS', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SUCCESS', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='FAILED', index=3, number=3,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=850,
  serialized_end=913,
)
_sym_db.RegisterEnumDescriptor(_STATUS)

Status = enum_type_wrapper.EnumTypeWrapper(_STATUS)
_FORMAT = _descriptor.EnumDescriptor(
  name='Format',
  full_name='parsing.Format',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='PARQUET', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='CSV', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='JSON', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='PICKLE', index=3, number=3,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=915,
  serialized_end=967,
)
_sym_db.RegisterEnumDescriptor(_FORMAT)

Format = enum_type_wrapper.EnumTypeWrapper(_FORMAT)
CLASS = 0
METHOD = 1
PENDING = 0
IN_PROGRESS = 1
SUCCESS = 2
FAILED = 3
PARQUET = 0
CSV = 1
JSON = 2
PICKLE = 3



_NAMESREQUEST = _descriptor.Descriptor(
  name='NamesRequest',
  full_name='parsing.NamesRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='appName', full_name='parsing.NamesRequest.appName', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='appRepo', full_name='parsing.NamesRequest.appRepo', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='language', full_name='parsing.NamesRequest.language', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='level', full_name='parsing.NamesRequest.level', index=3,
      number=4, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='isDistributed', full_name='parsing.NamesRequest.isDistributed', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='_isDistributed', full_name='parsing.NamesRequest._isDistributed',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=34,
  serialized_end=183,
)


_PARSEREQUEST = _descriptor.Descriptor(
  name='ParseRequest',
  full_name='parsing.ParseRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='appName', full_name='parsing.ParseRequest.appName', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='appRepo', full_name='parsing.ParseRequest.appRepo', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='language', full_name='parsing.ParseRequest.language', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='format', full_name='parsing.ParseRequest.format', index=3,
      number=4, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='level', full_name='parsing.ParseRequest.level', index=4,
      number=5, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='isDistributed', full_name='parsing.ParseRequest.isDistributed', index=5,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='_format', full_name='parsing.ParseRequest._format',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
    _descriptor.OneofDescriptor(
      name='_level', full_name='parsing.ParseRequest._level',
      index=1, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
    _descriptor.OneofDescriptor(
      name='_isDistributed', full_name='parsing.ParseRequest._isDistributed',
      index=2, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=186,
  serialized_end=399,
)


_PARSEREPLY = _descriptor.Descriptor(
  name='ParseReply',
  full_name='parsing.ParseReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='status', full_name='parsing.ParseReply.status', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=401,
  serialized_end=446,
)


_NAMES = _descriptor.Descriptor(
  name='Names',
  full_name='parsing.Names',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='names', full_name='parsing.Names.names', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=448,
  serialized_end=470,
)


_METADATA = _descriptor.Descriptor(
  name='MetaData',
  full_name='parsing.MetaData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='status', full_name='parsing.MetaData.status', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='name', full_name='parsing.MetaData.name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='format', full_name='parsing.MetaData.format', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='column_index', full_name='parsing.MetaData.column_index', index=3,
      number=4, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='row_index', full_name='parsing.MetaData.row_index', index=4,
      number=5, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='column_name', full_name='parsing.MetaData.column_name', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='row_name', full_name='parsing.MetaData.row_name', index=6,
      number=7, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='_column_name', full_name='parsing.MetaData._column_name',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
    _descriptor.OneofDescriptor(
      name='_row_name', full_name='parsing.MetaData._row_name',
      index=1, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=473,
  serialized_end=682,
)


_FILE = _descriptor.Descriptor(
  name='File',
  full_name='parsing.File',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='content', full_name='parsing.File.content', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=684,
  serialized_end=707,
)


_PARSEDATARESPONSE = _descriptor.Descriptor(
  name='ParseDataResponse',
  full_name='parsing.ParseDataResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='metadata', full_name='parsing.ParseDataResponse.metadata', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='file', full_name='parsing.ParseDataResponse.file', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='response', full_name='parsing.ParseDataResponse.response',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=709,
  serialized_end=810,
)

_NAMESREQUEST.fields_by_name['level'].enum_type = _GRANULARITY
_NAMESREQUEST.oneofs_by_name['_isDistributed'].fields.append(
  _NAMESREQUEST.fields_by_name['isDistributed'])
_NAMESREQUEST.fields_by_name['isDistributed'].containing_oneof = _NAMESREQUEST.oneofs_by_name['_isDistributed']
_PARSEREQUEST.fields_by_name['format'].enum_type = _FORMAT
_PARSEREQUEST.fields_by_name['level'].enum_type = _GRANULARITY
_PARSEREQUEST.oneofs_by_name['_format'].fields.append(
  _PARSEREQUEST.fields_by_name['format'])
_PARSEREQUEST.fields_by_name['format'].containing_oneof = _PARSEREQUEST.oneofs_by_name['_format']
_PARSEREQUEST.oneofs_by_name['_level'].fields.append(
  _PARSEREQUEST.fields_by_name['level'])
_PARSEREQUEST.fields_by_name['level'].containing_oneof = _PARSEREQUEST.oneofs_by_name['_level']
_PARSEREQUEST.oneofs_by_name['_isDistributed'].fields.append(
  _PARSEREQUEST.fields_by_name['isDistributed'])
_PARSEREQUEST.fields_by_name['isDistributed'].containing_oneof = _PARSEREQUEST.oneofs_by_name['_isDistributed']
_PARSEREPLY.fields_by_name['status'].enum_type = _STATUS
_METADATA.fields_by_name['status'].enum_type = _STATUS
_METADATA.fields_by_name['format'].enum_type = _FORMAT
_METADATA.oneofs_by_name['_column_name'].fields.append(
  _METADATA.fields_by_name['column_name'])
_METADATA.fields_by_name['column_name'].containing_oneof = _METADATA.oneofs_by_name['_column_name']
_METADATA.oneofs_by_name['_row_name'].fields.append(
  _METADATA.fields_by_name['row_name'])
_METADATA.fields_by_name['row_name'].containing_oneof = _METADATA.oneofs_by_name['_row_name']
_PARSEDATARESPONSE.fields_by_name['metadata'].message_type = _METADATA
_PARSEDATARESPONSE.fields_by_name['file'].message_type = _FILE
_PARSEDATARESPONSE.oneofs_by_name['response'].fields.append(
  _PARSEDATARESPONSE.fields_by_name['metadata'])
_PARSEDATARESPONSE.fields_by_name['metadata'].containing_oneof = _PARSEDATARESPONSE.oneofs_by_name['response']
_PARSEDATARESPONSE.oneofs_by_name['response'].fields.append(
  _PARSEDATARESPONSE.fields_by_name['file'])
_PARSEDATARESPONSE.fields_by_name['file'].containing_oneof = _PARSEDATARESPONSE.oneofs_by_name['response']
DESCRIPTOR.message_types_by_name['NamesRequest'] = _NAMESREQUEST
DESCRIPTOR.message_types_by_name['ParseRequest'] = _PARSEREQUEST
DESCRIPTOR.message_types_by_name['ParseReply'] = _PARSEREPLY
DESCRIPTOR.message_types_by_name['Names'] = _NAMES
DESCRIPTOR.message_types_by_name['MetaData'] = _METADATA
DESCRIPTOR.message_types_by_name['File'] = _FILE
DESCRIPTOR.message_types_by_name['ParseDataResponse'] = _PARSEDATARESPONSE
DESCRIPTOR.enum_types_by_name['Granularity'] = _GRANULARITY
DESCRIPTOR.enum_types_by_name['Status'] = _STATUS
DESCRIPTOR.enum_types_by_name['Format'] = _FORMAT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

NamesRequest = _reflection.GeneratedProtocolMessageType('NamesRequest', (_message.Message,), {
  'DESCRIPTOR' : _NAMESREQUEST,
  '__module__' : 'parse.v2.parse_pb2'
  # @@protoc_insertion_point(class_scope:parsing.NamesRequest)
  })
_sym_db.RegisterMessage(NamesRequest)

ParseRequest = _reflection.GeneratedProtocolMessageType('ParseRequest', (_message.Message,), {
  'DESCRIPTOR' : _PARSEREQUEST,
  '__module__' : 'parse.v2.parse_pb2'
  # @@protoc_insertion_point(class_scope:parsing.ParseRequest)
  })
_sym_db.RegisterMessage(ParseRequest)

ParseReply = _reflection.GeneratedProtocolMessageType('ParseReply', (_message.Message,), {
  'DESCRIPTOR' : _PARSEREPLY,
  '__module__' : 'parse.v2.parse_pb2'
  # @@protoc_insertion_point(class_scope:parsing.ParseReply)
  })
_sym_db.RegisterMessage(ParseReply)

Names = _reflection.GeneratedProtocolMessageType('Names', (_message.Message,), {
  'DESCRIPTOR' : _NAMES,
  '__module__' : 'parse.v2.parse_pb2'
  # @@protoc_insertion_point(class_scope:parsing.Names)
  })
_sym_db.RegisterMessage(Names)

MetaData = _reflection.GeneratedProtocolMessageType('MetaData', (_message.Message,), {
  'DESCRIPTOR' : _METADATA,
  '__module__' : 'parse.v2.parse_pb2'
  # @@protoc_insertion_point(class_scope:parsing.MetaData)
  })
_sym_db.RegisterMessage(MetaData)

File = _reflection.GeneratedProtocolMessageType('File', (_message.Message,), {
  'DESCRIPTOR' : _FILE,
  '__module__' : 'parse.v2.parse_pb2'
  # @@protoc_insertion_point(class_scope:parsing.File)
  })
_sym_db.RegisterMessage(File)

ParseDataResponse = _reflection.GeneratedProtocolMessageType('ParseDataResponse', (_message.Message,), {
  'DESCRIPTOR' : _PARSEDATARESPONSE,
  '__module__' : 'parse.v2.parse_pb2'
  # @@protoc_insertion_point(class_scope:parsing.ParseDataResponse)
  })
_sym_db.RegisterMessage(ParseDataResponse)


DESCRIPTOR._options = None

_PARSER = _descriptor.ServiceDescriptor(
  name='Parser',
  full_name='parsing.Parser',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=970,
  serialized_end=1369,
  methods=[
  _descriptor.MethodDescriptor(
    name='parseAll',
    full_name='parsing.Parser.parseAll',
    index=0,
    containing_service=None,
    input_type=_PARSEREQUEST,
    output_type=_PARSEREPLY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='getNames',
    full_name='parsing.Parser.getNames',
    index=1,
    containing_service=None,
    input_type=_NAMESREQUEST,
    output_type=_NAMES,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='getInteractions',
    full_name='parsing.Parser.getInteractions',
    index=2,
    containing_service=None,
    input_type=_PARSEREQUEST,
    output_type=_PARSEDATARESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='getCalls',
    full_name='parsing.Parser.getCalls',
    index=3,
    containing_service=None,
    input_type=_PARSEREQUEST,
    output_type=_PARSEDATARESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='getTFIDF',
    full_name='parsing.Parser.getTFIDF',
    index=4,
    containing_service=None,
    input_type=_PARSEREQUEST,
    output_type=_PARSEDATARESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='getWordCounts',
    full_name='parsing.Parser.getWordCounts',
    index=5,
    containing_service=None,
    input_type=_PARSEREQUEST,
    output_type=_PARSEDATARESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_PARSER)

DESCRIPTOR.services_by_name['Parser'] = _PARSER

# @@protoc_insertion_point(module_scope)