"""Descriptor configuration loading for Modbus function codes."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import json


class FieldType(Enum):
    U8 = "u8"
    U16 = "u16"
    U32 = "u32"
    U64 = "u64"
    I8 = "i8"
    I16 = "i16"
    I32 = "i32"
    I64 = "i64"
    BYTES = "bytes"
    BITS = "bits"

    def byte_size(self) -> Optional[int]:
        _sizes = {
            FieldType.U8: 1, FieldType.I8: 1,
            FieldType.U16: 2, FieldType.I16: 2,
            FieldType.U32: 4, FieldType.I32: 4,
            FieldType.U64: 8, FieldType.I64: 8,
        }
        return _sizes.get(self)


@dataclass
class FieldDescriptor:
    name: str
    field_type: FieldType
    length: Optional[int] = None
    length_from: Optional[str] = None
    scale: Optional[float] = None
    enum_map: Optional[Dict[int, str]] = None

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "FieldDescriptor":
        return FieldDescriptor(
            name=data["name"],
            field_type=FieldType(data["field_type"]),
            length=data.get("length"),
            length_from=data.get("length_from"),
            scale=data.get("scale"),
            enum_map=data.get("enum_map"),
        )


@dataclass
class FunctionDescriptor:
    function_code: int
    name: str
    request: List[FieldDescriptor] = field(default_factory=list)
    response: List[FieldDescriptor] = field(default_factory=list)

    @staticmethod
    def from_dict(fc: int, data: Dict[str, Any]) -> "FunctionDescriptor":
        return FunctionDescriptor(
            function_code=fc,
            name=data.get("name", f"fc_{fc}"),
            request=[FieldDescriptor.from_dict(f) for f in data.get("request", [])],
            response=[FieldDescriptor.from_dict(f) for f in data.get("response", [])],
        )


@dataclass
class Config:
    functions: Dict[int, FunctionDescriptor] = field(default_factory=dict)

    def get_descriptor(self, function_code: int) -> Optional[FunctionDescriptor]:
        return self.functions.get(function_code)

    def get_fields(self, function_code: int, direction: str) -> List[FieldDescriptor]:
        """Get field descriptors for a given function code and direction."""
        desc = self.get_descriptor(function_code)
        if desc is None:
            return []
        if direction == "c2s":
            return desc.request
        else:
            return desc.response

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Config":
        functions = {}
        for fc_str, fc_data in data.get("functions", {}).items():
            fc = int(fc_str)
            functions[fc] = FunctionDescriptor.from_dict(fc, fc_data)
        return Config(functions=functions)

    @staticmethod
    def load(path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Config.from_dict(data)
