"""Safe, explicit persistence for native Numeta library bundles."""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import inspect
import json
import math
import os
import platform
import shutil
import tempfile
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import numpy as np

from .ast import Procedure
from .array_shape import ArrayShape, SCALAR
from .compiler import Compiler
from .exceptions import (
    CorruptLibraryError,
    IncompatibleLibraryError,
    LegacyLibraryFormatError,
)
from .external_library import ExternalLibrary
from .library_artifacts import _persist_compiled_artifacts
from .library_linking import (
    _active_compiled_targets_by_name,
    _collect_compiled_target_closure,
)
from .library_signature import signature_id
from .native_abi import (
    format_metadata_mismatches,
    metadata_mismatches,
    native_runtime_abi,
)
from .pyc_extension import PyCExtension
from .settings import settings
from .signature import ParameterInfo
from ._version import __version__ as NUMETA_VERSION

BUNDLE_FORMAT = "numeta-library"
BUNDLE_FORMAT_VERSION = 2


def _make_persisted_procedure(name: str, arguments) -> Procedure:
    """Create an exact ``Procedure`` carrying only stable ABI metadata."""
    from .ast import Variable

    procedure = Procedure(name)
    procedure._dependencies = {}
    procedure.get_dependencies = MethodType(lambda self: self._dependencies, procedure)
    variables = {}
    for payload in arguments:
        variable = Variable(
            payload["name"],
            dtype=_decode_dtype(payload["dtype"]),
            shape=_decode_shape(payload["shape"], variables=variables),
            intent=payload["intent"],
            pointer=payload["pointer"],
            target=payload["target"],
            allocatable=payload["allocatable"],
            parameter=payload["parameter"],
            bind_c=payload["bind_c"],
            use_c_types=payload["use_c_types"],
            pass_by_value=payload["pass_by_value"],
            c_const=payload["c_const"],
            c_static=payload["c_static"],
            c_restrict=payload["c_restrict"],
            c_volatile=payload["c_volatile"],
        )
        procedure.add_variable(variable)
        variables[variable.name] = variable
    return procedure


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _encode_numpy_dtype(dtype) -> dict:
    dtype = np.dtype(dtype)
    if dtype.fields is not None:
        return {
            "kind": "struct",
            "names": list(dtype.names or ()),
            "fields": [
                {
                    "dtype": _encode_numpy_dtype(dtype.fields[name][0]),
                    "offset": dtype.fields[name][1],
                    "title": dtype.fields[name][2] if len(dtype.fields[name]) > 2 else None,
                }
                for name in dtype.names or ()
            ],
            "itemsize": dtype.itemsize,
            "aligned": bool(dtype.isalignedstruct),
        }
    if dtype.subdtype is not None:
        base, shape = dtype.subdtype
        return {"kind": "subarray", "base": _encode_numpy_dtype(base), "shape": list(shape)}
    return {"kind": "primitive", "str": dtype.str}


def _decode_numpy_dtype(payload) -> np.dtype:
    kind = payload.get("kind")
    if kind == "primitive":
        return np.dtype(payload["str"])
    if kind == "subarray":
        return np.dtype((_decode_numpy_dtype(payload["base"]), tuple(payload["shape"])))
    if kind == "struct":
        fields = payload["fields"]
        spec = {
            "names": payload["names"],
            "formats": [_decode_numpy_dtype(field["dtype"]) for field in fields],
            "offsets": [field["offset"] for field in fields],
            "itemsize": payload["itemsize"],
        }
        titles = [field.get("title") for field in fields]
        if any(title is not None for title in titles):
            spec["titles"] = titles
        return np.dtype(spec, align=payload.get("aligned", False))
    raise CorruptLibraryError(f"Unknown NumPy dtype encoding {kind!r}")


def _encode_dtype(dtype) -> dict:
    from .datatype import DataTypeMeta, VectorType

    if not isinstance(dtype, DataTypeMeta):
        raise TypeError(f"Expected a Numeta datatype, got {type(dtype)!r}")
    if issubclass(dtype, VectorType):
        return {
            "kind": "vector",
            "base": _encode_dtype(dtype.base_dtype()),
            "lanes": dtype.lanes(),
        }
    if dtype.is_struct():
        return {
            "kind": "struct",
            "name": dtype._name,
            "numpy_dtype": _encode_numpy_dtype(dtype.get_numpy()),
            "members": [
                {
                    "name": name,
                    "dtype": _encode_dtype(member_dtype),
                    "shape": _encode_shape(shape),
                }
                for name, member_dtype, shape in dtype._members
            ],
        }
    return {"kind": "builtin", "name": dtype._name}


def _datatype_by_name(name: str):
    from .datatype import DataType

    pending = list(DataType.__subclasses__())
    seen = set()
    while pending:
        dtype = pending.pop()
        if dtype in seen:
            continue
        seen.add(dtype)
        if getattr(dtype, "_name", None) == name and not getattr(dtype, "_is_struct", False):
            return dtype
        pending.extend(dtype.__subclasses__())
    raise CorruptLibraryError(f"Unknown Numeta datatype {name!r} in bundle manifest")


def _decode_dtype(payload):
    from .datatype import DataTypeMeta, make_struct_type, make_vector_type

    kind = payload.get("kind")
    if kind == "builtin":
        return _datatype_by_name(payload["name"])
    if kind == "vector":
        return make_vector_type(_decode_dtype(payload["base"]), payload["lanes"])
    if kind == "struct":
        members = [
            (member["name"], _decode_dtype(member["dtype"]), _decode_shape(member["shape"]))
            for member in payload["members"]
        ]
        np_dtype = _decode_numpy_dtype(payload["numpy_dtype"])
        existing = DataTypeMeta._np_dtype.get(np_dtype)
        if existing is not None:
            return existing
        return make_struct_type(np_dtype, members, name=payload["name"])
    raise CorruptLibraryError(f"Unknown datatype encoding {kind!r}")


def _encode_shape(shape: ArrayShape) -> dict:
    if shape.is_unknown:
        dims = None
    elif shape.is_shape_vector:
        return {
            "dims": None,
            "fortran_order": shape.fortran_order,
            "shape_vector": {
                "name": shape._shape.name,
                "rank": shape.rank,
            },
        }
    else:
        dims = [_encode_value(dim) for dim in shape.as_tuple()]
    return {"dims": dims, "fortran_order": shape.fortran_order}


def _decode_shape(payload, *, variables=None) -> ArrayShape:
    shape_vector = payload.get("shape_vector")
    if shape_vector is not None:
        if variables is None or shape_vector["name"] not in variables:
            raise CorruptLibraryError(
                f"Shape vector {shape_vector['name']!r} is missing from persisted arguments"
            )
        return ArrayShape.from_shape_vector(
            variables[shape_vector["name"]],
            shape_vector["rank"],
            fortran_order=bool(payload.get("fortran_order", False)),
        )
    dims = payload["dims"]
    if dims is not None:
        dims = tuple(_decode_value(dim) for dim in dims)
    return ArrayShape(dims, fortran_order=bool(payload.get("fortran_order", False)))


def _encode_value(value):
    from .datatype import ArrayType, DataTypeMeta, PointerType

    if value is inspect._empty:
        return {"kind": "inspect.empty"}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"kind": "float", "value": value.hex()}
    if isinstance(value, complex):
        return {
            "kind": "complex",
            "real": _encode_value(value.real),
            "imag": _encode_value(value.imag),
        }
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_encode_value(item) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_encode_value(item) for item in value]}
    if isinstance(value, dict):
        items = [[_encode_value(key), _encode_value(item)] for key, item in value.items()]
        items.sort(key=lambda pair: _canonical_json(pair[0]))
        return {"kind": "mapping", "items": items}
    if isinstance(value, slice):
        return {
            "kind": "slice",
            "start": _encode_value(value.start),
            "stop": _encode_value(value.stop),
            "step": _encode_value(value.step),
        }
    if isinstance(value, np.ndarray):
        return {
            "kind": "numpy.ndarray",
            "dtype": _encode_numpy_dtype(value.dtype),
            "shape": list(value.shape),
            "value": _encode_value(value.tolist()),
        }
    if isinstance(value, np.generic):
        return {
            "kind": "numpy.scalar",
            "dtype": _encode_numpy_dtype(value.dtype),
            "value": _encode_value(value.item()),
        }
    if isinstance(value, np.dtype):
        return {"kind": "numpy.dtype", "value": _encode_numpy_dtype(value)}
    if isinstance(value, DataTypeMeta):
        return {"kind": "numeta.dtype", "value": _encode_dtype(value)}
    if isinstance(value, PointerType):
        return {
            "kind": "numeta.pointer",
            "dtype": _encode_dtype(value.dtype),
            "const": value.const,
            "restrict": value.restrict,
            "volatile": value.volatile,
        }
    if isinstance(value, ArrayType):
        return {
            "kind": "numeta.array",
            "dtype": _encode_dtype(value.dtype),
            "shape": _encode_shape(value.shape),
        }
    if isinstance(value, ArrayShape):
        return {"kind": "numeta.shape", "value": _encode_shape(value)}
    if isinstance(value, type):
        if value in (bool, int, float, complex, str):
            return {"kind": "python.type", "name": value.__name__}
        try:
            dtype = np.dtype(value)
        except TypeError:
            dtype = None
        if dtype is not None:
            return {"kind": "numpy.type", "value": _encode_numpy_dtype(dtype)}
    if type(value).__name__ == "LiteralNode":
        return _encode_value(value.value)
    raise TypeError(
        "Cannot persist value of type "
        f"{type(value).__module__}.{type(value).__qualname__}; use a stable scalar, "
        "container, NumPy dtype/value, or Numeta type descriptor"
    )


def _decode_value(payload):
    from .datatype import ArrayType, PointerType

    if payload is None or isinstance(payload, (bool, int, float, str)):
        return payload
    if not isinstance(payload, dict) or "kind" not in payload:
        raise CorruptLibraryError(f"Invalid encoded value {payload!r}")
    kind = payload["kind"]
    if kind == "inspect.empty":
        return inspect._empty
    if kind == "float":
        return float.fromhex(payload["value"])
    if kind == "complex":
        return complex(_decode_value(payload["real"]), _decode_value(payload["imag"]))
    if kind in {"tuple", "list"}:
        items = [_decode_value(item) for item in payload["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "mapping":
        return {_decode_value(key): _decode_value(value) for key, value in payload["items"]}
    if kind == "slice":
        return slice(
            _decode_value(payload["start"]),
            _decode_value(payload["stop"]),
            _decode_value(payload["step"]),
        )
    if kind == "numpy.ndarray":
        dtype = _decode_numpy_dtype(payload["dtype"])
        return np.asarray(_decode_value(payload["value"]), dtype=dtype).reshape(payload["shape"])
    if kind == "numpy.scalar":
        dtype = _decode_numpy_dtype(payload["dtype"])
        return np.asarray(_decode_value(payload["value"]), dtype=dtype)[()]
    if kind == "numpy.dtype":
        return _decode_numpy_dtype(payload["value"])
    if kind == "numeta.dtype":
        return _decode_dtype(payload["value"])
    if kind == "numeta.pointer":
        return PointerType(
            dtype=_decode_dtype(payload["dtype"]),
            const=payload.get("const", False),
            restrict=payload.get("restrict", False),
            volatile=payload.get("volatile", False),
        )
    if kind == "numeta.array":
        return ArrayType(
            dtype=_decode_dtype(payload["dtype"]), shape=_decode_shape(payload["shape"])
        )
    if kind == "numeta.shape":
        return _decode_shape(payload["value"])
    if kind == "python.type":
        return {"bool": bool, "int": int, "float": float, "complex": complex, "str": str}[
            payload["name"]
        ]
    if kind == "numpy.type":
        return _decode_numpy_dtype(payload["value"]).type
    raise CorruptLibraryError(f"Unknown encoded value kind {kind!r}")


def _encode_parameter(parameter: ParameterInfo) -> dict:
    return {
        "name": parameter.name,
        "kind": parameter.kind.value,
        "default": _encode_value(parameter.default),
        "is_comptime": parameter.is_comptime,
    }


def _decode_parameter(payload) -> ParameterInfo:
    return ParameterInfo(
        payload["name"],
        inspect._ParameterKind(payload["kind"]),
        _decode_value(payload["default"]),
        bool(payload["is_comptime"]),
    )


def _cpu_features() -> list[str]:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return []
    for line in cpuinfo.read_text(errors="replace").splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() in {"flags", "features"}:
            return sorted(set(value.split()))
    return []


def _runtime_abi() -> dict:
    return native_runtime_abi()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checksums(bundle: Path) -> dict[str, str]:
    return {
        path.relative_to(bundle).as_posix(): _sha256(path)
        for path in sorted(bundle.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def _resolve_bundle_path(bundle: Path, relative: str) -> Path:
    candidate = (bundle / relative).resolve()
    try:
        candidate.relative_to(bundle.resolve())
    except ValueError as exc:
        raise CorruptLibraryError(f"Bundle path escapes its root: {relative!r}") from exc
    return candidate


@contextmanager
def _bundle_lock(parent: Path, name: str):
    import fcntl

    lock_id = hashlib.sha256(str(parent).encode()).hexdigest()[:16]
    lock_path = Path(tempfile.gettempdir()) / f"numeta-{name}-{lock_id}.lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _recover_bundle(parent: Path, name: str) -> Path:
    bundle = parent / f"{name}.numeta"
    backups = sorted(parent.glob(f".{name}.numeta.backup.*"), key=lambda path: path.stat().st_mtime)
    if not bundle.exists() and backups:
        os.replace(backups[-1], bundle)
        backups.pop()
    for backup in backups:
        shutil.rmtree(backup, ignore_errors=True)
    return bundle


def _iter_optional(value):
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return value
    return (value,)


def _external_id(dependency) -> str:
    identity = f"{getattr(dependency, 'name', '')}:{getattr(dependency, 'library_name', '')}"
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


def _serialize_external(dependency, stage: Path) -> tuple[str, dict, list[Path]]:
    external_id = _external_id(dependency)
    copied_objects = []
    object_paths = []
    object_dir = stage / "artifacts" / "external" / external_id
    for source_value in _iter_optional(dependency.obj_files):
        source = Path(source_value)
        if not source.exists():
            raise FileNotFoundError(f"Missing external object file {source}")
        object_dir.mkdir(parents=True, exist_ok=True)
        target = object_dir / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        copied_objects.append(target)
        object_paths.append(target.relative_to(stage).as_posix())

    payload = {
        "name": dependency.name,
        "library_name": getattr(dependency, "library_name", dependency.name),
        "to_link": bool(dependency.to_link),
        "object_files": object_paths,
        "include": [str(item) for item in _iter_optional(dependency.include)],
        "path": [str(item) for item in _iter_optional(dependency.path)],
        "rpath": [str(item) for item in _iter_optional(dependency.rpath)],
        "additional_flags": list(_iter_optional(dependency.additional_flags)),
    }
    return external_id, payload, copied_objects


def _serialize_function(function) -> dict:
    specializations = []
    for signature, compiled in function._compiled_functions.items():
        specializations.append(
            {
                "signature": _encode_value(signature),
                "signature_id": signature_id(signature),
                "symbol": compiled.func_name,
                "returns": _encode_value(function.return_signatures.get(signature, [])),
            }
        )
    return {
        "name": function.name,
        "backend": function.backend,
        "do_checks": function.do_checks,
        "compile_flags": list(function.compile_flags),
        "inline": function.inline,
        "simd_arch": function.simd_arch,
        "simd_features": list(function.simd_features),
        "c_name": function.c_name,
        "c_attributes": list(function.c_attributes),
        "c_linkage": function.c_linkage,
        "emit_mode": function.emit_mode,
        "use_c_dispatch": function._use_c_dispatch_instance,
        "params": [_encode_parameter(parameter) for parameter in function.params],
        "fixed_param_indices": list(function.fixed_param_indices),
        "n_positional_or_default_args": function.n_positional_or_default_args,
        "catch_var_positional_name": function.catch_var_positional_name,
        "specializations": specializations,
    }


def _serialize_global(key, target) -> dict:
    variables = []
    for variable in target.symbolic_function.variables.values():
        variables.append(
            {
                "name": variable.name,
                "dtype": _encode_dtype(variable.dtype),
                "shape": _encode_shape(variable._shape),
            }
        )
    return {"key": key, "symbol": target.func_name, "variables": variables}


def _serialize_argument(variable) -> dict:
    return {
        "name": variable.name,
        "dtype": _encode_dtype(variable.dtype),
        "shape": _encode_shape(variable._shape),
        "intent": variable.intent,
        "pointer": variable.pointer,
        "target": variable.target,
        "allocatable": variable.allocatable,
        "parameter": variable.parameter,
        "bind_c": variable.bind_c,
        "use_c_types": variable.use_c_types,
        "pass_by_value": variable.pass_by_value,
        "c_const": variable.c_const,
        "c_static": variable.c_static,
        "c_restrict": variable.c_restrict,
        "c_volatile": variable.c_volatile,
    }


def _restore_external(payload, bundle: Path) -> ExternalLibrary:
    object_files = [_resolve_bundle_path(bundle, path) for path in payload["object_files"]]
    external = ExternalLibrary(
        payload["name"],
        path=payload["path"],
        include=payload["include"],
        obj_files=object_files,
        rpath=payload["rpath"],
        additional_flags=payload["additional_flags"],
        to_link=payload["to_link"],
    )
    external.library_name = payload["library_name"]
    return external


def _restore_compiled_target(payload, bundle: Path):
    from .numeta_function import NumetaCompiledFunction

    symbol = payload["symbol"]
    symbolic = _make_persisted_procedure(symbol, payload.get("arguments", []))
    target = NumetaCompiledFunction.__new__(NumetaCompiledFunction)
    ExternalLibrary.__init__(target, symbol, to_link=True)
    target._library_name = payload["library_name"]
    target.func_name = symbol
    target.symbolic_function = symbolic
    symbolic.parent = target
    target._path = bundle / "libraries"
    target._rpath = target._path
    includes = [_resolve_bundle_path(bundle, path) for path in payload["include_dirs"]]
    objects = [_resolve_bundle_path(bundle, path) for path in payload["object_files"]]
    target._include = includes[0] if includes else bundle
    target._obj_files = objects[0] if objects else None
    target._source_files = [
        _resolve_bundle_path(bundle, path) for path in payload.get("source_files", [])
    ]
    target.do_checks = payload["do_checks"]
    target.compile_flags = list(payload["compile_flags"])
    target.backend = payload["backend"]
    target.simd_arch = payload["simd_arch"]
    target.simd_features = tuple(payload["simd_features"])
    target.c_attributes = tuple(payload["c_attributes"])
    target.c_linkage = payload["c_linkage"]
    target.emit_mode = payload["emit_mode"]
    target._requires_math = payload["requires_math"]
    target._compiler_identity = payload["compiler"]
    target._codegen_abi_settings = payload["codegen_settings"]
    target.compiled = True
    target._loaded_from_bundle = True
    return target


def _restore_function(payload, targets, bundle: Path):
    from .numeta_function import NumetaFunction

    function = NumetaFunction.__new__(NumetaFunction)
    state = {
        "name": payload["name"],
        "hidden": True,
        "external": True,
        "_path": None,
        "_rpath": None,
        "_include": None,
        "_obj_files": None,
        "additional_flags": None,
        "to_link": True,
        "namespaces": {},
        "procedures": {},
        "variables": {},
        "directory": bundle,
        "do_checks": payload["do_checks"],
        "compile_flags": list(payload["compile_flags"]),
        "backend": payload["backend"],
        "simd_arch": payload["simd_arch"],
        "simd_features": tuple(payload["simd_features"]),
        "c_name": payload["c_name"],
        "c_attributes": tuple(payload["c_attributes"]),
        "c_linkage": payload["c_linkage"],
        "emit_mode": payload["emit_mode"],
        "namer": None,
        "inline": payload["inline"],
        "_func": None,
        "params": [_decode_parameter(parameter) for parameter in payload["params"]],
        "fixed_param_indices": list(payload["fixed_param_indices"]),
        "n_positional_or_default_args": payload["n_positional_or_default_args"],
        "catch_var_positional_name": payload["catch_var_positional_name"],
        "return_signatures": {},
        "_compiled_functions": {},
        "_wrapper_specs": {},
        "_pyc_extensions": {},
        "_library_pyc_extension": None,
        "_fast_call": {},
        "_use_c_dispatch_instance": payload["use_c_dispatch"],
    }
    for specialization in payload["specializations"]:
        signature = _decode_value(specialization["signature"])
        if signature_id(signature) != specialization["signature_id"]:
            raise CorruptLibraryError(
                f"Signature identifier mismatch for function {payload['name']!r}"
            )
        state["return_signatures"][signature] = _decode_value(specialization["returns"])
        state["_compiled_functions"][signature] = targets[specialization["symbol"]]
    function.__setstate__(state)
    for signature in function._compiled_functions:
        function._wrapper_specs[signature] = function.build_wrapper_spec(signature)
    return function


def _restore_global(payload, target):
    from .ast import Namespace, Variable

    namespace = Namespace(target.func_name)
    for variable_payload in payload["variables"]:
        Variable(
            variable_payload["name"],
            dtype=_decode_dtype(variable_payload["dtype"]),
            shape=_decode_shape(variable_payload["shape"]),
            parent=namespace,
        )
    namespace.parent = target
    target.symbolic_function = namespace
    return target


def _validate_manifest(manifest: dict, bundle: Path) -> None:
    if manifest.get("format") != BUNDLE_FORMAT:
        raise CorruptLibraryError("Not a Numeta library bundle")
    if manifest.get("format_version") != BUNDLE_FORMAT_VERSION:
        raise IncompatibleLibraryError(
            f"Unsupported bundle format version {manifest.get('format_version')!r}; "
            f"expected {BUNDLE_FORMAT_VERSION}"
        )

    expected = manifest.get("checksums")
    if not isinstance(expected, dict):
        raise CorruptLibraryError("Bundle manifest has no checksum table")
    actual_files = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual_files != set(expected):
        raise CorruptLibraryError("Bundle file list does not match its checksum table")
    for relative, digest in expected.items():
        path = _resolve_bundle_path(bundle, relative)
        if _sha256(path) != digest:
            raise CorruptLibraryError(f"Checksum mismatch for {relative}")

    referenced_files = [manifest["core_library"]]
    if manifest.get("wrapper_library") is not None:
        referenced_files.append(manifest["wrapper_library"])
    for target in manifest["targets"].values():
        for field in ("object_files", "source_files", "module_files"):
            referenced_files.extend(target.get(field, []))
    for dependency in manifest["external_dependencies"].values():
        referenced_files.extend(dependency.get("object_files", []))
    missing_references = [path for path in referenced_files if path not in expected]
    if missing_references:
        raise CorruptLibraryError(
            "Bundle manifest references files outside its checksum table: "
            + ", ".join(sorted(missing_references))
        )

    saved_abi = manifest.get("abi", {})
    current_abi = _runtime_abi()
    if current_abi["system"] != "Linux":
        raise IncompatibleLibraryError("Numeta 0.6 native bundles support Linux only")
    mismatches = metadata_mismatches(saved_abi, current_abi)
    if mismatches:
        details = format_metadata_mismatches(mismatches)
        raise IncompatibleLibraryError(f"Native bundle ABI mismatch: {details}")

    function_check_policies = {}
    for function in manifest["functions"]:
        for specialization in function.get("specializations", []):
            function_check_policies[specialization["symbol"]] = function["do_checks"]
    for symbol, target in manifest["targets"].items():
        target_mismatches = metadata_mismatches(
            target["codegen_settings"],
            saved_abi["codegen_settings"],
            prefix=f"targets.{symbol}.codegen_settings",
        )
        if target_mismatches:
            details = format_metadata_mismatches(target_mismatches)
            raise CorruptLibraryError(f"Native target ABI metadata mismatch: {details}")
        expected_checks = function_check_policies.get(symbol)
        if expected_checks is not None and target["do_checks"] != expected_checks:
            raise CorruptLibraryError(
                f"Native target check policy does not match function for {symbol!r}"
            )

    wrapper_relative = manifest.get("wrapper_library")
    wrapper_cache = manifest.get("wrapper_cache_info")
    if wrapper_relative is not None:
        if not isinstance(wrapper_cache, dict):
            raise CorruptLibraryError("Native bundle wrapper has no cache metadata")
        required_wrapper_fields = {
            "backend",
            "compile_flags",
            "compiler",
            "simd_arch",
            "simd_features",
            "do_checks",
            "function_checks",
            "numpy_version",
        }
        missing_fields = sorted(required_wrapper_fields.difference(wrapper_cache))
        if missing_fields:
            raise CorruptLibraryError(
                "Native wrapper cache metadata is missing: " + ", ".join(missing_fields)
            )
        expected_function_checks = {
            specialization["symbol"]: function["do_checks"]
            for function in manifest["functions"]
            for specialization in function.get("specializations", [])
        }
        if wrapper_cache["function_checks"] != expected_function_checks:
            raise CorruptLibraryError(
                "Native wrapper check policies do not match persisted functions"
            )

        expected_wrapper = PyCExtension(
            name=manifest.get("name"),
            functions=[],
            do_checks=wrapper_cache["do_checks"],
            function_checks=wrapper_cache["function_checks"],
        ).build_cache_info(
            wrapper_cache["compile_flags"],
            backend=wrapper_cache["backend"],
            compiler_identity=wrapper_cache["compiler"],
            simd_arch=wrapper_cache["simd_arch"],
            simd_features=wrapper_cache["simd_features"],
        )
        compatibility_fields = (
            "format_version",
            "numeta_wrapper_abi_version",
            "python_soabi",
            "extension_suffix",
            "python_version",
            "numpy_c_abi",
            "platform",
            "machine",
            "wrapper_name",
            "codegen_settings",
        )
        wrapper_mismatches = metadata_mismatches(
            {key: wrapper_cache.get(key) for key in compatibility_fields},
            {key: expected_wrapper[key] for key in compatibility_fields},
        )
        if wrapper_mismatches:
            details = format_metadata_mismatches(wrapper_mismatches)
            raise IncompatibleLibraryError(f"Native wrapper cache mismatch: {details}")

    required_features = set(manifest.get("required_cpu_features", []))
    missing_features = required_features.difference(_cpu_features())
    if missing_features:
        raise IncompatibleLibraryError(
            "Native bundle requires unavailable CPU features: "
            + ", ".join(sorted(missing_features))
        )

    missing_libraries = [
        library
        for library in manifest.get("required_libraries", [])
        if ctypes.util.find_library(library) is None
    ]
    if missing_libraries:
        raise IncompatibleLibraryError(
            "Native bundle requires unavailable libraries: " + ", ".join(sorted(missing_libraries))
        )


def save_library_bundle(library, directory, compile_flags=None, timing_callback=None) -> Path:
    from .numeta_library import _emit_timing, _timing_phase

    save_start = __import__("time").perf_counter()
    parent = Path(directory).absolute()
    parent.mkdir(parents=True, exist_ok=True)
    if platform.system() != "Linux":
        raise IncompatibleLibraryError("Numeta 0.6 native bundles support Linux only")
    if library.name is None:
        raise ValueError("Library name must be set before saving")
    name = library.name

    with _bundle_lock(parent, name):
        bundle = _recover_bundle(parent, name)
        stage = Path(tempfile.mkdtemp(prefix=f".{name}.numeta.stage.", dir=parent))
        backup = None
        target_snapshots = {}
        try:
            libraries_dir = stage / "libraries"
            libraries_dir.mkdir(parents=True)
            roots = [
                compiled
                for function in library._entries.values()
                for compiled in function._compiled_functions.values()
            ]
            roots.extend(library._global_entries.values())
            active_targets = _active_compiled_targets_by_name(
                library._entries,
                library._global_entries,
            )
            targets = _collect_compiled_target_closure(roots, active_targets)
            for target in targets:
                target.validate_codegen_abi()

            procedures_infos = []
            for function in library._entries.values():
                for signature in function._compiled_functions:
                    function.construct_wrapper_spec(signature)
                procedures_infos.extend(function._wrapper_specs.values())
            from .numeta_function import NumetaFunction

            procedures_infos = NumetaFunction._deduplicate_wrapper_specs(procedures_infos)
            wrapped_functions = [
                function for function in library._entries.values() if function._compiled_functions
            ]
            wrapper_function_checks = {
                compiled.func_name: function.do_checks
                for function in wrapped_functions
                for compiled in function._compiled_functions.values()
            }
            wrapper_check_policies = set(wrapper_function_checks.values())
            wrapper_do_checks = next(
                iter(wrapper_check_policies),
                settings.default_do_checks,
            )
            if len(wrapper_check_policies) > 1:
                wrapper_do_checks = settings.default_do_checks
            resolved_flags = (
                settings.default_compile_flags if compile_flags is None else compile_flags
            )
            resolved_flags = Compiler._normalize_flags(resolved_flags)

            compiler = Compiler(
                settings.c_compiler,
                resolved_flags,
                setting="set_c_compiler",
                env_var="NUMETA_CC",
            )
            compiler_info = {"c": compiler.identity()}
            compiled_backends = {target.backend for target in targets}
            if "fortran" in compiled_backends:
                fortran_compiler = Compiler(
                    settings.fortran_compiler,
                    resolved_flags,
                    setting="set_fortran_compiler",
                    env_var="NUMETA_FC",
                )
                compiler_info["fortran"] = fortran_compiler.identity()

            compiled_artifacts = {}
            object_files = set()
            target_snapshots = {
                target: (
                    target._path,
                    target._rpath,
                    target._include,
                    target._obj_files,
                    list(getattr(target, "_source_files", [])),
                    target.library_name,
                    getattr(target, "_compiler_identity", None),
                )
                for target in targets
            }
            for target in targets:
                with _timing_phase(
                    timing_callback,
                    "save.persist_artifact",
                    symbol=target.func_name,
                ):
                    saved_obj, _saved_src, _saved_include, artifact = _persist_compiled_artifacts(
                        target, stage
                    )
                object_files.add(saved_obj)
                compiled_artifacts[target.func_name] = artifact

            external_payloads = {}
            target_dependencies = {}
            libraries = set()
            libraries_dirs = set()
            rpath_dirs = set()
            include_dirs = set()
            additional_flags = []
            seen_additional_flags = set()
            compiled_requires_math = False

            for target in targets:
                if target.backend == "c" and getattr(target, "_requires_math", False):
                    compiled_requires_math = True
                dependencies = []
                symbolic = getattr(target, "symbolic_function", None)
                values = symbolic.get_dependencies().values() if symbolic is not None else ()
                for dependency in values:
                    if (
                        hasattr(dependency, "func_name")
                        and dependency.func_name in compiled_artifacts
                    ):
                        dependencies.append({"kind": "compiled", "symbol": dependency.func_name})
                        continue
                    external_id = _external_id(dependency)
                    if external_id not in external_payloads:
                        external_id, external_payload, copied_objects = _serialize_external(
                            dependency, stage
                        )
                        external_payloads[external_id] = external_payload
                        object_files.update(copied_objects)
                    dependencies.append({"kind": "external", "id": external_id})
                    external_payload = external_payloads[external_id]
                    include_dirs.update(external_payload["include"])
                    if external_payload["to_link"]:
                        libraries.add(external_payload["library_name"])
                        libraries_dirs.update(external_payload["path"])
                        rpath_dirs.update(external_payload["rpath"])
                    for flag in external_payload["additional_flags"]:
                        if flag not in seen_additional_flags:
                            seen_additional_flags.add(flag)
                            additional_flags.append(flag)
                target_dependencies[target.func_name] = dependencies

            if "fortran" in compiled_backends:
                libraries.update({"gfortran", "m", "mvec"})
            if compiled_requires_math:
                libraries.add("m")

            with _timing_phase(timing_callback, "save.link", objects=len(object_files)):
                core_lib = compiler.compile_to_library(
                    name,
                    object_files,
                    libraries_dir,
                    libraries=libraries,
                    include_dirs=include_dirs,
                    libraries_dirs=libraries_dirs,
                    rpath_dirs=rpath_dirs,
                    additional_flags=additional_flags,
                )

            wrapper = None
            if procedures_infos:
                wrapper = PyCExtension(
                    name=name,
                    functions=procedures_infos,
                    do_checks=wrapper_do_checks,
                    function_checks=wrapper_function_checks,
                )
                wrapper_backend = "fortran" if "fortran" in compiled_backends else "c"
                simd_configurations = {
                    (
                        getattr(target, "simd_arch", settings.default_simd_arch),
                        tuple(
                            getattr(
                                target,
                                "simd_features",
                                settings.default_simd_features,
                            )
                        ),
                    )
                    for target in targets
                }
                if len(simd_configurations) == 1:
                    wrapper_simd_arch, wrapper_simd_features = next(iter(simd_configurations))
                else:
                    wrapper_simd_arch, wrapper_simd_features = None, ()
                with _timing_phase(timing_callback, "save.wrapper", reused=False):
                    wrapper.compile(
                        core_lib_name=name,
                        core_lib_path=libraries_dir,
                        directory=libraries_dir,
                        compile_flags=resolved_flags,
                        backend=wrapper_backend,
                        runtime_rpath="$ORIGIN",
                        simd_arch=wrapper_simd_arch,
                        simd_features=wrapper_simd_features,
                    )

            target_payloads = {}
            for target in targets:
                artifact = compiled_artifacts[target.func_name]
                target_payloads[target.func_name] = {
                    "symbol": target.func_name,
                    "library_name": name,
                    "backend": target.backend,
                    "do_checks": target.do_checks,
                    "compile_flags": list(target.compile_flags),
                    "simd_arch": getattr(target, "simd_arch", settings.default_simd_arch),
                    "simd_features": list(
                        getattr(target, "simd_features", settings.default_simd_features)
                    ),
                    "c_attributes": list(getattr(target, "c_attributes", ())),
                    "c_linkage": getattr(target, "c_linkage", None),
                    "emit_mode": getattr(target, "emit_mode", None),
                    "requires_math": bool(getattr(target, "_requires_math", False)),
                    "compiler": getattr(target, "_compiler_identity", None)
                    or compiler_info[target.backend],
                    "codegen_settings": target._codegen_abi_settings,
                    "arguments": [
                        _serialize_argument(argument)
                        for argument in getattr(target.symbolic_function, "arguments", {}).values()
                    ],
                    "object_files": artifact["object_files"],
                    "source_files": artifact["source_files"],
                    "include_dirs": artifact["include_dirs"],
                    "module_files": artifact["module_files"],
                    "dependencies": target_dependencies[target.func_name],
                }

            has_native_flags = any(
                flag in {"-march=native", "-mcpu=native"}
                for target in targets
                for flag in target.compile_flags
            )
            manifest = {
                "format": BUNDLE_FORMAT,
                "format_version": BUNDLE_FORMAT_VERSION,
                "created_by": {"name": "numeta", "version": NUMETA_VERSION},
                "name": name,
                "abi": _runtime_abi(),
                "required_cpu_features": _cpu_features() if has_native_flags else [],
                "toolchains": compiler_info,
                "required_libraries": sorted(libraries),
                "core_library": core_lib.relative_to(stage).as_posix(),
                "wrapper_library": (
                    Path(wrapper.lib_path).relative_to(stage).as_posix() if wrapper else None
                ),
                "wrapper_cache_info": wrapper.cache_info if wrapper else None,
                "targets": target_payloads,
                "external_dependencies": external_payloads,
                "functions": [
                    _serialize_function(function) for function in library._entries.values()
                ],
                "globals": [
                    _serialize_global(key, target)
                    for key, target in library._global_entries.items()
                ],
            }
            manifest["checksums"] = _checksums(stage)
            manifest_path = stage / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )

            if bundle.exists():
                backup = parent / f".{name}.numeta.backup.{uuid.uuid4().hex}"
                os.replace(bundle, backup)
            try:
                os.replace(stage, bundle)
            except Exception:
                if backup is not None and backup.exists() and not bundle.exists():
                    os.replace(backup, bundle)
                raise
            if backup is not None:
                shutil.rmtree(backup, ignore_errors=True)

            for target in targets:
                payload = target_payloads[target.func_name]
                target._path = bundle / "libraries"
                target._rpath = target._path
                target._include = _resolve_bundle_path(bundle, payload["include_dirs"][0])
                target._obj_files = _resolve_bundle_path(bundle, payload["object_files"][0])
                target._source_files = [
                    _resolve_bundle_path(bundle, path) for path in payload["source_files"]
                ]
                target.library_name = name
                target.compiled = True
            if wrapper is not None:
                wrapper.set_lib_path(_resolve_bundle_path(bundle, manifest["wrapper_library"]))
                for function in library._entries.values():
                    function._library_pyc_extension = wrapper
                    function._pyc_extensions = {}

            _emit_timing(
                timing_callback,
                "save.total",
                __import__("time").perf_counter() - save_start,
            )
            return _resolve_bundle_path(bundle, manifest["core_library"])
        except Exception:
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)
            for target, snapshot in target_snapshots.items():
                (
                    target._path,
                    target._rpath,
                    target._include,
                    target._obj_files,
                    target._source_files,
                    target.library_name,
                    target._compiler_identity,
                ) = snapshot
            raise


def load_library_bundle(
    library_class,
    name: str,
    directory,
    *,
    safe: bool = False,
    ignore_corrupt: bool | None = None,
):
    if safe:
        warnings.warn(
            "safe= is deprecated; JSON bundles never deserialize Python objects. "
            "Use ignore_corrupt= to control cache-miss behavior.",
            DeprecationWarning,
            stacklevel=2,
        )
    if ignore_corrupt is not None and safe and ignore_corrupt != safe:
        raise ValueError("safe and ignore_corrupt specify conflicting values")
    tolerate_corrupt = safe if ignore_corrupt is None else ignore_corrupt
    library_class._nm_validate_name(name)
    parent = Path(directory).absolute()

    with _bundle_lock(parent, name):
        bundle = _recover_bundle(parent, name)
    legacy = parent / f"{name}.pkl"
    if not bundle.exists() and legacy.exists():
        raise LegacyLibraryFormatError(
            f"Legacy pickle library {legacy} is not supported; rebuild it with Numeta 0.6"
        )

    result = library_class(name)
    try:
        manifest_path = bundle / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CorruptLibraryError(
                f"Cannot read bundle manifest {manifest_path}: {exc}"
            ) from exc
        _validate_manifest(manifest, bundle)
        if manifest.get("name") != name:
            raise CorruptLibraryError(
                f"Bundle name {manifest.get('name')!r} does not match requested name {name!r}"
            )

        external_dependencies = {
            external_id: _restore_external(payload, bundle)
            for external_id, payload in manifest["external_dependencies"].items()
        }
        targets = {
            symbol: _restore_compiled_target(payload, bundle)
            for symbol, payload in manifest["targets"].items()
        }
        for symbol, payload in manifest["targets"].items():
            dependencies = {}
            for dependency in payload["dependencies"]:
                if dependency["kind"] == "compiled":
                    value = targets[dependency["symbol"]]
                elif dependency["kind"] == "external":
                    value = external_dependencies[dependency["id"]]
                else:
                    raise CorruptLibraryError(f"Unknown dependency kind {dependency.get('kind')!r}")
                dependencies[getattr(value, "name", str(value))] = value
            targets[symbol].symbolic_function._dependencies = dependencies

        for function_payload in manifest["functions"]:
            function = _restore_function(function_payload, targets, bundle)
            result._entries[function.name] = function
        for global_payload in manifest["globals"]:
            target = _restore_global(global_payload, targets[global_payload["symbol"]])
            result._global_entries[global_payload["key"]] = target

        wrapper_relative = manifest.get("wrapper_library")
        if wrapper_relative is not None:
            wrapper_specs = []
            for function in result._entries.values():
                wrapper_specs.extend(function._wrapper_specs.values())
            from .numeta_function import NumetaFunction

            wrapper = PyCExtension(
                name=name,
                functions=NumetaFunction._deduplicate_wrapper_specs(wrapper_specs),
                do_checks=manifest["wrapper_cache_info"]["do_checks"],
                function_checks=manifest["wrapper_cache_info"]["function_checks"],
            )
            wrapper.cache_info = manifest.get("wrapper_cache_info")
            wrapper.set_lib_path(_resolve_bundle_path(bundle, wrapper_relative))
            for function in result._entries.values():
                function._library_pyc_extension = wrapper

        core_library = _resolve_bundle_path(bundle, manifest["core_library"])
        try:
            ctypes.CDLL(str(core_library), mode=getattr(ctypes, "RTLD_LOCAL", 0))
        except OSError as exc:
            raise IncompatibleLibraryError(
                f"Native bundle dependencies could not be loaded from {core_library}: {exc}"
            ) from exc
    except (CorruptLibraryError, IncompatibleLibraryError):
        if not tolerate_corrupt:
            raise
        warnings.warn(
            f"Failed to load NumetaLibrary {name!r}; treating it as a cache miss.",
            RuntimeWarning,
            stacklevel=2,
        )
        result = library_class(name)
    except (AttributeError, IndexError, KeyError, OverflowError, TypeError, ValueError) as exc:
        error = CorruptLibraryError(f"Invalid bundle manifest for {name!r}: {exc}")
        if not tolerate_corrupt:
            raise error from exc
        warnings.warn(
            f"Failed to load NumetaLibrary {name!r}; treating it as a cache miss.",
            RuntimeWarning,
            stacklevel=2,
        )
        result = library_class(name)

    loaded_names = {
        compiled.func_name
        for function in result._entries.values()
        for compiled in function._compiled_functions.values()
    }
    loaded_names.update(compiled.func_name for compiled in result._global_entries.values())
    if loaded_names:
        from .native_name_registry import native_name_registry

        native_name_registry.reserve_many(loaded_names)
    library_class.loaded.add(name)
    return result
