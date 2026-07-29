"""Stable, JSON-safe identifiers for compiled Numeta signatures."""

import hashlib
import json

import numpy as np

from .datatype import DataTypeMeta

SIGNATURE_ID_VERSION = 1


def signature_json_value(value):
    """Convert a supported signature value to a deterministic JSON value.

    Unsupported values are rejected rather than hashed from ``repr(value)``:
    default object representations commonly contain process-specific memory
    addresses and would violate the stability guarantee of signature IDs.
    """
    if isinstance(value, (tuple, list)):
        return [signature_json_value(item) for item in value]
    if isinstance(value, dict):
        items = [
            [signature_json_value(key), signature_json_value(item)] for key, item in value.items()
        ]
        return {
            "kind": "mapping",
            "items": sorted(
                items,
                key=lambda pair: json.dumps(
                    pair[0],
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            ),
        }
    if isinstance(value, np.dtype):
        return {
            "kind": "numpy.dtype",
            "str": value.str,
            "descr": value.descr,
            "itemsize": value.itemsize,
            "isalignedstruct": bool(value.isalignedstruct),
        }
    if isinstance(value, np.generic):
        return {
            "kind": "numpy.scalar",
            "dtype": signature_json_value(value.dtype),
            "value": signature_json_value(value.item()),
        }
    if isinstance(value, DataTypeMeta):
        return {"kind": "numeta.datatype", "name": value._name}
    if isinstance(value, type):
        if value in (bool, int, float, complex, str):
            return {
                "kind": "python.type",
                "module": value.__module__,
                "qualname": value.__qualname__,
            }
        try:
            dtype = np.dtype(value)
        except TypeError:
            dtype = None
        if dtype is not None:
            return {"kind": "numpy.scalar_type", "dtype": signature_json_value(dtype)}
        return {
            "kind": "python.type",
            "module": value.__module__,
            "qualname": value.__qualname__,
        }
    if type(value).__name__ == "LiteralNode":
        return signature_json_value(value.value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, complex):
        return {"kind": "python.complex", "real": value.real, "imag": value.imag}
    value_type = f"{type(value).__module__}.{type(value).__qualname__}"
    raise TypeError(
        f"Cannot create a stable signature ID for unsupported value of type {value_type}"
    )


def signature_to_jsonable(signature) -> dict:
    return {
        "version": SIGNATURE_ID_VERSION,
        "signature": signature_json_value(signature),
    }


def signature_id(signature) -> str:
    payload = json.dumps(
        signature_to_jsonable(signature),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sig-v{SIGNATURE_ID_VERSION}-{digest}"
