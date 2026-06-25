import numpy as np
from dataclasses import dataclass
from .fortran.fortran_type import FortranType
from .array_shape import ArrayShape, SCALAR, UNKNOWN


class DataTypeMeta(type):
    """Metaclass used for all data type classes."""

    _np_dtype = {}
    _ftype = {}
    _ftype_bind_c = {}

    def __new__(mcls, name, bases, attrs):
        cls = super().__new__(mcls, name, bases, attrs)

        # Register built-in numpy mapping when available
        np_type = attrs.get("_np_type")
        if np_type is not None:
            if np_type in DataTypeMeta._np_dtype:
                raise ValueError(f"DataType {np_type} already exists")
            DataTypeMeta._np_dtype[np_type] = cls

        ftype = attrs.get("_fortran_type")
        if ftype is not None:
            if ftype in DataTypeMeta._ftype:
                raise ValueError(f"FortranType {ftype} already exists")
            DataTypeMeta._ftype[ftype] = cls

        fortran_bind_c_type = attrs.get("_fortran_bind_c_type")
        if fortran_bind_c_type is not None:
            if fortran_bind_c_type in DataTypeMeta._ftype_bind_c:
                raise ValueError(f"FortranType {fortran_bind_c_type} already exists")
            DataTypeMeta._ftype_bind_c[fortran_bind_c_type] = cls

        return cls

    def __repr__(cls):
        return f"numeta.{cls._name}"

    def __call__(cls, *args, **kwargs):
        # StructType overrides this behaviour and must be instantiated normally
        value = args[0] if args else kwargs.get("value", None)
        name = kwargs.get("name", None)
        from .wrappers.scalar import scalar

        return scalar(cls, value, name=name)


class DataType(metaclass=DataTypeMeta):
    """Base class for all data type definitions."""

    _np_type = None
    _fortran_type = None
    _fortran_bind_c_type = None
    _cnp_type = None
    _capi_cast = staticmethod(lambda x: x)
    _name = "datatype"
    _is_struct = False
    _can_be_value = True

    @property
    def name(cls):
        return cls._name

    @classmethod
    def __class_getitem__(cls, key):
        if key is None:
            # It is a pointer
            return ArrayType(dtype=cls, shape=UNKNOWN)
        if not isinstance(key, tuple):
            key = (key,)
        new_key = []
        for k in key:
            if isinstance(k, slice):
                if k.start is not None or k.stop is not None or k.step is not None:
                    raise TypeError(f"Invalid type for array dimension: {type(k)}")
            elif not isinstance(k, int):
                raise TypeError(f"Invalid type for array dimension: {type(k)}")
            if isinstance(k, int) and k < 0:
                raise ValueError("Negative dimensions are not allowed")
            new_key.append(k if isinstance(k, int) else None)
        return ArrayType(dtype=cls, shape=ArrayShape(tuple(new_key)))

    @classmethod
    def is_np_dtype(cls, dtype):
        return dtype in DataTypeMeta._np_dtype

    @classmethod
    def from_np_dtype(cls, dtype):
        """Get the DataType class from a numpy dtype."""
        return DataTypeMeta._np_dtype[dtype]

    @classmethod
    def from_ftype(cls, ftype):
        """Get the DataType class from a FortranType."""
        from numeta.ast.types import Type as AstType

        if isinstance(ftype, AstType):
            ftype = FortranType(ftype.type, ftype.kind)
        if ftype in DataTypeMeta._ftype_bind_c:
            return DataTypeMeta._ftype_bind_c[ftype]
        return DataTypeMeta._ftype[ftype]

    @classmethod
    def is_struct(cls):
        return cls._is_struct

    @classmethod
    def can_be_value(cls):
        return cls._can_be_value

    @classmethod
    def get_numpy(cls):
        return cls._np_type

    @classmethod
    def get_cnumpy(cls):
        return cls._cnp_type

    @classmethod
    def get_fortran(cls, bind_c=None):
        if bind_c is None:
            from .settings import settings

            bind_c = settings.iso_C

        if bind_c:
            # Use lazy getter for bind_c types
            if hasattr(cls, "_get_bind_c_type"):
                ftype = cls._get_bind_c_type()
            else:
                ftype = cls._fortran_bind_c_type
            if ftype is not None and ftype not in DataTypeMeta._ftype_bind_c:
                DataTypeMeta._ftype_bind_c[ftype] = cls
            return ftype
        return cls._fortran_type

    @classmethod
    def get_capi_cast(cls, obj):
        return cls._capi_cast(obj)

    @classmethod
    def get_nbytes(cls):
        """Get the number of bytes used by this data type."""
        if cls._np_type is not None:
            return np.dtype(cls._np_type).itemsize
        raise NotImplementedError(f"DataType {cls._name} does not have a defined size")


_vector_type_cache = {}


def _short_c_name(dtype):
    if dtype is float64:
        return "f64"
    if dtype is float32:
        return "f32"
    if dtype is int64:
        return "i64"
    if dtype is int32:
        return "i32"
    return getattr(dtype, "_name", str(dtype)).replace("int", "i").replace("float", "f")


class VectorType(DataType):
    """Scalar-like logical SIMD vector dtype."""

    _np_type = None
    _fortran_type = None
    _fortran_bind_c_type = None
    _cnp_type = None
    _name = "vector"
    _is_vector = True

    @classmethod
    def __class_getitem__(cls, key):
        if key is None:
            return ArrayType(dtype=cls, shape=UNKNOWN)
        if not isinstance(key, tuple):
            key = (key,)
        new_key = []
        for k in key:
            if isinstance(k, slice):
                if k.start is not None or k.stop is not None or k.step is not None:
                    raise TypeError(f"Invalid type for array dimension: {type(k)}")
            elif not isinstance(k, int):
                raise TypeError(f"Invalid type for array dimension: {type(k)}")
            if isinstance(k, int) and k < 0:
                raise ValueError("Negative dimensions are not allowed")
            new_key.append(k if isinstance(k, int) else None)
        return ArrayType(dtype=cls, shape=ArrayShape(tuple(new_key)))

    @classmethod
    def get_numpy(cls):
        return None

    @classmethod
    def get_fortran(cls, bind_c=None):
        raise NotImplementedError("VectorType is only supported by the C backend")

    @classmethod
    def get_nbytes(cls):
        return cls._base_dtype.get_nbytes() * cls._lanes

    @classmethod
    def base_dtype(cls):
        return cls._base_dtype

    @classmethod
    def lanes(cls):
        return cls._lanes


def make_vector_type(base_dtype, lanes: int):
    base_dtype = get_datatype(base_dtype)
    lanes = int(lanes)
    if lanes <= 0:
        raise ValueError("Vector lanes must be positive")
    if base_dtype in {complex64, complex128, complex256, char, bool8, c_ptr}:
        raise NotImplementedError(
            "Complex, char, bool, and pointer SIMD vectors are not supported yet"
        )

    key = (base_dtype, lanes)
    if key in _vector_type_cache:
        return _vector_type_cache[key]

    short_name = _short_c_name(base_dtype)
    name = f"vec_{short_name}_{lanes}"
    c_name = f"nm_vec_{short_name}_{lanes}"
    cls = type(
        name,
        (VectorType,),
        {
            "_name": name,
            "name": name,
            "_base_dtype": base_dtype,
            "_lanes": lanes,
            "_cnp_type": c_name,
        },
    )
    _vector_type_cache[key] = cls
    return cls


class Vector:
    def __class_getitem__(cls, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("Vector requires Vector[base_dtype, lanes]")
        base_dtype, lanes = key
        return make_vector_type(base_dtype, lanes)


@dataclass(frozen=True)
class ArrayType:
    """Helper object returned by DataType[x] to describe array types."""

    dtype: DataType
    shape: ArrayShape

    @classmethod
    def __class_getitem__(cls, key):
        """
        Create an ArrayType with a given order and shape.
        If key is 'C' is C-style order, if 'F' is Fortran-style order.
        """
        if key != "C" and key != "F":
            raise ValueError(f"Invalid order: {key}, must be 'C' or 'F'")
        return cls(
            dtype=cls.dtype, shape=ArrayShape(cls.shape.as_tuple(), fortran_order=(key == "F"))
        )

    def __repr__(self):
        if self.shape.is_unknown:
            return f"{self.dtype._name}[*]"
        dims = ",".join(
            ":" if isinstance(d, slice) and d == slice(None) else str(d)
            for d in self.shape.as_tuple()
        )
        return f"{self.dtype._name}[{dims}]"

    def __call__(self, *args, **kwargs):
        value = args[0] if args else kwargs.get("value", None)
        name = kwargs.get("name", None)

        from .wrappers.empty import empty

        array = empty(self.shape, dtype=self.dtype, order=kwargs.get("order", "C"), name=name)
        if value is not None:
            array[:] = value

        return array


class int32(DataType):
    _np_type = np.int32
    _fortran_type = FortranType("integer", 4)
    _fortran_bind_c_type = None  # Set lazily
    _cnp_type = "npy_int32"
    _capi_cast = staticmethod(lambda x: f"PyLong_AsLongLong({x})")
    _name = "int32"

    @classmethod
    def _get_bind_c_type(cls):
        if cls._fortran_bind_c_type is None:
            from .fortran.external_modules.iso_c_binding import iso_c

            cls._fortran_bind_c_type = FortranType("integer", iso_c.c_int32)
        return cls._fortran_bind_c_type


class int64(DataType):
    _np_type = np.int64
    _fortran_type = FortranType("integer", 8)
    _fortran_bind_c_type = None  # Set lazily
    _cnp_type = "npy_int64"
    _capi_cast = staticmethod(lambda x: f"PyLong_AsLongLong({x})")
    _name = "int64"

    @classmethod
    def _get_bind_c_type(cls):
        if cls._fortran_bind_c_type is None:
            from .fortran.external_modules.iso_c_binding import iso_c

            cls._fortran_bind_c_type = FortranType("integer", iso_c.c_int64)
        return cls._fortran_bind_c_type


class size_t(DataType):
    _np_type = None
    _fortran_type = None
    _fortran_bind_c_type = None  # Set lazily
    _cnp_type = "npy_intp"
    _capi_cast = staticmethod(lambda x: f"PyLong_AsLongLong({x})")
    _name = "size_t"

    @classmethod
    def _get_bind_c_type(cls):
        if cls._fortran_bind_c_type is None:
            from .fortran.external_modules.iso_c_binding import iso_c

            cls._fortran_bind_c_type = FortranType("integer", iso_c.c_size_t)
        return cls._fortran_bind_c_type


class c_ptr(DataType):
    _np_type = None
    _fortran_type = None
    _fortran_bind_c_type = None  # Set lazily
    _cnp_type = "void*"
    _capi_cast = staticmethod(lambda x: f"PyLong_AsVoidPtr({x})")
    _name = "c_ptr"

    @classmethod
    def _get_bind_c_type(cls):
        if cls._fortran_bind_c_type is None:
            from .fortran.external_modules.iso_c_binding import iso_c

            cls._fortran_bind_c_type = FortranType("type", iso_c.c_ptr)
        return cls._fortran_bind_c_type


class float32(DataType):
    _np_type = np.float32
    _fortran_type = FortranType("real", 4)
    _fortran_bind_c_type = None  # Set lazily
    _cnp_type = "npy_float32"
    _capi_cast = staticmethod(lambda x: f"PyFloat_AsDouble({x})")
    _name = "float32"

    @classmethod
    def _get_bind_c_type(cls):
        if cls._fortran_bind_c_type is None:
            from .fortran.external_modules.iso_c_binding import iso_c

            cls._fortran_bind_c_type = FortranType("real", iso_c.c_float)
        return cls._fortran_bind_c_type


class float64(DataType):
    _np_type = np.float64
    _fortran_type = FortranType("real", 8)
    _fortran_bind_c_type = None  # Set lazily
    _cnp_type = "npy_float64"
    _capi_cast = staticmethod(lambda x: f"PyFloat_AsDouble({x})")
    _name = "float64"

    @classmethod
    def _get_bind_c_type(cls):
        if cls._fortran_bind_c_type is None:
            from .fortran.external_modules.iso_c_binding import iso_c

            cls._fortran_bind_c_type = FortranType("real", iso_c.c_double)
        return cls._fortran_bind_c_type


class float128(DataType):
    _np_type = np.longdouble
    _fortran_type = FortranType("real", 16)
    _fortran_bind_c_type = None  # Set lazily
    _cnp_type = "npy_longdouble"
    _capi_cast = staticmethod(lambda x: f"PyFloat_AsDouble({x})")
    _name = "float128"

    @classmethod
    def _get_bind_c_type(cls):
        if cls._fortran_bind_c_type is None:
            from .fortran.external_modules.iso_c_binding import iso_c

            cls._fortran_bind_c_type = FortranType("real", iso_c.c_long_double)
        return cls._fortran_bind_c_type


class complex64(DataType):
    _np_type = np.complex64
    _fortran_type = FortranType("complex", 4)
    _fortran_bind_c_type = None  # Set lazily
    _cnp_type = "npy_complex64"
    _capi_cast = staticmethod(
        lambda x: f"CMPLX(PyComplex_RealAsDouble({x}), PyComplex_ImagAsDouble({x}))",
    )
    _name = "complex64"

    @classmethod
    def _get_bind_c_type(cls):
        if cls._fortran_bind_c_type is None:
            from .fortran.external_modules.iso_c_binding import iso_c

            cls._fortran_bind_c_type = FortranType("complex", iso_c.c_float_complex)
        return cls._fortran_bind_c_type


class complex128(DataType):
    _np_type = np.complex128
    _fortran_type = FortranType("complex", 8)
    _fortran_bind_c_type = None  # Set lazily
    _cnp_type = "npy_complex128"
    _capi_cast = staticmethod(
        lambda x: f"CMPLX(PyComplex_RealAsDouble({x}), PyComplex_ImagAsDouble({x}))"
    )
    _name = "complex128"

    @classmethod
    def _get_bind_c_type(cls):
        if cls._fortran_bind_c_type is None:
            from .fortran.external_modules.iso_c_binding import iso_c

            cls._fortran_bind_c_type = FortranType("complex", iso_c.c_double_complex)
        return cls._fortran_bind_c_type


class complex256(DataType):
    _np_type = np.clongdouble
    _fortran_type = FortranType("complex", 16)
    _fortran_bind_c_type = None  # Set lazily
    _cnp_type = "npy_clongdouble"
    _capi_cast = staticmethod(
        lambda x: f"CMPLXL(PyComplex_RealAsDouble({x}), PyComplex_ImagAsDouble({x}))"
    )
    _name = "complex256"

    @classmethod
    def _get_bind_c_type(cls):
        if cls._fortran_bind_c_type is None:
            from .fortran.external_modules.iso_c_binding import iso_c

            cls._fortran_bind_c_type = FortranType("complex", iso_c.c_long_double_complex)
        return cls._fortran_bind_c_type


class bool8(DataType):
    _np_type = np.bool_
    _fortran_type = FortranType("logical", 1)
    _fortran_bind_c_type = None  # Set lazily
    _cnp_type = "npy_bool"
    _capi_cast = staticmethod(lambda x: f"PyObject_IsTrue({x})")
    _name = "bool8"

    @classmethod
    def _get_bind_c_type(cls):
        if cls._fortran_bind_c_type is None:
            from .fortran.external_modules.iso_c_binding import iso_c

            cls._fortran_bind_c_type = FortranType("logical", iso_c.c_bool)
        return cls._fortran_bind_c_type


class char(DataType):
    _np_type = np.str_
    _fortran_type = FortranType("character", 1)
    _fortran_bind_c_type = None  # Set lazily
    _cnp_type = "npy_str"
    _capi_cast = staticmethod(lambda x: f"PyUnicode_AsUTF8({x})")
    _name = "char"

    @classmethod
    def _get_bind_c_type(cls):
        if cls._fortran_bind_c_type is None:
            from .fortran.external_modules.iso_c_binding import iso_c

            cls._fortran_bind_c_type = FortranType("character", iso_c.c_char)
        return cls._fortran_bind_c_type


class StructType(DataType, metaclass=DataTypeMeta):
    """Metaclass used to build struct datatype classes."""

    _counter = 0
    _fortran_type = None
    _fortran_bind_c_type = None
    _np_type = None
    _cnp_type = None
    _capi_cast = staticmethod(lambda x: x)
    _name = "datatype"
    _is_struct = False
    _can_be_value = True
    _members = []

    @classmethod
    def c_declaration(cls):
        members_str = []
        for mname, dt, shape in cls._members:
            dec = f"{dt.get_cnumpy()} {mname}"
            if not shape.is_scalar and not shape.is_unknown:
                dec += "".join(f"[{d}]" for d in shape.as_tuple())
            members_str.append(dec)
        members_join = "; ".join(members_str)
        return f"typedef struct {{ {members_join} ;}} {cls._cnp_type};\n"

    @classmethod
    def __class_getitem__(cls, key) -> ArrayType:
        if key is None:
            # It is a pointer
            return ArrayType(dtype=cls, shape=UNKNOWN)
        if not isinstance(key, tuple):
            key = (key,)
        return ArrayType(dtype=cls, shape=ArrayShape(key))


def make_struct_type(np_dtype, members, name=None):
    """Create (or retrieve) a struct datatype class for ``members``."""

    key = tuple(members)
    if key in DataTypeMeta._np_dtype:
        return DataTypeMeta._np_dtype[key]

    if name is None:
        name = f"struct{StructType._counter}"

    StructType._counter += 1

    from .ast import StructType as AstStructType

    fortran_type = FortranType(
        "type",
        AstStructType(
            name,
            members,
        ),
    )

    attrs = {
        "_name": name,
        "name": name,
        "_members": members,
        "members": members,
        "_np_type": np_dtype,
        "_cnp_type": name,
        "_fortran_type": fortran_type,
        "_fortran_bind_c_type": fortran_type,
        "_is_struct": True,
        "_can_be_value": False,
    }

    new_cls = type(name, (StructType,), attrs)

    return new_cls


def get_struct_from_np_dtype(np_dtype):
    if np_dtype in DataTypeMeta._np_dtype:
        return DataTypeMeta._np_dtype[np_dtype]

    fields = []
    for name, (np_d, _) in np_dtype.base.fields.items():
        if np_d in DataTypeMeta._np_dtype:
            dtype = DataTypeMeta._np_dtype[np_d]
        elif np_d.base.type in DataTypeMeta._np_dtype:
            dtype = DataTypeMeta._np_dtype[np_d.base.type]
        elif np_d.fields is not None or (np_d.base.fields is not None):
            dtype = get_struct_from_np_dtype(np_d)
        else:
            raise ValueError(f"Invalid dtype {np_d.base.type}, {np_d.fields}")

        shape = SCALAR if len(np_d.shape) == 0 else ArrayShape(np_d.shape)
        fields.append((name, dtype, shape))

    struct_cls = make_struct_type(np_dtype, fields)
    DataTypeMeta._np_dtype[np_dtype] = struct_cls
    return struct_cls


def get_datatype(dtype):
    #
    # Numeta DataType
    #
    if isinstance(dtype, type) and issubclass(dtype, DataType):
        return dtype
    if isinstance(dtype, FortranType):
        return DataType.from_ftype(dtype)
    #
    # Python numeric types
    #
    if dtype is int:
        return int64
    elif dtype is float:
        return float64
    elif dtype is complex:
        return complex128
    elif dtype is bool:
        return bool8

    #
    # Numpy dtypes
    #

    # Canonalize dtype
    if isinstance(dtype, np.dtype):
        base = dtype.base.type
    else:
        base = getattr(dtype, "base", dtype).type if hasattr(dtype, "type") else dtype
        if isinstance(base, np.dtype):
            base = base.type

    # Check if it is a numpy dtype
    if DataType.is_np_dtype(base):
        return DataType.from_np_dtype(base)

    # It could be a struct
    if hasattr(dtype, "fields"):
        return get_struct_from_np_dtype(dtype)

    from numeta.ast.types import Type as AstType

    if isinstance(dtype, AstType):
        return DataType.from_ftype(dtype)

    raise ValueError(f"Invalid dtype {dtype}")
