import shlex


class SyntaxSettings:
    def __init__(
        self,
        c_like: bool = False,
        default_int=None,
        default_float=None,
        default_complex=None,
        default_bool=None,
        default_char=None,
        order: str = "F",
    ):
        self.__procedure_bind_c = False
        self.__struct_type_bind_c = False
        self.__array_lower_bound = 1
        self.__c_like_bounds = False
        self.__force_value = False
        self.order = order
        self.c_like = c_like

        if self.c_like:
            self.set_c_like()
        else:
            self.unset_c_like()

        self.DEFAULT_INT = None
        self.DEFAULT_INTEGER = None
        self.DEFAULT_FLOAT = None
        self.DEFAULT_REAL = None
        self.DEFAULT_COMPLEX = None
        self.DEFAULT_BOOL = None
        self.DEFAULT_LOGICAL = None
        self.DEFAULT_CHAR = None
        self.DEFAULT_CHARACTER = None
        if default_int is not None:
            self.set_default_datatype(default_int)
        if default_float is not None:
            self.set_default_datatype(default_float)
        if default_complex is not None:
            self.set_default_datatype(default_complex)
        if default_bool is not None:
            self.set_default_datatype(default_bool)
        if default_char is not None:
            self.set_default_datatype(default_char)

    def set_c_like(self):
        self.c_like = True
        self.set_array_lower_bound(0)
        self.set_procedure_bind_c()
        self.set_struct_type_bind_c()
        self.set_force_value()
        self.set_c_like_bounds()
        self.order = "C"

    def unset_c_like(self):
        self.c_like = False
        self.set_array_lower_bound(1)
        self.unset_procedure_bind_c()
        self.unset_struct_type_bind_c()
        self.unset_force_value()
        self.unset_c_like_bounds()
        self.order = "F"

    def set_array_order(self, order: str):
        if order == "C":
            self.order = "C"
        elif order == "F":
            self.order = "F"
        else:
            raise ValueError(f"Order {order} not supported")

    def set_default_datatype(self, dtype):
        from numeta.datatype import (
            DataType,
            int32,
            int64,
            float32,
            float64,
            float128,
            complex64,
            complex128,
            complex256,
            bool8,
            char,
        )

        if not (isinstance(dtype, type) and issubclass(dtype, DataType)):
            raise TypeError("dtype must be a DataType subclass")

        if dtype in (int32, int64):
            self.DEFAULT_INT = dtype
            self.DEFAULT_INTEGER = self.DEFAULT_INT
        elif dtype in (float32, float64, float128):
            self.DEFAULT_FLOAT = dtype
            self.DEFAULT_REAL = self.DEFAULT_FLOAT
        elif dtype in (complex64, complex128, complex256):
            self.DEFAULT_COMPLEX = dtype
        elif dtype == bool8:
            self.DEFAULT_BOOL = dtype
            self.DEFAULT_LOGICAL = self.DEFAULT_BOOL
        elif dtype == char:
            self.DEFAULT_CHAR = dtype
            self.DEFAULT_CHARACTER = self.DEFAULT_CHAR
        else:
            raise NotImplementedError(f"Unsupported DataType {dtype}")

    @property
    def procedure_bind_c(self):
        return self.__procedure_bind_c

    def set_procedure_bind_c(self):
        self.__procedure_bind_c = True

    def unset_procedure_bind_c(self):
        self.__procedure_bind_c = False

    @property
    def struct_type_bind_c(self):
        return self.__struct_type_bind_c

    def set_struct_type_bind_c(self):
        self.__struct_type_bind_c = True

    def unset_struct_type_bind_c(self):
        self.__struct_type_bind_c = False

    @property
    def array_lower_bound(self):
        return self.__array_lower_bound

    def set_array_lower_bound(self, value):
        try:
            self.__array_lower_bound = int(value)
        except ValueError as e:
            raise ValueError(f"Array lower bound must be an integer, got {value}") from e

    @property
    def c_like_bounds(self):
        return self.__c_like_bounds

    def set_c_like_bounds(self):
        self.__c_like_bounds = True

    def unset_c_like_bounds(self):
        self.__c_like_bounds = False

    @property
    def force_value(self):
        return self.__force_value

    def set_force_value(self):
        self.__force_value = True

    def unset_force_value(self):
        self.__force_value = False


class Settings:

    def __init__(
        self,
        iso_C,
        use_numpy_allocator=True,
        reorder_kwargs=True,
        add_shape_descriptors=True,
        ignore_fixed_shape_in_nested_calls=False,
        default_backend="fortran",
        default_do_checks=True,
        track_source_location=True,
        default_compile_flags="-O3 -march=native",
        default_simd_arch="scalar",
        default_simd_features=(),
        use_c_dispatch=True,
        use_c_signature_parser=True,
        c_temporary_allocation="heap",
        c_stack_temporary_max_bytes=32768,
        c_allow_vla_temporaries=False,
    ):
        """Initialize the settings.
        Parameters
        ----------
        iso_C : bool
            Whether to use ISO C compatibility mode.
        use_numpy_allocator : bool
            Whether to use the NumPy memory allocator.
        reorder_kwargs : bool
            If True, keyword arguments in the generated functions are reordered to ensure
            a unique and deterministic function signature, independent of the order in
            which keywords are passed. This is helpful for reproducibility but may be
            inconvenient when using numeta as a code generator.
        add_shape_descriptors : bool
            If True, shape descriptors are added to array arguments in the generated
            functions. Shape descriptors encode the dimensions of arrays, which is
            typically required when using JIT compilation. However, this can be
            undesirable if numeta is used purely as a code generator.
        ignore_fixed_shape_in_nested_calls : bool
            If True, instead of passing array in a nested call as fixed shape,
            they are passed with undefined dimensions.
            This can help limiting the number of generated functions if they have no dependence on the fixed shape.
        use_c_dispatch : bool
            If True (default), use the optimized C extension for argument parsing and dispatch
            if available. If False, force the use of the pure Python implementation.
        use_c_signature_parser : bool
            If True (default), use the C implementation in ``numeta.signature`` when available.
            If False, force ``numeta.signature`` helpers to use pure Python parsing logic.
        c_temporary_allocation : str
            Allocation policy for C-emitter internal temporaries: ``heap``, ``stack``, or ``auto``.
        c_stack_temporary_max_bytes : int
            Maximum compile-time-known temporary size for stack allocation when policy is ``auto``.
        c_allow_vla_temporaries : bool
            If True, runtime-sized internal C temporaries may be emitted as variable-length arrays.
        """
        self.iso_C = iso_C

        # Shared syntax settings for all backends.
        # If iso_C, start with C-like settings, otherwise Fortran-like.
        if iso_C:
            self.syntax = SyntaxSettings(c_like=True, order="C")
        else:
            self.syntax = SyntaxSettings(c_like=False, order="F")

        # Track current backend
        self.__default_backend = default_backend

        self.use_numpy_allocator = use_numpy_allocator
        self.__reorder_kwargs = reorder_kwargs
        self.__add_shape_descriptors = add_shape_descriptors
        self.__ignore_fixed_shape_in_nested_calls = ignore_fixed_shape_in_nested_calls
        self.__defaults_initialized = False
        self.set_default_backend(default_backend)
        self.set_default_do_checks(default_do_checks)
        self.set_track_source_location(track_source_location)
        self.set_default_compile_flags(default_compile_flags)
        self.set_default_simd_arch(default_simd_arch)
        self.set_default_simd_features(default_simd_features)
        self.use_c_dispatch = use_c_dispatch
        self.use_c_signature_parser = use_c_signature_parser
        self.c_temporary_allocation = c_temporary_allocation
        self.c_stack_temporary_max_bytes = c_stack_temporary_max_bytes
        self.c_allow_vla_temporaries = c_allow_vla_temporaries

    @staticmethod
    def _normalize_compile_flags(compile_flags):
        if isinstance(compile_flags, str):
            return tuple(shlex.split(compile_flags))
        return tuple(compile_flags)

    def set_default_from_datatype(self, dtype, *, iso_c: bool = False):
        """Set the default Fortran type using a :class:`DataType` subclass."""
        from .datatype import DataType

        if not isinstance(dtype, type) or not issubclass(dtype, DataType):
            raise TypeError("dtype must be a DataType subclass")

        self.syntax.set_default_datatype(dtype)

    def set_iso_C(self):
        """Set the ISO C compatibility mode."""
        self.iso_C = True
        self.syntax.set_c_like()
        self._setup_iso_c_defaults()
        self.__defaults_initialized = True

    def unset_iso_C(self):
        """Unset the ISO C compatibility mode."""
        self.iso_C = False
        self.syntax.unset_c_like()
        self._setup_fortran_defaults()
        self.__defaults_initialized = True

    def set_numpy_allocator(self):
        """Set whether to use the NumPy memory allocator."""
        self.use_numpy_allocator = True

    def unset_numpy_allocator(self):
        """Unset the NumPy memory allocator."""
        self.use_numpy_allocator = False

    @property
    def reorder_kwargs(self):
        """Return whether to reorder keyword arguments in the generated function."""
        return self.__reorder_kwargs

    def set_reorder_kwargs(self):
        """Set whether to reorder keyword arguments in the generated function."""
        self.__reorder_kwargs = True

    def unset_reorder_kwargs(self):
        """Unset the reordering of keyword arguments in the generated function."""
        self.__reorder_kwargs = False

    @property
    def add_shape_descriptors(self):
        """Return whether to add shape descriptors to array arguments in generated functions."""
        return self.__add_shape_descriptors

    def set_add_shape_descriptors(self):
        """Set whether to add shape descriptors to array arguments in generated functions."""
        self.__add_shape_descriptors = True

    def unset_add_shape_descriptors(self):
        """Unset the addition of shape descriptors to array arguments in generated functions."""
        self.__add_shape_descriptors = False

    @property
    def ignore_fixed_shape_in_nested_calls(self):
        return self.__ignore_fixed_shape_in_nested_calls

    @ignore_fixed_shape_in_nested_calls.setter
    def ignore_fixed_shape_in_nested_calls(self, value):
        self.__ignore_fixed_shape_in_nested_calls = value

    @property
    def default_backend(self):
        return self.__default_backend

    def set_default_backend(self, backend: str):
        if not isinstance(backend, str):
            raise TypeError("backend must be a string")
        backend = backend.lower()
        if backend not in {"fortran", "c"}:
            raise ValueError("backend must be 'fortran' or 'c'")
        self.__default_backend = backend

    @property
    def default_do_checks(self):
        return self.__default_do_checks

    def set_default_do_checks(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("default_do_checks must be a bool")
        self.__default_do_checks = value

    @property
    def track_source_location(self):
        return self.__track_source_location

    def set_track_source_location(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("track_source_location must be a bool")
        self.__track_source_location = value
        from numeta.ast.nodes.base_node import set_source_location_tracking

        set_source_location_tracking(value)

    def set_source_location_tracking(self):
        self.set_track_source_location(True)

    def unset_source_location_tracking(self):
        self.set_track_source_location(False)

    @property
    def default_compile_flags(self):
        return self.__default_compile_flags

    def set_default_compile_flags(self, compile_flags):
        if compile_flags is None:
            raise TypeError("compile_flags cannot be None")
        normalized = self._normalize_compile_flags(compile_flags)
        self.__default_compile_flags = normalized

    @property
    def default_simd_arch(self):
        return self.__default_simd_arch

    def set_default_simd_arch(self, arch: str):
        if not isinstance(arch, str):
            raise TypeError("simd_arch must be a string")
        arch = arch.lower()
        if arch not in {"scalar", "sse2", "avx", "avx2", "avx512f", "neon", "native"}:
            raise ValueError(
                "simd_arch must be one of: scalar, sse2, avx, avx2, avx512f, neon, native"
            )
        self.__default_simd_arch = arch

    @property
    def default_simd_features(self):
        return self.__default_simd_features

    def set_default_simd_features(self, features):
        if features is None:
            features = ()
        if isinstance(features, str):
            features = (features,)
        self.__default_simd_features = tuple(str(feature).lower() for feature in features)

    @property
    def use_c_dispatch(self):
        return self.__use_c_dispatch

    @use_c_dispatch.setter
    def use_c_dispatch(self, value):
        if not isinstance(value, bool):
            raise TypeError("use_c_dispatch must be a bool")
        self.__use_c_dispatch = value

    @property
    def use_c_signature_parser(self):
        return self.__use_c_signature_parser

    @use_c_signature_parser.setter
    def use_c_signature_parser(self, value):
        if not isinstance(value, bool):
            raise TypeError("use_c_signature_parser must be a bool")
        self.__use_c_signature_parser = value

    @property
    def c_temporary_allocation(self):
        return self.__c_temporary_allocation

    @c_temporary_allocation.setter
    def c_temporary_allocation(self, value):
        if not isinstance(value, str):
            raise TypeError("c_temporary_allocation must be a string")
        value = value.lower()
        if value not in {"heap", "stack", "auto"}:
            raise ValueError("c_temporary_allocation must be 'heap', 'stack', or 'auto'")
        self.__c_temporary_allocation = value

    @property
    def c_stack_temporary_max_bytes(self):
        return self.__c_stack_temporary_max_bytes

    @c_stack_temporary_max_bytes.setter
    def c_stack_temporary_max_bytes(self, value):
        if not isinstance(value, int):
            raise TypeError("c_stack_temporary_max_bytes must be an integer")
        if value < 0:
            raise ValueError("c_stack_temporary_max_bytes must be non-negative")
        self.__c_stack_temporary_max_bytes = value

    @property
    def c_allow_vla_temporaries(self):
        return self.__c_allow_vla_temporaries

    @c_allow_vla_temporaries.setter
    def c_allow_vla_temporaries(self, value):
        if not isinstance(value, bool):
            raise TypeError("c_allow_vla_temporaries must be a bool")
        self.__c_allow_vla_temporaries = value

    def _setup_iso_c_defaults(self):
        """Set ISO C default datatypes."""
        from .datatype import int64, float64, complex128, bool8, char

        self.set_default_from_datatype(int64, iso_c=True)
        self.set_default_from_datatype(float64, iso_c=True)
        self.set_default_from_datatype(complex128, iso_c=True)
        self.set_default_from_datatype(bool8, iso_c=True)
        self.set_default_from_datatype(char, iso_c=True)

    def _setup_fortran_defaults(self):
        """Set Fortran default datatypes."""
        from .datatype import int64, float64, complex128, bool8, char

        self.set_default_from_datatype(int64, iso_c=False)
        self.set_default_from_datatype(float64, iso_c=False)
        self.set_default_from_datatype(complex128, iso_c=False)
        self.set_default_from_datatype(bool8, iso_c=False)
        self.set_default_from_datatype(char, iso_c=False)

    def initialize_default_datatypes(self):
        """Initialize default datatypes once imports are ready."""
        if self.__defaults_initialized:
            return
        if self.iso_C:
            self._setup_iso_c_defaults()
        else:
            self._setup_fortran_defaults()
        self.__defaults_initialized = True


settings = Settings(iso_C=True, use_numpy_allocator=True)
