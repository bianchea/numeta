import inspect
import os
import platform
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .array_shape import ArrayShape, SCALAR, UNKNOWN
from .datatype import DataType, ArrayType, PointerType, get_datatype
from .settings import settings
from .ast import Variable
from .ast.expressions import ExpressionNode, GetAttr, GetItem
from .types_hint import comptime


def _compile_signature_extension(force=False):
    """Compile the C signature extension if needed."""
    import importlib.util

    # Check if already compiled
    spec = importlib.util.find_spec("numeta._signature")
    if spec is not None and not force:
        return True

    # Check if source exists
    c_file = Path(__file__).parent / "_signature.c"
    if not c_file.exists():
        return False

    # Try to compile
    try:
        from .compiler import Compiler

        module_dir = Path(__file__).parent

        # Use standard compiler flags
        std_flags = ["-O3", "-fPIC"]
        cc = "/usr/bin/gcc" if os.path.exists("/usr/bin/gcc") else "gcc"
        compiler = Compiler(cc, std_flags)

        include_dirs = [
            sysconfig.get_paths()["include"],
            np.get_include(),
        ]

        additional_flags = ["-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"]

        # First compile to object file
        obj_file, _ = compiler.compile_to_obj(
            name="_signature",
            directory=module_dir,
            sources=[c_file],
            include_dirs=include_dirs,
            additional_flags=additional_flags,
            obj_suffix=".o",
        )

        # Then link to shared library with correct Python extension name
        libraries = [f"python{sys.version_info.major}.{sys.version_info.minor}"]
        so_file = (
            module_dir
            / f"_signature.cpython-{sys.version_info.major}{sys.version_info.minor}-x86_64-linux-gnu.so"
        )

        # Use build_lib_command directly to control output filename
        command = compiler.build_lib_command(
            lib_file=so_file,
            obj_files=[obj_file],
            include_dirs=include_dirs,
            additional_flags=additional_flags,
            libraries=libraries,
            libraries_dirs=[],
            rpath_dirs=[],
        )

        compiler.run_command(command, cwd=module_dir)

        return so_file.exists()
    except Exception as e:
        return False


def _signature_extension_stale():
    import importlib.util

    c_file = Path(__file__).parent / "_signature.c"
    so_file = None
    try:
        spec = importlib.util.find_spec("numeta._signature")
    except ValueError:
        spec = None

    if spec is not None and spec.origin is not None:
        so_file = Path(spec.origin)
    else:
        module = sys.modules.get("numeta._signature")
        module_file = getattr(module, "__file__", None)
        if module_file is not None:
            so_file = Path(module_file)

    if so_file is None:
        return False
    if not so_file.exists() or not c_file.exists():
        return False
    return c_file.stat().st_mtime > so_file.stat().st_mtime


def _init_signature_module():
    """Initialize the signature module, compiling if necessary."""
    import importlib

    try:
        # Try importing first
        _signature = importlib.import_module("numeta._signature")
        if _signature_extension_stale():
            if not _compile_signature_extension(force=True):
                return None, False
            if "numeta._signature" in sys.modules:
                del sys.modules["numeta._signature"]
            _signature = importlib.import_module("numeta._signature")
    except ImportError:
        # Try compiling and importing again
        if not _compile_signature_extension():
            return None, False
        # Clear import cache and try again
        if "numeta._signature" in sys.modules:
            del sys.modules["numeta._signature"]
        _signature = importlib.import_module("numeta._signature")

    types_dict = {
        "ArrayType": ArrayType,
        "PointerType": PointerType,
        "DataType": DataType,
        "ExpressionNode": ExpressionNode,
        "Variable": Variable,
        "GetAttr": GetAttr,
        "GetItem": GetItem,
        "SCALAR": SCALAR,
        "UNKNOWN": UNKNOWN,
        "NumpyGeneric": np.generic,
    }

    constants_dict = {
        "KIND_POSITIONAL_ONLY": inspect.Parameter.POSITIONAL_ONLY.value,
        "KIND_POSITIONAL_OR_KEYWORD": inspect.Parameter.POSITIONAL_OR_KEYWORD.value,
        "KIND_VAR_POSITIONAL": inspect.Parameter.VAR_POSITIONAL.value,
        "KIND_KEYWORD_ONLY": inspect.Parameter.KEYWORD_ONLY.value,
        "KIND_VAR_KEYWORD": inspect.Parameter.VAR_KEYWORD.value,
        "INSPECT_EMPTY": inspect._empty,
    }

    _signature.init_globals(types_dict, constants_dict)
    return _signature, True


_c_signature_backend, _c_signature_backend_available = _init_signature_module()
if _c_signature_backend is None:
    _c_signature_backend_available = False

# Backward-compatible aliases kept for external imports/tests.
_signature = _c_signature_backend
_signature_c_available = _c_signature_backend_available


def _use_c_signature_parser_backend():
    return (
        _c_signature_backend_available
        and _c_signature_backend is not None
        and settings.use_c_signature_parser
    )


def _resolve_effective_intent(arg):
    target = arg
    while isinstance(target, (GetAttr, GetItem)):
        target = target.variable

    if isinstance(target, Variable) and target.intent != "in":
        return "inout"
    return "in"


def _is_scalar_shape(shape) -> bool:
    return isinstance(shape, ArrayShape) and shape.is_scalar


def _is_unknown_rank_shape(shape) -> bool:
    return isinstance(shape, ArrayShape) and shape.is_unknown


def _signature_dtype_token(dtype):
    np_dtype = dtype.get_numpy()
    return dtype if np_dtype is None else np_dtype


def _contains_symbolic_signature_arg(args, kwargs) -> bool:
    def is_symbolic(arg):
        if isinstance(arg, (ArrayType, PointerType, ExpressionNode)):
            return True
        return isinstance(arg, type) and issubclass(arg, DataType)

    return any(is_symbolic(arg) for arg in args) or any(is_symbolic(arg) for arg in kwargs.values())


@dataclass(frozen=True)
class ArgumentSpec:
    """
    This class is used to store the details of the arguments of the function.
    The ones that are compile-time are stored in the is_comptime attribute.
    """

    name: str
    is_comptime: bool = False
    comptime_value: Any = None
    datatype: DataType | None = None
    shape: ArrayShape | None = None
    rank: int = 0  # rank of the array, 0 for scalar
    intent: str = "inout"  # can be "in" or "inout"
    to_pass_by_value: bool = False
    is_keyword: bool = False
    c_const: bool = False
    c_restrict: bool = False
    c_volatile: bool = False
    explicit_pointer: bool = False


@dataclass(frozen=True)
class ParameterInfo:
    """Store metadata about a Python function parameter."""

    name: str
    kind: inspect._ParameterKind
    default: Any = inspect._empty
    is_comptime: bool = False


def parse_function_parameters(func):
    py_signature = inspect.signature(func)

    params = []
    catch_var_positional_name = "args"

    for name, parameter in py_signature.parameters.items():
        is_comp = func.__annotations__.get(name) is comptime
        params.append(ParameterInfo(name, parameter.kind, parameter.default, is_comp))

        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            catch_var_positional_name = name

    fixed_param_indices = [
        i
        for i, p in enumerate(params)
        if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    n_positional_or_default_args = len(fixed_param_indices)

    return params, fixed_param_indices, n_positional_or_default_args, catch_var_positional_name


def _get_signature_and_runtime_args_py(
    args,
    kwargs,
    *,
    params,
    fixed_param_indices,
    n_positional_or_default_args,
    catch_var_positional_name,
):
    """
    This method quickly extracts the signature and runtime arguments from the provided args.
    If the runtime arguments are not all numpy arrays or numeric types, the call is not to be
    executed.
    It returns a tuple of:
    - to_execute: a boolean indicating if the function can be executed with the provided args
    - signature: a tuple of the signature of the function
    - runtime_args: a list of runtime arguments to be passed to run the function
    A signature is a tuple of tuples, where each inner tuple represents an argument.
    - (name, dtype,) is scalar types passed by value
    - (name, dtype, 0) is for scalar types passed by reference
    - (name, dtype, rank) for numpy arrays
    - (name, dtype, rank, has_fortran_order) to set the Fortran order
    - (name, dtype, rank, has_fortran_order, intent) intent can be "in" or "inout"
    - (name, dtype, rank, has_fortran_order, intent, shape) if the shape is know at comptime
    **name** is a tuple is the argument comes from a variable positional argument (*args)
    """

    to_execute = True

    def get_signature_from_arg(arg, name):
        nonlocal to_execute

        if isinstance(arg, np.ndarray):
            arg_signature = (name, arg.dtype, len(arg.shape), np.isfortran(arg))
        elif isinstance(arg, (int, float, complex)):
            arg_signature = (
                name,
                type(arg),
            )
        elif isinstance(arg, np.generic):
            # it is a numpy scalar like np.int32(1) or np.float64(1.0) or a struct
            # A struct is mutable
            if arg.dtype.names is not None:
                arg_signature = (name, arg.dtype, 0)
            else:
                arg_signature = (
                    name,
                    arg.dtype,
                )
        elif isinstance(arg, ArrayType):
            to_execute = False
            dtype_token = _signature_dtype_token(arg.dtype)
            if arg.shape.is_unknown or (
                not settings.add_shape_descriptors and arg.shape.has_comptime_undefined_dims()
            ):
                # it is a pointer
                arg_signature = (name, dtype_token, None, arg.shape.fortran_order)
            elif arg.shape.has_comptime_undefined_dims():
                arg_signature = (
                    name,
                    dtype_token,
                    arg.shape.rank,
                    arg.shape.fortran_order,
                )
            else:
                arg_signature = (
                    name,
                    dtype_token,
                    arg.shape.rank,
                    arg.shape.fortran_order,
                    "inout",
                    arg.shape.as_tuple(),
                )
        elif isinstance(arg, PointerType):
            to_execute = False
            arg_signature = (
                name,
                arg,
            )
        elif isinstance(arg, type) and issubclass(arg, DataType):
            to_execute = False
            dtype_token = _signature_dtype_token(arg)
            arg_signature = (
                name,
                dtype_token,
            )
        elif isinstance(arg, ExpressionNode):
            to_execute = False
            dtype = arg.dtype
            intent = _resolve_effective_intent(arg)
            dtype_token = _signature_dtype_token(dtype)

            if _is_scalar_shape(arg._shape):
                if getattr(dtype, "_is_vector", False):
                    arg_signature = (
                        name,
                        dtype_token,
                    )
                elif intent == "inout":
                    arg_signature = (name, dtype.get_numpy(), 0, False, intent)
                else:
                    arg_signature = (
                        name,
                        dtype_token,
                    )
            elif _is_unknown_rank_shape(arg._shape) or (
                not settings.add_shape_descriptors and arg._shape.has_comptime_undefined_dims()
            ):
                arg_signature = (name, dtype_token, None, False, intent)
            elif arg._shape.has_comptime_undefined_dims():
                arg_signature = (
                    name,
                    dtype_token,
                    arg._shape.rank,
                    arg._shape.fortran_order,
                    intent,
                )
            else:
                if not settings.ignore_fixed_shape_in_nested_calls:
                    arg_signature = (
                        name,
                        dtype_token,
                        arg._shape.rank,
                        arg._shape.fortran_order,
                        intent,
                        arg._shape.as_tuple(),
                    )
                else:
                    arg_signature = (name, dtype_token, None, False, intent)
        else:
            raise ValueError(f"Argument {name} of type {type(arg)} is not supported")

        return arg_signature

    runtime_args = []
    signature = [None] * n_positional_or_default_args

    unused_kwargs = kwargs
    pos_idx = 0

    for fi, param_idx in enumerate(fixed_param_indices):
        param = params[param_idx]

        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            if pos_idx < len(args):
                arg = args[pos_idx]
                pos_idx += 1
            elif param.name in unused_kwargs:
                arg = unused_kwargs.pop(param.name)
            elif param.default is not inspect._empty:
                arg = param.default
            else:
                raise ValueError(f"Missing required argument: {param.name}")
        else:  # KEYWORD_ONLY
            if param.name in unused_kwargs:
                arg = unused_kwargs.pop(param.name)
            elif param.default is not inspect._empty:
                arg = param.default
            else:
                raise ValueError(f"Missing required argument: {param.name}")

        if param.is_comptime:
            signature[fi] = arg
        else:
            signature[fi] = get_signature_from_arg(arg, param.name)
            runtime_args.append(arg)

    # catch the *args variable positional arguments
    if pos_idx < len(args):
        for j, arg in enumerate(args[pos_idx:]):
            name = (catch_var_positional_name, j)
            signature.append(get_signature_from_arg(arg, name))
            runtime_args.append(arg)

    # catch the **kwargs variable keyword arguments
    unused_kwargs_keys = (
        unused_kwargs.keys() if not settings.reorder_kwargs else sorted(unused_kwargs.keys())
    )
    for name in unused_kwargs_keys:
        arg = unused_kwargs[name]
        signature.append(get_signature_from_arg(arg, name))
        runtime_args.append(arg)

    return to_execute, tuple(signature), runtime_args


def convert_signature_to_argument_specs(
    signature,
    *,
    params,
    fixed_param_indices,
    n_positional_or_default_args,
):
    """
    Converts a signature tuple into a list of ArgumentSpec objects.
    A signature is a tuple of tuples, where each inner tuple represents an argument.
    """

    def convert_arg_to_argument_spec(arg, is_keyword, name=None):
        name = arg[0] if name is None else name

        if len(arg) == 2 and isinstance(arg[1], PointerType):
            pointer = arg[1]
            return ArgumentSpec(
                name,
                datatype=pointer.dtype,
                rank=None,
                shape=UNKNOWN,
                intent="in" if pointer.const else "inout",
                is_keyword=is_keyword,
                c_const=pointer.const,
                c_restrict=pointer.restrict,
                c_volatile=pointer.volatile,
                explicit_pointer=True,
            )

        dtype = get_datatype(arg[1])
        if len(arg) == 2:
            # it is a numeric type or a string
            # So the intent will be always "in"
            # but complex numbers cannot be passed by value because of C
            ap = ArgumentSpec(
                name,
                datatype=dtype,
                shape=SCALAR,
                to_pass_by_value=dtype.can_be_value(),
                intent="in",
                is_keyword=is_keyword,
            )
        else:
            # for numpy arrays arg[1] is the rank, for the other types it is the shape
            rank = arg[2]

            fortran_order = False
            if len(arg) >= 4:
                fortran_order = arg[3]

            intent = "inout"
            if len(arg) >= 5:
                intent = arg[4]

            if rank is None:
                shape = UNKNOWN
            elif rank == 0:
                shape = SCALAR
            else:
                shape = ArrayShape([None] * rank, fortran_order=fortran_order)

            if len(arg) >= 6:
                # it means that the shape is known at comptime
                shape = ArrayShape(arg[5], fortran_order=fortran_order)

            ap = ArgumentSpec(
                name,
                datatype=dtype,
                rank=rank,
                shape=shape,
                intent=intent,
                is_keyword=is_keyword,
            )

        return ap

    signature_spec = []
    for i, arg in enumerate(signature):

        if i < n_positional_or_default_args:
            param = params[fixed_param_indices[i]]
            if param.is_comptime:
                ap = ArgumentSpec(
                    param.name,
                    is_comptime=True,
                    comptime_value=arg,
                    is_keyword=param.kind == inspect.Parameter.KEYWORD_ONLY,
                )
            else:
                is_keyword = param.kind == inspect.Parameter.KEYWORD_ONLY
                ap = convert_arg_to_argument_spec(arg, is_keyword)
        elif isinstance(arg[0], tuple):
            # it is a positional argument called with *args
            is_keyword = False
            name = f"{arg[0][0]}_{arg[0][1]}"
            ap = convert_arg_to_argument_spec(arg, is_keyword, name=name)
        else:
            is_keyword = True
            ap = convert_arg_to_argument_spec(arg, is_keyword)

        signature_spec.append(ap)

    return signature_spec


def get_signature_and_runtime_args(
    args,
    kwargs,
    *,
    params,
    fixed_param_indices,
    n_positional_or_default_args,
    catch_var_positional_name,
):
    if _use_c_signature_parser_backend() and not _contains_symbolic_signature_arg(args, kwargs):
        backend = _c_signature_backend
        assert backend is not None
        try:
            return backend.get_signature_and_runtime_args(
                args,
                kwargs,
                params,
                fixed_param_indices,
                n_positional_or_default_args,
                catch_var_positional_name,
                settings.add_shape_descriptors,
                settings.ignore_fixed_shape_in_nested_calls,
                settings.reorder_kwargs,
            )
        except TypeError:
            pass

    return _get_signature_and_runtime_args_py(
        args,
        kwargs,
        params=params,
        fixed_param_indices=fixed_param_indices,
        n_positional_or_default_args=n_positional_or_default_args,
        catch_var_positional_name=catch_var_positional_name,
    )


def fast_dispatch(
    args,
    kwargs,
    *,
    params,
    fixed_param_indices,
    n_positional_or_default_args,
    catch_var_positional_name,
    fast_call_dict,
):
    """Signature parse + dict lookup + call in one shot.

    Returns a 4-tuple (hit, to_execute, payload, runtime_args):
      hit=True  => payload is the actual function result (cache hit, called via Vectorcall)
      hit=False, to_execute=True  => payload is signature (cache miss, caller must load+call)
      hit=False, to_execute=False => payload is signature (symbolic, caller handles)
    """
    if _use_c_signature_parser_backend() and not _contains_symbolic_signature_arg(args, kwargs):
        backend = _c_signature_backend
        assert backend is not None
        try:
            return backend.fast_dispatch(
                args,
                kwargs,
                params,
                fixed_param_indices,
                n_positional_or_default_args,
                catch_var_positional_name,
                settings.add_shape_descriptors,
                settings.ignore_fixed_shape_in_nested_calls,
                settings.reorder_kwargs,
                fast_call_dict,
            )
        except TypeError:
            pass

    # Python fallback
    to_execute, sig, runtime_args = _get_signature_and_runtime_args_py(
        args,
        kwargs,
        params=params,
        fixed_param_indices=fixed_param_indices,
        n_positional_or_default_args=n_positional_or_default_args,
        catch_var_positional_name=catch_var_positional_name,
    )
    if to_execute and sig in fast_call_dict:
        result = fast_call_dict[sig](*runtime_args)
        return (True, True, result, None)
    return (False, to_execute, sig, runtime_args)


def compile_custom_signature_parser(name, params, directory):
    """
    Generate and compile a custom C signature parser for the given function parameters.

    This function creates an optimized C implementation of the signature parsing logic
    specific to the function's parameter structure. This avoids the overhead of
    generic Python signature parsing for simple functions.

    The generated parser:
    1. Checks argument types using fast C-API calls (PyArray_Check, etc).
    2. Constructs the signature tuple directly without Python function calls.
    3. Populates the runtime_args array directly.

    Args:
        name (str): The name of the function (used for naming the C function).
        params (list[ParameterInfo]): List of parameter information objects describing
                                      the expected arguments.
        directory (Path): The directory where the compiled shared library should be stored.

    Returns:
        tuple[str, str] | None: A tuple containing (library_path, entry_point_name)
                                if compilation succeeds. Returns None if the function
                                is not eligible for optimization (e.g. has varargs)
                                or if compilation fails.
    """
    # Only generate custom parser for simple functions:
    # - no *args/**kwargs
    # - no defaults
    # - only POSITIONAL_OR_KEYWORD parameters (no positional-only/keyword-only)
    # Only support POSITIONAL_OR_KEYWORD parameters (no positional-only, keyword-only, varargs)
    has_unsupported_kinds = any(
        getattr(p, "kind", inspect.Parameter.POSITIONAL_OR_KEYWORD)
        != inspect.Parameter.POSITIONAL_OR_KEYWORD
        for p in params
    )

    # Skip if there are default arguments, as the simple custom parser expects strict argument count
    has_defaults = any(
        getattr(p, "default", inspect.Parameter.empty) is not inspect.Parameter.empty
        for p in params
    )

    if has_defaults or has_unsupported_kinds:
        return None

    # Helper to extract parameter information
    params_info = []
    for i, p in enumerate(params):
        p_name = getattr(p, "name", str(p))
        is_comptime = getattr(p, "is_comptime", False)

        params_info.append(
            {
                "index": i,
                "name": p_name,
                "is_comptime": is_comptime,
            }
        )

    # Count required and optional args
    total_count = len(params_info)

    # Generate parser function name
    func_name = f"parse_{name}_signature"

    # Build interned string declarations for parameter names
    name_includes = []
    name_assignments = []
    for p in params_info:
        p_name = p["name"]
        var_name = f"str_{p_name}"
        name_includes.append(f"static PyObject *{var_name} = NULL;")
        name_assignments.append(
            f'    if (!{var_name}) {var_name} = PyUnicode_InternFromString("{p_name}");'
        )

    # Generate argument processing using NumPy C API directly
    arg_processing = []
    runtime_idx = 0
    for i in range(total_count):
        param_name = params_info[i]["name"]
        is_comptime = params_info[i]["is_comptime"]

        code = f"""    // Arg {i}: {param_name}{' (comptime)' if is_comptime else ''}
    {{
        PyObject *arg_{i} = PyTuple_GET_ITEM(args, {i});
"""

        if is_comptime:
            # Comptime argument: pass value directly to signature, skip runtime_args
            code += f"""        Py_INCREF(arg_{i});
        sig[{i}] = arg_{i};
    }}"""
        else:
            # Runtime argument: extract type signature and add to runtime_args
            # Unsupported argument kinds return -1 so BaseFunction can
            # fall back to the generic parser.
            code += f"""
        // Check if it's a numpy array using NumPy C API (fast!)
        if (PyArray_Check(arg_{i})) {{
            PyArrayObject *arr = (PyArrayObject*)arg_{i};
            PyObject *dtype = (PyObject*)PyArray_DESCR(arr);
            int ndim = PyArray_NDIM(arr);
            int is_f = PyArray_ISFORTRAN(arr);
            
            Py_INCREF(dtype);
            sig[{i}] = PyTuple_Pack(4, str_{param_name}, dtype,
                                    PyLong_FromLong(ndim), is_f ? Py_True : Py_False);
            Py_DECREF(dtype);
        }} else if (PyArray_CheckScalar(arg_{i})) {{
            // NumPy scalar (e.g. numpy.void, numpy.int64)
            PyObject *dtype = PyObject_GetAttrString(arg_{i}, "dtype");
            if (!dtype) return -1;
            
            PyObject *names = PyObject_GetAttrString(dtype, "names");
            int is_struct = (names && names != Py_None);
            Py_XDECREF(names);
            
            if (is_struct) {{
                sig[{i}] = PyTuple_Pack(3, str_{param_name}, dtype, PyLong_FromLong(0));
            }} else {{
                sig[{i}] = PyTuple_Pack(2, str_{param_name}, dtype);
            }}
            Py_DECREF(dtype);
        }} else if (PyLong_Check(arg_{i}) || PyFloat_Check(arg_{i}) || PyComplex_Check(arg_{i})) {{
            PyObject *t = (PyObject*)Py_TYPE(arg_{i});
            Py_INCREF(t);
            sig[{i}] = PyTuple_Pack(2, str_{param_name}, t);
            Py_DECREF(t);
        }} else {{
            return -1;
        }}
        if (!sig[{i}]) return -1;
        runtime_args[{runtime_idx}] = arg_{i};
    }}"""
            runtime_idx += 1

        arg_processing.append(code)

    # Generate template with NumPy C API initialization
    template = f"""#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// NumPy API initialization - only runs once
static int numpy_initialized = 0;
static void init_numpy() {{
    if (!numpy_initialized) {{
        if (_import_array() < 0) {{
            PyErr_Print();
            PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        }}
        numpy_initialized = 1;
    }}
}}

{chr(10).join(name_includes)}

int {func_name}(PyObject *args, PyObject **runtime_args, PyObject **sig, int *nargs_out, int *nruntime_out, void *self) {{
    init_numpy();  // Initialize NumPy on first call
    
{chr(10).join(name_assignments)}
    if (PyTuple_GET_SIZE(args) != {total_count}) {{
        PyErr_Format(PyExc_TypeError, "{name}() takes exactly {total_count} arguments (%%zd given)", PyTuple_GET_SIZE(args));
        return -1;
    }}
{chr(10).join(arg_processing)}
    *nargs_out = {total_count};
    *nruntime_out = {runtime_idx};
    return 0;
}}
"""
    parser_code = template

    # Setup paths
    # Use a hash of params to ensure uniqueness if name collides?
    # We'll use a random suffix or hash of the signature to allow recompilation if needed.
    import hashlib

    sig_hash = hashlib.md5(f"{name}_{params}".encode()).hexdigest()[:8]
    parser_name = f"_parser_{name}_{sig_hash}"

    parser_dir = directory / "parsers"
    parser_dir.mkdir(exist_ok=True)

    c_file = parser_dir / f"{parser_name}.c"

    # We construct the shared library path
    # Compiler.compile_to_library typically produces "lib{name}.so" in the directory
    # But we want to be sure about the name to load it later.
    # The Compiler class handles the platform specific extension?
    # compile_to_library hardcodes "lib{name}.so" in the implementation I read earlier.
    so_file = parser_dir / f"lib{parser_name}.so"

    # Check if already compiled
    if so_file.exists():
        return str(so_file), func_name

    # Write C code
    c_file.write_text(parser_code)

    # Compile using Compiler class
    try:
        from .compiler import Compiler

        # Get include paths dynamically
        py_include = sysconfig.get_paths()["include"]
        np_include = np.get_include()

        # Initialize compiler (GCC)
        cc = "/usr/bin/gcc" if os.path.exists("/usr/bin/gcc") else "gcc"
        compiler = Compiler(cc, ["-O3"])

        # Compile to shared library
        compiler.compile_to_library(
            name=parser_name,
            src_files=[c_file],
            directory=parser_dir,
            include_dirs=[py_include, np_include],
        )

        return str(so_file), func_name
    except Exception as e:
        # If compilation fails, return None (fallback to generic parser)
        # We print a warning so the user knows optimization failed but execution continues
        print(f"Warning: Failed to compile custom parser for {name}: {e}")
        return None


def get_signature_and_runtime_args_py(*args, **kwargs):
    """Python implementation - exposed for testing parity with C implementation."""
    return _get_signature_and_runtime_args_py(*args, **kwargs)
