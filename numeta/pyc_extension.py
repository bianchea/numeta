import numpy as np

NP_COMPLEX256 = getattr(np, "complex256", np.clongdouble)

import importlib.util
import platform
import sys
import sysconfig
from string import Template
from pathlib import Path


from .compiler import Compiler
from .native_abi import (
    NUMETA_WRAPPER_ABI_VERSION,
    codegen_abi_settings,
    format_metadata_mismatches,
    metadata_mismatches,
    numpy_c_abi_version,
)
from .settings import settings
from .wrapper_spec import WrapperSpec

WRAPPER_CACHE_FORMAT_VERSION = 3


class PyCExtension:
    SUFFIX = "__nm_pyext"

    def __init__(
        self,
        name,
        functions,
        do_checks=True,
        function_checks=None,
    ):
        self.name = f"{name}{self.SUFFIX}"
        self.functions = [WrapperSpec.coerce(function) for function in functions]
        self.do_checks = do_checks
        self.function_checks = dict(function_checks or {})
        self.lib_path = None
        self.cache_info = None

    def set_lib_path(self, path):
        self.lib_path = path

    def build_cache_info(
        self,
        compile_flags,
        backend=None,
        *,
        compiler_identity=None,
        simd_arch=None,
        simd_features=(),
    ):
        if backend is None:
            backend = settings.default_backend
        if compiler_identity is None:
            compiler_identity = Compiler(
                settings.c_compiler,
                compile_flags,
                setting="set_c_compiler",
                env_var="NUMETA_CC",
            ).identity()
        return {
            "format_version": WRAPPER_CACHE_FORMAT_VERSION,
            "numeta_wrapper_abi_version": NUMETA_WRAPPER_ABI_VERSION,
            "python_soabi": sysconfig.get_config_var("SOABI"),
            "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
            "python_version": [sys.version_info.major, sys.version_info.minor],
            "numpy_version": np.__version__,
            "numpy_c_abi": numpy_c_abi_version(),
            "platform": platform.system(),
            "machine": platform.machine(),
            "backend": backend,
            "compile_flags": Compiler._normalize_flags(compile_flags),
            "compiler": dict(compiler_identity),
            "simd_arch": simd_arch,
            "simd_features": list(simd_features),
            "wrapper_name": self.name,
            "do_checks": self.do_checks,
            "function_checks": dict(sorted(self.function_checks.items())),
            "codegen_settings": codegen_abi_settings(),
        }

    def set_cache_info(self, compile_flags, backend=None, **metadata):
        self.cache_info = self.build_cache_info(
            compile_flags,
            backend=backend,
            **metadata,
        )
        return self.cache_info

    def cache_mismatches(self, compile_flags, backend=None, **metadata):
        current = self.build_cache_info(
            compile_flags,
            backend=backend,
            **metadata,
        )
        return metadata_mismatches(getattr(self, "cache_info", None), current)

    def cache_matches(self, compile_flags, backend=None, **metadata):
        return not self.cache_mismatches(compile_flags, backend=backend, **metadata)

    def format_cache_mismatches(self, compile_flags, backend=None, **metadata) -> str:
        return format_metadata_mismatches(
            self.cache_mismatches(compile_flags, backend=backend, **metadata)
        )

    def compile(
        self,
        core_lib_name,
        core_lib_path,
        directory,
        compile_flags,
        backend=None,
        runtime_rpath=None,
        simd_arch=None,
        simd_features=(),
    ):
        if self.lib_path is not None:
            return self.lib_path

        if backend is None:
            backend = settings.default_backend

        if backend == "fortran":
            libraries = [
                "gfortran",
                "mvec",
                f"python{sys.version_info.major}.{sys.version_info.minor}",
                core_lib_name,
            ]
        elif backend == "c":
            libraries = [
                f"python{sys.version_info.major}.{sys.version_info.minor}",
                core_lib_name,
            ]
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        include_dirs = [sysconfig.get_paths()["include"], np.get_include()]
        additional_flags = ["-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"]

        compiler = Compiler(
            settings.c_compiler,
            compile_flags,
            setting="set_c_compiler",
            env_var="NUMETA_CC",
        )
        self.set_cache_info(
            compile_flags,
            backend=backend,
            compiler_identity=compiler.identity(),
            simd_arch=simd_arch,
            simd_features=simd_features,
        )

        wrapper_src = Path(directory) / f"{self.name}.c"
        self.write(wrapper_src)

        lib = compiler.compile_to_library(
            self.name,
            [wrapper_src],
            directory=directory,
            include_dirs=include_dirs,
            libraries=libraries,
            libraries_dirs=[core_lib_path],
            rpath_dirs=[core_lib_path if runtime_rpath is None else runtime_rpath],
            additional_flags=additional_flags,
        )
        self.set_lib_path(lib)
        return lib

    def load(self, func_name):
        if self.lib_path is None:
            raise RuntimeError("PyCExtension is not compiled yet.")

        spec = importlib.util.spec_from_file_location(self.name, self.lib_path)
        compiled_sub = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compiled_sub)
        return getattr(compiled_sub, func_name)

    def construct_module(self):
        from numeta.c.complex_compat import COMPLEX_COMPAT_HEADER

        template = """
#include <Python.h>
#include <numpy/arrayobject.h>
${complex_compat}

${numpy_wrappers} 

${struct_definitions}

${procedure_definitions}

// Method definition table
static PyMethodDef Methods[] = {
    ${module_procedures}
    {NULL, NULL, 0, NULL}  // Sentinel
};

// Module definition
static struct PyModuleDef ${name} = {
    PyModuleDef_HEAD_INIT,
    "${name}",  // Module name
    NULL,  // Optional module documentation
    -1,
    Methods
};

// Module initialization function
PyMODINIT_FUNC PyInit_${name}(void) {
    import_array();  // Initialize NumPy API
    return PyModule_Create(&${name});
}
"""

        substitutions = {"complex_compat": COMPLEX_COMPAT_HEADER}
        substitutions["name"] = f"{self.name}"

        from .wrappers.numpy_mem import numpy_mem

        substitutions["numpy_wrappers"] = numpy_mem.get_code()

        struct_definitions = {}
        procedure_definitions = []
        module_procedures = []

        for function_spec in self.functions:
            name = function_spec.name
            args_details = function_spec.arguments
            return_specs = function_spec.returns

            # find the structs
            for arg in args_details:
                if not arg.is_comptime and arg.datatype.is_struct():
                    # bfs to declare all nested structs
                    def bfs(struct):
                        for _, nested_dtype, _ in struct.members:
                            if (
                                nested_dtype.is_struct()
                                and nested_dtype.name not in struct_definitions
                            ):
                                bfs(nested_dtype)
                        struct_definitions[struct.name] = struct.c_declaration()

                    bfs(arg.datatype)

            module_procedures.append(
                "".join(
                    [
                        f'{{"{name}", (PyCFunction)(void(*)(void))nm_{name}, METH_FASTCALL, "Wrapper generated by numeta"}},'
                    ]
                )
            )

            procedure_definitions.append(
                self.construct_procedure(
                    name,
                    args_details,
                    return_specs,
                    shape_equalities=function_spec.shape_equalities,
                    do_checks=self.function_checks.get(name, self.do_checks),
                )
            )

        substitutions["module_procedures"] = "\n".join(module_procedures)
        substitutions["procedure_definitions"] = "\n".join(procedure_definitions)
        substitutions["struct_definitions"] = "".join(struct_definitions.values())

        module_template = Template(template)
        module_template = module_template.substitute(substitutions)

        return module_template

    def construct_procedure(
        self,
        name,
        args_details,
        return_specs,
        *,
        shape_equalities=(),
        do_checks=None,
    ):
        template = """
void ${fortran_name}(${fortran_args});

static PyObject* ${procedure_name}(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {

    if (nargs != ${n_args}) {
        PyErr_SetString(PyExc_TypeError, "${fortran_name} takes exactly ${n_args} arguments");
        return NULL;
    }

    ${initializations}

    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Error occurred while parsing arguments.");
        return NULL;
    }

    ${checks}

    ${return_args_declarations}

    ${fortran_name}(${call_args});

    ${return_args_conversion_to_numpy}

    ${return}
}
"""
        args = [a for a in args_details if not a.is_comptime]

        substitutions = {}
        substitutions["fortran_name"] = name
        substitutions["procedure_name"] = f"nm_{name}"
        substitutions["n_args"] = len(args)

        fortran_args = [self.get_fortran_args(var) for var in args]
        substitutions["initializations"] = "\n    ".join(
            [self.get_initialization(i, var) for i, var in enumerate(args)]
        )

        substitutions["checks"] = ""
        if self.do_checks if do_checks is None else do_checks:
            checks = [self.get_check(var) for var in args]
            checks.extend(
                self.get_shape_check(left, right, index)
                for index, equality in enumerate(shape_equalities)
                for left, right in ((equality.left, equality.right),)
            )
            substitutions["checks"] = "\n    ".join(checks)

        call_args = [self.get_call_args(var) for var in args]

        substitutions["return_args_declarations"] = "\n"
        substitutions["return_args_conversion_to_numpy"] = ""
        return_vars = []
        for i, (dtype, rank) in enumerate(return_specs):
            if rank != 0:
                substitutions[
                    "return_args_declarations"
                ] += f"\n    npy_intp out_shape_{i}[{rank}];"
                call_args.append(f"out_shape_{i}")
                fortran_args.append(f"npy_intp* out_shape_{i}")

                substitutions["return_args_declarations"] += f"\n    void* out_ptr_{i} = NULL;"
                call_args.append(f"&out_ptr_{i}")
                fortran_args.append(f"void** out_ptr_{i}")
                substitutions[
                    "return_args_conversion_to_numpy"
                ] += f"\n    PyObject* ret_out_{i} = PyArray_SimpleNewFromData({rank}, out_shape_{i}, {dtype.get_cnumpy().upper()}, out_ptr_{i});"
                substitutions[
                    "return_args_conversion_to_numpy"
                ] += f"\n    PyArray_ENABLEFLAGS((PyArrayObject*)ret_out_{i}, NPY_ARRAY_OWNDATA);"
                return_vars.append(f"ret_out_{i}")
            else:
                substitutions[
                    "return_args_declarations"
                ] += f"\n    {dtype.get_cnumpy()} out_val_{i};"
                call_args.append(f"&out_val_{i}")
                fortran_args.append(f"{dtype.get_cnumpy()}* out_val_{i}")
                substitutions[
                    "return_args_conversion_to_numpy"
                ] += f"\n    PyObject* ret_out_{i} = PyArray_Scalar(&out_val_{i}, PyArray_DescrFromType({dtype.get_cnumpy().upper()}), NULL);"
                return_vars.append(f"ret_out_{i}")

        substitutions["fortran_args"] = ", ".join(fortran_args)
        substitutions["call_args"] = ", ".join(call_args)

        if len(return_vars) == 0:
            substitutions["return"] = "Py_RETURN_NONE;"
        elif len(return_vars) == 1:
            substitutions["return"] = f"return {return_vars[0]};"
        else:
            return_str = f"PyObject* tuple = PyTuple_New({len(return_vars)});"
            for i, var in enumerate(return_vars):
                return_str += f"\n    Py_INCREF({var});"
                return_str += f"\n    PyTuple_SetItem(tuple, {i}, {var});"
            return_str += "\n    return tuple;"
            substitutions["return"] = return_str

        return Template(template).substitute(substitutions)

    def get_fortran_args(self, variable):
        """
        Returns the Fortran argument declaration for a given variable.
        Example for this function:

            void matmul(int32_t* a, int32_t* b, int32_t* c);

        this function returns the string 'int32_t* a' for the first argument.
        """

        if variable.to_pass_by_value:
            return f"{variable.datatype.get_cnumpy()} {variable.name}"
        elif settings.add_shape_descriptors and variable.shape.has_comptime_undefined_dims():
            return (
                f"npy_intp* {variable.name}_dims, {variable.datatype.get_cnumpy()}* {variable.name}"
            )
        else:
            return f"{variable.datatype.get_cnumpy()}* {variable.name}"

    def get_call_args(self, variable):
        """
        Returns the call argument declaration for a given variable.
        Example for this function:

            ciao((int32_t)a);

        this function returns the string '(int32_t)a' for the first argument.
        """
        if variable.to_pass_by_value:
            return f"({variable.datatype.get_cnumpy()}){variable.name}"
        elif variable.rank == 0 and variable.datatype.is_struct():
            return f"({variable.datatype.get_cnumpy()}*){variable.name}"
        elif settings.add_shape_descriptors and variable.shape.has_comptime_undefined_dims():
            return f"PyArray_DIMS({variable.name}), ({variable.datatype.get_cnumpy()}*)PyArray_DATA({variable.name})"
        else:
            return f"({variable.datatype.get_cnumpy()}*)PyArray_DATA({variable.name})"

    def get_initialization(self, i, variable):
        """
        Returns the declaration and initialization of the variable in the C API interface.
        For instance:

            int32_t a = (int32_t)PyLong_AsLongLong(args[0]);

        Where args[idx] is the corresponding argument in the Python function.
        """

        if variable.to_pass_by_value:
            if variable.datatype.get_numpy() in (np.complex64, np.complex128, NP_COMPLEX256):
                cast = variable.datatype.get_capi_cast(f"args[{i}]")
                result = f"{variable.datatype.get_cnumpy()} {variable.name};\n"
                result += f"    if (PyArray_IsScalar(args[{i}], ComplexFloating)) {{\n"
                result += f"        PyArray_ScalarAsCtype(args[{i}], &{variable.name});\n"
                result += f"    }} else if (PyComplex_Check(args[{i}])) {{\n"
                result += f"        {variable.name} = ({variable.datatype.get_cnumpy()}){cast};\n"
                result += "    } else {\n"
                result += f"        PyErr_SetString(PyExc_TypeError, \"Argument '{variable.name}' must be a complex scalar\");\n"
                result += "        return NULL;\n"
                result += "    }"
                return result
            cast = variable.datatype.get_capi_cast(f"args[{i}]")
            return f"{variable.datatype.get_cnumpy()} {variable.name} = ({variable.datatype.get_cnumpy()}){cast};"
        elif variable.rank == 0 and variable.datatype.is_struct():
            result = f"{variable.datatype.get_cnumpy()}* {variable.name} = NULL;\n"
            result += f"     PyArray_ScalarAsCtype(args[{i}], &{variable.name});"
            return result
        return f"PyArrayObject *{variable.name} = (PyArrayObject*)args[{i}];"

    def get_check(self, variable):
        if variable.rank != 0:
            check_array = f"""
    if (!PyArray_Check({variable.name})) {{
        PyErr_SetString(PyExc_TypeError, "Argument '{variable.name}' must be a NumPy array");
        return NULL;
    }}"""
            if variable.shape.fortran_order:
                check_farray = f"""
    if (!PyArray_ISFARRAY({variable.name})) {{
        PyErr_SetString(PyExc_ValueError, "Input array '{variable.name}' is not Fortran contiguous or aligned.");
        return NULL; 
    }}"""
            else:
                check_farray = f"""
    if (!PyArray_ISCARRAY({variable.name})) {{
        PyErr_SetString(PyExc_ValueError, "Input array '{variable.name}' is not C contiguous or aligned.");
        return NULL; 
    }}"""
            if variable.datatype.is_struct():
                check_align = f"""
    if (!PyDataType_FLAGCHK((PyArray_Descr*)PyArray_DESCR({variable.name}), NPY_ALIGNED_STRUCT)) {{
        PyErr_SetString(PyExc_ValueError, "Input struct {variable.name} dtype is not aligned, use align=True ");
        return NULL;
    }}"""
                return check_array + check_farray + check_align
            else:
                check_type = f"""
    if (PyArray_TYPE({variable.name}) != {variable.datatype.get_cnumpy().upper()}) {{
        PyErr_SetString(PyExc_ValueError, "Input array '{variable.name}' does not have the required type ({variable.datatype.get_numpy()}) "); 
        return NULL;
    }}"""
                return check_array + check_farray + check_type
        else:
            return ""

    @staticmethod
    def get_shape_check(left, right, index):
        return f"""
    if (PyArray_NDIM({left}) != PyArray_NDIM({right})) {{
        PyErr_SetString(PyExc_ValueError, "Array shapes for '{left}' and '{right}' must match");
        return NULL;
    }}
    for (int _nm_shape_dim_{index} = 0;
         _nm_shape_dim_{index} < PyArray_NDIM({left});
         _nm_shape_dim_{index}++) {{
        if (PyArray_DIM({left}, _nm_shape_dim_{index}) !=
            PyArray_DIM({right}, _nm_shape_dim_{index})) {{
            PyErr_SetString(PyExc_ValueError, "Array shapes for '{left}' and '{right}' must match");
            return NULL;
        }}
    }}"""

    def write(self, filename: Path):
        extension_module = self.construct_module()
        filename.parent.mkdir(exist_ok=True)
        filename.write_text(extension_module)
