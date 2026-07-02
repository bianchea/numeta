#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stddef.h>
#include <dlfcn.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

// Globals for types
static PyObject *ArrayType = NULL;
static PyObject *PointerType = NULL;
static PyObject *DataType = NULL;
static PyObject *ExpressionNode = NULL;
static PyObject *Variable = NULL;
static PyObject *GetAttr = NULL;
static PyObject *GetItem = NULL;
static PyObject *SCALAR = NULL;
static PyObject *UNKNOWN = NULL;
static PyObject *INSPECT_EMPTY = NULL;
static PyObject *NumpyGeneric = NULL;

// Globals for strings - interned for speed
static PyObject *str_kind = NULL;
static PyObject *str_name = NULL;
static PyObject *str_default = NULL;
static PyObject *str_is_comptime = NULL;
static PyObject *str_dtype = NULL;
static PyObject *str_names = NULL;
static PyObject *str_shape = NULL;
static PyObject *str_get_numpy = NULL;
static PyObject *str_has_comptime_undefined_dims = NULL;
static PyObject *str_fortran_order = NULL;
static PyObject *str_rank = NULL;
static PyObject *str_dims = NULL;
static PyObject *str_variable = NULL;
static PyObject *str_intent = NULL;
static PyObject *str_in = NULL;
static PyObject *str_inout = NULL;
static PyObject *str_underscore_shape = NULL;
static PyObject *str_current_builder = NULL;

// Constants
static int KIND_POSITIONAL_ONLY = 0;
static int KIND_POSITIONAL_OR_KEYWORD = 1;
static int KIND_VAR_POSITIONAL = 2;
static int KIND_KEYWORD_ONLY = 3;
static int KIND_VAR_KEYWORD = 4;

#ifndef PyTuple_ITEMS
#define PyTuple_ITEMS(op) (((PyTupleObject *)(op))->ob_item)
#endif

// ============================================================================
// Forward declarations for BaseFunction
// ============================================================================

typedef struct BaseFunctionObject BaseFunctionObject;

// Signature parser function type - returns 0 on success, -1 on error
// Populates sig[], runtime_args[], and sets *nargs_out (signature len) and *nruntime_out
typedef int (*SignatureParserFunc)(PyObject *args, PyObject **runtime_args, 
                                    PyObject **sig, int *nargs_out, int *nruntime_out, 
                                    void *self);

// ============================================================================
// BaseFunction Type
// ============================================================================

typedef struct {
    int add_shape_descriptors;
    int ignore_fixed_shape_in_nested_calls;
    int reorder_kwargs;
    int use_c_dispatch;
} Settings;

// Optimized instance check - inline and use fast path
static inline int is_instance_fast(PyObject *obj, PyObject *cls) {
    if (cls == NULL || obj == NULL) return 0;
    // Fast path: check if cls is a type and use PyObject_TypeCheck
    if (PyType_Check(cls)) {
        return PyObject_TypeCheck(obj, (PyTypeObject*)cls);
    }
    int result = PyObject_IsInstance(obj, cls);
    // Clear any exception that might have been set during the isinstance check
    if (result == -1) {
        PyErr_Clear();
        return 0;
    }
    return result;
}

// Fast path for numpy arrays - aggressively inlined
static inline PyObject* sig_from_ndarray(PyObject *arg, PyObject *name) {
    PyArrayObject *arr = (PyArrayObject*)arg;
    PyObject *dtype = (PyObject*)PyArray_DESCR(arr);
    int ndim = PyArray_NDIM(arr);
    int is_fortran = PyArray_ISFORTRAN(arr);
    
    // Use PyTuple_Pack for small fixed-size tuples - faster and cleaner
    PyObject *sig = PyTuple_Pack(4, name, dtype, PyLong_FromLong(ndim), 
                                  is_fortran ? Py_True : Py_False);
    return sig;
}

// Fast path for scalars - aggressively inlined
static inline PyObject* sig_from_scalar(PyObject *arg, PyObject *name) {
    PyObject *type = (PyObject*)Py_TYPE(arg);
    return PyTuple_Pack(2, name, type);
}

static PyObject* signature_dtype_token(PyObject *dtype) {
    PyObject *numpy_dtype = PyObject_CallMethodObjArgs(dtype, str_get_numpy, NULL);
    if (!numpy_dtype) {
        return NULL;
    }
    if (numpy_dtype == Py_None) {
        Py_DECREF(numpy_dtype);
        PyErr_SetString(PyExc_TypeError, "Numeta dtype has no NumPy signature token");
        return NULL;
    }
    return numpy_dtype;
}

static int dtype_is_vector(PyObject *dtype) {
    PyObject *flag = PyObject_GetAttrString(dtype, "_is_vector");
    if (!flag) {
        PyErr_Clear();
        return 0;
    }
    int result = PyObject_IsTrue(flag);
    Py_DECREF(flag);
    if (result < 0) {
        PyErr_Clear();
        return 0;
    }
    return result;
}

// ArrayType handler with Python-parity signature construction
static PyObject* sig_from_arraytype(PyObject *arg, PyObject *name,
                                     int *to_execute, Settings *settings) {
    *to_execute = 0;

    PyObject *shape = PyObject_GetAttr(arg, str_shape);
    if (!shape) return NULL;

    PyObject *dtype = PyObject_GetAttr(arg, str_dtype);
    if (!dtype) {
        Py_DECREF(shape);
        return NULL;
    }

    PyObject *numpy_dtype = signature_dtype_token(dtype);
    if (!numpy_dtype) {
        Py_DECREF(shape);
        Py_DECREF(dtype);
        return NULL;
    }

    int shape_is_unknown = (shape == UNKNOWN);
    int has_comptime = 0;
    PyObject *has_res = PyObject_CallMethodObjArgs(shape, str_has_comptime_undefined_dims, NULL);
    if (has_res) {
        has_comptime = PyObject_IsTrue(has_res);
        Py_DECREF(has_res);
    } else {
        PyErr_Clear();
    }

    PyObject *fortran_order = PyObject_GetAttr(shape, str_fortran_order);
    if (!fortran_order) {
        PyErr_Clear();
        Py_INCREF(Py_False);
        fortran_order = Py_False;
    }

    PyObject *sig = NULL;
    if (shape_is_unknown || (!settings->add_shape_descriptors && has_comptime)) {
        sig = PyTuple_Pack(4, name, numpy_dtype, Py_None, fortran_order);
    } else if (has_comptime) {
        PyObject *rank = PyObject_GetAttr(shape, str_rank);
        if (!rank) {
            Py_DECREF(shape);
            Py_DECREF(dtype);
            Py_DECREF(numpy_dtype);
            Py_DECREF(fortran_order);
            return NULL;
        }
        sig = PyTuple_Pack(4, name, numpy_dtype, rank, fortran_order);
        Py_DECREF(rank);
    } else {
        PyObject *rank = PyObject_GetAttr(shape, str_rank);
        PyObject *dims = PyObject_GetAttr(shape, str_dims);
        if (!rank || !dims) {
            Py_XDECREF(rank);
            Py_XDECREF(dims);
            Py_DECREF(shape);
            Py_DECREF(dtype);
            Py_DECREF(numpy_dtype);
            Py_DECREF(fortran_order);
            return NULL;
        }
        sig = PyTuple_Pack(6, name, numpy_dtype, rank, fortran_order, str_inout, dims);
        Py_DECREF(rank);
        Py_DECREF(dims);
    }

    Py_DECREF(shape);
    Py_DECREF(dtype);
    Py_DECREF(numpy_dtype);
    Py_DECREF(fortran_order);
    return sig;
}

// Optimized numpy scalar handler
static inline PyObject* sig_from_numpy_scalar(PyObject *arg, PyObject *name) {
    PyObject *dtype = PyObject_GetAttr(arg, str_dtype);
    if (!dtype) return NULL;
    
    PyObject *names = PyObject_GetAttr(dtype, str_names);
    if (!names) { Py_DECREF(dtype); return NULL; }
    
    PyObject *sig;
    if (names != Py_None) {
        // Structured dtype
        sig = PyTuple_Pack(3, name, dtype, PyLong_FromLong(0));
    } else {
        // Simple dtype
        sig = PyTuple_Pack(2, name, dtype);
    }
    Py_DECREF(names);
    return sig;
}

// Fast path for intent checking - cache the check
static inline PyObject* get_intent_str(PyObject *arg) {
    PyObject *target = arg;
    Py_INCREF(target);

    while (is_instance_fast(target, GetAttr) || is_instance_fast(target, GetItem)) {
        PyObject *next_target = PyObject_GetAttr(target, str_variable);
        Py_DECREF(target);
        if (!next_target) {
            PyErr_Clear();
            return str_in;
        }
        target = next_target;
    }

    if (!is_instance_fast(target, Variable)) {
        Py_DECREF(target);
        return str_in;
    }

    PyObject *intent_obj = PyObject_GetAttr(target, str_intent);
    Py_DECREF(target);

    if (!intent_obj) {
        PyErr_Clear();
        return str_inout;
    }

    PyObject *result = str_inout;
    int is_in = PyObject_RichCompareBool(intent_obj, str_in, Py_EQ);
    if (is_in == 1) {
        result = str_in;
    } else if (is_in < 0) {
        PyErr_Clear();
    }
    Py_DECREF(intent_obj);
    return result;
}

// Optimized ExpressionNode handler
static PyObject* sig_from_expression_node(PyObject *arg, PyObject *name,
                                          int *to_execute, Settings *settings) {
    *to_execute = 0;
    
    PyObject *dtype = PyObject_GetAttr(arg, str_dtype);
    if (!dtype) {
        return NULL;
    }
    
    PyObject *numpy_dtype = PyObject_CallMethodObjArgs(dtype, str_get_numpy, NULL);
    if (!numpy_dtype) { 
        Py_DECREF(dtype); 
        return NULL; 
    }
    
    PyObject *shape = PyObject_GetAttr(arg, str_underscore_shape);
    if (!shape) { 
        Py_DECREF(dtype); 
        Py_DECREF(numpy_dtype); 
        return NULL; 
    }
    
    int shape_is_scalar = (shape == SCALAR);
    int shape_is_unknown = (shape == UNKNOWN);
    PyObject *intent_str = get_intent_str(arg);
    
    int has_comptime_dims = 0;
    // Only check for comptime dims if not scalar or unknown
    if (!shape_is_scalar && !shape_is_unknown) {
        PyObject *res = PyObject_CallMethodObjArgs(shape, str_has_comptime_undefined_dims, NULL);
        if (res) { 
            has_comptime_dims = PyObject_IsTrue(res); 
            Py_DECREF(res); 
        } else {
            PyErr_Clear();
        }
    }
    
    PyObject *sig = NULL;
    
    if (shape_is_scalar) {
        if (dtype_is_vector(dtype)) {
            sig = PyTuple_Pack(2, name, numpy_dtype);
        } else if (intent_str == str_inout) {
            sig = PyTuple_Pack(5, name, numpy_dtype, PyLong_FromLong(0), 
                              Py_False, intent_str);
        } else {
            sig = PyTuple_Pack(2, name, numpy_dtype);
        }
    }
    else if (shape_is_unknown || (!settings->add_shape_descriptors && has_comptime_dims)) {
        sig = PyTuple_Pack(5, name, numpy_dtype, Py_None, Py_False, intent_str);
    }
    else if (has_comptime_dims) {
        PyObject *rank = PyObject_GetAttr(shape, str_rank);
        PyObject *fortran_order = PyObject_GetAttr(shape, str_fortran_order);
        sig = PyTuple_New(5);
        if (sig) {
            Py_INCREF(name);
            PyTuple_SET_ITEM(sig, 0, name);
            PyTuple_SET_ITEM(sig, 1, numpy_dtype);
            PyTuple_SET_ITEM(sig, 2, rank ? rank : Py_None);
            PyTuple_SET_ITEM(sig, 3, fortran_order ? fortran_order : Py_False);
            Py_INCREF(intent_str);
            PyTuple_SET_ITEM(sig, 4, intent_str);
        }
        Py_XDECREF(fortran_order);
    }
    else {
        if (!settings->ignore_fixed_shape_in_nested_calls) {
            PyObject *rank = PyObject_GetAttr(shape, str_rank);
            PyObject *fortran_order = PyObject_GetAttr(shape, str_fortran_order);
            PyObject *dims = PyObject_GetAttr(shape, str_dims);
            sig = PyTuple_New(6);
            if (sig) {
                Py_INCREF(name);
                PyTuple_SET_ITEM(sig, 0, name);
                PyTuple_SET_ITEM(sig, 1, numpy_dtype);
                PyTuple_SET_ITEM(sig, 2, rank ? rank : Py_None);
                PyTuple_SET_ITEM(sig, 3, fortran_order ? fortran_order : Py_False);
                Py_INCREF(intent_str);
                PyTuple_SET_ITEM(sig, 4, intent_str);
                PyTuple_SET_ITEM(sig, 5, dims ? dims : Py_None);
            }
            Py_XDECREF(fortran_order);
        } else {
            sig = PyTuple_Pack(5, name, numpy_dtype, Py_None, Py_False, intent_str);
        }
    }
    
    Py_DECREF(shape);
    Py_DECREF(dtype);
    return sig;
}

static PyObject* sig_from_pointertype(PyObject *arg, PyObject *name, int *to_execute) {
    *to_execute = 0;
    PyErr_SetString(PyExc_TypeError, "PointerType signatures require Python parser fallback");
    return NULL;
}

// Main signature extraction - optimized dispatcher
static PyObject* get_signature_from_arg(PyObject *arg, PyObject *name, 
                                         int *to_execute, Settings *settings) {
    // Fast paths in order of likelihood
    if (PyArray_Check(arg)) {
        return sig_from_ndarray(arg, name);
    }
    
    if (PyLong_Check(arg) || PyFloat_Check(arg) || PyComplex_Check(arg)) {
        return sig_from_scalar(arg, name);
    }
    
    if (is_instance_fast(arg, NumpyGeneric)) {
        return sig_from_numpy_scalar(arg, name);
    }
    
    if (is_instance_fast(arg, ArrayType)) {
        return sig_from_arraytype(arg, name, to_execute, settings);
    }

    if (is_instance_fast(arg, PointerType)) {
        return sig_from_pointertype(arg, name, to_execute);
    }
    
    if (is_instance_fast(arg, ExpressionNode)) {
        return sig_from_expression_node(arg, name, to_execute, settings);
    }
    
    // DataType class handling (e.g. nm.float64)
    if (PyType_Check(arg) && PyObject_IsSubclass(arg, DataType)) {
        *to_execute = 0;
        PyObject *numpy_dtype = signature_dtype_token(arg);
        if (!numpy_dtype) return NULL;
        PyObject *sig = PyTuple_Pack(2, name, numpy_dtype);
        Py_DECREF(numpy_dtype);
        return sig;
    }
    
    PyErr_Format(PyExc_ValueError, "Argument %R type not supported", name);
    return NULL;
}

// ============================================================================
// Core signature parsing helper
// ============================================================================
// Shared by get_signature_and_runtime_args() and fast_dispatch().
//
// Writes runtime args into runtime_args_buf[0..nruntime_out-1].
// Writes signature items into sig_list (a pre-allocated PyList).
// Returns 0 on success, -1 on error (with exception set).
// The caller owns the references placed in runtime_args_buf (borrowed+INCREF'd).

#define MAX_STACK_ARGS 64

typedef struct {
    PyObject *py_args;
    PyObject *kwargs;
    PyObject *params;
    PyObject *fixed_param_indices;
    int n_pos_def_args;
    const char *catch_var_name_str;
    Settings settings;
} ParseInput;

typedef struct {
    int to_execute;
    PyObject *sig_tuple;        // New reference (tuple)
    PyObject **runtime_args_buf; // C array of new refs (caller must DECREF)
    Py_ssize_t nruntime;
    // For heap-allocated buffer cleanup
    PyObject **heap_buf;        // Non-NULL if we had to heap-allocate
} ParseResult;

// caller_stack_buf: a stack-allocated buffer from the CALLER's frame (not ours)
// caller_stack_buf_size: its capacity
// If the number of args exceeds this, we heap-allocate internally.
static int _parse_signature_core(ParseInput *input, ParseResult *out,
                                  PyObject **caller_stack_buf,
                                  Py_ssize_t caller_stack_buf_size) {
    PyObject *py_args = input->py_args;
    PyObject *kwargs = input->kwargs;
    PyObject *params = input->params;
    PyObject *fixed_param_indices = input->fixed_param_indices;
    const char *catch_var_name_str = input->catch_var_name_str;
    Settings *settings = &input->settings;

    int to_execute = 1;

    Py_ssize_t args_len = PyTuple_Size(py_args);
    Py_ssize_t num_fixed = PyList_Size(fixed_param_indices);

    PyObject *unused_kwargs = PyDict_Copy(kwargs);
    if (!unused_kwargs) return -1;

    // Count positional args consumed by fixed params (first pass)
    Py_ssize_t pos_idx = 0;
    for (Py_ssize_t i = 0; i < num_fixed; i++) {
        PyObject *param_idx_obj = PyList_GetItem(fixed_param_indices, i);
        Py_ssize_t param_idx = PyLong_AsSsize_t(param_idx_obj);
        PyObject *param = PyList_GetItem(params, param_idx);
        PyObject *p_kind = PyObject_GetAttr(param, str_kind);
        long kind = PyLong_AsLong(p_kind);
        Py_DECREF(p_kind);

        if (kind == KIND_POSITIONAL_ONLY || kind == KIND_POSITIONAL_OR_KEYWORD) {
            if (pos_idx < args_len) {
                pos_idx++;
            }
        }
    }

    Py_ssize_t var_args_count = args_len - pos_idx;
    Py_ssize_t var_kwargs_count = PyDict_Size(unused_kwargs);
    Py_ssize_t total_sig_size = num_fixed + var_args_count + var_kwargs_count;

    // Allocate runtime_args buffer: use caller's stack buf if it fits, else heap
    PyObject **runtime_args_buf;
    PyObject **heap_buf = NULL;

    // total_sig_size is an upper bound on runtime args count
    if (total_sig_size <= caller_stack_buf_size) {
        runtime_args_buf = caller_stack_buf;
    } else {
        heap_buf = (PyObject **)PyMem_Malloc(total_sig_size * sizeof(PyObject *));
        if (!heap_buf) {
            Py_DECREF(unused_kwargs);
            PyErr_NoMemory();
            return -1;
        }
        runtime_args_buf = heap_buf;
    }
    Py_ssize_t nruntime = 0;

    // Signature list (pre-allocated)
    PyObject *signature = PyList_New(total_sig_size);
    if (!signature) {
        Py_DECREF(unused_kwargs);
        if (heap_buf) PyMem_Free(heap_buf);
        return -1;
    }

    // Second pass: actual processing
    pos_idx = 0;
    Py_ssize_t sig_idx = 0;

    for (Py_ssize_t i = 0; i < num_fixed; i++) {
        PyObject *param_idx_obj = PyList_GetItem(fixed_param_indices, i);
        Py_ssize_t param_idx = PyLong_AsSsize_t(param_idx_obj);
        PyObject *param = PyList_GetItem(params, param_idx);

        PyObject *p_kind = PyObject_GetAttr(param, str_kind);
        PyObject *p_name = PyObject_GetAttr(param, str_name);
        PyObject *p_default = PyObject_GetAttr(param, str_default);
        PyObject *p_is_comptime = PyObject_GetAttr(param, str_is_comptime);

        if (!p_kind || !p_name || !p_default || !p_is_comptime) {
            Py_XDECREF(p_kind); Py_XDECREF(p_name); Py_XDECREF(p_default); Py_XDECREF(p_is_comptime);
            goto error;
        }

        long kind = PyLong_AsLong(p_kind);
        int is_comptime = PyObject_IsTrue(p_is_comptime);
        PyObject *arg = NULL;

        if (kind == KIND_POSITIONAL_ONLY || kind == KIND_POSITIONAL_OR_KEYWORD) {
            if (pos_idx < args_len) {
                arg = PyTuple_GetItem(py_args, pos_idx);
                Py_INCREF(arg);
                pos_idx++;
            } else {
                arg = PyDict_GetItem(unused_kwargs, p_name);
                if (arg) {
                    Py_INCREF(arg);
                    PyDict_DelItem(unused_kwargs, p_name);
                } else if (p_default != INSPECT_EMPTY) {
                    arg = p_default;
                    Py_INCREF(arg);
                } else {
                    PyErr_Format(PyExc_ValueError, "Missing required argument: %S", p_name);
                    Py_DECREF(p_kind); Py_DECREF(p_name); Py_DECREF(p_default); Py_DECREF(p_is_comptime);
                    goto error;
                }
            }
        } else {
            arg = PyDict_GetItem(unused_kwargs, p_name);
            if (arg) {
                Py_INCREF(arg);
                PyDict_DelItem(unused_kwargs, p_name);
            } else if (p_default != INSPECT_EMPTY) {
                arg = p_default;
                Py_INCREF(arg);
            } else {
                PyErr_Format(PyExc_ValueError, "Missing required argument: %S", p_name);
                Py_DECREF(p_kind); Py_DECREF(p_name); Py_DECREF(p_default); Py_DECREF(p_is_comptime);
                goto error;
            }
        }

        if (is_comptime) {
            PyList_SET_ITEM(signature, sig_idx++, arg);
        } else {
            PyObject *sig_item = get_signature_from_arg(arg, p_name, &to_execute, settings);
            if (!sig_item) {
                Py_DECREF(arg);
                Py_DECREF(p_kind); Py_DECREF(p_name); Py_DECREF(p_default); Py_DECREF(p_is_comptime);
                goto error;
            }
            PyList_SET_ITEM(signature, sig_idx++, sig_item);
            // Store in C buffer (new ref — arg is already INCREF'd)
            runtime_args_buf[nruntime++] = arg;
            // Don't DECREF arg here — ownership transferred to buffer
            goto skip_decref_arg;
        }

skip_decref_arg:
        Py_DECREF(p_kind); Py_DECREF(p_name); Py_DECREF(p_default); Py_DECREF(p_is_comptime);
        continue;
    }

    // Catch *args
    if (pos_idx < args_len) {
        for (Py_ssize_t j = 0; pos_idx < args_len; pos_idx++, j++) {
            PyObject *arg = PyTuple_GetItem(py_args, pos_idx);
            PyObject *name_tuple = PyTuple_Pack(2, PyUnicode_FromString(catch_var_name_str),
                                               PyLong_FromLong(j));
            if (!name_tuple) goto error;

            PyObject *sig_item = get_signature_from_arg(arg, name_tuple, &to_execute, settings);
            Py_DECREF(name_tuple);
            if (!sig_item) goto error;

            PyList_SET_ITEM(signature, sig_idx++, sig_item);
            Py_INCREF(arg);  // We borrow from tuple, need our own ref
            runtime_args_buf[nruntime++] = arg;
        }
    }

    // Catch **kwargs
    PyObject *keys = PyDict_Keys(unused_kwargs);
    if (settings->reorder_kwargs) {
        PyList_Sort(keys);
    }

    Py_ssize_t num_keys = PyList_Size(keys);
    for (Py_ssize_t k = 0; k < num_keys; k++) {
        PyObject *key = PyList_GetItem(keys, k);
        PyObject *val = PyDict_GetItem(unused_kwargs, key);

        PyObject *sig_item = get_signature_from_arg(val, key, &to_execute, settings);
        if (!sig_item) {
            Py_DECREF(keys);
            goto error;
        }
        PyList_SET_ITEM(signature, sig_idx++, sig_item);
        Py_INCREF(val);
        runtime_args_buf[nruntime++] = val;
    }
    Py_DECREF(keys);
    Py_DECREF(unused_kwargs);

    // Trim signature list to actual size
    if (sig_idx < total_sig_size) {
        PyList_SetSlice(signature, sig_idx, total_sig_size, NULL);
    }

    PyObject *sig_tuple = PyList_AsTuple(signature);
    Py_DECREF(signature);
    if (!sig_tuple) {
        // Clean up runtime_args_buf refs
        for (Py_ssize_t i = 0; i < nruntime; i++) Py_DECREF(runtime_args_buf[i]);
        if (heap_buf) PyMem_Free(heap_buf);
        return -1;
    }

    out->to_execute = to_execute;
    out->sig_tuple = sig_tuple;
    out->runtime_args_buf = runtime_args_buf;
    out->nruntime = nruntime;
    out->heap_buf = heap_buf;
    return 0;

error:
    Py_DECREF(unused_kwargs);
    Py_DECREF(signature);
    // Clean up any runtime args we already stored
    for (Py_ssize_t i = 0; i < nruntime; i++) Py_DECREF(runtime_args_buf[i]);
    if (heap_buf) PyMem_Free(heap_buf);
    return -1;
}

// Helper to build a Python list from the runtime_args C buffer
static PyObject* _runtime_args_to_list(PyObject **buf, Py_ssize_t n) {
    PyObject *list = PyList_New(n);
    if (!list) return NULL;
    for (Py_ssize_t i = 0; i < n; i++) {
        Py_INCREF(buf[i]);
        PyList_SET_ITEM(list, i, buf[i]);
    }
    return list;
}

// Helper to clean up a ParseResult (DECREF runtime args + free heap buf)
static void _parse_result_cleanup(ParseResult *r) {
    for (Py_ssize_t i = 0; i < r->nruntime; i++) {
        Py_DECREF(r->runtime_args_buf[i]);
    }
    if (r->heap_buf) PyMem_Free(r->heap_buf);
}

// ============================================================================
// get_signature_and_runtime_args — original API, unchanged behavior
// ============================================================================
static PyObject* get_signature_and_runtime_args(PyObject *self, PyObject *args) {
    PyObject *py_args;
    PyObject *kwargs;
    PyObject *params;
    PyObject *fixed_param_indices;
    int n_pos_def_args;
    char *catch_var_name_str;
    int add_shape_descriptors;
    int ignore_fixed_shape;
    int reorder_kwargs;

    if (!PyArg_ParseTuple(args, "OOOOisipp", &py_args, &kwargs, &params, &fixed_param_indices,
                          &n_pos_def_args, &catch_var_name_str,
                          &add_shape_descriptors, &ignore_fixed_shape, &reorder_kwargs)) {
        return NULL;
    }

    ParseInput input = {
        py_args, kwargs, params, fixed_param_indices,
        n_pos_def_args, catch_var_name_str,
        {add_shape_descriptors, ignore_fixed_shape, reorder_kwargs}
    };
    ParseResult result;
    PyObject *stack_buf[MAX_STACK_ARGS];

    if (_parse_signature_core(&input, &result, stack_buf, MAX_STACK_ARGS) < 0) {
        return NULL;
    }

    // Build the Python list from the C buffer
    PyObject *runtime_args_list = _runtime_args_to_list(result.runtime_args_buf, result.nruntime);
    _parse_result_cleanup(&result);

    if (!runtime_args_list) {
        Py_DECREF(result.sig_tuple);
        return NULL;
    }

    PyObject *ret = PyTuple_Pack(3, result.to_execute ? Py_True : Py_False,
                                 result.sig_tuple, runtime_args_list);
    Py_DECREF(result.sig_tuple);
    Py_DECREF(runtime_args_list);
    return ret;
}

// ============================================================================
// fast_dispatch — signature parse + dict lookup + Vectorcall in C
// ============================================================================
// Returns a 4-tuple: (hit, to_execute, payload, runtime_args_or_None)
//   hit=True,  to_execute=True  => payload is the actual function result
//   hit=False, to_execute=True  => payload is signature tuple (cache miss, Python must load)
//   hit=False, to_execute=False => payload is signature tuple (symbolic, Python handles)
static PyObject* fast_dispatch(PyObject *self, PyObject *args) {
    PyObject *py_args;
    PyObject *kwargs;
    PyObject *params;
    PyObject *fixed_param_indices;
    int n_pos_def_args;
    char *catch_var_name_str;
    int add_shape_descriptors;
    int ignore_fixed_shape;
    int reorder_kwargs;
    PyObject *fast_call_dict;

    if (!PyArg_ParseTuple(args, "OOOOisippO", &py_args, &kwargs, &params, &fixed_param_indices,
                          &n_pos_def_args, &catch_var_name_str,
                          &add_shape_descriptors, &ignore_fixed_shape, &reorder_kwargs,
                          &fast_call_dict)) {
        return NULL;
    }

    ParseInput input = {
        py_args, kwargs, params, fixed_param_indices,
        n_pos_def_args, catch_var_name_str,
        {add_shape_descriptors, ignore_fixed_shape, reorder_kwargs}
    };
    ParseResult result;
    PyObject *stack_buf[MAX_STACK_ARGS];

    if (_parse_signature_core(&input, &result, stack_buf, MAX_STACK_ARGS) < 0) {
        return NULL;
    }

    // --- Not executable (symbolic) → return to Python ---
    if (!result.to_execute) {
        PyObject *runtime_args_list = _runtime_args_to_list(result.runtime_args_buf, result.nruntime);
        _parse_result_cleanup(&result);
        if (!runtime_args_list) {
            Py_DECREF(result.sig_tuple);
            return NULL;
        }
        PyObject *ret = PyTuple_Pack(4, Py_False, Py_False,
                                     result.sig_tuple, runtime_args_list);
        Py_DECREF(result.sig_tuple);
        Py_DECREF(runtime_args_list);
        return ret;
    }

    // --- Executable: try cache lookup ---
    // PyDict_GetItem returns borrowed ref, does NOT set exception on miss
    PyObject *func = PyDict_GetItem(fast_call_dict, result.sig_tuple);

    if (func == NULL) {
        // Cache miss → return to Python for compilation + loading
        PyObject *runtime_args_list = _runtime_args_to_list(result.runtime_args_buf, result.nruntime);
        _parse_result_cleanup(&result);
        if (!runtime_args_list) {
            Py_DECREF(result.sig_tuple);
            return NULL;
        }
        PyObject *ret = PyTuple_Pack(4, Py_False, Py_True,
                                     result.sig_tuple, runtime_args_list);
        Py_DECREF(result.sig_tuple);
        Py_DECREF(runtime_args_list);
        return ret;
    }

    // --- Cache hit! Call via Vectorcall ---
    PyObject *call_result = PyObject_Vectorcall(
        func,
        result.runtime_args_buf,
        (size_t)result.nruntime,
        NULL  // no kwnames
    );

    // Clean up
    _parse_result_cleanup(&result);

    if (!call_result) {
        // Function raised an exception
        Py_DECREF(result.sig_tuple);
        return NULL;
    }

    PyObject *ret = PyTuple_Pack(4, Py_True, Py_True,
                                 call_result, Py_None);
    Py_DECREF(result.sig_tuple);
    Py_DECREF(call_result);
    return ret;
}

// ============================================================================
// BaseFunction Type - The optimized base class for NumetaFunction
// ============================================================================

typedef struct BaseFunctionObject {
    PyObject_HEAD
    PyObject *params;              // List of params
    PyObject *fixed_param_indices; // List of indices
    int n_pos_def_args;
    PyObject *catch_var_name_obj;  // PyUnicode or Py_None
    const char *catch_var_name_cstr; // Pointer to internal buffer of above (or NULL)
    PyObject *fast_call_dict;      // Dict[Signature, CompiledFunc]
    PyObject *BuilderHelperCls;    // BuilderHelper class for symbolic check
    Settings settings;
    SignatureParserFunc parser_func;  // Custom generated parser (NULL = use generic)
} BaseFunctionObject;

static void BaseFunction_dealloc(BaseFunctionObject *self) {
    Py_XDECREF(self->params);
    Py_XDECREF(self->fixed_param_indices);
    Py_XDECREF(self->catch_var_name_obj);
    Py_XDECREF(self->fast_call_dict);
    Py_XDECREF(self->BuilderHelperCls);
    // Note: dlclose is tricky with Python - we'll skip explicit close
    // The OS will clean up when the process exits
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *BaseFunction_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    BaseFunctionObject *self = (BaseFunctionObject *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->params = NULL;
        self->fixed_param_indices = NULL;
        self->n_pos_def_args = 0;
        self->catch_var_name_obj = NULL;
        self->catch_var_name_cstr = NULL;
        self->fast_call_dict = NULL;
        self->BuilderHelperCls = NULL;
        self->parser_func = NULL;
        // Default settings
        self->settings.add_shape_descriptors = 0;
        self->settings.ignore_fixed_shape_in_nested_calls = 0;
        self->settings.reorder_kwargs = 0;
        self->settings.use_c_dispatch = 1;
    }
    return (PyObject *)self;
}

// _configure_dispatch(params, fixed_indices, n_pos, catch_name, fast_call_dict, builder_helper, settings...)
static PyObject *BaseFunction_configure_dispatch(BaseFunctionObject *self, PyObject *args) {
    PyObject *params;
    PyObject *fixed_indices;
    int n_pos;
    PyObject *catch_name;
    PyObject *fast_call_dict;
    PyObject *builder_helper;
    int add_shape, ignore_fixed, reorder, use_c_dispatch;

    if (!PyArg_ParseTuple(args, "OOiOOOiiii", 
        &params, &fixed_indices, &n_pos, &catch_name, &fast_call_dict, &builder_helper,
        &add_shape, &ignore_fixed, &reorder, &use_c_dispatch)) {
        return NULL;
    }

    Py_XINCREF(params);
    Py_XDECREF(self->params);
    self->params = params;

    Py_XINCREF(fixed_indices);
    Py_XDECREF(self->fixed_param_indices);
    self->fixed_param_indices = fixed_indices;

    self->n_pos_def_args = n_pos;

    Py_XINCREF(catch_name);
    Py_XDECREF(self->catch_var_name_obj);
    self->catch_var_name_obj = catch_name;

    if (catch_name == Py_None) {
        self->catch_var_name_cstr = NULL;
    } else if (PyUnicode_Check(catch_name)) {
        self->catch_var_name_cstr = PyUnicode_AsUTF8(catch_name);
        if (!self->catch_var_name_cstr) return NULL;
    } else {
        PyErr_SetString(PyExc_TypeError, "catch_var_name must be str or None");
        return NULL;
    }

    Py_XINCREF(fast_call_dict);
    Py_XDECREF(self->fast_call_dict);
    self->fast_call_dict = fast_call_dict;

    Py_XINCREF(builder_helper);
    Py_XDECREF(self->BuilderHelperCls);
    self->BuilderHelperCls = builder_helper;

    self->settings.add_shape_descriptors = add_shape;
    self->settings.ignore_fixed_shape_in_nested_calls = ignore_fixed;
    self->settings.reorder_kwargs = reorder;
    self->settings.use_c_dispatch = use_c_dispatch;

    Py_RETURN_NONE;
}

static int should_delegate_symbolic_arg(PyObject *arg) {
    if (is_instance_fast(arg, ArrayType)) return 1;
    if (is_instance_fast(arg, PointerType)) return 1;
    if (is_instance_fast(arg, ExpressionNode)) return 1;
    if (PyType_Check(arg)) {
        int is_dtype = PyObject_IsSubclass(arg, DataType);
        if (is_dtype == 1) return 1;
        if (is_dtype < 0) PyErr_Clear();
    }
    return 0;
}

static int should_delegate_symbolic_call(PyObject *args, PyObject *kwargs) {
    Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    for (Py_ssize_t i = 0; i < nargs; ++i) {
        if (should_delegate_symbolic_arg(PyTuple_GET_ITEM(args, i))) return 1;
    }
    if (kwargs && PyDict_Check(kwargs)) {
        PyObject *key;
        PyObject *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if (should_delegate_symbolic_arg(value)) return 1;
        }
    }
    return 0;
}

static PyObject *call_python_fallback(BaseFunctionObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *method = PyObject_GetAttrString((PyObject*)self, "_python_call");
    if (!method) return NULL;
    PyObject *result = PyObject_Call(method, args, kwargs);
    Py_DECREF(method);
    return result;
}

static PyObject *BaseFunction_call(BaseFunctionObject *self, PyObject *args, PyObject *kwargs) {
    // Check if configured
    if (!self->params || !self->fast_call_dict) {
        PyErr_SetString(PyExc_RuntimeError, "BaseFunction dispatch not configured");
        return NULL;
    }

    // Check if C dispatch is disabled via settings (fallback to Python implementation)
    if (!self->settings.use_c_dispatch) {
        return call_python_fallback(self, args, kwargs);
    }
    
    // Default kwargs to empty dict if NULL (call from python with no kwargs)
    PyObject *tmp_kwargs = NULL;
    if (kwargs == NULL) {
        tmp_kwargs = PyDict_New();
        if (!tmp_kwargs) return NULL;
        kwargs = tmp_kwargs;
    }

    if (should_delegate_symbolic_call(args, kwargs)) {
        PyObject *result = call_python_fallback(self, args, kwargs);
        Py_XDECREF(tmp_kwargs);
        return result;
    }

    ParseInput input = {
        args, kwargs, 
        self->params, 
        self->fixed_param_indices,
        self->n_pos_def_args, 
        self->catch_var_name_cstr,
        self->settings
    };

    ParseResult result;
    PyObject *stack_buf[MAX_STACK_ARGS];
    PyObject *ret_val = NULL;
    PyObject *func = NULL;

    // --- Check Symbolic Context (BuilderHelper.current_builder) ---
    int force_symbolic = 0;
    if (self->BuilderHelperCls && self->BuilderHelperCls != Py_None) {
        PyObject *builder = PyObject_GetAttr(self->BuilderHelperCls, str_current_builder);
        if (builder) {
            if (builder != Py_None) force_symbolic = 1;
            Py_DECREF(builder);
        } else {
            PyErr_Clear();
        }
    }

    // --- Try Custom Parser First (Fast Path) ---
    // If we have a custom parser, no kwargs, and not in symbolic mode, use it
    Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    if (self->parser_func && !force_symbolic && (!kwargs || PyDict_Size(kwargs) == 0)) {
        
        // Use separate buffers for runtime_args and sig
        PyObject *sig_buf[MAX_STACK_ARGS] = {NULL};
        int parsed_nargs = 0;
        int parsed_nruntime = 0;
        int parser_result = self->parser_func(args, stack_buf, sig_buf, &parsed_nargs, &parsed_nruntime, self);
        
        if (parser_result == 0) {
            // Parser succeeded
            PyObject **runtime_args = stack_buf;
            PyObject **sig = sig_buf;
            
            // Build signature tuple
            PyObject *sig_tuple = PyTuple_New(parsed_nargs);
            if (!sig_tuple) {
                Py_XDECREF(tmp_kwargs);
                return NULL;
            }
            for (int i = 0; i < parsed_nargs; i++) {
                PyTuple_SET_ITEM(sig_tuple, i, sig[i]);  // Steals reference
            }
            
            // Lookup compiled function
            func = PyDict_GetItem(self->fast_call_dict, sig_tuple);
            
            if (func) {
                // Cache hit - call directly
                Py_DECREF(sig_tuple);  // We don't need this anymore
                Py_XDECREF(tmp_kwargs);
                return PyObject_Vectorcall(func, runtime_args, (size_t)parsed_nruntime, NULL);
            }
            
            // Cache miss - fall through to slow path with the parsed data
            // We need to construct runtime_args_list from runtime_args
            PyObject *runtime_args_list = PyList_New(parsed_nruntime);
            if (!runtime_args_list) {
                Py_DECREF(sig_tuple);
                Py_XDECREF(tmp_kwargs);
                return NULL;
            }
            for (int i = 0; i < parsed_nruntime; i++) {
                Py_INCREF(runtime_args[i]);
                PyList_SET_ITEM(runtime_args_list, i, runtime_args[i]);
            }
            
            Py_XDECREF(tmp_kwargs);
            ret_val = PyObject_CallMethod((PyObject*)self, "_handle_cache_miss", "OO", 
                                          sig_tuple, runtime_args_list);
            
            Py_DECREF(sig_tuple);
            Py_DECREF(runtime_args_list);
            return ret_val;
        }

        // Parser failed: release any partially-constructed signature entries,
        // clear parser error state, and fall back to the generic parser.
        for (int i = 0; i < MAX_STACK_ARGS; i++) {
            Py_XDECREF(sig_buf[i]);
        }
        if (PyErr_Occurred()) {
            PyErr_Clear();
        }
    }

    // --- Slow Path: Generic Parser ---
    if (_parse_signature_core(&input, &result, stack_buf, MAX_STACK_ARGS) < 0) {
        Py_XDECREF(tmp_kwargs);
        if (PyErr_Occurred()) {
            PyErr_Clear();
        }
        PyObject *method = PyObject_GetAttrString((PyObject*)self, "_python_call");
        if (!method) return NULL;
        ret_val = PyObject_Call(method, args, kwargs);
        Py_DECREF(method);
        return ret_val;
    }

    // Clean up temporary kwargs if we created it
    Py_XDECREF(tmp_kwargs);

    // --- Not executable (symbolic) OR Forced Symbolic ---
    if (!result.to_execute || force_symbolic) {
        _parse_result_cleanup(&result);
        Py_DECREF(result.sig_tuple);

        PyObject *method = PyObject_GetAttrString((PyObject*)self, "_python_call");
        if (!method) return NULL;
        ret_val = PyObject_Call(method, args, kwargs);
        Py_DECREF(method);
        return ret_val;
    }

    // --- Executable: Cache Lookup ---
    func = PyDict_GetItem(self->fast_call_dict, result.sig_tuple);

    if (func == NULL) {
        // Cache miss -> Compilation required
        PyObject *runtime_args_list = _runtime_args_to_list(result.runtime_args_buf, result.nruntime);
        _parse_result_cleanup(&result);
        if (!runtime_args_list) {
            Py_DECREF(result.sig_tuple);
            return NULL;
        }

        // Callback: self._handle_cache_miss(signature, runtime_args)
        ret_val = PyObject_CallMethod((PyObject*)self, "_handle_cache_miss", "OO", 
                                      result.sig_tuple, runtime_args_list);
        
        Py_DECREF(result.sig_tuple);
        Py_DECREF(runtime_args_list);
        return ret_val;
    }

    // --- Cache Hit: Vectorcall ---
    ret_val = PyObject_Vectorcall(
        func,
        result.runtime_args_buf,
        (size_t)result.nruntime,
        NULL
    );

    _parse_result_cleanup(&result);
    Py_DECREF(result.sig_tuple);
    
    return ret_val;
}

// _set_custom_parser(lib_path, func_name) - Load custom parser from compiled .so
static PyObject *BaseFunction_set_custom_parser(BaseFunctionObject *self, PyObject *args) {
    const char *lib_path;
    const char *func_name;
    
    if (!PyArg_ParseTuple(args, "ss", &lib_path, &func_name)) {
        return NULL;
    }
    
    // Load the library with RTLD_GLOBAL so it can see symbols from _signature.so
    // Note: dlclose is tricky with Python - we don't store the handle
    // The OS will clean up when the process exits
    void *handle = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        PyErr_Format(PyExc_RuntimeError, "Failed to load parser library: %s", dlerror());
        return NULL;
    }
    
    // Get the function pointer
    SignatureParserFunc parser = (SignatureParserFunc)dlsym(handle, func_name);
    if (!parser) {
        dlclose(handle);
        PyErr_Format(PyExc_RuntimeError, "Failed to find parser function '%s': %s", func_name, dlerror());
        return NULL;
    }
    
    // Store the parser function (we don't store the handle - OS cleans up on exit)
    self->parser_func = parser;
    
    Py_RETURN_NONE;
}

static PyMethodDef BaseFunction_methods[] = {
    {"_configure_dispatch", (PyCFunction)BaseFunction_configure_dispatch, METH_VARARGS,
     "Configure the C-level dispatch state"},
    {"_set_custom_parser", (PyCFunction)BaseFunction_set_custom_parser, METH_VARARGS,
     "Load and set a custom signature parser from a compiled shared library"},
    {NULL}  /* Sentinel */
};

static PyTypeObject BaseFunctionType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numeta._signature.BaseFunction",
    .tp_doc = "Optimized dispatch base class",
    .tp_basicsize = sizeof(BaseFunctionObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = BaseFunction_new,
    .tp_dealloc = (destructor)BaseFunction_dealloc,
    .tp_call = (ternaryfunc)BaseFunction_call,
    .tp_methods = BaseFunction_methods,
};

static PyObject *init_globals(PyObject *self, PyObject *args) {
    PyObject *types_dict;
    PyObject *constants_dict;

    if (!PyArg_ParseTuple(args, "OO", &types_dict, &constants_dict)) {
        return NULL;
    }

    #define LOAD_TYPE(name) \
        name = PyDict_GetItemString(types_dict, #name); \
        Py_XINCREF(name);

    LOAD_TYPE(ArrayType);
    LOAD_TYPE(PointerType);
    LOAD_TYPE(DataType);
    LOAD_TYPE(ExpressionNode);
    LOAD_TYPE(Variable);
    LOAD_TYPE(GetAttr);
    LOAD_TYPE(GetItem);
    LOAD_TYPE(SCALAR);
    LOAD_TYPE(UNKNOWN);
    LOAD_TYPE(NumpyGeneric);
    #undef LOAD_TYPE

    PyObject *tmp;
    #define LOAD_CONST(name) \
        tmp = PyDict_GetItemString(constants_dict, #name); \
        if (tmp) name = PyLong_AsLong(tmp);

    LOAD_CONST(KIND_POSITIONAL_ONLY);
    LOAD_CONST(KIND_POSITIONAL_OR_KEYWORD);
    LOAD_CONST(KIND_VAR_POSITIONAL);
    LOAD_CONST(KIND_KEYWORD_ONLY);
    LOAD_CONST(KIND_VAR_KEYWORD);
    #undef LOAD_CONST

    INSPECT_EMPTY = PyDict_GetItemString(constants_dict, "INSPECT_EMPTY");
    Py_XINCREF(INSPECT_EMPTY);

    // Init interned strings
    str_kind = PyUnicode_InternFromString("kind");
    str_name = PyUnicode_InternFromString("name");
    str_default = PyUnicode_InternFromString("default");
    str_is_comptime = PyUnicode_InternFromString("is_comptime");
    str_dtype = PyUnicode_InternFromString("dtype");
    str_names = PyUnicode_InternFromString("names");
    str_shape = PyUnicode_InternFromString("shape");
    str_get_numpy = PyUnicode_InternFromString("get_numpy");
    str_has_comptime_undefined_dims = PyUnicode_InternFromString("has_comptime_undefined_dims");
    str_fortran_order = PyUnicode_InternFromString("fortran_order");
    str_rank = PyUnicode_InternFromString("rank");
    str_dims = PyUnicode_InternFromString("dims");
    str_variable = PyUnicode_InternFromString("variable");
    str_intent = PyUnicode_InternFromString("intent");
    str_in = PyUnicode_InternFromString("in");
    str_inout = PyUnicode_InternFromString("inout");
    str_underscore_shape = PyUnicode_InternFromString("_shape");
    str_current_builder = PyUnicode_InternFromString("current_builder");

    Py_RETURN_NONE;
}

static PyMethodDef SignatureMethods[] = {
    {"get_signature_and_runtime_args", get_signature_and_runtime_args, METH_VARARGS, "Optimize signature parsing"},
    {"fast_dispatch", fast_dispatch, METH_VARARGS, "Parse signature + dict lookup + vectorcall dispatch"},
    {"init_globals", init_globals, METH_VARARGS, "Initialize globals"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef signaturemodule = {
    PyModuleDef_HEAD_INIT,
    "_signature",
    NULL,
    -1,
    SignatureMethods
};

PyMODINIT_FUNC PyInit__signature(void) {
    import_array();
    
    if (PyType_Ready(&BaseFunctionType) < 0)
        return NULL;

    PyObject *m = PyModule_Create(&signaturemodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&BaseFunctionType);
    if (PyModule_AddObject(m, "BaseFunction", (PyObject *)&BaseFunctionType) < 0) {
        Py_DECREF(&BaseFunctionType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
