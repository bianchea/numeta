import numeta as nm
import numpy as np


def _snapshot_settings():
    return (
        nm.settings.default_backend,
        nm.settings.default_do_checks,
        nm.settings.default_compile_flags,
    )


def _restore_settings(snapshot):
    backend, do_checks, compile_flags = snapshot
    nm.settings.set_default_backend(backend)
    nm.settings.set_default_do_checks(do_checks)
    nm.settings.set_default_compile_flags(compile_flags)


def test_jit_defaults_from_settings():
    snapshot = _snapshot_settings()
    try:
        nm.settings.set_default_backend("c")
        nm.settings.set_default_do_checks(False)
        nm.settings.set_default_compile_flags("-O2 -march=native")

        @nm.jit(backend=None, do_checks=None, compile_flags=None)
        def add_one(a):
            a[0] += 1

        arr = np.zeros(4, dtype=np.float64)
        add_one(arr)

        signature = add_one.get_signature(arr)
        compiled = add_one._compiled_functions[signature]
        assert compiled.backend == "c"
        assert add_one._pyc_extensions[signature].do_checks is False
        assert add_one.compile_flags == ["-O2", "-march=native"]
    finally:
        _restore_settings(snapshot)


def test_compile_flags_parsing_uses_shlex():
    snapshot = _snapshot_settings()
    try:
        nm.settings.set_default_compile_flags('-O2 -DTEST="hello world"')

        @nm.jit
        def add_one(a):
            a[0] += 1

        assert add_one.compile_flags == ["-O2", "-DTEST=hello world"]
    finally:
        _restore_settings(snapshot)


def test_compiler_setting_precedes_environment(monkeypatch):
    original_c = nm.settings._Settings__c_compiler
    original_fortran = nm.settings._Settings__fortran_compiler
    try:
        nm.settings.set_c_compiler(None)
        nm.settings.set_fortran_compiler(None)
        monkeypatch.setenv("NUMETA_CC", "env-cc")
        monkeypatch.setenv("NUMETA_FC", "env-fc")
        assert nm.settings.c_compiler == "env-cc"
        assert nm.settings.fortran_compiler == "env-fc"

        nm.settings.set_c_compiler("explicit-cc")
        nm.settings.set_fortran_compiler("explicit-fc")
        assert nm.settings.c_compiler == "explicit-cc"
        assert nm.settings.fortran_compiler == "explicit-fc"
    finally:
        nm.settings.set_c_compiler(original_c)
        nm.settings.set_fortran_compiler(original_fortran)
