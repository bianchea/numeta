import numpy as np
import os
import shutil
import sys
import tempfile

from pathlib import Path
import pickle
import warnings
from types import MappingProxyType
from typing import Iterable

from .numeta_function import NumetaFunction, NumetaCompiledFunction
from .native_name_registry import native_name_registry
from .pyc_extension import PyCExtension
from .compiler import Compiler
from .settings import settings
from .datatype import DataTypeMeta, make_struct_type


def _pickleable_or_none(value, *, description: str):
    if value is None:
        return None
    try:
        pickle.dumps(value)
    except Exception:
        warnings.warn(
            f"{description} is not pickleable; pass it again with reattach=True "
            "before compiling new signatures from the loaded library.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    return value


def _rebuild_struct_type(np_dtype, members, name):
    members = list(members)

    existing = DataTypeMeta._np_dtype.get(np_dtype)
    if existing is not None:
        return existing

    existing = DataTypeMeta._np_dtype.get(tuple(members))
    if existing is not None:
        return existing

    return make_struct_type(np_dtype, members, name=name)


def _artifact_dir_for_compiled(directory: Path, compiled: NumetaCompiledFunction) -> Path:
    return directory / "artifacts" / "compiled" / compiled.func_name


def _source_suffix_for(compiled: NumetaCompiledFunction) -> str:
    if compiled.backend == "fortran":
        return "_src.f90"
    if compiled.backend == "c":
        return "_src.c"
    raise ValueError(f"Unsupported backend: {compiled.backend}")


def _source_path_for(compiled: NumetaCompiledFunction) -> Path:
    return Path(compiled._path) / f"{compiled.func_name}{_source_suffix_for(compiled)}"


def _copy_if_different(source: Path, target: Path) -> None:
    if source.absolute() == target.absolute():
        return
    shutil.copy2(source, target)


def _persist_compiled_artifacts(
    compiled: NumetaCompiledFunction,
    directory: Path,
) -> tuple[Path, Path | None, Path]:
    old_obj_file = compiled.obj_files[0]

    target_dir = _artifact_dir_for_compiled(directory, compiled)
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_obj_file = target_dir / old_obj_file.name
    _copy_if_different(old_obj_file, saved_obj_file)

    saved_src_file = None
    source_paths = []
    for source_file in getattr(compiled, "_source_files", ()):
        source_path = Path(source_file)
        if source_path.exists():
            source_paths.append(source_path)

    generated_source = _source_path_for(compiled)
    if generated_source.exists() and generated_source not in source_paths:
        source_paths.append(generated_source)

    for source_path in source_paths:
        target_source = target_dir / source_path.name
        _copy_if_different(source_path, target_source)
        if saved_src_file is None and source_path.name.endswith(_source_suffix_for(compiled)):
            saved_src_file = target_source

    for side_product in Path(compiled._path).glob("*.mod"):
        _copy_if_different(side_product, target_dir / side_product.name)

    return saved_obj_file, saved_src_file, target_dir


def _validate_function_level_compatibility(
    old_func: NumetaFunction,
    new_func: NumetaFunction,
) -> None:
    if old_func.backend != new_func.backend:
        raise ValueError("Cannot replace function with different backend")

    if tuple(old_func.compile_flags) != tuple(new_func.compile_flags):
        raise ValueError(
            "Cannot replace function with different compile_flags in minimal incremental mode"
        )

    if old_func.do_checks != new_func.do_checks:
        raise ValueError(
            "Cannot replace function with different do_checks in minimal incremental mode"
        )

    if old_func.inline or new_func.inline:
        raise ValueError("replace() does not support inline functions yet")


def _validate_specialization_compatibility(
    old_func: NumetaFunction,
    new_func: NumetaFunction,
    signature,
) -> None:
    old_spec = old_func._wrapper_specs.get(signature)
    if old_spec is None:
        old_spec = old_func.build_wrapper_spec(signature)

    new_spec = new_func._wrapper_specs.get(signature)
    if new_spec is None:
        new_spec = new_func.build_wrapper_spec(signature)

    old_name, old_args, old_returns = old_spec
    new_name, new_args, new_returns = new_spec

    if old_name != new_name:
        raise AssertionError("replacement did not preserve compiled symbol name")

    if old_args != new_args:
        raise ValueError(
            f"Replacement for {old_func.name!r} changed argument ABI for signature {signature!r}"
        )

    if old_returns != new_returns:
        raise ValueError(
            f"Replacement for {old_func.name!r} changed return ABI for signature {signature!r}"
        )


class NumetaLibrary:
    __slots__ = "name", "_entries", "_global_entries"
    loaded = set()
    _reserved_names = {
        "name",
        "functions",
        "loaded",
        "register",
        "remove",
        "replace",
        "list_functions",
        "signatures",
        "compiled_symbols",
        "signature_for_call",
        "compiled_function",
        "save",
        "load",
        "print_f90_files",
        "__getitem__",
        "__contains__",
        "__iter__",
        "__len__",
        "__repr__",
    }

    def __init__(self, name: str | None = None) -> None:
        if name is not None:
            self._nm_validate_name(name)
        self.name = name
        self._entries: dict = {}
        self._global_entries: dict = {}

    @classmethod
    def _nm_validate_name(cls, name: str) -> None:
        if name == PyCExtension.SUFFIX or name.endswith(PyCExtension.SUFFIX):
            raise ValueError(
                f"Library name '{name}' is reserved because it ends with {PyCExtension.SUFFIX}."
            )
        if name in cls.loaded:
            raise ValueError(f"Already using a library called {name}")
        if native_name_registry.is_reserved(name):
            raise ValueError(
                f"Library name '{name}' conflicts with a compiled function library name."
            )
        wrapper_module = f"{name}{PyCExtension.SUFFIX}"
        if wrapper_module in sys.modules:
            raise ValueError(
                f"Library name '{name}' conflicts with an existing Numeta wrapper module."
            )

    def register(self, function: NumetaFunction) -> NumetaFunction:
        if not isinstance(function, NumetaFunction):
            raise TypeError("register expects a NumetaFunction")
        if function.name in self._reserved_names:
            raise ValueError(
                f"Function name '{function.name}' is reserved by NumetaLibrary methods"
            )
        existing = self._entries.get(function.name)
        if existing is not None and existing is not function:
            raise ValueError(f"Function '{function.name}' already registered")
        self._entries[function.name] = function
        return function

    def remove(self, name: str) -> NumetaFunction:
        return self._entries.pop(name)

    def replace(
        self,
        name_or_function: str | NumetaFunction,
        function: NumetaFunction | None = None,
        *,
        compile_now: bool = True,
        require_existing_specializations: bool = True,
        signatures: Iterable | None = None,
        non_selected: str = "preserve",
    ) -> NumetaFunction:
        valid_non_selected = {"preserve", "invalidate", "error"}
        if non_selected not in valid_non_selected:
            raise ValueError(
                "non_selected must be one of "
                f"{', '.join(sorted(repr(policy) for policy in valid_non_selected))}"
            )

        if function is None:
            if not isinstance(name_or_function, NumetaFunction):
                raise TypeError("replace(function) expects a NumetaFunction")
            name = name_or_function.name
            new_func = name_or_function
        else:
            if not isinstance(name_or_function, str):
                raise TypeError("replace(name, function) expects name to be a string")
            name = name_or_function
            new_func = function

        if not isinstance(new_func, NumetaFunction):
            raise TypeError("replacement must be a NumetaFunction")

        if name not in self._entries:
            raise KeyError(f"Cannot replace unknown function {name!r}")

        old_func = self._entries[name]
        if require_existing_specializations and not old_func._compiled_functions:
            raise ValueError(
                f"Function {name!r} has no compiled specializations to replace. "
                "Register the new function normally or call replace(..., "
                "require_existing_specializations=False)."
            )

        if new_func is old_func:
            raise ValueError(
                f"replace() requires a distinct replacement function, "
                f"not the same object as the library entry"
            )

        if new_func._compiled_functions:
            raise ValueError(
                "Replacement function already has compiled specializations. "
                "Call replacement.clear() before lib.replace(...)."
            )

        _validate_function_level_compatibility(old_func, new_func)

        old_signatures = tuple(old_func._compiled_functions)
        if signatures is None:
            selected_signatures = old_signatures
            preserve_unselected = False
        else:
            selected_signatures = tuple(dict.fromkeys(signatures))
            if not selected_signatures:
                raise ValueError("signatures must contain at least one specialization")

            missing_signatures = [
                signature
                for signature in selected_signatures
                if signature not in old_func._compiled_functions
            ]
            if missing_signatures:
                raise KeyError(
                    f"Cannot replace unknown specialization(s) for {name!r}: "
                    f"{missing_signatures!r}"
                )

            unselected_signatures = [
                signature for signature in old_signatures if signature not in selected_signatures
            ]
            if non_selected == "error" and unselected_signatures:
                raise ValueError(
                    f"replace(..., non_selected='error') received {len(unselected_signatures)} "
                    f"unselected specialization(s) for {name!r}"
                )
            preserve_unselected = non_selected == "preserve"

        original_state = {
            "name": new_func.name,
            "return_signatures": new_func.return_signatures.copy(),
            "_compiled_functions": new_func._compiled_functions.copy(),
            "_wrapper_specs": new_func._wrapper_specs.copy(),
            "_pyc_extensions": new_func._pyc_extensions.copy(),
            "_library_pyc_extension": new_func._library_pyc_extension,
            "_fast_call": new_func._fast_call.copy(),
        }
        names_added_by_replace = []

        try:
            for signature in selected_signatures:
                old_compiled = old_func._compiled_functions[signature]
                old_symbol = old_compiled.func_name
                if not native_name_registry.is_reserved(old_symbol):
                    names_added_by_replace.append(old_symbol)

                new_func._wrapper_specs.pop(signature, None)
                new_func._pyc_extensions.pop(signature, None)
                new_func._fast_call.pop(signature, None)
                new_func.construct_compiled_target(
                    signature,
                    forced_name=old_symbol,
                    allow_existing_name=True,
                )
                new_func.construct_wrapper_spec(signature)
                _validate_specialization_compatibility(old_func, new_func, signature)

                if compile_now:
                    new_func._compiled_functions[signature].compile_obj()

            if preserve_unselected:
                for signature in old_signatures:
                    if signature in selected_signatures:
                        continue
                    new_func._compiled_functions[signature] = old_func._compiled_functions[
                        signature
                    ]
                    if signature in old_func.return_signatures:
                        new_func.return_signatures[signature] = old_func.return_signatures[
                            signature
                        ]
                    if signature in old_func._wrapper_specs:
                        new_func._wrapper_specs[signature] = old_func._wrapper_specs[signature]
                    else:
                        new_func._wrapper_specs[signature] = old_func.build_wrapper_spec(signature)
                    if signature in old_func._pyc_extensions:
                        new_func._pyc_extensions[signature] = old_func._pyc_extensions[signature]
                    if signature in old_func._fast_call:
                        new_func._fast_call[signature] = old_func._fast_call[signature]

        except Exception:
            new_func.name = original_state["name"]
            new_func.return_signatures = original_state["return_signatures"]
            new_func._compiled_functions = original_state["_compiled_functions"]
            new_func._wrapper_specs = original_state["_wrapper_specs"]
            new_func._pyc_extensions = original_state["_pyc_extensions"]
            new_func._library_pyc_extension = original_state["_library_pyc_extension"]
            new_func._fast_call = original_state["_fast_call"]
            raise

        new_func.name = name
        new_func._library_pyc_extension = None
        if not preserve_unselected:
            new_func._fast_call.clear()
        self._entries[name] = new_func

        if self.name in NumetaLibrary.loaded:
            warnings.warn(
                "Incremental replacement relinks the library on disk. Already-loaded function "
                "pointers may still point to the old shared object. Reload in a fresh process "
                "for guaranteed behavior.",
                RuntimeWarning,
            )

        return new_func

    def list_functions(self) -> list[str]:
        return list(self._entries)

    def signatures(self, name: str) -> list:
        return list(self._entries[name]._compiled_functions)

    def compiled_symbols(self, name: str) -> dict:
        return {
            signature: compiled.func_name
            for signature, compiled in self._entries[name]._compiled_functions.items()
        }

    def signature_for_call(self, name: str, *args, **kwargs):
        return self._entries[name].get_signature(*args, **kwargs)

    def compiled_function(self, name: str, signature) -> NumetaCompiledFunction:
        try:
            return self._entries[name]._compiled_functions[signature]
        except KeyError as exc:
            if name not in self._entries:
                raise
            raise KeyError(
                f"Function {name!r} has no compiled specialization for signature {signature!r}"
            ) from exc

    @property
    def functions(self) -> MappingProxyType:
        return MappingProxyType(self._entries)

    def __getitem__(self, name: str) -> NumetaFunction:
        return self._entries[name]

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __iter__(self):
        return iter(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, size={len(self)})"

    def __getattr__(self, name) -> NumetaFunction:
        if name in self._entries:
            return self._entries[name]
        raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")

    def _nm_add(self, function: NumetaFunction) -> None:
        self.register(function)

    def _nm_add_global(self, global_target: NumetaCompiledFunction) -> None:
        if not isinstance(global_target, NumetaCompiledFunction):
            raise TypeError("global registration expects a NumetaCompiledFunction")
        existing = self._global_entries.get(global_target.library_name)
        if existing is not None and existing is not global_target:
            raise ValueError(f"Global namespace '{global_target.library_name}' already registered")
        self._global_entries[global_target.library_name] = global_target

    def _nm_get(self, name) -> NumetaFunction | None:
        return self._entries.get(name)

    def write_code(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for nm_function in self._entries.values():
            for compiled_target in nm_function._compiled_functions.values():
                if compiled_target.backend == "fortran":
                    fortran_src = directory / f"{compiled_target.func_name}_src.f90"
                    from .ir import FortranEmitter, lower_procedure

                    ir_proc = lower_procedure(compiled_target.symbolic_function)
                    emitter = FortranEmitter()
                    fortran_src.write_text(emitter.emit_procedure(ir_proc))
                elif compiled_target.backend == "c":
                    from numeta.c.emitter import CEmitter
                    from .ir import lower_procedure

                    c_src = directory / f"{compiled_target.func_name}_src.c"
                    ir_proc = lower_procedure(compiled_target.symbolic_function)
                    emitter = CEmitter()
                    c_code, _requires_math = emitter.emit_procedure(ir_proc)
                    c_src.write_text(c_code)
                else:
                    raise ValueError(f"Unsupported backend: {compiled_target.backend}")

        for global_target in self._global_entries.values():
            if global_target.backend == "fortran":
                fortran_src = directory / f"{global_target.func_name}_src.f90"
                from .ast.namespace import Namespace
                from .fortran.fortran_syntax import render_stmt_lines

                if not isinstance(global_target.symbolic_function, Namespace):
                    raise ValueError("Global target must be backed by a namespace")
                lines = render_stmt_lines(
                    global_target.symbolic_function.get_declaration(), indent=0
                )
                fortran_src.write_text("".join(lines))
            elif global_target.backend == "c":
                from .ast.namespace import Namespace
                from numeta.c.emitter import CEmitter

                c_src = directory / f"{global_target.func_name}_src.c"
                if not isinstance(global_target.symbolic_function, Namespace):
                    raise ValueError("Global target must be backed by a namespace")
                emitter = CEmitter()
                c_code, _requires_math = emitter.emit_namespace(global_target.symbolic_function)
                c_src.write_text(c_code)
            else:
                raise ValueError(f"Unsupported backend: {global_target.backend}")

    def save(
        self,
        directory: str | Path,
        compile_flags: str | Iterable[str] | None = None,
    ) -> Path:
        directory = Path(directory).absolute()
        directory.mkdir(parents=True, exist_ok=True)

        if self.name is None:
            raise ValueError("Library name must be set before saving")
        name = self.name

        #
        # Create the interface of only the functions owned by the library
        #

        procedures_infos = []
        for function in self._entries.values():
            procedures_infos.extend(function._wrapper_specs.values())
        procedures_infos = NumetaFunction._deduplicate_wrapper_specs(procedures_infos)

        pyc_extension = PyCExtension(
            name=self.name,
            functions=procedures_infos,
        )

        resolved_flags = settings.default_compile_flags if compile_flags is None else compile_flags
        wrapper_config_function = next(iter(self._entries.values()), None)
        wrapper_compile_flags = (
            wrapper_config_function.compile_flags
            if wrapper_config_function is not None
            else resolved_flags
        )
        wrapper_backend = (
            wrapper_config_function.backend if wrapper_config_function is not None else None
        )
        pyc_extension.set_cache_info(wrapper_compile_flags, backend=wrapper_backend)
        compiler = Compiler("gcc", compile_flags=resolved_flags)

        obj_files: set[Path] = set()
        dependencies = {}
        pickle_path = directory / f"{self.name}.pkl"
        temp_pickle_path: Path | None = None

        def build_function_state(obj: NumetaFunction) -> dict:
            return {
                "name": obj.name,
                "hidden": obj.hidden,
                "external": obj.external,
                "_path": obj._path,
                "_rpath": obj._rpath,
                "_include": obj._include,
                "_obj_files": obj._obj_files,
                "additional_flags": obj.additional_flags,
                "to_link": obj.to_link,
                "namespaces": obj.namespaces,
                "procedures": obj.procedures,
                "variables": obj.variables,
                "directory": obj.directory,
                "do_checks": obj.do_checks,
                "compile_flags": obj.compile_flags,
                "backend": obj.backend,
                "namer": _pickleable_or_none(obj.namer, description=f"namer for {obj.name!r}"),
                "inline": obj.inline,
                "_func": None,
                "params": obj.params,
                "fixed_param_indices": obj.fixed_param_indices,
                "n_positional_or_default_args": obj.n_positional_or_default_args,
                "catch_var_positional_name": obj.catch_var_positional_name,
                "return_signatures": obj.return_signatures,
                "_compiled_functions": obj._compiled_functions,
                "_wrapper_specs": obj._wrapper_specs,
                "_pyc_extensions": obj._pyc_extensions,
                "_library_pyc_extension": obj._library_pyc_extension,
                "_fast_call": {},
                "_use_c_dispatch_instance": obj._use_c_dispatch_instance,
            }

        def build_compiled_function_state(obj: NumetaCompiledFunction) -> dict:
            saved_obj, saved_src, saved_include = _persist_compiled_artifacts(obj, directory)
            return {
                # Loaded functions link against the combined library but keep
                # func_name as the exported procedure symbol.
                "name": name,
                "hidden": obj.hidden,
                "external": obj.external,
                "_path": directory,
                "_rpath": directory,
                "_include": directory,
                "_obj_files": saved_obj,
                "_source_files": [saved_src] if saved_src is not None else [],
                "additional_flags": obj.additional_flags,
                "to_link": obj.to_link,
                "namespaces": obj.namespaces,
                "procedures": obj.procedures,
                "variables": obj.variables,
                "func_name": obj.func_name,
                "symbolic_function": obj.symbolic_function,
                "do_checks": obj.do_checks,
                "compile_flags": obj.compile_flags,
                "backend": obj.backend,
                "_requires_math": obj._requires_math,
                "compiled": True,
            }

        # We need to compiled ALL the NumetaFunctions not only the one directly owned by the library

        class RewritingPickler(pickle.Pickler):

            def reducer_override(self, obj):  # type: ignore[override]
                nonlocal dependencies
                nonlocal obj_files
                if isinstance(obj, DataTypeMeta) and getattr(obj, "_is_struct", False):
                    return (
                        _rebuild_struct_type,
                        (obj._np_type, tuple(obj._members), obj._name),
                    )

                if isinstance(obj, NumetaFunction):
                    state = build_function_state(obj)
                    state["_pyc_extensions"] = {}
                    state["_library_pyc_extension"] = pyc_extension
                    return (NumetaFunction.__new__, (NumetaFunction,), state)

                if isinstance(obj, NumetaCompiledFunction):
                    state = build_compiled_function_state(obj)
                    obj_files.add(Path(state["_obj_files"]))
                    dependencies |= obj.symbolic_function.get_dependencies()
                    return (NumetaCompiledFunction.__new__, (NumetaCompiledFunction,), state)

                return NotImplemented

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=directory,
                prefix=f".{self.name}.",
                suffix=".pkl.tmp",
                delete=False,
            ) as f:
                temp_pickle_path = Path(f.name)
                payload = {
                    "version": 2,
                    "entries": list(self._entries.values()),
                    "global_entries": dict(self._global_entries),
                }
                RewritingPickler(f).dump(payload)

            libraries = set()
            libraries_dirs = set()
            rpath_dirs = set()
            include_dirs = set()
            additional_flags = set()

            processed_compiled = set()
            processed_external = set()
            pending_dependencies = list(dependencies.values())

            while pending_dependencies:
                lib = pending_dependencies.pop()

                if isinstance(lib, NumetaCompiledFunction):
                    marker = id(lib)
                    if marker in processed_compiled:
                        continue
                    processed_compiled.add(marker)

                    saved_obj, _saved_src, _saved_include = _persist_compiled_artifacts(
                        lib, directory
                    )
                    obj_files.add(saved_obj)
                    pending_dependencies.extend(lib.symbolic_function.get_dependencies().values())
                    continue

                marker = id(lib)
                if marker in processed_external:
                    continue
                processed_external.add(marker)

                if lib.include is not None:
                    if isinstance(lib.include, (list, tuple, set)):
                        include_dirs |= set(lib.include)
                    else:
                        include_dirs.add(lib.include)

                if lib.to_link:
                    libraries.add(getattr(lib, "library_name", lib.name))
                    if lib.path is not None:
                        libraries_dirs.add(str(lib.path))
                    if lib.rpath is not None:
                        rpath_dirs.add(str(lib.rpath))

                if lib.additional_flags is not None:
                    if isinstance(lib.additional_flags, str):
                        additional_flags.add(tuple(lib.additional_flags.split()))
                    else:
                        additional_flags.add(tuple(lib.additional_flags))

            lib = compiler.compile_to_library(
                name,
                obj_files,
                directory,
                libraries=libraries,
                include_dirs=include_dirs,
                libraries_dirs=libraries_dirs,
                rpath_dirs=rpath_dirs,
                additional_flags=additional_flags,
            )

            if procedures_infos:
                pyc_extension.compile(
                    core_lib_name=name,
                    core_lib_path=directory,
                    directory=directory,
                    compile_flags=wrapper_compile_flags,
                    backend=wrapper_backend,
                )

            os.replace(temp_pickle_path, pickle_path)
        except Exception:
            if temp_pickle_path is not None:
                temp_pickle_path.unlink(missing_ok=True)
            raise

        return lib

    @classmethod
    def load(
        cls,
        name: str,
        directory: str | Path,
        *,
        safe: bool = False,
    ) -> "NumetaLibrary":
        cls._nm_validate_name(name)
        directory = Path(directory).absolute()

        result = NumetaLibrary(name)

        try:
            with open(directory / f"{name}.pkl", "rb") as handle:
                payload = pickle.load(handle)

            if isinstance(payload, dict) and "entries" in payload:
                entries = payload["entries"]
                global_entries = payload.get("global_entries", {})
            else:
                entries = payload
                global_entries = {}

            for func in entries:
                if isinstance(func, NumetaFunction):
                    result._entries[func.name] = func
                elif isinstance(func, NumetaCompiledFunction):
                    result._global_entries[func.func_name] = func

            if isinstance(global_entries, dict):
                result._global_entries.update(global_entries)
            else:
                for func in global_entries:
                    result._global_entries[func.func_name] = func
        except (EOFError, pickle.UnpicklingError) as exc:
            if not safe:
                raise
            warnings.warn(
                f"Failed to load NumetaLibrary '{name}' cache from {directory / f'{name}.pkl'}: {exc}. "
                "Treating it as a cache miss.",
                RuntimeWarning,
            )

        restored_extensions = set()
        for func in result._entries.values():
            wrapper = getattr(func, "_library_pyc_extension", None)
            if wrapper is None or id(wrapper) in restored_extensions:
                continue
            restored_extensions.add(id(wrapper))

            wrapper_path = directory / f"lib{wrapper.name}.so"
            if wrapper_path.exists() and wrapper.cache_matches(
                func.compile_flags, backend=func.backend
            ):
                wrapper.set_lib_path(wrapper_path)
            else:
                wrapper.set_lib_path(None)

        loaded_names: set[str] = set()
        for func in result._entries.values():
            for compiled in func._compiled_functions.values():
                loaded_names.add(compiled.func_name)

        for compiled in result._global_entries.values():
            loaded_names.add(compiled.func_name)

        if loaded_names:
            native_name_registry.reserve_many(loaded_names)
        NumetaLibrary.loaded.add(name)

        return result
