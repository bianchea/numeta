import sys
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Iterable
import warnings

from .library_linking import (
    _active_compiled_targets_by_name,
    _collect_compiled_target_closure,
    _link_plan_for_compiled_target,
)
from .library_replacement import (
    adopt_compiled_target_state as _replace_compiled_target_state,
    single_global_variable as _single_global_variable,
    validate_function_compatibility as _validate_function_level_compatibility,
    validate_global_compatibility as _validate_global_replacement_compatibility,
    validate_specialization_compatibility as _validate_specialization_compatibility,
)
from .library_signature import signature_id as _signature_id
from .library_signature import signature_to_jsonable as _signature_to_jsonable
from .native_name_registry import native_name_registry
from .numeta_function import NumetaCompiledFunction, NumetaFunction
from .pyc_extension import PyCExtension
from .settings import settings


def _emit_timing(timing_callback, phase: str, elapsed_s: float, **metadata) -> None:
    if timing_callback is None:
        return
    event = {"phase": phase, "elapsed_s": elapsed_s}
    event.update(metadata)
    timing_callback(event)


@contextmanager
def _timing_phase(timing_callback, phase: str, **metadata):
    start = perf_counter()
    try:
        yield
    finally:
        _emit_timing(timing_callback, phase, perf_counter() - start, **metadata)


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
        "replace_global_constant",
        "clear_generated_state",
        "list_functions",
        "signatures",
        "signature_id",
        "signature_ids",
        "signature_from_id",
        "signature_to_json",
        "compiled_symbols",
        "signature_for_call",
        "compiled_function",
        "dependency_objects_for_symbol",
        "link_plan_for_symbol",
        "compile_replacement",
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

    def replace_global_constant(
        self,
        name: str,
        *,
        value=None,
        shape=None,
        dtype=None,
        order: str | None = None,
        directory: str | Path | None = None,
        backend: str | None = None,
        compile_now: bool = True,
        allow_shape_change: bool = False,
    ):
        namespace_name = f"{name}_namespace"
        if namespace_name not in self._global_entries:
            raise KeyError(f"Cannot replace unknown global constant {name!r}")

        old_target = self._global_entries[namespace_name]
        old_var = _single_global_variable(old_target)
        if old_var.name != name:
            raise KeyError(
                f"Global namespace {namespace_name!r} contains {old_var.name!r}, not {name!r}"
            )

        if shape is None:
            shape = old_var._shape
        if dtype is None:
            dtype = old_var.dtype
        if order is None:
            order = "F" if old_var._shape.fortran_order else "C"
        if directory is None:
            directory = old_target._path
        if backend is None:
            backend = old_target.backend

        from .wrappers.declare_global_constant import make_global_constant_target

        new_var, new_target = make_global_constant_target(
            shape,
            dtype=dtype,
            order=order,
            name=name,
            value=value,
            directory=str(directory) if directory is not None else None,
            backend=backend,
        )
        _validate_global_replacement_compatibility(
            old_target,
            new_target,
            allow_shape_change=allow_shape_change,
        )

        _replace_compiled_target_state(old_target, new_target)
        self._global_entries[namespace_name] = old_target

        if compile_now:
            old_target.compile_obj()

        return new_var

    def clear_generated_state(
        self,
        *,
        functions: Iterable[str | NumetaFunction] | str | NumetaFunction | None = None,
        include_globals: bool = False,
        release_names: bool = True,
    ) -> None:
        if functions is None:
            selected_functions = list(self._entries.values())
        elif isinstance(functions, str):
            selected_functions = [self._entries[functions]]
        elif isinstance(functions, NumetaFunction):
            selected_functions = [functions]
        else:
            selected_functions = [
                self._entries[item] if isinstance(item, str) else item for item in functions
            ]

        for function in selected_functions:
            if not isinstance(function, NumetaFunction):
                raise TypeError("functions must contain names or NumetaFunction instances")
            function.clear_generated_state(release_names=release_names)

        if include_globals:
            if release_names:
                native_name_registry.release_many(
                    global_target.func_name for global_target in self._global_entries.values()
                )
            self._global_entries.clear()

    def replace(
        self,
        name_or_function: str | NumetaFunction,
        function: NumetaFunction | None = None,
        *,
        compile_now: bool = True,
        require_existing_specializations: bool = True,
        signatures: Iterable | None = None,
        non_selected: str = "preserve",
        mode: str | None = None,
        timing_callback=None,
    ) -> NumetaFunction:
        replace_start = perf_counter()
        valid_non_selected = {"preserve", "invalidate", "error"}
        if non_selected not in valid_non_selected:
            raise ValueError(
                "non_selected must be one of "
                f"{', '.join(sorted(repr(policy) for policy in valid_non_selected))}"
            )
        valid_modes = {"eager", "lazy"}
        if mode is None:
            mode = "lazy" if signatures is None and not compile_now else "eager"
        elif mode not in valid_modes:
            raise ValueError(
                "mode must be one of " f"{', '.join(sorted(repr(mode) for mode in valid_modes))}"
            )
        if mode == "lazy" and compile_now:
            raise ValueError("replace(..., mode='lazy') cannot use compile_now=True")
        if mode == "lazy" and signatures is not None:
            raise ValueError("replace(..., mode='lazy') cannot also pass signatures")

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
        if mode == "lazy":
            selected_signatures = ()
            unselected_signatures = list(old_signatures)
            if non_selected == "error" and unselected_signatures:
                raise ValueError(
                    f"replace(..., mode='lazy', non_selected='error') received "
                    f"{len(unselected_signatures)} existing specialization(s) for {name!r}"
                )
            preserve_unselected = non_selected == "preserve"
        elif signatures is None:
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

        original_state = new_func.snapshot_generated_state()
        names_added_by_replace = []

        try:
            for signature in selected_signatures:
                old_compiled = old_func._compiled_functions[signature]
                old_symbol = old_compiled.func_name
                if not native_name_registry.is_reserved(old_symbol):
                    names_added_by_replace.append(old_symbol)

                with _timing_phase(
                    timing_callback,
                    "replace.construct",
                    function=name,
                    signature_id=_signature_id(signature),
                    symbol=old_symbol,
                ):
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
                    with _timing_phase(
                        timing_callback,
                        "replace.compile_obj",
                        function=name,
                        signature_id=_signature_id(signature),
                        symbol=old_symbol,
                    ):
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
            new_func.restore_generated_state(original_state)
            native_name_registry.release_many(names_added_by_replace)
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

        _emit_timing(
            timing_callback,
            "replace.total",
            perf_counter() - replace_start,
            function=name,
            signatures=len(selected_signatures),
        )
        return new_func

    def compile_replacement(
        self,
        name_or_function: str | NumetaFunction,
        function: NumetaFunction | None = None,
        *,
        signature=None,
        signatures: Iterable | None = None,
        require_existing_specializations: bool = True,
        non_selected: str = "preserve",
        timing_callback=None,
    ):
        if signature is not None and signatures is not None:
            raise ValueError("Pass either signature or signatures, not both")

        if function is None:
            if not isinstance(name_or_function, NumetaFunction):
                raise TypeError("compile_replacement(function) expects a NumetaFunction")
            name = name_or_function.name
        else:
            if not isinstance(name_or_function, str):
                raise TypeError("compile_replacement(name, function) expects name to be a string")
            name = name_or_function

        selected_signatures = None
        single_signature = signature is not None
        if signature is not None:
            selected_signatures = (self._resolve_signature_reference(name, signature),)
        elif signatures is not None:
            selected_signatures = tuple(
                self._resolve_signature_reference(name, item) for item in signatures
            )

        replaced = self.replace(
            name_or_function,
            function,
            compile_now=True,
            require_existing_specializations=require_existing_specializations,
            signatures=selected_signatures,
            non_selected=non_selected,
            timing_callback=timing_callback,
        )

        if selected_signatures is None:
            selected_signatures = tuple(replaced._compiled_functions)

        artifacts = []
        for selected_signature in selected_signatures:
            plan = self.link_plan_for_symbol(name, selected_signature)
            artifacts.append(
                {
                    "function": name,
                    "signature": selected_signature,
                    "signature_id": self.signature_id(name, selected_signature),
                    **plan,
                }
            )

        if single_signature:
            return artifacts[0]
        return artifacts

    def _resolve_signature_reference(self, name: str, signature):
        if isinstance(signature, str) and signature.startswith(f"sig-v{SIGNATURE_ID_VERSION}-"):
            return self.signature_from_id(name, signature)
        return signature

    def _compiled_target_for_symbol(
        self,
        symbol_or_function: str | NumetaFunction | NumetaCompiledFunction,
        signature=None,
    ) -> NumetaCompiledFunction:
        active_targets = _active_compiled_targets_by_name(
            self._entries,
            self._global_entries,
        )
        if isinstance(symbol_or_function, NumetaCompiledFunction):
            return active_targets.get(symbol_or_function.func_name, symbol_or_function)

        if isinstance(symbol_or_function, NumetaFunction):
            name = symbol_or_function.name
            function = self._entries.get(name, symbol_or_function)
        elif isinstance(symbol_or_function, str):
            if signature is None and symbol_or_function not in self._entries:
                target = active_targets.get(symbol_or_function)
                if target is not None:
                    return target
                raise KeyError(f"Unknown compiled symbol {symbol_or_function!r}")
            name = symbol_or_function
            function = self._entries[name]
        else:
            raise TypeError("Expected a function name, compiled symbol, or NumetaFunction")

        if signature is not None:
            signature = self._resolve_signature_reference(name, signature)
            try:
                return function._compiled_functions[signature]
            except KeyError as exc:
                raise KeyError(
                    f"Function {name!r} has no compiled specialization for signature {signature!r}"
                ) from exc

        if len(function._compiled_functions) != 1:
            raise ValueError(
                f"Function {name!r} has {len(function._compiled_functions)} compiled "
                "specialization(s); pass signature=... to select one"
            )
        return next(iter(function._compiled_functions.values()))

    def dependency_objects_for_symbol(
        self,
        symbol_or_function: str | NumetaFunction | NumetaCompiledFunction,
        signature=None,
        *,
        include_root: bool = True,
    ) -> list[Path]:
        plan = self.link_plan_for_symbol(symbol_or_function, signature)
        if include_root:
            return list(plan["object_files"])
        return list(plan["dependency_objects"])

    def link_plan_for_symbol(
        self,
        symbol_or_function: str | NumetaFunction | NumetaCompiledFunction,
        signature=None,
    ) -> dict:
        target = self._compiled_target_for_symbol(symbol_or_function, signature)
        active_targets = _active_compiled_targets_by_name(
            self._entries,
            self._global_entries,
        )
        return _link_plan_for_compiled_target(target, active_targets)

    def list_functions(self) -> list[str]:
        return list(self._entries)

    def signatures(self, name: str) -> list:
        return list(self._entries[name]._compiled_functions)

    def signature_to_json(self, signature) -> dict:
        return _signature_to_jsonable(signature)

    def signature_id(self, name: str, signature) -> str:
        if name not in self._entries:
            raise KeyError(name)
        return _signature_id(signature)

    def signature_ids(self, name: str) -> dict:
        return {
            signature: self.signature_id(name, signature)
            for signature in self._entries[name]._compiled_functions
        }

    def signature_from_id(self, name: str, signature_id: str):
        for signature in self._entries[name]._compiled_functions:
            if self.signature_id(name, signature) == signature_id:
                return signature
        raise KeyError(f"Function {name!r} has no compiled signature with id {signature_id!r}")

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
        existing = self._global_entries.get(global_target.func_name)
        if existing is not None and existing is not global_target:
            raise ValueError(f"Global namespace '{global_target.func_name}' already registered")
        self._global_entries[global_target.func_name] = global_target

    def _nm_get(self, name) -> NumetaFunction | None:
        return self._entries.get(name)

    def write_code(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        roots = []
        for nm_function in self._entries.values():
            roots.extend(nm_function._compiled_functions.values())
        roots.extend(self._global_entries.values())

        active_targets = _active_compiled_targets_by_name(
            self._entries,
            self._global_entries,
        )
        compiled_targets = _collect_compiled_target_closure(roots, active_targets)

        for compiled_target in compiled_targets:
            compiled_target.write_source(directory)

    def save(
        self,
        directory: str | Path,
        compile_flags: str | Iterable[str] | None = None,
        timing_callback=None,
    ) -> Path:
        """Persist this library as a validated native bundle."""
        from .library_bundle import save_library_bundle

        return save_library_bundle(
            self,
            directory,
            compile_flags=compile_flags,
            timing_callback=timing_callback,
        )

    @classmethod
    def load(
        cls,
        name: str,
        directory: str | Path,
        *,
        safe: bool = False,
        ignore_corrupt: bool | None = None,
    ) -> "NumetaLibrary":
        """Load a validated native bundle without deserializing Python objects."""
        from .library_bundle import load_library_bundle

        return load_library_bundle(
            cls,
            name,
            directory,
            safe=safe,
            ignore_corrupt=ignore_corrupt,
        )
