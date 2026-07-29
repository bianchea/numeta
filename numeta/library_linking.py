"""Dependency traversal and link-plan construction for Numeta libraries."""

from pathlib import Path
from typing import Iterable

from .numeta_function import NumetaCompiledFunction


def _iter_optional_sequence(value):
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return value
    return (value,)


def _iter_flags(value):
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(value.split())
    return tuple(value)


def _append_unique(target: list, seen: set, values) -> None:
    for value in values:
        if value is None:
            continue
        marker = str(value)
        if marker in seen:
            continue
        seen.add(marker)
        target.append(value)


def _active_compiled_targets_by_name(
    entries: dict,
    global_entries: dict,
) -> dict[str, NumetaCompiledFunction]:
    targets = {}
    for nm_function in entries.values():
        for compiled_target in nm_function._compiled_functions.values():
            targets[compiled_target.func_name] = compiled_target
    for global_target in global_entries.values():
        targets[global_target.func_name] = global_target
    return targets


def _collect_compiled_target_closure(
    roots: Iterable[NumetaCompiledFunction],
    active_targets: dict[str, NumetaCompiledFunction],
) -> list[NumetaCompiledFunction]:
    targets = {}
    pending = list(roots)
    while pending:
        target = pending.pop(0)
        if not isinstance(target, NumetaCompiledFunction):
            continue
        target = active_targets.get(target.func_name, target)
        if target.func_name in targets:
            continue
        targets[target.func_name] = target

        symbolic = getattr(target, "symbolic_function", None)
        if symbolic is None:
            continue
        try:
            dependencies = symbolic.get_dependencies().values()
        except Exception as exc:
            raise ValueError(
                f"Could not collect generated dependencies for {target.func_name!r}"
            ) from exc
        pending.extend(dep for dep in dependencies if isinstance(dep, NumetaCompiledFunction))

    return list(targets.values())


def _link_plan_for_compiled_target(
    root: NumetaCompiledFunction,
    active_targets: dict[str, NumetaCompiledFunction],
) -> dict:
    root = active_targets.get(root.func_name, root)
    compiled_targets = _collect_compiled_target_closure([root], active_targets)

    object_files = []
    dependency_objects = []
    include_dirs = []
    libraries = set()
    libraries_dirs = []
    rpath_dirs = []
    additional_flags = []
    object_seen = set()
    dependency_object_seen = set()
    include_seen = set()
    library_dir_seen = set()
    rpath_seen = set()
    additional_flag_seen = set()
    processed_external = set()
    compiled_backends = set()
    compiled_requires_math = False
    root_object_file = None

    for compiled_target in compiled_targets:
        compiled_target = active_targets.get(compiled_target.func_name, compiled_target)
        compiled_backends.add(compiled_target.backend)
        if compiled_target.backend == "c" and getattr(compiled_target, "_requires_math", False):
            compiled_requires_math = True

        current_objects = [Path(obj_file) for obj_file in compiled_target.obj_files]
        if compiled_target is root and current_objects:
            root_object_file = current_objects[0]
        _append_unique(object_files, object_seen, current_objects)
        if compiled_target is not root:
            _append_unique(dependency_objects, dependency_object_seen, current_objects)

        _append_unique(include_dirs, include_seen, _iter_optional_sequence(compiled_target.include))

        symbolic = getattr(compiled_target, "symbolic_function", None)
        if symbolic is None:
            continue
        for dependency in symbolic.get_dependencies().values():
            if isinstance(dependency, NumetaCompiledFunction):
                continue
            marker = id(dependency)
            if marker in processed_external:
                continue
            processed_external.add(marker)

            external_objects = [Path(obj) for obj in _iter_optional_sequence(dependency.obj_files)]
            _append_unique(object_files, object_seen, external_objects)
            _append_unique(dependency_objects, dependency_object_seen, external_objects)

            _append_unique(include_dirs, include_seen, _iter_optional_sequence(dependency.include))

            if dependency.to_link:
                libraries.add(getattr(dependency, "library_name", dependency.name))
                _append_unique(
                    libraries_dirs, library_dir_seen, _iter_optional_sequence(dependency.path)
                )
                _append_unique(rpath_dirs, rpath_seen, _iter_optional_sequence(dependency.rpath))

            _append_unique(
                additional_flags,
                additional_flag_seen,
                _iter_flags(dependency.additional_flags),
            )

    if "fortran" in compiled_backends:
        libraries.update({"gfortran", "m", "mvec"})
    if compiled_requires_math:
        libraries.add("m")

    if root_object_file is None:
        raise ValueError(f"Could not determine object file for {root.func_name!r}")

    return {
        "symbol": root.func_name,
        "object_file": root_object_file,
        "dependency_objects": dependency_objects,
        "object_files": object_files,
        "include_dirs": include_dirs,
        "libraries": sorted(libraries),
        "libraries_dirs": libraries_dirs,
        "rpath_dirs": rpath_dirs,
        "additional_flags": additional_flags,
        "backend": root.backend,
        "compile_flags": list(root.compile_flags),
        "requires_math": compiled_requires_math,
    }
