"""Persistence and relocation of compiled Numeta library artifacts."""

import shutil
from pathlib import Path
from typing import Iterable

from .numeta_function import NumetaCompiledFunction

ARTIFACT_MANIFEST_VERSION = 1


def _artifact_dir_for_compiled(directory: Path, compiled: NumetaCompiledFunction) -> Path:
    return directory / "artifacts" / "compiled" / compiled.func_name


def _source_suffix_for(compiled: NumetaCompiledFunction) -> str:
    if compiled.backend == "fortran":
        return "_src.f90"
    if compiled.backend == "c":
        return "_src.c"
    raise ValueError(f"Unsupported backend: {compiled.backend}")


def _object_suffix_for(compiled: NumetaCompiledFunction) -> str:
    if compiled.backend == "fortran":
        return "_fortran.o"
    if compiled.backend == "c":
        return "_c.o"
    raise ValueError(f"Unsupported backend: {compiled.backend}")


def _source_path_for(compiled: NumetaCompiledFunction) -> Path:
    return Path(compiled._path) / f"{compiled.func_name}{_source_suffix_for(compiled)}"


def _artifact_object_for(
    compiled: NumetaCompiledFunction,
    artifact_dir: Path,
) -> Path | None:
    saved_obj = getattr(compiled, "_obj_files", None)
    if saved_obj is not None:
        candidate = artifact_dir / Path(saved_obj).name
        if candidate.exists():
            return candidate

    preferred = artifact_dir / f"{compiled.func_name}{_object_suffix_for(compiled)}"
    if preferred.exists():
        return preferred

    matches = sorted(artifact_dir.glob("*.o"))
    if len(matches) == 1:
        return matches[0]

    return None


def _load_legacy_compiled_artifacts(
    compiled: NumetaCompiledFunction,
    directory: Path,
) -> None:
    directory = directory.absolute()
    artifact_dir = _artifact_dir_for_compiled(directory, compiled)

    compiled._path = directory
    compiled._rpath = directory
    if not hasattr(compiled, "_source_files"):
        compiled._source_files = []
    if not hasattr(compiled, "_requires_math"):
        compiled._requires_math = False

    if artifact_dir.exists():
        compiled._include = artifact_dir

        artifact_obj = _artifact_object_for(compiled, artifact_dir)
        if artifact_obj is not None:
            compiled._obj_files = artifact_obj

        source_files = []
        for source_file in getattr(compiled, "_source_files", ()) or ():
            source_path = artifact_dir / Path(source_file).name
            if source_path.exists() and source_path not in source_files:
                source_files.append(source_path)

        generated_source = artifact_dir / (f"{compiled.func_name}{_source_suffix_for(compiled)}")
        if generated_source.exists() and generated_source not in source_files:
            source_files.append(generated_source)
        compiled._source_files = source_files
    else:
        include = getattr(compiled, "_include", None)
        if include is None or not Path(include).exists():
            compiled._include = directory


def _apply_loaded_compiled_artifact_manifest(
    compiled: NumetaCompiledFunction,
    directory: Path,
    manifest: dict,
) -> None:
    artifact = manifest.get(compiled.func_name)
    if artifact is None:
        raise KeyError(f"Missing compiled artifact metadata for {compiled.func_name!r}")

    if artifact.get("version") != ARTIFACT_MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported artifact manifest version for {compiled.func_name!r}: "
            f"{artifact.get('version')!r}"
        )

    object_files = artifact.get("object_files") or []
    include_dirs = artifact.get("include_dirs") or []
    source_files = artifact.get("source_files") or []
    if not object_files:
        raise ValueError(f"No object files recorded for {compiled.func_name!r}")
    if not include_dirs:
        raise ValueError(f"No include dirs recorded for {compiled.func_name!r}")

    resolved_objects = [
        _resolve_artifact_path(directory, object_file) for object_file in object_files
    ]
    resolved_includes = [
        _resolve_artifact_path(directory, include_dir) for include_dir in include_dirs
    ]
    resolved_sources = [
        _resolve_artifact_path(directory, source_file) for source_file in source_files
    ]

    missing_paths = [path for path in (*resolved_objects, *resolved_includes) if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing persisted artifact(s) for " f"{compiled.func_name!r}: {missing_paths!r}"
        )

    compiled.name = artifact.get("func_name", compiled.func_name)
    compiled.func_name = artifact.get("func_name", compiled.func_name)
    compiled.library_name = artifact.get("library_name", compiled.library_name)
    compiled.backend = artifact.get("backend", compiled.backend)
    compiled._path = directory
    compiled._rpath = directory
    compiled._include = resolved_includes[0]
    compiled._obj_files = resolved_objects[0]
    compiled._source_files = [path for path in resolved_sources if path.exists()]
    if not hasattr(compiled, "_requires_math"):
        compiled._requires_math = False
    compiled.compiled = True


def _load_compiled_artifact_graph(
    directory: Path,
    roots: Iterable[NumetaCompiledFunction],
    manifest: dict | None = None,
) -> None:
    pending = list(roots)
    seen: set[int] = set()

    while pending:
        compiled = pending.pop()
        if not isinstance(compiled, NumetaCompiledFunction):
            continue
        marker = id(compiled)
        if marker in seen:
            continue
        seen.add(marker)

        if manifest:
            _apply_loaded_compiled_artifact_manifest(compiled, directory, manifest)
        else:
            _load_legacy_compiled_artifacts(compiled, directory)

        symbolic = getattr(compiled, "symbolic_function", None)
        if symbolic is None:
            continue
        symbolic.parent = compiled
        try:
            dependencies = symbolic.get_dependencies().values()
        except Exception:
            continue
        pending.extend(dep for dep in dependencies if isinstance(dep, NumetaCompiledFunction))


def _copy_if_different(source: Path, target: Path) -> None:
    if source.absolute() == target.absolute():
        return
    shutil.copy2(source, target)


def _relative_artifact_path(directory: Path, path: Path) -> str:
    return path.absolute().relative_to(directory.absolute()).as_posix()


def _resolve_artifact_path(directory: Path, path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return directory / resolved


def _compiled_artifact_manifest_entry(
    compiled: NumetaCompiledFunction,
    directory: Path,
    *,
    saved_obj_file: Path,
    saved_src_file: Path | None,
    saved_include: Path,
) -> dict:
    source_files = []
    if saved_src_file is not None:
        source_files.append(_relative_artifact_path(directory, saved_src_file))

    return {
        "version": ARTIFACT_MANIFEST_VERSION,
        "func_name": compiled.func_name,
        "library_name": compiled.library_name,
        "backend": compiled.backend,
        "object_files": [_relative_artifact_path(directory, saved_obj_file)],
        "source_files": source_files,
        "include_dirs": [_relative_artifact_path(directory, saved_include)],
        "module_files": [
            _relative_artifact_path(directory, mod_file)
            for mod_file in sorted(saved_include.glob("*.mod"))
        ],
    }


def _persist_compiled_artifacts(
    compiled: NumetaCompiledFunction,
    directory: Path,
) -> tuple[Path, Path | None, Path, dict]:
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

    compiled._obj_files = saved_obj_file
    compiled._include = target_dir
    if saved_src_file is not None:
        compiled._source_files = [saved_src_file]

    root_obj_file = directory / saved_obj_file.name
    if root_obj_file != saved_obj_file:
        root_obj_file.unlink(missing_ok=True)
    if old_obj_file.parent == directory and old_obj_file != saved_obj_file:
        old_obj_file.unlink(missing_ok=True)

    return (
        saved_obj_file,
        saved_src_file,
        target_dir,
        _compiled_artifact_manifest_entry(
            compiled,
            directory,
            saved_obj_file=saved_obj_file,
            saved_src_file=saved_src_file,
            saved_include=target_dir,
        ),
    )
