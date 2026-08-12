from __future__ import annotations

from pathlib import Path
import os
import shlex
import shutil
import subprocess as sp
from typing import Iterable

from .exceptions import CompilationError, ToolchainNotFoundError


class Compiler:
    def __init__(
        self,
        compiler,
        compile_flags: str | Iterable[str],
        *,
        setting=None,
        env_var=None,
    ) -> None:
        self.compiler = self._resolve_executable(
            compiler,
            setting=setting,
            env_var=env_var,
        )
        self.compile_flags = self._normalize_flags(compile_flags)

    @staticmethod
    def _resolve_executable(compiler, *, setting=None, env_var=None) -> str:
        try:
            compiler = os.fspath(compiler)
        except TypeError as exc:
            raise TypeError("compiler must be a path-like value") from exc
        if not compiler:
            raise ValueError("compiler cannot be empty")

        if os.sep in compiler:
            candidate = Path(compiler).expanduser().absolute()
            resolved = (
                str(candidate) if candidate.is_file() and os.access(candidate, os.X_OK) else None
            )
        else:
            resolved = shutil.which(compiler)
        if resolved is None:
            raise ToolchainNotFoundError(
                compiler,
                setting=setting,
                env_var=env_var,
            )
        return resolved

    @staticmethod
    def _normalize_flags(compile_flags: str | Iterable[str]) -> list[str]:
        if isinstance(compile_flags, str):
            return shlex.split(compile_flags)
        return list(compile_flags)

    def run_command(self, command: list[str], cwd: Path) -> sp.CompletedProcess[str]:
        try:
            sp_run = sp.run(
                command,
                cwd=cwd,
                stdout=sp.PIPE,
                stderr=sp.PIPE,
                text=True,
            )
        except OSError as exc:
            raise CompilationError(
                command=command,
                cwd=Path(cwd),
                stderr=str(exc),
            ) from exc
        if sp_run.returncode != 0:
            raise CompilationError(
                command=command,
                cwd=Path(cwd),
                stdout=sp_run.stdout,
                stderr=sp_run.stderr,
            )
        return sp_run

    def identity(self) -> dict:
        """Return compiler provenance without applying user compile flags."""
        result = self.run_command([self.compiler, "--version"], cwd=Path.cwd())
        first_line = result.stdout.splitlines()[:1]
        return {
            "executable": self.compiler,
            "version": first_line[0] if first_line else "unknown",
        }

    def build_obj_command(
        self,
        *,
        obj_file: Path,
        source_files: Iterable[Path],
        include_dirs: Iterable[str],
        additional_flags: Iterable[str],
    ) -> list[str]:
        command = [self.compiler]
        command.extend(["-fopenmp"])
        command.extend(self.compile_flags)
        command.extend(["-fPIC", "-c", "-o", str(obj_file)])
        command.extend([f"{file}" for file in source_files])
        command.extend([f"-I{inc_dir}" for inc_dir in include_dirs])
        command.extend(additional_flags)
        return command

    def build_lib_command(
        self,
        *,
        lib_file: Path,
        obj_files: Iterable[Path],
        include_dirs: Iterable[str],
        additional_flags: Iterable[str],
        libraries: Iterable[str],
        libraries_dirs: Iterable[Path],
        rpath_dirs: Iterable[Path],
    ) -> list[str]:
        command = [self.compiler]
        command.extend(self.compile_flags)
        command.extend(["-fopenmp"])
        command.extend(["-fPIC", "-shared", "-o", str(lib_file)])
        command.extend([str(obj) for obj in obj_files])
        command.extend([f"-l{lib}" for lib in libraries])
        command.extend([f"-L{lib_dir}" for lib_dir in libraries_dirs])
        command.extend([f"-I{inc_dir}" for inc_dir in include_dirs])
        if rpath_dirs:
            command.extend([f"-L{lib_dir}" for lib_dir in rpath_dirs])
            command.append("-Wl,--enable-new-dtags")
            command.extend([f"-Wl,-rpath,{lib_dir}" for lib_dir in rpath_dirs])
        command.extend(additional_flags)
        return command

    def compile_to_obj(
        self,
        *,
        name: str,
        directory: Path,
        sources: Iterable[Path],
        include_dirs: Iterable[str] = (),
        additional_flags: Iterable[str] = (),
        obj_suffix: str = "_fortran.o",
    ) -> tuple[Path, str]:
        """
        Compile Fortran source files using gfortran and return the resulting object file.
        """
        obj_file = directory / f"{name}{obj_suffix}"

        command = self.build_obj_command(
            obj_file=obj_file,
            source_files=sources,
            include_dirs=include_dirs,
            additional_flags=additional_flags,
        )

        self.run_command(command, cwd=directory)

        return obj_file, str(directory)

    def compile_to_library(
        self,
        name: str,
        src_files: Iterable[Path],
        directory: Path,
        include_dirs: Iterable[str] = (),
        libraries: Iterable[str] = (),
        libraries_dirs: Iterable[Path] = (),
        rpath_dirs: Iterable[Path] = (),
        additional_flags: Iterable[str] = (),
    ) -> Path:
        """
        Compile Fortran source files using gfortran and return the resulting object file.
        """

        lib_file = directory / f"lib{name}.so"
        command = self.build_lib_command(
            lib_file=lib_file,
            obj_files=src_files,
            include_dirs=include_dirs,
            additional_flags=additional_flags,
            libraries=libraries,
            libraries_dirs=libraries_dirs,
            rpath_dirs=rpath_dirs,
        )

        self.run_command(command, cwd=lib_file.parent)

        return lib_file
