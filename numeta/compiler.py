from __future__ import annotations

from pathlib import Path
import os
import shlex
import subprocess as sp
import textwrap
from typing import Iterable

from .exceptions import CompilationError


class Compiler:
    def __init__(self, compiler, compile_flags: str | Iterable[str]) -> None:
        if compiler == "gcc" and os.path.exists("/usr/bin/gcc"):
            compiler = "/usr/bin/gcc"
        self.compiler = compiler
        self.compile_flags = self._normalize_flags(compile_flags)

    @staticmethod
    def _normalize_flags(compile_flags: str | Iterable[str]) -> list[str]:
        if isinstance(compile_flags, str):
            return shlex.split(compile_flags)
        return list(compile_flags)

    def run_command(self, command: list[str], cwd: Path) -> sp.CompletedProcess[str]:
        sp_run = sp.run(
            command,
            cwd=cwd,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            text=True,
        )
        if sp_run.returncode != 0:
            error_message = "Error while compiling, the command was:\n"
            error_message += " ".join(command) + "\n"
            error_message += "The output was:\n"
            error_message += textwrap.indent(sp_run.stdout, "    ")
            error_message += textwrap.indent(sp_run.stderr, "    ")
            raise CompilationError(error_message)
        return sp_run

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
