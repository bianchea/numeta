import sys
import subprocess
import os
import shutil
from pathlib import Path
import pytest


def test_c_extension_fallback():
    """
    Verifies that NumetaFunction falls back to Python implementation gracefully
    when the C extension (numeta._signature) is missing or fails to import.
    """
    script = """
import sys
import os

# Poison the module cache with a dummy module that lacks BaseFunction
from types import ModuleType
dummy = ModuleType("numeta._signature")
# We purposefully do not add BaseFunction to it
# But we add init_globals so signature.py doesn't crash on import
dummy.init_globals = lambda *args: None
sys.modules["numeta._signature"] = dummy


try:
    from numeta import numeta_function
    from numeta.numeta_function import NumetaFunction, BaseFunction
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. Verify instantiation doesn't crash and skips custom parser
def simple_func(a, b):
    return a + b

try:
    nf = NumetaFunction(simple_func)
except Exception as e:
    print(f"Error instantiating NumetaFunction: {e}")
    sys.exit(1)

# Verify capability flag on instance
if nf.uses_c_dispatch:
    print(f"Error: uses_c_dispatch should be False, got {nf.uses_c_dispatch}")
    sys.exit(1)

# 4. Verify no custom parser side effects (optional, but ensure no errors logged)
# (We assume silence is success here as per the design)

print("SUCCESS")
"""

    # Ensure we run from the project root or where 'numeta' package is resolvable
    cwd = os.getcwd()
    env = os.environ.copy()
    if "PYTHONPATH" not in env:
        env["PYTHONPATH"] = cwd
    else:
        env["PYTHONPATH"] = f"{cwd}:{env['PYTHONPATH']}"

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, cwd=cwd
    )

    if result.returncode != 0:
        pytest.fail(f"Fallback script failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")

    if "SUCCESS" not in result.stdout:
        pytest.fail(f"Fallback script did not report success:\nSTDOUT: {result.stdout}")


def test_import_without_signature_extension_does_not_compile(tmp_path):
    source = Path(__file__).resolve().parents[1] / "numeta"
    package = tmp_path / "numeta"
    shutil.copytree(
        source,
        package,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so", "*.o"),
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pathlib, numeta; "
            "p = pathlib.Path(numeta.__file__).parent; "
            "assert not list(p.glob('*.so')); assert not list(p.glob('*.o'))",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
