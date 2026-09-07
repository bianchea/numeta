import ctypes.util
from pathlib import Path
import re

import numpy as np
import pytest

import numeta as nm


def example(section):
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    body = readme.split(section, 1)[1]
    return re.search(r"```python\n(.*?)```", body, flags=re.S).group(1)


@pytest.mark.parametrize(
    "section",
    [
        "## Quick Start\n",
        "### Conditional Statements\n",
        "### Parallel Loop Example\n",
        "### Conversion and snapshots\n",
    ],
)
def test_readme_control_examples(section, backend):
    previous = nm.settings.default_backend
    nm.settings.set_default_backend(backend)
    try:
        namespace = {}
        exec(compile(example(section), str(Path(__file__)), "exec"), namespace)
        if "conditional_example" in namespace:
            out = np.zeros(4)
            namespace["conditional_example"](4, out)
            np.testing.assert_array_equal(out, [0, 1, 2, 2])
        if "pmul" in namespace:
            a = np.arange(6.0).reshape(2, 3)
            b = np.arange(12.0).reshape(3, 4)
            out = np.zeros((2, 4))
            namespace["pmul"](a, b, out)
            np.testing.assert_allclose(out, a @ b)
    finally:
        nm.settings.set_default_backend(previous)


def test_readme_blas_lp64_rectangular_example(backend):
    if ctypes.util.find_library("blas") is None:
        pytest.skip("LP64 BLAS is not installed")
    previous = nm.settings.default_backend
    nm.settings.set_default_backend(backend)
    try:
        namespace = {}
        exec(
            compile(example("### How to Link an External Library\n"), str(Path(__file__)), "exec"),
            namespace,
        )
        a = np.asfortranarray(np.arange(6.0).reshape(2, 3))
        b = np.asfortranarray(np.arange(12.0).reshape(3, 4))
        out = np.zeros((2, 4), order="F")
        namespace["matmul"](a, b, out)
        np.testing.assert_allclose(out, a @ b)
    finally:
        nm.settings.set_default_backend(previous)
