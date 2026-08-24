import numeta as nm
import numpy as np
from pathlib import Path


def _read_generated_source(tmp_path: Path, func_name: str, backend: str) -> str:
    suffix = "f90" if backend == "fortran" else "c"
    return (tmp_path / f"{func_name}_src.{suffix}").read_text()


def test_scalar_accumulator_uses_explicit_runtime_updates(tmp_path, backend):
    func_name = f"rebind_accumulator_{backend}"

    @nm.jit(backend=backend, directory=str(tmp_path), namer=lambda *spec: func_name)
    def reduce_kernel(result_stack, shell_result_stack):
        for i_batch in nm.range(result_stack.shape[0]):
            for result_index in nm.range(result_stack.shape[1]):
                for flat_index in nm.range(result_stack.shape[2]):
                    value = nm.scalar(shell_result_stack[i_batch, result_index, 0, flat_index])
                    for i_thread in nm.range(1, shell_result_stack.shape[2]):
                        value[:] += shell_result_stack[i_batch, result_index, i_thread, flat_index]
                    result_stack[i_batch, result_index, flat_index] = value

    shell_result_stack = np.zeros((2, 3, 4, 5), dtype=np.float64, order="F")
    result_stack = np.zeros((2, 3, 5), dtype=np.float64, order="F")

    for i_batch in range(shell_result_stack.shape[0]):
        for result_index in range(shell_result_stack.shape[1]):
            for i_thread in range(shell_result_stack.shape[2]):
                for flat_index in range(shell_result_stack.shape[3]):
                    shell_result_stack[i_batch, result_index, i_thread, flat_index] = (
                        1000.0 * i_batch + 100.0 * result_index + 10.0 * i_thread + flat_index
                    )

    reduce_kernel(result_stack, shell_result_stack)

    source = _read_generated_source(tmp_path, func_name, backend)
    if backend == "fortran":
        assert "do fc_i4 = 1_c_int64_t" in source
        assert "end do\n                result_stack" in source
        assert "fc_s" in source
    else:
        assert "for (fc_i4 = 1;" in source
        assert "fc_s" in source


def test_scalar_accumulator_with_explicit_storage(backend):
    @nm.jit(backend=backend)
    def reduce_kernel(result_stack, shell_result_stack):
        for i_batch in nm.range(result_stack.shape[0]):
            for result_index in nm.range(result_stack.shape[1]):
                for flat_index in nm.range(result_stack.shape[2]):
                    value = nm.scalar(nm.f8)
                    value[:] = shell_result_stack[i_batch, result_index, 0, flat_index]
                    for i_thread in nm.range(1, shell_result_stack.shape[2]):
                        value[:] += shell_result_stack[i_batch, result_index, i_thread, flat_index]
                    result_stack[i_batch, result_index, flat_index] = value

    shell_result_stack = np.zeros((2, 3, 4, 5), dtype=np.float64, order="F")
    result_stack = np.zeros((2, 3, 5), dtype=np.float64, order="F")

    for i_batch in range(shell_result_stack.shape[0]):
        for result_index in range(shell_result_stack.shape[1]):
            for i_thread in range(shell_result_stack.shape[2]):
                for flat_index in range(shell_result_stack.shape[3]):
                    shell_result_stack[i_batch, result_index, i_thread, flat_index] = (
                        1000.0 * i_batch + 100.0 * result_index + 10.0 * i_thread + flat_index
                    )

    reduce_kernel(result_stack, shell_result_stack)

    expected = shell_result_stack.sum(axis=2)
    np.testing.assert_allclose(result_stack, expected)
