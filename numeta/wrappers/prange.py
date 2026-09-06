from __future__ import annotations
from typing import Any, Iterator

from numeta.builder_helper import BuilderHelper
from numeta.settings import settings
from numeta.fortran.external_modules.omp import omp
from numeta.ast.variable import Variable


def prange(*args, **kwargs: Any) -> Iterator[Variable]:
    if len(args) == 1:
        start = 0
        stop = args[0]
        step = None
    elif len(args) == 2:
        start = args[0]
        stop = args[1]
        step = None
    elif len(args) == 3:
        start = args[0]
        stop = args[1]
        step = args[2]
    else:
        raise ValueError("Invalid number of arguments")

    builder = BuilderHelper.get_current_builder()
    I = builder.generate_local_variables("fc_i", dtype=settings.syntax.DEFAULT_INT)

    loop = omp.parallel_for(I, start, stop - 1, step=step, **kwargs)
    builder.pending_iterators[id(loop)] = loop
    with loop:
        yield I
        builder.pending_iterators.pop(id(loop))
