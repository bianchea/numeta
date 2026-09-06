"""Structured OpenMP options and data-sharing inference."""

from .ast.variable import Variable
from .exceptions import NumetaError, raise_with_source
from .trace_validation import storage_base


def parallel_options(
    *, default="none", schedule="static", chunk=None, num_threads=None, private=None, shared=None
):
    if default not in {"none", "private", "shared"}:
        raise ValueError("OpenMP default must be 'none', 'private', or 'shared'")
    if schedule not in {"static", "dynamic", "guided", "auto", "runtime"}:
        raise ValueError("OpenMP schedule must be static, dynamic, guided, auto, or runtime")
    for name, value in (("chunk", chunk), ("num_threads", num_threads)):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"OpenMP {name} must be a positive compile-time integer")
    if chunk is not None and schedule in {"auto", "runtime"}:
        raise ValueError(f"OpenMP schedule={schedule!r} does not accept a chunk")
    result = dict(default=default, schedule=schedule, chunk=chunk, num_threads=num_threads)
    for name, values in (("private", private), ("shared", shared)):
        result[name] = []
        for value in () if values is None else values:
            variable = storage_base(value)
            if variable is None:
                raise TypeError(f"OpenMP {name} entries must refer to storage")
            if all(variable is not existing for existing in result[name]):
                result[name].append(variable)
    if {id(v) for v in result["private"]} & {id(v) for v in result["shared"]}:
        raise ValueError("OpenMP shared and private clauses overlap")
    return result


def finalize_parallel(loop):
    from .ast.statements import Assignment, For

    options = loop.openmp
    captured = {}
    private = {id(v): v for v in options["private"]}
    shared = {id(v): v for v in options["shared"]}
    private[id(loop.iterator)] = loop.iterator
    writes = []

    def statements(body):
        for statement in body:
            yield statement
            if hasattr(statement, "scope"):
                yield from statements(statement.scope.body)
            for branch in getattr(statement, "orelse", ()):
                yield from statements(branch.scope.body)

    for entity in loop.extract_entities():
        if isinstance(entity, Variable) and not entity.parameter and entity.parent is None:
            captured[id(entity)] = entity
    for statement in statements(loop.scope.body):
        if isinstance(statement, For):
            private[id(statement.iterator)] = statement.iterator
        if isinstance(statement, Assignment):
            target = storage_base(statement.target)
            if target is not None:
                writes.append((target, statement))
    for key, variable in list(captured.items()):
        if variable._trace_sequence > loop._trace_sequence:
            private[key] = variable
        elif key not in private:
            if options["default"] == "private" and key not in shared:
                private[key] = variable
            else:
                shared[key] = variable
    for variable in list(shared.values()):
        if not variable._shape.is_unknown:
            for dimension in variable._shape.iter_dims():
                if hasattr(dimension, "extract_entities"):
                    for entity in dimension.extract_entities():
                        if isinstance(entity, Variable):
                            shared[id(entity)] = entity
    for key in private:
        shared.pop(key, None)
    for variable, statement in writes:
        if (
            variable._shape.is_scalar
            and id(variable) not in private
            and not getattr(statement, "atomic", False)
        ):
            raise_with_source(
                NumetaError,
                f"Parallel loop writes captured scalar {variable.name!r}. Declare it private, "
                "use nm.omp.atomic_update_add, or move its storage inside the loop.",
                statement,
            )
    options["private"] = list(private.values())
    options["shared"] = list(shared.values())
    options["default"] = "none"


def render_parallel(options, name=lambda value: value):
    clauses = ["default(none)"]
    schedule = options["schedule"]
    if options["chunk"] is not None:
        schedule += f", {options['chunk']}"
    clauses.append(f"schedule({schedule})")
    if options["num_threads"] is not None:
        clauses.append(f"num_threads({options['num_threads']})")
    for sharing in ("private", "shared"):
        if options[sharing]:
            clauses.append(f"{sharing}({', '.join(name(v) for v in options[sharing])})")
    return " ".join(clauses)
