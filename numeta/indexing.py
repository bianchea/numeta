from numeta.settings import settings


def get_slice_dim(slice_, max_dim):
    start = slice_.start
    if start is None:
        start = settings.syntax.array_lower_bound
    stop = slice_.stop
    if stop is None and max_dim is not None:
        stop = max_dim
    if stop is None:
        return None
    span = stop - start + settings.syntax.array_lower_bound
    step = slice_.step
    if step is None:
        return span
    if isinstance(step, int) and step <= 0:
        raise NotImplementedError("Negative step slicing is not implemented for shape extraction")
    return (span + step - 1) // step


def merge_slices(base_slice, key):
    lbound = settings.syntax.array_lower_bound

    if base_slice.start is None and base_slice.stop is None:
        return slice(key.start, key.stop, None)

    base_start = base_slice.start if base_slice.start is not None else lbound

    if key.start is None:
        new_start = base_slice.start
    elif base_slice.start is None:
        new_start = key.start
    else:
        new_start = base_start + key.start - lbound

    if key.stop is None:
        new_stop = base_slice.stop
    elif base_slice.start is None:
        new_stop = key.stop
    else:
        new_stop = base_start + key.stop - lbound

    return slice(new_start, new_stop, None)


def merge_scalar_index(base_slice, key):
    lbound = settings.syntax.array_lower_bound
    base_start = base_slice.start if base_slice.start is not None else lbound
    return base_start + key - lbound


def to_fortran_index(index):
    shift = 1 - settings.syntax.array_lower_bound
    if shift == 0:
        return index
    if hasattr(index, "value") and isinstance(index.value, int):
        return int(index.value) + shift
    if isinstance(index, int):
        return index + shift
    return index + shift


def to_fortran_slice_start(start):
    if start is None:
        return 1
    return to_fortran_index(start)


def to_fortran_slice_stop(stop):
    if stop is None:
        return None
    shift = (0 if settings.syntax.c_like_bounds else 1) - settings.syntax.array_lower_bound
    if shift == 0:
        return stop
    if hasattr(stop, "value") and isinstance(stop.value, int):
        return int(stop.value) + shift
    if isinstance(stop, int):
        return stop + shift
    return stop + shift
