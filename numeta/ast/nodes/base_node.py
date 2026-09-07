from abc import ABC, abstractmethod
import sys
from itertools import count


_TRACK_SOURCE_LOCATION = True
_TRACE_SEQUENCE = count()
_INTERNAL_AST = "numeta/ast/"
_INTERNAL_WRAPPERS = "numeta/wrappers/"
_INTERNAL_BUILDER = "numeta/builder_helper.py"
_INTERNAL_FUNCTION = "numeta/numeta_function.py"
_INTERNAL_IR = "numeta/ir/"
_INTERNAL_C = "numeta/c/"
_INTERNAL_FORTRAN = "numeta/fortran/"


def set_source_location_tracking(enabled: bool) -> None:
    global _TRACK_SOURCE_LOCATION
    _TRACK_SOURCE_LOCATION = enabled


class Node(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trace_sequence = next(_TRACE_SEQUENCE)
        from numeta.ast.scope import Scope

        self._trace_scope = Scope.current_scope
        if _TRACK_SOURCE_LOCATION:
            self._source_location = self._capture_source_location()
        else:
            self._source_location = None

    def _capture_source_location(self):
        return capture_source_location()

    @property
    def source_location(self):
        """Get the source location where this node was created."""
        return self._source_location

    @abstractmethod
    def extract_entities(self):
        """Extract the nested entities of the node."""


def capture_source_location():
    """Capture the source location (filename, line) where this node was created."""
    if not _TRACK_SOURCE_LOCATION:
        return None
    try:
        frame = sys._getframe(1)
        while frame:
            filename = frame.f_code.co_filename
            is_numeta_internal = (
                _INTERNAL_AST in filename
                or _INTERNAL_WRAPPERS in filename
                or _INTERNAL_BUILDER in filename
                or _INTERNAL_FUNCTION in filename
                or _INTERNAL_IR in filename
                or _INTERNAL_C in filename
                or _INTERNAL_FORTRAN in filename
                or "numeta/datatype.py" in filename
                or "numeta/trace_validation.py" in filename
                or "numeta/exceptions.py" in filename
                or "numeta/type_rules.py" in filename
            )
            if not is_numeta_internal:
                return {
                    "filename": filename,
                    "lineno": frame.f_lineno,
                    "function": frame.f_code.co_name,
                }
            frame = frame.f_back
    except Exception:
        pass
    return None
