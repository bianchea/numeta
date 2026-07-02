from .nodes import NamedEntity
from .expressions import ExpressionNode
from numeta.array_shape import ArrayShape, SCALAR
from numeta.settings import settings


class Variable(NamedEntity, ExpressionNode):
    def __init__(
        self,
        name,
        dtype=None,
        shape=SCALAR,
        intent=None,
        pointer=False,
        target=False,
        allocatable=False,
        parameter=False,
        assign=None,
        parent=None,
        bind_c=False,
        use_c_types=False,
        pass_by_value=None,
        c_const=False,
        c_restrict=False,
        c_volatile=False,
    ):
        # Note: NamedEntity.__init__ calls Node.__init__ which captures source location
        super().__init__(name, parent=parent)

        if dtype is not None:
            from numeta.datatype import get_datatype

            self.__dtype = get_datatype(dtype)
        else:
            self.__dtype = None

        self.use_c_types = use_c_types
        if not isinstance(shape, ArrayShape):
            self.__shape = ArrayShape(shape, fortran_order=True)
        else:
            self.__shape = shape
        self.allocatable = allocatable
        self.parameter = parameter
        self.assign = assign
        self.intent = intent
        self.pointer = pointer
        self.target = target
        # Note that bind c make the variable global
        self.bind_c = bind_c
        self.pass_by_value = pass_by_value
        self.c_const = bool(c_const)
        self.c_restrict = bool(c_restrict)
        self.c_volatile = bool(c_volatile)

        from .namespace import Namespace

        if isinstance(self.parent, Namespace):
            self.parent.add_variable(self)

    @property
    def dtype(self):
        return self.__dtype

    @property
    def _shape(self):
        return self.__shape

    def _set_shape(self, shape):
        if not isinstance(shape, ArrayShape):
            self.__shape = ArrayShape(shape, fortran_order=self._shape.fortran_order)
        else:
            self.__shape = shape

    def get_with_updated_variables(self, variables_couples):
        for old_variable, new_variable in variables_couples:
            if old_variable.name == self.name:
                return new_variable
        return self

    def get_declaration(self):
        from .statements import VariableDeclaration

        return VariableDeclaration(self)

    @property
    def real(self):
        from .expressions import Re

        return Re(self)

    @real.setter
    def real(self, value):
        from .expressions import Re
        from .statements import Assignment

        return Assignment(Re(self), value)

    @property
    def imag(self):
        from .expressions import Im

        return Im(self)

    @imag.setter
    def imag(self, value):
        from .expressions import Im
        from .statements import Assignment

        return Assignment(Im(self), value)

    @property
    def shape(self):
        return self._shape.as_tuple()

    def __setitem__(self, key, value):
        """Materialize assignment only through indexed or sliced writes.

        Numeta records generated assignments when Python executes ``variable[...] = value``
        or augmented forms such as ``variable[...] += value`` that lower to ``__setitem__``.
        Bare rebinding like ``variable += value`` does not materialize an assignment.
        """
        from .statements import Assignment

        if isinstance(key, slice) and key.start is None and key.stop is None and key.step is None:
            # if the variable is assigned to itself, do nothing, needed for the += and -= operators
            if self is value:
                return
            Assignment(self, value)
        else:
            Assignment(self[key], value)

    def copy(self):
        return Variable(
            self.name,
            dtype=self.dtype,
            shape=self._shape,
            intent=self.intent,
            pointer=self.pointer,
            target=self.target,
            allocatable=self.allocatable,
            parameter=self.parameter,
            assign=self.assign,
            parent=self.parent,
            use_c_types=self.use_c_types,
            pass_by_value=self.pass_by_value,
            c_const=self.c_const,
            c_restrict=self.c_restrict,
            c_volatile=self.c_volatile,
        )

    @property
    def pass_by_value(self):
        if self.__pass_by_value is not None:
            return self.__pass_by_value
        return settings.syntax.force_value and self._shape.is_scalar and self.intent == "in"

    @pass_by_value.setter
    def pass_by_value(self, value):
        if value is not None and not isinstance(value, bool):
            raise TypeError("pass_by_value must be a bool or None")
        if value is True:
            if not self._shape.is_scalar:
                raise ValueError("pass_by_value=True is only valid for scalar variables")
        self.__pass_by_value = value
