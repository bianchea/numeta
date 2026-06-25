from .expression_node import ExpressionNode
from numeta.ast.tools import check_node
from numeta.exceptions import NumetaTypeError, raise_with_source


class BinaryOperationNode(ExpressionNode):
    __slots__ = ["op", "left", "right", "_shape_cache"]

    def __init__(self, left, op, right):
        super().__init__()
        self.op = op
        self.left = check_node(left)
        self.right = check_node(right)
        self._shape_cache = None

    @property
    def dtype(self):
        """Return the DataType of the expression."""
        from numeta.type_rules import binary_result_dtype

        left_dtype = getattr(self.left, "dtype", None)
        right_dtype = getattr(self.right, "dtype", None)
        return binary_result_dtype(left_dtype, right_dtype, self.op)

    @property
    def _shape(self):
        """Return the shape of the expression if any."""
        cached_shape = self._shape_cache
        if cached_shape is not None:
            return cached_shape

        left_shape = self.left._shape
        right_shape = self.right._shape

        # This is a simplification. It doesn't handle broadcasting correctly.
        # For now, we'll just return the shape of the left operand.
        if left_shape.is_scalar:
            result_shape = right_shape
        elif right_shape.is_scalar:
            result_shape = left_shape
        else:
            result_shape = left_shape

        self._shape_cache = result_shape
        return result_shape

    def get_with_updated_variables(self, variables_couples):
        return BinaryOperationNode(
            self.left.get_with_updated_variables(variables_couples),
            self.op,
            self.right.get_with_updated_variables(variables_couples),
        )

    def extract_entities(self):
        yield from self.left.extract_entities()
        yield from self.right.extract_entities()


class BinaryOperationNodeNoPar(BinaryOperationNode):
    pass


class EqBinaryNode(BinaryOperationNode):
    __slots__ = ["op", "left", "right", "_shape_cache"]

    def __init__(self, left, right):
        # faster than calling super().__init__(left, '.eq.', right)
        ExpressionNode.__init__(self)
        self.op = ".eq."
        self.left = check_node(left)
        self.right = check_node(right)
        self._shape_cache = None

    def __bool__(self):
        try:
            return self.left.name == self.right.name
        except AttributeError:
            raise_with_source(
                NumetaTypeError,
                f"Do not use '==' operator for non-NamedEntity: {type(self.left)} and {type(self.right)}",
                source_node=self,
            )
        # TODO: Too slow


class NeBinaryNode(BinaryOperationNode):
    def __init__(self, left, right):
        ExpressionNode.__init__(self)
        self.op = ".ne."
        # self.left = left
        # self.right = right
        self.left = check_node(left)
        self.right = check_node(right)
        self._shape_cache = None

    def __bool__(self):
        try:
            return self.left.name != self.right.name
        except AttributeError:
            raise_with_source(
                NumetaTypeError,
                f"Do not use '!=' operator for non-NamedEntity: {type(self.left)} and {type(self.right)}",
                source_node=self,
            )

        # TODO: Too slow
