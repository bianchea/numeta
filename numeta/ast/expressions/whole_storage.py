from .expression_node import ExpressionNode


class WholeStorage(ExpressionNode):
    """An explicit ``[:]`` reference to an expression's complete storage.

    The node exists only while tracing so Python augmented assignment can be
    distinguished from augmented assignment on a bare Variable. Lowering
    normalizes it to the wrapped storage reference.
    """

    def __init__(self, variable):
        super().__init__()
        self.variable = variable

    @property
    def dtype(self):
        return self.variable.dtype

    @property
    def _shape(self):
        return self.variable._shape

    def extract_entities(self):
        yield from self.variable.extract_entities()

    def get_with_updated_variables(self, variables_couples):
        return type(self)(self.variable.get_with_updated_variables(variables_couples))
