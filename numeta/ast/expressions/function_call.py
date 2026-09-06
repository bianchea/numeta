from numeta.ast.tools import check_node
from .expression_node import ExpressionNode


class FunctionCall(ExpressionNode):
    __slots__ = ["function", "arguments"]

    def __init__(self, function, *arguments):
        super().__init__()
        self.function = function
        self.arguments = [check_node(arg) for arg in arguments]
        from numeta.builder_helper import BuilderHelper

        builder = BuilderHelper.current_builder
        if builder is not None and not getattr(function, "pure", False):
            builder.effectful_calls.append(self)

    @property
    def dtype(self):
        return self.function.dtype

    @property
    def _shape(self):
        return self.function._shape

    def extract_entities(self):
        yield self.function
        for arg in self.arguments:
            yield from arg.extract_entities()

    def get_with_updated_variables(self, variables_couples):
        new_args = [arg.get_with_updated_variables(variables_couples) for arg in self.arguments]
        return FunctionCall(self.function, *new_args)
