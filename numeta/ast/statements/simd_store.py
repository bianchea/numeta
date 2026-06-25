from __future__ import annotations

from numeta.ast.tools import check_node

from .statement import Statement


class VStore(Statement):
    def __init__(self, array, index, value, aligned=False, add_to_scope=True):
        super().__init__(add_to_scope=add_to_scope)
        self.array = check_node(array)
        self.index = check_node(index)
        self.value = check_node(value)
        self.aligned = bool(aligned)

    @property
    def children(self):
        return [self.array, self.index, self.value]

    def get_with_updated_variables(self, variables_couples):
        return type(self)(
            self.array.get_with_updated_variables(variables_couples),
            self.index.get_with_updated_variables(variables_couples),
            self.value.get_with_updated_variables(variables_couples),
            aligned=self.aligned,
            add_to_scope=False,
        )
