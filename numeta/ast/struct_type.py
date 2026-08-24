from .nodes import NamedEntity
from numeta.exceptions import raise_with_source
from numeta.array_shape import ArrayShape, SCALAR


class StructType(NamedEntity):
    """
    A structured type. Used to define structs.

    Parameters
    ----------
    name : str
        The name of the struct type.
    fields : list of tuples
        The fields of the struct type, each tuple containing the name, datatype, and dimension.
    """

    def __init__(self, name, fields):
        super().__init__(name)
        self.fields = [
            (field_name, dtype, SCALAR if len(field) == 2 else field[2])
            for field in fields
            for field_name, dtype in [field[:2]]
        ]
        self.fields = [
            (field_name, dtype, shape if isinstance(shape, ArrayShape) else ArrayShape(shape))
            for field_name, dtype, shape in self.fields
        ]
        for name, _, shape in self.fields:
            if shape.has_comptime_undefined_dims():
                raise_with_source(
                    ValueError,
                    f"Struct type '{name}' cannot have compile-time undefined dimensions.",
                    source_node=self,
                )
        self.parent = None

    def get_declaration(self):
        from .statements import StructTypeDeclaration

        return StructTypeDeclaration(self)
