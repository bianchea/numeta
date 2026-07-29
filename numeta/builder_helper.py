from .ast import Variable, Scope, Procedure
from .exceptions import NumetaError
from .settings import settings

import numpy as np


class BuilderHelper:
    current_builder = None

    @classmethod
    def set_current_builder(cls, builder):
        cls.current_builder = builder

    @classmethod
    def get_current_builder(cls):
        if cls.current_builder is None:
            raise NumetaError("The current builder is not initialized")
        return cls.current_builder

    def __init__(self, numeta_function, symbolic_function, signature):
        self.numeta_function = numeta_function
        self.symbolic_function = symbolic_function
        self.signature = signature

        self.prefix_counter = {}
        self.allocated_arrays = {}

        if self.numeta_function.backend == "c" or settings.use_numpy_allocator:
            self.allocate_array = self._allocate_array_numpy
            self.deallocate_array = self._deallocate_array_numpy
        else:
            self.allocate_array = self._allocate_array
            self.deallocate_array = self._deallocate_array

    @classmethod
    def generate_local_variables(cls, prefix, allocate=False, name=None, **kwargs):
        """
        TODO:
        TO DEPRECATE in some way, use:
        builder = BuilderHelper.get_current_builder()
        bilder.generate_local_variable(...)
        problem: sometimes is not required to have a builder (i.e. fixed name) so maybe set a default one (?).
        """
        if name is None:
            builder = cls.get_current_builder()
            if prefix not in builder.prefix_counter:
                builder.prefix_counter[prefix] = 0
            builder.prefix_counter[prefix] += 1
            name = f"{prefix}{builder.prefix_counter[prefix]}"
        if allocate:
            builder = cls.get_current_builder()
            return builder.allocate_array(name, **kwargs)
        return Variable(name, **kwargs)

    def generate_local_variable(self, prefix, allocate=False, name=None, **kwargs):
        return BuilderHelper.generate_local_variables(
            prefix, allocate=allocate, name=name, **kwargs
        )

    def _allocate_array(self, name, shape, **kwargs):
        from .ast import Allocate, If, Allocated, Not

        variable = Variable(name, shape=shape, allocatable=True, **kwargs)
        with If(Not(Allocated(variable))):
            Allocate(variable, *shape.iter_dims())
        self.allocated_arrays[name] = variable
        return variable

    def _deallocate_array(self, array):
        from numeta.ast import Deallocate, If, Allocated

        with If(Allocated(array)):
            Deallocate(array)

    def _allocate_array_numpy(self, name, shape, **kwargs):
        from .ast import PointerAssignment
        from .ast.expressions import ArrayConstructor
        from .wrappers import numpy_mem
        from .fortran.external_modules.iso_c_binding import iso_c
        from .array_shape import ArrayShape
        from .datatype import DataType, c_ptr

        # create a c pointer variable that will be also deallocated
        variable_ptr = Variable(f"{name}_c_ptr", dtype=c_ptr)
        self.allocated_arrays[name] = variable_ptr

        dtype = kwargs["dtype"]

        size = dtype.get_nbytes()
        for dim in shape.iter_dims():
            size *= dim

        if isinstance(size, int):
            size = np.intp(size)

        # allocate memory with the numpy allocator
        numpy_mem.numpy_allocate(variable_ptr, size)

        # Fortran is so versone
        # create fortran pointer (with lower bound 1)
        variable_lb1 = Variable(
            f"{name}_f_ptr_lb1", dtype=dtype, shape=ArrayShape((None,)), pointer=True
        )
        # point the fortran pointer to the allocated memory
        iso_c.c_f_pointer(variable_ptr, variable_lb1, ArrayConstructor(size))

        variable = Variable(name, shape=shape, pointer=True, **kwargs)

        # assign the fortran pointer with the proper lower bound
        PointerAssignment(variable, shape, variable_lb1)

        return variable

    def _deallocate_array_numpy(self, array):
        from .wrappers import numpy_mem

        numpy_mem.numpy_deallocate(array)

    @staticmethod
    def _is_trivial_shape_dim(dim) -> bool:
        from .ast.expressions import LiteralNode, GetItem

        if isinstance(dim, int):
            return True
        if isinstance(dim, np.integer):
            return True
        if isinstance(dim, LiteralNode):
            return True
        if isinstance(dim, Variable):
            return True
        if isinstance(dim, GetItem):
            idx = dim.sliced
            return isinstance(dim.variable, Variable) and isinstance(idx, int)
        return False

    def _materialize_non_trivial_shape_dims(self, dims):
        from .array_shape import ArrayShape
        from .ast.tools import check_node
        from .datatype import size_t

        checked_dims = []
        needs_materialization = False
        for dim in dims:
            if dim is None:
                raise NotImplementedError(
                    "Cannot materialize allocation shape with unresolved dimension None."
                )
            if isinstance(dim, int):
                node = dim
            elif isinstance(dim, np.integer):
                node = int(dim)
            else:
                node = check_node(dim)
            checked_dims.append(node)
            if not self._is_trivial_shape_dim(node):
                needs_materialization = True

        if not needs_materialization:
            return tuple(checked_dims)

        rank = len(checked_dims)
        tmp_shape = self.generate_local_variable(
            "nm_shape", dtype=size_t, shape=ArrayShape((rank,))
        )

        for i, node in enumerate(checked_dims):
            tmp_shape[i] = node

        return tuple(tmp_shape[i] for i in range(rank))

    @classmethod
    def normalize_allocation_shape(cls, shape):
        from .array_shape import ArrayShape

        if not isinstance(shape, ArrayShape) or shape.is_unknown or shape.is_shape_vector:
            return shape
        try:
            builder = cls.get_current_builder()
        except NumetaError:
            return shape

        normalized_dims = builder._materialize_non_trivial_shape_dims(shape.iter_dims())
        return ArrayShape(normalized_dims, fortran_order=shape.fortran_order)

    def build(self, *args, **kwargs):
        old_builder = self.current_builder
        self.set_current_builder(self)

        old_scope = Scope.current_scope
        self.symbolic_function.scope.enter()
        try:
            return_variables = self.numeta_function.run_symbolic(*args, **kwargs)

            if return_variables is None:
                return_variables = []
            elif not isinstance(return_variables, (list, tuple)):
                return_variables = [return_variables]

            from .array_shape import ArrayShape
            from .ast import Shape
            from .datatype import DataType, size_t

            ret = []
            for i, var in enumerate(return_variables):
                expr = None
                if isinstance(var, Variable):
                    if var.name in self.allocated_arrays:

                        rank = var._shape.rank
                        shape = Variable(
                            f"fc_out_shape_{i}",
                            dtype=size_t,
                            shape=ArrayShape((rank,)),
                            intent="out",
                        )
                        self.symbolic_function.add_variable(shape)
                        # add to the symbolic function
                        shape[:] = Shape(var)

                        ptr = self.allocated_arrays.pop(var.name)
                        ptr.intent = "out"
                        self.symbolic_function.add_variable(ptr)

                        ret.append((var.dtype, rank))
                    elif var._shape.is_scalar and var.name not in self.symbolic_function.arguments:

                        var.intent = "out"
                        self.symbolic_function.add_variable(var)

                        ret.append((var.dtype, 0))
                    else:
                        if var._shape.is_scalar:
                            tmp = BuilderHelper.generate_local_variables("fc_s", dtype=var.dtype)
                            tmp[:] = var
                            tmp.intent = "out"
                            self.symbolic_function.add_variable(tmp)
                            ret.append((var.dtype, 0))
                            continue
                        expr = var
                else:
                    # it is an expression
                    expr = var

                if expr is not None:
                    # We have to copy the expression in a new array
                    expr_shape = expr._shape
                    if expr_shape.is_scalar:
                        tmp = BuilderHelper.generate_local_variables("fc_s", dtype=expr.dtype)
                        tmp[:] = expr
                        tmp.intent = "out"
                        self.symbolic_function.add_variable(tmp)
                        ret.append((expr.dtype, 0))
                        continue

                    if expr_shape.is_unknown:
                        raise NotImplementedError(
                            "Returning arrays with unknown shape is not supported yet."
                        )

                    rank = expr_shape.rank
                    shape = Variable(
                        f"fc_out_shape_{i}",
                        dtype=size_t,
                        shape=ArrayShape((rank,)),
                        intent="out",
                    )
                    self.symbolic_function.add_variable(shape)

                    normalized_dims = self._materialize_non_trivial_shape_dims(
                        expr_shape.iter_dims()
                    )
                    for j, dim in enumerate(normalized_dims):
                        shape[j] = dim

                    from .wrappers import empty

                    tmp = empty(
                        shape,
                        dtype=expr.dtype,
                        order="F" if expr_shape.fortran_order else "C",
                    )

                    tmp[:] = expr

                    ptr = self.allocated_arrays.pop(tmp.name)
                    ptr.intent = "out"
                    self.symbolic_function.add_variable(ptr)

                    ret.append((expr.dtype, rank))

            for array in self.allocated_arrays.values():
                self.deallocate_array(array)

            return ret
        finally:
            self.symbolic_function.scope.exit()
            Scope.current_scope = old_scope
            self.set_current_builder(old_builder)

    def inline(self, function, *arguments):
        """Inline ``function`` with the given ``arguments`` into the current scope."""
        # Avoid heavy imports at module load time
        if not isinstance(function, Procedure):
            raise TypeError("Unsupported function type for inline call")

        from .ast.tools import check_node

        args = [check_node(arg) for arg in arguments]
        if len(args) != len(function.arguments):
            raise ValueError("Incorrect number of arguments for inlined subroutine")
        variables_couples = list(zip(function.arguments.values(), args))
        for local_variable in function.get_local_variables().values():
            new_local_variable = self.generate_local_variables(
                "nm_inline_",
                dtype=local_variable.dtype,
                shape=local_variable._shape,
                intent=local_variable.intent,
                pointer=local_variable.pointer,
                target=local_variable.target,
                allocatable=local_variable.allocatable,
                parameter=local_variable.parameter,
                assign=local_variable.assign,
                bind_c=local_variable.bind_c,
            )
            variables_couples.append((local_variable, new_local_variable))

        for stmt in function.scope.get_statements():
            Scope.add_to_current_scope(stmt.get_with_updated_variables(variables_couples))
