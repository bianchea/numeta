from numeta.ast import ExternalNamespace
from numeta.ast import For, Comment
from numeta.ast.function import Function
from numeta.array_shape import SCALAR
from numeta.settings import settings

syntax_settings = settings.syntax


class OmpComment(Comment):
    """
    Hack to handle OpenMP statements.
    """

    def __init__(self, comment, add_to_scope=False):
        super().__init__(comment, add_to_scope=add_to_scope)
        self.prefix = "!$omp "


class OmpFor(For):
    def __init__(
        self,
        *args,
        default="none",
        schedule="static",
        private=None,
        shared=None,
        chunk=None,
        num_threads=None,
        **kwargs,
    ):
        from numeta.parallel import parallel_options

        self.openmp = parallel_options(
            default=default,
            schedule=schedule,
            private=private,
            shared=shared,
            chunk=chunk,
            num_threads=num_threads,
        )
        super().__init__(*args, **kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        if exc_type is None:
            from numeta.parallel import finalize_parallel

            finalize_parallel(self)

    def get_with_updated_variables(self, variables_couples):
        result = super().get_with_updated_variables(variables_couples)
        result.openmp = dict(self.openmp)
        for clause in ("shared", "private"):
            result.openmp[clause] = [
                v.get_with_updated_variables(variables_couples) for v in self.openmp[clause]
            ]
        return result


class omp_get_thread_num(Function):
    @property
    def dtype(self):
        from numeta.datatype import int64

        return syntax_settings.DEFAULT_INT or int64

    @property
    def _shape(self):
        return SCALAR


class omp_get_max_threads(Function):
    @property
    def dtype(self):
        from numeta.datatype import int64

        return syntax_settings.DEFAULT_INT or int64

    @property
    def _shape(self):
        return SCALAR


class omp_get_wtime(Function):
    @property
    def dtype(self):
        from numeta.datatype import float64

        return syntax_settings.DEFAULT_FLOAT or float64

    @property
    def _shape(self):
        return SCALAR


class OmpNamespace(ExternalNamespace):
    def __init__(self):
        super().__init__("omp_lib", None)

        self.procedures["omp_get_thread_num"] = omp_get_thread_num(
            "omp_get_thread_num",
            [],
            parent=self,
        )
        self.procedures["omp_get_max_threads"] = omp_get_max_threads(
            "omp_get_max_threads",
            [],
            parent=self,
        )
        self.procedures["omp_get_wtime"] = omp_get_wtime(
            "omp_get_wtime",
            [],
            parent=self,
        )

    def parallel_for(self, *args, **kwargs):
        return OmpFor(*args, **kwargs)

    def atomic_update_op(self, variable, to_assign, op):
        from numeta.ast.statements import Assignment, Comment
        from numeta.ast.expressions.binary_operation_node import (
            BinaryOperationNodeNoPar,
        )

        Comment(f"$omp atomic update", add_to_scope=True)
        statement = Assignment(
            variable,
            BinaryOperationNodeNoPar(variable, op, to_assign),
            add_to_scope=True,
        )
        statement.atomic = True

    def atomic_update_add(self, variable, to_assign):
        self.atomic_update_op(variable, to_assign, "+")

    def AtomicUpdateAdd(self, variable, to_assign):
        self.atomic_update_op(variable, to_assign, "+")

    def ATOMIC_UPDATE_ADD(self, variable, to_assign):
        self.atomic_update_op(variable, to_assign, "+")

    def atomic_update_sub(self, variable, to_assign):
        self.atomic_update_op(variable, to_assign, "-")

    def AtomicUpdateSub(self, variable, to_assign):
        self.atomic_update_op(variable, to_assign, "-")

    def ATOMIC_UPDATE_SUB(self, variable, to_assign):
        self.atomic_update_op(variable, to_assign, "-")

    def atomic_update_mul(self, variable, to_assign):
        self.atomic_update_op(variable, to_assign, "*")

    def AtomicUpdateMul(self, variable, to_assign):
        self.atomic_update_op(variable, to_assign, "*")

    def ATOMIC_UPDATE_MUL(self, variable, to_assign):
        self.atomic_update_op(variable, to_assign, "*")

    def atomic_update_div(self, variable, to_assign):
        self.atomic_update_op(variable, to_assign, "/")

    def AtomicUpdateDiv(self, variable, to_assign):
        self.atomic_update_op(variable, to_assign, "/")

    def ATOMIC_UPDATE_DIV(self, variable, to_assign):
        self.atomic_update_op(variable, to_assign, "/")


class _LazyOmp:
    _instance = None

    def __getattr__(self, name):
        if self._instance is None:
            self._instance = OmpNamespace()
        return getattr(self._instance, name)


omp = _LazyOmp()
