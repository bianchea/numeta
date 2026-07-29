from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IRNode:
    source: object | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IRType:
    name: str
    kind: str | None = None
    bitwidth: int | None = None
    struct: "IRStruct | None" = None
    is_vector: bool = False
    vector_lanes: int | None = None
    vector_base: "IRType | None" = None


@dataclass(frozen=True)
class IRShape:
    rank: int | None
    dims: tuple[Any, ...] | None
    order: str
    index_base: int = 0


@dataclass(frozen=True)
class IRValueType:
    dtype: IRType
    shape: IRShape | None


@dataclass
class IRVar(IRNode):
    name: str = ""
    vtype: IRValueType | None = None
    intent: str | None = None
    storage: str = "value"
    is_const: bool = False
    is_arg: bool = False
    allocatable: bool = False
    pointer: bool = False
    target: bool = False
    parameter: bool = False
    c_static: bool = False
    bind_c: bool = False
    assign: object | None = None
    pass_by_value: bool = False


@dataclass
class IRStruct(IRNode):
    name: str = ""
    members: list[tuple[str, IRValueType]] = field(default_factory=list)


@dataclass
class IRModule(IRNode):
    name: str = ""
    globals: list[IRVar | IRStruct] = field(default_factory=list)
    procedures: list[IRProcedure] = field(default_factory=list)


@dataclass
class IRProcedure(IRNode):
    name: str = ""
    args: list[IRVar] = field(default_factory=list)
    locals: list[IRVar] = field(default_factory=list)
    body: list[IRNode] = field(default_factory=list)
    result: IRVar | None = None


@dataclass
class IRStmt(IRNode):
    pass


@dataclass
class IRExpr(IRNode):
    vtype: IRValueType | None = None


@dataclass
class IRLiteral(IRExpr):
    value: Any = None


@dataclass
class IRVarRef(IRExpr):
    var: IRVar | None = None


@dataclass
class IRBinary(IRExpr):
    op: str = ""
    left: IRExpr | None = None
    right: IRExpr | None = None


@dataclass
class IRUnary(IRExpr):
    op: str = ""
    operand: IRExpr | None = None


@dataclass
class IRCall(IRStmt):
    func: IRExpr | None = None
    args: list[IRExpr] = field(default_factory=list)


@dataclass
class IRCallExpr(IRExpr):
    callee: IRExpr | None = None
    args: list[IRExpr] = field(default_factory=list)


@dataclass
class IRSlice(IRNode):
    start: IRExpr | None = None
    stop: IRExpr | None = None
    step: IRExpr | None = None


@dataclass
class IRGetItem(IRExpr):
    base: IRExpr | None = None
    indices: list[IRExpr | IRSlice] = field(default_factory=list)


@dataclass
class IRGetAttr(IRExpr):
    base: IRExpr | None = None
    name: str = ""


@dataclass
class IRIntrinsic(IRExpr):
    name: str = ""
    args: list[IRExpr] = field(default_factory=list)


@dataclass
class IRAssign(IRStmt):
    target: IRExpr | None = None
    value: IRExpr | None = None


@dataclass
class IRIf(IRStmt):
    cond: IRExpr | None = None
    then: list[IRNode] = field(default_factory=list)
    else_: list[IRNode] = field(default_factory=list)


@dataclass
class IRFor(IRStmt):
    var: IRVar | None = None
    start: IRExpr | None = None
    stop: IRExpr | None = None
    step: IRExpr | None = None
    body: list[IRNode] = field(default_factory=list)


@dataclass
class IRWhile(IRStmt):
    cond: IRExpr | None = None
    body: list[IRNode] = field(default_factory=list)


@dataclass
class IRReturn(IRStmt):
    value: IRExpr | None = None


@dataclass
class IRPrint(IRStmt):
    values: list[IRExpr] = field(default_factory=list)


@dataclass
class IRAllocate(IRStmt):
    var: IRExpr | None = None
    dims: list[IRExpr] = field(default_factory=list)


@dataclass
class IRDeallocate(IRStmt):
    var: IRExpr | None = None


@dataclass
class IRSimdStore(IRStmt):
    array: IRExpr | None = None
    index: IRExpr | None = None
    value: IRExpr | None = None
    aligned: bool = False


@dataclass
class IROpaqueExpr(IRExpr):
    payload: object | None = None


@dataclass
class IROpaqueStmt(IRStmt):
    payload: object | None = None
