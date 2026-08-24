"""Compatibility errors for the removed source-dependent control-flow API."""

_MESSAGE = (
    "nm.cond()/nm.endif() were removed because they inspected and re-executed Python "
    "source. Use explicit Numeta control flow instead, for example:\n\n"
    "    with nm.If(condition):\n"
    "        out[:] = value\n"
    "    with nm.Else():\n"
    "        out[:] = other\n"
)


def cond(condition):
    raise RuntimeError(_MESSAGE)


def endif():
    raise RuntimeError(_MESSAGE)
