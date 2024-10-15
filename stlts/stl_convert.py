from __future__ import annotations

from .stl import (
    Alw,
    And,
    Atomic,
    Bottom,
    BoundedAlw,
    BoundedEv,
    Ev,
    Or,
    Release,
    StlFormula,
    Top,
    Until,
)


def str_interval(interval: tuple[float, float]) -> str:
    return f'[{interval[0]}, {interval[1]}]'


def str_atomic(
    variables,
    coefficients,
    constant,
    sense,
) -> str:
    combined = ' + '.join(
        f"{coeff:.5} * {var}" for var, coeff in zip(variables, coefficients)
    )
    if coefficients == [1] and constant == 0:
        return f'{variables[0]} {sense} 0'
    elif coefficients == [1]:
        return f'{variables[0]} {sense} {-constant:.5}'
    elif constant == 0:
        return f'{combined} {sense} 0'
    else:
        return f"{combined} + {constant:.5} {sense} 0"


def to_latex(phi: StlFormula) -> str:
    if isinstance(phi, Atomic):
        sanitized_name = str_atomic(
            variables=phi.predicate.inequality.expr.variables,
            coefficients=phi.predicate.inequality.expr.coefficients,
            constant=phi.predicate.inequality.expr.constant,
            sense=phi.predicate.inequality.sense,
        ).replace('_', '\\_')
        return f'\\text{{{sanitized_name}}}'
    elif isinstance(phi, Top):
        return '\\top'
    elif isinstance(phi, Bottom):
        return '\\bot'
    elif isinstance(phi, And):
        content = '\\wedge'.join([f'{to_latex(child)}' for child in phi.children])
        return f'({content})'
    elif isinstance(phi, Or):
        content = '\\vee'.join([f'{to_latex(child)}' for child in phi.children])
        return f'({content})'
    elif isinstance(phi, Alw):
        return f'\\Box {to_latex(phi.children[0])}'
    elif isinstance(phi, Ev):
        return f'\\Diamond {to_latex(phi.children[0])}'
    elif isinstance(phi, Until):
        return (
            f'({to_latex(phi.children[0])} \\mathop{{\\mathcal{{U}}}}'
            f' {to_latex(phi.children[1])})'
        )
    elif isinstance(phi, Release):
        return (
            f'({to_latex(phi.children[0])} \\mathop{{\\mathcal{{R}}}}'
            f' {to_latex(phi.children[1])})'
        )
    elif isinstance(phi, BoundedAlw):
        return f'\\Box_{{{str_interval(phi.interval)}}} {to_latex(phi.child)}'
    elif isinstance(phi, BoundedEv):
        return f'\\Diamond_{{{str_interval(phi.interval)}}} {to_latex(phi.child)}'
    else:
        raise ValueError(f"Unknown STL formula type: {type(phi)}")


def to_breach(phi: StlFormula) -> str:
    if isinstance(phi, Atomic):
        return str_atomic(
            variables=[v + '[t]' for v in phi.predicate.inequality.expr.variables],
            coefficients=phi.predicate.inequality.expr.coefficients,
            constant=phi.predicate.inequality.expr.constant,
            sense=phi.predicate.inequality.sense + '=',
        )
    elif isinstance(phi, Top):
        return 'true'
    elif isinstance(phi, Bottom):
        return 'false'
    elif isinstance(phi, And):
        return f'({" and ".join([to_breach(child) for child in phi.children])})'
    elif isinstance(phi, Or):
        return f'({" or ".join([to_breach(child) for child in phi.children])})'
    elif isinstance(phi, Alw):
        return f'(alw({to_breach(phi.children[0])}))'
    elif isinstance(phi, Ev):
        return f'(ev({to_breach(phi.children[0])}))'
    elif isinstance(phi, Until):
        return f'({to_breach(phi.children[0])} until {to_breach(phi.children[1])})'
    elif isinstance(phi, Release):
        return f'({to_breach(phi.children[0])} release {to_breach(phi.children[1])})'
    elif isinstance(phi, BoundedAlw):
        return f'(alw_{str_interval(phi.interval)} ({to_breach(phi.child)}))'
    elif isinstance(phi, BoundedEv):
        return f'(ev_{str_interval(phi.interval)} ({to_breach(phi.child)}))'
    else:
        raise ValueError(f"Unknown STL formula type: {type(phi)}")


def to_stlmc(phi: StlFormula) -> str:
    if isinstance(phi, Atomic):
        return str_atomic(
            variables=phi.predicate.inequality.expr.variables,
            coefficients=phi.predicate.inequality.expr.coefficients,
            constant=phi.predicate.inequality.expr.constant,
            sense=phi.predicate.inequality.sense + '=',
        )
    elif isinstance(phi, Top):
        return 'true'
    elif isinstance(phi, Bottom):
        return 'false'
    elif isinstance(phi, And):
        return f'({" and ".join([to_stlmc(child) for child in phi.children])})'
    elif isinstance(phi, Or):
        return f'({" or ".join([to_stlmc(child) for child in phi.children])})'
    elif isinstance(phi, Alw):
        return f'([][0, inf)({to_stlmc(phi.children[0])}))'
    elif isinstance(phi, Ev):
        return f'(<>[0, inf)({to_stlmc(phi.children[0])}))'
    elif isinstance(phi, Until):
        return f'({to_breach(phi.children[0])} U[0,inf) {to_stlmc(phi.children[1])})'
    elif isinstance(phi, Release):
        return f'({to_breach(phi.children[0])} R[0,inf) {to_stlmc(phi.children[1])})'
    elif isinstance(phi, BoundedAlw):
        return f'([]{str_interval(phi.interval)} ({to_stlmc(phi.child)}))'
    elif isinstance(phi, BoundedEv):
        return f'(<>{str_interval(phi.interval)} ({to_stlmc(phi.child)}))'
    else:
        raise ValueError(f"Unknown STL formula type: {type(phi)}")
