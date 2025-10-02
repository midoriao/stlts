# stlts

This is a Python package for synthesizing a trace that satisfies a given Signal Temporal Logic (STL) formula. The package implements the MILP-based synthesis algorithm proposed in the paper:

> Sota Sato, Jie An, Zhenya Zhang, Ichiro Hasuo. Optimization-Based Model Checking and Trace Synthesis for Complex STL Specifications. 36th International Conference on Computer-Aided Verification, 2024 [[doi](https://doi.org/10.1007/978-3-031-65633-0_13)] [[arXiv](https://arxiv.org/abs/2408.06983)]


### Requirements

- Python with version 3.8 or higher
- License for [Gurobi optimizer](https://www.gurobi.com/)

### Install

```sh
pip install stlts
```

Optionally, install from the cloned repo (requires pip version 22.0 or higher):

```sh
git clone https://github.com/midoriao/stlts
cd stlts
pip install -e .
```

### Usage

You can run a synthesis for example model and spec:

```python
import gurobipy as gp
from stlts import benchmarks

milp = gp.Model()

benchmark_name = 'rnc1'
bound = 5

prob = benchmarks.get_benchmark(milp, benchmark_name, N=bound, delta=0.1)

prob.search_satisfaction()

if prob.has_solution:
    print(prob.get_trace_result(interpolation=False))
else:
    print(f'No trace found with bound {bound}')
```

To try another  STL formula, it can be specified in a DSL style.

```python
import gurobipy as gp
from stlts import benchmarks
from stlts.linear_expression import LinearExpression as L
from stlts.stl import Atomic, BoundedAlw, Ev


milp = gp.Model()

benchmark_name = 'rnc1'
bound = 5

prob = benchmarks.get_benchmark(milp, benchmark_name, N=bound, delta=0.1)

# Atomic proposition is given in linear inequality form
danger = Atomic(L.f(1.0, 'x1', -1.0, 'x2') <= 10)
stl_spec = Ev(BoundedAlw((0, 5), danger))

prob.initialize_milp_formulation(stl_spec)
prob.search_satisfaction()
```

Here’s a list of the supported STL operators:

- `Atomic(p)`: Atomic proposition. Its content `p` is given in linear inequality form.
- `And(psi1, psi2, ...)`: This operator specifies that all formulas `psi1, psi2, ...` hold.
- `Or(psi1, psi2, ...)`: This operator specifies that at least one formula out of `psi1, psi2, ...` holds.
- `BoundedAlw([a,b], psi)`: This operator specifies that a property must hold at all times within a given interval `[a,b]`.
- `Alw(psi)`: This operator specifies that a property must hold at all times. Semantically it is equivalent to `BoundedAlw([0, infty], psi)`.
- `BoundedEv([a,b], psi)`, `Ev(psi)`: This operator specifies that a property must hold become true at some point within a given interval.
- `BoundedUntil([a,b], psi1, psi2)`, `Until(psi1, psi2)`: This operator specifies that `psi1` must be true until `psi2`becomes true within an interval `[a, b]`.
- `BoundedRelease([a,b], psi1, psi2)`, `Release(psi1, psi2)`: This operator is dual to the Until operator and specifies that `psi2` must hold true until and including when `psi1` becomes true, within an interval `[a,b]`.

Note that we do not provide `Not` operator, since our formula must be in NNF (negation-normal form). Instead, we provide a method `phi.negation()` to get an equivalent formula in NNF to the negation of `phi`.

### Supplementary Material

The code for replicating the experiments in the paper is available at [Zenodo](https://doi.org/10.5281/zenodo.11001313).
