# Hippox: High-order Polynomial Projection Operators for JAX

![image](https://user-images.githubusercontent.com/36138121/212599815-02825a92-8d4b-4330-878d-30b32765e345.png)


## What is Hippox?

Hippox provides a simple dataclass for initializing High-order Polynomial Projection Operators (HiPPOs) as parameters in JAX neural network libraries such as Flax and Haiku.

## Example

Here is an example of initializing HiPPO parameters inside a Haiku module:

```python
import haiku as hk 
from hippox.main import Hippo

class MyHippoModule(hk.Module):
    def __init__(self, state_size, measure)
        _hippo = Hippo(state_size=state_size, measure=measure)
        _hippo()

        self._lambda_real = hk.get_parameter(
            'lambda_real',
            shape=[state_size,],
            init = _hippo.lambda_initializer('real')
        )
        self._lambda_imag = hk.get_parameter(
            'lambda_imaginary',
            shape=[state_size,],
            init = _hippo.lambda_initializer('imaginary')
        )
        self._state_matrix = self._lambda_real + 1j * self._lambda_imag

        self._input_matrix = hk.get_parameter(
            'input_matrix',
            shape=[state_size, 1],
            init=_hippo.b_initializer()
        )

    def __call__(input, prev_state):
        new_state = self._state_matrix @ prev_state + self._input_matrix @ input
        return new_state

```

If using a library (such as Equinox) which does not require an initializer function but simply takes JAX ndarrays for parameterization, then you can call the HiPPO matrices directly as a property of the base class after it has been called:

```python
import equinox as eqx
from hippox.main import Hippo

class MyHippoModule(eqx.Module):
    A: jnp.ndarray
    B: jnp.ndarray

    def __init__(self, state_size, measure)
        _hippo = Hippo(state_size=state_size, measure=measure)
        _hippo_params = _hippo()
        
        self.A = _hippo_params.state_matrix
        self.B = _hippo_params.input_matrix

    def __call__(input, prev_state):
        new_state = self.A @ prev_state + self.B @ input
        return new_state

```

## Installation
hippox can be easily installed through PyPi:
```
pip install hippox
```

## References

### Repositories
1. https://github.com/HazyResearch/state-spaces - Original paper implementations in PyTorch

2. https://github.com/srush/annotated-s4 - JAX implementation of S4 models (S4, S4D, DSS)

### Papers

1. HiPPO: Recurrent Memory with Optimal Polynomial Projections:   https://arxiv.org/abs/2008.07669 - Original paper which introduced HiPPOs

2. Efficiently Modeling Long Sequences with Structured State Spaces:      https://arxiv.org/abs/2111.00396 - S4 paper, introduces normal/diagonal plus low rank decomposition

3. How to Train Your HiPPO: State Space Models with Generalized Orthogonal Basis Projections: https://arxiv.org/abs/2206.12037 - Generalizes and explains the core principals behind HiPPO

4. On the Parameterization and Initialization of Diagonal State Space Models: https://arxiv.org/abs/2206.11893 - S4D paper, details and explains the diagonal only parameterization 
