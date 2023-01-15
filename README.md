# Hippox

High-order Polynomial Projection Operators for JAX

## What is Hippox?

Hippox provides a simple dataclass for initializing High-order Polynomial Projection Operators (HiPPOs) as parameters in JAX neural network libraries such as Flax and Haiku.

## Example

Here is an example of initializing HiPPO parameters inside a Haiku module:

```python
class MyHippoModule(hk.Module):
    def __init__(self, state_size, measure)
        _hippo = Hippo(state_size=state_size, measure=measure)
        _hippo()

        self._lambda_real = hk.get_parameter(
            'lambda_real',
            shape=[state_size,]
            init = _hippo.lambda_initializer('real')
        )
        self._lambda_imag = hk.get_parameter(
            'lambda_imaginary',
            shape=[state_size,]
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
class MyHippoModule(equinox.Module):
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


