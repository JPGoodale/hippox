# Hippox: High-order Polynomial Projection Operators for JAX

!(data:image/png;base64,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)

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


