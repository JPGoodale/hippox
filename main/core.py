import dataclasses
import jax.numpy as jnp
from typing import Optional, Callable, NamedTuple, Tuple
from .basis_measures import hippo_legs, hippo_legt, hippo_lagt, hippo_fout
from .diagonalization import scaled_diagonal, block_diagonal, diagonalize, eigenvector_transform


class HippoParams(NamedTuple):
    """
    Base class for storing matrices directly derived from HiPPO.
    """

    state_matrix: jnp.ndarray
    eigenvector_pair: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None
    input_matrix: Optional[jnp.ndarray] = None
    low_rank_term: Optional[jnp.ndarray] = None


@dataclasses.dataclass
class Hippo:
    """
    Base class for initialization of HiPPOs (High-order Polynomial Projection Operators) derived from various basis measures.
    """

    state_size: int
    measure_family: str
    conj_sym: bool = False
    dplr: bool = False
    diagonalize: bool = True
    block_diagonal: bool = False
    n_blocks: Optional[int] = None

    def __call__(self) -> HippoParams:
        if self.block_diagonal:
            assert self.n_blocks is not None
            self.state_size = self.state_size // self.n_blocks

        if self.measure_family == 'legs':
            params = hippo_legs(self.state_size)

        elif self.measure_family == 'legt':
            params = hippo_legt(
                self.state_size, lmu=False
            )

        elif self.measure_family == 'lmu':
            params = hippo_legt(
                self.state_size, lmu=True,
            )

        elif self.measure_family == 'lagt':
            params = hippo_lagt(
                self.state_size, generalized=False
            )

        elif self.measure_family == 'glagt':
            params = hippo_lagt(
                self.state_size, generalized=True
            )

        elif self.measure_family == 'fout':
            params = hippo_fout(
                self.state_size, diagonal=False,
                decay=False, double=False
            )

        elif self.measure_family == 'foud':
            params = hippo_fout(
                self.state_size, diagonal=True,
                decay=False, double=False
            )

        elif self.measure_family == 'fourier_decay':
            params = hippo_fout(
                self.state_size, diagonal=False,
                decay=True, double=False
            )

        elif self.measure_family == 'fourier_double':
            params = hippo_fout(
                self.state_size, diagonal=False,
                decay=False, double=True
            )

        elif self.measure_family == 'linear':
            params = scaled_diagonal(
                self.state_size, 'linear',
                self.conj_sym
            )

        elif self.measure_family == 'inverse':
            params = scaled_diagonal(
                self.state_size, 'inverse',
                self.conj_sym
            )
        else:
            raise ValueError(
                'Invalid measure_family argument:', self.measure_family
            )

        if self.diagonalize:
            params = diagonalize(
                params, self.state_size, self.measure_family,
                self.conj_sym, self.dplr
            )
            if self.block_diagonal:
                params = block_diagonal(params, self.state_size, self.n_blocks, self.conj_sym)

        self._params = params
        return self._params

    def b_initializer(self):
        """
        Input matrix parameter initializer for use in Modules across various jax neural net libraries i.e. Haiku, Flax etc... Here's an example using Haiku:

        class MyHippoModule(hk.Module):
            def __init__(self, state_size, measure)
                _hippo = Hippo(state_size=state_size, measure=measure)
                _hippo()

                self._lambda_real = hk.get_parameter(
                    'lambda_imaginary',
                    shape=[state_size,]
                    init=_hippo.lambda_initializer('real')
                )
                self._lambda_imag = hk.get_parameter(
                    'lambda_imaginary',
                    shape=[state_size,]
                    init=_hippo.lambda_initializer('imaginary')
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


        If using a library such as Equinox which does not require an initializer function but simply takes jax n-dimensional arrays for setting parameters, then you can call them directly as a property of the class. Here's an example:

        class MyHippoModule(equinox.Module):
            A: jnp.ndarray
            B: jnp.ndarray
            def __init__(self, state_size, measure)
                _hippo = Hippo(state_size=state_size, measure=measure)
                _hippo_params = _hippo()
                self.A = _hippo_params.state_matrix
                self.B = _hippo_params.input_matrix
            def __call__(input, state):
                new_state = self.A @ state + self.B @ input
                return new state

        :return initializer function of signature: f(key, shape) if dplr is True, otherwise jnp.ones
        """
        if self.dplr:
            return lambda key, shape: self._params.input_matrix
        else:
            return jnp.ones

    def low_rank_initializer(self):
        """
        Low rank term parameter initializer for use in Modules across various jax neural net libraries i.e. Haiku, Flax etc... Here's an example using Haiku:

        class MyHippoModule(hk.Module):
            def __init__(self, state_size, measure)
                _hippo = Hippo(state_size=state_size, measure=measure)
                _hippo()

                self._lambda_real = hk.get_parameter(
                    'lambda_imaginary',
                    shape=[state_size,]
                    init=_hippo.lambda_initializer('real')
                )
                self._lambda_imag = hk.get_parameter(
                    'lambda_imaginary',
                    shape=[state_size,]
                    init=_hippo.lambda_initializer('imaginary')
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


        If using a library such as Equinox which does not require
        an initializer function but simply takes jax n-dimensional
        arrays for setting parameters, then you can call them directly
        as a property of the class. Here's an example:

        class MyHippoModule(equinox.Module):
            A: jnp.ndarray
            B: jnp.ndarray
            def __init__(self, state_size, measure)
                _hippo = Hippo(state_size=state_size, measure=measure)
                _hippo_params = _hippo()
                self.A = _hippo_params.state_matrix
                self.B = _hippo_params.input_matrix
            def __call__(input, state):
                new_state = self.A @ state + self.B @ input
                return new state

        :return initializer function of signature: f(key, shape)

        """

        if self.dplr:
            return lambda key, shape: self._params.low_rank_term
        else:
            raise ValueError('Keep_input_matrix_and_low_term must be True')

    def lambda_initializer(self, return_array: str) -> Callable:
        """
        Splits state matrix back into real and imaginary parts for parameter initialization in Modules across various neural net libraries i.e. Haiku, Flax etc... Here's an example using Haiku:

        class MyHippoModule(hk.Module):
            def __init__(self, state_size, measure)
                _hippo = Hippo(state_size=state_size, measure=measure)
                _hippo()

                self._lambda_real = hk.get_parameter(
                    'lambda_imaginary',
                    shape=[state_size,]
                    init=_hippo.lambda_initializer('real')
                )
                self._lambda_imag = hk.get_parameter(
                    'lambda_imaginary',
                    shape=[state_size,]
                    init=_hippo.lambda_initializer('imaginary')
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

        If using a library such as Equinox which does not require an initializer function but simply takes jax n-dimensional arrays for setting parameters, then you can call them directly as a property of the class. Here's an example:

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
                return new state

        :param return_array: str summoning either 'real', 'imaginary' or 'both'
        :return: initializer function of signature: f(key, shape)

        """
        if return_array == 'real':
            return lambda key, shape: self._params.state_matrix.real
        elif return_array == 'imaginary':
            return lambda key, shape: self._params.state_matrix.imag
        elif return_array == 'full':
            return lambda key, shape: self._params.state_matrix
        else:
            raise ValueError('Invalid return array:', return_array)

    def eigenvector_transform(
            self,
            input_fn,
            key, shape,
            inverse: bool = True,
            concatenate: bool = False
    ) -> jnp.ndarray:
        """
        Wrapper method for global function of the same name which applies either a transformation of the input array by the inverse eigenvector or  a transformation of the eigenvector by an input array. Unlike the function, this method assumes that the input array you are providing is the return value of a jax initialization function requiring key and shape arguments, such as jax.nn.initializers.normal(). It also provides an option for splitting the transformed array into real and imaginary parts and concatenating them back together for parameter initialization such as in the original S5 implementation: https://github.com/lindermanlab/S5

        :param input_fn: initialization function of signature: f(key, shape)
        :param key: jax PRNGKey for input function
        :param shape: list or tuple providing the shape of for the input function
        :param inverse: bool deciding whether to apply inverse transformation
        :param concatenate: bool deciding whether to split and concatenate transformed array
        :return: transformed array

        """
        array = input_fn(key, shape)
        transformed = eigenvector_transform(self._params.eigenvector_pair, array, inverse, self.measure_family)
        if concatenate:
            transformed_real = transformed.real
            transformed_imag = transformed.imag
            transformed = jnp.concatenate((transformed_real[..., None], transformed_imag[..., None]), axis=-1)
        return transformed
