import chex
import jax.numpy as jnp
from typing import Optional, Callable
from .basis_measures import hippo_legs, hippo_legt, hippo_lagt, hippo_fout, HippoParams
from .diagonalization import scaled_diagonal, block_diagonal, diagonalize, eigenvector_transform


@chex.dataclass
class Hippo:
    """
    Base class for initializing HiPPOs (High-order Polynomial Projection Operators) derived from various basis measures. Uses the Chex dataclass rather than the standard library one for compatibility with JAX pytrees and therefore does not take positional arguments when being called.
    :param state_size: int specifying the size of the hidden state
    :param basis_measure: str specifying the basis measure family to generate HiPPO from
    :param conj_sym: bool deciding whether conjugate symmetry is enforced
    :param dplr: bool deciding whether to keep low rank term and original B matrix
    :param diagonalize: bool deciding whether to diagonalize HiPPO matrix
    :param block_diagonal: bool deciding whether to split diagonal HiPPO into blocks as in: https://arxiv.org/abs/2208.04933
    :param n_blocks: int specifying the number of blocks to split state matrix into if block_diagonal is True
    """

    state_size: int
    basis_measure: str
    conj_sym: bool = False
    dplr: bool = False
    diagonalize: bool = True
    block_diagonal: bool = False
    n_blocks: Optional[int] = None

    def __call__(self) -> HippoParams:
        if self.block_diagonal:
            assert self.n_blocks is not None
            self.state_size = self.state_size // self.n_blocks

        if self.basis_measure == 'legs':
            params = hippo_legs(self.state_size)

        elif self.basis_measure == 'legt':
            params = hippo_legt(
                self.state_size, lmu=False
            )

        elif self.basis_measure == 'lmu':
            params = hippo_legt(
                self.state_size, lmu=True,
            )

        elif self.basis_measure == 'lagt':
            params = hippo_lagt(
                self.state_size, generalized=False
            )

        elif self.basis_measure == 'glagt':
            params = hippo_lagt(
                self.state_size, generalized=True
            )

        elif self.basis_measure == 'fout':
            params = hippo_fout(
                self.state_size, diagonal=False,
                decay=False, double=False
            )

        elif self.basis_measure == 'foud':
            params = hippo_fout(
                self.state_size, diagonal=True,
                decay=False, double=False
            )

        elif self.basis_measure == 'fourier_decay':
            params = hippo_fout(
                self.state_size, diagonal=False,
                decay=True, double=False
            )

        elif self.basis_measure == 'fourier_double':
            params = hippo_fout(
                self.state_size, diagonal=False,
                decay=False, double=True
            )

        elif self.basis_measure == 'linear':
            params = scaled_diagonal(
                self.state_size, 'linear',
                self.conj_sym
            )

        elif self.basis_measure == 'inverse':
            params = scaled_diagonal(
                self.state_size, 'inverse',
                self.conj_sym
            )
        else:
            raise ValueError(
                'Invalid basis_measure argument:', self.basis_measure
            )

        if self.diagonalize:
            params = diagonalize(
                params, self.state_size, self.basis_measure,
                self.conj_sym, self.dplr
            )
            if self.block_diagonal:
                params = block_diagonal(params, self.state_size, self.n_blocks, self.conj_sym, self.basis_measure)

        self._params = params
        return self._params

    def b_initializer(self):
        """
        Input matrix parameter initializer for use in Modules across various jax neural net libraries i.e. Haiku, Flax etc...

        :return initializer function of signature: f(key, shape) if dplr is True, otherwise jnp.ones
        """
        if self.dplr:
            return lambda key, shape: self._params.input_matrix
        else:
            return jnp.ones

    def low_rank_initializer(self):
        """
        Low rank term parameter initializer for use in Modules across various jax neural net libraries i.e. Haiku, Flax etc...

        :return initializer function of signature: f(key, shape)

        """

        if self.dplr:
            return lambda key, shape: self._params.low_rank_term
        else:
            raise ValueError('Keep_input_matrix_and_low_term must be True')

    def lambda_initializer(self, return_array: str) -> Callable:
        """
        Splits state matrix back into real and imaginary parts for parameter initialization in Modules across various neural net libraries i.e. Haiku, Flax etc...

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
            input,
            inverse: bool = True,
            concatenate: bool = False
    ) -> Callable:
        """
        Wrapper method for global function of the same name which applies either a transformation of the input array by the inverse eigenvector or a transformation of the eigenvector by an input array. Unlike the function, this method assumes that the input array you are providing is the return value of a jax initialization function requiring key and shape arguments, such as jax.nn.initializers.normal(). It also provides an option for splitting the transformed array into real and imaginary parts and concatenating them back together for parameter initialization such as in the original S5 implementation: https://github.com/lindermanlab/S5

        :param input: input array
        :param inverse: bool deciding whether to apply inverse transformation
        :param concatenate: bool deciding whether to split and concatenate transformed array
        :return: transformed array

        """
        transformed = eigenvector_transform(self._params.eigenvector_pair, input, inverse, self.dplr)
        if concatenate:
            transformed_real = transformed.real
            transformed_imag = transformed.imag
            transformed = jnp.concatenate((transformed_real[..., None], transformed_imag[..., None]), axis=-1)
        return lambda key, shape: transformed
