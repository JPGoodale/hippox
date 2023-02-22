import jax.numpy as jnp
from jax.numpy import newaxis
from jax.numpy.linalg import eigh
from jax.scipy.linalg import block_diag
from typing import Tuple
from .basis_measures import HippoParams


def diagonalize(
        params: HippoParams,
        N: int,
        measure: str,
        conj_sym: bool,
        dplr: bool,

) -> HippoParams:
    """
    Decomposes the hippo state matrix into diagonal form, and diagonal plus low rank if dplr is True, which essentially means just keeping the input matrix and low rank term as well as the diagonal state matrix and it's eigenvectors.

    :param params: HippoParams object containing original state and input matrices
    :param N: state size
    :param measure: basis measure family
    :param conj_sym: bool deciding whether to enforce conjugate symmetry
    :param dplr:
    :return: HippoParams object containing diagonalized state matrix and it's eigenvectors

    """
    if measure in ['linear', 'inverse']:
        return params

    else:
        low_rank_term = low_rank(N, measure)
        if measure == 'legs':
            normal_plus_low_rank = params.state_matrix + low_rank_term[:, newaxis] * low_rank_term[newaxis, :]
        else:
            normal_plus_low_rank = params.state_matrix + jnp.sum(
                low_rank_term[:, newaxis] * low_rank_term[..., newaxis],
                axis=-3
            )

        Lambda_imag, eigenvector = eigh(normal_plus_low_rank * -1j)
        diagonal = jnp.diagonal(normal_plus_low_rank)
        Lambda_real = jnp.mean(diagonal) * jnp.ones_like(diagonal)
        Lambda = Lambda_real + 1j * Lambda_imag

        if conj_sym:
            Lambda = Lambda[:N // 2]
            eigenvector = eigenvector[:, :N // 2]

        inverse_eigenvector = eigenvector.conj().T
        eigenvector_pair = (eigenvector, inverse_eigenvector)

        if dplr:
            input_matrix = eigenvector_transform(eigenvector_pair, params.input_matrix, inverse=True, dplr=True)
            low_rank_term = eigenvector_transform(eigenvector_pair, low_rank_term, inverse=True, dplr=True)
            return HippoParams(
                state_matrix=Lambda,
                eigenvector_pair=eigenvector_pair,
                input_matrix=input_matrix,
                low_rank_term=low_rank_term
            )
        else:
            return HippoParams(
                state_matrix=Lambda,
                eigenvector_pair=eigenvector_pair,
            )


def scaled_diagonal(N: int, scaling: str, conj_sym: bool):
    """
    Creates either the linear or inverse approximations of the diagonal Hippo state matrix, the inverse being derived from the scaled Legendre (HiPPO-LegS) and the linear from the translated Fourier (HiPPO-FouT).

    :param N: state_size
    :param scaling: str specifying which scaling method to apply
    :param conj_sym: bool deciding whether to enforce conjugate symmetry
    :return: HippoParams object containing scaled_diagonal approximation as state matrix
    """
    if conj_sym:
        _N = N//2
    else:
        _N = N
    n = jnp.arange(_N)

    lambda_real = -0.5 * jnp.ones(_N)
    if scaling == 'linear':
        lambda_imaginary = jnp.pi * n
    elif scaling == 'inverse':
        lambda_imaginary = 1/jnp.pi * N * (N / (2*n + 1) - 1)
    else:
        raise ValueError(
            f'Not a valid scaling: {scaling}, options are confined to linear and inverse'
        )
    _lambda = lambda_real + 1j * lambda_imaginary

    return HippoParams(state_matrix=_lambda)


def block_diagonal(
        params: HippoParams,
        block_size: int,
        n_blocks: int,
        conj_sym: bool,
        measure: str,
) -> HippoParams:
    """
    Splits the diagonal Hippo state matrix and it's eigenvectors into blocks as in: https://arxiv.org/abs/2208.04933.

    :param params: HippoParams object containing state matrix and eigenvector pair
    :param block_size: size of each block
    :param n_blocks: number of blocks to split into
    :param conj_sym: bool deciding whether to enforce conjugate symmetry
    :param measure: string specifying the measure family of input HippoParams
    :return: HippoParams object containing state matrix and eigenvector pair in block structure

    """
    if conj_sym:
        block_size = block_size // 2

    state_matrix = (params.state_matrix * jnp.ones((n_blocks, block_size))).ravel()
    if measure in ['linear', 'inverse']:
        return HippoParams(state_matrix=state_matrix)
    else:
        eigenvector = block_diag(*([params.eigenvector_pair[0]] * n_blocks))
        inverse_eigenvector = block_diag(*([params.eigenvector_pair[1]] * n_blocks))
        return HippoParams(state_matrix=state_matrix, eigenvector_pair=(eigenvector, inverse_eigenvector))


def low_rank(
        N: int,
        measure: str,
) -> jnp.ndarray:
    """
    Creates the low rank term for various measure families.

    :param N: state size
    :param measure: basis measure family
    :return: jnp.ndarray
    """
    n = jnp.arange(N)

    if measure == 'legs':
        P = jnp.sqrt(n + 0.5)

    elif measure == 'legt':
        P = jnp.sqrt(2 * n + 1)
        P0 = P.at[0::2].set(0)
        P1 = P.at[1::2].set(0)
        P = jnp.stack([P0, P1], axis=0)
        P *= 2 ** (-0.5)

    elif measure == 'lagt':
        P = jnp.sqrt(0.5) * jnp.ones((1, N))

    elif measure == 'fout':
        P = jnp.zeros(N)
        P = P.at[0::2].set(jnp.sqrt(2))
        P = P.at[0].set(1)
        P = P[newaxis, :]

    elif measure == 'fourier_decay':
        P = jnp.zeros(N)
        P = P.at[0::2].set(jnp.sqrt(2))
        P = P.at[0].set(1)
        P = P[newaxis, :]
        P = P / jnp.sqrt(2)

    elif measure == 'fourier_double':
        P = jnp.zeros(N)
        P = P.at[0::2].set(jnp.sqrt(2))
        P = P.at[0].set(1)
        P = P[newaxis, :]
        P = jnp.sqrt(2) * P

    elif measure == 'foud':
        P = jnp.zeros((1, N))

    else:
        raise ValueError(
            'invalid measure argument:', measure
        )

    return P


def eigenvector_transform(
        eigenvector_pair: Tuple[jnp.ndarray, jnp.ndarray],
        array: jnp.ndarray, inverse: bool, dplr: bool
) -> jnp.ndarray:
    """
    Applies either a transformation of the input array by the inverse eigenvector or a transformation of the eigenvector by an input array. See wrapper method of base Hippo class for an example in a parameter initialization context.

    :param eigenvector_pair: tuple containing the eigenvectors indexed from regular to inverse
    :param array: input n-dimensional array
    :param inverse: decides whether to apply inverse transformation
    :param dplr: bool specifying whether the input is in diagonal plus low rank representation
    :return: transformed array

    """
    if inverse:
        if dplr:
            return jnp.einsum('ij, ...j -> ...i', eigenvector_pair[1], array)
        else:
            return eigenvector_pair[1] @ array
    else:
        array = array[..., 0] + 1j * array[..., 1]
        return array @ eigenvector_pair[0]
