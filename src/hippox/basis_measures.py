import jax.numpy as jnp
import jax.scipy.special as jss
import scipy.special as ss
from jax.numpy import newaxis
from typing import Tuple, NamedTuple, Optional


class HippoParams(NamedTuple):
    """
    Base class for storing matrices directly derived from HiPPO.
    """

    state_matrix: jnp.ndarray
    eigenvector_pair: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None
    input_matrix: Optional[jnp.ndarray] = None
    low_rank_term: Optional[jnp.ndarray] = None


def hippo_legs(N: int) -> HippoParams:
    """
    Creates the scaled Legendre basis measure HiPPO

    :param N: state size
    :return: HippoParams object containing state and input matrices

    """
    n = jnp.arange(N)
    polynomial = jnp.sqrt(2 * n + 1)

    A = polynomial[:, newaxis] * polynomial[newaxis, :]
    A = jnp.tril(A) - jnp.diag(n)

    A = -A
    B = polynomial

    return HippoParams(state_matrix=A, input_matrix=B)


def hippo_legt(
        N: int,
        lmu: bool,
) -> HippoParams:
    """
    Creates the translated Legendre basis measure HiPPO

    :param N: state_size
    :param lmu: bool deciding whether to use Legendre memory unit initialization used in: https://proceedings.neurips.cc/paper/2019/file/952285b9b7e7a1be5aa7849f32ffff05-Paper.pdf
    :return: HippoParams object containing state and input matrices

    """
    n = jnp.arange(N)
    j, i = jnp.meshgrid(n, n)
    polynomial = (2 * n + 1)

    if lmu:
        A = jnp.where(i < j,
                      x=-1,
                      y=(-1.) ** (i - j + 1),
                      ) * (polynomial[:, newaxis])

        B = (-1.) ** n[:, newaxis] * (polynomial[:, newaxis])

    else:
        polynomial = jnp.sqrt(polynomial)

        A = jnp.where(i < j,
                      x=(-1.) ** (i - j),
                      y=1,
                      )
        A = polynomial[:, newaxis] * A * polynomial[newaxis, :]

        A = -A
        B = polynomial[: newaxis]

        A *= 0.5
        B *= 0.5

    return HippoParams(state_matrix=A, input_matrix=B)


def hippo_lagt(
        N: int,
        generalized: bool,
        beta: float = 1.0,
) -> HippoParams:
    """
    Creates the translated Laguerre basis measure HiPPO

    :param N: state size
    :param generalized: bool deciding whether to use the generalized initialization (GLagT)
    :param beta: constant
    :return: HippoParams object containing state and input matrices

    """
    if generalized:
        n = jnp.arange(N)
        alpha = 0.0
        beta = 0.01

        A = -jnp.eye(N) * (1 + beta) \
            / 2 - (jnp.tril(jnp.ones((N, N)), -1))

        B = ss.binom(alpha + n, n)[:, newaxis]

        polynomial = jnp.exp(0.5 * (jss.gammaln(n + alpha + 1) - jss.gammaln(n + 1)))

        A = (1. / polynomial[:, newaxis]) * A * polynomial[newaxis, :]
        B = (1. / polynomial[:, newaxis]) * B * (
                jnp.exp(-0.5 * jss.gammaln(1 - alpha)) * beta ** ((1 - alpha) / 2)
        )

    else:
        A = jnp.eye(N) \
            / 2 - (jnp.tril(jnp.ones((N, N))))

        B = beta * jnp.ones(N)

    return HippoParams(state_matrix=A, input_matrix=B)


def hippo_fout(
        N: int,
        diagonal: bool,
        decay: bool,
        double: bool,
) -> HippoParams:
    """
    Creates the translated Fourier basis measure HiPPO

    :param N: state size
    :param diagonal: bool deciding whether to use the diagonal initialization (FouD)
    :param decay: bool deciding whether to use the decay initialization (Fourier_decay)
    :param double: bool deciding whether to use the double initialization (Fourier_double)
    :return: HippoParams object containing state and input matrices

    """
    if N % 2 != 0:
        raise ValueError('N must be an even value for this projection')
    frequencies = jnp.arange(N // 2)
    if double:
        frequencies = frequencies * 2

    if diagonal:
        stack = jnp.stack(
            [frequencies, jnp.zeros(N // 2)],
            axis=-1
        ).reshape(-1)[:-1]

        A = 2 * jnp.pi * (-jnp.diag(stack, 1) + jnp.diag(stack, -1))

    else:
        stack = jnp.stack(
            [jnp.zeros(N // 2), frequencies],
            axis=-1
        ).reshape(-1)[1:]

        A = jnp.pi * (-jnp.diag(stack, 1) + jnp.diag(stack, -1))

    B = jnp.zeros(N)
    B = B.at[0::2].set(jnp.sqrt(2))
    B = B.at[0].set(1)

    if diagonal:
        A = A - 0.5 * jnp.eye(N)
    elif decay:
        A = A - 0.5 * B[:, newaxis] * B[newaxis, :]
        B = 0.5 * B
    elif double:
        A = A - B[:, newaxis] * B[newaxis, :] * 2
        B = B * 2
    else:
        A = A - B[:, newaxis] * B[newaxis, :]

    return HippoParams(state_matrix=A, input_matrix=B)
