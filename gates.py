#!/usr/bin/env python
import numpy as np
from scipy.linalg import expm, norm, dft

from openfermion.ops import BosonOperator, QuadOperator
from openfermion.transforms import get_sparse_operator
from strawberryfields.utils import random_interferometer


# ============================================================
# Gates
# ============================================================

def cubic_phase(gamma, cutoff, offset=20):
    r"""Cubic phase gate e^{-i\gamma x^3}.

    Args:
        gamma (float): gate parameter.
        cutoff (int): the Fock basis truncation of the returned unitary.
        offset (int): the Fock basis truncation used to calculate the
            matrix exponential. Setting this larger can increase numerical
            accuracy in the returned truncated unitary.

    Returns:
        array: a [cutoff, cutoff] complex array representing
            the action of the cubic phase gate in the truncated
            Fock basis.
    """
    x3 = QuadOperator('q0 q0 q0')
    U = expm(get_sparse_operator(-1j*gamma*x3, trunc=cutoff+offset, hbar=2).toarray())
    return U[:cutoff, :cutoff]


def cross_kerr(kappa, cutoff):
    r"""Cross-Kerr interaction e^{-i\kappa n_1 n_2}.

    Args:
        kappa (float): Kerr interaction strength.
        cutoff (int): the Fock basis truncation of the returned unitary.

    Returns:
        array: a [cutoff, cutoff] complex array representing
            the action of the cross-Kerr interaction in the truncated
            Fock basis.
    """
    n0 = BosonOperator('0^ 0')
    n1 = BosonOperator('1^ 1')
    U = expm(get_sparse_operator(1j*kappa*n0*n1, trunc=cutoff).toarray())
    return U


def random_unitary(size, cutoff):
    r"""Random unitary in the Fock basis.

    Args:
        size (int): size of the random unitary. The unitary will
            act on Fock basis elements |0> to |size-1>.
        cutoff (int): the Fock basis truncation of the returned unitary.
            Must be larger than size.

    Returns:
        array: a [cutoff, cutoff] complex array representing
            the action of the random unitary in the truncated
            Fock basis.
    """
    U = np.identity(cutoff, dtype=np.complex128)
    U[:size, :size] = random_interferometer(size)
    return U


def DFT(size, cutoff):
    r"""Discrete Fourier transform in the Fock basis.

    Args:
        size (int): size of the DFT. The unitary will
            act on Fock basis elements |0> to |size-1>.
        cutoff (int): the Fock basis truncation of the returned unitary.
            Must be larger than size.

    Returns:
        array: a [cutoff, cutoff] complex array representing
            the action of the DFT in the truncated
            Fock basis.
    """
    U = np.identity(cutoff, dtype=np.complex128)
    U[:size, :size] = dft(size)/np.sqrt(size)
    return U


# ============================================================
# Gate auxillary functions
# ============================================================


def min_cutoff(U, p, gate_cutoff, cutoff):
    """Calculate the minimum sim cutoff such that U is still unitary.

    This is done by looping through a smaller and smaller cutoff
    dimension, and stopping when the max norms of the columns is
    significantly less than one. This is a measure of the
    non-unitarity of a matrix.

    Args:
        U (array): the unitary to analyze, of size [cutoff^2, cutoff^2]
        p (float): the precision value p=1-|U_j| at which the column
            U_j of the unitary is no longer considered unitary.
        gate_cutoff (int): the gate cutoff truncation to analyze.
        cutoff (int): the simulation Fock basis truncation.

    Returns:
        array: the minimum simulation cutoff such that the columns
            of the target gate remain unitary.
    """
    min_cutoff = cutoff + 1

    # get the number of modes the gate acts on
    m = get_modes(U, cutoff)

    for n in range(cutoff, gate_cutoff, -1):
        norms = 1 - norm(U[:n**m, :gate_cutoff**m], axis=0)
        eps = max(norms)
        if eps > p:
            min_cutoff = n+1
            break
    else:
        min_cutoff = gate_cutoff + 1

    return min_cutoff


def get_modes(U, cutoff):
    """Get the number of modes a gate unitary acts on.

    Args:
        U (array): the unitary to analyze, of size [cutoff^2, cutoff^2]
        cutoff (int): the simulation Fock basis truncation.

    Returns:
        int: the number of modes
    """
    return int(np.log(U.shape[0])/np.log(cutoff))


def unitary_state_fidelity(V, U):
    r"""The state fidelity of the target unitary and the learnt unitary
    applied to the equal superposition state.

    This function returns the following state fidelity:

    .. math:: \langle \psi_d \mid V^\dagger U \mid \psi_d\rangle

    where :math:`|\psi_d\rangle=\frac{1}{\sqrt{d}}\sum_{n=0}^{d-1} |n\rangle`
    is the equal superposition state, :math:`V` the target unitary, and
    :math:`U` the learnt unitary.

    The target unitary should have 2m dimensions, where m is the number of modes,
    with each dimension of size cutoff.

    The learnt unitary should have m+1 dimensions, where m is the number of modes,
    with the first dimension of size d^m, where d is the gate cutoff, and
    all subsequent dimensions are of size cutoff.

    Args:
        V (array): the target unitary.
        U (array): the learnt unitary.

    Returns:
        tuple (array, array, float): Returns a tuple containing V|psi_d>,
            U|psi_d>, and the fidelity.
    """
    # simulation cutoff
    c = V.shape[0]
    # number of modes
    m = get_modes(V, c)
    # gate cutoff
    d = np.int(U.shape[0]**(1/m))

    if m == 1:
        # single mode unitary
        state1 = np.sum(V[:d, :], axis=0)/np.sqrt(d)
        state2 = np.sum(U[:d, :], axis=0)/np.sqrt(d)
    elif m == 2:
        # two mode unitary
        # reshape the target unitary to be shape [c^2, d^2]
        Ut = np.einsum('ijkl->ikjl', V[:, :d, :, :d]).reshape(c**2, d**2)
        # reshape the learnt unitary to be shape [d^2, c^2]
        Ul = np.einsum('ijkl->ijkl', U.reshape(d, d, c, c)).reshape(d**2, c**2)

        eq_sup_state = np.full([d**2], 1/d)

        state1 = Ut @ eq_sup_state
        state2 = Ul.T @ eq_sup_state

    # calculate the fidelity
    fidelity = np.abs(np.vdot(state1, state2))**2
    return state1, state2, fidelity
