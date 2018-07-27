#!/usr/bin/env python
import numpy as np


# ============================================================
# States
# ============================================================

def single_photon(cutoff):
    r"""Single photon state |1>.

    Args:
        cutoff (int): the Fock basis truncation of the returned state vector.

    Returns:
        array: a size [cutoff] complex array state vector.
    """
    state = np.zeros([cutoff])
    state[1] = 1
    return state


def ON(N, a, cutoff):
    """ON state |0> + a|N>.

    Args:
        N (int): the occupied N-photon Fock state.
        a (complex): amplitude of the N-photon Fock state.
        cutoff (int): the Fock basis truncation of the returned state vector.

    Returns:
        array: a size [cutoff] complex array state vector.
    """
    state = np.zeros([cutoff])
    state[0] = 1
    state[N] = a
    return state/np.linalg.norm(state)


def hex_GKP(mu, delta, cutoff):
    r"""Hex GKP state.

    The Hex GKP state is defined by

    .. math::
        |mu> = \sum_{n_1,n_2=-\infty}^\infty e^{-i(q+\sqrt{3}p)/2}
            \sqrt{4\pi/\sqrt{3}d}(dn_1+\mu) e^{iq\sqrt{4\pi/\sqrt{3}d}n_2}|0>

    where d is the dimension of a code space, \mu=0,1,...,d-1, |0> is the
    vacuum state, and the states are modulated by a Gaussian envelope in the
    case of finite energy:

    ..math:: e^{-\Delta ^2 n}|\mu>

    Args:
        N (int): the occupied N-photon Fock state.
        a (complex): amplitude of the N-photon Fock state.
        cutoff (int): the Fock basis truncation of the returned state vector.

    Returns:
        array: a size [cutoff] complex array state vector.
    """
    state = np.zeros([cutoff])
    return state/np.linalg.norm(state)


def random_state(cutoff):
    r"""Random state.

    Args:
        cutoff (int): the Fock basis truncation of the returned state vector.

    Returns:
        array: a size [cutoff] complex array state vector.
    """
    state = np.random.randn(cutoff) + 1j*np.random.randn(cutoff)
    return state/np.linalg.norm(state)


def NOON(N, cutoff):
    r"""The two-mode NOON state |N0>+|0N>.

    Args:
        N (int): the occupied N-photon Fock state.
        cutoff (int): the Fock basis truncation of the returned state vector.

    Returns:
        array: a size [cutoff^2] complex array state vector.
    """
    state = np.zeros([cutoff, cutoff])
    state[0, N] = 1
    state[N, 0] = 1
    return state.flatten()/np.linalg.norm(state)

# ============================================================
# State auxillary functions
# ============================================================

def correct_global_phase(state):
    # Corrects global phase of wavefunction
    maxentry = np.argmax(np.abs(state))
    phase = state[maxentry]/np.abs(state[maxentry])
    return state/phase

