# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from scipy.special import factorial as fac


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


def hex_GKP(mu, d, delta, cutoff, nmax=7):
    r"""Hexagonal GKP code state.

    The Hex GKP state is defined by

    .. math::
        |mu> = \sum_{n_1,n_2=-\infty}^\infty e^{-i(q+\sqrt{3}p)/2}
            \sqrt{4\pi/\sqrt{3}d}(dn_1+\mu) e^{iq\sqrt{4\pi/\sqrt{3}d}n_2}|0>

    where d is the dimension of a code space, \mu=0,1,...,d-1, |0> is the
    vacuum state, and the states are modulated by a Gaussian envelope in the
    case of finite energy:

    ..math:: e^{-\Delta ^2 n}|\mu>

    Args:
        d (int): the dimension of the code space.
        mu (int): mu=0,1,...,d-1.
        delta (float): width of the modulating Gaussian envelope.
        cutoff (int): the Fock basis truncation of the returned state vector.
        nmax (int): the Hex GKP state |mu> is calculated by performing the
            sum using n1,n1=-nmax,...,nmax.

    Returns:
        array: a size [cutoff] complex array state vector.
    """
    n1 = np.arange(-nmax, nmax+1)[:, None]
    n2 = np.arange(-nmax, nmax+1)[None, :]

    n1sq = n1**2
    n2sq = n2**2

    sqrt3 = np.sqrt(3)

    arg1 = -1j*np.pi*n2*(d*n1+mu)/d
    arg2 = -np.pi*(d**2*n1sq+n2sq-d*n1*(n2-2*mu)-n2*mu+mu**2)/(sqrt3*d)
    arg2 *= 1-np.exp(-2*delta**2)

    amplitude = (np.exp(arg1)*np.exp(arg2)).flatten()[:, None]

    alpha = np.sqrt(np.pi/(2*sqrt3*d)) * (sqrt3*(d*n1+mu) - 1j*(d*n1-2*n2+mu))
    alpha *= np.exp(-delta**2)

    alpha = alpha.flatten()[:, None]
    n = np.arange(cutoff)[None, :]
    coherent = np.exp(-0.5*np.abs(alpha)**2)*alpha**n/np.sqrt(fac(n))

    hex_state = np.sum(amplitude*coherent, axis=0)
    return hex_state/np.linalg.norm(hex_state)


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
    """Corrects the global phase of wavefunction."""
    maxentry = np.argmax(np.abs(state))
    phase = state[maxentry]/np.abs(state[maxentry])
    return state/phase
