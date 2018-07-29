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

import tensorflow as tf

import strawberryfields as sf
from strawberryfields.ops import *


def one_mode_variational_quantum_circuit(cutoff, input_state=None, batch_size=None,
        depth=25, active_sd=0.0001, passive_sd=0.1, **kwargs):
    """Construct the one mode circuit ansatz, and return the learnt unitary and gate parameters.

    Args:
        cutoff (int): the simulation Fock basis truncation.
        input_state (int): the multi-mode state vector to initialise the circuit in.
            If used in batch mode, the first dimension of input_state must match the batch
            size of the circuit.
        batch_size (int): the number of parallel batches used to evaluate the circuit.
        depth (int): the number of layers to use to construct the circuit.
        active_sd (float): the normal standard deviation used to initialise the active gate parameters.
        passive_sd (float): the normal standard deviation used to initialise the passive gate parameters.

    Returns:
        tuple (state, parameters): a tuple contiaining the state vector and a list of the
            gate parameters, as tensorflow tensors.
    """
    # Random initialization of gate parameters.
    # Gate parameters begin close to zero, with values drawn from Normal(0, sdev)
    with tf.name_scope('variables'):
        # displacement parameters
        d_r = tf.Variable(tf.random_normal(shape=[depth], stddev=active_sd))
        d_phi = tf.Variable(tf.random_normal(shape=[depth], stddev=passive_sd))
        # rotation parameter
        r1 = tf.Variable(tf.random_normal(shape=[depth], stddev=passive_sd))
        # squeezing parameters
        sq_r = tf.Variable(tf.random_normal(shape=[depth], stddev=active_sd))
        sq_phi = tf.Variable(tf.random_normal(shape=[depth], stddev=passive_sd))
        # rotation parameter
        r2 = tf.Variable(tf.random_normal(shape=[depth], stddev=passive_sd))
        # Kerr gate parameter
        kappa = tf.Variable(tf.random_normal(shape=[depth], stddev=active_sd))

    # Array of all parameters
    parameters = [d_r, d_phi, r1, sq_r, sq_phi, r2, kappa]

    # Gate layer: D-R-S-R-K
    def layer(i, q, m):
        with tf.name_scope('layer_{}'.format(i)):
            Dgate(d_phi[i]) | q[m]
            Rgate(r1[i]) | q[m]
            Sgate(sq_r[i], sq_phi[i]) | q[m]
            Rgate(r2[i]) | q[m]
            Kgate(kappa[i]) | q[m]

        return q

    # Start SF engine
    eng, q = sf.Engine(1)

    # construct the circuit
    with eng:
        if input_state is not None:
            Ket(input_state) | q
        for k in range(depth):
            q = layer(k, q, 0)

    if batch_size is not None:
        state = eng.run('tf', cutoff_dim=cutoff, eval=False, batch_size=batch_size)
    else:
        state = eng.run('tf', cutoff_dim=cutoff, eval=False)

    # Extract the state vector
    ket = state.ket()

    return ket, parameters


def two_mode_variational_quantum_circuit(cutoff, input_state=None, batch_size=None,
        depth=25, active_sd=0.0001, passive_sd=0.1, **kwargs):
    """Construct the two mode circuit ansatz, and return the learnt unitary and gate parameters.

    Args:
        cutoff (int): the simulation Fock basis truncation.
        input_state (int): the multi-mode state vector to initialise the circuit in.
            If used in batch mode, the first dimension of input_state must match the batch
            size of the circuit.
        batch_size (int): the number of parallel batches used to evaluate the circuit.
        depth (int): the number of layers to use to construct the circuit.
        active_sd (float): the normal standard deviation used to initialise the active gate parameters.
        passive_sd (float): the normal standard deviation used to initialise the passive gate parameters.

    Returns:
        tuple (state, parameters): a tuple contiaining the state vector and a list of the
            gate parameters, as tensorflow tensors.
    """
    # Random initialization of gate parameters.
    # Gate parameters begin close to zero, with values drawn from Normal(0, sdev)
    with tf.name_scope('variables'):
        # interferometer1 parameters
        theta1 = tf.Variable(tf.random_normal(shape=[depth], stddev=passive_sd))
        phi1 = tf.Variable(tf.random_normal(shape=[depth], stddev=passive_sd))
        r1 = tf.Variable(tf.random_normal(shape=[depth], stddev=passive_sd))
        # squeeze gate
        sq_r = tf.Variable(tf.random_normal(shape=[2, depth], stddev=active_sd))
        sq_phi = tf.Variable(tf.random_normal(shape=[2, depth], stddev=passive_sd))
        # interferometer2 parameters
        theta2 = tf.Variable(tf.random_normal(shape=[depth], stddev=passive_sd))
        phi2 = tf.Variable(tf.random_normal(shape=[depth], stddev=passive_sd))
        r2 = tf.Variable(tf.random_normal(shape=[depth], stddev=passive_sd))
        # displacement gate
        d_r = tf.Variable(tf.random_normal(shape=[2, depth], stddev=active_sd))
        d_phi = tf.Variable(tf.random_normal(shape=[2, depth], stddev=passive_sd))
        # kerr gate
        kappa = tf.Variable(tf.random_normal(shape=[2, depth], stddev=active_sd))

    # Array of all parameters
    parameters = [theta1, phi1, r1, sq_r, sq_phi, theta2, phi2, r2, d_r, d_phi, kappa]

    # Gate layer: U-(SxS)-U-(DxD)-(KxK)
    def layer(i, q):
        with tf.name_scope('layer_{}'.format(i)):
            BSgate(theta1[k], phi1[k]) | (q[0], q[1])
            Rgate(r1[i]) | q[0]

            for m in range(2):
                Sgate(sq_r[m, i], sq_phi[m, i]) | q[m]

            BSgate(theta2[k], phi2[k]) | (q[0], q[1])
            Rgate(r2[i]) | q[0]

            for m in range(2):
                Dgate(d_r[m, i],  d_phi[m, i]) | q[m]
                Kgate(kappa[m, i]) | q[m]
        return q

    # Start SF engine
    eng, q = sf.Engine(2)

    # construct the circuit
    with eng:
        if input_state is not None:
            Ket(input_state) | q
        for k in range(depth):
            q = layer(k, q)

    if batch_size is not None:
        state = eng.run('tf', cutoff_dim=cutoff, eval=False, batch_size=batch_size)
    else:
        state = eng.run('tf', cutoff_dim=cutoff, eval=False)

    # Extract the state vector
    ket = state.ket()

    return ket, parameters


def variational_quantum_circuit(*, modes, cutoff, input_state=None, batch_size=None,
        depth=25, active_sd=0.0001, passive_sd=0.1, **kwargs):
    """Construct the variational quantum circuit ansatz, and return the output state and gate parameters.

    This is a wrapper function for the one_mode_variational_quantum_circuit and
    two_mode_variational_quantum_circuit functions.

    Args:
        modes (int): the number of modes (1 or 2) for the variational quantum circuit.
        cutoff (int): the simulation Fock basis truncation.
        input_state (int): the multi-mode state vector to initialise the circuit in.
            If used in batch mode, the first dimension of input_state must match the batch
            size of the circuit.
        batch_size (int): the number of parallel batches used to evaluate the circuit.
        depth (int): the number of layers to use to construct the circuit.
        active_sd (float): the normal standard deviation used to initialise the active gate parameters.
        passive_sd (float): the normal standard deviation used to initialise the passive gate parameters.

    Returns:
        tuple (state, parameters): a tuple contiaining the state vector and a list of the
            gate parameters, as tensorflow tensors.
    """
    if modes == 2:
        return two_mode_variational_quantum_circuit(cutoff, input_state, batch_size, depth, active_sd, passive_sd, **kwargs)

    return one_mode_variational_quantum_circuit(cutoff, input_state, batch_size, depth, active_sd, passive_sd, **kwargs)
