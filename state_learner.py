#!/usr/bin/env python3
import os
import time
import argparse

import numpy as np
from numpy.polynomial.hermite import hermval

import tensorflow as tf

import strawberryfields as sf
from strawberryfields.ops import *

from states import single_photon, ON, hex_GKP, random_state, NOON, correct_global_phase
from plots import wigner_3D_plot, wavefunction_plot, two_mode_wavefunction_plot, plot_cost


# ===============================================================================
# Hyperparameters
# ===============================================================================

# Set the default hyperparameters
HP = {
    #name of the simulation
    'name': 'state_synthesis',
    # default output directory
    'out_dir': 'sim_results',
    # State parameters
    'params': [5],
    # Cutoff dimension
    'cutoff': 10,
    # Number of layers
    'depth': 20,
    # Number of steps in optimization routine performing gradient descent
    'reps': 1000,
    # Penalty coefficient to ensure the state is normalized
    'penalty_strength': 0,
    # Standard deviation of initial parameters
    'sdev': 0.1
}


# ===============================================================================
# Auxillary functions
# ===============================================================================

def parse_arguments(defaults):
    """Parse optional command line arguments.

    Args:
        defaults (dict): a dictionary containing the default hyperparameters.

    Returns:
        dict: a dictionary containing the simulation hyperparameters, updated
            with values passed as command line arguments.
    """

    parser = argparse.ArgumentParser(description='Quantum state preparation learning.')
    # output arguments
    parser.add_argument('-n', '--name',
        type=str, default=defaults["name"], help='Simulation name.')
    parser.add_argument('-o', '--outdir',
        type=str, default=defaults["out_dir"], help='Output directory')
    parser.add_argument('-s', '--dump-reps',
        type=int, default=100, help='Steps at which to save output')
    parser.add_argument('-D', '--debug',
        action='store_true', help="Debug mode")
    # simulation settings
    parser.add_argument('-r', '--reps',
        type=int, default=defaults["reps"], help='Optimization steps')
    parser.add_argument('-p', '--param',
        type=float, nargs='+', default=defaults["params"], help='State parameters')
    parser.add_argument('-c', '--cutoff',
        type=int, default=defaults["cutoff"], help='Fock basis truncation')
    parser.add_argument('-d', '--depth',
        type=int, default=defaults["depth"], help='Number of layers')
    parser.add_argument('-P', '--penalty-strength',
        type=int, default=defaults["penalty_strength"], help='Regularisation penalty strength')
    args = parser.parse_args()

    hyperparams = {}
    hyperparams.update(defaults)
    hyperparams.update(vars(args))

    if args.debug:
        hyperparams['depth'] = 1
        hyperparams['reps'] = 5
        hyperparams['name'] += "_debug"

    hyperparams['simulation_name'] = "{}_d{}_c{}_r{}".format(
        hyperparams['name'], hyperparams['depth'], hyperparams['cutoff'], hyperparams['reps'])

    hyperparams['out_dir'] = os.path.join(args.outdir, hyperparams['simulation_name'], '')
    hyperparams['board_name'] = os.path.join('TensorBoard', hyperparams['simulation_name'], '')

    # save the simulation details and results
    if not os.path.exists(hyperparams['out_dir']):
        os.makedirs(hyperparams['out_dir'])

    return hyperparams


def one_mode_circuit(cutoff, depth=25, sdev=0.1, **kwargs):
    """Construct the one mode circuit ansatz, and return the state vector and gate parameters.

    Args:
        cutoff (int): the Fock basis truncation.
        depth (int): the number of layers to use to construct the circuit.
        sdev (float): the normal standard deviation used to initialise the gate parameters.

    Returns:
        tuple (state, parameters): a tuple contiaining the state vector and a list of the
            gate parameters, as tensorflow tensors.
    """

    # Layer architecture

    # Random initialization of gate parameters.
    # Gate parameters begin close to zero, with values drawn from Normal(0, sdev)
    with tf.name_scope('variables'):
        # displacement parameters
        d_r = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))
        d_phi = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))
        # rotation parameter
        r1 = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))
        # squeezing parameters
        sq_r = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))
        sq_phi = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))
        # rotation parameter
        r2 = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))
        # Kerr gate parameter
        kappa = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))

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

    # Construct the circuit

    # Start SF engine
    eng, q = sf.Engine(1)

    # construct the circuit
    with eng:
        for k in range(depth):
            q = layer(k, q, 0)

    state = eng.run('tf', cutoff_dim=cutoff, eval=False)

    # Extract the state vector
    ket = state.ket()

    return ket, parameters


def two_mode_circuit(cutoff, depth=25, sdev=0.1, **kwargs):
    """Construct the two mode circuit ansatz, and return the state vector and gate parameters.

    Args:
        cutoff (int): the Fock basis truncation.
        depth (int): the number of layers to use to construct the circuit.
        sdev (float): the normal standard deviation used to initialise the gate parameters.

    Returns:
        tuple (state, parameters): a tuple contiaining the state vector and a list of the
            gate parameters, as tensorflow tensors.
    """

    # Random initialization of gate parameters.
    # Gate parameters begin close to zero, with values drawn from Normal(0, sdev)
    with tf.name_scope('variables'):
        # interferometer1 parameters
        theta1 = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))
        phi1 = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))
        r1 = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))
        # squeeze gate
        sq_r = tf.Variable(tf.random_normal(shape=[2, depth], stddev=sdev))
        sq_phi = tf.Variable(tf.random_normal(shape=[2, depth], stddev=sdev))
        # interferometer2 parameters
        theta2 = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))
        phi2 = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))
        r2 = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))
        # displacement gate
        d_r = tf.Variable(tf.random_normal(shape=[2, depth], stddev=sdev))
        d_phi = tf.Variable(tf.random_normal(shape=[2, depth], stddev=sdev))
        # kerr gate
        kappa = tf.Variable(tf.random_normal(shape=[2, depth], stddev=sdev))


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
        for k in range(depth):
            q = layer(k, q)

    state = eng.run('tf', cutoff_dim=cutoff, eval=False)

    # Extract the state vector
    ket = tf.reshape(state.ket(), [-1])

    return ket, parameters


def state_fidelity(ket, target_state):
    """Calculate the fidelity between the target and output state."""
    fidelity = tf.abs(tf.reduce_sum(tf.conj(ket) * target_state)) ** 2
    return fidelity


def optimize(ket, target_state, parameters, cutoff, reps=1000, penalty_strength=100,
        out_dir='sim_results', simulation_name='state_learning', board_name='TensorBoard',
        dump_reps=100, **kwargs):
    """The optimization routine."""

    # ===============================================================================
    # Loss function
    # ===============================================================================

    fidelity = state_fidelity(ket, target_state)
    tf.summary.scalar('fidelity', fidelity)

    # loss function to minimise
    loss = 1 - fidelity
    tf.summary.scalar('loss', loss)

    # ===============================================================================
    # Regularisation
    # ===============================================================================

    # calculate the norm of the state
    state_norm = tf.abs(tf.reduce_sum(tf.conj(ket) * ket)) ** 2
    tf.summary.scalar('norm', state_norm)

    # penalty
    penalty = penalty_strength * (state_norm - 1)**2
    tf.summary.scalar('penalty', penalty)

    # Overall cost function
    cost = loss + penalty
    tf.summary.scalar('cost', cost)

    # ===============================================================================
    # Set up the tensorflow session
    # ===============================================================================

    # Using Adam algorithm for optimization
    optimiser = tf.train.AdamOptimizer()
    minimize_cost = optimiser.minimize(cost)

    # Begin Tensorflow session
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(board_name)
    merge = tf.summary.merge_all()

    # ===============================================================================
    # Run the optimization
    # ===============================================================================

    # Keeps track of fidelity to target state
    fid_progress = []
    # keep track of the cost function
    cost_progress = []

    # Keeps track of best state and fidelity during optimization
    best_state = np.zeros(cutoff)
    best_fid = 0

    start = time.time()
    print('Beginning optimization')

    # Run optimization
    for i in range(reps):

        _, cost_val, fid_val, ket_val, norm_val, penalty_val, params_val = session.run(
            [minimize_cost, cost, fidelity, ket, state_norm, penalty, parameters])
        # Stores fidelity at each step
        cost_progress.append(cost_val)
        fid_progress.append(fid_val)

        if i % dump_reps == 0:
            # print progress
            print("Rep: {} Cost: {:.4f} Fidelity: {:.4f} Norm: {:.4f}".format(
                i, cost_val, fid_val, norm_val))

            if i > 0:
                # save results file
                np.savez(os.path.join(out_dir, simulation_name+'.npz'),
                    **sim_results)


        if i > 0 and fid_val > best_fid:
            best_fid = fid_val
            min_cost = cost_val
            best_state = correct_global_phase(ket_val)

            end = time.time()

            sim_results = {
                # sim details
                'name': HP['name'],
                'target_state': target_state,
                'state_params': HP['params'],
                'cutoff': cutoff,
                'depth': HP['depth'],
                'reps': reps,
                'penalty_strength': penalty_strength,
                'best_runtime': end-start,
                # optimization results
                'best_rep': i,
                'min_cost': cost_val,
                'fidelity': best_fid,
                'cost_progress': np.array(cost_progress),
                'fid_progress': np.array(fid_progress),
                'penalty': penalty_val,
                # optimization output
                'learnt_state': best_state,
                'params': params_val,
                'd_r': params_val[0],
                'd_phi': params_val[1],
                'r1': params_val[2],
                'sq_r': params_val[3],
                'sq_phi': params_val[4],
                'r2': params_val[5],
                'kappa': params_val[6]
            }

    end = time.time()
    print("Elapsed time is {} seconds".format(np.round(end - start)))
    print("Final cost = ", cost_val)
    print("Minimum cost = ", min_cost)
    print("Optimum fidelity = ", best_fid)

    sim_results['runtime'] = end-start
    sim_results['cost_progress'] = np.array(cost_progress)
    sim_results['fid_progress'] = np.array(fid_progress)

    np.savez(os.path.join(out_dir, simulation_name+'.npz'), **sim_results)

    return sim_results


def save_plots(modes, target_state, best_state, cost_progress, offset=-0.11, l=5,
        out_dir='sim_results', simulation_name='state_learner', **kwargs):
    """Generate and save plots"""

    if modes == 1:
        # generate a wigner function plot of the target state
        fig1, ax1 = wigner_3D_plot(target_state, offset=offset, l=l)
        fig1.savefig(os.path.join(out_dir, simulation_name+'_targetWigner.png'))

        # generate a wigner function plot of the learnt state
        fig2, ax2 = wigner_3D_plot(best_state, offset=offset, l=l)
        fig2.savefig(os.path.join(out_dir, simulation_name+'_learntWigner.png'))

        # generate a wavefunction plot of the target state
        figW1, axW1 = wavefunction_plot(target_state, l=l)
        figW1.savefig(os.path.join(out_dir, simulation_name+'_targetWavefunction.png'))

        # generate a wavefunction plot of the learnt state
        figW2, axW2 = wavefunction_plot(best_state, l=l)
        figW2.savefig(os.path.join(out_dir, simulation_name+'_learntWavefunction.png'))
    else:
        # generate a 3D wavefunction plot of the target state
        figW1, axW1 = two_mode_wavefunction_plot(target_state, l=l)
        figW1.savefig(os.path.join(out_dir, simulation_name+'_targetWavefunction.png'))

        # generate a 3D wavefunction plot of the learnt state
        figW2, axW2 = two_mode_wavefunction_plot(best_state, l=l)
        figW2.savefig(os.path.join(out_dir, simulation_name+'_learntWavefunction.png'))

    # generate a cost function plot
    figC, axC = plot_cost(cost_progress)
    figC.savefig(os.path.join(out_dir, simulation_name+'_cost.png'))


if __name__ == "__main__":
    # update hyperparameters with command line arguments
    HP = parse_arguments(HP)

    # set the target state
    target_state = single_photon(HP['cutoff'])
    modes = 1

    # set the target state
    target_state = NOON(5, HP['cutoff'])
    modes = 2

    # calculate the learnt state and return the gate parameters
    if modes == 1:
        ket, parameters = one_mode_circuit(**HP)
    elif modes == 2:
        ket, parameters = two_mode_circuit(**HP)

    # perform the optimization
    res = optimize(ket, target_state, parameters, **HP)

    # save plots
    save_plots(modes, res['learnt_state'], target_state, res['cost_progress'], **HP)
