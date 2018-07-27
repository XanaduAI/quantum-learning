#!/usr/bin/env python3
import os
import time
from itertools import product
import argparse

import numpy as np

import tensorflow as tf

import strawberryfields as sf
from strawberryfields.ops import *

from gates import (cubic_phase, DFT, random_unitary, cross_kerr,
    min_cutoff, unitary_state_fidelity)

from plots import (wigner_3D_plot, wavefunction_plot,
    two_mode_wavefunction_plot, plot_cost, one_mode_unitary_plots)


# ===============================================================================
# Hyperparameters
# ===============================================================================

# Set the default hyperparameters
HP = {
    #name of the simulation
    'name': 'gate_synthesis',
    # default output directory
    'out_dir': 'sim_results',
    # Gate parameters
    'params': [0.01],
    # Precision
    'eps': 0.0001,
    # offset when calculating matrix exponential
    'offset': 20,
    # Cutoff dimension
    'cutoff': 16,
    # Gate cutoff/truncation
    'gate_cutoff': 5,
    # Number of layers
    'depth': 25,
    # Number of steps in optimization routine performing gradient descent
    'reps': 1000,
    # Penalty coefficient to ensure the state is normalized
    'penalty_strength': 0,
    # Standard deviation of active initial parameters
    'active_sd': 0.0001,
    # Standard deviation of passive initial parameters
    'passive_sd': 1
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

    parser = argparse.ArgumentParser(description='Quantum gate synthesis.')
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
        type=float, nargs='+', default=defaults["params"], help='Gate parameters')
    parser.add_argument('-c', '--cutoff',
        type=int, default=defaults["cutoff"], help='Fock basis truncation')
    parser.add_argument('-g', '--gate-cutoff',
        type=int, default=defaults["gate_cutoff"], help='Gate/unitary truncation')
    parser.add_argument('-t', '--offset',
        type=int, default=defaults["offset"], help='Matrix exponential offset')
    parser.add_argument('-d', '--depth',
        type=int, default=defaults["depth"], help='Number of layers')
    parser.add_argument('-P', '--penalty-strength',
        type=int, default=defaults["penalty_strength"], help='Regularisation penalty strength')
    args = parser.parse_args()

    hyperparams = {}
    hyperparams.update(defaults)
    hyperparams.update(vars(args))

    hyperparams['batch_size'] = hyperparams['gate_cutoff']

    if args.debug:
        hyperparams['depth'] = 1
        hyperparams['reps'] = 5
        hyperparams['name'] += "_debug"

    hyperparams['simulation_name'] = "{}_d{}_c{}_g{}_r{}".format(
        hyperparams['name'], hyperparams['depth'], hyperparams['cutoff'], hyperparams['gate_cutoff'], hyperparams['reps'])

    hyperparams['out_dir'] = os.path.join(args.outdir, hyperparams['simulation_name'], '')
    hyperparams['board_name'] = os.path.join('TensorBoard', hyperparams['simulation_name'], '')

    # save the simulation details and results
    if not os.path.exists(hyperparams['out_dir']):
        os.makedirs(hyperparams['out_dir'])

    return hyperparams


def one_mode_circuit(cutoff, gate_cutoff, depth=25, active_sd=0.0001, passive_sd=0.1, **kwargs):
    """Construct the one mode circuit ansatz, and return the learnt unitary and gate parameters.

    Args:
        cutoff (int): the simulation Fock basis truncation.
        gate_cutoff (int): the unitary Fock basis truncation. Must be less than or equal to cutoff.
        depth (int): the number of layers to use to construct the circuit.
        active_sd (float): the normal standard deviation used to initialise the active gate parameters.
        passive_sd (float): the normal standard deviation used to initialise the passive gate parameters.

    Returns:
        tuple (state, parameters): a tuple contiaining the state vector and a list of the
            gate parameters, as tensorflow tensors.
    """

    # Layer architecture

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

    # produce 1 x batch_size array of initial states
    in_state = np.arange(gate_cutoff)

    # Start SF engine
    eng, q = sf.Engine(1)

    # construct the circuit
    with eng:
        Fock(in_state) | q[0]
        for k in range(depth):
            q = layer(k, q, 0)

    state = eng.run('tf', cutoff_dim=cutoff, eval=False, batch_size=gate_cutoff)

    # Extract the state vector
    ket = state.ket()

    return ket, parameters


def real_unitary_overlaps(ket, target_unitary, gate_cutoff):
    """Calculate the overlaps between the target and output unitaries."""

    in_state = np.arange(gate_cutoff)

    # extract action of the target unitary acting on
    # the allowed input fock states. This produces an array
    # with elements indexed by (k,l), the action based on the output state.
    target_kets = np.array([target_unitary[:, i] for i in in_state])
    target_kets = tf.constant(target_kets, dtype=tf.complex64)

    # real overlaps
    overlaps = tf.real(tf.einsum('bi,bi->b', tf.conj(target_kets), ket))
    return overlaps


def optimize(ket, target_unitary, parameters, cutoff, gate_cutoff, reps=1000, penalty_strength=100,
        out_dir='sim_results', simulation_name='gate_synthesis', board_name='TensorBoard',
        dump_reps=100, **kwargs):
    """The optimization routine."""

    # ===============================================================================
    # Loss function
    # ===============================================================================

    in_state = np.arange(gate_cutoff)

    # real overlaps
    overlaps = real_unitary_overlaps(ket, target_unitary, gate_cutoff)
    for idx, state in enumerate(in_state.T):
        tf.summary.scalar('overlap_{}'.format(state), tf.abs(overlaps[idx]))

    # average of the real overlaps
    mean_overlap = tf.reduce_mean(overlaps)
    tf.summary.scalar("mean_overlap", mean_overlap)

    # loss function
    loss = tf.reduce_sum(tf.abs(overlaps - 1))
    tf.summary.scalar('loss', loss)


    # ===============================================================================
    # Regularisation
    # ===============================================================================

    # calculate the norms of the states
    state_norms = tf.abs(tf.einsum('bi,bi->b', ket, tf.conj(ket)))
    norm_deviation = tf.reduce_sum((state_norms - 1)**2)/gate_cutoff

    # penalty
    penalty = penalty_strength*norm_deviation
    tf.summary.scalar('penalty', penalty)

    # overall cost function
    cost = loss + penalty
    tf.summary.scalar('cost', cost)

    # ===============================================================================
    # Set up the tensorflow session
    # ===============================================================================

    # Using Adam algorithm for optimization
    optimiser = tf.train.AdamOptimizer()
    min_cost_optimize = optimiser.minimize(cost)

    # Begin Tensorflow session
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(board_name)
    merge = tf.summary.merge_all()

    # ===============================================================================
    # Run the optimization
    # ===============================================================================

    # Keeps track of fidelity to target unitary
    overlap_progress = []
    # keep track of the cost function
    cost_progress = []

    # Keeps track of best learnt unitary and overlaps during optimization
    best_mean_overlap = 0
    best_min_overlap = 0
    best_max_overlap = 0

    start = time.time()
    print('Beginning optimization')


    # Run optimization
    for i in range(reps):

        _, cost_val, overlaps_val, ket_val, penalty_val, params_val = session.run(
            [min_cost_optimize, cost, overlaps, ket, penalty, parameters])
        mean_overlap_val = np.mean(overlaps_val)
        min_overlap_val = min(overlaps_val)
        max_overlap_val = max(overlaps_val)

        # store cost at each step
        cost_progress.append(cost_val)
        overlap_progress.append(overlaps_val)

        if i % dump_reps == 0:
            # print progress
            print("Rep: {} Cost: {:.4f} Overlaps: Mean = {:.4f}, Min = {:.4f}, Max = {:.4f}".format(
                i, cost_val, mean_overlap_val, min_overlap_val, max_overlap_val))

            summary = session.run(merge)
            writer.add_summary(summary, i)

            if i > 0:
                # save results file
                np.savez(os.path.join(out_dir, simulation_name+'.npz'),
                    **sim_results)

        if i > 0 and mean_overlap_val > best_mean_overlap:
            best_mean_overlap = mean_overlap_val
            best_min_overlap = min_overlap_val
            best_max_overlap = max_overlap_val

            min_cost = cost_val
            eq_state_target, eq_state_learnt, state_fid = unitary_state_fidelity(target_unitary, ket_val)

            end = time.time()

            sim_results = {
                # sim details
                'name': HP['name'],
                'target_unitary': target_unitary,
                'U_param': HP['params'],
                'eps': HP['eps'],
                'cutoff': cutoff,
                'gate_cutoff': gate_cutoff,
                'depth': HP['depth'],
                'reps': HP['reps'],
                'penalty_strength': HP['penalty_strength'],
                'best_runtime': end-start,
                # optimization results
                'best_rep': i,
                'mean_overlap': mean_overlap_val,
                'min_overlap': min_overlap_val,
                'max_overlap': max_overlap_val,
                'min_cost': cost_val,
                'cost_progress': np.array(cost_progress),
                'mean_overlap_progress': np.mean(np.array(overlap_progress), axis=1),
                'min_overlap_progress': np.min(np.array(overlap_progress), axis=1),
                'max_overlap_progress': np.max(np.array(overlap_progress), axis=1),
                'penalty': penalty_val,
                # optimization output
                'U_output': ket_val,
                'params': params_val,
                'r1': params_val[0],
                'sq_r': params_val[1],
                'sq_phi': params_val[2],
                'r2': params_val[3],
                'disp_r': params_val[4],
                'disp_phi': params_val[5],
                'kappa': params_val[6],
                # equal superposition state test
                'eq_state_learnt': eq_state_learnt,
                'eq_state_target': eq_state_target,
                'eq_state_fidelity': state_fid
            }


    end = time.time()
    print("\nElapsed time is {} seconds".format(np.round(end - start)))
    print("Final cost = ", cost_val)
    print("Minimum cost = ", min_cost)

    print("\nMean overlap = {}".format(best_mean_overlap))
    print("Min overlap = {}".format(best_min_overlap))
    print("Max overlap = {}".format(best_max_overlap))

    print("\nEqual superposition state fidelity = ", state_fid)
    print("Target state = ", eq_state_target)
    print("Learnt state = ", eq_state_learnt)

    # save the simulation details and results
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    sim_results['runtime'] = end-start
    sim_results['cost_progress'] = np.array(cost_progress),
    sim_results['mean_overlap_progress'] = np.mean(np.array(overlap_progress), axis=1)
    sim_results['min_overlap_progress'] = np.min(np.array(overlap_progress), axis=1)
    sim_results['max_overlap_progress'] = np.max(np.array(overlap_progress), axis=1)

    np.savez(os.path.join(out_dir, simulation_name+'.npz'), **sim_results)
    return sim_results


def save_plots(modes, target_unitary, learnt_unitary, eq_state_learnt, eq_state_target,
        cost_progress, offset=-0.11, l=5, out_dir='sim_results',
        simulation_name='gate_synthesis', **kwargs):
    """Generate and save plots"""

    if modes == 1:
        # generate a wigner function plot of the target state
        fig1, ax1 = wigner_3D_plot(eq_state_target, offset=offset, l=l)
        fig1.savefig(os.path.join(out_dir, simulation_name+'_targetWigner.png'))

        # generate a wigner function plot of the learnt state
        fig2, ax2 = wigner_3D_plot(eq_state_learnt, offset=offset, l=l)
        fig2.savefig(os.path.join(out_dir, simulation_name+'_learntWigner.png'))

        # generate a wavefunction plot of the target state
        figW1, axW1 = one_mode_unitary_plots(target_unitary, learnt_unitary)
        figW1.savefig(os.path.join(out_dir, simulation_name+'_unitaryPlot.png'))
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

    # ===============================================================================
    # Target unitary
    # ===============================================================================

    # set the target state
    target_unitary = random_unitary(5, HP['cutoff'])
    modes = 1

    # check cutoff is high enough for precision value
    Ubig = random_unitary(5, HP['cutoff']+60)
    min_cut = min_cutoff(Ubig, HP['eps'], HP['gate_cutoff'], HP['cutoff']+60)
    if min_cut >  HP['cutoff']:
        print("Warning! Minimum cutoff for specificed precision "
              "is {}, but the current cutoff is {}.".format(
                min_cut,  HP['cutoff']))

    # calculate the learnt state and return the gate parameters
    if modes == 1:
        ket, parameters = one_mode_circuit(**HP)
    elif modes == 2:
        ket, parameters = two_mode_circuit(**HP)

    # perform the optimization
    res = optimize(ket, target_unitary, parameters, **HP)

    # save plots
    save_plots(modes, **res)
