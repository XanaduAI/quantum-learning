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
import os
import time
from itertools import product
import argparse
import json

import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf

import strawberryfields as sf
from strawberryfields.ops import *

from learner.circuits import variational_quantum_circuit

from learner.gates import (cubic_phase, DFT, random_unitary, cross_kerr, get_modes,
    unitary_state_fidelity, sample_average_fidelity, process_fidelity, average_fidelity)

from learner.plots import (wigner_3D_plot, wavefunction_plot,
    two_mode_wavefunction_plot, plot_cost, one_mode_unitary_plots, two_mode_unitary_plots)


# ===============================================================================
# Hyperparameters
# ===============================================================================

# Set the default hyperparameters
HP = {
    #name of the simulation
    'name': 'random_gif',
    # default output directory
    'out_dir': 'sim_results',
    # Target unitary function. This function accepts an optional
    # list of gate parameters, along with required keyword argument
    # `cutoff`, which determines the Fock basis truncation.
    'target_unitary_fn': random_unitary,
    # Dictionary of target unitary function arguments
    'target_params': {'size': 4},
    # Cutoff dimension
    'cutoff': 10,
    # Gate cutoff/truncation
    'gate_cutoff': 4,
    # Number of layers
    'depth': 25,
    # Number of steps in optimization routine performing gradient descent
    'reps': 2000,
    # Penalty coefficient to ensure the state is normalized
    'penalty_strength': 0,
    # Standard deviation of active initial parameters
    'active_sd': 0.0001,
    # Standard deviation of passive initial parameters
    'passive_sd': 0.1,
    # Does the target unitary map Fock states within the gate
    # cutoff outside of the gate cutoff? If unsure, set to True.
    'maps_outside': False,
}


# ===============================================================================
# Parse command line arguments
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
    parser.add_argument('-o', '--out-dir',
        type=str, default=defaults["out_dir"], help='Output directory')
    parser.add_argument('-s', '--dump-reps',
        type=int, default=100, help='Steps at which to save output')
    parser.add_argument('-D', '--debug',
        action='store_true', help="Debug mode")
    # simulation settings
    parser.add_argument('-r', '--reps',
        type=int, default=defaults["reps"], help='Optimization steps')
    parser.add_argument('-p', '--target-params',
        type=json.loads, default=defaults["target_params"], help='Gate parameters')
    parser.add_argument('-c', '--cutoff',
        type=int, default=defaults["cutoff"], help='Fock basis truncation')
    parser.add_argument('-g', '--gate-cutoff',
        type=int, default=defaults["gate_cutoff"], help='Gate/unitary truncation')
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

    hyperparams['ID'] = "{}_d{}_c{}_g{}_r{}".format(
        hyperparams['name'], hyperparams['depth'], hyperparams['cutoff'], hyperparams['gate_cutoff'], hyperparams['reps'])

    hyperparams['out_dir'] = os.path.join(args.out_dir, hyperparams['ID'], '')
    hyperparams['board_name'] = os.path.join('TensorBoard', hyperparams['ID'], '')

    # save the simulation details and results
    if not os.path.exists(hyperparams['out_dir']):
        os.makedirs(hyperparams['out_dir'])

    return hyperparams


# ===============================================================================
# Optimization functions
# ===============================================================================

def real_unitary_overlaps(ket, target_unitary, gate_cutoff, cutoff):
    """Calculate the overlaps between the target and output unitaries.

    Args:
        ket (tensor): tensorflow tensor representing the output (batched) state vector
            of the circuit. This can be used to determine the learnt unitary.
            This tensor must be of size [gate_cutoff, cutoff, ..., cutoff].
        target_unitary (array): the target unitary.
        gate_cutoff (int): the number of input-output relations. Must be less than
            or equal to the simulation cutoff.
    """
    m = len(ket.shape)-1

    if m == 1:
        # one mode unitary
        in_state = np.arange(gate_cutoff)

        # extract action of the target unitary acting on
        # the allowed input fock states.
        target_kets = np.array([target_unitary[:, i] for i in in_state])
        target_kets = tf.constant(target_kets, dtype=tf.complex64)

        # real overlaps
        overlaps = tf.real(tf.einsum('bi,bi->b', tf.conj(target_kets), ket))

    elif m == 2:
        # two mode unitary
        fock_states = np.arange(gate_cutoff)
        in_state = np.array(list(product(fock_states, fock_states)))

        # extract action of the target unitary acting on
        # the allowed input fock states.
        target_unitary_sf = np.einsum('ijkl->ikjl', target_unitary.reshape([cutoff]*4))
        target_kets = np.array([target_unitary_sf[:, i, :, j] for i, j in in_state])
        target_kets = tf.constant(target_kets, dtype=tf.complex64)

        # real overlaps
        overlaps = tf.real(tf.einsum('bij,bij->b', tf.conj(target_kets), ket))

    for idx, state in enumerate(in_state.T):
        tf.summary.scalar('overlap_{}'.format(state), tf.abs(overlaps[idx]))

    return overlaps


def optimize(ket, target_unitary, parameters, cutoff, gate_cutoff, reps=1000, penalty_strength=0,
        out_dir='sim_results', ID='gate_synthesis', board_name='TensorBoard',
        dump_reps=100, **kwargs):
    """The optimization routine.

    Args:
        ket (tensor): tensorflow tensor representing the output (batched) state vector
            of the circuit. This can be used to determine the learnt unitary.
            This tensor must be of size [gate_cutoff, cutoff, ..., cutoff].
        target_unitary (array): the target unitary.
        parameters (list): list of the tensorflow variables representing the gate
            parameters to be optimized in the variational quantum circuit.
        gate_cutoff (int): the number of input-output relations. Must be less than
            or equal to the simulation cutoff.
        reps (int): the number of optimization repititions.
        penalty_strength (float): the strength of the penalty to apply to optimized states
            deviating from a norm of 1.
        out_dir (str): directory to store saved output.
        ID (str): the ID of the simulation. The optimization output is saved in the directory
            out_dir/ID.
        board_name (str): the folder to store data for TensorBoard.
        dump_reps (int): the repitition frequency at which to save output.

    Returns:
        dict: a dictionary containing the hyperparameters and results of the optimization.
    """

    d = gate_cutoff
    c = cutoff
    m = len(ket.shape)-1

    # ===============================================================================
    # Loss function
    # ===============================================================================

    # real overlaps
    overlaps = real_unitary_overlaps(ket, target_unitary, gate_cutoff, cutoff)

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
    if m == 1:
        # one mode unitary
        state_norms = tf.abs(tf.einsum('bi,bi->b', ket, tf.conj(ket)))
    elif m == 2:
        # two mode unitary
        state_norms = tf.abs(tf.einsum('bij,bij->b', ket, tf.conj(ket)))

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

        if m == 1:
            learnt_unitary = ket_val.T
        elif m == 2:
            learnt_unitary = ket_val.reshape(d**2, c**2).T

        c = learnt_unitary.shape[0]
        d = learnt_unitary.shape[1]
        Ur = learnt_unitary[:d, :d]

        vmax = np.max([Ur.real, Ur.imag])
        vmin = np.min([Ur.real, Ur.imag])
        cmax = max(vmax, vmin)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].matshow(Ur.real, cmap=plt.get_cmap('Reds'), vmin=-cmax, vmax=cmax)
        ax[1].matshow(Ur.imag, cmap=plt.get_cmap('Greens'), vmin=-cmax, vmax=cmax)

        for a in ax.ravel():
            a.tick_params(bottom=False,labelbottom=False,
                          top=False,labeltop=False,
                          left=False,labelleft=False,
                          right=False,labelright=False)

        ax[0].set_xlabel(r'$\mathrm{Re}(U)$')
        ax[1].set_xlabel(r'$\mathrm{Im}(U)$')

        for a in ax.ravel():
            a.tick_params(color='white', labelcolor='white')
            for spine in a.spines.values():
                spine.set_edgecolor('white')

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, '{}.png'.format(i).zfill(4)))
        plt.close(fig)

        if i % dump_reps == 0:
            # print progress
            print("Rep: {} Cost: {:.4f} Overlaps: Mean = {:.4f}, Min = {:.4f}, Max = {:.4f}".format(
                i, cost_val, mean_overlap_val, min_overlap_val, max_overlap_val))

            summary = session.run(merge)
            writer.add_summary(summary, i)

            if i > 0:
                # save results file
                np.savez(os.path.join(out_dir, ID+'.npz'),
                    **sim_results)

        if i > 0 and mean_overlap_val > best_mean_overlap:
            end = time.time()

            best_mean_overlap = mean_overlap_val
            best_min_overlap = min_overlap_val
            best_max_overlap = max_overlap_val

            min_cost = cost_val

            if m == 1:
                learnt_unitary = ket_val.T
            elif m == 2:
                learnt_unitary = ket_val.reshape(d**2, c**2).T

            eq_state_target, eq_state_learnt, state_fid = unitary_state_fidelity(target_unitary, learnt_unitary, cutoff)
            Fe = process_fidelity(target_unitary, learnt_unitary, cutoff)
            avgF = average_fidelity(target_unitary, learnt_unitary, cutoff)
            # avgFs = sample_average_fidelity(target_unitary, learnt_unitary, cutoff)

            sim_results = {
                # sim details
                'name': HP['name'],
                'ID': HP['ID'],
                'target_unitary': target_unitary,
                'target_params': HP['target_params'],
                # 'eps': HP['eps'],
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
                'process_fidelity': Fe,
                'avg_fidelity': avgF,
                'min_cost': cost_val,
                'cost_progress': np.array(cost_progress),
                'mean_overlap_progress': np.mean(np.array(overlap_progress), axis=1),
                'min_overlap_progress': np.min(np.array(overlap_progress), axis=1),
                'max_overlap_progress': np.max(np.array(overlap_progress), axis=1),
                'penalty': penalty_val,
                # optimization output
                'learnt_unitary': learnt_unitary,
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

    avgFs = sample_average_fidelity(target_unitary, learnt_unitary, cutoff)
    sim_results['sample_avg_fidelity'] = avgFs

    print("\nProcess fidelity = {}".format(Fe))
    print("Average fidelity = {}".format(avgF))
    print("Sampled average fidelity = {}".format(avgFs))

    print("\nEqual superposition state fidelity = ", state_fid)
    # print("Target state = ", eq_state_target)
    # print("Learnt state = ", eq_state_learnt)

    # save the simulation details and results
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    sim_results['runtime'] = end-start
    sim_results['cost_progress'] = np.array(cost_progress)
    sim_results['mean_overlap_progress'] = np.mean(np.array(overlap_progress), axis=1)
    sim_results['min_overlap_progress'] = np.min(np.array(overlap_progress), axis=1)
    sim_results['max_overlap_progress'] = np.max(np.array(overlap_progress), axis=1)

    np.savez(os.path.join(out_dir, ID+'.npz'), **sim_results)
    return sim_results


def save_plots(target_unitary, learnt_unitary, eq_state_learnt, eq_state_target,
        cost_progress, *, modes, offset=-0.11, l=5, out_dir='sim_results',
        ID='gate_synthesis', **kwargs):
    """Generate and save plots"""

    square = not kwargs.get('maps_outside', True)

    if modes == 1:
        # generate a wigner function plot of the target state
        fig1, ax1 = wigner_3D_plot(eq_state_target, offset=offset, l=l)
        fig1.savefig(os.path.join(out_dir, ID+'_targetWigner.png'))

        # generate a wigner function plot of the learnt state
        fig2, ax2 = wigner_3D_plot(eq_state_learnt, offset=offset, l=l)
        fig2.savefig(os.path.join(out_dir, ID+'_learntWigner.png'))

        # generate a matrix plot of the target and learnt unitaries
        figW1, axW1 = one_mode_unitary_plots(target_unitary, learnt_unitary, square=square)
        figW1.savefig(os.path.join(out_dir, ID+'_unitaryPlot.png'))
    elif modes == 2:
        # generate a 3D wavefunction plot of the target state
        figW1, axW1 = two_mode_wavefunction_plot(eq_state_target, l=l)
        figW1.savefig(os.path.join(out_dir, ID+'_targetWavefunction.png'))

        # generate a 3D wavefunction plot of the learnt state
        figW2, axW2 = two_mode_wavefunction_plot(eq_state_learnt, l=l)
        figW2.savefig(os.path.join(out_dir, ID+'_learntWavefunction.png'))

        # generate a matrix plot of the target and learnt unitaries
        figM1, axM1 = two_mode_unitary_plots(target_unitary, learnt_unitary, square=square)
        figM1.savefig(os.path.join(out_dir, ID+'_unitaryPlot.png'))

    # generate a cost function plot
    figC, axC = plot_cost(cost_progress)
    figC.savefig(os.path.join(out_dir, ID+'_cost.png'))


# ===============================================================================
# Main script
# ===============================================================================

if __name__ == "__main__":
    # update hyperparameters with command line arguments
    HP = parse_arguments(HP)

    target_unitary = HP['target_unitary_fn'](cutoff=HP['cutoff'], **HP['target_params'])
    HP['modes'] = get_modes(target_unitary, HP['cutoff'])
    HP['batch_size'] = HP['gate_cutoff']**HP['modes']

    print('------------------------------------------------------------------------')
    print('Hyperparameters:')
    print('------------------------------------------------------------------------')
    for key, val in HP.items():
        print("{}: {}".format(key, val))
    print('------------------------------------------------------------------------')

    # produce a batch_size array of one mode initial Fock states
    in_ket = np.zeros([HP['gate_cutoff'], HP['cutoff']])
    np.fill_diagonal(in_ket, 1)

    if HP['modes'] == 2:
        # take the outer product of the one mode input states
        in_ket = np.einsum('ij,kl->ikjl', in_ket, in_ket)
        # reshape to be the correct shape for Strawberry Fields
        in_ket = in_ket.reshape(HP['gate_cutoff']**2, HP['cutoff'], HP['cutoff'])

    # calculate the learnt state and return the gate parameters
    print('Constructing variational quantum circuit...')
    ket, parameters = variational_quantum_circuit(input_state=in_ket, **HP)

    # perform the optimization
    print('Beginning optimization...')
    res = optimize(ket, target_unitary, parameters, **HP)

    # save plots
    print('Generating plots...')
    save_plots(target_unitary, res['learnt_unitary'], res['eq_state_learnt'],
        res['eq_state_target'], res['cost_progress'], **HP)
