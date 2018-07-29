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
import argparse
import json

import numpy as np

import tensorflow as tf

import strawberryfields as sf
from strawberryfields.ops import *

from learner.circuits import variational_quantum_circuit
from learner.states import single_photon, ON, hex_GKP, random_state, NOON, correct_global_phase
from learner.plots import wigner_3D_plot, wavefunction_plot, two_mode_wavefunction_plot, plot_cost

# ===============================================================================
# Hyperparameters
# ===============================================================================

# Set the default hyperparameters
HP = {
    #name of the simulation
    'name': 'single_photon',
    # default output directory
    'out_dir': 'sim_results',
    # Target states function. This function accepts an optional
    # list of gate parameters, along with the keyword argument
    # `cutoff`, which determines the Fock basis truncation.
    'target_state_fn': NOON,
    # Dictionary of target state function parameters
    'state_params': {'N':5},
    # Cutoff dimension
    'cutoff': 10,
    # Number of layers
    'depth': 20,
    # Number of steps in optimization routine performing gradient descent
    'reps': 1000,
    # Penalty coefficient to ensure the state is normalized
    'penalty_strength': 0,
    # Standard deviation of active initial parameters
    'active_sd': 0.1,
    # Standard deviation of passive initial parameters
    'passive_sd': 0.1
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

    parser = argparse.ArgumentParser(description='Quantum state preparation learning.')
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
    parser.add_argument('-p', '--state-params',
        type=json.loads, default=defaults["state_params"], help='State parameters')
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

    hyperparams['ID'] = "{}_d{}_c{}_r{}".format(
        hyperparams['name'], hyperparams['depth'], hyperparams['cutoff'], hyperparams['reps'])

    hyperparams['out_dir'] = os.path.join(args.out_dir, hyperparams['ID'], '')
    hyperparams['board_name'] = os.path.join('TensorBoard', hyperparams['ID'], '')

    # save the simulation details and results
    if not os.path.exists(hyperparams['out_dir']):
        os.makedirs(hyperparams['out_dir'])

    return hyperparams


# ===============================================================================
# Optimization functions
# ===============================================================================

def state_fidelity(ket, target_state):
    """Calculate the fidelity between the target and output state."""
    fidelity = tf.abs(tf.reduce_sum(tf.conj(ket) * target_state)) ** 2
    return fidelity


def optimize(ket, target_state, parameters, cutoff, reps=1000, penalty_strength=0,
        out_dir='sim_results', ID='state_learning', board_name='TensorBoard',
        dump_reps=100, **kwargs):
    """The optimization routine.

    Args:
        ket (tensor): tensorflow tensor representing the output state vector of the circuit.
        target_state (array): the target state.
        parameters (list): list of the tensorflow variables representing the gate
            parameters to be optimized in the variational quantum circuit.
        cutoff (int): the simulation Fock basis truncation.
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


    # ===============================================================================
    # Loss function
    # ===============================================================================

    fidelity = state_fidelity(ket, target_state)
    tf.summary.scalar('fidelity', fidelity)

    # loss function to minimise
    loss = tf.abs(tf.reduce_sum(tf.conj(ket) * target_state) - 1)
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
                np.savez(os.path.join(out_dir, ID+'.npz'),
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
                'state_params': HP['state_params'],
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

    np.savez(os.path.join(out_dir, ID+'.npz'), **sim_results)

    return sim_results


def save_plots(target_state, best_state, cost_progress, *, modes, offset=-0.11, l=5,
        out_dir='sim_results', ID='state_learner', **kwargs):
    """Generate and save plots"""

    if modes == 1:
        # generate a wigner function plot of the target state
        fig1, ax1 = wigner_3D_plot(target_state, offset=offset, l=l)
        fig1.savefig(os.path.join(out_dir, ID+'_targetWigner.png'))

        # generate a wigner function plot of the learnt state
        fig2, ax2 = wigner_3D_plot(best_state, offset=offset, l=l)
        fig2.savefig(os.path.join(out_dir, ID+'_learntWigner.png'))

        # generate a wavefunction plot of the target state
        figW1, axW1 = wavefunction_plot(target_state, l=l)
        figW1.savefig(os.path.join(out_dir, ID+'_targetWavefunction.png'))

        # generate a wavefunction plot of the learnt state
        figW2, axW2 = wavefunction_plot(best_state, l=l)
        figW2.savefig(os.path.join(out_dir, ID+'_learntWavefunction.png'))
    elif modes == 2:
        # generate a 3D wavefunction plot of the target state
        figW1, axW1 = two_mode_wavefunction_plot(target_state, l=l)
        figW1.savefig(os.path.join(out_dir, ID+'_targetWavefunction.png'))

        # generate a 3D wavefunction plot of the learnt state
        figW2, axW2 = two_mode_wavefunction_plot(best_state, l=l)
        figW2.savefig(os.path.join(out_dir, ID+'_learntWavefunction.png'))

    # generate a cost function plot
    figC, axC = plot_cost(cost_progress)
    figC.savefig(os.path.join(out_dir, ID+'_cost.png'))


# ===============================================================================
# Main script
# ===============================================================================

if __name__ == "__main__":
    # update hyperparameters with command line arguments
    HP = parse_arguments(HP)

    # set the target state
    target_state = HP['target_state_fn'](cutoff=HP['cutoff'], **HP['state_params'])
    HP['modes'] = int(np.log(target_state.shape[0])/np.log(HP['cutoff']))

    print('------------------------------------------------------------------------')
    print('Hyperparameters:')
    print('------------------------------------------------------------------------')
    for key, val in HP.items():
        print("{}: {}".format(key, val))
    print('------------------------------------------------------------------------')

    # calculate the learnt state and return the gate parameters
    print('Constructing variational quantum circuit...')
    ket, parameters = variational_quantum_circuit(**HP)

    # flatten ket to take into account two mode states
    ket = tf.reshape(ket, [-1])

    # perform the optimization
    print('Beginning optimization...')
    res = optimize(ket, target_state, parameters, **HP)

    # save plots
    print('Generating plots...')
    save_plots(res['learnt_state'], target_state, res['cost_progress'], **HP)
