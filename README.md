
<img align="left" src="https://github.com/XanaduAI/quantum-learning/blob/master/static/photon.gif">

# Quantum state learning and gate synthesis


This repository contains the source code used to produce the results presented in *"Machine learning method for state preparation and gate synthesis on photonic quantum computers"* [Quantum Science and Technology, Volume 4, Number 2 ](https://arxiv.org/abs/1807.10781).

## Contents

* `state_learner.py`: a Python script to automate quantum state learning using continuous-variable (CV) variational quantum circuits. Simply specify your one- or two-mode target state, along with other hyperparameters, and this script automatically constructs and optimizes the variational quantum circuit.

* `gate_synthesis.py`: a Python script to automate quantum gate synthesis using continuous-variable (CV) variational quantum circuits. Simply specify your one- or two-mode target unitary, along with other hyperparameters, and this script automatically constructs and optimizes the variational quantum circuit.

* `learner`: a Python module containing the following importable Python files:
	- `states.py`: functions to generate the states analyzed in the paper.
	- `gates.py`: functions to generate the states analyzed in the paper.
	- `plots.py`: functions to generate the plots and visualizations in the paper.
	- `circuits.py`: functions to construct the one-mode and two-mode variational circuits as described in the paper *[Continuous-variable quantum neural networks](https://arxiv.org/abs/1806.06871)*.

* Jupyter notebooks: two Jupyter notebooks are also provided, [`StateLearning.ipynb`](https://github.com/XanaduAI/quantum-learning/blob/master/notebooks/StateLearning.ipynb) and [`GateSynthesis.ipynb`](https://github.com/XanaduAI/quantum-learning/blob/master/notebooks/GateSynthesis.ipynb), walking through the process of state learning and gate synthesis respectively.

* Results and data presented in [arXiv:1807.10781](https://arxiv.org/abs/1807.10781): contains two sub folders; `gate_results` and `state_results`, each with NumPy `npz` files containing the hyperparameters and circuit parameters of the synthesized gates and learnt states presented in arXiv:1807.10781. Refer to these folder for more informtion on the data contained.

<p align="center">
	<img src="https://github.com/XanaduAI/quantum-learning/blob/master/static/random.gif">
</p>

## Requirements

To construct and optimize the variational quantum circuits, these scripts and notebooks use the TensorFlow backend of [Strawberry Fields](https://github.com/XanaduAI/strawberryfields). In addition, matplotlib is required for generating output plots, and [OpenFermion](https://github.com/quantumlib/OpenFermion) is used to construct target gate unitaries.


## Using the scripts

To use the scripts, simply set the hyperparameters - either by modifying the default hyperparameters in the file itself, or passing the relevant command line arguments - and then run the script using Python 3:

```bash
python3 state_learner.py
```

The outputs of the simulations will be saved in the directory `out_dir/simulation_ID`, with `out_dir` set by the hyperparameter dictionary, and `simulation_ID` determined automatically based on the simulation name.

After every optimization, plots and visualisations of the target and learnt state/gate are generated, as well as a NumPy multi-array file `simulation_ID.npz`. This file contains all the hyperparameters that characterize the simulation, as well as the results - including the target and learnt state/gate, and the optimized variational circuit gate parameters.

To access the saved data, the file can be loaded using NumPy:

```python
results = np.load('simulation_ID.npz')
```

The individual hyperparameters and results can then be accessed via the respective key. For example, to extract the learnt state, as well as a list of the variational circuit layer squeezing magnitudes:

```python
learnt_state = results['learnt_state']
squeezing = results['sq_r']
```

For a list of all available keys, simply run `print(results.keys())`.

### State learner hyperparameters

The following hyperparameters can be set for the script `state_learner.py`:


|   Hyperparameter  | Command line argument |                                                                                                                                                                       Description                                                                                                                                                                       |
| ----------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`            | `-n`/`--name`         | The name of the simulation                                                                                                                                                                                                                                                                                                                              |
| `out_dir`         | `-o`/`--out-dir`      | Output directory for saving the simulation results                                                                                                                                                                                                                                                                                                      |
| `target_state_fn` | n/a                   | Function for generating the target state for optimization. This function can accept an optional list of state parameters, along with the required keyword argument `cutoff` which determines the Fock basis truncation. The function must return a NumPy array of length `[cutoff]` for single mode states, and length `[cutoff^2]` for two mode states. |
| `state_params`    | `-p`/`--state-params` | Optional dictionary of state parameters to pass to the target state function, for example `{"N": 3}`.                                                                                                                                                                                                                                             |
| `cutoff`          | `-c`/`--cutoff`       | The simulation Fock basis truncation.                                                                                                                                                                                                                                                                                                                   |
| `depth`           | `-d`/`--depth`        | Number of layers in the variational quantum circuit.                                                                                                                                                                                                                                                                                                    |
| `reps`            | `-r`/`--reps`         | Number of optimization steps to perform.                                                                                                                                                                                                                                                                                                                |
| `active_sd`       | n/a                   | Standard deviation of initial photon non-preserving gate parameters in the variational quantum circuit.                                                                                                                                                                                                                                                 |
| `passive_sd`      | n/a                   | Standard deviation of initial photon preserving gate parameters in the variational quantum circuit.                                                                                                                                                                                                                                                 |

The target state function can be defined manually in Python and added to the hyperparameters dictionary, or imported from the file `learners/states.py`. After the optimization is complete, the state learning script will automatically generate the following plots:

* Cost function vs. optimization step
* Wigner functions of the target state and the learnt state (for one mode states only)
* Wavefunctions of the target state and the learnt state.

### Gate synthesis hyperparameters

The following hyperparameters can be set for the script `gate_synthesis.py`:


|   Hyperparameter  | Command line argument |                                                                                                                                                                       Description                                                                                                                                                                       |
| ----------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`            | `-n`/`--name`         | The name of the simulation                                                                                                                                                                                                                                                                                                                              |
| `out_dir`         | `-o`/`--out-dir`      | Output directory for saving the simulation results                                                                                                                                                                                                                                                                                                      |
| `target_unitary_fn` | n/a                   | Function for generating the target unitary for synthesis. This function can accept an optional list of gate parameters, along with the required keyword argument `cutoff` which determines the Fock basis truncation. The function must return a NumPy array of size `[cutoff, cutoff]` for single mode unitaries, and size `[cutoff^2, cutoff^2]` for two mode unitaries. |
| `target_params`    | `-p`/`--target-params` | Optional dictionary of gate parameters to pass to the target unitary function, for example `{"gamma": 0.01}`.                                                                                                                                                                                                                                             |
| `cutoff`          | `-c`/`--cutoff`       | The simulation Fock basis truncation.                                                                                                                                                                                                                                                                                                                   |
| `gate_cutoff`          | `-g`/`--gate-cutoff`       | the d-dimensional subspace in which the target unitary acts. The value of the gate cutoff must be less than or equal to the simulation cutoff.	                                                                                                                                                                                                                                                                                                                |
| `depth`           | `-d`/`--depth`        | Number of layers in the variational quantum circuit.                                                                                                                                                                                                                                                                                                    |
| `reps`            | `-r`/`--reps`         | Number of optimization steps to perform.                                                                                                                                                                                                                                                                                                                |
| `active_sd`       | n/a                   | Standard deviation of initial photon non-preserving gate parameters in the variational quantum circuit.                                                                                                                                                                                                                                                 |
| `passive_sd`      | n/a                   | Standard deviation of initial photon preserving gate parameters in the variational quantum circuit.                                                                                                                                                                                                                                                 |
| `maps_outside`      | n/a                   | Set to `True` if the target unitary maps Fock states within the d-dimensional subspace specified by the gate cutoff to Fock states outside of the d-dimensional subspace. If unsure, set to True.                                                                                                                                                                                                                                              |


The target unitary function can be defined manually in Python and added to the hyperparameters dictionary, or imported from the file `learners/gates.py`. After the optimization is complete, the gate synthesis script will automatically calculate the process fidelity and average fidelity of the two unitaries, and generate the following plots:

* Cost function vs. optimization step
* Wigner functions of the target unitary and the learnt unitary applied to the equal superposition state (for one mode states only)
* Wavefunctions of the target unitary and the learnt unitary applied to the equal superposition state (for two mode states only)
* Matrix plots of the real and imaginary elements of the target unitary and learnt unitary.

## Authors

Juan Miguel Arrazola, Thomas R. Bromley, Josh Izaac, Casey R. Myers, Kamil Brádler, and Nathan Killoran.

If you are doing any research using this source code and Strawberry Fields, please cite the following two papers:

> Juan Miguel Arrazola, Thomas R. Bromley, Josh Izaac, Casey R. Myers, Kamil Brádler, and Nathan Killoran. Machine learning method for state preparation and gate synthesis on photonic quantum computers. arXiv, 2018. [arXiv:1807.10781](https://arxiv.org/abs/1807.10781)

> Nathan Killoran, Josh Izaac, Nicolás Quesada, Ville Bergholm, Matthew Amy, and Christian Weedbrook. Strawberry Fields: A Software Platform for Photonic Quantum Computing. arXiv, 2018. [arXiv:1804.03159](https://arxiv.org/abs/1804.03159)

## License

This source code is free and open source, released under the Apache License, Version 2.0.
