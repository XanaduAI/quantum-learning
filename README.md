# Quantum state learning and gate synthesis

<img align="left" src="https://github.com/XanaduAI/quantum-learning/blob/master/static/photon.gif">

This repository contains the source code used to produce the results presented in *"Machine learning method for state preparation and gate synthesis on photonic
quantum computers"*.

It includes the following content:

* `state_learner.py`: a Python script to automate quantum state learning using continuous-variable (CV) variational quantum circuits. Simply specify your one- or two-mode target state, along with other hyperparameters, and this script automatically constructs and optimizes the variational quantum circuit.

* `gate_synthesis.py`: a Python script to automate quantum gate synthesis using continuous-variable (CV) variational quantum circuits. Simply specify your one- or two-mode target unitary, along with other hyperparameters, and this script automatically constructs and optimizes the variational quantum circuit.

* Jupyter notebooks: two Jupyter notebooks are also provided, `StateLearning.ipynb` and `GateSynthesis.ipynb`, walking through the process of state learning and gate synthesis respectively.

## Requirements

To construct and optimize the variational quantum circuits, these scripts and notebooks use the TensorFlow backend of [Strawberry Fields](https://github.com/XanaduAI/strawberryfields). In addition, matplotlib is required for generating output plots, and [OpenFermion](https://github.com/quantumlib/OpenFermion) is used to construct target gate unitaries.


## Using the scripts

To use the scripts, simply set the hyperparameters - either by modifying the default hyperparameters in the file itself, or passing the relevant command line arguments - and then run the script using Python 3:

```bash
	python3 state_learner.py
```

```bash
	python3 gate_synthesis.py
```

The results of the simulations will be saved in the directory `out_dir/simulation_ID`, with `out_dir` set by the hyperparameter dictionary, and `simulation_ID` determined automatically based on the simulation name.

### State learner hyperparameters

The following hyperparameters can be set for the script `state_learner.py`:


|   Hyperparameter  | Command line argument |                                                                                                                                                                     Description                                                                                                                                                                     |
| ----------------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`            | `-n`/`--name`         | The name of the simulation                                                                                                                                                                                                                                                                                                                          |
| `out_dir`         | `-o`/`--out-dir`      | Output directory for saving the simulation results                                                                                                                                                                                                                                                                                                  |
| `target_state_fn` |                       | Function for generating the target state for optimization. This function can accept an optional list of gate parameters, along with the required keyword argument `cutoff` which determines Fock basis truncation. The function must return a NumPy array of length `[cutoff]` for single mode states, and length `[cutoff^2]` for two mode states. |
|                   |                       |                                                                                                                                                                                                                                                                                                                                                     |


## Authors

Juan Miguel Arrazola, Thomas R. Bromley, Josh Izaac, Casey R. Myers, Kamil Brádler, and Nathan Killoran.

If you are doing any research using this source code and Strawberry Fields, please cite the following two papers:

> Nathan Killoran, Josh Izaac, Nicolás Quesada, Ville Bergholm, Matthew Amy, and Christian Weedbrook. Strawberry Fields: A Software Platform for Photonic Quantum Computing. arXiv, 2018. arXiv:1804.03159

## License

This source code is free and open source, released under the Apache License, Version 2.0.
