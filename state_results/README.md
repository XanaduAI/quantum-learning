# Learnt state data

This folder contains the saved NumPy npz files containing hyperparameters, learnt circuit parameters, and other properties and data for the learnt states presented in [arXiv:1807.10781](https://arxiv.org/abs/1807.10781). These states include 4 single mode states,

* Single photon state

* ON state with a=1, N=9

* Hex GKP state with mu=1, delta=0.3

* Random state

and one two mode state,

* The NOON state with N=5.

To access the saved data, the file can be loaded using NumPy:

```python
results = np.load('Single_photon_state.npz')
```

The individual hyperparameters and results can then be accessed via the respective key. For example, to extract the learnt state, as well as a list of the variational circuit layer squeezing magnitudes:

```python
learnt_state = results['ket']
squeezing = results['sq_r']
```

### Available keys

For a list of all available keys, see the table below.


|       Keys      |   Value type   |                                        Description                                        |
|-----------------|----------------|-------------------------------------------------------------------------------------------|
| `cutoff`        | integer        | The simulation Fock basis truncation.                                                     |
| `depth`         | integer        | Number of layers in the variational quantum circuit.                                      |
| `reps`          | integer        | Number of optimization steps to performed.                                                |
| `cost`          | float          | Minimum value of the cost achieved during the optimization.                               |
| `fidelity`      | float          | Maximum value of the fidelity achieved during the optimization.                           |
| `cost_function` | array[float]   | Value of the cost function for each optimization step.                                    |
| `sq_r`          | array[float]   | Squeezing magnitude of each layer in the variational quantum circuit.                     |
| `sq_phi`        | array[float]   | Squeezing phase of each layer in the variational quantum circuit.                         |
| `disp_r`        | array[float]   | Displacement magnitude of each layer in the variational quantum circuit.                  |
| `disp_phi`      | array[float]   | Displacement phase of each layer in the variational quantum circuit.                      |
| `r`             | array[float]   | Phase space rotation of leach layer.                                                      |
| `kappa`         | array[float]   | Non-linear Kerr interaction strength of each layer.                                       |
| `ket`           | array[complex] | Variational quantum circuit output/learnt state vector corresponding to maximum fidelity. |



## Authors

Juan Miguel Arrazola, Thomas R. Bromley, Josh Izaac, Casey R. Myers, Kamil Brádler, and Nathan Killoran.

If you are doing any research using this source code and Strawberry Fields, please cite the following two papers:

> Juan Miguel Arrazola, Thomas R. Bromley, Josh Izaac, Casey R. Myers, Kamil Brádler, and Nathan Killoran. Machine learning method for state preparation and gate synthesis on photonic quantum computers. arXiv, 2018. [arXiv:1807.10781](https://arxiv.org/abs/1807.10781)

> Nathan Killoran, Josh Izaac, Nicolás Quesada, Ville Bergholm, Matthew Amy, and Christian Weedbrook. Strawberry Fields: A Software Platform for Photonic Quantum Computing. arXiv, 2018. [arXiv:1804.03159](https://arxiv.org/abs/1804.03159)

## License

This source code is free and open source, released under the Apache License, Version 2.0.
