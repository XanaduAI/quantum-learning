# Quantum state learning and gate synthesis

<img align="left" width="100" height="100" src="https://github.com/XanaduAI/quantum-learning/blob/master/static/photon.gif">

This repository contains the source code used to produce the results presented in *"Machine learning method for state preparation and gate synthesis on photonic
quantum computers"*.

It includes the following content:

* `state_learner.py`: a Python script to automate quantum state learning using continuous-variable (CV) variational quantum circuits. Simply specify your one- or two-mode target state, along with other hyperparameters, and this script automatically constructs and optimizes the variational quantum circuit.

* `gate_synthesis.py`: a Python script to automate quantum gate synthesis using continuous-variable (CV) variational quantum circuits. Simply specify your one- or two-mode target unitary, along with other hyperparameters, and this script automatically constructs and optimizes the variational quantum circuit.

* Jupyter notebooks: two Jupyter notebooks are also provided, `StateLearning.ipynb` and `GateSynthesis.ipynb`, walking through the process of state learning and gate synthesis respectively.

## Requirements

To construct and optimize the variational quantum circuits, these scripts and notebooks use the TensorFlow backend of [Strawberry Fields](https://github.com/XanaduAI/strawberryfields). In addition, matplotlib is required for generating output plots, and [OpenFermion](https://github.com/quantumlib/OpenFermion) is used to construct target gate unitaries.


## Using the scripts



## Authors

Juan Miguel Arrazola, Thomas R. Bromley, Josh Izaac, Casey R. Myers, Kamil Brádler, and Nathan Killoran.

If you are doing any research using this source code and Strawberry Fields, please cite the following two papers:

> Nathan Killoran, Josh Izaac, Nicolás Quesada, Ville Bergholm, Matthew Amy, and Christian Weedbrook. Strawberry Fields: A Software Platform for Photonic Quantum Computing. arXiv, 2018. arXiv:1804.03159

## License

This source code is free and open source, released under the Apache License, Version 2.0.
