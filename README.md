# Iliad

Pure NumPy / SciPy implementation of samplers based on Hamiltonian mechanics. Includes implementations of the following methods:

* Hamiltonian Monte Carlo from Neal, R. (2011), Handbook of Markov Chain Monte Carlo. Chapman and Hall/CRC.
* Riemannian Manifold Hamiltonian Monte Carlo from Girolami, M. and Calderhead, B. (2011), Riemann manifold Langevin and Hamiltonian Monte Carlo methods. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 73: 123-214.
* Lagrangian Monte Carlo from Lan S., Stathopoulos V., Shahbaba B., and Girolami M. (2015), Markov Chain Monte Carlo from Lagrangian Dynamics. J Comput Graph Stat: 24(2):357-378. 

In order to install this software, you should execute `pip install -e .` from the directory containing this README. Additionally, you will require the [Odyssey](https://github.com/JamesBrofos/Odyssey).

This code contains significant components that are identical to, or adapted from, a previous implementation of this software released at [Evaluating the Implicit Midpoint Integrator](https://github.com/JamesBrofos/Evaluating-the-Implicit-Midpoint-Integrator) and [Thresholds in Hamiltonian Monte Carlo](https://github.com/JamesBrofos/Thresholds-in-Hamiltonian-Monte-Carlo), which were provided under an MIT License.
