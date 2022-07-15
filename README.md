# MicroTOuNN
## A Generalized Framework for Microstructural Optimization using Neural Networks
[Arxiv](https://arxiv.org/abs/2207.06512)

Saketh Sridhara, Aaditya Chandrasekhar and Krishnan Suresh \
[Engineering Representations and Simulation Lab](https://ersl.wisc.edu)
University of Wisconsin-Madison, Madison, WI, USA

### Abstract

Microstructures, i.e., architected materials, are designed today, typically, by maximizing an objective, such as bulk modulus, subject to a volume constraint. However, in many applications, it is often more appropriate to impose constraints on other physical quantities of interest.

In this paper, we consider such generalized microstructural optimization problems where any of the microstructural quantities, namely, bulk, shear, Poisson ratio, or volume, can serve as the objective, while the remaining can serve as constraints. In particular, we propose here a neural-network (NN)  framework to solve such problems. The framework relies on the classic density formulation of microstructural optimization, but the density field is represented through the NN's weights and biases.

The main characteristics of the proposed NN framework are: (1) it supports automatic differentiation, eliminating the need for manual sensitivity derivations, (2) smoothing filters  are not required due to implicit filtering, (3) the framework can be easily extended to multiple-materials, and (4) a high-resolution microstructural topology can be recovered through a simple post-processing step. The framework is illustrated through a variety of microstructural optimization problems.

### Requirements

1.  Numpy
2.  Pytorch
3.  Matplotlib
4.  [Torch_sparse_solve](https://github.com/flaport/torch_sparse_solve) for faster solve 
