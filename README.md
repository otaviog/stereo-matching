# Stereo Matching Experiment

I found PyTorch an excellent tool to protoype CUDA-based kernels for fast implementation of computer vision algorithms.
In this repository I revamp my old stereo matching research with CUDA and rewriten them using modern CUDA along with Pytorch.

**Methods**

* Cost measurements:
  * Sum of squared distances
  * Birchfield
* Aggregation mechanisms:
  * Winner takes all
  * Dynamic Programming
  * Semiglobal matching

**Note for AI practioners**: The operations aren't differentiable.

## Getting Started

**Requirements**:

* Cuda toolkit installed
* Pytorch

```shell
$ pip install .
```


## Benchmarking

