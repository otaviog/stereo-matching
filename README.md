# Stereo Matching Experiment

I found PyTorch an excellent tool to prototype CUDA-based kernels for experimenting with computer vision algorithms.
I revamp my old stereo matching code with CUDA in this repository and rewrite them using modern CUDA and Pytorch.

Here's a video from the old implementation:

[![Stereo matching](http://img.youtube.com/vi/EuIyLVVpwGs/0.jpg)](http://www.youtube.com/watch?v=EuIyLVVpwGs "Stereo matching sample old")


**Methods**

* Cost measurements:
  * Sum of squared distances (SSD)
  * Birchfield
* Aggregation mechanisms:
  * Semiglobal matching
* Disparity reducer
  * Winner takes all
  * Dynamic Programming

**Note for AI practitioners**: The operations aren't differentiable.

## Getting Started

TBD

**Requirements**:

* Cuda toolkit installed
* Pytorch

```shell
$ pip install .
```


## Benchmarking

TBD: Benchmarks

## Optimization Log

TBD: Add a log of optimization attempts to test what works and what not work.


