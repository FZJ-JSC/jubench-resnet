# JUPITER Benchmark Suite: ResNet

[![DOI](https://zenodo.org/badge/831446381.svg)](https://zenodo.org/badge/latestdoi/831446381) [![Static Badge](https://img.shields.io/badge/DOI%20(Suite)-10.5281%2Fzenodo.12737073-blue)](https://zenodo.org/badge/latestdoi/764615316)

This benchmark is part of the [JUPITER Benchmark Suite](https://github.com/FZJ-JSC/jubench). See the repository of the suite for some general remarks.

This repository contains the ResNet benchmark. [`DESCRIPTION.md`](DESCRIPTION.md) contains details for compilation, execution, and evaluation.

The benchmark is included from https://github.com/HelmholtzAI-FZJ/tf_cnn_benchmarks (a fork of https://github.com/tensorflow/benchmarks). It uses the `main` branch, compatible with Tensorflow 2 and Horovod.

The source code of the ResNet benchmark is included in the `./src/` subdirectory as a submodule from the upstream repository at [github.com/HelmholtzAI-FZJ/tf_cnn_benchmarks](https://github.com/HelmholtzAI-FZJ/tf_cnn_benchmarks).