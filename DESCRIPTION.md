# TensorFlow ResNet50-ImageNet Benchmark

## Purpose

This benchmark serves to measure the performance of a system on one of the most important Machine Learning (ML) benchmark tasks: The training of a ResNet50 model on the ImageNet dataset. The key metric, is the throughput, that is the number of training samples that can be used per second to perform the training. It is based on the ML framework TensorFlow.

The benchmark is based on the implementation `tf_cnn_benchmark`. This is a TensorFlow benchmark containing implementations of several popular convolutional models, and is designed to be as fast as possible. `tf_cnn_benchmark`s supports both running on a single machine or running in distributed mode across multiple hosts. Even though it is considered deprecated at the time of writing, it is still widely applied.

_While this description tries to be agnostic with respect to the benchmarking infrastructure, we consider JUBE as our reference and give examples with it._

## Source

Archive Name: `tf_resnet-bench.tar.gz`

The file holds instructions to run the benchmark, corresponding JUBE scripts, and
configuration files to run the tf_resnet_benchmark. The sources
for the benchmark are included in the `src` directory which are equivalent to commit `6099d0d` ofhttps://github.com/HelmholtzAI-FZJ/tf_cnn_benchmarks.git.

## Building

`tf_cnn_benchmark` is a Python package that does not need to be compiled. It depends on  TensorFlow, which can run on CPUs, GPUs, and other accelerators. The benchmark is intended for the GPU version only.

The benchmark depends on some packages: It needs Python3, a working [TensorFlow](https://www.tensorflow.org) 2.x and its dependencies. It also depends on [Horovod](https://horovod.readthedocs.io) for multi-node parallelization. 

### Software Environment

The execution was tested with Tensorflow 2.6.0 and Horovod 0.24.3. In the reference environment, the software environment, is set up is by loading the software modules for Tensorflow and Horovod.
```bash
module load GCC OpenMPI TensorFlow Horovod
```

Submissions with any versions of the used libraries as long as they are freely available in public repositories, is acceptable. The version / commit hash need to be specified. If the version used is not available in the main TensorFlow and Horovod repository, a clear indication of a plan to and a track record of upstreaming changes of the fork needs to be recognizable.

Environment variables that control the execution such as `HOROVOD_ENABLE_XLA_OPS` can be specified at will.  Optimisations that potentially change the result must not be used.

## Execution

Before going into the execution details, we briefly comment on the scale.

### Parameters

The code will measure the throughput emulating the training with a gradient-based method such a stochastic gradient descent. The hyperparameter batch size affects both the efficiency of the execution and the convergence of the algorithm. For small batch sizes, the efficiency will be low, as communication and overhead take a larger fraction of the execution time. For large batch sizes, convergence is slowed down, and for even larger batch sizes, the algorithm will not reach as good optima as for small batch sizes. In the literature, training this system has been achieved successfully with batch sizes of up to 32k. As our reference execution size is not ideal for this number, the global batch size is limited to a slightly smaller number: 20480. This batch size shall not be exceeded. The code accepts as input the _local batch size_, that is the number of samples that is executed per GPU. The global batch size is obtained by multiplying this number with the number of GPUs used. 

### Command Line

The code can be run from the command line like the following:

```bash
$RUN_PREFIX python /path/to/tf_cnn_benchmarks.py --model resnet50_v2 --batch_size 512 -variable_update horovod --use_fp16=True --xla_compile=True
```
In this setup, the code uses random data as input. The code also supports using a processed version of Imagenet as input. The location is then specified with the parameter `--data_dir` but this is **not required** for the submission of benchmark results.  

### Accuracy

The numerical accuracy of the floating point operations must not be reduced compared to the baseline. 

### JUBE

The JUBE step `run` calls the aforementioned command line with the correct modules. It also cares about the MPI distribution by submitting a script to the batch system. The latter is achieved by populating a batch submission script template (via `platform.xml`) with information specified in the top of the script relating to the number of nodes and tasks per node. Via dependencies, the JUBE step `run` calls the JUBE step `download` automatically.

To submit a self-contained benchmark run to the batch system, call `jube run benchmark/jube/default.yaml` (please adapt the script as needed). JUBE will generate the necessary configuration and files, and submit the benchmark to the batch engine.

The following parameters of the JUBE script might need to be adapted:
- `n_gpu`: GPUs/node
- `taskspernode`: must be equal to `n_gpu`
- `queue`: SLURM queue to use
- `modules`: to be sourced before building and running

After a run, JUBE can also be used to extract the runtime of the program (with
`jube analyse` and `jube result`).

The parameter `data_dir` controls the location of input data (instead of random data). This is **not required** for the submission.

## Verification

The application should run through successfully without any exceptions or error
codes generated. An overview table indicating successful operation is generated,
similar to the following:

```
Generating training model
Initializing graph
Running warm up
Done warm up
Step	Img/sec	total_loss
1	images/sec: 2222.7 +/- 0.0 (jitter = 0.0)	nan
10	images/sec: 2285.6 +/- 14.1 (jitter = 42.7)	nan
20	images/sec: 2304.9 +/- 14.0 (jitter = 30.2)	nan
30	images/sec: 2303.2 +/- 10.3 (jitter = 42.1)	nan
40	images/sec: 2292.3 +/- 8.8 (jitter = 39.7)	nan
50	images/sec: 2288.4 +/- 7.5 (jitter = 45.9)	nan
60	images/sec: 2296.2 +/- 8.0 (jitter = 50.9)	nan
70	images/sec: 2297.9 +/- 7.1 (jitter = 45.5)	nan
80	images/sec: 2299.9 +/- 7.1 (jitter = 48.6)	nan
90	images/sec: 2297.7 +/- 6.4 (jitter = 47.8)	nan
100	images/sec: 2296.2 +/- 6.1 (jitter = 47.8)	nan
----------------------------------------------------------------
total images/sec: 18356.70
----------------------------------------------------------------
```
Note that this table was generated with two nodes (eight GPUs) only. 

## Results

The output of "total images/sec" (throughput) is the most convenient metric in every day life and very useful to make comparisons. To homogenize with other benchmark, this is **not** the number compared. For the comparison the throughput (18356.70 in the example above) is converted into the **time a hypothetical training would require**. This conversion is done by assuming 90 epochs with 1,281,167 training samples of ImageNet, with the simple formula
```
[ time_to_report_in_seconds ]  = [epochs] * [images] / [throughput images/second]
```
Using the verification result above (images/sec: 18356.70), we obtain a duration of 90 * 1281167 / 18356.70 = 6281.36 seconds. This is the metric to be reported. 

### JUBE

Using `jube analyse` and a subsequent `jube result` prints an overview table
with the number of nodes, batch size, runtime and images/sec. The throughput metric is
shown as `imagespersec[s]`.

Using `jube analyse` and a subsequent `jube result` prints an overview table
with the number of nodes, tasks per node, and runtime.

## Baseline

On JUWELS Booster, the code was executed with JUBE, achieving a total throughput of 109163.45 images / second on 10 nodes.
For a submission, it is required that the degree of parallelization is chosen such that the result is faster than this baseline. 

The output of JUBE is given in this block. 
```
jube result -a run
result:
|   jobid |    systemname | error_code | runtime[s] | batch_size | nodes | imagespersec_avg | time_to_report_in_seconds |
|---------|---------------|------------|------------|------------|-------|------------------|---------------------------|
| 9480786 | juwelsbooster |          0 |     110.49 |        512 |    10 |        109163.45 |                   1056.27 |

```
The benchmark metric to be reported in this case is 1056.27 seconds.
