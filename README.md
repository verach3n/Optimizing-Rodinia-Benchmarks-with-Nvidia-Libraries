# DD2360HT22Project
## Group 7 - Optimizing Rodinia Benchmarks with Nvidia Libraries
This project aims to improve the performance of the [Rodinia benchmarks](https://www.cs.virginia.edu/rodinia/doku.php) by using [Nvidia libraries](https://www.cs.virginia.edu/rodinia/doku.php) such as cuSOLVER and cuBLAS.

## Table of Contents
- Installation
- Usage
- File Structure

## Installation
To use this project, you will need to install the following dependencies:
- Nvidia CUDA Toolkit: Follow the instructions [here](https://developer.nvidia.com/cuda-downloads) to install CUDA on your system.
- cuBLAS library: Follow the instructions [here](https://developer.nvidia.com/cublas) to install cuBLAS.
- cuSOLVER library: Follow the instructions [here](https://developer.nvidia.com/cusolver) to install cuSOLVER.

You will also need a GPU with CUDA support.

## Usage
To use the optimized benchmarks:
1. Run Gaussian with the desired benchmark and input parameters `./<filename> -s <matrix-dimension>`

For example:
`./gaussian -s 1024`


2. Run lud with the desired benchmark and input parameters. `./<filename> -s <matrix-dimension>`

For example:
`./lud -s 20000`.

3. Run combination with the desired benchmark and input parameters. `./<filename> -s <matrix-dimension>`

For example:
`./cusolver -s 20000`.

Also in each folder, there is a corresponding `run.sh` file that is used to profile the running process for a series of inputs. You can run it using the `./run.sh` command.

## File Structure

This project consists of the following files and directories:

- cuBLAS: This directory contains the source code for the project.  
  - gaussian:  
    - common: This directory required to run the executables.   
    - managedm: This directory contains implementation of gaussian using managed memory in `gaussian_manegedm.cu`.  
    - pinnedm: This directory contains implementation of gaussian using pinned memory in `gaussian_pinnedm.cu`.    
    - regular: This directory contains implementation of gaussian using regular memory in `gaussian.cu`.   
    - mdt3_output.txt: This file contains the processed data of the profiling of `gaussian_manegedm.cu`.
    - pdt3_output.txt: This file contains the processed data of the profiling of `gaussian_pinnedm.cu`.
    - rdt3_output.txt: This file contains the processed data of the profiling of `gaussian.cu`.
  - lud/cuda: 
    - lud_kernel.cu: This file contains kernel function.  
    - lud.cu: This file contains main function.  
 - cuSOLVER: This file contains the main function of the program.  
    - getrf: `cusolver_getrf_example.cu` is the cu file using cuSOLVER library to implement both gaussian and lud algorithm.
