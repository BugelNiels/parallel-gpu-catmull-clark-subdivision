<br />
<p align="center">
  <h1 align="center">Parallel CPU Catmull-Clark Subdivision</h1>

  <p align="center">
    A fast GPU-based Catmull-Clark subdivision algorithm using CUDA
  </p>
</p>

## About The Project

This project is a CUDA implementation of the method described in the paper: [A Halfedge Refinement Rule for Parallel Catmull-Clark Subdivision](https://onlinelibrary.wiley.com/doi/10.1111/cgf.14381?af=R).
It also has a number of optimizations to accelerate it even further. Supports quad meshes, non-quad meshes and meshes with boundaries.

The project works by reading `.obj` files from the `models` directory. You can optionally indicate a location of where to write the result. This will be an `.obj` file in the `results` directory. 

## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

You need to the following to be able to compile and run the project

* [Make](https://www.gnu.org/software/make/) ([Windows Version](http://gnuwin32.sourceforge.net/packages/make.htm))
* [CUDA](https://developer.nvidia.com/cuda-toolkit)

### Setup

To set up the program, run the following commands:
```sh
    git clone git@github.com:BugelNiels/cuda-subdivision.git
    cd cuda-subdivision
    make
```
This should work on both Linux and Windows

### Running

You can run the project as follows:

- **Linux**:

    ```sh
    ./subdivide <subdivision level> <obj file> <optional: result obj file>
    ```
    Example:     
    ```sh
    ./subdivide 4 bigguy bigguy_result
    ```
    Will subdivide the mesh `models/bigguy.obj` 4 times and write the result to `results/bigguy_result.obj`

- **Windows**:

    ```sh
    subdivide.exe <subdivision level> <obj file> <optional: result obj file>
    ```
    Example:     
    ```sh
    subdivide.exe 4 bigguy bigguy_result
    ```
    Will subdivide the mesh `models/bigguy.obj` 4 times and write the result to `results/bigguy_result.obj`

The `benchmark.sh` script can be used to benchmark. Note that this script is for linux terminals only (but can be adjusted to work for windows as well). The timings in milliseconds will be written to a file called `timings.txt`.