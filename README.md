# QCircPy

QCircPy is a Quantum Computer Simulation and Benchmarking package on Python with GPU and CPU flexibility and performance. It allows the user to benchmark and run simple Quantum Circuits on a GPU or CPU with `numpy` and `cupy`.

This project is for educational purposes.




## Installation

### Prerequisites

QCircPy requires `numpy`, which is installed automatically, and `cupy`, which the user should install manually based on their CUDA version.

For users using CUDA 11.x:

```cmd
pip install cupy-cuda11x
```

or:

```cmd
py -m pip install cupy-cuda11x
```

For users using CUDA 12.x:

```cmd
pip install cupy-cuda12x
```

or:

```cmd
py - m pip install cupy-cuda12x
```

Then, finally:

### Installation

```cmd
pip install qcircpy
```

or:

```cmd
py -m pip install qcircpy
```



## Usage

It is recommended to use the `engine` subpackage as an interface to QCircPy's functionalities. 

```py
import qcircpy.engine as qp
```

Detailed usage can be found in [USAGE.md](USAGE.md)
