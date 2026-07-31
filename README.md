# AutoGrad

A tiny **single-header automatic differentiation engine** written in C. (499 semi-colons)

AutoGrad implements a reusable **static computation graph** supporting:

* Forward propagation
* Reverse-mode automatic differentiation (backpropagation)
* Topological sorting of computation graphs
* Gradient descent using a precomputed topological ordering

Unlike dynamic autograd systems, the computation graph is **constructed once** and reused throughout training, avoiding repeated graph allocation and improving execution efficiency.

---

## Features

* Single-header library (`autograd.h`)
* Static computation graph
* Forward and backward propagation
* Reverse-mode automatic differentiation
* Topological graph traversal
* Gradient descent
* Neural network primitives
  * `Neuron`
  * `Layer`
  * `MLP`
* Model serialization/deserialization
* MNIST dataset parser
* Minimal dependencies (standard C library)

---

## Supported Operations

Current scalar operations include:

* Addition (`+`)
* Subtraction (`-`)
* Multiplication (`*`)
* Summation (`Σ`)
* Hyperbolic tangent (`tanh`)

Additional mathematical operations can be added by defining the corresponding forward and backward functions.

---

# Getting Started

Include the header wherever the API is required:

```c
#include "autograd.h"
```

In **exactly one** source file:

```c
#define AUTOGRAD_IMPLEMENTATION
#include "autograd.h"
```

Compile normally:

```bash
gcc train.c parser.c -lm -o train
gcc test.c parser.c -lm -o test
```

---

# Project Structure

```text
.
├── autograd.h          # Single-header autograd library
├── parser.c
├── parser.h            # MNIST parser
├── train.c             # Training program
├── test.c              # Model evaluation
├── model.agm           # Saved model
├── FORMAT.txt          # Format description of agm files
├── dataset/
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
└── README.md
```

---

# Core Idea

Unlike the previous autograd implementation in [CMNIST](https://github.com/Beginner10617/CMNIST.git), this engine constructs the graph only once.

AutoGrad instead separates:

1. Graph construction
2. Graph execution

The graph is allocated once and can then be reused for:

* Forward propagation
* Backpropagation
* Gradient descent
* Updating input values

No graph reconstruction is required between iterations.

---

# Forward Pass

Each `Value` node stores:

* Scalar value (`data`)
* Gradient (`grad`)
* References to parent nodes
* Forward function
* Backward function

Forward execution is performed in topological order.

The ordering only needs to be computed once:

```c
ValueList topo;
topoSortList(output, &topo);
```

The same ordering can then be reused throughout training.

---

# Backward Pass

Gradients are computed using reverse-mode automatic differentiation.

```c
output->grad = 1;
```

Then iterate in reverse topological order:

```text
for node in reverse(topological_order):
    node->_backward(node)
```

Each operation propagates gradients to its parent nodes according to the chain rule.

---

# Neural Network API

The library provides lightweight neural-network abstractions built directly on top of the autograd engine:

* `Neuron`
* `Layer`
* `MLP`

These components support:

* Forward inference
* Backpropagation
* Gradient descent
* Parameter serialization

---

# MNIST Example

The repository includes a complete handwritten digit classification example.

Components include:

* Binary MNIST dataset parser
* MLP training program (`train.c`)
* Model serialization (`model.agm`)
* Model evaluation (`test.c`)

Example evaluation output:

```text
loading dataset header
-------------------------
magic number: 2051
number of images: 10000
number of rows: 28
number of columns: 28
-------------------------
-------------------------
magic number: 2049
number of images: 10000
-------------------------
setting up tree
loading dataset
8997 correct out of 10000 cases 89.970% accuracy
```

---

# Model Format

Trained models are stored in the binary `.agm` format.

The serialized model contains:

* Network architecture
* Activation functions
* Bias parameters
* Weight parameters

This allows trained networks to be loaded without retraining.

---

# Design

Each `Value` stores only the information necessary for automatic differentiation:

* Current scalar value
* Gradient
* Parent nodes
* Forward callback
* Backward callback

Graphs are constructed manually using helper functions such as:

```c
setAdd(out, x, y);
```

Because graph topology never changes, allocations occur only during graph construction.

---

# Future Work

Possible extensions include:

* Division
* Exponential
* Logarithm
* Power
* ReLU
* Sigmoid
* Softmax
* Additional optimizers
  * Momentum
  * RMSProp
  * Adam
* Vector and tensor operations
* SIMD optimizations
* Batch training support

---

# Inspiration

While building a handwritten digit classifier in C in
[CMNIST](https://github.com/Beginner10617/CMNIST.git),
I noticed that rebuilding the computation graph every forward pass resulted in repeated memory allocation and deallocation.

This project explores a reusable static graph design where the computation graph is constructed once and then reused across all 
