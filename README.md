# AutoGrad

A tiny automatic differentiation engine written in C.

This project implements a reusable computational graph for:

- Forward propagation
- Reverse-mode automatic differentiation (backpropagation)
- Topological sorting of a computation graph
- Gradient descent using a precomputed topological order

Currently supported operations:

- Addition (`+`)
- Subtraction (`-`)
- Multiplication (`*`)
- Summation (`Σ`)
- Hyperbolic tangent (`tanh`)

In addition, basic neural network components have been implemented:

- Neuron
- Layer
- Multi-Layer Perceptron (MLP)

These components are implemented but not yet thoroughly tested.

---

## Build

Compile with:

```bash
gcc main.c autoGrad.c -o output
```

Run:

```bash
./output
```

---

## Core Idea

Unlike the previous autograd implementation in
[CMNIST](https://github.com/Beginner10617/CMNIST.git),
which dynamically rebuilt the computation graph during every forward pass,
this engine only requires constructing the graph once.

After construction, the same graph can be reused for:

- Multiple forward passes
- Multiple backward passes
- Updating input values without reallocating graph nodes

This makes the engine behave more like a static computation graph framework.

---

## Example

A simple toy example of the computation graph below is implemented in `main.c`.
The output has been verified against expected mathematical values.

<div align="center">
<img src="example-1.jpg" alt="Example computation graph" width="400" height="167">
</div>

---

## Forward Pass

Each node stores:

- Its current scalar value
- Gradient value
- References to previous nodes
- The operation used to compute it

Calling:

```c
node->_forward(node);
```

computes the node's value using its dependencies.

Forward execution should follow a topological ordering of the graph.

To achieve this, a `ValueList` can be computed once using `topoSortList()` after graph construction.

The resulting order can then be reused during training:

```text
forward()
backward()
gradientDescent()
```

Unlike the previous implementation, these operations do not allocate new graph memory.

---

## Backward Pass

Gradients are propagated using reverse-mode autodiff.

To begin backpropagation, initialize the output gradient:

```c
C->grad = 1;
```

Then execute backward functions in reverse topological order:

```c
C->_backward(C);
B->_backward(B);
A->_backward(A);
```

Each operation accumulates gradients into its parent nodes.

---

## Neural Network Components

The project includes basic neural network abstractions:

- `Neuron`
- `Layer`
- `MLP`

These structures are implemented on top of the autograd engine but are still undergoing testing.

The intended validation target is the MNIST handwritten digit dataset.

---

## MNIST Dataset Support

A parser for the binary MNIST dataset files has been added.

The plan is to use this parser to train and validate the implemented MLP architecture.

---

## Design

Each `Value` node contains:

- Scalar value (`data`)
- Gradient (`grad`)
- References to parent nodes (`_prev`)
- Forward function pointer (`_forward`)
- Backward function pointer (`_backward`)

Graphs are manually constructed using helper functions such as:

```c
setAdd(out, x, y);
```

This allows graph structure reuse without repeated allocation.

---

## Possible Improvements

- Additional mathematical operations
  - Exponentials
  - `ReLU`
  - Additional activation functions
- Better testing coverage
- Training and evaluating the neural network on MNIST
- Mini-batch support and optimizers

---

## Inspiration

While building a digit classifier in C in
[CMNIST](https://github.com/Beginner10617/CMNIST.git),
I noticed repeated memory allocation and deallocation caused by rebuilding the computation graph every forward pass.

This project explores a reusable graph design where the computation graph is constructed once and reused across iterations to avoid that overhead.
