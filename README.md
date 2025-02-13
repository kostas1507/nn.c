# nn.c

This is a minimal implementation of a neural network in C that should shed some light on how neural networks learn.

It aims to be as transparent as possible by using only C with no external functions.

The network uses the softsign activation function and mean squared error (MSE) loss function (to avoid pow()).

It has a fixed architecture: two inputs, one hidden layer with two neurons, and one output. It can learn logic gates.

Because everything is static, gradients can be pre-computed, so backpropagation does not need to be implemented.

**Note:** This project will not be very helpful if you have no idea how neural networks work. For that you should checkout [Karpathy's video](https://www.youtube.com/watch?v=VMj-3S1tku0&t=6918s).

