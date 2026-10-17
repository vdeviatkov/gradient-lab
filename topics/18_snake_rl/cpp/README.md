# Snake reinforcement learning — C++

[Topic and results](../README.md) · [Idea](../idea.md) · [C++ setup and library boundaries](../../../docs/implementations/cpp.md)

## Implementation

Canonical algorithm source:

- [reinforcement.cpp](reinforcement.cpp)
- [snake.cpp](snake.cpp)

Public interfaces:

- [reinforcement.hpp](include/ml_scratch/reinforcement.hpp)
- [snake.hpp](include/ml_scratch/snake.hpp)

Topic-specific support:

- [reinforcement_checkpoint.cpp](utils/reinforcement_checkpoint.cpp)
- [snake_render.cpp](utils/snake_render.cpp)

Reuses canonical implementations from [Backpropagation and gradient checking](../../07_backpropagation/cpp/README.md), [Optimizers](../../10_optimizers/cpp/README.md).

The [experiment driver](main.cpp) owns this topic’s data, configuration, evaluation, and reporting.

## Run

From the repository root after the shared C++ build:

```bash
./build/cpp_snake_rl
```

Dataset requirements and measured results are documented in the [topic README](../README.md).

## Tests

- [reinforcement_test.cpp](tests/reinforcement_test.cpp)
- [snake_test.cpp](tests/snake_test.cpp)

These tests are registered with CTest by the local [CMakeLists.txt](CMakeLists.txt).
