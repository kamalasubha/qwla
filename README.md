# QWLA: Quantum computing without the linear algebra

<div align="center">
  <img src="https://github.com/user-attachments/assets/5b65a96e-4fcc-4af9-b580-a09c509a38cc" width="300" />
</div>
This is a tiny quantum circuit simulator.
The simulator is written in Python. It represents the state as a dictionary and applies operations in a functional style, using map, filter, reduce, etc.

## Usage

The `state.py` file contains the `State` class, which is a quantum state and supports a number of operations.

There are also some example circuits in the root directory.

The easiest way to use this with `uv`.
To run an example, simply:

```bash
uv run example_epr.py
```




