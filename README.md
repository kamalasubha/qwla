# QWLA: Quantum computing without the linear algebra

<div align="center">
  <img src="https://github.com/user-attachments/assets/5b65a96e-4fcc-4af9-b580-a09c509a38cc" width="300" />
</div>
This is a tiny quantum circuit simulator written in Python. 
It represents a quantum state as a dictionary and applies operations in a functional style, using map, filter, reduce, etc.

## technical details
For all technical details and an introduction to quantum computing without the linear algebra, see the accompanying paper: TBD

## structure
- The `state.py` file contains the `State` class, which is a quantum state and supports a number of operations (Clifford + T).
- There are example circuits in the `examples/` folder.


## usage
The easiest way to use QWLA is with [uv](https://docs.astral.sh/uv/getting-started/installation/).
To run an example, simply:

```bash
uv run examples/epr.py
```




