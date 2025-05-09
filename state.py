from math import sqrt, pi
from cmath import exp
from functional import seq
from bitarray import frozenbitarray as bitarray
from bitarray import bitarray as mut_bitarray
import random

# Complex number i for quantum operations
img = 1j

# Helper functions for bit manipulation
def set_bit(x: bitarray, i: int, v: int) -> bitarray:
    """Set the i-th bit of bitarray x to value v (0 or 1)"""
    new = mut_bitarray(x)
    new[i] = v
    return bitarray(new)

def flip(x: bitarray, i: int) -> bitarray:
    """Flip (negate) the i-th bit of bitarray x"""
    mask = bitarray("0" * i + "1" + "0" * (len(x) - i - 1))
    return x ^ mask

class State:
    """
    Quantum state using a dictionary-like structure mapping bitstrings to amplitudes.
    """
    def __init__(self, n_qubits: int, m_bits: int = 0):
        """
        Initialize a quantum state with n_qubits qubits and m_bits classical bits.
        
        Args:
            n_qubits: Number of qubits in the system
            m_bits: Number of classical bits for measurement results
        
        The state starts in 0...0 (ground state).
        """
        self.n_qubits = n_qubits
        self.m_bits = m_bits
        self.state = seq(
            [
                (bitarray(format(i, f"0{n_qubits}b")), 1.0 if i == 0 else 0.0)
                for i in range(2**n_qubits)
            ]
        )
        self.cbits = [0] * m_bits

    def x(self, j: int):
        """
        Apply the NOT gate to the j-th qubit.
        
        This gate flips the basis states where qubit j is present.
        """
        self.state = self.state.smap(lambda b, a: (flip(b, j), a))
        return self

    def cx(self, ctrl: int, trgt: int):
        """
        Apply the CX (controlled-NOT) gate with control qubit ctrl and target qubit trgt.
        """
        self.state = self.state.smap(
            lambda b, a: (b if not b[ctrl] else flip(b, trgt), a)
        )
        return self

    def s(self, j: int):
        """
        Apply the S (phase) gate to the j-th qubit.
        """
        self.state = self.state.smap(lambda b, a: (b, (img ** b[j]) * a))
        return self

    def t(self, j: int):
        """
        Apply the T gate to the j-th qubit.
        """
        phase = exp(img * pi / 4)
        self.state = self.state.smap(lambda b, a: (b, (phase ** b[j]) * a))
        return self
    
    def h(self, j: int):
        """
        Apply the Hadamard gate to the j-th qubit.
        """
        self.state = (
            self.state.smap(
                lambda b, a: [
                    (set_bit(b, j, 0), a / sqrt(2)),
                    (set_bit(b, j, 1), ((1 - 2 * b[j]) * a / sqrt(2))),
                ]
            )
            .flatten()
            .reduce_by_key(lambda x, y: x + y)
        )
        return self
    
    def measure(self, j: int, cbit: int = None):
        """
        Measure the j-th qubit.
        
        Args:
            j: Index of qubit to measure
            cbit: Optional classical bit to store the measurement result
            
        Returns:
            The state after measurement (collapsed)
        """
        # compute the probability of 0
        prob_0 = (
            self.state.filter(lambda s: not s[0][j])
            .smap(lambda b, a: a.conjugate() * a)
            .sum()
        )
        
        measurement = random.random() < prob_0
        
        if cbit is not None:
            self.cbits[cbit] = measurement
            
        if random.random() < prob_0:
            # Collapse to 0 state
            self.state = self.state.smap(lambda b, a: (b, a * (not b[j]))).smap(
                lambda b, a: (b, a / sqrt(prob_0))
            )
        else:
            # Collapse to 1 state
            self.state = self.state.smap(lambda b, a: (b, a * b[j])).smap(
                lambda b, a: (b, a / sqrt(1.0 - prob_0))
            )
        
        if cbit is not None:
            self.cbits[cbit] = 0 if random.random() < prob_0 else 1
        return self
    
    def __str__(self):
        """
        Return a string representation of the quantum state.
        
        Format: Each basis state with its corresponding amplitude.
        """
        # sort the list by basis state, starting at 00
        self.state = self.state.sorted(key=lambda x: x[0].to01())
 
        return "\n".join([f"{b.to01()}: {a}" for b, a in self.state])
