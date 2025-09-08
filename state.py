"""
Optimized quantum state simulator using functional programming techniques.
This module implements an efficient quantum state class that supports various quantum gates
and operations without requiring matrix algebra.
"""

from math import sqrt, pi, cos, sin
from cmath import exp
import random
from typing import Optional, Dict, Tuple
from functional import seq
from bitarray import frozenbitarray as bitarray
from bitarray import bitarray as mut_bitarray


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


def get_bit(x: bitarray, i: int) -> int:
    """Get the i-th bit of bitarray x"""
    return int(x[i])


class State:
    """
    Optimized quantum state using a dictionary-like structure mapping bitstrings to amplitudes.
    """

    def __init__(self, n_qubits: int, n_bits: int = 0):
        """
        Initialize a quantum state with n_qubits qubits and n_bits classical bits.

        Args:
            n_qubits: Number of qubits in the system
            n_bits: Number of classical bits for measurement results

        The state starts in |00...0⟩ (ground state).
        """
        assert n_qubits > 0 and n_bits >= 0

        self.n_qubits = n_qubits
        self.m_bits = n_bits
        
        # Initialize with only the ground state to save memory
        ground_state = bitarray(format(0, f"0{n_qubits}b"))
        self.state = seq([(ground_state, 1.0 + 0j)])
        self.cbits = [0] * n_bits
        
        # Tolerance for numerical precision
        self.tolerance = 1e-12

    def _clean_state(self):
        """Remove negligible amplitudes to maintain numerical stability"""
        self.state = self.state.filter(lambda entry: abs(entry[1]) > self.tolerance)
        return self

    def _normalize(self):
        """Normalize the quantum state"""
        norm_squared = sum(abs(amp) ** 2 for _, amp in self.state).real
        if norm_squared > 0:
            norm = sqrt(norm_squared)
            self.state = self.state.smap(lambda b, a: (b, a / norm))
        return self

    def x(self, j: int):
        """
        Apply the Pauli-X (NOT) gate to the j-th qubit.
        """
        # print(f"-> Applying X gate to qubit {j}")
        self.state = self.state.smap(lambda b, a: (flip(b, j), a))
        return self._clean_state()

    def y(self, j: int):
        """
        Apply the Pauli-Y gate to the j-th qubit.
        """
        # print(f"-> Applying Y gate to qubit {j}")
        self.state = self.state.smap(
            lambda b, a: (flip(b, j), a * (1j if not b[j] else -1j))
        )
        return self._clean_state()

    def z(self, j: int):
        """
        Apply the Pauli-Z gate to the j-th qubit.
        """
        # print(f"-> Applying Z gate to qubit {j}")
        self.state = self.state.smap(
            lambda b, a: (b, a * (-1 if b[j] else 1))
        )
        return self._clean_state()

    def cx(self, control: int, target: int):
        """
        Apply the CNOT gate with control and target qubits.
        """
        # print(f"-> Applying CNOT gate with control {control} and target {target}")
        self.state = self.state.smap(
            lambda b, a: (b if not b[control] else flip(b, target), a)
        )
        return self._clean_state()

    def cz(self, control: int, target: int):
        """
        Apply the controlled-Z gate.
        """
        # print(f"-> Applying CZ gate with control {control} and target {target}")
        self.state = self.state.smap(
            lambda b, a: (b, a * (-1 if (b[control] and b[target]) else 1))
        )
        return self._clean_state()

    def s(self, j: int):
        """
        Apply the S (phase) gate to the j-th qubit.
        """
        # print(f"-> Applying S gate to qubit {j}")
        self.state = self.state.smap(lambda b, a: (b, a * (1j if b[j] else 1)))
        return self._clean_state()

    def s_dagger(self, j: int):
        """
        Apply the S† gate to the j-th qubit.
        """
        # print(f"-> Applying S† gate to qubit {j}")
        self.state = self.state.smap(lambda b, a: (b, a * (-1j if b[j] else 1)))
        return self._clean_state()

    def t(self, j: int):
        """
        Apply the T gate to the j-th qubit.
        """
        # print(f"-> Applying T gate to qubit {j}")
        phase = exp(1j * pi / 4)
        self.state = self.state.smap(lambda b, a: (b, a * (phase if b[j] else 1)))
        return self._clean_state()

    def t_dagger(self, j: int):
        """
        Apply the T† gate to the j-th qubit.
        """
        # print(f"-> Applying T† gate to qubit {j}")
        phase = exp(-1j * pi / 4)
        self.state = self.state.smap(lambda b, a: (b, a * (phase if b[j] else 1)))
        return self._clean_state()

    def h(self, j: int):
        """
        Apply the Hadamard gate to the j-th qubit.
        Optimized implementation that avoids unnecessary operations.
        """
        # print(f"-> Applying Hadamard gate to qubit {j}")
        inv_sqrt2 = 1 / sqrt(2)
        
        # Group states by the target qubit value for efficient processing
        states_0 = []  # States where qubit j is 0
        states_1 = []  # States where qubit j is 1
        
        for bits, amp in self.state:
            if bits[j] == 0:
                states_0.append((bits, amp))
            else:
                states_1.append((bits, amp))
        
        new_states = []
        
        # Process states where qubit j was 0
        for bits, amp in states_0:
            bits_0 = bits  # Qubit j stays 0
            bits_1 = flip(bits, j)  # Qubit j becomes 1
            new_amp = amp * inv_sqrt2
            new_states.extend([(bits_0, new_amp), (bits_1, new_amp)])
        
        # Process states where qubit j was 1
        for bits, amp in states_1:
            bits_0 = flip(bits, j)  # Qubit j becomes 0
            bits_1 = bits  # Qubit j stays 1
            new_amp = amp * inv_sqrt2
            new_states.extend([(bits_0, new_amp), (bits_1, -new_amp)])
        
        # Combine amplitudes for identical basis states
        state_dict = {}
        for bits, amp in new_states:
            key = bits.to01()
            if key in state_dict:
                state_dict[key] = (bits, state_dict[key][1] + amp)
            else:
                state_dict[key] = (bits, amp)
        
        self.state = seq(list(state_dict.values()))
        return self._clean_state()

    def rx(self, qubit: int, angle: float):
        """
        Apply rotation around X-axis by angle θ.
        """
        # print(f"-> Applying RX gate to qubit {qubit}, angle {angle:.3f}")
        cos_half = cos(angle / 2)
        sin_half = sin(angle / 2)
        
        new_states = []
        for bits, amp in self.state:
            if bits[qubit] == 0:
                # |0⟩ component
                bits_0 = bits
                bits_1 = flip(bits, qubit)
                new_states.extend([
                    (bits_0, amp * cos_half),
                    (bits_1, amp * (-1j * sin_half))
                ])
            else:
                # |1⟩ component
                bits_0 = flip(bits, qubit)
                bits_1 = bits
                new_states.extend([
                    (bits_0, amp * (-1j * sin_half)),
                    (bits_1, amp * cos_half)
                ])
        
        # Combine amplitudes for identical basis states
        state_dict = {}
        for bits, amp in new_states:
            key = bits.to01()
            if key in state_dict:
                state_dict[key] = (bits, state_dict[key][1] + amp)
            else:
                state_dict[key] = (bits, amp)
        
        self.state = seq(list(state_dict.values()))
        return self._clean_state()

    def ry(self, qubit: int, angle: float):
        """
        Apply rotation around Y-axis by angle θ.
        RY(θ) = cos(θ/2)|0⟩⟨0| + cos(θ/2)|1⟩⟨1| + sin(θ/2)|0⟩⟨1| - sin(θ/2)|1⟩⟨0|
        """
        # print(f"-> Applying RY gate to qubit {qubit}, angle {angle:.3f}")
        cos_half = cos(angle / 2)
        sin_half = sin(angle / 2)
        
        new_states = []
        for bits, amp in self.state:
            if bits[qubit] == 0:
                # |0⟩ component
                bits_0 = bits
                bits_1 = flip(bits, qubit)
                new_states.extend([
                    (bits_0, amp * cos_half),
                    (bits_1, amp * sin_half)
                ])
            else:
                # |1⟩ component
                bits_0 = flip(bits, qubit)
                bits_1 = bits
                new_states.extend([
                    (bits_0, amp * (-sin_half)),
                    (bits_1, amp * cos_half)
                ])
        
        # Combine amplitudes for identical basis states
        state_dict = {}
        for bits, amp in new_states:
            key = bits.to01()
            if key in state_dict:
                state_dict[key] = (bits, state_dict[key][1] + amp)
            else:
                state_dict[key] = (bits, amp)
        
        self.state = seq(list(state_dict.values()))
        return self._clean_state()

    def rz(self, qubit: int, angle: float):
        """
        Apply rotation around Z-axis by angle θ.
        """
        # print(f"-> Applying RZ gate to qubit {qubit}, angle {angle:.3f}")
        self.state = self.state.smap(
            lambda b, a: (b, a * exp(1j * angle / 2 * (-1 if b[qubit] else 1)))
        )
        return self._clean_state()

    def cp(self, control: int, target: int, angle: float):
        """
        Apply controlled phase gate with improved numerical stability.
        """
        # print(f"-> Applying CP gate with control {control}, target {target}, angle {angle:.3f}")
        phase = exp(1j * angle)
        self.state = self.state.smap(
            lambda b, a: (b, a * phase if (b[control] and b[target]) else a)
        )
        return self._clean_state()

    def swap(self, i: int, j: int):
        """
        Swap two qubits with optimized bit manipulation.
        """
        # print(f"-> Applying SWAP gate between qubits {i} and {j}")
        new_states = []
        for bits, amp in self.state:
            # Create new bitarray with swapped bits
            new_bits = mut_bitarray(bits)
            new_bits[i], new_bits[j] = new_bits[j], new_bits[i]
            new_states.append((bitarray(new_bits), amp))
        
        self.state = seq(new_states)
        return self._clean_state()

    def measure(self, j: int, cbit: Optional[int] = None):
        """
        Measure the j-th qubit with improved numerical stability.
        """
        # print(f"-> Measuring qubit {j}")
        
        # Calculate probabilities with better numerical precision
        prob_0 = 0.0
        prob_1 = 0.0
        
        for bits, amp in self.state:
            prob = abs(amp) ** 2
            if bits[j] == 0:
                prob_0 += prob
            else:
                prob_1 += prob
        
        # Normalize probabilities to handle numerical errors
        total_prob = prob_0 + prob_1
        if total_prob > 0:
            prob_0 /= total_prob
            prob_1 /= total_prob
        
        # print(f"\tProbability of |0⟩: {prob_0:.3f}")
        # print(f"\tProbability of |1⟩: {prob_1:.3f}")
        
        # Generate measurement result
        measurement = int(random.random() >= prob_0)
        # print(f"\tMeasurement result: {measurement}")
        
        # Store in classical register if specified
        if cbit is not None:
            self.cbits[cbit] = measurement
        
        # Collapse the state
        target_prob = prob_0 if measurement == 0 else prob_1
        if target_prob > 0:
            norm_factor = sqrt(target_prob)
            new_states = []
            
            for bits, amp in self.state:
                if bits[j] == measurement:
                    new_states.append((bits, amp / norm_factor))
            
            self.state = seq(new_states)
        else:
            # Handle edge case where measurement probability is 0
            self.state = seq([])
        
        return self._clean_state()

    def get_probabilities(self) -> Dict[str, float]:
        """
        Get measurement probabilities for all basis states.
        """
        probs = {}
        for bits, amp in self.state:
            prob = abs(amp) ** 2
            if prob > self.tolerance:
                probs[bits.to01()] = prob
        return probs

    def get_amplitude(self, bitstring: str) -> complex:
        """
        Get the amplitude for a specific basis state.
        """
        target_bits = bitarray(bitstring)
        for bits, amp in self.state:
            if bits == target_bits:
                return amp
        return 0.0 + 0j

    def is_normalized(self) -> bool:
        """
        Check if the state is properly normalized.
        """
        norm_squared = sum(abs(amp) ** 2 for _, amp in self.state).real
        return abs(norm_squared - 1.0) < self.tolerance

    def __str__(self):
        """
        Return a string representation of the quantum state.
        """
        if not self.state:
            return "Empty quantum state"
        
        # Sort states by bitstring for consistent output
        sorted_states = sorted(
            [(bits.to01(), amp) for bits, amp in self.state],
            key=lambda x: x[0]
        )
        
        # Filter out negligible amplitudes for display
        significant_states = [
            (bits, amp) for bits, amp in sorted_states 
            if abs(amp) > self.tolerance
        ]
        
        if not significant_states:
            return "Quantum state: (all amplitudes negligible)"
        
        result = "Quantum state:\n"
        
        for bits, amp in significant_states:
            if abs(amp.imag) < self.tolerance:
                # Real amplitude
                result += f"|{bits}⟩: {amp.real:.6f}\n"
            elif abs(amp.real) < self.tolerance:
                # Pure imaginary amplitude
                result += f"|{bits}⟩: {amp.imag:.6f}i\n"
            else:
                # Complex amplitude
                result += f"|{bits}⟩: {amp:.6f}\n"
        
        # Add normalization check
        if self.is_normalized():
            result += "✓ State is normalized\n"
        else:
            norm_squared = sum(abs(amp) ** 2 for _, amp in sorted_states).real
            result += f"⚠ State norm²: {norm_squared:.6f}\n"
        
        # Add classical register values if they exist
        if self.m_bits > 0:
            result += f"\nClassical register: {self.cbits}"
        
        return result.rstrip()