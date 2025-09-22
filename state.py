"""
Optimized quantum state simulator using functional programming techniques.
This module implements an efficient quantum state class that supports various quantum gates
and operations without requiring matrix algebra.
"""

from math import sqrt, pi, cos, sin
from cmath import exp
import random
from typing import Optional, Dict, Tuple
from collections import defaultdict
import numpy as np

# For compatibility, keep the module imports even though we'll minimize their use
try:
    from functional import seq
except ImportError:
    seq = None

try:
    from bitarray import frozenbitarray as bitarray
    from bitarray import bitarray as mut_bitarray
except ImportError:
    bitarray = None
    mut_bitarray = None



def set_bit(x, i: int, v: int):
    """Set the i-th bit of x to value v (0 or 1)"""
    if isinstance(x, int):
        if v:
            return x | (1 << i)
        else:
            return x & ~(1 << i)
    else:
        # Fallback for bitarray
        new = mut_bitarray(x)
        new[i] = v
        return bitarray(new)


def flip(x, i: int):
    """Flip (negate) the i-th bit of x"""
    if isinstance(x, int):
        return x ^ (1 << i)
    else:
        # Fallback for bitarray
        mask = bitarray("0" * i + "1" + "0" * (len(x) - i - 1))
        return x ^ mask


def get_bit(x, i: int) -> int:
    """Get the i-th bit of x"""
    if isinstance(x, int):
        return (x >> i) & 1
    else:
        # Fallback for bitarray
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
        
        # Use integer representation internally for much faster operations
        # But maintain compatibility with the functional seq interface
        self._state_dict = {0: 1.0 + 0j}  # Integer keys for O(1) access
        self.cbits = [0] * n_bits if n_bits > 0 else []
        
        # Tolerance for numerical precision
        self.tolerance = 1e-12
        
        # Optimization flags
        self._needs_cleaning = False
        self._is_normalized = True
        
        # Cached values for performance
        self._inv_sqrt2 = 1.0 / sqrt(2)
        
        # Maintain backward compatibility
        self._update_seq_state()

    def _update_seq_state(self):
        """Update the seq state for backward compatibility"""
        if seq and bitarray:
            # Convert internal dict to seq format
            state_list = []
            for int_bits, amp in self._state_dict.items():
                if abs(amp) > self.tolerance:
                    bit_str = format(int_bits, f'0{self.n_qubits}b')
                    bits = bitarray(bit_str)
                    state_list.append((bits, amp))
            self.state = seq(state_list)
        else:
            # Fallback to simple list
            self.state = list(self._state_dict.items())

    def _int_to_bitarray(self, int_bits: int):
        """Convert integer to bitarray for compatibility"""
        if bitarray:
            bit_str = format(int_bits, f'0{self.n_qubits}b')
            return bitarray(bit_str)
        return int_bits

    def _clean_state(self):
        """Remove negligible amplitudes to maintain numerical stability"""
        if self._needs_cleaning:
            self._state_dict = {k: v for k, v in self._state_dict.items() 
                               if abs(v) > self.tolerance}
            self._needs_cleaning = False
        self._update_seq_state()
        return self

    def _normalize(self):
        """Normalize the quantum state"""
        if not self._is_normalized:
            norm_squared = sum(abs(amp) ** 2 for amp in self._state_dict.values())
            if norm_squared > 0:
                norm = sqrt(norm_squared)
                for k in self._state_dict:
                    self._state_dict[k] /= norm
            self._is_normalized = True
        self._update_seq_state()
        return self

    def x(self, j: int):
        """
        Apply the Pauli-X (NOT) gate to the j-th qubit.
        """
        new_state = {}
        bit_mask = 1 << (self.n_qubits - j - 1)
        
        for int_bits, amp in self._state_dict.items():
            new_bits = int_bits ^ bit_mask
            new_state[new_bits] = amp
        
        self._state_dict = new_state
        self._update_seq_state()
        return self._clean_state()

    def y(self, j: int):
        """
        Apply the Pauli-Y gate to the j-th qubit.
        """
        new_state = {}
        bit_mask = 1 << (self.n_qubits - j - 1)
        
        for int_bits, amp in self._state_dict.items():
            new_bits = int_bits ^ bit_mask
            # Check if bit j is 1 or 0
            if int_bits & bit_mask:
                new_state[new_bits] = amp * (-1j)
            else:
                new_state[new_bits] = amp * 1j
        
        self._state_dict = new_state
        self._needs_cleaning = True
        self._update_seq_state()
        return self._clean_state()

    def z(self, j: int):
        """
        Apply the Pauli-Z gate to the j-th qubit.
        """
        bit_mask = 1 << (self.n_qubits - j - 1)
        
        # In-place modification for better performance
        for int_bits in list(self._state_dict.keys()):
            if int_bits & bit_mask:
                self._state_dict[int_bits] *= -1
        
        self._update_seq_state()
        return self._clean_state()

    def cx(self, control: int, target: int):
        """
        Apply the CNOT gate with control and target qubits.
        """
        new_state = {}
        control_mask = 1 << (self.n_qubits - control - 1)
        target_mask = 1 << (self.n_qubits - target - 1)
        
        for int_bits, amp in self._state_dict.items():
            if int_bits & control_mask:  # control bit is 1
                new_bits = int_bits ^ target_mask  # flip target
                new_state[new_bits] = amp
            else:
                new_state[int_bits] = amp
        
        self._state_dict = new_state
        self._update_seq_state()
        return self._clean_state()

    def cz(self, control: int, target: int):
        """
        Apply the controlled-Z gate.
        """
        control_mask = 1 << (self.n_qubits - control - 1)
        target_mask = 1 << (self.n_qubits - target - 1)
        
        # In-place modification
        for int_bits in list(self._state_dict.keys()):
            if (int_bits & control_mask) and (int_bits & target_mask):
                self._state_dict[int_bits] *= -1
        
        self._update_seq_state()
        return self._clean_state()

    def s(self, j: int):
        """
        Apply the S (phase) gate to the j-th qubit.
        """
        bit_mask = 1 << (self.n_qubits - j - 1)
        
        for int_bits in list(self._state_dict.keys()):
            if int_bits & bit_mask:
                self._state_dict[int_bits] *= 1j
        
        self._update_seq_state()
        return self._clean_state()

    def s_dagger(self, j: int):
        """
        Apply the S† gate to the j-th qubit.
        """
        bit_mask = 1 << (self.n_qubits - j - 1)
        
        for int_bits in list(self._state_dict.keys()):
            if int_bits & bit_mask:
                self._state_dict[int_bits] *= -1j
        
        self._update_seq_state()
        return self._clean_state()

    def t(self, j: int):
        """
        Apply the T gate to the j-th qubit.
        """
        bit_mask = 1 << (self.n_qubits - j - 1)
        phase = exp(1j * pi / 4)
        
        for int_bits in list(self._state_dict.keys()):
            if int_bits & bit_mask:
                self._state_dict[int_bits] *= phase
        
        self._update_seq_state()
        return self._clean_state()

    def t_dagger(self, j: int):
        """
        Apply the T† gate to the j-th qubit.
        """
        bit_mask = 1 << (self.n_qubits - j - 1)
        phase = exp(-1j * pi / 4)
        
        for int_bits in list(self._state_dict.keys()):
            if int_bits & bit_mask:
                self._state_dict[int_bits] *= phase
        
        self._update_seq_state()
        return self._clean_state()

    def h(self, j: int):
        """
        Apply the Hadamard gate to the j-th qubit.
        """
        new_state = defaultdict(complex)
        bit_mask = 1 << (self.n_qubits - j - 1)
        
        for int_bits, amp in self._state_dict.items():
            amp_scaled = amp * self._inv_sqrt2
            
            if int_bits & bit_mask:  # bit j is 1
                # |1⟩ → (|0⟩ - |1⟩)/√2
                new_bits_0 = int_bits ^ bit_mask  # flip to 0
                new_state[new_bits_0] += amp_scaled
                new_state[int_bits] -= amp_scaled
            else:  # bit j is 0
                # |0⟩ → (|0⟩ + |1⟩)/√2
                new_bits_1 = int_bits ^ bit_mask  # flip to 1
                new_state[int_bits] += amp_scaled
                new_state[new_bits_1] += amp_scaled
        
        # Clean up zeros and convert back to regular dict
        self._state_dict = {k: v for k, v in new_state.items() 
                           if abs(v) > self.tolerance}
        self._update_seq_state()
        return self

    def rx(self, qubit: int, angle: float):
        """
        Apply rotation around X-axis by angle θ.
        """
        cos_half = cos(angle / 2)
        sin_half = sin(angle / 2)
        i_sin_half = -1j * sin_half
        
        new_state = defaultdict(complex)
        bit_mask = 1 << (self.n_qubits - qubit - 1)
        
        for int_bits, amp in self._state_dict.items():
            if int_bits & bit_mask:  # bit is 1
                bits_0 = int_bits ^ bit_mask
                new_state[bits_0] += amp * i_sin_half
                new_state[int_bits] += amp * cos_half
            else:  # bit is 0
                bits_1 = int_bits ^ bit_mask
                new_state[int_bits] += amp * cos_half
                new_state[bits_1] += amp * i_sin_half
        
        self._state_dict = dict(new_state)
        self._needs_cleaning = True
        self._update_seq_state()
        return self._clean_state()

    def ry(self, qubit: int, angle: float):
        """
        Apply rotation around Y-axis by angle θ.
        """
        cos_half = cos(angle / 2)
        sin_half = sin(angle / 2)
        
        new_state = defaultdict(complex)
        bit_mask = 1 << (self.n_qubits - qubit - 1)
        
        for int_bits, amp in self._state_dict.items():
            if int_bits & bit_mask:  # bit is 1
                bits_0 = int_bits ^ bit_mask
                new_state[bits_0] += amp * (-sin_half)
                new_state[int_bits] += amp * cos_half
            else:  # bit is 0
                bits_1 = int_bits ^ bit_mask
                new_state[int_bits] += amp * cos_half
                new_state[bits_1] += amp * sin_half
        
        self._state_dict = dict(new_state)
        self._needs_cleaning = True
        self._update_seq_state()
        return self._clean_state()

    def rz(self, qubit: int, angle: float):
        """
        Apply rotation around Z-axis by angle θ.
        """
        bit_mask = 1 << (self.n_qubits - qubit - 1)
        phase_0 = exp(1j * angle / 2)
        phase_1 = exp(-1j * angle / 2)
        
        # In-place modification
        for int_bits in list(self._state_dict.keys()):
            if int_bits & bit_mask:
                self._state_dict[int_bits] *= phase_1
            else:
                self._state_dict[int_bits] *= phase_0
        
        self._needs_cleaning = True
        self._update_seq_state()
        return self._clean_state()

    def cp(self, control: int, target: int, angle: float):
        """
        Apply controlled phase gate - CRITICAL for QFT in Shor's algorithm.
        Optimized for performance.
        """
        control_mask = 1 << (self.n_qubits - control - 1)
        target_mask = 1 << (self.n_qubits - target - 1)
        phase = exp(1j * angle)
        
        # In-place modification for better performance
        for int_bits in list(self._state_dict.keys()):
            if (int_bits & control_mask) and (int_bits & target_mask):
                self._state_dict[int_bits] *= phase
        
        self._update_seq_state()
        return self._clean_state()

    def swap(self, i: int, j: int):
        """
        Swap two qubits - used in QFT.
        Optimized bit manipulation using integers.
        """
        if i == j:
            return self
        
        new_state = {}
        bit_i = self.n_qubits - i - 1
        bit_j = self.n_qubits - j - 1
        mask_i = 1 << bit_i
        mask_j = 1 << bit_j
        
        for int_bits, amp in self._state_dict.items():
            # Extract bits at positions i and j
            bit_val_i = (int_bits >> bit_i) & 1
            bit_val_j = (int_bits >> bit_j) & 1
            
            # Only swap if bits are different
            if bit_val_i != bit_val_j:
                # Clear both bits
                new_bits = int_bits & ~mask_i & ~mask_j
                # Set swapped bits
                if bit_val_i:
                    new_bits |= mask_j
                if bit_val_j:
                    new_bits |= mask_i
                new_state[new_bits] = amp
            else:
                new_state[int_bits] = amp
        
        self._state_dict = new_state
        self._update_seq_state()
        return self._clean_state()

    def measure(self, j: int, cbit: Optional[int] = None):
        """
        Measure the j-th qubit - critical for extracting period in Shor's algorithm.
        Optimized with numpy for faster probability calculation.
        """
        bit_mask = 1 << (self.n_qubits - j - 1)
        
        # Separate states by bit value
        states_0 = []
        states_1 = []
        amps_0 = []
        amps_1 = []
        
        for int_bits, amp in self._state_dict.items():
            if int_bits & bit_mask:
                states_1.append(int_bits)
                amps_1.append(amp)
            else:
                states_0.append(int_bits)
                amps_0.append(amp)
        
        # Calculate probabilities using numpy if available, else fallback
        if amps_0:
            if np:
                prob_0 = np.sum(np.abs(np.array(amps_0, dtype=np.complex128))**2)
            else:
                prob_0 = sum(abs(a)**2 for a in amps_0)
        else:
            prob_0 = 0.0
            
        if amps_1:
            if np:
                prob_1 = np.sum(np.abs(np.array(amps_1, dtype=np.complex128))**2)
            else:
                prob_1 = sum(abs(a)**2 for a in amps_1)
        else:
            prob_1 = 0.0
        
        # Normalize probabilities
        total_prob = prob_0 + prob_1
        if total_prob > 0:
            prob_0 /= total_prob
            prob_1 /= total_prob
        
        # Generate measurement result
        measurement = int(random.random() >= prob_0)
        
        # Store in classical register if specified
        if cbit is not None and cbit < self.m_bits:
            self.cbits[cbit] = measurement
        
        # Collapse the state
        if measurement == 0 and prob_0 > 0:
            norm_factor = 1.0 / sqrt(prob_0)
            self._state_dict = {state: amp * norm_factor 
                               for state, amp in zip(states_0, amps_0)}
        elif measurement == 1 and prob_1 > 0:
            norm_factor = 1.0 / sqrt(prob_1)
            self._state_dict = {state: amp * norm_factor 
                               for state, amp in zip(states_1, amps_1)}
        else:
            self._state_dict = {}
        
        self._is_normalized = True
        self._update_seq_state()
        return self._clean_state()

    def get_probabilities(self) -> Dict[str, float]:
        """
        Get measurement probabilities for all basis states.
        """
        probs = {}
        for int_bits, amp in self._state_dict.items():
            prob = abs(amp) ** 2
            if prob > self.tolerance:
                bitstring = format(int_bits, f'0{self.n_qubits}b')
                probs[bitstring] = prob
        return probs

    def get_amplitude(self, bitstring: str) -> complex:
        """
        Get the amplitude for a specific basis state.
        """
        int_bits = int(bitstring, 2)
        return self._state_dict.get(int_bits, 0.0 + 0j)

    def is_normalized(self) -> bool:
        """
        Check if the state is properly normalized.
        """
        norm_squared = sum(abs(amp) ** 2 for amp in self._state_dict.values())
        return abs(norm_squared - 1.0) < self.tolerance

    def __str__(self):
        """
        Return a string representation of the quantum state.
        """
        if not self._state_dict:
            return "Empty quantum state"
        
        # Sort states by integer value (equivalent to bitstring sorting)
        sorted_items = sorted(self._state_dict.items())
        
        # Filter out negligible amplitudes for display
        significant_states = [
            (format(int_bits, f'0{self.n_qubits}b'), amp)
            for int_bits, amp in sorted_items
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
        norm_squared = sum(abs(amp) ** 2 for _, amp in self._state_dict.items())
        if abs(norm_squared - 1.0) < self.tolerance:
            result += "✓ State is normalized\n"
        else:
            result += f"⚠ State norm²: {norm_squared:.6f}\n"
        
        # Add classical register values if they exist
        if self.m_bits > 0:
            result += f"\nClassical register: {self.cbits}"
        
        return result.rstrip()