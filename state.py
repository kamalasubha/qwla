"""
Highly optimized quantum state simulator with advanced performance optimizations.
This module implements a quantum state class with vectorization, caching, gate fusion,
and sophisticated memory management while maintaining perfect mathematical accuracy.
"""

import numpy as np
from math import sqrt, pi, cos, sin, log2
from cmath import exp
import random
from typing import Optional, Dict, Tuple, List, Union
from collections import defaultdict
import functools
from bitarray import frozenbitarray as bitarray
from bitarray import bitarray as mut_bitarray
import threading


# Global thread-local storage for performance
_thread_local = threading.local()

def get_thread_cache():
    """Get thread-local cache for storing computed values"""
    if not hasattr(_thread_local, 'cache'):
        _thread_local.cache = {}
    return _thread_local.cache


class FastBitArray:
    """
    Optimized bit array implementation using integer representation for small arrays.
    Falls back to bitarray for larger systems.
    """
    __slots__ = ('_value', '_length', '_use_int')
    
    def __init__(self, data):
        if isinstance(data, str):
            self._length = len(data)
            if self._length <= 64:  # Use integer for small bit arrays
                self._use_int = True
                self._value = int(data, 2)
            else:
                self._use_int = False
                self._value = bitarray(data)
        elif isinstance(data, int):
            self._length = data.bit_length() if data > 0 else 1
            self._use_int = True
            self._value = data
        else:
            self._use_int = False
            self._value = data
            self._length = len(data)
    
    @classmethod
    def from_int(cls, value: int, length: int):
        """Create FastBitArray from integer value and length"""
        obj = cls.__new__(cls)
        obj._value = value
        obj._length = length
        obj._use_int = length <= 64
        if not obj._use_int:
            # Convert to bitarray for large systems
            obj._value = bitarray(format(value, f'0{length}b'))
        return obj
    
    def __getitem__(self, index):
        if self._use_int:
            return (self._value >> (self._length - 1 - index)) & 1
        return int(self._value[index])
    
    def __eq__(self, other):
        if isinstance(other, FastBitArray):
            return self._value == other._value and self._length == other._length
        return False
    
    def __hash__(self):
        return hash((self._value, self._length))
    
    def flip(self, index):
        """Return new FastBitArray with bit at index flipped"""
        if self._use_int:
            mask = 1 << (self._length - 1 - index)
            return FastBitArray.from_int(self._value ^ mask, self._length)
        else:
            new_bits = mut_bitarray(self._value)
            new_bits[index] = not new_bits[index]
            return FastBitArray(bitarray(new_bits))
    
    def set_bit(self, index, value):
        """Return new FastBitArray with bit at index set to value"""
        if self._use_int:
            if value:
                mask = 1 << (self._length - 1 - index)
                return FastBitArray.from_int(self._value | mask, self._length)
            else:
                mask = ~(1 << (self._length - 1 - index))
                return FastBitArray.from_int(self._value & mask, self._length)
        else:
            new_bits = mut_bitarray(self._value)
            new_bits[index] = bool(value)
            return FastBitArray(bitarray(new_bits))
    
    def to_string(self):
        """Convert to binary string representation"""
        if self._use_int:
            return format(self._value, f'0{self._length}b')
        return self._value.to01()
    
    def copy(self):
        """Create a copy of this FastBitArray"""
        return FastBitArray.from_int(self._value, self._length)


@functools.lru_cache(maxsize=256)
def cached_rotation_matrix(gate_type: str, angle: float) -> np.ndarray:
    """Cache frequently used rotation matrices"""
    if gate_type == 'rx':
        c, s = cos(angle/2), sin(angle/2)
        return np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)
    elif gate_type == 'ry':
        c, s = cos(angle/2), sin(angle/2)
        return np.array([[c, -s], [s, c]], dtype=complex)
    elif gate_type == 'rz':
        return np.array([[exp(-1j*angle/2), 0], [0, exp(1j*angle/2)]], dtype=complex)
    else:
        raise ValueError(f"Unknown gate type: {gate_type}")


@functools.lru_cache(maxsize=64)
def cached_standard_gates():
    """Cache standard gate matrices"""
    return {
        'x': np.array([[0, 1], [1, 0]], dtype=complex),
        'y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'z': np.array([[1, 0], [0, -1]], dtype=complex),
        'h': np.array([[1, 1], [1, -1]], dtype=complex) / sqrt(2),
        's': np.array([[1, 0], [0, 1j]], dtype=complex),
        's_dag': np.array([[1, 0], [0, -1j]], dtype=complex),
        't': np.array([[1, 0], [0, exp(1j*pi/4)]], dtype=complex),
        't_dag': np.array([[1, 0], [0, exp(-1j*pi/4)]], dtype=complex),
    }


class GateFuser:
    """
    Optimize quantum circuits by fusing consecutive gates on the same qubit.
    """
    def __init__(self):
        self.pending_gates = defaultdict(list)
        self.gates = cached_standard_gates()
    
    def add_gate(self, qubit: int, gate_type: str, angle: Optional[float] = None):
        """Add a gate to the fusion queue"""
        if angle is not None:
            matrix = cached_rotation_matrix(gate_type, angle)
        else:
            matrix = self.gates[gate_type]
        
        self.pending_gates[qubit].append(matrix)
    
    def flush_qubit(self, qubit: int) -> Optional[np.ndarray]:
        """Get fused matrix for a qubit and clear its queue"""
        if qubit not in self.pending_gates or not self.pending_gates[qubit]:
            return None
        
        # Multiply matrices in reverse order (last gate applied first)
        result = self.pending_gates[qubit][-1]
        for matrix in reversed(self.pending_gates[qubit][:-1]):
            result = matrix @ result
        
        self.pending_gates[qubit].clear()
        return result
    
    def flush_all(self) -> Dict[int, np.ndarray]:
        """Get all fused matrices and clear all queues"""
        result = {}
        for qubit in list(self.pending_gates.keys()):
            matrix = self.flush_qubit(qubit)
            if matrix is not None:
                result[qubit] = matrix
        return result


class OptimizedStateVector:
    """
    Optimized state vector with automatic sparse/dense switching
    """
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.tolerance = 1e-12
        
        # Start with sparse representation
        self.is_sparse = True
        self.states = [FastBitArray.from_int(0, n_qubits)]
        self.amplitudes = np.array([1.0 + 0j], dtype=complex)
        self.dense_vector = None
        
        # Switch to dense for small systems or many states
        if n_qubits <= 10:
            self.to_dense()
    
    def to_dense(self):
        """Convert to dense representation"""
        if not self.is_sparse:
            return
        
        self.dense_vector = np.zeros(2**self.n_qubits, dtype=complex)
        for state, amp in zip(self.states, self.amplitudes):
            if state._use_int:
                self.dense_vector[state._value] = amp
            else:
                index = int(state.to_string(), 2)
                self.dense_vector[index] = amp
        
        self.states = None
        self.amplitudes = None
        self.is_sparse = False
    
    def to_sparse(self):
        """Convert to sparse representation"""
        if self.is_sparse:
            return
        
        # Find non-zero elements
        non_zero_indices = np.nonzero(self.dense_vector)[0]
        if len(non_zero_indices) == 0:
            self.states = []
            self.amplitudes = np.array([], dtype=complex)
        else:
            self.states = [FastBitArray.from_int(int(i), self.n_qubits) for i in non_zero_indices]
            self.amplitudes = self.dense_vector[non_zero_indices].copy()
        
        self.dense_vector = None
        self.is_sparse = True
    
    def optimize_representation(self):
        """Automatically choose best representation"""
        if self.is_sparse:
            state_count = len(self.states)
            if state_count > 2**(self.n_qubits - 2):  # > 25% filled
                self.to_dense()
        else:
            non_zero_count = np.count_nonzero(np.abs(self.dense_vector) > self.tolerance)
            if non_zero_count < 2**(self.n_qubits - 3):  # < 12.5% filled
                self.to_sparse()
    
    def clean_state(self):
        """Remove negligible amplitudes"""
        if self.is_sparse:
            if len(self.states) > 0:
                mask = np.abs(self.amplitudes) > self.tolerance
                self.states = [s for s, m in zip(self.states, mask) if m]
                self.amplitudes = self.amplitudes[mask]
        else:
            self.dense_vector[np.abs(self.dense_vector) <= self.tolerance] = 0.0
    
    def apply_single_qubit_gate(self, qubit: int, matrix: np.ndarray):
        """Apply single-qubit gate efficiently"""
        if self.is_sparse:
            self._apply_single_qubit_sparse(qubit, matrix)
        else:
            self._apply_single_qubit_dense(qubit, matrix)
        
        self.clean_state()
    
    def _apply_single_qubit_sparse(self, qubit: int, matrix: np.ndarray):
        """Apply single-qubit gate to sparse representation"""
        state_dict = defaultdict(complex)
        
        for state, amp in zip(self.states, self.amplitudes):
            bit = state[qubit]
            
            if bit == 0:
                # |0⟩ contribution
                state_dict[state] += matrix[0, 0] * amp
                state_dict[state.flip(qubit)] += matrix[1, 0] * amp
            else:
                # |1⟩ contribution
                state_dict[state.flip(qubit)] += matrix[0, 1] * amp
                state_dict[state] += matrix[1, 1] * amp
        
        # Convert back to arrays
        new_states = []
        new_amplitudes = []
        for state, amp in state_dict.items():
            if abs(amp) > self.tolerance:
                new_states.append(state)
                new_amplitudes.append(amp)
        
        self.states = new_states
        self.amplitudes = np.array(new_amplitudes, dtype=complex)
    
    def _apply_single_qubit_dense(self, qubit: int, matrix: np.ndarray):
        """Apply single-qubit gate to dense representation"""
        n = 2**self.n_qubits
        new_vector = np.zeros(n, dtype=complex)
        
        for i in range(n):
            bit = (i >> (self.n_qubits - 1 - qubit)) & 1
            if bit == 0:
                j = i | (1 << (self.n_qubits - 1 - qubit))
                new_vector[i] += matrix[0, 0] * self.dense_vector[i] + matrix[0, 1] * self.dense_vector[j]
                new_vector[j] += matrix[1, 0] * self.dense_vector[i] + matrix[1, 1] * self.dense_vector[j]
        
        self.dense_vector = new_vector
    
    def __iter__(self):
        """Iterate over (state, amplitude) pairs"""
        if self.is_sparse:
            for state, amp in zip(self.states, self.amplitudes):
                if abs(amp) > self.tolerance:
                    yield state, amp
        else:
            for i, amp in enumerate(self.dense_vector):
                if abs(amp) > self.tolerance:
                    yield FastBitArray.from_int(i, self.n_qubits), amp


class State:
    """
    Highly optimized quantum state simulator with advanced performance features.
    """
    
    def __init__(self, n_qubits: int, n_bits: int = 0):
        """
        Initialize quantum state with maximum optimization.
        """
        assert n_qubits > 0 and n_bits >= 0
        
        self.n_qubits = n_qubits
        self.m_bits = n_bits
        self.cbits = [0] * n_bits
        
        # Use optimized state vector
        self.state_vector = OptimizedStateVector(n_qubits)
        
        # Gate fusion for performance
        self.gate_fuser = GateFuser()
        self.auto_fuse = True
        
        # Performance monitoring
        self.operation_count = 0
        self.optimization_stats = {
            'gates_fused': 0,
            'state_cleanings': 0,
            'representation_switches': 0
        }
        
        # Tolerance settings
        self.tolerance = 1e-12
    
    def _maybe_fuse_gate(self, qubit: int, gate_type: str, angle: Optional[float] = None):
        """Add gate to fusion queue if auto-fusion is enabled"""
        if self.auto_fuse:
            self.gate_fuser.add_gate(qubit, gate_type, angle)
        else:
            self._apply_gate_immediately(qubit, gate_type, angle)
    
    def _apply_gate_immediately(self, qubit: int, gate_type: str, angle: Optional[float] = None):
        """Apply gate immediately without fusion"""
        if angle is not None:
            matrix = cached_rotation_matrix(gate_type, angle)
        else:
            gates = cached_standard_gates()
            matrix = gates[gate_type]
        
        self.state_vector.apply_single_qubit_gate(qubit, matrix)
        self.operation_count += 1
    
    def _flush_gates(self, qubit: Optional[int] = None):
        """Flush pending gates for optimization"""
        if qubit is not None:
            matrix = self.gate_fuser.flush_qubit(qubit)
            if matrix is not None:
                self.state_vector.apply_single_qubit_gate(qubit, matrix)
                self.optimization_stats['gates_fused'] += 1
        else:
            fused_gates = self.gate_fuser.flush_all()
            for q, matrix in fused_gates.items():
                self.state_vector.apply_single_qubit_gate(q, matrix)
                self.optimization_stats['gates_fused'] += 1
    
    # Optimized gate implementations
    def x(self, j: int):
        """Apply Pauli-X gate with gate fusion"""
        #print(f"-> Applying X gate to qubit {j}")
        self._maybe_fuse_gate(j, 'x')
        return self
    
    def y(self, j: int):
        """Apply Pauli-Y gate with gate fusion"""
        #print(f"-> Applying Y gate to qubit {j}")
        self._maybe_fuse_gate(j, 'y')
        return self
    
    def z(self, j: int):
        """Apply Pauli-Z gate with gate fusion"""
        #print(f"-> Applying Z gate to qubit {j}")
        self._maybe_fuse_gate(j, 'z')
        return self
    
    def h(self, j: int):
        """Apply Hadamard gate with optimizations"""
        #print(f"-> Applying Hadamard gate to qubit {j}")
        self._flush_gates(j)  # Flush before non-commuting gate
        self._maybe_fuse_gate(j, 'h')
        return self
    
    def s(self, j: int):
        """Apply S gate with gate fusion"""
        #print(f"-> Applying S gate to qubit {j}")
        self._maybe_fuse_gate(j, 's')
        return self
    
    def s_dagger(self, j: int):
        """Apply S† gate with gate fusion"""
        #print(f"-> Applying S† gate to qubit {j}")
        self._maybe_fuse_gate(j, 's_dag')
        return self
    
    def t(self, j: int):
        """Apply T gate with gate fusion"""
        #print(f"-> Applying T gate to qubit {j}")
        self._maybe_fuse_gate(j, 't')
        return self
    
    def t_dagger(self, j: int):
        """Apply T† gate with gate fusion"""
        #print(f"-> Applying T† gate to qubit {j}")
        self._maybe_fuse_gate(j, 't_dag')
        return self
    
    def rx(self, qubit: int, angle: float):
        """Apply RX rotation with caching"""
        #print(f"-> Applying RX gate to qubit {qubit}, angle {angle:.3f}")
        self._maybe_fuse_gate(qubit, 'rx', angle)
        return self
    
    def ry(self, qubit: int, angle: float):
        """Apply RY rotation with caching"""
        #print(f"-> Applying RY gate to qubit {qubit}, angle {angle:.3f}")
        self._maybe_fuse_gate(qubit, 'ry', angle)
        return self
    
    def rz(self, qubit: int, angle: float):
        """Apply RZ rotation with caching"""
        #print(f"-> Applying RZ gate to qubit {qubit}, angle {angle:.3f}")
        self._maybe_fuse_gate(qubit, 'rz', angle)
        return self
    
    def cx(self, control: int, target: int):
        """Apply CNOT gate with optimizations"""
        #print(f"-> Applying CNOT gate with control {control} and target {target}")
        
        # Flush gates on involved qubits
        self._flush_gates(control)
        self._flush_gates(target)
        
        # Apply CNOT
        if self.state_vector.is_sparse:
            self._apply_cnot_sparse(control, target)
        else:
            self._apply_cnot_dense(control, target)
        
        self.state_vector.clean_state()
        return self
    
    def _apply_cnot_dense(self, control: int, target: int):
        """Optimized CNOT for dense representation"""
        n = 2**self.n_qubits
        new_vector = self.state_vector.dense_vector.copy()
        
        for i in range(n):
            if (i >> (self.n_qubits - 1 - control)) & 1:  # Control is 1
                j = i ^ (1 << (self.n_qubits - 1 - target))  # Flip target
                new_vector[i], new_vector[j] = self.state_vector.dense_vector[j], self.state_vector.dense_vector[i]
        
        self.state_vector.dense_vector = new_vector
    
    def _apply_cnot_sparse(self, control: int, target: int):
        """Optimized CNOT for sparse representation"""
        new_states = []
        new_amplitudes = []
        
        for state, amp in zip(self.state_vector.states, self.state_vector.amplitudes):
            if state[control] == 1:  # Control is 1, flip target
                new_state = state.flip(target)
            else:  # Control is 0, no change
                new_state = state
            
            new_states.append(new_state)
            new_amplitudes.append(amp)
        
        self.state_vector.states = new_states
        self.state_vector.amplitudes = np.array(new_amplitudes, dtype=complex)
    
    def cp(self, control: int, target: int, angle: float):
        """Apply controlled phase gate with optimizations"""
        #print(f"-> Applying CP gate with control {control}, target {target}, angle {angle:.3f}")
        
        self._flush_gates(control)
        self._flush_gates(target)
        
        phase = exp(1j * angle)
        
        if self.state_vector.is_sparse:
            for i, state in enumerate(self.state_vector.states):
                if state[control] == 1 and state[target] == 1:
                    self.state_vector.amplitudes[i] *= phase
        else:
            n = 2**self.n_qubits
            for i in range(n):
                if ((i >> (self.n_qubits - 1 - control)) & 1) and ((i >> (self.n_qubits - 1 - target)) & 1):
                    self.state_vector.dense_vector[i] *= phase
        
        return self
    
    def swap(self, i: int, j: int):
        """Optimized SWAP gate implementation"""
        #print(f"-> Applying SWAP gate between qubits {i} and {j}")
        
        self._flush_gates(i)
        self._flush_gates(j)
        
        if self.state_vector.is_sparse:
            new_states = []
            for state in self.state_vector.states:
                # Swap bits
                if state[i] != state[j]:
                    new_state = state.flip(i).flip(j)
                else:
                    new_state = state
                new_states.append(new_state)
            self.state_vector.states = new_states
        else:
            # Dense SWAP implementation
            n = 2**self.n_qubits
            new_vector = self.state_vector.dense_vector.copy()
            for idx in range(n):
                bit_i = (idx >> (self.n_qubits - 1 - i)) & 1
                bit_j = (idx >> (self.n_qubits - 1 - j)) & 1
                if bit_i != bit_j:
                    new_idx = idx ^ (1 << (self.n_qubits - 1 - i)) ^ (1 << (self.n_qubits - 1 - j))
                    new_vector[new_idx] = self.state_vector.dense_vector[idx]
                else:
                    new_vector[idx] = self.state_vector.dense_vector[idx]
            self.state_vector.dense_vector = new_vector
        
        return self
    
    def measure(self, j: int, cbit: Optional[int] = None):
        """Highly optimized measurement"""
        #print(f"-> Measuring qubit {j}")
        
        # Flush all pending gates before measurement
        self._flush_gates()
        
        # Calculate probabilities
        prob_0 = prob_1 = 0.0
        
        if self.state_vector.is_sparse:
            for state, amp in zip(self.state_vector.states, self.state_vector.amplitudes):
                prob = abs(amp) ** 2
                if state[j] == 0:
                    prob_0 += prob
                else:
                    prob_1 += prob
        else:
            for i, amp in enumerate(self.state_vector.dense_vector):
                prob = abs(amp) ** 2
                if (i >> (self.n_qubits - 1 - j)) & 1:
                    prob_1 += prob
                else:
                    prob_0 += prob
        
        # Normalize probabilities
        total_prob = prob_0 + prob_1
        if total_prob > 0:
            prob_0 /= total_prob
            prob_1 /= total_prob
        
        #print(f"\tProbability of |0⟩: {prob_0:.6f}")
        #print(f"\tProbability of |1⟩: {prob_1:.6f}")
        
        # Generate measurement result
        measurement = int(random.random() >= prob_0)
        #print(f"\tMeasurement result: {measurement}")
        
        # Store in classical register
        if cbit is not None:
            self.cbits[cbit] = measurement
        
        # Collapse state
        target_prob = prob_0 if measurement == 0 else prob_1
        if target_prob > 0:
            norm_factor = sqrt(target_prob)
            
            if self.state_vector.is_sparse:
                new_states = []
                new_amplitudes = []
                for state, amp in zip(self.state_vector.states, self.state_vector.amplitudes):
                    if state[j] == measurement:
                        new_states.append(state)
                        new_amplitudes.append(amp / norm_factor)
                
                self.state_vector.states = new_states
                self.state_vector.amplitudes = np.array(new_amplitudes, dtype=complex)
            else:
                new_vector = np.zeros_like(self.state_vector.dense_vector)
                for i in range(len(self.state_vector.dense_vector)):
                    if ((i >> (self.n_qubits - 1 - j)) & 1) == measurement:
                        new_vector[i] = self.state_vector.dense_vector[i] / norm_factor
                self.state_vector.dense_vector = new_vector
        
        self.state_vector.clean_state()
        self.state_vector.optimize_representation()
        return self
    
    def get_probabilities(self) -> Dict[str, float]:
        """Get measurement probabilities with optimization"""
        self._flush_gates()
        
        probs = {}
        for state, amp in self.state_vector:
            prob = abs(amp) ** 2
            if prob > self.tolerance:
                probs[state.to_string()] = float(prob)
        
        return probs
    
    def get_amplitude(self, bitstring: str) -> complex:
        """Get amplitude for specific state"""
        self._flush_gates()
        
        if self.state_vector.is_sparse:
            target = FastBitArray(bitstring)
            for state, amp in zip(self.state_vector.states, self.state_vector.amplitudes):
                if state == target:
                    return complex(amp)
            return 0.0 + 0j
        else:
            index = int(bitstring, 2)
            return complex(self.state_vector.dense_vector[index])
    
    def get_state_count(self) -> int:
        """Get number of non-zero amplitude states"""
        if self.state_vector.is_sparse:
            return len(self.state_vector.states)
        else:
            return np.count_nonzero(np.abs(self.state_vector.dense_vector) > self.tolerance)
    
    def get_optimization_stats(self) -> Dict:
        """Get performance optimization statistics"""
        return {
            **self.optimization_stats,
            'operation_count': self.operation_count,
            'state_count': self.get_state_count(),
            'is_sparse': self.state_vector.is_sparse,
            'pending_gates': sum(len(gates) for gates in self.gate_fuser.pending_gates.values()),
        }
    
    def __str__(self):
        """Enhanced string representation with optimization info"""
        self._flush_gates()
        
        result = []
        state_count = 0
        
        for state, amp in self.state_vector:
            if abs(amp) > self.tolerance:
                state_count += 1
                if abs(amp.imag) < self.tolerance:
                    result.append(f"|{state.to_string()}⟩: {amp.real:.6f}")
                elif abs(amp.real) < self.tolerance:
                    result.append(f"|{state.to_string()}⟩: {amp.imag:.6f}i")
                else:
                    result.append(f"|{state.to_string()}⟩: {amp:.6f}")
        
        if not result:
            return "Empty quantum state"
        
        # Sort for consistent output
        result.sort()
        
        header = f"Quantum state ({state_count} non-zero amplitudes):\n"
        body = "\n".join(result[:10])
        
        if len(result) > 10:
            body += f"\n... and {len(result) - 10} more states"
        
        # Add optimization info
        stats = self.get_optimization_stats()
        footer = f"\n\nOptimization stats: {stats['gates_fused']} gates fused, "
        footer += f"{'sparse' if stats['is_sparse'] else 'dense'} representation"
        
        # Add classical register
        if self.m_bits > 0:
            footer += f"\nClassical register: {self.cbits}"
        
        return header + body + footer