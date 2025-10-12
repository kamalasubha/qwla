import math
import random
import time
from fractions import Fraction
from typing import List, Tuple, Optional
import numpy as np
import sys

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.synthesis.qft import synth_qft_full

class ShorsAlgorithmQiskit:
    """
    Implementation of Shor's factorization algorithm using Qiskit
    """
    
    def __init__(self, backend='aer_simulator', shots=1024):
        """
        Initialize Shor's algorithm with Qiskit backend
        
        Args:
            backend: Qiskit backend to use ('aer_simulator', 'qasm_simulator', etc.)
            shots: Number of measurement shots
        """
        self.backend = AerSimulator()
        self.backend.set_options(seed_simulator=42)
        self.shots = shots
        
    def create_qft(self, n_qubits: int, inverse: bool = False) -> QuantumCircuit:
        """
        Create Quantum Fourier Transform circuit
        
        Args:
            n_qubits: Number of qubits
            inverse: If True, create inverse QFT
            
        Returns:
            QFT QuantumCircuit
        """
        return synth_qft_full(num_qubits=n_qubits, do_swaps=True, inverse=inverse)
    
    def controlled_unitary(self, qc: QuantumCircuit, control: int, 
                          target_qubits: List[int], a: int, N: int, power: int):
        """
        Implement controlled-U^(2^power) gate for modular exponentiation
        
        This is a simplified version - in practice, this would be implemented
        using modular arithmetic circuits
        """
        # Calculate a^(2^power) mod N
        a_power = pow(a, 2**power, N)
        
        # Apply controlled operations based on the result
        # This is highly simplified - actual implementation requires
        # quantum modular multiplication circuits
        
        # Add phase based on the modular exponentiation result
        phase = 2 * np.pi * a_power / N
        for target in target_qubits:
            qc.cp(phase, control, target)
    
    def create_order_finding_circuit(self, a: int, N: int) -> Tuple[QuantumCircuit, int]:
        """
        Create the quantum circuit for order finding (core of Shor's algorithm)
        
        Args:
            a: Base for modular exponentiation
            N: Number to find the order of a
            
        Returns:
            Tuple of (QuantumCircuit, number of counting qubits)
        """
        # Determine number of qubits needed
        n_count = 2 * N.bit_length() + 1  # Counting qubits (for precision)
        n_target = N.bit_length()  # Target qubits
        
        # Limit for simulation feasibility
        n_count = min(n_count, 8)  # Reduce for simulation
        n_target = min(n_target, 4)

        if N > 1000:
            n_count = min(n_count, 12)  # More qubits for large N
            n_target = min(n_target, 6)
        
        # Create quantum registers
        counting_reg = QuantumRegister(n_count, 'counting')
        target_reg = QuantumRegister(n_target, 'target')
        classical_reg = ClassicalRegister(n_count, 'measurement')
        
        # Create quantum circuit
        qc = QuantumCircuit(counting_reg, target_reg, classical_reg)
        
        # Step 1: Initialize counting register in superposition
        for i in range(n_count):
            qc.h(counting_reg[i])
        
        # Step 2: Initialize target register to |1⟩ (for multiplication)
        qc.x(target_reg[0])
        
        # Step 3: Apply controlled modular exponentiation
        for i in range(n_count):
            self.controlled_unitary(
                qc, counting_reg[i], 
                list(range(n_count, n_count + n_target)),
                a, N, i
            )
        
        # Step 4: Apply inverse QFT to counting register
        qft_inverse = self.create_qft(n_count, inverse=True)
        qc.append(qft_inverse, counting_reg[:])
        
        # Step 5: Measure counting register
        qc.measure(counting_reg, classical_reg)
        
        return qc, n_count
    
    def run_quantum_order_finding(self, a: int, N: int) -> Optional[int]:
        """
        Execute quantum order finding circuit
        
        Args:
            a: Base for modular exponentiation  
            N: Number to find order
            
        Returns:
            The period r if found, None otherwise
        """
        # Create the quantum circuit
        qc, n_count = self.create_order_finding_circuit(a, N)
        
        # Transpile and run the circuit
        transpiled_qc = transpile(qc, self.backend)
        job = self.backend.run(transpiled_qc, shots=self.shots)
        result = job.result()
        counts = result.get_counts(qc)
        
        # Analyze measurement results
        measured_phases = []
        for output in counts:
            decimal = int(output, 2)
            phase = decimal / (2 ** n_count)
            measured_phases.append(phase)
        
        # Find the period using continued fractions
        for phase in measured_phases:
            if phase == 0:
                continue
                
            # Use continued fractions to extract period
            frac = Fraction(phase).limit_denominator(N)
            r = frac.denominator
            
            # Verify the period
            if r < N and pow(a, r, N) == 1:
                return r
        
        return None
    
    def shors_algorithm(self, N: int, max_attempts: int) -> Tuple[int, int]:
        """
        Complete Shor's algorithm implementation
        
        Args:
            N: Number to factor
            max_attempts: Maximum number of attempts
            
        Returns:
            Tuple of factors (p, q) where N = p * q
        """
        """ # Check trivial cases
        if N % 2 == 0:
            return (2, N // 2)
        
        # Check if N is a perfect power
        for k in range(2, int(math.log2(N)) + 1):
            root = N ** (1/k)
            if abs(round(root) ** k - N) < 1e-10:
                factor = int(round(root))
                return (factor, N // factor) """
        
        # Main Shor's algorithm loop
        for attempt in range(max_attempts):
            # Step 1: Choose random a coprime to N
            a = random.randint(2, N - 1)
            gcd_val = math.gcd(a, N)
            
            if gcd_val > 1:
                print ("attempt", attempt)
                return (gcd_val, N // gcd_val)
            
            # Step 2: Find period using quantum order finding
            r = self.run_quantum_order_finding(a, N)
            
            if r is None:
                continue
            
            if r % 2 != 0:
                continue
            
            # Step 3: Use period to find factors
            x = pow(a, r // 2, N)
            
            if x == N - 1:
                continue
            
            factor1 = math.gcd(x - 1, N)
            factor2 = math.gcd(x + 1, N)
            
            if 1 < factor1 < N:
                print ("attempt", attempt)
                return (factor1, N // factor1)
            
            if 1 < factor2 < N:
                print ("attempt", attempt)
                return (factor2, N // factor2)
        
        # Fallback to classical method
        return self.pollard_rho(N)
    
    def pollard_rho(self, n: int) -> Tuple[int, int]:
        """Optimized Pollard's rho with Brent's improvement"""
        print ("Fallback to pollard_rho")
        if n % 2 == 0:
            return (2, n // 2)
        
        # Brent's improvement to Pollard's rho
        y, c, m = random.randint(1, n - 1), random.randint(1, n - 1), random.randint(1, n - 1)
        g, r, q = 1, 1, 1
        
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = (q * abs(x - y)) % n
                
                g = math.gcd(q, n)
                k += m
            
            r *= 2
        
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g > 1:
                    break
        
        return (g, n // g) if g != n else (1, n)

    def benchmark_analysis(self, test_numbers: List[int], max_attempts: int) -> None:
        """
        Benchmark and analyze Shor's algorithm performance
        """
        print("=" * 80)
        print("SHOR'S ALGORITHM BENCHMARK WITH QISKIT")
        print("=" * 80)
        
        
        for N in test_numbers:
            print(f"\n{'='*40}")
            print(f"Factoring N = {N} ({N.bit_length()} bits)")
            print(f"{'='*40}")

            # Actual factorization
            start = time.time()
            factors_classical = self.pollard_rho(N)
            classical_time = time.time() - start

            # Run Shor's algorithm
            start_time = time.time()
            factors_quantum = self.shors_algorithm(N, max_attempts)
            quantum_time = time.time() - start_time

            print(f"\nResults:")
            print(f"  Classical: {factors_classical} in {classical_time:.6f}s")
            print(f"  Quantum: {factors_quantum} in {quantum_time:.6f}s")
                



def main():
    """
    Main demonstration of Shor's algorithm with Qiskit
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    max_attempts = int(sys.argv[1])
    # Initialize Shor's algorithm
    shor = ShorsAlgorithmQiskit(shots=2048)
    
    # Test numbers matching benchmark_factorization_optimized.py
    test_numbers = [
              10,  # 2 × 5 (4 bits)
              14,  # 2 × 7 (4 bits)
              15,  # 3 × 5 (4 bits)
              21,  # 3 × 7 (5 bits)
              35,  # 5 × 7 (6 bits)
             143,  # 11 × 13 (8 bits)
             187,  # 11 × 17 (8 bits)
             209,  # 11 × 19 (8 bits)
             221,  # 13 × 17 (8 bits)
             247,  # 13 × 19 (8 bits)
             323,  # 17 × 19 (9 bits)
             713,  # 23 × 31 (10 bits)
             899,  # 29 × 31 (10 bits)
            1081,  # 23 × 47 (11 bits)
            1147,  # 31 × 37 (11 bits)
            1403,  # 23 × 61 (11 bits)
            1517,  # 37 × 41 (11 bits)
            1643,  # 31 × 53 (11 bits)
            1739,  # 37 × 47 (11 bits)
            1927,  # 41 × 47 (11 bits)
            2021,  # 43 × 47 (11 bits)
            2419,  # 41 × 59 (12 bits)
            2501,  # 41 × 61 (12 bits)
            2867,  # 47 × 61 (12 bits)
            3233,  # 53 × 61 (12 bits)
            3599,  # 59 × 61 (12 bits)
            8137,  # 79 × 103 (13 bits)
           11413,  # 101 × 113 (14 bits)
           14039,  # 101 × 139 (14 bits)
           16837,  # 113 × 149 (15 bits)
           17767,  # 109 × 163 (15 bits)
           17869,  # 107 × 167 (15 bits)
           18419,  # 113 × 163 (15 bits)
           22879,  # 137 × 167 (15 bits)
           25591,  # 157 × 163 (15 bits)
           26219,  # 157 × 167 (15 bits)
           26671,  # 149 × 179 (15 bits)
           32399,  # 179 × 181 (15 bits)
           50851,  # 211 × 241 (16 bits)
           69451,  # 199 × 349 (17 bits)
           72299,  # 197 × 367 (17 bits)
           95951,  # 229 × 419 (17 bits)
          111281,  # 257 × 433 (17 bits)
          146633,  # 331 × 443 (18 bits)
          190999,  # 389 × 491 (18 bits)
          215549,  # 439 × 491 (18 bits)
          220459,  # 449 × 491 (18 bits)
          253991,  # 499 × 509 (18 bits)
          858343,  # 733 × 1171 (20 bits)
          921269,  # 757 × 1217 (20 bits)
          954113]  # 719 × 1327 (20 bits)
    
    # Run benchmark
    shor.benchmark_analysis(test_numbers, max_attempts)


if __name__ == "__main__":
    main()