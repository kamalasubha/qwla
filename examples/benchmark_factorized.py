import time
import math
import random
from typing import List, Tuple, Optional
from fractions import Fraction
from state import State
import sys

class ImprovedQuantumFactorization:
    
    def __init__(self):
        self.classical_results = []
        self.quantum_results = []
        self.circuit_depth_classical = {}
        self.circuit_depth_quantum = {}
    
    # ============= IMPROVED QUANTUM COMPONENTS =============
    
    def quantum_fourier_transform(self, state: State, qubits: List[int], inverse: bool = False):
        n = len(qubits)
        
        if inverse:
            qubits = qubits[::-1]
        
        for j in range(n):
            # Apply Hadamard to create superposition
            state.h(qubits[j])
            
            # Apply controlled phase rotations
            for k in range(j + 1, n):
                angle = math.pi / (2 ** (k - j))
                if not inverse:
                    state.cp(qubits[k], qubits[j], angle)
                else:
                    state.cp(qubits[k], qubits[j], -angle)
        
        # Swap qubits for proper ordering
        for i in range(n // 2):
            state.swap(qubits[i], qubits[n - 1 - i])
    
    def modular_exponentiation_circuit(self, state: State, a: int, N: int, 
                                      control_qubits: List[int], 
                                      target_qubits: List[int]):
        n_controls = len(control_qubits)
        
        # Initialize target register to |1⟩
        state.x(target_qubits[0])
        
        # Apply controlled modular multiplication
        for i in range(n_controls):
            power = 2 ** i
            a_power = pow(a, power, N)
            
            # Simplified controlled multiplication
            # In real quantum computer, this would be a complex circuit
            if a_power % 2 == 1:
                state.cx(control_qubits[i], target_qubits[0])
            
            # Add phase based on the power
            phase = 2 * math.pi * a_power / N
            state.cp(control_qubits[i], target_qubits[0], phase)
    
    def improved_quantum_order_finding(self, a: int, N: int) -> int:
        # Determine number of qubits needed
        # We need 2n qubits for n-bit number to achieve high precision
        n_bits = N.bit_length()
        n_counting_qubits = 2 * n_bits + 3  # Extra qubits for precision
        n_target_qubits = n_bits
        
        # Limit qubits for simulation feasibility
        n_counting_qubits = min(n_counting_qubits, 10)
        n_target_qubits = min(n_target_qubits, 4)
        
        state = State(n_counting_qubits + n_target_qubits, n_counting_qubits)
        
        counting_qubits = list(range(n_counting_qubits))
        target_qubits = list(range(n_counting_qubits, n_counting_qubits + n_target_qubits))
        
        # Step 1: Initialize counting register in superposition
        for q in counting_qubits:
            state.h(q)
        
        # Step 2: Apply controlled modular exponentiation
        self.modular_exponentiation_circuit(state, a, N, counting_qubits, target_qubits)
        
        # Step 3: Apply inverse QFT to extract phase
        self.quantum_fourier_transform(state, counting_qubits, inverse=True)
        
        # Step 4: Measure counting register
        for i, q in enumerate(counting_qubits):
            state.measure(q, i)
        
        # Step 5: Extract period using continued fractions
        measured_value = sum(state.cbits[i] * (2 ** i) for i in range(n_counting_qubits))
        
        if measured_value == 0:
            return -1
        
        # Use continued fractions to find the period
        phase = measured_value / (2 ** n_counting_qubits)
        frac = Fraction(phase).limit_denominator(N)
        
        r = frac.denominator
        
        # Verify the period
        if r < N and pow(a, r, N) == 1:
            return r
        
        # If verification fails, try classical refinement
        for mult in range(1, 10):
            test_r = r * mult
            if test_r < N and pow(a, test_r, N) == 1:
                return test_r
        
        return -1
    
    def enhanced_shors_algorithm(self, N: int, max_attempts: int) -> Tuple[int, int]:
        
        for attempt in range(max_attempts):
            # Step 1: Choose random a coprime to N
            a = random.randint(2, N - 1)
            gcd_val = math.gcd(a, N)
            
            if gcd_val > 1:
                print ("attempt", attempt)
                return (gcd_val, N // gcd_val)
            
            # Step 2: Find period using quantum order finding
            r = self.improved_quantum_order_finding(a, N)
            
            if r == -1 or r % 2 != 0:
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
    
    # ============= CLASSICAL METHODS (kept for comparison) =============
    
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
    
    # ============= PERFORMANCE ANALYSIS =============
    
    def benchmark_with_analysis(self, numbers: List[int], max_attempts: int) -> None:
        print("=" * 80)
        print("ENHANCED QUANTUM VS CLASSICAL FACTORIZATION ANALYSIS")
        print("=" * 80)
        
        for n in numbers:
            print(f"\n{'='*40}")
            print(f"Factoring N = {n} ({n.bit_length()} bits)")
            print(f"{'='*40}")
            
            # Actual factorization
            start = time.time()
            factors_classical = self.pollard_rho(n)
            classical_time = time.time() - start
            
            start = time.time()
            factors_quantum = self.enhanced_shors_algorithm(n, max_attempts)
            quantum_time = time.time() - start
            
            print(f"\nResults:")
            print(f"  Classical: {factors_classical} in {classical_time:.6f}s")
            print(f"  Quantum: {factors_quantum} in {quantum_time:.6f}s")
            


def main():
    """Enhanced demonstration with proper test cases"""
    random.seed(42)
    max_attempts = int(sys.argv[1])
    factorizer = ImprovedQuantumFactorization()
    
    # Test with increasingly large semiprimes (products of two primes)
    # These are the types of numbers used in RSA encryption
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
    
    
    factorizer.benchmark_with_analysis(test_numbers, max_attempts)

if __name__ == "__main__":
    main()