import time
import math
import random
from typing import List, Tuple, Optional
from state import State

class FactorizationComparison:
    def __init__(self):
        self.classical_results = []
        self.quantum_results = []
    
    # Classical Factorization Methods
    
    def trial_division(self, n: int) -> Tuple[int, int]:
        """Basic trial division factorization"""
        if n <= 1:
            return (1, n)
        
        # Check for factor 2
        if n % 2 == 0:
            return (2, n // 2)
        
        # Check odd factors up to sqrt(n)
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return (i, n // i)
        
        return (1, n)  # Prime number
    
    def pollard_rho(self, n: int) -> Tuple[int, int]:
        """Pollard's rho algorithm for factorization"""
        if n % 2 == 0:
            return (2, n // 2)
        
        def f(x):
            return (x * x + 1) % n
        
        x = random.randint(2, n - 2)
        y = x
        d = 1
        
        while d == 1:
            x = f(x)
            y = f(f(y))
            d = math.gcd(abs(x - y), n)
            
            if d == n:  # Failure, try again
                return self.trial_division(n)
        
        return (d, n // d)
    
    def fermat_factorization(self, n: int) -> Tuple[int, int]:
        """Fermat's factorization method"""
        if n % 2 == 0:
            return (2, n // 2)
        
        a = math.ceil(math.sqrt(n))
        b_squared = a * a - n
        
        while not self.is_perfect_square(b_squared):
            a += 1
            b_squared = a * a - n
            if a > (n + 1) // 2:  # Avoid infinite loop
                return self.trial_division(n)
        
        b = int(math.sqrt(b_squared))
        return (a - b, a + b)
    
    def is_perfect_square(self, n: int) -> bool:
        """Check if a number is a perfect square"""
        if n < 0:
            return False
        root = int(math.sqrt(n))
        return root * root == n
    
    # Quantum Shor's Algorithm (simplified)
    
    def quantum_order_finding(self, a: int, N: int, n_qubits: int = 6) -> int:
        """Simplified quantum order finding for Shor's algorithm"""
        state = State(n_qubits + 2, n_qubits)
        
        # Create superposition
        for i in range(n_qubits):
            state.h(i)
        
        # Simplified controlled operations
        for i in range(min(3, n_qubits)):
            state.cx(i, n_qubits)
            if a % 2 == 0:
                state.x(n_qubits + 1)
        
        # Approximate QFT
        for i in range(n_qubits):
            state.h(i)
            for j in range(i + 1, min(i + 2, n_qubits)):
                angle = math.pi / (2 ** (j - i))
                state.cp(j, i, angle)
        
        # Measure
        for i in range(n_qubits):
            state.measure(i, i)
        
        # Extract order
        measured_value = sum(state.cbits[i] * (2 ** i) for i in range(n_qubits))
        
        if measured_value == 0:
            return self.classical_order_finding(a, N)
        
        potential_order = (2 ** n_qubits) // measured_value
        
        # Verify order
        if potential_order > 0 and pow(a, potential_order, N) == 1:
            return potential_order
        
        return self.classical_order_finding(a, N)
    
    def classical_order_finding(self, a: int, N: int) -> int:
        order = 1
        current = a % N
        while current != 1 and order < N:
            current = (current * a) % N
            order += 1
        return order if current == 1 else -1
    
    def shors_algorithm(self, N: int) -> Tuple[int, int]:
        if N % 2 == 0:
            return (2, N // 2)
        
        for _ in range(5):  # Limited attempts
            a = random.randint(2, N - 1)
            g = math.gcd(a, N)
            if g > 1:
                return (g, N // g)
            
            r = self.quantum_order_finding(a, N)
            
            if r != -1 and r % 2 == 0:
                half_power = pow(a, r // 2, N)
                if half_power != 1 and half_power != N - 1:
                    factor1 = math.gcd(half_power - 1, N)
                    factor2 = math.gcd(half_power + 1, N)
                    
                    if 1 < factor1 < N:
                        return (factor1, N // factor1)
                    elif 1 < factor2 < N:
                        return (factor2, N // factor2)
        
        return self.trial_division(N)  # Fallback
    
    # Performance Testing
    
    def benchmark_classical(self, numbers: List[int]) -> List[dict]:
        """Benchmark classical factorization methods"""
        results = []
        
        for n in numbers:
            print(f"\nTesting classical methods for N = {n}")
            
            # Trial Division
            start_time = time.time()
            factors_trial = self.trial_division(n)
            trial_time = time.time() - start_time
            
            # Pollard's Rho
            start_time = time.time()
            factors_pollard = self.pollard_rho(n)
            pollard_time = time.time() - start_time
            
            # Fermat's Method
            start_time = time.time()
            factors_fermat = self.fermat_factorization(n)
            fermat_time = time.time() - start_time
            
            result = {
                'number': n,
                'trial_division': {'factors': factors_trial, 'time': trial_time},
                'pollard_rho': {'factors': factors_pollard, 'time': pollard_time},
                'fermat': {'factors': factors_fermat, 'time': fermat_time}
            }
            results.append(result)
            
            print(f"  Trial Division: {factors_trial} ({trial_time:.6f}s)")
            print(f"  Pollard's Rho: {factors_pollard} ({pollard_time:.6f}s)")
            print(f"  Fermat Method: {factors_fermat} ({fermat_time:.6f}s)")
        
        return results
    
    def benchmark_quantum(self, numbers: List[int]) -> List[dict]:
        """Benchmark Shor's quantum algorithm"""
        results = []
        
        for n in numbers:
            print(f"\nTesting Shor's algorithm for N = {n}")
            
            start_time = time.time()
            factors_shor = self.shors_algorithm(n)
            shor_time = time.time() - start_time
            
            result = {
                'number': n,
                'shors_algorithm': {'factors': factors_shor, 'time': shor_time}
            }
            results.append(result)
            
            print(f"  Shor's Algorithm: {factors_shor} ({shor_time:.6f}s)")
        
        return results
    
    def compare_performance(self, numbers: List[int]) -> None:
        """Compare classical vs quantum factorization performance"""
        print("=" * 60)
        print("FACTORIZATION PERFORMANCE COMPARISON")
        print("=" * 60)
        
        # Run benchmarks
        classical_results = self.benchmark_classical(numbers)
        quantum_results = self.benchmark_quantum(numbers)
        
        # Analysis
        print(f"\n{'=' * 60}")
        print("PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        for i, n in enumerate(numbers):
            classical = classical_results[i]
            quantum = quantum_results[i]
            
            print(f"\nNumber: {n}")
            print("-" * 20)
            
            # Find fastest classical method
            fastest_classical = min(
                classical['trial_division']['time'],
                classical['pollard_rho']['time'],
                classical['fermat']['time']
            )
            
            quantum_time = quantum['shors_algorithm']['time']
            
            print(f"Fastest Classical: {fastest_classical:.6f}s")
            print(f"Quantum (Shor's): {quantum_time:.6f}s")
            
            if quantum_time < fastest_classical:
                speedup = fastest_classical / quantum_time
                print(f"Quantum Advantage: {speedup:.2f}x faster")
            else:
                slowdown = quantum_time / fastest_classical
                print(f"Classical Advantage: {slowdown:.2f}x faster")
                print("(Note: Quantum advantage emerges for larger numbers)")
        

def main():
    """Main demonstration function"""
    random.seed(42)  # For reproducibility
    
    comparison = FactorizationComparison()
    
    # Test with various composite numbers
    test_numbers = [15, 21, 35, 77, 91, 143, 187, 221, 1000]
    
    print("Factorization Methods Comparison")
    print("This demo compares classical and quantum factorization approaches")
    print(f"Testing numbers: {test_numbers}")
    
    comparison.compare_performance(test_numbers)

if __name__ == "__main__":
    main()